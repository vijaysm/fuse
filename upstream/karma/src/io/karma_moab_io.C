#include "karma_moab_io.h"

#include "karma_dof_map.h"
#include "karma_exceptions.h"
#include "petsc_raii.h"
#include "../physics/common/impls/collective_signature.h"

#include "moab/Core.hpp"
#include "moab/Interface.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
template < class RealType > void append_ieee_little_endian( std::vector< unsigned char >& bytes, RealType value )
{
    static_assert( std::numeric_limits< RealType >::is_iec559 &&
                       ( sizeof( RealType ) == 4 || sizeof( RealType ) == 8 ),
                   "checkpoint scalar encoding requires IEEE binary32 or binary64 PetscReal" );
    using Bits = std::conditional_t< sizeof( RealType ) == 4, std::uint32_t, std::uint64_t >;
    const Bits bits = std::bit_cast< Bits >( value );
    for( unsigned shift = 0; shift < sizeof( Bits ) * 8; shift += 8 )
        bytes.push_back( static_cast< unsigned char >( bits >> shift ) );
}

template < class RealType > RealType read_ieee_little_endian( std::span< const unsigned char > bytes,
                                                              std::size_t offset )
{
    static_assert( std::numeric_limits< RealType >::is_iec559 &&
                       ( sizeof( RealType ) == 4 || sizeof( RealType ) == 8 ),
                   "checkpoint scalar encoding requires IEEE binary32 or binary64 PetscReal" );
    using Bits = std::conditional_t< sizeof( RealType ) == 4, std::uint32_t, std::uint64_t >;
    Bits bits = 0;
    for( unsigned byte = 0; byte < sizeof( Bits ); ++byte )
        bits |= static_cast< Bits >( bytes[offset + byte] ) << ( 8 * byte );
    return std::bit_cast< RealType >( bits );
}
}  // namespace

namespace KARMA::detail
{
namespace
{
thread_local CheckpointGatherStats last_checkpoint_gather_stats;
}

CheckpointGatherStats checkpoint_gather_stats() noexcept
{
    return last_checkpoint_gather_stats;
}

void set_checkpoint_gather_stats( std::uint64_t local_records, std::uint64_t received_records ) noexcept
{
    last_checkpoint_gather_stats = { local_records, received_records };
}

bool checkpoint_values_are_finite( const PetscScalar* values, PetscInt count ) noexcept
{
    for( PetscInt i = 0; i < count; ++i )
        if( PetscIsInfOrNanScalar( values[i] ) ) return false;
    return true;
}

void copy_checkpoint_values( const PetscScalar* source, PetscScalar* destination, PetscInt count ) noexcept
{
    for( PetscInt i = 0; i < count; ++i ) destination[i] = source[i];
}

std::vector< unsigned char > encode_checkpoint_scalar( PetscScalar value )
{
    std::vector< unsigned char > bytes;
    append_ieee_little_endian( bytes, PetscRealPart( value ) );
#if defined(PETSC_USE_COMPLEX)
    append_ieee_little_endian( bytes, PetscImaginaryPart( value ) );
#endif
    return bytes;
}

PetscScalar decode_checkpoint_scalar( std::span< const unsigned char > bytes )
{
#if defined(PETSC_USE_COMPLEX)
    constexpr std::size_t components = 2;
#else
    constexpr std::size_t components = 1;
#endif
    if( bytes.size() != components * sizeof( PetscReal ) )
        KARMA_THROW( InvalidArgument( "checkpoint scalar byte width is invalid" ) );
    const PetscReal real = read_ieee_little_endian< PetscReal >( bytes, 0 );
#if defined(PETSC_USE_COMPLEX)
    return PetscCMPLX( real, read_ieee_little_endian< PetscReal >( bytes, sizeof( PetscReal ) ) );
#else
    return real;
#endif
}

namespace
{
void sync_path( std::string_view path, int flags, bool require_directory, const char* description )
{
    const std::string name( path );
    int descriptor = -1;
    do descriptor = ::open( name.c_str(), flags );
    while( descriptor == -1 && errno == EINTR );
    if( descriptor == -1 )
        KARMA_THROW( InvalidArgument( std::string( "open checkpoint " ) + description + ": " + std::strerror( errno ) ) );

    struct stat status{};
    int stat_result = -1;
    do stat_result = ::fstat( descriptor, &status );
    while( stat_result != 0 && errno == EINTR );
    const int stat_error = stat_result == 0 ? 0 : errno;
    const bool expected_type = stat_error == 0 &&
                               ( require_directory ? S_ISDIR( status.st_mode ) : S_ISREG( status.st_mode ) );
    if( !expected_type )
    {
        const int close_result = ::close( descriptor );
        const int close_error = close_result == 0 ? 0 : errno;
        std::string message = stat_error != 0
                                  ? std::string( "inspect checkpoint " ) + description + ": " +
                                        std::strerror( stat_error )
                                  : std::string( "checkpoint sync path is not a " ) +
                                        ( require_directory ? "directory" : "regular file" );
        if( close_error != 0 ) message += "; close also failed: " + std::string( std::strerror( close_error ) );
        KARMA_THROW( InvalidArgument( message ) );
    }

    int sync_result = -1;
    const char* sync_operation = "fsync";
#if defined(__APPLE__)
    if( !require_directory )
    {
        sync_operation = "F_FULLFSYNC";
#if defined(F_FULLFSYNC)
        do sync_result = ::fcntl( descriptor, F_FULLFSYNC );
        while( sync_result != 0 && errno == EINTR );
#else
        errno = ENOTSUP;
#endif
    }
    else
#endif
    {
        do sync_result = ::fsync( descriptor );
        while( sync_result != 0 && errno == EINTR );
    }
    const int sync_error = sync_result == 0 ? 0 : errno;
    const int close_result = ::close( descriptor );
    const int close_error = close_result == 0 ? 0 : errno;
    if( sync_error != 0 )
    {
        std::string message = std::string( sync_operation ) + " checkpoint " + description + ": " +
                              std::strerror( sync_error );
        if( close_error != 0 ) message += "; close also failed: " + std::string( std::strerror( close_error ) );
        KARMA_THROW( InvalidArgument( message ) );
    }
    if( close_error != 0 )
        KARMA_THROW( InvalidArgument( std::string( "close checkpoint " ) + description +
                                      " after successful sync: " + std::strerror( close_error ) ) );
}
}  // namespace

void remove_checkpoint_temporaries( std::string_view directory, std::string_view snapshot,
                                    std::string_view staged )
{
    const auto remove_path = []( std::string_view name, bool require_directory ) -> std::string {
        const std::filesystem::path path( name );
        std::error_code error;
        const auto status = std::filesystem::symlink_status( path, error );
        if( error == std::errc::no_such_file_or_directory ||
            ( !error && status.type() == std::filesystem::file_type::not_found ) )
            return {};
        if( error ) return "inspect checkpoint temporary path '" + path.string() + "': " + error.message();
        const bool expected_type = require_directory ? std::filesystem::is_directory( status )
                                                     : std::filesystem::is_regular_file( status );
        const char* description = require_directory ? "reservation directory" : "file";
        const std::string operation = "remove checkpoint temporary " + std::string( description ) + " '" +
                                      path.string() + "': ";
        if( !expected_type ) return operation + "path has the wrong type";
        const bool removed = std::filesystem::remove( path, error );
        if( error ) return operation + error.message();
        if( removed ) return {};
        const auto remaining = std::filesystem::symlink_status( path, error );
        if( error == std::errc::no_such_file_or_directory ||
            ( !error && remaining.type() == std::filesystem::file_type::not_found ) )
            return {};
        return operation + ( error ? error.message() : "path still exists" );
    };

    std::string failures;
    for( const auto& error : { remove_path( snapshot, false ), remove_path( staged, false ),
                               remove_path( directory, true ) } )
        if( !error.empty() ) failures += ( failures.empty() ? "" : "; " ) + error;
    if( !failures.empty() ) KARMA_THROW( InvalidArgument( failures ) );
}

void check_checkpoint_barrier( int error, std::string_view operation )
{
    if( error != MPI_SUCCESS )
        KARMA_THROW( InvalidArgument( std::string( operation ) + " failed with MPI error " + std::to_string( error ) ) );
}

CheckpointPublicationState checkpoint_result_collective_failed( CheckpointPublicationState state ) noexcept
{
    return state == CheckpointPublicationState::Published ? state : CheckpointPublicationState::Indeterminate;
}

std::string checkpoint_publication_result_error( CheckpointPublicationState state, std::string_view error )
{
    const char* publication = state == CheckpointPublicationState::Published
                                  ? "checkpoint destination was published, but"
                                  : "checkpoint publication is indeterminate; the destination may have been published because";
    return std::string( publication ) + " collectively propagating the rename result failed: " + std::string( error );
}

[[noreturn]] void throw_checkpoint_publication_durability_error( std::string_view error )
{
    KARMA_THROW( InvalidArgument(
        "checkpoint destination has already been atomically replaced but crash durability could not be confirmed: " +
        std::string( error ) ) );
}

void sync_checkpoint_file( std::string_view filename )
{
    sync_path( filename, O_RDONLY, false, "file" );
}

void sync_checkpoint_directory( std::string_view directory )
{
    sync_path( directory, O_RDONLY, true, "directory" );
}
}  // namespace KARMA::detail

namespace
{
constexpr std::uint32_t schema_version = 6;
constexpr std::uint32_t maximum_checkpoint_variables = 100000;
constexpr std::uint32_t maximum_manifest_records = 1000000;
constexpr std::uint32_t maximum_manifest_bytes = 256u * 1024u * 1024u;
constexpr std::size_t maximum_chunk_records = 1024;
constexpr const char* manifest_tag_name = "KARMA_FIELD_MANIFEST";
constexpr const char* applicability_tag_name = "KARMA_V6_APPLICABILITY_ID";
constexpr const char* applicability_global_ids_tag_name = "KARMA_V6_APPLICABILITY_GLOBAL_IDS";
constexpr const char* stable_entity_id_tag_name = "KARMA_V6_STABLE_ENTITY_ID";
constexpr const char* chunk_id_tag_name = "KARMA_V6_CHUNK_ID";
constexpr const char* field_tag_prefix = "KARMA_V6_F";

enum class StorageBinding : std::uint32_t { GeometryVertex, ElementPacked, GlobalDofArray };
enum class TagStorage : std::uint32_t { Dense, Sparse };

struct BasisDescriptor
{
    KARMA::ReferenceCell cell{};
    KARMA::CoefficientRepresentation representation{};
    std::uint64_t hash{};
    std::vector< unsigned char > descriptor;
};

struct Layout
{
    unsigned variable{};
    KARMA::GeometryVariant variant{};
    StorageBinding binding{};
    std::string tag_name;
    std::uint64_t descriptor_hash{};
    std::vector< unsigned char > descriptor;
    moab::EntityType entity_type = moab::MBMAXTYPE;
    TagStorage tag_storage{};
    unsigned components{};
    unsigned coefficients_per_component{};
    std::uint64_t entity_count{};
    std::uint64_t candidate_count{};
    std::optional< std::uint64_t > applicability_id;
    std::vector< std::size_t > basis_to_connectivity;
    std::uint64_t record_count{};
    std::vector< std::uint32_t > chunk_ids;
    std::vector< std::uint64_t > chunk_checksums;
    std::vector< std::uint32_t > chunk_records;
};

struct VariableMetadata
{
    KARMA::VariableSpec specification;
    std::vector< BasisDescriptor > bases;
    std::vector< Layout > layouts;
};

struct Metadata
{
    unsigned dimension{};
    std::string system_name;
    std::vector< VariableMetadata > variables;
};

struct EntityPayload
{
    std::vector< PetscScalar > values;
    std::vector< unsigned char > present;
};

struct Payload
{
    Layout layout;
    std::map< moab::EntityHandle, EntityPayload > entities;
    std::map< KARMA::DofKey, PetscScalar > records;
};

void moab_check( moab::ErrorCode error, const std::string& operation )
{
    if( error != moab::MB_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( operation + " failed with MOAB error " + std::to_string( error ) ) );
}

void petsc_check( PetscErrorCode error, const std::string& operation )
{
    if( error != PETSC_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( operation + " failed with PETSc error " + std::to_string( error ) ) );
}

void mpi_check( int error, const std::string& operation )
{
    if( error != MPI_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( operation + " failed with MPI error " + std::to_string( error ) ) );
}

int checked_int( std::size_t size, const char* description )
{
    if( size > static_cast< std::size_t >( std::numeric_limits< int >::max() ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( description ) + " exceeds int range" ) );
    return static_cast< int >( size );
}

std::string extension( std::string_view filename )
{
    std::string value = std::filesystem::path( filename ).extension().string();
    std::transform( value.begin(), value.end(), value.begin(), []( unsigned char c ) {
        return static_cast< char >( std::tolower( c ) );
    } );
    return value;
}

bool is_h5m( std::string_view filename )
{
    const std::string suffix = extension( filename );
    return suffix == ".h5m" || suffix == ".mhdf";
}

void validate_format( std::string_view filename, bool checkpoint )
{
    const std::string suffix = extension( filename );
    if( suffix == ".exo" || suffix == ".ex2" || suffix == ".e" || suffix == ".gen" || suffix == ".exodus" )
        KARMA_THROW( KARMA::InvalidArgument( "ExodusII is unavailable because MOAB was built without NetCDF" ) );
    if( checkpoint && !is_h5m( filename ) )
        KARMA_THROW( KARMA::InvalidArgument( "exact checkpoints require an H5M filename" ) );
}

void append_word( std::vector< unsigned char >& bytes, std::uint32_t value )
{
    for( unsigned shift = 0; shift < 32; shift += 8 ) bytes.push_back( static_cast< unsigned char >( value >> shift ) );
}

void append_u64( std::vector< unsigned char >& bytes, std::uint64_t value )
{
    append_word( bytes, static_cast< std::uint32_t >( value ) );
    append_word( bytes, static_cast< std::uint32_t >( value >> 32 ) );
}

void append_signed_word( std::vector< unsigned char >& bytes, int value )
{
    append_word( bytes, static_cast< std::uint32_t >( static_cast< std::int32_t >( value ) ) );
}

void append_string( std::vector< unsigned char >& bytes, std::string_view value )
{
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( value.size() ) ) );
    bytes.insert( bytes.end(), value.begin(), value.end() );
}

void append_bytes( std::vector< unsigned char >& output, std::span< const unsigned char > value )
{
    append_word( output, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( value.size() ) ) );
    output.insert( output.end(), value.begin(), value.end() );
}

std::uint32_t read_word( std::span< const unsigned char > bytes, std::size_t& offset )
{
    if( offset > bytes.size() || bytes.size() - offset < 4 )
        KARMA_THROW( KARMA::InvalidArgument( "truncated checkpoint manifest" ) );
    std::uint32_t value = 0;
    for( unsigned shift = 0; shift < 32; shift += 8 ) value |= static_cast< std::uint32_t >( bytes[offset++] ) << shift;
    return value;
}

std::uint64_t read_u64( std::span< const unsigned char > bytes, std::size_t& offset )
{
    const std::uint64_t low = read_word( bytes, offset );
    return low | ( static_cast< std::uint64_t >( read_word( bytes, offset ) ) << 32 );
}

std::string read_string( std::span< const unsigned char > bytes, std::size_t& offset )
{
    const std::size_t size = read_word( bytes, offset );
    if( size > maximum_manifest_bytes || offset > bytes.size() || size > bytes.size() - offset )
        KARMA_THROW( KARMA::InvalidArgument( "invalid checkpoint manifest string" ) );
    std::string value( reinterpret_cast< const char* >( bytes.data() + offset ), size );
    offset += size;
    return value;
}

std::vector< unsigned char > read_bytes( std::span< const unsigned char > bytes, std::size_t& offset )
{
    const std::size_t size = read_word( bytes, offset );
    if( size > maximum_manifest_bytes || offset > bytes.size() || size > bytes.size() - offset )
        KARMA_THROW( KARMA::InvalidArgument( "invalid checkpoint manifest payload" ) );
    std::vector< unsigned char > value( bytes.begin() + static_cast< std::ptrdiff_t >( offset ),
                                        bytes.begin() + static_cast< std::ptrdiff_t >( offset + size ) );
    offset += size;
    return value;
}

std::uint64_t descriptor_hash( std::span< const unsigned char > bytes )
{
    std::uint64_t result = 1469598103934665603ULL;
    for( const unsigned char byte : bytes )
    {
        result ^= byte;
        result *= 1099511628211ULL;
    }
    return result;
}

std::vector< unsigned char > basis_descriptor( const KARMA::KARMABasis& basis )
{
    std::vector< unsigned char > bytes;
    append_string( bytes, basis.name() );
    append_string( bytes, basis.compatibility_id() );
    const auto key = basis.key();
    append_word( bytes, static_cast< unsigned >( key.cell ) );
    append_word( bytes, static_cast< unsigned >( key.family ) );
    append_word( bytes, key.order );
    append_word( bytes, static_cast< unsigned >( key.continuity ) );
    append_word( bytes, static_cast< unsigned >( basis.coefficient_representation() ) );
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( basis.size() ) ) );
    const auto points = basis.interpolation_points();
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( points.size() ) ) );
    for( const auto& point : points )
        for( unsigned d = 0; d < 3; ++d )
        {
            const double value = point( d );
            append_u64( bytes, std::bit_cast< std::uint64_t >( value ) );
        }
    const auto associations = basis.dof_associations();
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( associations.size() ) ) );
    for( const auto& association : associations )
    {
        append_word( bytes, association.entity_dimension );
        append_word( bytes, association.local_entity_index );
        append_word( bytes, association.slot );
    }
    const auto transforms = basis.orientation_transforms();
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( transforms.size() ) ) );
    for( const auto& transform : transforms )
    {
        append_word( bytes, transform.entity_dimension );
        append_word( bytes, transform.local_entity_index );
        append_word( bytes, transform.orientation_code );
        append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( transform.local_to_canonical.size() ) ) );
        for( const auto value : transform.local_to_canonical ) append_word( bytes, static_cast< std::uint32_t >( value ) );
        append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( transform.signs.size() ) ) );
        for( const int sign : transform.signs ) append_signed_word( bytes, sign );
    }
    return bytes;
}

std::vector< unsigned char > field_descriptor( const KARMA::VariableSpec& variable,
                                               const Layout& layout,
                                               std::span< const unsigned char > basis )
{
    std::vector< unsigned char > bytes;
    append_string( bytes, variable.name );
    append_word( bytes, static_cast< unsigned >( variable.family ) );
    append_word( bytes, variable.order );
    append_word( bytes, static_cast< unsigned >( variable.continuity ) );
    append_word( bytes, variable.components );
    append_word( bytes, static_cast< unsigned >( layout.variant ) );
    append_word( bytes, static_cast< unsigned >( layout.binding ) );
    append_word( bytes, static_cast< unsigned >( layout.entity_type ) );
    append_word( bytes, static_cast< unsigned >( layout.tag_storage ) );
    append_word( bytes, layout.components );
    append_word( bytes, layout.coefficients_per_component );
    append_u64( bytes, layout.entity_count );
    append_u64( bytes, layout.candidate_count );
    append_word( bytes, layout.applicability_id ? 1u : 0u );
    if( layout.applicability_id ) append_u64( bytes, *layout.applicability_id );
    append_word( bytes, static_cast< std::uint32_t >(
                            KARMA::detail::checked_mpi_count( layout.basis_to_connectivity.size() ) ) );
    for( const auto ordinal : layout.basis_to_connectivity )
        append_word( bytes, static_cast< std::uint32_t >( ordinal ) );
    append_bytes( bytes, basis );
    return bytes;
}

std::string field_tag_name( unsigned variable, KARMA::GeometryVariant variant, StorageBinding binding,
                            std::uint64_t hash );

constexpr std::uint32_t petsc_real_kind()
{
#if defined(PETSC_USE_REAL_SINGLE)
    return 1;
#elif defined(PETSC_USE_REAL_DOUBLE)
    return 2;
#elif defined(PETSC_USE_REAL___FLOAT128)
    return 3;
#elif defined(PETSC_USE_REAL___FP16)
    return 4;
#else
    return 0;
#endif
}

std::vector< unsigned char > encode_metadata( const Metadata& metadata )
{
    std::vector< unsigned char > bytes;
    append_word( bytes, schema_version );
    append_word( bytes, metadata.dimension );
    append_word( bytes, sizeof( PetscScalar ) );
    append_word( bytes, petsc_real_kind() );
    append_word( bytes, 0u );
#if defined(PETSC_USE_COMPLEX)
    append_word( bytes, 1 );
#else
    append_word( bytes, 0 );
#endif
    append_string( bytes, metadata.system_name );
    append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( metadata.variables.size() ) ) );
    for( const auto& variable : metadata.variables )
    {
        const auto& specification = variable.specification;
        append_string( bytes, specification.name );
        append_word( bytes, static_cast< unsigned >( specification.family ) );
        append_word( bytes, specification.order );
        append_word( bytes, static_cast< unsigned >( specification.continuity ) );
        append_word( bytes, specification.components );
        append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( variable.bases.size() ) ) );
        for( const auto& basis : variable.bases )
        {
            append_word( bytes, static_cast< unsigned >( basis.cell ) );
            append_word( bytes, static_cast< unsigned >( basis.representation ) );
            append_u64( bytes, basis.hash );
            append_bytes( bytes, basis.descriptor );
        }
        append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( variable.layouts.size() ) ) );
        for( const auto& layout : variable.layouts )
        {
            append_word( bytes, static_cast< unsigned >( layout.variant ) );
            append_word( bytes, static_cast< unsigned >( layout.binding ) );
            append_string( bytes, layout.tag_name );
            append_u64( bytes, layout.descriptor_hash );
            append_bytes( bytes, layout.descriptor );
            append_word( bytes, static_cast< unsigned >( layout.entity_type ) );
            append_word( bytes, static_cast< unsigned >( layout.tag_storage ) );
            append_word( bytes, layout.components );
            append_word( bytes, layout.coefficients_per_component );
            append_u64( bytes, layout.entity_count );
            append_u64( bytes, layout.candidate_count );
            append_word( bytes, layout.applicability_id ? 1u : 0u );
            if( layout.applicability_id ) append_u64( bytes, *layout.applicability_id );
             append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( layout.basis_to_connectivity.size() ) ) );
             for( const auto ordinal : layout.basis_to_connectivity ) append_word( bytes, static_cast< std::uint32_t >( ordinal ) );
             append_u64( bytes, layout.record_count );
             append_word( bytes, static_cast< std::uint32_t >( KARMA::detail::checked_mpi_count( layout.chunk_records.size() ) ) );
             for( std::size_t chunk = 0; chunk < layout.chunk_records.size(); ++chunk )
             {
                 append_word( bytes, layout.chunk_ids.at( chunk ) );
                 append_word( bytes, layout.chunk_records[chunk] );
                 append_u64( bytes, layout.chunk_checksums.at( chunk ) );
             }
         }

    }
    if( bytes.size() > maximum_manifest_bytes ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint manifest is too large" ) );
    return bytes;
}

Metadata decode_metadata( std::span< const unsigned char > encoded )
{
    if( encoded.empty() || encoded.size() > maximum_manifest_bytes )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint manifest size is invalid" ) );
    std::size_t offset = 0;
    if( read_word( encoded, offset ) != schema_version )
        KARMA_THROW( KARMA::InvalidArgument( "unsupported checkpoint schema version" ) );
    Metadata result;
    result.dimension = read_word( encoded, offset );
    if( result.dimension < 1 || result.dimension > 3 )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint mesh dimension is out of range" ) );
    if( read_word( encoded, offset ) != sizeof( PetscScalar ) )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint PetscScalar size does not match this build" ) );
    if( read_word( encoded, offset ) != petsc_real_kind() )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint PETSc real representation does not match this build" ) );
    if( read_word( encoded, offset ) != 0u )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint scalar encoding is not little-endian" ) );
#if defined(PETSC_USE_COMPLEX)
    constexpr unsigned complex_scalar = 1;
#else
    constexpr unsigned complex_scalar = 0;
#endif
    if( read_word( encoded, offset ) != complex_scalar )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint scalar kind does not match this PETSc build" ) );
    result.system_name = read_string( encoded, offset );
    const std::size_t variable_count = read_word( encoded, offset );
    if( variable_count == 0 || variable_count > maximum_checkpoint_variables )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint variable count is invalid" ) );
    result.variables.reserve( variable_count );
    std::size_t total_records = 0;
    for( std::size_t i = 0; i < variable_count; ++i )
    {
        VariableMetadata metadata;
        auto& variable = metadata.specification;
        variable.name = read_string( encoded, offset );
        const unsigned family = read_word( encoded, offset );
        variable.order = read_word( encoded, offset );
        const unsigned continuity = read_word( encoded, offset );
        variable.components = read_word( encoded, offset );
        if( variable.name.empty() || family > static_cast< unsigned >( KARMA::FieldBasisFamily::UserDefined ) ||
            continuity > static_cast< unsigned >( KARMA::BasisContinuity::SE ) || variable.components == 0 ||
            variable.order > KARMA::maximum_basis_order )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint variable metadata is out of range" ) );
        variable.family = static_cast< KARMA::FieldBasisFamily >( family );
        variable.continuity = static_cast< KARMA::BasisContinuity >( continuity );
        const std::size_t basis_count = read_word( encoded, offset );
        if( basis_count == 0 || basis_count > 17 ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint basis count is invalid" ) );
        std::optional< unsigned > previous_cell;
        for( std::size_t basis = 0; basis < basis_count; ++basis )
        {
            const unsigned cell = read_word( encoded, offset );
            const unsigned representation = read_word( encoded, offset );
            const std::uint64_t hash = read_u64( encoded, offset );
            auto descriptor = read_bytes( encoded, offset );
            if( cell > static_cast< unsigned >( KARMA::ReferenceCell::Pyramid ) ||
                representation > static_cast< unsigned >( KARMA::CoefficientRepresentation::Modal ) ||
                descriptor.empty() || descriptor_hash( descriptor ) != hash )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint basis descriptor is invalid" ) );
            if( previous_cell && *previous_cell >= cell )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint basis descriptors are not canonical" ) );
            previous_cell = cell;
            metadata.bases.push_back( { static_cast< KARMA::ReferenceCell >( cell ),
                                        static_cast< KARMA::CoefficientRepresentation >( representation ), hash,
                                        std::move( descriptor ) } );
        }
        const std::size_t layout_count = read_word( encoded, offset );
        total_records += layout_count;
        if( layout_count == 0 || total_records > maximum_manifest_records )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint layout count is invalid" ) );
        std::set< std::string > tag_names;
        std::optional< unsigned > previous_variant;
        for( std::size_t record = 0; record < layout_count; ++record )
        {
            Layout layout;
            layout.variable = static_cast< unsigned >( i );
            const unsigned variant = read_word( encoded, offset );
            const unsigned binding = read_word( encoded, offset );
            layout.tag_name = read_string( encoded, offset );
            layout.descriptor_hash = read_u64( encoded, offset );
            layout.descriptor = read_bytes( encoded, offset );
            const unsigned entity_type = read_word( encoded, offset );
            const unsigned storage = read_word( encoded, offset );
            layout.components = read_word( encoded, offset );
            layout.coefficients_per_component = read_word( encoded, offset );
            layout.entity_count = read_u64( encoded, offset );
            layout.candidate_count = read_u64( encoded, offset );
            const unsigned has_set = read_word( encoded, offset );
            if( has_set > 1 ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability marker is invalid" ) );
            if( has_set ) layout.applicability_id = read_u64( encoded, offset );
            const std::size_t permutation_size = read_word( encoded, offset );
            if( permutation_size > 1024 ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint permutation is too large" ) );
            for( std::size_t p = 0; p < permutation_size; ++p ) layout.basis_to_connectivity.push_back( read_word( encoded, offset ) );
            layout.record_count = read_u64( encoded, offset );
            const std::size_t chunk_count = read_word( encoded, offset );
            if( chunk_count > maximum_manifest_records )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk count is invalid" ) );
            for( std::size_t chunk = 0; chunk < chunk_count; ++chunk )
            {
                layout.chunk_ids.push_back( read_word( encoded, offset ) );
                layout.chunk_records.push_back( read_word( encoded, offset ) );
                layout.chunk_checksums.push_back( read_u64( encoded, offset ) );
            }
            if( variant > static_cast< unsigned >( KARMA::GeometryVariant::Pyramid14 ) || binding > 2 ||
                entity_type >= moab::MBMAXTYPE || storage > 1 || layout.tag_name.empty() ||
                layout.components != variable.components || layout.coefficients_per_component == 0 ||
                layout.descriptor.empty() || descriptor_hash( layout.descriptor ) != layout.descriptor_hash ||
                layout.entity_count == 0 || layout.entity_count > layout.candidate_count || !tag_names.insert( layout.tag_name ).second )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field layout is invalid" ) );
            if( previous_variant && *previous_variant >= variant )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint layouts are not canonical" ) );
            previous_variant = variant;
            layout.variant = static_cast< KARMA::GeometryVariant >( variant );
            layout.binding = static_cast< StorageBinding >( binding );
            layout.entity_type = static_cast< moab::EntityType >( entity_type );
            layout.tag_storage = static_cast< TagStorage >( storage );
            const bool vertex = layout.binding == StorageBinding::GeometryVertex;
            const bool global = layout.binding == StorageBinding::GlobalDofArray;
            const bool sparse_occupancy = layout.entity_count <= layout.candidate_count / 5;
            const auto expected_type = vertex ? moab::MBVERTEX : KARMA::KARMAReferenceTopology( layout.variant ).topology();
            const auto layout_cell = KARMA::reference_cell( layout.variant );
            const auto basis_record = std::find_if( metadata.bases.begin(), metadata.bases.end(),
                                                    [layout_cell]( const BasisDescriptor& basis ) {
                                                        return basis.cell == layout_cell;
                                                    } );
            const bool nodal_cg = variable.continuity == KARMA::BasisContinuity::cG &&
                                  basis_record != metadata.bases.end() &&
                                  basis_record->representation == KARMA::CoefficientRepresentation::Nodal;
            if( global && ( variable.continuity != KARMA::BasisContinuity::cG ||
                            layout.tag_storage != TagStorage::Sparse || layout.applicability_id ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint global array basis is invalid" ) );
            const std::size_t expected_chunks = static_cast< std::size_t >(
                ( layout.record_count + maximum_chunk_records - 1 ) / maximum_chunk_records );
            if( global && ( layout.record_count == 0 || layout.chunk_records.size() != expected_chunks ||
                            layout.chunk_ids.size() != layout.chunk_records.size() ||
                            layout.chunk_records.size() != layout.chunk_checksums.size() ||
                            std::accumulate( layout.chunk_records.begin(), layout.chunk_records.end(),
                                             std::uint64_t{} ) != layout.record_count ||
                            std::any_of( layout.chunk_records.begin(), layout.chunk_records.end(),
                                         []( std::uint32_t count ) {
                                             return count == 0 || count > maximum_chunk_records;
                                         } ) ||
                            std::any_of( layout.chunk_records.begin(), layout.chunk_records.end() - 1,
                                         []( std::uint32_t count ) { return count != maximum_chunk_records; } ) ||
                            layout.chunk_records.back() !=
                                ( ( layout.record_count - 1 ) % maximum_chunk_records + 1 ) ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint global array chunk layout is invalid" ) );
            if( layout.tag_name != field_tag_name( layout.variable, layout.variant, layout.binding,
                                                   layout.descriptor_hash ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field tag name is invalid" ) );
            if( layout.entity_type != expected_type ||
                ( vertex && ( !nodal_cg || layout.entity_type != moab::MBVERTEX ||
                              layout.coefficients_per_component != 1 ||
                              layout.basis_to_connectivity.empty() ) ) ||
                ( !vertex && !layout.basis_to_connectivity.empty() ) ||
                ( !global && ( layout.record_count != 0 || !layout.chunk_ids.empty() ||
                               !layout.chunk_records.empty() || !layout.chunk_checksums.empty() ) ) ||
                ( !global && ( ( layout.tag_storage == TagStorage::Sparse ) != sparse_occupancy ) ) ||
                ( !global && layout.tag_storage == TagStorage::Sparse && layout.applicability_id ) ||
                ( !global && layout.tag_storage == TagStorage::Dense && layout.entity_count < layout.candidate_count &&
                  !layout.applicability_id ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint binding layout is invalid" ) );
            metadata.layouts.push_back( std::move( layout ) );
        }
        result.variables.push_back( std::move( metadata ) );
    }
    if( offset != encoded.size() ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint manifest has trailing bytes" ) );
    return result;
}

void validate_tag( const moab::Interface& database, moab::Tag tag, moab::DataType expected_type,
                   moab::TagType expected_storage, int expected_bytes, const char* description )
{
    moab::DataType data_type = moab::MB_TYPE_OPAQUE;
    moab::TagType storage = moab::MB_TAG_ANY;
    moab_check( database.tag_get_data_type( tag, data_type ), std::string( "query " ) + description + " type" );
    moab_check( database.tag_get_type( tag, storage ), std::string( "query " ) + description + " storage" );
    if( data_type != expected_type || storage != expected_storage )
        KARMA_THROW( KARMA::InvalidArgument( std::string( description ) + " has an incompatible MOAB tag schema" ) );
    int bytes = 0;
    const moab::ErrorCode width_error = database.tag_get_bytes( tag, bytes );
    if( expected_bytes == moab::MB_VARIABLE_LENGTH )
    {
        if( width_error != moab::MB_VARIABLE_DATA_LENGTH )
            KARMA_THROW( KARMA::InvalidArgument( std::string( description ) + " must be variable-length" ) );
    }
    else
    {
        moab_check( width_error, std::string( "query " ) + description + " width" );
        if( bytes != expected_bytes )
            KARMA_THROW( KARMA::InvalidArgument( std::string( description ) + " has an incompatible MOAB tag width" ) );
    }
}

void set_manifest( moab::Interface& database, const Metadata& metadata )
{
    const auto bytes = encode_metadata( metadata );
    moab::Tag tag = nullptr;
    moab_check( database.tag_get_handle( manifest_tag_name, moab::MB_VARIABLE_LENGTH, moab::MB_TYPE_OPAQUE, tag,
                                         moab::MB_TAG_MESH | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT ),
                "create checkpoint manifest tag" );
    const moab::EntityHandle root = 0;
    const void* pointer = bytes.data();
    const int size = checked_int( bytes.size(), "checkpoint manifest" );
    moab_check( database.tag_set_by_ptr( tag, &root, 1, &pointer, &size ), "set checkpoint manifest" );
}

std::vector< unsigned char > read_manifest_bytes( const moab::Interface& database )
{
    moab::Tag tag = nullptr;
    moab_check( database.tag_get_handle( manifest_tag_name, tag ), "find checkpoint manifest" );
    validate_tag( database, tag, moab::MB_TYPE_OPAQUE, moab::MB_TAG_MESH, moab::MB_VARIABLE_LENGTH,
                  "checkpoint manifest" );
    const moab::EntityHandle root = 0;
    const void* pointer = nullptr;
    int size = 0;
    moab_check( database.tag_get_by_ptr( tag, &root, 1, &pointer, &size ), "read checkpoint manifest" );
    if( pointer == nullptr || size <= 0 || static_cast< std::uint32_t >( size ) > maximum_manifest_bytes )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint manifest is empty or too large" ) );
    const auto* first = static_cast< const unsigned char* >( pointer );
    return { first, first + size };
}

void collective_error( MPI_Comm communicator, std::string_view local_error, bool* communicator_usable = nullptr )
{
    KARMA::detail::collective_signature_error( communicator, local_error, communicator_usable );
}

Metadata broadcast_manifest( MPI_Comm communicator, const moab::Interface* rank_zero_database,
                             const std::string* rank_zero_filename )
{
    int rank = 0;
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(manifest broadcast)" );
    std::vector< unsigned char > bytes;
    std::string local_error;
    if( rank == 0 )
        try
        {
            if( rank_zero_database )
                bytes = read_manifest_bytes( *rank_zero_database );
            else
            {
                moab::Core database;
                moab_check( database.load_file( rank_zero_filename->c_str() ), "load checkpoint manifest" );
                bytes = read_manifest_bytes( database );
            }
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
    collective_error( communicator, std::move( local_error ) );
    std::uint64_t size = bytes.size();
    mpi_check( MPI_Bcast( &size, 1, MPI_UINT64_T, 0, communicator ), "MPI_Bcast(manifest size)" );
    if( size == 0 || size > maximum_manifest_bytes )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint manifest broadcast size is invalid" ) );
    local_error.clear();
    try { bytes.resize( static_cast< std::size_t >( size ) ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint manifest allocation failure"; }
    collective_error( communicator, std::move( local_error ) );
    mpi_check( MPI_Bcast( bytes.data(), static_cast< int >( size ), MPI_UNSIGNED_CHAR, 0, communicator ),
               "MPI_Bcast(manifest bytes)" );
    return decode_metadata( bytes );
}

void require_collective_arguments( MPI_Comm communicator, std::string_view filename,
                                   std::string_view options, bool replicated, std::string_view system_name = {} )
{
    int ranks = 1;
    mpi_check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(I/O arguments)" );
    std::vector< unsigned char > serialized;
    std::string local_error;
    try
    {
        append_string( serialized, filename );
        append_string( serialized, options );
        append_word( serialized, replicated ? 1u : 0u );
        append_string( serialized, system_name );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown I/O argument serialization failure"; }
    collective_error( communicator, std::move( local_error ) );
    int local_size = 0;
    local_error.clear();
    try { local_size = KARMA::detail::checked_mpi_count( serialized.size() ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown I/O argument size failure"; }
    collective_error( communicator, std::move( local_error ) );
    std::vector< int > sizes;
    KARMA::detail::prepare_collectively( communicator, [&] { sizes.resize( static_cast< std::size_t >( ranks ) ); },
                                         "I/O argument count allocation failed" );
    mpi_check( MPI_Allgather( &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(I/O argument sizes)" );
    KARMA::detail::MpiLayout layout;
    local_error.clear();
    try { layout = KARMA::detail::checked_mpi_displacements( sizes, 1 ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown I/O argument gather layout failure"; }
    collective_error( communicator, std::move( local_error ) );
    std::vector< unsigned char > gathered;
    local_error.clear();
    try { gathered.resize( layout.total_count ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown I/O argument gather allocation failure"; }
    collective_error( communicator, std::move( local_error ) );
    mpi_check( MPI_Allgatherv( serialized.data(), local_size, MPI_UNSIGNED_CHAR, gathered.data(), sizes.data(),
                               layout.displacements.data(), MPI_UNSIGNED_CHAR, communicator ),
               "MPI_Allgatherv(I/O arguments)" );
    for( int rank = 0; rank < ranks; ++rank )
    {
        const auto begin = gathered.begin() + layout.displacements[static_cast< std::size_t >( rank )];
        if( sizes[static_cast< std::size_t >( rank )] != local_size ||
            !std::equal( begin, begin + sizes[static_cast< std::size_t >( rank )], serialized.begin() ) )
            KARMA_THROW( KARMA::InvalidArgument( "I/O filename, options, or distribution state differ across ranks" ) );
    }
}

void rank_zero_write( const KARMA::KARMAMesh& mesh, std::string_view filename, std::string_view options )
{
    int rank = 0;
    mpi_check( MPI_Comm_rank( mesh.communicator(), &rank ), "MPI_Comm_rank(rank-zero write)" );
    std::string failure;
    if( rank == 0 )
        try { mesh.write( filename, options ); }
        catch( const std::exception& error ) { failure = error.what(); }
    collective_error( mesh.communicator(), std::move( failure ) );
}

void collective_petsc_check( MPI_Comm communicator, PetscErrorCode error, const char* operation )
{
    collective_error( communicator, error == PETSC_SUCCESS ? std::string_view{} : std::string_view( operation ) );
}

KARMA::GeometryVariant element_variant( const KARMA::KARMAMesh& mesh, moab::EntityHandle element )
{
    const moab::EntityHandle* connectivity = nullptr;
    int count = 0;
    moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get checkpoint topology" );
    return KARMA::KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                     static_cast< std::size_t >( count ) );
}

bool bitwise_equal( const PetscScalar& first, const PetscScalar& second )
{
    return std::memcmp( &first, &second, sizeof( PetscScalar ) ) == 0;
}

#if defined(PETSC_USE_COMPLEX)
constexpr std::size_t scalar_components = 2;
#else
constexpr std::size_t scalar_components = 1;
#endif
constexpr std::size_t scalar_byte_width = scalar_components * sizeof( PetscReal );
constexpr std::size_t global_record_size = KARMA::DofKey::width * sizeof( std::uint64_t ) + scalar_byte_width;
constexpr std::size_t chunk_header_size = 4 * sizeof( std::uint32_t );

std::vector< unsigned char > encode_global_records(
    std::uint32_t chunk, std::map< KARMA::DofKey, PetscScalar >::const_iterator first,
    std::map< KARMA::DofKey, PetscScalar >::const_iterator last )
{
    const auto count = static_cast< std::uint32_t >( std::distance( first, last ) );
    std::vector< unsigned char > bytes;
    bytes.reserve( chunk_header_size + static_cast< std::size_t >( count ) * global_record_size );
    append_word( bytes, schema_version );
    append_word( bytes, chunk );
    append_word( bytes, count );
    append_word( bytes, static_cast< std::uint32_t >( global_record_size ) );
    for( ; first != last; ++first )
    {
        for( const auto word : first->first.words ) append_u64( bytes, word );
        const auto scalar = KARMA::detail::encode_checkpoint_scalar( first->second );
        bytes.insert( bytes.end(), scalar.begin(), scalar.end() );
    }
    return bytes;
}

void validate_dof_key( const KARMA::DofKey& key, unsigned variable )
{
    const std::uint64_t identities = key.words[5];
    if( key.words[0] != variable || key.words[2] > 3 || key.words[3] >= moab::MBMAXTYPE ||
        identities == 0 || identities > ( KARMA::DofKey::width - 6 ) / 3 )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF key is malformed" ) );
    const std::size_t used = 6 + 3 * static_cast< std::size_t >( identities );
    if( std::any_of( key.words.begin() + static_cast< std::ptrdiff_t >( used ), key.words.end(),
                     []( std::uint64_t word ) { return word != 0; } ) )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF key has nonzero padding" ) );
}

std::string field_tag_name( unsigned variable, KARMA::GeometryVariant variant, StorageBinding binding,
                            std::uint64_t hash )
{
    const char* marker = binding == StorageBinding::GeometryVertex ? "_V_" :
                         binding == StorageBinding::GlobalDofArray ? "_A_" : "_P_";
    return field_tag_prefix + std::to_string( variable ) + "_G" +
           std::to_string( static_cast< unsigned >( variant ) ) + marker + std::to_string( hash );
}

std::vector< Payload > build_payloads( KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system,
                                       std::span< const PetscScalar > values, Metadata& metadata )
{
    metadata.dimension = mesh.dimension();
    metadata.system_name = system.name();
    metadata.variables.resize( system.n_variables() );
    std::vector< Payload > payloads;
    for( unsigned variable = 0; variable < system.n_variables(); ++variable )
    {
        auto& variable_metadata = metadata.variables[variable];
        variable_metadata.specification = system.variable( variable );
        std::map< KARMA::GeometryVariant, std::vector< moab::EntityHandle > > groups;
        for( const auto element : mesh.elements() ) groups[element_variant( mesh, element )].push_back( element );
        std::set< KARMA::ReferenceCell > described_cells;
        for( const auto& [variant, elements] : groups )
        {
            const auto cell = KARMA::reference_cell( variant );
            const auto basis = system.basis_registry().create(
                { cell, variable_metadata.specification.family, variable_metadata.specification.order,
                  variable_metadata.specification.continuity } );
            if( described_cells.insert( cell ).second )
            {
                auto descriptor = basis_descriptor( basis );
                variable_metadata.bases.push_back( { cell, basis.coefficient_representation(),
                                                     descriptor_hash( descriptor ), std::move( descriptor ) } );
            }
            const KARMA::KARMAReferenceTopology topology( variant );
            const auto permutation = variable_metadata.specification.continuity == KARMA::BasisContinuity::cG &&
                                             basis.coefficient_representation() == KARMA::CoefficientRepresentation::Nodal
                                         ? topology.node_bijection( basis.interpolation_points() )
                                         : std::nullopt;
            const StorageBinding binding = permutation && mesh.dimension() < 3
                                               ? StorageBinding::GeometryVertex
                                               : variable_metadata.specification.continuity == KARMA::BasisContinuity::cG
                                                     ? StorageBinding::GlobalDofArray
                                                     : StorageBinding::ElementPacked;
            Payload payload;
            payload.layout.variable = variable;
            payload.layout.variant = variant;
            payload.layout.binding = binding;
            payload.layout.entity_type = binding == StorageBinding::GeometryVertex ? moab::MBVERTEX : topology.topology();
            payload.layout.components = variable_metadata.specification.components;
            payload.layout.coefficients_per_component = binding == StorageBinding::GeometryVertex ? 1u : static_cast< unsigned >( basis.size() );
            if( binding == StorageBinding::GeometryVertex ) payload.layout.basis_to_connectivity = *permutation;
            for( const auto element : elements )
            {
                std::vector< PetscInt > indices;
                system.dof_map().dof_indices( element, indices, variable );
                const std::size_t expected = variable_metadata.specification.components * basis.size();
                if( indices.size() != expected ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint element DoF width is invalid" ) );
                if( binding == StorageBinding::ElementPacked )
                {
                    auto& destination = payload.entities[element];
                    destination.values.resize( expected );
                    destination.present.assign( expected, 1 );
                    for( std::size_t i = 0; i < expected; ++i )
                    {
                        if( indices[i] < 0 || static_cast< std::size_t >( indices[i] ) >= values.size() )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF index is out of range" ) );
                        destination.values[i] = values[static_cast< std::size_t >( indices[i] )];
                        if( PetscIsInfOrNanScalar( destination.values[i] ) )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint solution contains a nonfinite value" ) );
                    }
                }
                else if( binding == StorageBinding::GlobalDofArray )
                {
                    const auto& keys = system.dof_map().dof_keys( element );
                    std::vector< PetscInt > all_indices;
                    system.dof_map().dof_indices( element, all_indices );
                    if( keys.size() != all_indices.size() )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF key width is invalid" ) );
                    for( std::size_t i = 0; i < expected; ++i )
                    {
                        const auto position = std::find( all_indices.begin(), all_indices.end(), indices[i] );
                        if( indices[i] < 0 || static_cast< std::size_t >( indices[i] ) >= values.size() ||
                            position == all_indices.end() )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF record is invalid" ) );
                        const PetscScalar value = values[static_cast< std::size_t >( indices[i] )];
                        if( PetscIsInfOrNanScalar( value ) )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint solution contains a nonfinite value" ) );
                        const auto key = keys[static_cast< std::size_t >( position - all_indices.begin() )];
                        const auto [entry, inserted] = payload.records.emplace( key, value );
                        if( !inserted && !bitwise_equal( entry->second, value ) )
                            KARMA_THROW( KARMA::InvalidArgument( "shared checkpoint DoF has inconsistent values" ) );
                    }
                }
                else
                {
                    const moab::EntityHandle* connectivity = nullptr;
                    int count = 0;
                    moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get checkpoint connectivity" );
                    if( static_cast< std::size_t >( count ) != permutation->size() )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint geometry permutation width is invalid" ) );
                    for( unsigned component = 0; component < variable_metadata.specification.components; ++component )
                        for( std::size_t basis_ordinal = 0; basis_ordinal < basis.size(); ++basis_ordinal )
                        {
                            const PetscInt index = indices[component * basis.size() + basis_ordinal];
                            if( index < 0 || static_cast< std::size_t >( index ) >= values.size() )
                                KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF index is out of range" ) );
                            const PetscScalar value = values[static_cast< std::size_t >( index )];
                            if( PetscIsInfOrNanScalar( value ) )
                                KARMA_THROW( KARMA::InvalidArgument( "checkpoint solution contains a nonfinite value" ) );
                            auto& destination = payload.entities[connectivity[permutation->at( basis_ordinal )]];
                            if( destination.values.empty() )
                            {
                                destination.values.resize( variable_metadata.specification.components );
                                destination.present.assign( variable_metadata.specification.components, 0 );
                            }
                            if( destination.present[component] && !bitwise_equal( destination.values[component], value ) )
                                KARMA_THROW( KARMA::InvalidArgument( "shared checkpoint vertex has inconsistent values" ) );
                            destination.values[component] = value;
                            destination.present[component] = 1;
                        }
                }
            }
            moab::Range candidates;
            moab_check( mesh.moab()->get_entities_by_type( 0, payload.layout.entity_type, candidates ),
                        "query checkpoint tag candidates" );
            payload.layout.entity_count = binding == StorageBinding::GlobalDofArray ? elements.size()
                                                                                     : payload.entities.size();
            payload.layout.candidate_count = candidates.size();
            if( payload.layout.entity_count == 0 || payload.layout.entity_count > payload.layout.candidate_count )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field occupancy is invalid" ) );
            payload.layout.tag_storage = binding == StorageBinding::GlobalDofArray
                                             ? TagStorage::Sparse
                                             : payload.layout.entity_count <= payload.layout.candidate_count / 5
                                                   ? TagStorage::Sparse : TagStorage::Dense;
            if( binding == StorageBinding::GlobalDofArray )
                payload.layout.record_count = payload.records.size();
            if( binding != StorageBinding::GlobalDofArray && payload.layout.tag_storage == TagStorage::Dense &&
                payload.layout.entity_count < payload.layout.candidate_count )
                payload.layout.applicability_id =
                    static_cast< std::uint64_t >( variable ) *
                        ( static_cast< unsigned >( KARMA::GeometryVariant::Pyramid14 ) + 1 ) +
                    static_cast< unsigned >( variant ) + 1;
            payload.layout.descriptor = field_descriptor( variable_metadata.specification, payload.layout,
                                                          basis_descriptor( basis ) );
            payload.layout.descriptor_hash = descriptor_hash( payload.layout.descriptor );
            payload.layout.tag_name = field_tag_name( variable, variant, binding, payload.layout.descriptor_hash );
            payloads.push_back( std::move( payload ) );
        }
    }
    return payloads;
}

void set_field_tags( moab::Interface& database, std::vector< Payload >& payloads, Metadata& metadata )
{
    std::uint32_t next_chunk_id = 0;
    for( auto& payload : payloads )
    {
        auto& layout = payload.layout;
        if( layout.binding == StorageBinding::GlobalDofArray )
        {
            moab::Tag chunk_id_tag = nullptr;
            moab_check( database.tag_get_handle( chunk_id_tag_name, 1, moab::MB_TYPE_INTEGER, chunk_id_tag,
                                                 moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                        "create checkpoint chunk identifier tag" );
            auto first = payload.records.cbegin();
            while( first != payload.records.cend() )
            {
                auto last = first;
                const auto count = static_cast< std::uint32_t >(
                    std::min( maximum_chunk_records,
                              static_cast< std::size_t >( std::distance( first, payload.records.cend() ) ) ) );
                std::advance( last, count );
                const std::uint32_t chunk_id = next_chunk_id++;
                auto bytes = encode_global_records( chunk_id, first, last );
                layout.chunk_ids.push_back( chunk_id );
                layout.chunk_records.push_back( count );
                layout.chunk_checksums.push_back( descriptor_hash( bytes ) );
                moab::EntityHandle meshset = 0;
                moab_check( database.create_meshset( moab::MESHSET_SET, meshset ),
                            "create checkpoint global array chunk" );
                const int identifier = checked_int( chunk_id, "checkpoint chunk identifier" );
                moab_check( database.tag_set_data( chunk_id_tag, &meshset, 1, &identifier ),
                            "set checkpoint chunk identifier" );
                const std::string name = layout.tag_name + "_C" + std::to_string( chunk_id );
                moab::Tag chunk_tag = nullptr;
                moab_check( database.tag_get_handle( name.c_str(), checked_int( bytes.size(), "checkpoint chunk" ),
                                                     moab::MB_TYPE_OPAQUE, chunk_tag,
                                                     moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                            "create checkpoint global array chunk tag" );
                moab_check( database.tag_set_data( chunk_tag, &meshset, 1, bytes.data() ),
                            "write checkpoint global array chunk" );
                first = last;
            }
            metadata.variables.at( layout.variable ).layouts.push_back( layout );
            continue;
        }
        moab::Range candidates;
        moab_check( database.get_entities_by_type( 0, layout.entity_type, candidates ), "query checkpoint field candidates" );
        if( candidates.size() != layout.candidate_count )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint candidate count changed during write" ) );
        const std::size_t width = static_cast< std::size_t >( layout.components ) * layout.coefficients_per_component;
        const int bytes = checked_int( width * scalar_byte_width, "checkpoint field tag width" );
        moab::Tag tag = nullptr;
        const moab::TagType storage = layout.tag_storage == TagStorage::Sparse ? moab::MB_TAG_SPARSE : moab::MB_TAG_DENSE;
        moab_check( database.tag_get_handle( layout.tag_name.c_str(), bytes, moab::MB_TYPE_OPAQUE, tag,
                                             storage | moab::MB_TAG_CREAT ),
                    "create checkpoint field tag " + layout.tag_name );
        validate_tag( database, tag, moab::MB_TYPE_OPAQUE, storage, bytes, "checkpoint field" );
        if( layout.tag_storage == TagStorage::Dense )
        {
            std::vector< unsigned char > safe_defaults( candidates.size() * width * scalar_byte_width );
            moab_check( database.tag_set_data( tag, candidates, safe_defaults.data() ), "initialize dense checkpoint field" );
            if( layout.entity_count < layout.candidate_count )
            {
                moab::Tag applicability = nullptr;
                moab_check( database.tag_get_handle( applicability_tag_name, 1, moab::MB_TYPE_INTEGER,
                                                     applicability, moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                            "create checkpoint applicability identifier tag" );
                const std::uint64_t identifier = *layout.applicability_id;
                if( identifier > static_cast< std::uint64_t >( std::numeric_limits< int >::max() ) )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability identifier exceeds int range" ) );
                const int stored_identifier = static_cast< int >( identifier );
                const void* value = &stored_identifier;
                moab::Range selectors;
                moab_check( database.get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &applicability,
                                                                   &value, 1, selectors ),
                            "find checkpoint applicability set" );
                if( selectors.size() > 1 )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability identifier is ambiguous" ) );
                moab::EntityHandle selector = 0;
                if( selectors.empty() )
                {
                    moab_check( database.create_meshset( moab::MESHSET_SET, selector ),
                                "create checkpoint applicability set" );
                    moab_check( database.tag_set_data( applicability, &selector, 1, &stored_identifier ),
                                "set checkpoint applicability identifier" );
                }
                else
                {
                    selector = selectors.front();
                    moab::Range old_participants;
                    moab_check( database.get_entities_by_handle( selector, old_participants ),
                                "read old checkpoint applicability set" );
                    if( !old_participants.empty() )
                        moab_check( database.remove_entities( selector, old_participants ),
                                    "clear checkpoint applicability set" );
                }
                moab::Range participants;
                for( const auto& [entity, ignored] : payload.entities ) { (void)ignored; participants.insert( entity ); }
                moab_check( database.add_entities( selector, participants ), "populate checkpoint applicability set" );
                moab::Tag stable_ids = nullptr;
                const int unset_id = -1;
                moab_check( database.tag_get_handle( stable_entity_id_tag_name, 1, moab::MB_TYPE_INTEGER, stable_ids,
                                                     moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &unset_id ),
                            "create checkpoint stable entity identifier tag" );
                std::vector< int > candidate_ids( candidates.size() );
                std::iota( candidate_ids.begin(), candidate_ids.end(), 0 );
                moab_check( database.tag_set_data( stable_ids, candidates, candidate_ids.data() ),
                            "write checkpoint stable entity identifiers" );
                std::vector< int > participant_ids( participants.size() );
                moab_check( database.tag_get_data( stable_ids, participants, participant_ids.data() ),
                            "read checkpoint applicability stable identifiers" );
                std::sort( participant_ids.begin(), participant_ids.end() );
                moab::Tag applicability_ids = nullptr;
                moab_check( database.tag_get_handle( applicability_global_ids_tag_name, moab::MB_VARIABLE_LENGTH,
                                                     moab::MB_TYPE_INTEGER, applicability_ids,
                                                     moab::MB_TAG_SPARSE | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT ),
                            "create checkpoint applicability stable identifier tag" );
                const void* ids = participant_ids.data();
                const int id_count = checked_int( participant_ids.size(), "checkpoint applicability identifiers" );
                moab_check( database.tag_set_by_ptr( applicability_ids, &selector, 1, &ids, &id_count ),
                            "write checkpoint applicability stable identifiers" );
            }
        }
        for( const auto& [entity, entity_payload] : payload.entities )
        {
            if( std::find( entity_payload.present.begin(), entity_payload.present.end(), 0 ) != entity_payload.present.end() )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field payload is incomplete" ) );
            std::vector< unsigned char > encoded;
            encoded.reserve( entity_payload.values.size() * scalar_byte_width );
            for( const auto value : entity_payload.values )
            {
                const auto scalar = KARMA::detail::encode_checkpoint_scalar( value );
                encoded.insert( encoded.end(), scalar.begin(), scalar.end() );
            }
            moab_check( database.tag_set_data( tag, &entity, 1, encoded.data() ),
                        "set checkpoint field payload" );
        }
        metadata.variables.at( layout.variable ).layouts.push_back( layout );
    }
}

void validate_basis_descriptors( const Metadata& metadata, const KARMA::KARMABasisRegistry& registry,
                                 const KARMA::KARMAMesh* mesh = nullptr )
{
    std::set< KARMA::ReferenceCell > mesh_cells;
    if( mesh )
        for( const auto element : mesh->elements() ) mesh_cells.insert( KARMA::reference_cell( element_variant( *mesh, element ) ) );
    for( const auto& variable : metadata.variables )
    {
        std::set< KARMA::ReferenceCell > described;
        for( const auto& descriptor : variable.bases )
        {
            if( !described.insert( descriptor.cell ).second )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint contains a duplicate basis descriptor" ) );
            const auto basis = registry.create( { descriptor.cell, variable.specification.family,
                                                  variable.specification.order, variable.specification.continuity } );
            const auto actual = basis_descriptor( basis );
            if( basis.coefficient_representation() != descriptor.representation ||
                descriptor_hash( actual ) != descriptor.hash || actual != descriptor.descriptor )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint basis descriptor does not match the registry" ) );
        }
        if( mesh && described != mesh_cells )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint basis descriptors do not cover the mesh" ) );
    }
}

moab::Range expected_participants( const KARMA::KARMAMesh& mesh, const Layout& layout );

void validate_metadata( const Metadata& metadata, const KARMA::KARMASystem& system )
{
    if( metadata.dimension != system.mesh().dimension() || metadata.system_name != system.name() ||
        metadata.variables.size() != system.n_variables() )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint does not match the destination system" ) );
    for( unsigned i = 0; i < system.n_variables(); ++i )
    {
        const auto& expected = metadata.variables[i].specification;
        const auto& actual = system.variable( i );
        if( expected.name != actual.name || expected.family != actual.family || expected.order != actual.order ||
            expected.continuity != actual.continuity || expected.components != actual.components )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint variable metadata does not match the destination system" ) );
        for( const auto& layout : metadata.variables[i].layouts )
        {
            const auto basis = system.basis_registry().create(
                { KARMA::reference_cell( layout.variant ), actual.family, actual.order, actual.continuity } );
            const auto permutation = actual.continuity == KARMA::BasisContinuity::cG &&
                                             basis.coefficient_representation() == KARMA::CoefficientRepresentation::Nodal
                                         ? KARMA::KARMAReferenceTopology( layout.variant ).node_bijection(
                                               basis.interpolation_points() )
                                         : std::nullopt;
            const StorageBinding binding = permutation && system.mesh().dimension() < 3
                                               ? StorageBinding::GeometryVertex
                                               : actual.continuity == KARMA::BasisContinuity::cG
                                                     ? StorageBinding::GlobalDofArray
                                                     : StorageBinding::ElementPacked;
            moab::Range candidates;
            moab_check( system.mesh().moab()->get_entities_by_type( 0,
                        binding == StorageBinding::GeometryVertex ? moab::MBVERTEX
                                                                 : KARMA::KARMAReferenceTopology( layout.variant ).topology(),
                        candidates ), "query canonical checkpoint candidates" );
            std::uint64_t entity_count = 0;
            for( const auto element : system.mesh().elements() )
                if( element_variant( system.mesh(), element ) == layout.variant ) ++entity_count;
            if( binding == StorageBinding::GeometryVertex )
                entity_count = expected_participants( system.mesh(), layout ).size();
            const TagStorage storage = binding == StorageBinding::GlobalDofArray ||
                                               entity_count <= candidates.size() / 5
                                           ? TagStorage::Sparse : TagStorage::Dense;
            if( layout.binding != binding || layout.entity_type !=
                    ( binding == StorageBinding::GeometryVertex ? moab::MBVERTEX
                                                                : KARMA::KARMAReferenceTopology( layout.variant ).topology() ) ||
                layout.tag_storage != storage || layout.components != actual.components ||
                layout.coefficients_per_component !=
                    ( binding == StorageBinding::GeometryVertex ? 1u : static_cast< unsigned >( basis.size() ) ) ||
                layout.entity_count != entity_count || layout.candidate_count != candidates.size() ||
                layout.applicability_id !=
                    ( storage == TagStorage::Dense && entity_count < candidates.size()
                          ? std::optional< std::uint64_t >(
                                static_cast< std::uint64_t >( i ) *
                                    ( static_cast< unsigned >( KARMA::GeometryVariant::Pyramid14 ) + 1 ) +
                                static_cast< unsigned >( layout.variant ) + 1 )
                          : std::nullopt ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint layout is not canonical for the destination mesh" ) );
        }
    }
    validate_basis_descriptors( metadata, system.basis_registry(), &system.mesh() );
}

void validate_chunk_namespace( KARMA::KARMAMesh& mesh, const Metadata& metadata )
{
    std::map< std::uint32_t, std::string > declared;
    std::uint32_t expected_id = 0;
    for( const auto& variable : metadata.variables )
        for( const auto& layout : variable.layouts )
            if( layout.binding == StorageBinding::GlobalDofArray )
                for( std::size_t chunk = 0; chunk < layout.chunk_ids.size(); ++chunk )
                {
                    const auto id = layout.chunk_ids[chunk];
                    if( id != expected_id++ ||
                        !declared.emplace( id, layout.tag_name + "_C" + std::to_string( id ) ).second )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace is not canonical" ) );
                }

    moab::Tag chunk_id_tag = nullptr;
    const moab::ErrorCode id_result = mesh.moab()->tag_get_handle( chunk_id_tag_name, chunk_id_tag );
    if( declared.empty() )
    {
        if( id_result == moab::MB_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace contains an undeclared identifier tag" ) );
        return;
    }
    moab_check( id_result, "find checkpoint chunk identifier tag" );
    validate_tag( *mesh.moab(), chunk_id_tag, moab::MB_TYPE_INTEGER, moab::MB_TAG_SPARSE,
                  sizeof( int ), "checkpoint chunk identifier" );
    moab::Range meshsets;
    for( unsigned type = moab::MBVERTEX; type < moab::MBMAXTYPE; ++type )
    {
        moab::Range tagged;
        moab_check( mesh.moab()->get_entities_by_type_and_tag( 0, static_cast< moab::EntityType >( type ),
                                                                &chunk_id_tag, nullptr, 1, tagged ),
                    "enumerate checkpoint chunk identifiers" );
        if( type != moab::MBENTITYSET && !tagged.empty() )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace contains a non-meshset" ) );
        meshsets.merge( tagged );
    }
    if( meshsets.size() != declared.size() )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace has missing or extra meshsets" ) );
    std::set< std::uint32_t > found_ids;
    for( const auto meshset : meshsets )
    {
        int id = -1;
        moab_check( mesh.moab()->tag_get_data( chunk_id_tag, &meshset, 1, &id ),
                    "read checkpoint chunk identifier" );
        if( id < 0 || !declared.contains( static_cast< std::uint32_t >( id ) ) ||
            !found_ids.insert( static_cast< std::uint32_t >( id ) ).second )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace has an invalid identifier" ) );
    }

    std::map< std::string, moab::EntityHandle > declared_tags;
    for( const auto meshset : meshsets )
    {
        int id = -1;
        moab_check( mesh.moab()->tag_get_data( chunk_id_tag, &meshset, 1, &id ),
                    "read checkpoint chunk identifier" );
        declared_tags.emplace( declared.at( static_cast< std::uint32_t >( id ) ), meshset );
    }
    std::set< std::string > actual_tags;
    std::vector< moab::Tag > tags;
    moab_check( mesh.moab()->tag_get_tags( tags ), "enumerate checkpoint chunk tags" );
    for( const auto tag : tags )
    {
        std::string name;
        moab_check( mesh.moab()->tag_get_name( tag, name ), "read checkpoint chunk tag name" );
        if( name.starts_with( field_tag_prefix ) && name.find( "_A_" ) != std::string::npos &&
            name.rfind( "_C" ) != std::string::npos )
        {
            actual_tags.insert( name );
            const auto expected = declared_tags.find( name );
            if( expected == declared_tags.end() ) continue;
            for( unsigned type = moab::MBVERTEX; type < moab::MBMAXTYPE; ++type )
            {
                moab::Range tagged;
                moab_check( mesh.moab()->get_entities_by_type_and_tag( 0, static_cast< moab::EntityType >( type ),
                                                                        &tag, nullptr, 1, tagged ),
                            "enumerate checkpoint chunk payload entities" );
                if( type != moab::MBENTITYSET && !tagged.empty() )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk payload is attached to a non-meshset" ) );
                if( type == moab::MBENTITYSET &&
                    ( tagged.size() != 1 || tagged.front() != expected->second ) )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk payload meshset is invalid" ) );
            }
        }
    }
    std::set< std::string > expected_tags;
    for( const auto& [name, meshset] : declared_tags ) { (void)meshset; expected_tags.insert( name ); }
    if( actual_tags != expected_tags )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk namespace has undeclared payload tags" ) );
}

moab::Range expected_participants( const KARMA::KARMAMesh& mesh, const Layout& layout )
{
    moab::Range expected;
    for( const auto element : mesh.elements() )
        if( element_variant( mesh, element ) == layout.variant )
        {
            if( layout.binding == StorageBinding::ElementPacked )
                expected.insert( element );
            else
            {
                const moab::EntityHandle* connectivity = nullptr;
                int count = 0;
                moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get checkpoint connectivity" );
                expected.insert( connectivity, connectivity + count );
            }
        }
    return expected;
}

moab::Range participants_for( KARMA::KARMAMesh& mesh, const Layout& layout, moab::Tag tag )
{
    moab::Range participants;
    if( layout.tag_storage == TagStorage::Sparse )
        moab_check( mesh.moab()->get_entities_by_type_and_tag( 0, layout.entity_type, &tag, nullptr, 1, participants ),
                    "query sparse checkpoint field membership" );
    else if( layout.entity_count < layout.candidate_count )
    {
        if( !layout.applicability_id ||
            *layout.applicability_id > static_cast< std::uint64_t >( std::numeric_limits< int >::max() ) )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability identifier is invalid" ) );
        moab::Tag applicability = nullptr;
        moab_check( mesh.moab()->tag_get_handle( applicability_tag_name, applicability ),
                    "find checkpoint applicability identifier tag" );
        const int identifier = static_cast< int >( *layout.applicability_id );
        const void* value = &identifier;
        moab::Range selectors;
        moab_check( mesh.moab()->get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &applicability,
                                                               &value, 1, selectors ),
                    "find checkpoint applicability set" );
        if( selectors.size() != 1 )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability set is invalid" ) );
        const moab::EntityHandle selector = selectors.front();
        moab::Tag applicability_ids = nullptr;
        moab_check( mesh.moab()->tag_get_handle( applicability_global_ids_tag_name, applicability_ids ),
                    "find checkpoint applicability stable identifiers" );
        validate_tag( *mesh.moab(), applicability_ids, moab::MB_TYPE_INTEGER, moab::MB_TAG_SPARSE,
                      moab::MB_VARIABLE_LENGTH, "checkpoint applicability stable identifiers" );
        const void* ids_data = nullptr;
        int id_count = 0;
        moab_check( mesh.moab()->tag_get_by_ptr( applicability_ids, &selector, 1, &ids_data, &id_count ),
                    "read checkpoint applicability stable identifiers" );
        if( ids_data == nullptr || id_count < 0 || static_cast< std::uint64_t >( id_count ) != layout.entity_count )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability stable identifiers are invalid" ) );
        const auto* ids = static_cast< const int* >( ids_data );
        if( std::any_of( ids, ids + id_count, []( int id ) { return id < 0; } ) ||
            std::adjacent_find( ids, ids + id_count, std::greater_equal< int >{} ) != ids + id_count )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability stable identifiers are not canonical" ) );
        moab::Range candidates;
        moab_check( mesh.moab()->get_entities_by_type( 0, layout.entity_type, candidates ),
                    "query checkpoint applicability candidates" );
        moab::Tag stable_ids = nullptr;
        moab_check( mesh.moab()->tag_get_handle( stable_entity_id_tag_name, stable_ids ),
                    "find destination stable entity identifiers" );
        validate_tag( *mesh.moab(), stable_ids, moab::MB_TYPE_INTEGER, moab::MB_TAG_DENSE,
                      sizeof( int ), "checkpoint stable entity identifier" );
        std::vector< int > candidate_ids( candidates.size() );
        moab_check( mesh.moab()->tag_get_data( stable_ids, candidates, candidate_ids.data() ),
                    "read destination stable entity identifiers" );
        std::map< int, moab::EntityHandle > by_id;
        auto entity = candidates.begin();
        for( const int candidate_id : candidate_ids )
        {
            if( candidate_id < 0 || !by_id.emplace( candidate_id, *entity++ ).second )
                KARMA_THROW( KARMA::InvalidArgument( "destination stable entity identifiers are invalid" ) );
        }
        for( int i = 0; i < id_count; ++i )
        {
            const auto found = by_id.find( ids[i] );
            if( found == by_id.end() )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint applicability stable identifier is missing" ) );
            participants.insert( found->second );
        }
    }
    else
        participants = expected_participants( mesh, layout );
    return participants;
}

std::map< KARMA::DofKey, PetscScalar > read_global_records( KARMA::KARMAMesh& mesh, const Layout& layout )
{
    std::map< KARMA::DofKey, PetscScalar > result;
    std::optional< KARMA::DofKey > previous_key;
    moab::Tag chunk_id_tag = nullptr;
    moab_check( mesh.moab()->tag_get_handle( chunk_id_tag_name, chunk_id_tag ),
                "find checkpoint chunk identifier tag" );
    for( std::size_t chunk = 0; chunk < layout.chunk_records.size(); ++chunk )
    {
        const std::uint32_t chunk_id = layout.chunk_ids[chunk];
        const std::string name = layout.tag_name + "_C" + std::to_string( chunk_id );
        moab::Tag tag = nullptr;
        moab_check( mesh.moab()->tag_get_handle( name.c_str(), tag ), "find checkpoint global array chunk" );
        const std::size_t bytes = chunk_header_size +
                                  static_cast< std::size_t >( layout.chunk_records[chunk] ) * global_record_size;
        validate_tag( *mesh.moab(), tag, moab::MB_TYPE_OPAQUE, moab::MB_TAG_SPARSE,
                      checked_int( bytes, "checkpoint global array chunk" ), "checkpoint global array chunk" );
        moab::Range participants;
        moab_check( mesh.moab()->get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &tag, nullptr, 1,
                                                                participants ),
                    "find checkpoint global array chunk meshset" );
        if( participants.size() != 1 )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint global array chunk is missing or duplicated" ) );
        const moab::EntityHandle meshset = participants.front();
        int identifier = -1;
        moab_check( mesh.moab()->tag_get_data( chunk_id_tag, &meshset, 1, &identifier ),
                    "read checkpoint chunk identifier" );
        if( identifier != static_cast< int >( chunk_id ) )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint chunk identifier does not match the manifest" ) );
        std::vector< unsigned char > encoded( bytes );
        moab_check( mesh.moab()->tag_get_data( tag, &meshset, 1, encoded.data() ),
                    "read checkpoint global array chunk" );
        if( descriptor_hash( encoded ) != layout.chunk_checksums[chunk] )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint global array chunk checksum is invalid" ) );
        std::size_t offset = 0;
        if( read_word( encoded, offset ) != schema_version || read_word( encoded, offset ) != chunk_id ||
            read_word( encoded, offset ) != layout.chunk_records[chunk] ||
            read_word( encoded, offset ) != global_record_size )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint global array chunk header is invalid" ) );
        for( std::uint32_t record = 0; record < layout.chunk_records[chunk]; ++record )
        {
            KARMA::DofKey key;
            for( auto& word : key.words ) word = read_u64( encoded, offset );
            validate_dof_key( key, layout.variable );
            if( previous_key && !( *previous_key < key ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF keys are not strictly increasing" ) );
            previous_key = key;
            const PetscScalar value = KARMA::detail::decode_checkpoint_scalar(
                std::span< const unsigned char >( encoded ).subspan( offset, scalar_byte_width ) );
            offset += scalar_byte_width;
            if( PetscIsInfOrNanScalar( value ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF value is nonfinite" ) );
            if( !result.emplace( key, value ).second )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint contains a duplicate global DoF key" ) );
        }
    }
    if( result.size() != layout.record_count )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF record count is invalid" ) );
    return result;
}

std::map< PetscInt, PetscScalar > collect_restored_fields( KARMA::KARMAMesh& mesh,
                                                            const KARMA::KARMASystem& system,
                                                            const Metadata& metadata )

{
    validate_metadata( metadata, system );
    validate_chunk_namespace( mesh, metadata );
    std::map< PetscInt, PetscScalar > restored;
    for( unsigned variable = 0; variable < system.n_variables(); ++variable )
    {
        const auto& specification = metadata.variables[variable].specification;
        std::map< KARMA::GeometryVariant, const Layout* > layouts;
        std::map< KARMA::GeometryVariant, std::map< KARMA::DofKey, PetscScalar > > global_records;
        std::map< KARMA::GeometryVariant, std::set< KARMA::DofKey > > consumed_global_keys;
        for( const auto& layout : metadata.variables[variable].layouts )
        {
            if( !layouts.emplace( layout.variant, &layout ).second )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint has duplicate variable geometry layouts" ) );
            if( layout.binding == StorageBinding::GlobalDofArray )
            {
                global_records.emplace( layout.variant, read_global_records( mesh, layout ) );
                continue;
            }
            moab::Range candidates;
            moab_check( mesh.moab()->get_entities_by_type( 0, layout.entity_type, candidates ),
                        "query checkpoint field candidates" );
            if( candidates.size() != layout.candidate_count )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field candidate count does not match the mesh" ) );
            moab::Tag tag = nullptr;
            moab_check( mesh.moab()->tag_get_handle( layout.tag_name.c_str(), tag ), "find checkpoint field tag" );
            const moab::TagType storage = layout.tag_storage == TagStorage::Sparse ? moab::MB_TAG_SPARSE : moab::MB_TAG_DENSE;
            const std::size_t width = static_cast< std::size_t >( layout.components ) * layout.coefficients_per_component;
            validate_tag( *mesh.moab(), tag, moab::MB_TYPE_OPAQUE, storage,
                          checked_int( width * scalar_byte_width, "checkpoint field tag width" ), "checkpoint field" );
            const moab::Range participants = participants_for( mesh, layout, tag );
            const moab::Range expected = expected_participants( mesh, layout );
            if( participants.size() != layout.entity_count || participants != expected )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field participants do not match the manifest layout" ) );
            for( const auto entity : participants )
                if( mesh.moab()->type_from_handle( entity ) != layout.entity_type )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint field contains a wrong entity type" ) );
        }
        for( const auto element : mesh.elements() )
        {
            const auto variant = element_variant( mesh, element );
            const auto found_layout = layouts.find( variant );
            if( found_layout == layouts.end() ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint field layout is undeclared" ) );
            const Layout& layout = *found_layout->second;
            const auto basis = system.basis_registry().create(
                { KARMA::reference_cell( variant ), specification.family, specification.order, specification.continuity } );
            const auto actual_descriptor = field_descriptor( specification, layout, basis_descriptor( basis ) );
            if( actual_descriptor != layout.descriptor || descriptor_hash( actual_descriptor ) != layout.descriptor_hash )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint layout descriptor does not match the basis" ) );
            if( layout.components != specification.components ||
                ( layout.binding == StorageBinding::ElementPacked && layout.coefficients_per_component != basis.size() ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint field width does not match the basis" ) );
            moab::Tag tag = nullptr;
            const std::size_t width = static_cast< std::size_t >( layout.components ) * layout.coefficients_per_component;
            if( layout.binding != StorageBinding::GlobalDofArray )
            {
                moab_check( mesh.moab()->tag_get_handle( layout.tag_name.c_str(), tag ), "find checkpoint field tag" );
                const moab::TagType storage = layout.tag_storage == TagStorage::Sparse ? moab::MB_TAG_SPARSE : moab::MB_TAG_DENSE;
                validate_tag( *mesh.moab(), tag, moab::MB_TYPE_OPAQUE, storage,
                              checked_int( width * scalar_byte_width, "checkpoint field tag width" ), "checkpoint field" );
            }
            std::vector< PetscInt > indices;
            system.dof_map().dof_indices( element, indices, variable );
            if( indices.size() != specification.components * basis.size() )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint destination DoF width is invalid" ) );
            if( layout.binding == StorageBinding::GlobalDofArray )
            {
                const auto& keys = system.dof_map().dof_keys( element );
                std::vector< PetscInt > all_indices;
                system.dof_map().dof_indices( element, all_indices );
                if( keys.size() != all_indices.size() )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint destination DoF key width is invalid" ) );
                auto& records = global_records.at( variant );
                for( std::size_t i = 0; i < indices.size(); ++i )
                {
                    const auto position = std::find( all_indices.begin(), all_indices.end(), indices[i] );
                    if( position == all_indices.end() )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint destination DoF key is missing" ) );
                    const auto& key = keys[static_cast< std::size_t >( position - all_indices.begin() )];
                    const auto value = records.find( key );
                    if( value == records.end() )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint is missing a destination global DoF key" ) );
                    consumed_global_keys[variant].insert( key );
                    const auto [entry, inserted] = restored.emplace( indices[i], value->second );
                    if( !inserted && !bitwise_equal( entry->second, value->second ) )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint global DoF values disagree" ) );
                }
            }
            else if( layout.binding == StorageBinding::ElementPacked )
            {
                std::vector< unsigned char > encoded( width * scalar_byte_width );
                moab_check( mesh.moab()->tag_get_data( tag, &element, 1, encoded.data() ), "read packed checkpoint field" );
                for( std::size_t i = 0; i < width; ++i )
                {
                    const PetscScalar value = KARMA::detail::decode_checkpoint_scalar(
                        std::span< const unsigned char >( encoded ).subspan( i * scalar_byte_width,
                                                                            scalar_byte_width ) );
                    if( PetscIsInfOrNanScalar( value ) ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint field payload is nonfinite" ) );
                    const auto [entry, inserted] = restored.emplace( indices[i], value );
                    if( !inserted && !bitwise_equal( entry->second, value ) )
                        KARMA_THROW( KARMA::InvalidArgument( "checkpoint contains inconsistent shared packed DoF values" ) );
                }
            }
            else
            {
                const auto expected_permutation = KARMA::KARMAReferenceTopology( variant ).node_bijection( basis.interpolation_points() );
                if( !expected_permutation || *expected_permutation != layout.basis_to_connectivity )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint geometry vertex permutation does not match the basis" ) );
                const moab::EntityHandle* connectivity = nullptr;
                int count = 0;
                moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get restore connectivity" );
                for( unsigned component = 0; component < specification.components; ++component )
                    for( std::size_t basis_ordinal = 0; basis_ordinal < basis.size(); ++basis_ordinal )
                    {
                        std::vector< unsigned char > encoded( specification.components * scalar_byte_width );
                        const auto vertex = connectivity[layout.basis_to_connectivity.at( basis_ordinal )];
                        moab_check( mesh.moab()->tag_get_data( tag, &vertex, 1, encoded.data() ), "read vertex checkpoint field" );
                        const PetscScalar value = KARMA::detail::decode_checkpoint_scalar(
                            std::span< const unsigned char >( encoded ).subspan( component * scalar_byte_width,
                                                                                scalar_byte_width ) );
                        if( PetscIsInfOrNanScalar( value ) ) KARMA_THROW( KARMA::InvalidArgument( "checkpoint field payload is nonfinite" ) );
                        const auto [entry, inserted] = restored.emplace( indices[component * basis.size() + basis_ordinal], value );
                        if( !inserted && !bitwise_equal( entry->second, value ) )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint contains inconsistent shared vertex values" ) );
                    }
            }
        }
        for( const auto& [variant, records] : global_records )
            if( consumed_global_keys[variant].size() != records.size() )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint contains an extra global DoF record" ) );
        std::set< KARMA::GeometryVariant > mesh_variants;
        for( const auto element : mesh.elements() ) mesh_variants.insert( element_variant( mesh, element ) );
        if( layouts.size() != mesh_variants.size() )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint field layouts do not match the mesh" ) );
    }
    if( restored.size() != static_cast< std::size_t >( system.n_dofs() ) )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint does not contain every system DoF" ) );
    return restored;
}

void restore_fields( KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system, Vec solution,
                     const Metadata* supplied_metadata = nullptr )
{
    const MPI_Comm communicator = mesh.communicator();
    std::string local_error;
    PetscInt local_size = 0;
    try
    {
        system.validate_vector_layout( solution, "checkpoint destination" );
        petsc_check( VecGetLocalSize( solution, &local_size ), "VecGetLocalSize(checkpoint destination)" );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );

    std::optional< KARMA::PetscVecArrayRead > destination_inspection;
    local_error.clear();
    try { destination_inspection.emplace( solution ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint destination array acquisition failure"; }
    collective_error( communicator, local_error );
    const bool finite = KARMA::detail::checkpoint_values_are_finite( destination_inspection->get(), local_size );
    collective_petsc_check( communicator, destination_inspection->release(),
                            "VecRestoreArrayRead(checkpoint destination)" );
    collective_error( communicator,
                      finite ? std::string_view{} : std::string_view( "checkpoint destination contains a nonfinite value" ) );

    std::map< PetscInt, PetscScalar > restored;
    local_error.clear();
    try
    {
        const Metadata metadata = supplied_metadata ? *supplied_metadata
                                                     : broadcast_manifest( mesh.communicator(), mesh.moab(), nullptr );
        restored = collect_restored_fields( mesh, system, metadata );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    KARMA::PetscOwner< Vec, VecDestroy > staged;
    collective_petsc_check( communicator, VecDuplicate( solution, staged.put() ), "VecDuplicate(checkpoint restore)" );
    PetscInt first = 0;
    PetscInt last = 0;
    collective_petsc_check( communicator, VecGetOwnershipRange( staged.get(), &first, &last ),
                            "VecGetOwnershipRange(checkpoint restore)" );
    collective_petsc_check( communicator, VecSet( staged.get(), 0.0 ), "VecSet(checkpoint staging)" );
    PetscErrorCode set_error = PETSC_SUCCESS;
    for( auto iterator = restored.lower_bound( first ); iterator != restored.end() && iterator->first < last; ++iterator )
        if( set_error == PETSC_SUCCESS )
            set_error = VecSetValue( staged.get(), iterator->first, iterator->second, INSERT_VALUES );
    collective_petsc_check( communicator, set_error, "VecSetValue(checkpoint restore)" );
    const PetscErrorCode begin_error = VecAssemblyBegin( staged.get() );
    const PetscErrorCode end_error = VecAssemblyEnd( staged.get() );
    collective_petsc_check( communicator, begin_error != PETSC_SUCCESS ? begin_error : end_error,
                            "VecAssembly(checkpoint restore)" );

    std::optional< KARMA::PetscVecArrayRead > staged_values;
    std::optional< KARMA::PetscVecArrayWrite > destination_values;
    local_error.clear();
    try
    {
        staged_values.emplace( staged.get() );
        destination_values.emplace( solution );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint commit array acquisition failure"; }
    collective_error( communicator, local_error );

    bool copy_completed = false;
    try
    {
        KARMA::detail::copy_checkpoint_values( staged_values->get(), destination_values->get(), last - first );
        copy_completed = true;
    }
    catch( const std::exception& error )
    {
        local_error = std::string( "checkpoint destination may be partially committed after copy failure: " ) + error.what();
    }
    catch( ... ) { local_error = "checkpoint destination may be partially committed after unknown copy failure"; }

    const PetscErrorCode destination_restore_error = destination_values->release();
    const PetscErrorCode staged_restore_error = staged_values->release();
    if( destination_restore_error != PETSC_SUCCESS )
        local_error = copy_completed
                          ? "checkpoint destination values were copied, but VecRestoreArray(checkpoint commit) failed"
                          : "checkpoint destination may be partially committed, and VecRestoreArray(checkpoint commit) failed";
    else if( staged_restore_error != PETSC_SUCCESS && local_error.empty() )
        local_error = "checkpoint destination values were copied, but VecRestoreArrayRead(checkpoint commit) failed";
    collective_error( communicator, local_error );
}

struct CheckpointTemporaryPaths
{
    std::filesystem::path directory;
    std::filesystem::path snapshot;
    std::filesystem::path staged;
};

class CheckpointReservation
{
  public:
    explicit CheckpointReservation( std::filesystem::path&& directory ) noexcept : _directory( std::move( directory ) ) {}
    ~CheckpointReservation() noexcept
    {
        if( _active ) (void)::rmdir( _directory.c_str() );
    }
    CheckpointReservation( const CheckpointReservation& ) = delete;
    CheckpointReservation& operator=( const CheckpointReservation& ) = delete;
    void activate() noexcept { _active = true; }
    void release() noexcept { _active = false; }
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return _directory; }

  private:
    std::filesystem::path _directory;
    bool _active = false;
};

CheckpointTemporaryPaths checkpoint_temporary_paths( MPI_Comm communicator, std::string_view filename )
{
    int rank = 0;
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(checkpoint temporary name)" );
    std::string name;
    std::string local_error;
    std::optional< CheckpointReservation > reservation;
    if( rank == 0 )
        try
        {
            const std::filesystem::path destination( filename );
            if( !destination.parent_path().empty() && !std::filesystem::is_directory( destination.parent_path() ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint destination parent is not a directory" ) );
            const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
            for( unsigned attempt = 0;; ++attempt )
            {
                auto base = destination.parent_path() /
                    ( "." + destination.filename().string() + ".karma-checkpoint-" + std::to_string( nonce ) + "-" +
                      std::to_string( attempt ) );
                if( ::mkdir( base.c_str(), S_IRWXU ) == 0 )
                {
                    reservation.emplace( std::move( base ) );
                    reservation->activate();
                    name = reservation->directory().string();
                    break;
                }
                if( errno != EEXIST )
                    KARMA_THROW( KARMA::InvalidArgument( "reserve checkpoint temporary directory: " +
                                                         std::string( std::strerror( errno ) ) ) );
            }
        }
        catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    int size = 0;
    local_error.clear();
    if( rank == 0 )
        try { size = checked_int( name.size(), "checkpoint temporary name" ); }
        catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    mpi_check( MPI_Bcast( &size, 1, MPI_INT, 0, communicator ), "MPI_Bcast(checkpoint temporary name size)" );
    local_error.clear();
    try { name.resize( static_cast< std::size_t >( size ) ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint temporary name allocation failure"; }
    collective_error( communicator, std::move( local_error ) );
    mpi_check( MPI_Bcast( name.data(), size, MPI_CHAR, 0, communicator ), "MPI_Bcast(checkpoint temporary name)" );
    std::optional< CheckpointTemporaryPaths > paths;
    local_error.clear();
    try
    {
        paths.emplace( CheckpointTemporaryPaths{ name, std::filesystem::path( name ) / "snapshot.h5m",
                                                 std::filesystem::path( name ) / "staged.h5m" } );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint temporary path construction failure"; }
    collective_error( communicator, std::move( local_error ) );
    if( reservation ) reservation->release();
    return std::move( *paths );
}

void remove_checkpoint_temporaries_and_sync( const CheckpointTemporaryPaths& paths )
{
    KARMA::detail::remove_checkpoint_temporaries( paths.directory.string(), paths.snapshot.string(),
                                                   paths.staged.string() );
    const auto parent = paths.directory.parent_path().empty() ? std::filesystem::path( "." )
                                                              : paths.directory.parent_path();
    KARMA::detail::sync_checkpoint_directory( parent.string() );
}

void stage_checkpoint( const KARMA::KARMAMesh& mesh, const CheckpointTemporaryPaths& paths,
                       std::string_view options, std::vector< Payload >& payloads, Metadata& metadata )
{
    mesh.write( paths.snapshot.string(), options );
    {
        moab::Core staged;
        moab_check( staged.load_file( paths.snapshot.string().c_str() ), "load checkpoint staging snapshot" );
        set_field_tags( staged, payloads, metadata );
        set_manifest( staged, metadata );
        moab_check( staged.write_file( paths.staged.string().c_str() ), "write staged checkpoint" );
    }
}

void collective_barrier( MPI_Comm communicator, const char* operation, bool& communicator_usable )
{
    const int error = MPI_Barrier( communicator );
    if( error != MPI_SUCCESS ) communicator_usable = false;
    KARMA::detail::check_checkpoint_barrier( error, operation );
}

std::string cleanup_checkpoint( MPI_Comm communicator, const CheckpointTemporaryPaths& paths, int rank,
                                bool& communicator_usable )
{
    std::string local_error;
    if( rank == 0 )
        try { remove_checkpoint_temporaries_and_sync( paths ); }
        catch( const std::exception& error ) { local_error = error.what(); }
    if( !communicator_usable ) return local_error;
    try { collective_error( communicator, local_error, &communicator_usable ); }
    catch( const std::exception& error ) { return error.what(); }
    return {};
}

[[noreturn]] void throw_checkpoint_published_cleanup_error( std::string_view error )
{
    KARMA_THROW( KARMA::InvalidArgument( "checkpoint destination was published, but temporary cleanup failed: " +
                                         std::string( error ) ) );
}

[[noreturn]] void rethrow_after_checkpoint_cleanup(
    std::exception_ptr primary, MPI_Comm communicator, const CheckpointTemporaryPaths& paths, int rank,
    bool& communicator_usable, KARMA::detail::CheckpointPublicationState publication_state )
{
    const std::string cleanup_error = cleanup_checkpoint( communicator, paths, rank, communicator_usable );
    if( cleanup_error.empty() ) std::rethrow_exception( primary );
    std::string primary_error = "checkpoint operation failed";
    try { std::rethrow_exception( primary ); }
    catch( const std::exception& error ) { primary_error = error.what(); }
    catch( ... ) {}
    const std::string combined = primary_error + "; checkpoint cleanup also failed: " + cleanup_error;
    if( publication_state == KARMA::detail::CheckpointPublicationState::Published )
        KARMA::detail::throw_checkpoint_publication_durability_error( combined );
    KARMA_THROW( KARMA::InvalidArgument( combined ) );
}

void publish_checkpoint( MPI_Comm communicator, const CheckpointTemporaryPaths& paths, std::string_view filename,
                         KARMA::detail::CheckpointPublicationState& publication_state, bool& communicator_usable )
{
    collective_barrier( communicator, "MPI_Barrier(checkpoint staging close)", communicator_usable );
    int rank = 0;
    const int rank_error = MPI_Comm_rank( communicator, &rank );
    if( rank_error != MPI_SUCCESS ) communicator_usable = false;
    mpi_check( rank_error, "MPI_Comm_rank(checkpoint publish)" );
    std::string local_error;
    if( rank == 0 )
        try { KARMA::detail::sync_checkpoint_file( paths.staged.string() ); }
        catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ), &communicator_usable );

    local_error.clear();
    if( rank == 0 )
        try
        {
            const std::string destination( filename );
            if( ::rename( paths.staged.c_str(), destination.c_str() ) != 0 )
                KARMA_THROW( KARMA::InvalidArgument( "atomically publish checkpoint: " +
                                                     std::string( std::strerror( errno ) ) ) );
            publication_state = KARMA::detail::CheckpointPublicationState::Published;
        }
        catch( const std::exception& error ) { local_error = error.what(); }
    try
    {
        collective_error( communicator, std::move( local_error ), &communicator_usable );
    }
    catch( const std::exception& error )
    {
        if( communicator_usable ) throw;
        publication_state = KARMA::detail::checkpoint_result_collective_failed( publication_state );
        KARMA_THROW( KARMA::InvalidArgument(
            KARMA::detail::checkpoint_publication_result_error( publication_state, error.what() ) ) );
    }
    publication_state = KARMA::detail::CheckpointPublicationState::Published;

    local_error.clear();
    if( rank == 0 )
        try
        {
            const std::filesystem::path destination( filename );
            const auto parent = destination.parent_path().empty() ? std::filesystem::path( "." )
                                                                  : destination.parent_path();
            KARMA::detail::sync_checkpoint_directory( parent.string() );
        }
        catch( const std::exception& error ) { local_error = error.what(); }
    try
    {
        collective_error( communicator, std::move( local_error ), &communicator_usable );
    }
    catch( const KARMA::InvalidArgument& error )
    {
        KARMA::detail::throw_checkpoint_publication_durability_error( error.what() );
    }
}

void write_staged_checkpoint( const KARMA::KARMAMesh& mesh, std::string_view filename, std::string_view options,
                              std::vector< Payload >& payloads, Metadata& metadata )
{
    const auto paths = checkpoint_temporary_paths( mesh.communicator(), filename );
    int rank = 0;
    mpi_check( MPI_Comm_rank( mesh.communicator(), &rank ), "MPI_Comm_rank(checkpoint staging)" );
    bool communicator_usable = true;
    auto publication_state = KARMA::detail::CheckpointPublicationState::Unpublished;
    try
    {
        std::string local_error;
        if( rank == 0 )
            try { stage_checkpoint( mesh, paths, options, payloads, metadata ); }
            catch( const std::exception& error ) { local_error = error.what(); }
        collective_error( mesh.communicator(), std::move( local_error ), &communicator_usable );
        publish_checkpoint( mesh.communicator(), paths, filename, publication_state, communicator_usable );
    }
    catch( ... )
    {
        rethrow_after_checkpoint_cleanup( std::current_exception(), mesh.communicator(), paths, rank,
                                          communicator_usable, publication_state );
    }
    const std::string cleanup_error = cleanup_checkpoint( mesh.communicator(), paths, rank, communicator_usable );
    if( !cleanup_error.empty() ) throw_checkpoint_published_cleanup_error( cleanup_error );
    try { collective_barrier( mesh.communicator(), "MPI_Barrier(checkpoint cleanup)", communicator_usable ); }
    catch( const std::exception& error ) { throw_checkpoint_published_cleanup_error( error.what() ); }
}

std::map< KARMA::DofKey, PetscScalar > collect_owned_values_by_key( const KARMA::KARMASystem& system,
                                                                     Vec vector )
{
    constexpr std::size_t record_size = KARMA::DofKey::width * sizeof( std::uint64_t ) + scalar_byte_width;
    constexpr std::uint64_t records_per_round = maximum_chunk_records;
    const MPI_Comm communicator = system.communicator();
    system.validate_vector_layout( vector, "checkpoint solution" );
    int rank = 0;
    int ranks = 1;
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(checkpoint DoF gather)" );
    mpi_check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(checkpoint DoF gather)" );
    const PetscInt first = system.dof_map().first_owned_dof();
    const PetscInt end = system.dof_map().end_owned_dof();
    const std::size_t local_count = static_cast< std::size_t >( end - first );
    std::vector< std::optional< KARMA::DofKey > > owned_keys;
    std::vector< unsigned char > encoded;
    std::string local_error;
    try
    {
        owned_keys.resize( local_count );
        std::map< KARMA::DofKey, PetscInt > key_indices;
        for( const auto element : system.mesh().elements() )
        {
            std::vector< PetscInt > indices;
            system.dof_map().dof_indices( element, indices );
            const auto& keys = system.dof_map().dof_keys( element );
            if( indices.size() != keys.size() )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF key width is invalid" ) );
            for( std::size_t i = 0; i < keys.size(); ++i )
            {
                const PetscInt index = indices[i];
                if( index < first || index >= end ) continue;
                const std::size_t local_index = static_cast< std::size_t >( index - first );
                if( owned_keys[local_index] && *owned_keys[local_index] != keys[i] )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint owned DoF has conflicting identities" ) );
                const auto [known, inserted] = key_indices.emplace( keys[i], index );
                if( !inserted && known->second != index )
                    KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF identity maps to multiple owned indices" ) );
                owned_keys[local_index] = keys[i];
            }
        }
        if( std::any_of( owned_keys.begin(), owned_keys.end(), []( const auto& key ) { return !key; } ) )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint owned DoF has no stable identity" ) );

        if( local_count > std::numeric_limits< std::size_t >::max() / record_size )
            KARMA_THROW( KARMA::InvalidArgument( "checkpoint owned DoF payload exceeds addressable memory" ) );
        encoded.reserve( local_count * record_size );
        KARMA::PetscVecArrayRead values( vector );
        for( std::size_t i = 0; i < local_count; ++i )
        {
            const PetscScalar value = values.get()[i];
            if( PetscIsInfOrNanScalar( value ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint solution contains a nonfinite value" ) );
            for( const auto word : owned_keys[i]->words ) append_u64( encoded, word );
            const auto scalar = KARMA::detail::encode_checkpoint_scalar( value );
            encoded.insert( encoded.end(), scalar.begin(), scalar.end() );
        }
        const PetscErrorCode restore_error = values.release();
        if( restore_error != PETSC_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "VecRestoreArrayRead failed" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown checkpoint owned DoF preparation failure"; }
    collective_error( communicator, std::move( local_error ) );

    const std::uint64_t local_records = local_count;
    std::vector< std::uint64_t > record_counts;
    if( rank == 0 )
        try { record_counts.resize( static_cast< std::size_t >( ranks ) ); }
        catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    mpi_check( MPI_Gather( &local_records, 1, MPI_UINT64_T, rank == 0 ? record_counts.data() : nullptr, 1,
                           MPI_UINT64_T, 0, communicator ),
               "MPI_Gather(checkpoint owned record counts)" );

    std::map< KARMA::DofKey, PetscScalar > result;
    std::uint64_t received_records = 0;
    for( int source = 0; source < ranks; ++source )
    {
        std::uint64_t source_records = rank == 0 ? record_counts[static_cast< std::size_t >( source )] : 0;
        mpi_check( MPI_Bcast( &source_records, 1, MPI_UINT64_T, 0, communicator ),
                   "MPI_Bcast(checkpoint source record count)" );
        const std::uint64_t rounds = source_records / records_per_round +
                                     ( source_records % records_per_round != 0 ? 1 : 0 );
        for( std::uint64_t round = 0; round < rounds; ++round )
        {
            const std::uint64_t first_record = round * records_per_round;
            const std::uint64_t round_records = std::min( records_per_round, source_records - first_record );
            const int round_bytes = checked_int( static_cast< std::size_t >( round_records ) * record_size,
                                                 "checkpoint gather round" );
            const int send_bytes = rank == source ? round_bytes : 0;
            std::vector< int > byte_counts;
            std::vector< int > displacements;
            std::vector< unsigned char > gathered;
            if( rank == 0 )
                try
                {
                    byte_counts.assign( static_cast< std::size_t >( ranks ), 0 );
                    byte_counts[static_cast< std::size_t >( source )] = round_bytes;
                    const auto layout = KARMA::detail::checked_mpi_displacements( byte_counts, 1 );
                    displacements = layout.displacements;
                    gathered.resize( layout.total_count );
                }
                catch( const std::exception& error ) { local_error = error.what(); }
                catch( ... ) { local_error = "unknown checkpoint gather round allocation failure"; }
            collective_error( communicator, std::move( local_error ) );
            const unsigned char* send = send_bytes == 0 ? nullptr :
                encoded.data() + static_cast< std::size_t >( first_record ) * record_size;
            mpi_check( MPI_Gatherv( send, send_bytes, MPI_UNSIGNED_CHAR,
                                    rank == 0 ? gathered.data() : nullptr,
                                    rank == 0 ? byte_counts.data() : nullptr,
                                    rank == 0 ? displacements.data() : nullptr,
                                    MPI_UNSIGNED_CHAR, 0, communicator ),
                       "MPI_Gatherv(checkpoint owned records)" );
            if( rank == 0 && local_error.empty() )
                try
                {
                    for( std::size_t offset = 0; offset < gathered.size(); offset += record_size )
                    {
                        KARMA::DofKey key;
                        std::size_t cursor = offset;
                        for( auto& word : key.words ) word = read_u64( gathered, cursor );
                        const PetscScalar value = KARMA::detail::decode_checkpoint_scalar(
                            std::span< const unsigned char >( gathered ).subspan( cursor, scalar_byte_width ) );
                        if( PetscIsInfOrNanScalar( value ) )
                            KARMA_THROW( KARMA::InvalidArgument( "checkpoint solution contains a nonfinite value" ) );
                        const auto [entry, inserted] = result.emplace( key, value );
                        if( !inserted )
                        {
                            if( !bitwise_equal( entry->second, value ) )
                                KARMA_THROW( KARMA::InvalidArgument( "distributed checkpoint DoF values disagree" ) );
                            KARMA_THROW( KARMA::InvalidArgument( "distributed checkpoint DoF identity is duplicated" ) );
                        }
                        ++received_records;
                    }
                }
                catch( const std::exception& error ) { local_error = error.what(); }
                catch( ... ) { local_error = "unknown checkpoint gathered DoF validation failure"; }
        }
    }
    collective_error( communicator, std::move( local_error ) );
    KARMA::detail::set_checkpoint_gather_stats( local_records, rank == 0 ? received_records : 0 );
    return result;
}

std::vector< PetscScalar > values_for_system(
    const KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system,
    const std::map< KARMA::DofKey, PetscScalar >& source_values )
{
    std::vector< PetscScalar > values( static_cast< std::size_t >( system.n_dofs() ) );
    std::vector< unsigned char > present( values.size(), 0 );
    std::set< KARMA::DofKey > keys;
    for( const auto element : mesh.elements() )
    {
        std::vector< PetscInt > indices;
        system.dof_map().dof_indices( element, indices );
        const auto& element_keys = system.dof_map().dof_keys( element );
        if( indices.size() != element_keys.size() )
            KARMA_THROW( KARMA::InvalidArgument( "staged checkpoint DoF key width is invalid" ) );
        for( std::size_t i = 0; i < element_keys.size(); ++i )
        {
            const auto value = source_values.find( element_keys[i] );
            if( value == source_values.end() || indices[i] < 0 ||
                static_cast< std::size_t >( indices[i] ) >= values.size() )
                KARMA_THROW( KARMA::InvalidArgument( "staged checkpoint DoF identity is invalid" ) );
            const std::size_t index = static_cast< std::size_t >( indices[i] );
            if( present[index] && !bitwise_equal( values[index], value->second ) )
                KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF index has conflicting values" ) );
            values[index] = value->second;
            present[index] = 1;
            keys.insert( element_keys[i] );
        }
    }
    if( keys.size() != source_values.size() ||
        std::any_of( present.begin(), present.end(), []( unsigned char value ) { return value == 0; } ) )
        KARMA_THROW( KARMA::InvalidArgument( "checkpoint DoF identities do not exactly cover the system" ) );
    return values;
}

void write_partitioned_staged_checkpoint( const KARMA::KARMAMesh& mesh, std::string_view filename,
                                           std::string_view options, const KARMA::KARMASystem& system,
                                           const std::map< KARMA::DofKey, PetscScalar >& source_values )
{
    const auto paths = checkpoint_temporary_paths( mesh.communicator(), filename );
    const std::string parallel_options = options.empty() ? "PARALLEL=WRITE_PART" : std::string( options );
    int rank = 0;
    mpi_check( MPI_Comm_rank( mesh.communicator(), &rank ), "MPI_Comm_rank(partitioned checkpoint staging)" );
    bool communicator_usable = true;
    auto publication_state = KARMA::detail::CheckpointPublicationState::Unpublished;
    try
    {
        std::string local_error;
        try { mesh.write( paths.snapshot.string(), parallel_options ); }
        catch( const std::exception& error ) { local_error = error.what(); }
        collective_error( mesh.communicator(), std::move( local_error ), &communicator_usable );
        local_error.clear();
        if( rank == 0 )
            try
            {
                KARMA::KARMAMesh staged_mesh( mesh.dimension(), MPI_COMM_SELF );
                staged_mesh.read( paths.snapshot.string() );
                KARMA::KARMASystem staged_system( staged_mesh, system.name(), system.basis_registry() );
                for( unsigned variable = 0; variable < system.n_variables(); ++variable )
                    staged_system.add_variable( system.variable( variable ) );
                staged_system.init();
                const auto staged_values = values_for_system( staged_mesh, staged_system, source_values );
                Metadata metadata;
                auto payloads = build_payloads( staged_mesh, staged_system, staged_values, metadata );
                set_field_tags( *staged_mesh.moab(), payloads, metadata );
                set_manifest( *staged_mesh.moab(), metadata );
                staged_mesh.write( paths.staged.string() );
            }
            catch( const std::exception& error ) { local_error = error.what(); }
        collective_error( mesh.communicator(), std::move( local_error ), &communicator_usable );
        publish_checkpoint( mesh.communicator(), paths, filename, publication_state, communicator_usable );
    }
    catch( ... )
    {
        rethrow_after_checkpoint_cleanup( std::current_exception(), mesh.communicator(), paths, rank,
                                          communicator_usable, publication_state );
    }
    const std::string cleanup_error = cleanup_checkpoint( mesh.communicator(), paths, rank, communicator_usable );
    if( !cleanup_error.empty() ) throw_checkpoint_published_cleanup_error( cleanup_error );
    try { collective_barrier( mesh.communicator(), "MPI_Barrier(checkpoint cleanup)", communicator_usable ); }
    catch( const std::exception& error ) { throw_checkpoint_published_cleanup_error( error.what() ); }
}
}  // namespace

namespace KARMA
{
KARMALoadedCheckpoint::KARMALoadedCheckpoint( std::unique_ptr< KARMAMesh > mesh,
                                              std::unique_ptr< KARMASystem > system )
    : _mesh( std::move( mesh ) ), _system( std::move( system ) )
{
    if( !_mesh || !_system || &_system->mesh() != _mesh.get() )
        KARMA_THROW( InvalidArgument( "loaded checkpoint requires a system bound to its owned mesh" ) );
}

KARMAMoabIO::KARMAMoabIO( KARMAMesh& mesh ) noexcept : _input( &mesh ), _output( &mesh ) {}
KARMAMoabIO::KARMAMoabIO( const KARMAMesh& mesh ) noexcept : _output( &mesh ) {}

void KARMAMoabIO::read_mesh( std::string_view filename, std::string_view options )
{
    std::string local_error;
    try
    {
        if( _input == nullptr ) KARMA_THROW( InvalidArgument( "read_mesh requires a mutable mesh" ) );
        validate_format( filename, false );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    const MPI_Comm communicator = _output->communicator();
    collective_error( communicator, std::move( local_error ) );
    require_collective_arguments( communicator, filename, options, false );
    _input->read( filename, options );
}

void KARMAMoabIO::write_mesh( std::string_view filename, std::string_view options ) const
{
    std::string local_error;
    try { validate_format( filename, false ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    if( _output->n_ranks() > 1 )
    {
        collective_error( _output->communicator(), std::move( local_error ) );
        require_collective_arguments( _output->communicator(), filename, options, _output->is_replicated() );
    }
    else if( !local_error.empty() ) KARMA_THROW( InvalidArgument( local_error ) );
    if( _output->n_ranks() == 1 ) _output->write( filename, options );
    else if( _output->is_replicated() )
    {
        if( options.find( "PARALLEL=" ) != std::string_view::npos )
            KARMA_THROW( InvalidArgument( "replicated output does not accept parallel MOAB write options" ) );
        rank_zero_write( *_output, filename, options );
    }
    else
    {
        if( !is_h5m( filename ) ) KARMA_THROW( InvalidArgument( "partitioned output requires H5M" ) );
        const std::string parallel_options = options.empty() ? "PARALLEL=WRITE_PART" : std::string( options );
        if( parallel_options.find( "PARALLEL=WRITE_PART" ) == std::string::npos )
            KARMA_THROW( InvalidArgument( "partitioned output requires PARALLEL=WRITE_PART" ) );
        _output->write( filename, parallel_options );
    }
}

void KARMAMoabIO::write_checkpoint( std::string_view filename, const KARMASystem& system, Vec solution,
                                    std::string_view options )
{
    std::string local_error;
    try
    {
        validate_format( filename, true );
        if( _input == nullptr || &system.mesh() != _input )
            KARMA_THROW( InvalidArgument( "checkpoint writing requires the mutable system mesh" ) );
        if( system.mesh().n_ranks() > 1 && system.mesh().is_replicated() &&
            options.find( "PARALLEL=" ) != std::string_view::npos )
            KARMA_THROW( InvalidArgument( "replicated checkpoint output does not accept parallel MOAB write options" ) );
        if( system.mesh().n_ranks() > 1 && !system.mesh().is_replicated() &&
            options.find( "PARALLEL=WRITE_PART" ) == std::string_view::npos && !options.empty() )
            KARMA_THROW( InvalidArgument( "partitioned checkpoint output requires PARALLEL=WRITE_PART" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    if( system.mesh().n_ranks() > 1 )
    {
        collective_error( system.mesh().communicator(), std::move( local_error ) );
        require_collective_arguments( system.mesh().communicator(), filename, options, system.mesh().is_replicated(),
                                      system.name() );
    }
    else if( !local_error.empty() ) KARMA_THROW( InvalidArgument( local_error ) );
    const auto source_values = collect_owned_values_by_key( system, solution );
    auto& mesh = *_input;
    if( mesh.n_ranks() > 1 && !mesh.is_replicated() )
    {
        write_partitioned_staged_checkpoint( mesh, filename, options, system, source_values );
        return;
    }
    int rank = 0;
    mpi_check( MPI_Comm_rank( mesh.communicator(), &rank ), "MPI_Comm_rank(checkpoint)" );
    local_error.clear();
    Metadata metadata;
    std::vector< Payload > payloads;
    if( rank == 0 )
        try
        {
            const auto values = values_for_system( mesh, system, source_values );
            payloads = build_payloads( mesh, system, values, metadata );
        }
        catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( mesh.communicator(), std::move( local_error ) );
    write_staged_checkpoint( mesh, filename, options, payloads, metadata );
}

void KARMAMoabIO::restore_solution( const KARMASystem& system, Vec solution )
{
    std::string local_error;
    if( &system.mesh() != _input ) local_error = "restore system belongs to another mesh";
    collective_error( system.communicator(), std::move( local_error ) );
    restore_fields( *_input, system, solution );
}

KARMALoadedCheckpoint KARMAMoabIO::load_checkpoint( std::string_view filename, MPI_Comm communicator,
                                                     KARMABasisRegistry registry )
{
    std::string local_error;
    try { validate_format( filename, true ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    require_collective_arguments( communicator, filename, {}, true );

    const std::string file( filename );
    std::optional< Metadata > metadata;
    local_error.clear();
    try
    {
        metadata = broadcast_manifest( communicator, nullptr, &file );
        validate_basis_descriptors( *metadata, registry );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );

    std::unique_ptr< KARMAMesh > mesh;
    local_error.clear();
    try
    {
        mesh = std::make_unique< KARMAMesh >( metadata->dimension, communicator );
        mesh->read( filename );
        validate_basis_descriptors( *metadata, registry, mesh.get() );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );

    std::unique_ptr< KARMASystem > system;
    local_error.clear();
    try
    {
        system = std::make_unique< KARMASystem >( *mesh, metadata->system_name, std::move( registry ) );
        for( const auto& variable : metadata->variables ) system->add_variable( variable.specification );
        system->init();
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, std::move( local_error ) );
    restore_fields( *mesh, *system, system->solution(), &*metadata );
    return KARMALoadedCheckpoint( std::move( mesh ), std::move( system ) );
}
}  // namespace KARMA
