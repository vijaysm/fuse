// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
#include "karma_dof_map.h"

#include "karma_exceptions.h"
#include "karma_reference_topology.h"
#include "../physics/common/impls/collective_signature.h"

#include "moab/CN.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <utility>

namespace
{
constexpr std::size_t key_words = KARMA::DofKey::width;
using Key                       = KARMA::DofKey;
using EntityIdentity            = std::array< std::uint64_t, 3 >;

void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}
void mpi_check( int error, const char* operation )
{
    if( error != MPI_SUCCESS ) KARMA_THROW( KARMA::InvalidArgument( std::string( "MPI error in " ) + operation ) );
}
void moab_check( moab::ErrorCode error, const char* operation )
{
    if( error != moab::MB_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( std::string( "MOAB error in " ) + operation ) );
}

std::uint64_t coordinate_bits( KARMA::Real value )
{
    if( !std::isfinite( value ) ) KARMA_THROW( KARMA::InvalidArgument( "nonfinite mesh coordinate" ) );
    if( value == 0.0 ) value = 0.0;
    return std::bit_cast< std::uint64_t >( value );
}

EntityIdentity geometric_vertex_identity( const KARMA::KARMAMesh& mesh, moab::EntityHandle vertex )
{
    double coordinates[3]{};
    moab_check( mesh.moab()->get_coords( &vertex, 1, coordinates ), "get_coords(DoF vertex identity)" );
    return { coordinate_bits( coordinates[0] ), coordinate_bits( coordinates[1] ), coordinate_bits( coordinates[2] ) };
}

EntityIdentity conventional_entity_identity( const KARMA::KARMAMesh& mesh, moab::EntityHandle entity )
{
    if( mesh.global_id( entity ) < 0 )
        KARMA_THROW( KARMA::InvalidArgument( "DoF identity requires valid KARMA global IDs" ) );
    moab::Tag global_id = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ), "get GLOBAL_ID for DoF identity" );
    int id = 0;
    moab_check( mesh.moab()->tag_get_data( global_id, &entity, 1, &id ), "read GLOBAL_ID for DoF identity" );
    if( id <= 0 ) KARMA_THROW( KARMA::InvalidArgument( "DoF identity requires a positive MOAB GLOBAL_ID" ) );
    return { static_cast< std::uint64_t >( id ), 0, 0 };
}

std::vector< EntityIdentity > canonical_cycle( const std::vector< EntityIdentity >& local, unsigned& orientation )
{
    std::vector< EntityIdentity > best;
    orientation = 0;
    for( unsigned reflected = 0; reflected < 2; ++reflected )
        for( unsigned rotation = 0; rotation < local.size(); ++rotation )
        {
            std::vector< EntityIdentity > candidate( local.size() );
            for( unsigned local_index = 0; local_index < local.size(); ++local_index )
            {
                const unsigned canonical_index = reflected == 0
                                                     ? ( local_index + rotation ) % local.size()
                                                     : ( rotation + local.size() - local_index ) % local.size();
                candidate[canonical_index]     = local[local_index];
            }
            if( best.empty() || candidate < best )
            {
                best        = std::move( candidate );
                orientation = reflected == 0 ? rotation : static_cast< unsigned >( local.size() ) + rotation;
            }
        }
    return best;
}

Key make_key( unsigned variable, unsigned component, unsigned dimension, moab::EntityType type, unsigned slot,
              const std::vector< EntityIdentity >& entities )
{
    if( 6 + 3 * entities.size() > key_words ) KARMA_THROW( KARMA::InvalidArgument( "DoF key exceeds fixed storage" ) );
    Key key{};
    key.words[0]       = variable;
    key.words[1]       = component;
    key.words[2]       = dimension;
    key.words[3]       = static_cast< std::uint64_t >( type );
    key.words[4]       = slot;
    key.words[5]       = entities.size();
    std::size_t offset = 6;
    for( const auto& entity : entities )
        for( const auto word : entity )
            key.words[offset++] = word;
    return key;
}

template < class Words >
std::uint64_t fixed_hash( const Words& words, std::uint64_t seed = 1469598103934665603ULL )
{
    std::uint64_t hash = seed;
    for( const auto word : words )
        for( unsigned byte = 0; byte < 8; ++byte )
        {
            hash ^= ( word >> ( 8 * byte ) ) & 0xffU;
            hash *= 1099511628211ULL;
        }
    return hash;
}

KARMA::GeometryVariant element_variant( const KARMA::KARMAMesh& mesh, moab::EntityHandle element )
{
    const moab::EntityHandle* connectivity = nullptr;
    int count                              = 0;
    moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get_connectivity(DoF variant)" );
    return KARMA::KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                     static_cast< std::size_t >( count ) );
}

struct MeshIdentity
{
    bool replicated = false;
    std::map< moab::EntityHandle, EntityIdentity > vertices;
    std::map< moab::EntityHandle, std::vector< EntityIdentity > > elements;
};

void collective_identity_check( MPI_Comm communicator, int rank, int ranks, const std::string& local_error );

std::vector< std::uint64_t > local_geometric_signature( const KARMA::KARMAMesh& mesh, MeshIdentity& identity )
{
    std::set< EntityIdentity > unique_vertices;
    for( const auto vertex : mesh.vertices() )
    {
        const auto key = geometric_vertex_identity( mesh, vertex );
        if( !unique_vertices.insert( key ).second )
            KARMA_THROW( KARMA::InvalidArgument( "replicated DoF numbering requires unique vertex coordinates" ) );
        identity.vertices.emplace( vertex, key );
    }

    std::set< std::vector< EntityIdentity > > unique_elements;
    std::map< moab::EntityHandle, std::vector< EntityIdentity > > element_signatures;
    for( const auto element : mesh.elements() )
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count                              = 0;
        moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get_connectivity(mesh signature)" );
        std::vector< EntityIdentity > signature;
        signature.push_back( { static_cast< std::uint64_t >( mesh.moab()->type_from_handle( element ) ),
                               static_cast< std::uint64_t >( count ), 0 } );
        std::vector< EntityIdentity > nodes;
        nodes.reserve( static_cast< std::size_t >( count ) );
        for( int i = 0; i < count; ++i )
            nodes.push_back( identity.vertices.at( connectivity[i] ) );
        // Replicated mode proves equality only for the exact reference-node role sequence.
        signature.insert( signature.end(), nodes.begin(), nodes.end() );
        if( !unique_elements.insert( signature ).second )
            KARMA_THROW( KARMA::InvalidArgument(
                "partitioned DoF numbering requires a stable partition identity: replicated mesh has duplicate element signatures" ) );
        element_signatures.emplace( element, std::move( signature ) );
    }
    for( const auto& [element, signature] : element_signatures )
    {
        const auto ordinal = std::distance( unique_elements.begin(), unique_elements.find( signature ) );
        identity.elements.emplace(
            element, std::vector< EntityIdentity >{ { static_cast< std::uint64_t >( ordinal ), 0, 0 } } );
    }

    std::vector< std::uint64_t > result{ static_cast< std::uint64_t >( mesh.dimension() ),
                                         static_cast< std::uint64_t >( unique_vertices.size() ),
                                         static_cast< std::uint64_t >( unique_elements.size() ) };
    for( const auto& vertex : unique_vertices )
        result.insert( result.end(), vertex.begin(), vertex.end() );
    for( const auto& element : unique_elements )
    {
        result.push_back( element.size() );
        for( const auto& word : element )
            result.insert( result.end(), word.begin(), word.end() );
    }
    return result;
}

MeshIdentity establish_identity( const KARMA::KARMAMesh& mesh, MPI_Comm communicator, int rank, int ranks )
{
    MeshIdentity identity;
    if( ranks == 1 )
    {
        for( const auto vertex : mesh.vertices() )
        {
            (void)geometric_vertex_identity( mesh, vertex );
            identity.vertices.emplace( vertex, conventional_entity_identity( mesh, vertex ) );
        }
        for( const auto element : mesh.elements() )
            identity.elements.emplace( element,
                                       std::vector< EntityIdentity >{ conventional_entity_identity( mesh, element ) } );
        return identity;
    }
    if( !mesh.is_replicated() )
    {
        std::string local_error;
        try
        {
            moab::Tag global_id = nullptr;
            moab_check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ), "get GLOBAL_ID for DoF identity" );
            for( const auto vertex : mesh.vertices() )
            {
                int id = 0;
                moab_check( mesh.moab()->tag_get_data( global_id, &vertex, 1, &id ),
                            "read vertex GLOBAL_ID for DoF identity" );
                identity.vertices.emplace( vertex, EntityIdentity{ static_cast< std::uint64_t >( id ), 0, 0 } );
            }
            for( const auto element : mesh.elements() )
            {
                int id = 0;
                moab_check( mesh.moab()->tag_get_data( global_id, &element, 1, &id ),
                            "read element GLOBAL_ID for DoF identity" );
                identity.elements.emplace(
                    element, std::vector< EntityIdentity >{ { static_cast< std::uint64_t >( id ), 0, 0 } } );
            }
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
        collective_identity_check( communicator, rank, ranks, local_error );
        return identity;
    }

    int local_valid = 1;
    std::string local_error;
    std::vector< std::uint64_t > signature;
    try
    {
        signature = local_geometric_signature( mesh, identity );
    }
    catch( const std::exception& error )
    {
        local_valid = 0;
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    try
    {
        (void)KARMA::detail::checked_mpi_count( signature.size() );
    }
    catch( const std::logic_error& )
    {
        local_valid = 0;
    }
    if( !KARMA::detail::collective_validation_succeeded( communicator, local_valid != 0 ) )
        KARMA_THROW( KARMA::InvalidArgument( "mesh signature exceeds MPI count range" ) );
    const int local_words = KARMA::detail::checked_mpi_count( signature.size() );
    std::vector< int > counts;
    KARMA::detail::prepare_collectively( communicator, [&] { counts.resize( static_cast< std::size_t >( ranks ) ); },
                                         "mesh signature count allocation failed" );
    mpi_check( MPI_Allgather( &local_words, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(mesh signature sizes)" );
    KARMA::detail::MpiLayout layout;
    try
    {
        layout = KARMA::detail::checked_mpi_displacements( counts, 1 );
    }
    catch( const std::logic_error& )
    {
        local_valid = 0;
    }
    if( !KARMA::detail::collective_validation_succeeded( communicator, local_valid != 0 ) )
        KARMA_THROW( KARMA::InvalidArgument( "mesh signature displacement overflow" ) );
    std::vector< std::uint64_t > gathered;
    KARMA::detail::prepare_collectively( communicator, [&] { gathered.resize( layout.total_count ); },
                                         "mesh signature gather allocation failed" );
    mpi_check( MPI_Allgatherv( signature.data(), local_words, MPI_UINT64_T, gathered.data(), counts.data(),
                               layout.displacements.data(), MPI_UINT64_T, communicator ),
               "MPI_Allgatherv(mesh signatures)" );
    bool same = true;
    for( int source = 0; source < ranks; ++source )
    {
        const auto source_size = counts[static_cast< std::size_t >( source )];
        const auto begin       = gathered.begin() + layout.displacements[static_cast< std::size_t >( source )];
        same = same && source_size == local_words && std::equal( begin, begin + source_size, signature.begin() );
    }
    const int local_same = same ? 1 : 0;
    int all_same         = 0;
    mpi_check( MPI_Allreduce( &local_same, &all_same, 1, MPI_INT, MPI_MIN, communicator ),
               "MPI_Allreduce(mesh signature agreement)" );
    if( !all_same )
        KARMA_THROW( KARMA::InvalidArgument( "partitioned DoF numbering requires a stable partition identity" ) );
    identity.vertices.clear();
    identity.elements.clear();
    for( const auto vertex : mesh.vertices() )
        identity.vertices.emplace( vertex, conventional_entity_identity( mesh, vertex ) );
    for( const auto element : mesh.elements() )
        identity.elements.emplace( element,
                                   std::vector< EntityIdentity >{ conventional_entity_identity( mesh, element ) } );
    identity.replicated = true;
    return identity;
}

std::vector< EntityIdentity > subentity_identities( const KARMA::KARMAMesh& mesh, const MeshIdentity& identity,
                                                    moab::EntityHandle element,
                                                    const KARMA::DofAssociation& association, unsigned& orientation )
{
    const moab::EntityType type            = mesh.moab()->type_from_handle( element );
    const moab::EntityHandle* connectivity = nullptr;
    int count                              = 0;
    moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get_connectivity(DoF key)" );
    if( association.entity_dimension == 0 )
    {
        orientation = 0;
        return { identity.vertices.at( connectivity[association.local_entity_index] ) };
    }
    if( association.entity_dimension == static_cast< unsigned >( moab::CN::Dimension( type ) ) )
    {
        orientation = 0;
        return identity.elements.at( element );
    }

    const moab::EntityType sub_type =
        moab::CN::SubEntityType( type, association.entity_dimension, association.local_entity_index );
    const int sub_count = moab::CN::VerticesPerEntity( sub_type );
    std::array< int, moab::MAX_SUB_ENTITY_VERTICES > indices{};
    moab::CN::SubEntityVertexIndices( type, association.entity_dimension, association.local_entity_index,
                                      indices.data() );
    std::vector< EntityIdentity > local;
    local.reserve( static_cast< std::size_t >( sub_count ) );
    for( int i = 0; i < sub_count; ++i )
        local.push_back( identity.vertices.at( connectivity[indices[i]] ) );
    return canonical_cycle( local, orientation );
}

unsigned canonical_slot( const KARMA::KARMABasis& basis, const KARMA::DofAssociation& association,
                          unsigned orientation )
{
    if( association.entity_dimension == 0 ) return association.slot;
    const auto transforms = basis.orientation_transforms();
    const auto transform  = std::find_if( transforms.begin(), transforms.end(), [&]( const auto& candidate ) {
        return candidate.entity_dimension == association.entity_dimension &&
               candidate.local_entity_index == association.local_entity_index &&
               candidate.orientation_code == orientation;
    } );
    if( transform == transforms.end() ) return association.slot;
    if( association.slot >= transform->local_to_canonical.size() )
        KARMA_THROW( KARMA::InvalidArgument( "basis orientation transform does not cover the DoF slot" ) );
    return static_cast< unsigned >( transform->local_to_canonical[association.slot] );
}

void collective_identity_check( MPI_Comm communicator, int, int, const std::string& local_error )
{
    KARMA::detail::collective_signature_error( communicator, local_error );
}

std::vector< std::uint64_t > variable_signature( const std::vector< KARMA::VariableSpec >& variables )
{
    std::vector< std::uint64_t > signature{ variables.size() };

    for( const auto& variable : variables )
    {
        signature.push_back( variable.name.size() );
        for( const unsigned char character : variable.name ) signature.push_back( character );
        signature.push_back( static_cast< std::uint64_t >( variable.family ) );
        signature.push_back( variable.order );
        signature.push_back( static_cast< std::uint64_t >( variable.continuity ) );
        signature.push_back( variable.components );
    }
    return signature;
}

std::vector< std::uint64_t > basis_signature( const std::set< KARMA::ReferenceCell >& cells,
                                              const std::vector< KARMA::VariableSpec >& variables,
                                              const KARMA::KARMABasisRegistry& registry )
{
    std::vector< std::uint64_t > signature{ cells.size() };
    for( const auto cell : cells )
    {
        signature.push_back( static_cast< std::uint64_t >( cell ) );
        for( const auto& variable : variables )
        {
            const auto basis = registry.create( { cell, variable.family, variable.order, variable.continuity } );
            signature.push_back( basis.size() );
            const std::string compatibility = basis.compatibility_id();
            signature.push_back( compatibility.size() );
            for( const unsigned char character : compatibility ) signature.push_back( character );
            const auto associations = basis.dof_associations();
            signature.push_back( associations.size() );
            for( const auto& association : associations )
            {
                signature.push_back( association.entity_dimension );
                signature.push_back( association.local_entity_index );
                signature.push_back( association.slot );
            }
            const auto transforms = basis.orientation_transforms();
            signature.push_back( transforms.size() );
            for( const auto& transform : transforms )
            {
                signature.push_back( transform.entity_dimension );
                signature.push_back( transform.local_entity_index );
                signature.push_back( transform.orientation_code );
                signature.push_back( transform.local_to_canonical.size() );
                for( const auto value : transform.local_to_canonical ) signature.push_back( value );
                signature.push_back( transform.signs.size() );
                for( const auto sign : transform.signs ) signature.push_back( static_cast< std::uint64_t >( sign ) );
            }
        }
    }
    return signature;
}

std::set< KARMA::ReferenceCell > gather_cell_types( const KARMA::KARMAMesh& mesh, MPI_Comm communicator,
                                                     int rank, int ranks )
{
    std::set< KARMA::ReferenceCell > local_cells;
    std::string local_error;
    try
    {
        for( const auto element : mesh.elements() )
            local_cells.insert( KARMA::reference_cell( element_variant( mesh, element ) ) );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    std::vector< std::uint64_t > local;
    try
    {
        local.reserve( local_cells.size() );
        for( const auto cell : local_cells ) local.push_back( static_cast< std::uint64_t >( cell ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_identity_check( communicator, rank, ranks, local_error );
    int local_count = 0;
    try
    {
        local_count = KARMA::detail::checked_mpi_count( local.size() );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< int > counts;
    KARMA::detail::prepare_collectively( communicator, [&] { counts.resize( static_cast< std::size_t >( ranks ) ); },
                                         "cell type count allocation failed" );
    mpi_check( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(cell type counts)" );
    KARMA::detail::MpiLayout layout;
    try
    {
        layout = KARMA::detail::checked_mpi_displacements( counts, 1 );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< std::uint64_t > gathered;
    KARMA::detail::prepare_collectively( communicator, [&] { gathered.resize( layout.total_count ); },
                                         "cell type gather allocation failed" );
    mpi_check( MPI_Allgatherv( local.data(), local_count, MPI_UINT64_T, gathered.data(), counts.data(),
                               layout.displacements.data(), MPI_UINT64_T, communicator ),
               "MPI_Allgatherv(cell types)" );
    std::set< KARMA::ReferenceCell > result;
    for( const auto value : gathered ) result.insert( static_cast< KARMA::ReferenceCell >( value ) );
    return result;
}

void require_matching_signature( MPI_Comm communicator, int rank, int ranks,
                                  const std::vector< std::uint64_t >& signature )
{
    int local_size = 0;
    std::string local_error;
    try
    {
        local_size = KARMA::detail::checked_mpi_count( signature.size() );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< int > sizes;
    KARMA::detail::prepare_collectively( communicator, [&] { sizes.resize( static_cast< std::size_t >( ranks ) ); },
                                         "discretization signature count allocation failed" );
    mpi_check( MPI_Allgather( &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(discretization signature sizes)" );
    KARMA::detail::MpiLayout layout;
    try
    {
        layout = KARMA::detail::checked_mpi_displacements( sizes, 1 );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< std::uint64_t > gathered;
    KARMA::detail::prepare_collectively( communicator, [&] { gathered.resize( layout.total_count ); },
                                         "discretization signature gather allocation failed" );
    mpi_check( MPI_Allgatherv( signature.data(), local_size, MPI_UINT64_T, gathered.data(), sizes.data(),
                               layout.displacements.data(), MPI_UINT64_T, communicator ),
               "MPI_Allgatherv(discretization signatures)" );
    const auto own_begin = gathered.begin() + layout.displacements[static_cast< std::size_t >( rank )];
    for( int source = 0; source < ranks; ++source )
    {
        const auto begin = gathered.begin() + layout.displacements[static_cast< std::size_t >( source )];
        if( sizes[static_cast< std::size_t >( source )] != local_size ||
            !std::equal( begin, begin + sizes[static_cast< std::size_t >( source )], own_begin ) )
            KARMA_THROW( KARMA::InvalidArgument( "discretization signature differs across ranks" ) );
    }
}

std::map< Key, int > gather_key_owners( const std::map< Key, int >& local_records,
                                        MPI_Comm communicator, int ranks )
{
    constexpr std::size_t record_words = key_words + 1;
    if( ranks == 1 ) return local_records;

    int local_words = 0;
    int local_valid = 1;
    try
    {
        local_words = KARMA::detail::checked_mpi_word_count( local_records.size(), record_words );
    }
    catch( const std::logic_error& )
    {
        local_valid = 0;
    }
    if( !KARMA::detail::collective_validation_succeeded( communicator, local_valid != 0 ) )
        KARMA_THROW( KARMA::InvalidArgument( "MPI DoF record word count overflow" ) );

    std::vector< std::uint64_t > local;
    std::vector< int > counts;
    KARMA::detail::prepare_collectively( communicator, [&] {
        local.reserve( static_cast< std::size_t >( local_words ) );
        for( const auto& [key, owner] : local_records )
        {
            local.insert( local.end(), key.begin(), key.end() );
            local.push_back( static_cast< std::uint64_t >( owner ) );
        }
        counts.resize( static_cast< std::size_t >( ranks ) );
    }, "DoF record preparation failed" );

    mpi_check( MPI_Allgather( &local_words, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(DoF record word counts)" );
    KARMA::detail::MpiLayout layout;
    local_valid = 1;
    try
    {
        layout = KARMA::detail::checked_mpi_displacements( counts, record_words );
    }
    catch( const std::logic_error& )
    {
        local_valid = 0;
    }
    if( !KARMA::detail::collective_validation_succeeded( communicator, local_valid != 0 ) )
        KARMA_THROW( KARMA::InvalidArgument( "invalid MPI DoF record count or displacement" ) );
    std::vector< std::uint64_t > gathered;
    KARMA::detail::prepare_collectively( communicator, [&] { gathered.resize( layout.total_count ); },
                                         "DoF record gather allocation failed" );
    mpi_check( MPI_Allgatherv( local.data(), local_words, MPI_UINT64_T, gathered.data(), counts.data(),
                               layout.displacements.data(), MPI_UINT64_T, communicator ),
               "MPI_Allgatherv(DoF records)" );

    std::map< Key, int > global_records;
    std::map< Key, std::set< int > > reporters;
    std::string local_error;
    try
    {
        for( int source = 0; source < ranks; ++source )
            for( int source_offset = 0; source_offset < counts[static_cast< std::size_t >( source )];
                 source_offset += static_cast< int >( record_words ) )
            {
                const std::size_t offset = static_cast< std::size_t >(
                    layout.displacements[static_cast< std::size_t >( source )] + source_offset );
                Key key{};
                std::copy_n( gathered.begin() + static_cast< std::ptrdiff_t >( offset ), key_words, key.begin() );
                const auto owner_word = gathered[offset + key_words];
                if( owner_word >= static_cast< std::uint64_t >( ranks ) )
                {
                    local_error = "DoF record has an invalid owner";
                    break;
                }
                const int owner = static_cast< int >( owner_word );
                reporters[key].insert( source );
                const auto [found, inserted] = global_records.emplace( key, owner );
                if( !inserted && found->second != owner )
                {
                    local_error = "copies of a DoF key report conflicting owners";
                    break;
                }
            }
        if( local_error.empty() )
            for( const auto& [key, owner] : global_records )
                if( !reporters.at( key ).contains( owner ) )
                {
                    local_error = "DoF owner does not report the key";
                    break;
                }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown gathered DoF record validation failure"; }
    int rank = 0;
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(DoF records)" );
    collective_identity_check( communicator, rank, ranks, local_error );
    return global_records;
}
}  // namespace

namespace KARMA
{
namespace detail
{
int checked_mpi_count( std::size_t count )
{
    if( count > static_cast< std::size_t >( std::numeric_limits< int >::max() ) )
        KARMA_THROW( InvalidArgument( "MPI count overflow" ) );
    return static_cast< int >( count );
}

int checked_mpi_word_count( std::size_t count, std::size_t words_per_record )
{
    if( words_per_record != 0 &&
        count > static_cast< std::size_t >( std::numeric_limits< int >::max() ) / words_per_record )
        KARMA_THROW( InvalidArgument( "MPI word count overflow" ) );
    return checked_mpi_count( count * words_per_record );
}

MpiLayout checked_mpi_displacements( std::span< const int > counts, std::size_t words_per_record )
{
    MpiLayout layout{ std::vector< int >( counts.size() ), 0 };
    for( std::size_t source = 0; source < counts.size(); ++source )
    {
        const int count = counts[source];
        if( count < 0 || ( words_per_record != 0 && static_cast< std::size_t >( count ) % words_per_record != 0 ) )
            KARMA_THROW( InvalidArgument( "invalid MPI serialized record count" ) );
        layout.displacements[source] = checked_mpi_count( layout.total_count );
        if( static_cast< std::size_t >( count ) >
            static_cast< std::size_t >( std::numeric_limits< int >::max() ) - layout.total_count )
            KARMA_THROW( InvalidArgument( "MPI displacement overflow" ) );
        layout.total_count += static_cast< std::size_t >( count );
    }
    return layout;
}

PetscInt checked_petsc_int_size( std::size_t size )
{
    if( size > static_cast< std::size_t >( PETSC_MAX_INT ) )
        KARMA_THROW( InvalidArgument( "size exceeds PetscInt range" ) );
    return static_cast< PetscInt >( size );
}

bool collective_validation_succeeded( MPI_Comm communicator, bool local_valid )
{
    const int local = local_valid ? 1 : 0;
    int global      = 0;
    mpi_check( MPI_Allreduce( &local, &global, 1, MPI_INT, MPI_MIN, communicator ),
               "MPI_Allreduce(validation agreement)" );
    return global != 0;
}
}  // namespace detail

CanonicalSubentity canonicalize_cyclic_vertex_ids( std::span< const PetscInt > cyclic_vertex_ids )
{
    if( cyclic_vertex_ids.size() < 2 || cyclic_vertex_ids.size() > 4 )
        KARMA_THROW( InvalidArgument( "canonical orientation requires an edge, triangle, or quadrilateral" ) );
    std::vector< EntityIdentity > local;
    local.reserve( cyclic_vertex_ids.size() );
    for( const PetscInt id : cyclic_vertex_ids ) local.push_back( { static_cast< std::uint64_t >( id ), 0, 0 } );
    unsigned orientation = 0;
    const auto canonical = canonical_cycle( local, orientation );
    CanonicalSubentity result{ {}, orientation };
    result.vertex_ids.reserve( canonical.size() );
    for( const auto& identity : canonical ) result.vertex_ids.push_back( static_cast< PetscInt >( identity[0] ) );
    return result;
}

unsigned canonical_orientation_code( std::span< const PetscInt > cyclic_vertex_ids )
{
    return canonicalize_cyclic_vertex_ids( cyclic_vertex_ids ).orientation_code;
}

struct KARMADofMap::Impl
{
    const KARMAMesh* mesh = nullptr;
    std::map< moab::EntityHandle, std::vector< Key > > element_keys;
    std::map< moab::EntityHandle, std::uint64_t > element_ordinals;
    std::map< moab::EntityHandle, std::vector< std::size_t > > variable_offsets;
    std::map< moab::EntityHandle, std::vector< std::size_t > > orientation_offsets;
    std::map< moab::EntityHandle, std::vector< DofOrientation > > orientation_data;
    std::map< std::tuple< moab::EntityHandle, unsigned, unsigned >, Key > vertex_keys;
    std::map< Key, PetscInt > global_indices;
    std::vector< Key > owned_keys;
    std::map< PetscInt, std::set< PetscInt > > adjacency;
    PetscInt first_owned = 0;
    bool adjacency_initialized = false;
};

KARMADofMap::KARMADofMap() = default;

KARMADofMap::KARMADofMap( const KARMAMesh& mesh, const std::vector< VariableSpec >& variables,
                          const KARMABasisRegistry& registry )
    : _impl( std::make_unique< Impl >() ), _variables( variables )
{
    _impl->mesh           = &mesh;
    MPI_Comm communicator = mesh.parallel_comm()->comm();
    int ranks = 1, rank = 0;
    mpi_check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size" );
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank" );

    std::string local_error;
    std::vector< std::uint64_t > signature;
    try
    {
        if( variables.empty() ) KARMA_THROW( InvalidArgument( "system has no variables" ) );
        signature = variable_signature( variables );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    require_matching_signature( communicator, rank, ranks, signature );
    std::set< ReferenceCell > cells;
    local_error.clear();
    try
    {
        cells = gather_cell_types( mesh, communicator, rank, ranks );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    local_error.clear();
    try
    {
        signature = basis_signature( cells, variables, registry );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    require_matching_signature( communicator, rank, ranks, signature );
    if( ranks > 1 && !mesh.is_replicated() )
    {
        moab::Tag global_id = nullptr;
        const bool has_global_id = mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS;
        const bool all_have_global_id = detail::collective_validation_succeeded( communicator, has_global_id );
        if( all_have_global_id )
        {
            moab::Range visible = mesh.vertices();
            visible.merge( mesh.elements() );
            const moab::ErrorCode exchange_error = mesh.parallel_comm()->exchange_tags( global_id, visible );
            local_error = exchange_error == moab::MB_SUCCESS ? "" : "failed to exchange visible GLOBAL_ID values";
            collective_identity_check( communicator, rank, ranks, local_error );
        }
    }
    mesh.ensure_global_ids();

    const MeshIdentity identity = establish_identity( mesh, communicator, rank, ranks );
    for( const auto& [element, value] : identity.elements )
        _impl->element_ordinals.emplace( element, value.front()[0] );

    std::map< Key, int > local_records;
    local_error.clear();
    try
    {
    for( const auto element : mesh.elements() )
    {
        auto& keys                = _impl->element_keys[element];
        auto& offsets             = _impl->variable_offsets[element];
        auto& orientation_offsets = _impl->orientation_offsets[element];
        auto& orientation_data    = _impl->orientation_data[element];
        const ReferenceCell cell  = reference_cell( element_variant( mesh, element ) );
        for( unsigned variable = 0; variable < variables.size(); ++variable )
        {
            offsets.push_back( keys.size() );
            orientation_offsets.push_back( orientation_data.size() );
            const auto& specification = variables[variable];
            const KARMABasis basis =
                registry.create( { cell, specification.family, specification.order, specification.continuity } );
            const auto associations = basis.dof_associations();
            for( unsigned component = 0; component < specification.components; ++component )
                for( std::size_t ordinal = 0; ordinal < associations.size(); ++ordinal )
                {
                    const auto& association = associations[ordinal];
                    unsigned orientation    = 0;
                    std::vector< EntityIdentity > entities;
                    unsigned key_dimension = mesh.dimension();
                    moab::EntityType key_type = mesh.moab()->type_from_handle( element );
                    unsigned slot = static_cast< unsigned >( ordinal );
                    moab::EntityHandle owner_entity = element;
                    int owner = rank;
                    if( specification.continuity == BasisContinuity::cG )
                    {
                        entities      = subentity_identities( mesh, identity, element, association, orientation );
                        key_dimension = association.entity_dimension;
                        slot          = canonical_slot( basis, association, orientation );
                        if( association.entity_dimension == 0 )
                        {
                            const moab::EntityHandle* connectivity = nullptr;
                            int count = 0;
                            moab_check( mesh.moab()->get_connectivity( element, connectivity, count ),
                                        "get_connectivity(DoF owner)" );
                            owner_entity = connectivity[association.local_entity_index];
                            key_type = moab::MBVERTEX;
                        }
                        else if( association.entity_dimension != mesh.dimension() )
                        {
                            key_type = moab::CN::SubEntityType( mesh.moab()->type_from_handle( element ),
                                                                association.entity_dimension,
                                                                association.local_entity_index );
                            if( !mesh.is_replicated() )
                            {
                                moab::EntityHandle associated = 0;
                                moab_check( mesh.moab()->side_element( element, association.entity_dimension,
                                                                       association.local_entity_index, associated ),
                                            "side_element(DoF owner)" );
                                if( associated == 0 )
                                    KARMA_THROW( InvalidArgument(
                                        "partitioned cG layout requires explicit associated MOAB subentities" ) );
                                owner_entity = associated;
                            }
                        }
                    }
                    else
                        entities = identity.elements.at( element );
                    keys.push_back( make_key( variable, component, key_dimension, key_type, slot, entities ) );
                    if( specification.continuity == BasisContinuity::cG && association.entity_dimension == 0 &&
                        association.slot == 0 )
                        _impl->vertex_keys.emplace( std::tuple{ owner_entity, variable, component }, keys.back() );
                    if( !mesh.is_replicated() )
                    {
                        if( owner_entity != 0 )
                            moab_check( mesh.parallel_comm()->get_owner( owner_entity, owner ), "get_owner(DoF)" );
                    }
                    else if( ranks > 1 )
                        owner = static_cast< int >( fixed_hash( keys.back().words ) % static_cast< unsigned >( ranks ) );
                    const auto [record, inserted] = local_records.emplace( keys.back(), owner );
                    if( !inserted && record->second != owner )
                        KARMA_THROW( InvalidArgument( "local copies of a DoF key report conflicting owners" ) );
                }
            std::set< std::pair< unsigned, unsigned > > orientable_entities;
            for( const auto& transform : basis.orientation_transforms() )
                orientable_entities.emplace( transform.entity_dimension, transform.local_entity_index );
            for( const auto [dimension, entity] : orientable_entities )
            {
                const DofAssociation association{ dimension, entity, 0 };
                unsigned orientation = 0;
                const auto entities  = subentity_identities( mesh, identity, element, association, orientation );
                orientation_data.push_back( { association, orientation, {} } );
                auto& mapping = orientation_data.back().local_to_canonical_vertices;
                if( entities.size() == 2 )
                {
                    const auto p = local_to_canonical_vertices( static_cast< EdgeOrientation >( orientation ) );
                    mapping.assign( p.begin(), p.end() );
                }
                else if( entities.size() == 3 )
                {
                    const auto p = local_to_canonical_vertices( static_cast< TriangleOrientation >( orientation ) );
                    mapping.assign( p.begin(), p.end() );
                }
                else if( entities.size() == 4 )
                {
                    const auto p =
                        local_to_canonical_vertices( static_cast< QuadrilateralOrientation >( orientation ) );
                    mapping.assign( p.begin(), p.end() );
                }
            }
        }
        offsets.push_back( keys.size() );
        orientation_offsets.push_back( orientation_data.size() );
    }
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    const std::map< Key, int > global_records = gather_key_owners( local_records, communicator, ranks );
    local_error.clear();
    try
    {
        const PetscInt global_dof_count = detail::checked_petsc_int_size( global_records.size() );
        std::vector< std::pair< Key, int > > ordered( global_records.begin(), global_records.end() );
        std::stable_sort( ordered.begin(), ordered.end(), []( const auto& left, const auto& right ) {
            return left.second != right.second ? left.second < right.second : left.first < right.first;
        } );
        for( std::size_t i = 0; i < ordered.size(); ++i )
            _impl->global_indices.emplace( ordered[i].first, static_cast< PetscInt >( i ) );
        _n_dofs  = global_dof_count;
        _n_owned = static_cast< PetscInt >( std::count_if( ordered.begin(), ordered.end(), [rank]( const auto& record ) {
            return record.second == rank;
        } ) );
        _impl->first_owned =
            static_cast< PetscInt >( std::count_if( ordered.begin(), ordered.end(), [rank]( const auto& record ) {
                return record.second < rank;
            } ) );
        for( PetscInt index = _impl->first_owned; index < _impl->first_owned + _n_owned; ++index )
            _local_to_global.push_back( index );
        _impl->owned_keys.resize( static_cast< std::size_t >( _n_owned ) );
        for( const auto& [key, index] : _impl->global_indices )
            if( index >= _impl->first_owned && index < _impl->first_owned + _n_owned )
                _impl->owned_keys[static_cast< std::size_t >( index - _impl->first_owned )] = key;
        for( const auto& [key, owner] : local_records )
        {
            (void)owner;
            const PetscInt index = _impl->global_indices.at( key );
            if( index < _impl->first_owned || index >= _impl->first_owned + _n_owned ) _ghosts.push_back( index );
        }
        std::sort( _ghosts.begin(), _ghosts.end() );
        _local_to_global.insert( _local_to_global.end(), _ghosts.begin(), _ghosts.end() );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    struct MappingGuard
    {
        ISLocalToGlobalMapping value = nullptr;
        ~MappingGuard()
        {
            if( value ) ISLocalToGlobalMappingDestroy( &value );
        }
    } mapping;
    local_error.clear();
    try
    {
        petsc_check( ISLocalToGlobalMappingCreate( communicator, 1,
                                                   detail::checked_petsc_int_size( _local_to_global.size() ),
                                                   _local_to_global.data(), PETSC_COPY_VALUES, &mapping.value ),
                     "ISLocalToGlobalMappingCreate" );
        if( !mesh.is_replicated() )
            _assembly_elements.assign( mesh.owned_elements().begin(), mesh.owned_elements().end() );
        else
            for( const auto element : mesh.elements() )
            {
                const auto& element_identity = identity.elements.at( element ).front();
                const int owner = static_cast< int >( fixed_hash( element_identity ) % static_cast< unsigned >( ranks ) );
                if( owner == rank ) _assembly_elements.push_back( element );
            }
        for( const auto& [element, keys] : _impl->element_keys )
        {
            (void)element;
            for( const auto& row : keys )
                for( const auto& column : keys )
                    _impl->adjacency[_impl->global_indices.at( row )].insert( _impl->global_indices.at( column ) );
        }
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    add_element_adjacency( 0, 0 );
    _mapping = std::exchange( mapping.value, nullptr );
}

KARMADofMap::~KARMADofMap()
{
    if( _mapping ) ISLocalToGlobalMappingDestroy( &_mapping );
}
KARMADofMap::KARMADofMap( KARMADofMap&& other ) noexcept : _impl( std::move( other._impl ) ),
                                                           _variables( std::move( other._variables ) ),
                                                           _n_dofs( other._n_dofs ),
                                                           _n_owned( other._n_owned ),
                                                           _local_to_global( std::move( other._local_to_global ) ),
                                                           _ghosts( std::move( other._ghosts ) ),
                                                           _assembly_elements( std::move( other._assembly_elements ) ),
                                                           _diagonal_nnz( std::move( other._diagonal_nnz ) ),
                                                           _off_diagonal_nnz( std::move( other._off_diagonal_nnz ) ),
                                                           _mapping( std::exchange( other._mapping, nullptr ) )
{
}
KARMADofMap& KARMADofMap::operator=( KARMADofMap&& other ) noexcept
{
    if( this != &other )
    {
        if( _mapping ) ISLocalToGlobalMappingDestroy( &_mapping );
        _impl              = std::move( other._impl );
        _variables         = std::move( other._variables );
        _n_dofs            = other._n_dofs;
        _n_owned           = other._n_owned;
        _local_to_global   = std::move( other._local_to_global );
        _ghosts            = std::move( other._ghosts );
        _assembly_elements = std::move( other._assembly_elements );
        _diagonal_nnz      = std::move( other._diagonal_nnz );
        _off_diagonal_nnz  = std::move( other._off_diagonal_nnz );
        _mapping           = std::exchange( other._mapping, nullptr );
    }
    return *this;
}

void KARMADofMap::dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices ) const
{
    const auto found = _impl->element_keys.find( element );
    if( found == _impl->element_keys.end() ) KARMA_THROW( InvalidArgument( "unknown element in dof_indices" ) );
    indices.clear();
    for( const auto& key : found->second )
        indices.push_back( _impl->global_indices.at( key ) );
}
const std::vector< DofKey >& KARMADofMap::dof_keys( moab::EntityHandle element ) const
{
    const auto found = _impl->element_keys.find( element );
    if( found == _impl->element_keys.end() ) KARMA_THROW( InvalidArgument( "unknown element in dof_keys" ) );
    return found->second;
}
PetscInt KARMADofMap::global_index( const DofKey& key ) const
{
    if( !_impl ) KARMA_THROW( InvalidArgument( "DoF map is not initialized" ) );
    const auto found = _impl->global_indices.find( key );
    if( found == _impl->global_indices.end() ) KARMA_THROW( InvalidArgument( "unknown DoF key" ) );
    return found->second;
}
std::pair< DofKey, PetscInt > KARMADofMap::owned_key_index( std::size_t ordinal ) const
{
    if( !_impl ) KARMA_THROW( InvalidArgument( "DoF map is not initialized" ) );
    if( ordinal >= static_cast< std::size_t >( _n_owned ) )
        KARMA_THROW( InvalidArgument( "owned DoF ordinal is out of range" ) );
    const PetscInt index = _impl->first_owned + static_cast< PetscInt >( ordinal );
    return { _impl->owned_keys[ordinal], index };
}
std::vector< std::pair< DofKey, PetscInt > > KARMADofMap::owned_key_indices() const
{
    std::vector< std::pair< DofKey, PetscInt > > result;
    result.reserve( static_cast< std::size_t >( _n_owned ) );
    for( std::size_t ordinal = 0; ordinal < static_cast< std::size_t >( _n_owned ); ++ordinal )
        result.push_back( owned_key_index( ordinal ) );
    return result;
}
void KARMADofMap::dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices, unsigned variable ) const
{
    if( variable >= _variables.size() ) KARMA_THROW( InvalidArgument( "variable index is out of range" ) );
    std::vector< PetscInt > all;
    dof_indices( element, all );
    const auto& offsets = _impl->variable_offsets.at( element );
    indices.assign( all.begin() + static_cast< std::ptrdiff_t >( offsets[variable] ),
                    all.begin() + static_cast< std::ptrdiff_t >( offsets[variable + 1] ) );
}
void KARMADofMap::local_dof_indices( moab::EntityHandle element, std::vector< PetscInt >& indices ) const
{
    dof_indices( element, indices );
    std::map< PetscInt, PetscInt > inverse;
    for( std::size_t i = 0; i < _local_to_global.size(); ++i )
        inverse.emplace( _local_to_global[i], static_cast< PetscInt >( i ) );
    for( auto& index : indices )
        index = inverse.at( index );
}
PetscInt KARMADofMap::vertex_dof( moab::EntityHandle vertex, unsigned variable, unsigned component ) const
{
    if( variable >= _variables.size() || component >= _variables[variable].components )
        KARMA_THROW( InvalidArgument( "vertex_dof variable or component is out of range" ) );
    const auto vertex_key = _impl->vertex_keys.find( std::tuple{ vertex, variable, component } );
    if( vertex_key == _impl->vertex_keys.end() ) KARMA_THROW( InvalidArgument( "variable has no shared vertex DoF" ) );
    const auto found = _impl->global_indices.find( vertex_key->second );
    if( found == _impl->global_indices.end() ) KARMA_THROW( InvalidArgument( "variable has no shared vertex DoF" ) );
    return found->second;
}
std::vector< DofStorage > KARMADofMap::storage( moab::EntityHandle element, unsigned variable,
                                                  const KARMABasis& basis ) const
{
    if( variable >= _variables.size() ) KARMA_THROW( InvalidArgument( "variable index is out of range" ) );
    const auto& specification = _variables[variable];
    if( basis.key().cell != reference_cell( element_variant( *_impl->mesh, element ) ) ||
        basis.key().family != specification.family || basis.key().order != specification.order ||
        basis.key().continuity != specification.continuity )
        KARMA_THROW( InvalidArgument( "storage basis does not match the element and variable" ) );
    const auto key_found = _impl->element_keys.find( element );
    if( key_found == _impl->element_keys.end() ) KARMA_THROW( InvalidArgument( "unknown element in storage" ) );
    const auto& offsets = _impl->variable_offsets.at( element );
    const auto associations = basis.dof_associations();
    if( offsets[variable + 1] - offsets[variable] != associations.size() * specification.components )
        KARMA_THROW( InvalidArgument( "storage basis size does not match the element DoF layout" ) );
    std::vector< DofStorage > result;
    result.reserve( associations.size() * specification.components );
    const std::size_t begin = offsets[variable];
    for( unsigned component = 0; component < specification.components; ++component )
        for( std::size_t ordinal = 0; ordinal < associations.size(); ++ordinal )
        {
            const std::size_t offset = begin + component * associations.size() + ordinal;
            const auto& key = key_found->second[offset];
            result.push_back( { _impl->global_indices.at( key ), component, associations[ordinal],
                                static_cast< unsigned >( key[4] ), specification.continuity != BasisContinuity::cG } );
        }
    return result;
}
std::vector< DofOrientation > KARMADofMap::orientations( moab::EntityHandle element, unsigned variable ) const
{
    if( variable >= _variables.size() ) KARMA_THROW( InvalidArgument( "variable index is out of range" ) );
    const auto& all     = _impl->orientation_data.at( element );
    const auto& offsets = _impl->orientation_offsets.at( element );
    return { all.begin() + static_cast< std::ptrdiff_t >( offsets[variable] ),
             all.begin() + static_cast< std::ptrdiff_t >( offsets[variable + 1] ) };
}
PetscInt KARMADofMap::first_owned_dof() const noexcept
{ return _impl ? _impl->first_owned : 0; }

bool KARMADofMap::owns_element_for_assembly( moab::EntityHandle element ) const noexcept
{ return std::find( _assembly_elements.begin(), _assembly_elements.end(), element ) != _assembly_elements.end(); }
std::uint64_t KARMADofMap::canonical_element_ordinal( moab::EntityHandle element ) const
{
    const auto found = _impl->element_ordinals.find( element );
    if( found == _impl->element_ordinals.end() ) KARMA_THROW( InvalidArgument( "unknown element identity" ) );
    return found->second;
}
void KARMADofMap::add_element_adjacency( moab::EntityHandle first, moab::EntityHandle second )
{
    MPI_Comm communicator = _impl->mesh->parallel_comm()->comm();
    int rank = 0, ranks = 1;
    mpi_check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(adjacency)" );
    mpi_check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(adjacency)" );

    std::string local_error;
    std::set< std::pair< PetscInt, PetscInt > > local_pairs;
    try
    {
        if( ( first == 0 ) != ( second == 0 ) )
            KARMA_THROW( InvalidArgument( "adjacency requires two elements or the (0,0) no-op" ) );
        if( !_impl->adjacency_initialized )
            for( const auto& [row, columns] : _impl->adjacency )
                for( const PetscInt column : columns ) local_pairs.emplace( row, column );
        if( first != 0 )
        {
            const auto& first_keys  = _impl->element_keys.at( first );
            const auto& second_keys = _impl->element_keys.at( second );
            for( const auto& row : first_keys )
                for( const auto& column : second_keys )
                {
                    const PetscInt row_index    = _impl->global_indices.at( row );
                    const PetscInt column_index = _impl->global_indices.at( column );
                    local_pairs.emplace( row_index, column_index );
                    local_pairs.emplace( column_index, row_index );
                }
        }
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    int local_words = 0;
    try
    {
        local_words = detail::checked_mpi_word_count( local_pairs.size(), 2 );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< PetscInt > local;
    local_error.clear();
    try
    {
        local.reserve( static_cast< std::size_t >( local_words ) );
        for( const auto [row, column] : local_pairs )
        {
            local.push_back( row );
            local.push_back( column );
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< int > sizes;
    KARMA::detail::prepare_collectively( communicator, [&] { sizes.resize( static_cast< std::size_t >( ranks ) ); },
                                         "adjacency count allocation failed" );
    mpi_check( MPI_Allgather( &local_words, 1, MPI_INT, sizes.data(), 1, MPI_INT, communicator ),
               "MPI_Allgather(adjacency pair counts)" );
    detail::MpiLayout layout;
    local_error.clear();
    try
    {
        layout = detail::checked_mpi_displacements( sizes, 2 );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );
    std::vector< PetscInt > gathered;
    local_error.clear();
    try
    {
        gathered.resize( layout.total_count );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_identity_check( communicator, rank, ranks, local_error );
    mpi_check( MPI_Allgatherv( local.data(), local_words, MPIU_INT, gathered.data(), sizes.data(),
                                layout.displacements.data(), MPIU_INT, communicator ),
               "MPI_Allgatherv(adjacency pairs)" );

    std::set< std::pair< PetscInt, PetscInt > > global_pairs;
    local_error.clear();
    try
    {
        for( std::size_t offset = 0; offset < gathered.size(); offset += 2 )
        {
            const PetscInt row    = gathered[offset];
            const PetscInt column = gathered[offset + 1];
            if( row < 0 || row >= _n_dofs || column < 0 || column >= _n_dofs )
                KARMA_THROW( InvalidArgument( "adjacency pair has an invalid global DoF index" ) );
            global_pairs.emplace( row, column );
        }
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    collective_identity_check( communicator, rank, ranks, local_error );

    local_error.clear();
    try
    {
        if( !_impl->adjacency_initialized ) _impl->adjacency.clear();
        for( const auto [row, column] : global_pairs )
            if( row >= _impl->first_owned && row < _impl->first_owned + _n_owned )
                _impl->adjacency[row].insert( column );
        _impl->adjacency_initialized = true;
        _diagonal_nnz.assign( static_cast< std::size_t >( _n_owned ), 0 );
        _off_diagonal_nnz.assign( static_cast< std::size_t >( _n_owned ), 0 );
        for( PetscInt row = _impl->first_owned; row < _impl->first_owned + _n_owned; ++row )
            for( const PetscInt column : _impl->adjacency[row] )
                ++( column >= _impl->first_owned && column < _impl->first_owned + _n_owned
                        ? _diagonal_nnz[static_cast< std::size_t >( row - _impl->first_owned )]
                        : _off_diagonal_nnz[static_cast< std::size_t >( row - _impl->first_owned )] );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_identity_check( communicator, rank, ranks, local_error );
}
}  // namespace KARMA
