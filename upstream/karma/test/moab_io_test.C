#include "karma_moab_io.h"

#include <petsc.h>

#include "moab/CN.hpp"
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace
{
int failures = 0;

void check( bool condition, const std::string& message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error != PETSC_SUCCESS ) throw std::runtime_error( std::string( operation ) + " failed" );
}

void moab_check( moab::ErrorCode error, const char* operation )
{
    if( error != moab::MB_SUCCESS ) throw std::runtime_error( std::string( operation ) + " failed" );
}

void fill_solution( Vec solution )
{
    PetscInt first = 0;
    PetscInt last  = 0;
    petsc_check( VecGetOwnershipRange( solution, &first, &last ), "VecGetOwnershipRange" );
    for( PetscInt index = first; index < last; ++index )
    {
        const PetscScalar value = static_cast< PetscScalar >( 0.25 + 1.5 * static_cast< double >( index ) );
        petsc_check( VecSetValue( solution, index, value, INSERT_VALUES ), "VecSetValue" );
    }
    petsc_check( VecAssemblyBegin( solution ), "VecAssemblyBegin" );
    petsc_check( VecAssemblyEnd( solution ), "VecAssemblyEnd" );
}

void compare_vectors( Vec actual, Vec expected, const std::string& message )
{
    PetscBool equal = PETSC_FALSE;
    petsc_check( VecEqual( actual, expected, &equal ), "VecEqual" );
    check( equal == PETSC_TRUE, message );
}

std::vector< unsigned char > vector_bytes( Vec vector )
{
    PetscInt local_size = 0;
    petsc_check( VecGetLocalSize( vector, &local_size ), "VecGetLocalSize(vector snapshot)" );
    const PetscScalar* values = nullptr;
    petsc_check( VecGetArrayRead( vector, &values ), "VecGetArrayRead(vector snapshot)" );
    std::vector< unsigned char > result( static_cast< std::size_t >( local_size ) * sizeof( PetscScalar ) );
    std::memcpy( result.data(), values, result.size() );
    petsc_check( VecRestoreArrayRead( vector, &values ), "VecRestoreArrayRead(vector snapshot)" );
    return result;
}

template < class Function > bool throws_invalid( Function&& function )
{
    try
    {
        function();
    }
    catch( const KARMA::InvalidArgument& )
    {
        return true;
    }
    return false;
}

template < class Function > bool throws_invalid_containing( Function&& function, std::string_view text )
{
    try
    {
        function();
    }
    catch( const KARMA::InvalidArgument& error )
    {
        return std::string_view( error.what() ).find( text ) != std::string_view::npos;
    }
    return false;
}

struct OrientedEdgeBasis
{
    [[nodiscard]] std::string name() const { return "checkpoint-oriented-edge"; }
    [[nodiscard]] std::string compatibility_id() const { return "test.oriented-edge.v1"; }
    [[nodiscard]] KARMA::CoefficientRepresentation coefficient_representation() const noexcept
    { return KARMA::CoefficientRepresentation::Modal; }
    [[nodiscard]] KARMA::BasisKey key() const noexcept
    {
        return { KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::UserDefined, 3,
                 KARMA::BasisContinuity::cG };
    }
    [[nodiscard]] std::size_t size() const noexcept { return 8; }
    void evaluate_values( const KARMA::Point&, std::span< Real > values ) const
    { std::fill( values.begin(), values.end(), 0.0 ); }
    void evaluate( const KARMA::Point&, std::span< Real > values, std::span< KARMA::Gradient > gradients ) const
    {
        std::fill( values.begin(), values.end(), 0.0 );
        std::fill( gradients.begin(), gradients.end(), KARMA::Gradient::Zero() );
    }
    [[nodiscard]] std::vector< KARMA::Point > interpolation_points() const
    { return std::vector< KARMA::Point >( size(), KARMA::Point::Zero() ); }
    [[nodiscard]] std::vector< KARMA::DofAssociation > dof_associations() const
    {
        std::vector< KARMA::DofAssociation > result;
        for( unsigned edge = 0; edge < 4; ++edge )
            for( unsigned slot = 0; slot < 2; ++slot ) result.push_back( { 1, edge, slot } );
        return result;
    }
    [[nodiscard]] std::vector< KARMA::OrientationTransform > orientation_transforms() const
    {
        std::vector< KARMA::OrientationTransform > result;
        for( unsigned edge = 0; edge < 4; ++edge )
        {
            result.push_back( { 1, edge, 0, { 0, 1 }, { 1, 1 } } );
            result.push_back( { 1, edge, 1, { 1, 0 }, { 1, 1 } } );
        }
        return result;
    }
};

struct DifferentEdgeBasis : OrientedEdgeBasis
{
    [[nodiscard]] std::string compatibility_id() const { return "test.oriented-edge.v2"; }
};

void register_oriented_basis( KARMA::KARMABasisRegistry& registry )
{
    const KARMA::BasisKey key{ KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::UserDefined, 3,
                               KARMA::BasisContinuity::cG };
    registry.register_factory( key, [] { return KARMA::KARMABasis( OrientedEdgeBasis{} ); } );
}

void register_different_basis( KARMA::KARMABasisRegistry& registry )
{
    const KARMA::BasisKey key{ KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::UserDefined, 3,
                               KARMA::BasisContinuity::cG };
    registry.register_factory( key, [] { return KARMA::KARMABasis( DifferentEdgeBasis{} ); } );
}

moab::EntityHandle associated_entity( KARMA::KARMAMesh& mesh, moab::EntityHandle element,
                                      const KARMA::DofStorage& storage )
{
    if( storage.element_local || storage.association.entity_dimension == mesh.dimension() ) return element;
    if( storage.association.entity_dimension == 0 )
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get semantic connectivity" );
        return connectivity[storage.association.local_entity_index];
    }
    moab::EntityHandle entity = 0;
    moab_check( mesh.moab()->side_element( element, static_cast< int >( storage.association.entity_dimension ),
                                           static_cast< int >( storage.association.local_entity_index ), entity ),
                "get semantic subentity" );
    if( entity == 0 )
    {
        moab::Range created;
        moab_check( mesh.moab()->get_adjacencies( &element, 1,
                                                  static_cast< int >( storage.association.entity_dimension ),
                                                  true, created, moab::Interface::UNION ),
                    "create semantic subentities" );
        moab_check( mesh.moab()->side_element( element, static_cast< int >( storage.association.entity_dimension ),
                                               static_cast< int >( storage.association.local_entity_index ), entity ),
                    "get created semantic subentity" );
    }
    check( entity != 0, "semantic associated entity exists" );
    return entity;
}

void fill_semantic_solution( KARMA::KARMASystem& system, Vec solution )
{
    auto& mesh = system.mesh();
    std::map< PetscInt, PetscScalar > values;
    for( unsigned variable = 0; variable < system.n_variables(); ++variable )
    {
        const auto& specification = system.variable( variable );
        for( const auto element : mesh.elements() )
        {
            const moab::EntityHandle* connectivity = nullptr;
            int count = 0;
            moab_check( mesh.moab()->get_connectivity( element, connectivity, count ), "get semantic topology" );
            const auto cell = KARMA::reference_cell( KARMA::KARMAReferenceTopology::from_moab(
                mesh.moab()->type_from_handle( element ), static_cast< std::size_t >( count ) ) );
            const auto basis = system.basis_registry().create(
                { cell, specification.family, specification.order, specification.continuity } );
            for( const auto& storage : system.dof_map().storage( element, variable, basis ) )
            {
                const auto entity = associated_entity( mesh, element, storage );
                KARMA::Point center = KARMA::Point::Zero();
                if( storage.association.entity_dimension == 0 && !storage.element_local )
                {
                    double coordinates[3]{};
                    moab_check( mesh.moab()->get_coords( &entity, 1, coordinates ), "get semantic vertex coordinates" );
                    center = KARMA::Point( coordinates[0], coordinates[1], coordinates[2] );
                }
                else
                    center = mesh.centroid( entity );
                std::uint64_t coordinate_hash = 1469598103934665603ULL;
                for( unsigned dimension = 0; dimension < 3; ++dimension )
                {
                    const auto bits = std::bit_cast< std::uint64_t >( static_cast< double >( center( dimension ) ) );
                    coordinate_hash ^= bits;
                    coordinate_hash *= 1099511628211ULL;
                }
                const std::uint64_t semantic =
                    ( static_cast< std::uint64_t >( variable ) << 44 ) |
                    ( static_cast< std::uint64_t >( storage.component ) << 36 ) |
                    ( static_cast< std::uint64_t >( storage.association.entity_dimension ) << 32 ) |
                    ( static_cast< std::uint64_t >( storage.canonical_slot ) << 24 ) |
                    ( coordinate_hash & 0xFFFFFFULL );
                const PetscScalar value = static_cast< PetscScalar >( semantic );
                const auto [found, inserted] = values.emplace( storage.index, value );
                check( inserted || found->second == value, "semantic values agree for shared cG DoFs" );
            }
        }
    }
    PetscInt first = 0, last = 0;
    petsc_check( VecGetOwnershipRange( solution, &first, &last ), "VecGetOwnershipRange(semantic solution)" );
    petsc_check( VecSet( solution, 0.0 ), "VecSet(semantic solution)" );
    for( auto iterator = values.lower_bound( first ); iterator != values.end() && iterator->first < last; ++iterator )
        petsc_check( VecSetValue( solution, iterator->first, iterator->second, INSERT_VALUES ),
                     "VecSetValue(semantic solution)" );
    petsc_check( VecAssemblyBegin( solution ), "VecAssemblyBegin(semantic solution)" );
    petsc_check( VecAssemblyEnd( solution ), "VecAssemblyEnd(semantic solution)" );
}

void add_reference_element( KARMA::KARMAMesh& mesh, KARMA::GeometryVariant variant )
{
    const KARMA::KARMAReferenceTopology topology( variant );
    std::vector< double > coordinates;
    for( const auto& point : topology.reference_nodes() )
        for( unsigned dimension = 0; dimension < 3; ++dimension ) coordinates.push_back( point( dimension ) );
    moab::Range vertices;
    moab_check( mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( topology.node_count() ), vertices ),
                "create reference vertices" );
    std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
    moab::EntityHandle element = 0;
    moab_check( mesh.moab()->create_element( topology.topology(), connectivity.data(),
                                             static_cast< int >( connectivity.size() ), element ),
                "create reference element" );
    mesh.assign_ids();
    mesh.update();
}

struct DatabaseInventory
{
    std::vector< unsigned char > digest;

    bool operator==( const DatabaseInventory& ) const = default;
};

void append_inventory_word( std::vector< unsigned char >& digest, std::uint64_t value )
{
    for( unsigned shift = 0; shift < 64; shift += 8 )
        digest.push_back( static_cast< unsigned char >( value >> shift ) );
}

void append_inventory_bytes( std::vector< unsigned char >& digest, const void* data, std::size_t size )
{
    append_inventory_word( digest, size );
    if( size != 0 )
    {
        const auto* bytes = static_cast< const unsigned char* >( data );
        digest.insert( digest.end(), bytes, bytes + size );
    }
}

void append_inventory_string( std::vector< unsigned char >& digest, std::string_view value )
{
    append_inventory_bytes( digest, value.data(), value.size() );
}

DatabaseInventory database_inventory( moab::Interface& database )
{
    DatabaseInventory result;
    std::map< moab::EntityHandle, std::uint64_t > entity_order;
    std::vector< moab::Range > entities( static_cast< std::size_t >( moab::MBMAXTYPE ) );
    for( unsigned type = moab::MBVERTEX; type < moab::MBMAXTYPE; ++type )
    {
        auto& range = entities[type];
        moab_check( database.get_entities_by_type( 0, static_cast< moab::EntityType >( type ), range ),
                    "enumerate inventory entities" );
        std::uint64_t ordinal = 0;
        for( const auto entity : range ) entity_order.emplace( entity, ordinal++ );
    }
    const auto append_entity = [&]( moab::EntityHandle entity ) {
        const auto found = entity_order.find( entity );
        if( entity == 0 || found != entity_order.end() )
        {
            append_inventory_word( result.digest, 0 );
            append_inventory_word( result.digest, entity == 0 ? moab::MBMAXTYPE : database.type_from_handle( entity ) );
            append_inventory_word( result.digest, entity == 0 ? 0 : found->second );
            return;
        }
        append_inventory_word( result.digest, 1 );
        append_inventory_bytes( result.digest, &entity, sizeof( entity ) );
    };
    const auto append_payload = [&]( moab::DataType data_type, const void* data, std::size_t size ) {
        append_inventory_word( result.digest, size );
        if( data_type != moab::MB_TYPE_HANDLE || size % sizeof( moab::EntityHandle ) != 0 )
        {
            if( size != 0 )
            {
                const auto* bytes = static_cast< const unsigned char* >( data );
                result.digest.insert( result.digest.end(), bytes, bytes + size );
            }
            return;
        }
        for( std::size_t offset = 0; offset < size; offset += sizeof( moab::EntityHandle ) )
        {
            moab::EntityHandle handle = 0;
            std::memcpy( &handle, static_cast< const unsigned char* >( data ) + offset, sizeof( handle ) );
            append_entity( handle );
        }
    };
    const auto payload_bytes = []( moab::DataType data_type, int count ) {
        if( count < 0 ) throw std::runtime_error( "negative variable inventory tag length" );
        std::size_t width = 0;
        switch( data_type )
        {
            case moab::MB_TYPE_OPAQUE: width = 1; break;
            case moab::MB_TYPE_INTEGER: width = sizeof( int ); break;
            case moab::MB_TYPE_DOUBLE: width = sizeof( double ); break;
            case moab::MB_TYPE_HANDLE: width = sizeof( moab::EntityHandle ); break;
            default: throw std::runtime_error( "unsupported variable inventory tag data type" );
        }
        const auto typed_count = static_cast< std::size_t >( count );
        if( typed_count > std::numeric_limits< std::size_t >::max() / width )
            throw std::runtime_error( "variable inventory tag byte size overflow" );
        return typed_count * width;
    };
    for( unsigned type = moab::MBVERTEX; type < moab::MBMAXTYPE; ++type )
        for( const auto entity : entities[type] )
        {
            append_entity( entity );
            if( type == moab::MBVERTEX )
            {
                double coordinates[3]{};
                moab_check( database.get_coords( &entity, 1, coordinates ), "read inventory coordinates" );
                append_inventory_bytes( result.digest, coordinates, sizeof( coordinates ) );
            }
            else if( type != moab::MBENTITYSET )
            {
                const moab::EntityHandle* connectivity = nullptr;
                int count = 0;
                moab_check( database.get_connectivity( entity, connectivity, count ), "read inventory connectivity" );
                append_inventory_word( result.digest, static_cast< std::uint64_t >( count ) );
                for( int i = 0; i < count; ++i ) append_entity( connectivity[i] );
            }
        }
    std::vector< moab::Tag > tags;
    moab_check( database.tag_get_tags( tags ), "enumerate inventory tags" );
    std::sort( tags.begin(), tags.end(), [&]( const moab::Tag first, const moab::Tag second ) {
        std::string first_name;
        std::string second_name;
        moab_check( database.tag_get_name( first, first_name ), "read inventory tag name" );
        moab_check( database.tag_get_name( second, second_name ), "read inventory tag name" );
        return first_name < second_name;
    } );
    for( const auto tag : tags )
    {
        std::string name;
        moab::DataType data_type{};
        moab::TagType storage{};
        int bytes = 0;
        int length = 0;
        moab_check( database.tag_get_name( tag, name ), "read inventory tag name" );
        moab_check( database.tag_get_data_type( tag, data_type ), "read inventory tag data type" );
        moab_check( database.tag_get_type( tag, storage ), "read inventory tag storage" );
        const moab::ErrorCode width_error = database.tag_get_bytes( tag, bytes );
        const moab::ErrorCode length_error = database.tag_get_length( tag, length );
        append_inventory_string( result.digest, name );
        append_inventory_word( result.digest, data_type );
        append_inventory_word( result.digest, storage );
        append_inventory_word( result.digest, width_error );
        append_inventory_word( result.digest, bytes );
        append_inventory_word( result.digest, length_error );
        append_inventory_word( result.digest, length );
        const void* default_value = nullptr;
        int default_size = 0;
        const moab::ErrorCode default_error = database.tag_get_default_value( tag, default_value, default_size );
        append_inventory_word( result.digest, default_error );
        if( default_error == moab::MB_SUCCESS )
            append_payload( data_type, default_value, payload_bytes( data_type, default_size ) );
        for( unsigned type = moab::MBVERTEX; type < moab::MBMAXTYPE; ++type )
        {
            moab::Range tagged;
            moab_check( database.get_entities_by_type_and_tag( 0, static_cast< moab::EntityType >( type ), &tag,
                                                                nullptr, 1, tagged ),
                        "enumerate inventory tag values" );
            append_inventory_word( result.digest, tagged.size() );
            for( const auto entity : tagged )
            {
                append_entity( entity );
                if( width_error == moab::MB_SUCCESS )
                {
                    std::vector< unsigned char > value( static_cast< std::size_t >( bytes ) );
                    moab_check( database.tag_get_data( tag, &entity, 1, value.data() ), "read inventory tag value" );
                    append_payload( data_type, value.data(), value.size() );
                }
                else
                {
                    const void* value = nullptr;
                    int size = 0;
                    moab_check( database.tag_get_by_ptr( tag, &entity, 1, &value, &size ),
                                "read variable inventory tag value" );
                    append_payload( data_type, value, payload_bytes( data_type, size ) );
                }
            }
        }
        const moab::EntityHandle root = 0;
        if( width_error == moab::MB_SUCCESS )
        {
            std::vector< unsigned char > value( static_cast< std::size_t >( bytes ) );
            const moab::ErrorCode root_error = database.tag_get_data( tag, &root, 1, value.data() );
            append_inventory_word( result.digest, root_error );
            if( root_error == moab::MB_SUCCESS ) append_payload( data_type, value.data(), value.size() );
        }
        else
        {
            const void* value = nullptr;
            int size = 0;
            const moab::ErrorCode root_error = database.tag_get_by_ptr( tag, &root, 1, &value, &size );
            append_inventory_word( result.digest, root_error );
            if( root_error == moab::MB_SUCCESS )
                append_payload( data_type, value, payload_bytes( data_type, size ) );
        }
    }
    for( const auto meshset : entities[moab::MBENTITYSET] )
    {
        append_entity( meshset );
        unsigned options = 0;
        moab_check( database.get_meshset_options( meshset, options ), "read inventory meshset options" );
        append_inventory_word( result.digest, options );
        if( options & moab::MESHSET_ORDERED )
        {
            std::vector< moab::EntityHandle > contents;
            moab_check( database.get_entities_by_handle( meshset, contents ), "read ordered inventory meshset" );
            append_inventory_word( result.digest, contents.size() );
            for( const auto entity : contents ) append_entity( entity );
        }
        else
        {
            moab::Range contents;
            moab_check( database.get_entities_by_handle( meshset, contents ), "read unordered inventory meshset" );
            append_inventory_word( result.digest, contents.size() );
            for( const auto entity : contents ) append_entity( entity );
        }
        moab::Range parents;
        moab::Range children;
        moab_check( database.get_parent_meshsets( meshset, parents ), "read inventory meshset parents" );
        moab_check( database.get_child_meshsets( meshset, children ), "read inventory meshset children" );
        append_inventory_word( result.digest, parents.size() );
        for( const auto parent : parents ) append_entity( parent );
        append_inventory_word( result.digest, children.size() );
        for( const auto child : children ) append_entity( child );
    }
    return result;
}

DatabaseInventory collective_database_inventory( moab::Interface& database )
{
    DatabaseInventory result;
    std::string local_error;
    try
    {
        result = database_inventory( database );
    }
    catch( const std::exception& exception )
    {
        local_error = exception.what();
    }
    catch( ... )
    {
        local_error = "unknown exception";
    }

    const int local_failed = local_error.empty() ? 0 : 1;
    int any_failed = 0;
    MPI_Allreduce( &local_failed, &any_failed, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD );
    if( any_failed )
        throw std::runtime_error( local_failed ? "database inventory failed: " + local_error
                                               : "database inventory failed on another rank" );
    return result;
}

template < class Value >
void set_variable_tag( moab::Interface& database, moab::Tag tag, moab::EntityHandle entity,
                       const std::vector< Value >& values )
{
    const void* data = values.data();
    const int count = static_cast< int >( values.size() );
    moab_check( database.tag_set_by_ptr( tag, &entity, 1, &data, &count ), "set variable inventory tag" );
}

void variable_typed_tag_inventory_tracks_trailing_values()
{
    moab::Core database;
    const double coordinates[] = { 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 0.0 };
    moab::Range vertices;
    moab_check( database.create_vertices( coordinates, 4, vertices ), "create variable tag inventory vertices" );
    const moab::EntityHandle entity = vertices.front();
    const moab::EntityHandle root = 0;

    std::array< int, 3 > integer_default{ 11, 22, 33 };
    moab::Tag integer_tag = nullptr;
    moab_check( database.tag_get_handle( "VARIABLE_INTEGER", integer_default.size(), moab::MB_TYPE_INTEGER,
                                         integer_tag, moab::MB_TAG_SPARSE | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT,
                                         integer_default.data() ),
                "create variable integer tag" );
    std::vector< int > integers{ 1, 2, 3 };
    set_variable_tag( database, integer_tag, entity, integers );

    moab::Tag double_tag = nullptr;
    moab_check( database.tag_get_handle( "VARIABLE_DOUBLE", moab::MB_VARIABLE_LENGTH, moab::MB_TYPE_DOUBLE,
                                         double_tag, moab::MB_TAG_MESH | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT ),
                "create variable double tag" );
    std::vector< double > doubles{ 1.25, 2.5, 3.75 };
    set_variable_tag( database, double_tag, root, doubles );

    moab::Tag handle_tag = nullptr;
    moab_check( database.tag_get_handle( "VARIABLE_HANDLE", moab::MB_VARIABLE_LENGTH, moab::MB_TYPE_HANDLE,
                                         handle_tag, moab::MB_TAG_SPARSE | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT ),
                "create variable handle tag" );
    std::vector< moab::EntityHandle > handles{ vertices[1], vertices[2] };
    set_variable_tag( database, handle_tag, entity, handles );

    const auto baseline = database_inventory( database );
    const int original_default = integer_default.back();
    integer_default.back() = 44;
    moab_check( database.tag_delete( integer_tag ), "delete variable integer tag" );
    moab_check( database.tag_get_handle( "VARIABLE_INTEGER", integer_default.size(), moab::MB_TYPE_INTEGER,
                                         integer_tag, moab::MB_TAG_SPARSE | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT,
                                         integer_default.data() ),
                "recreate variable integer tag with mutated default" );
    set_variable_tag( database, integer_tag, entity, integers );
    check( database_inventory( database ) != baseline,
           "trailing variable integer default mutation changes the inventory digest" );
    integer_default.back() = original_default;
    moab_check( database.tag_delete( integer_tag ), "delete mutated variable integer tag" );
    moab_check( database.tag_get_handle( "VARIABLE_INTEGER", integer_default.size(), moab::MB_TYPE_INTEGER,
                                         integer_tag, moab::MB_TAG_SPARSE | moab::MB_TAG_VARLEN | moab::MB_TAG_CREAT,
                                         integer_default.data() ),
                "restore variable integer tag default" );
    set_variable_tag( database, integer_tag, entity, integers );
    check( database_inventory( database ) == baseline,
           "restoring the variable integer default restores the inventory digest" );

    const auto check_mutation = [&]( moab::Tag tag, moab::EntityHandle tagged_entity, auto& values, auto replacement,
                                     const char* description ) {
        const auto original = values.back();
        values.back() = replacement;
        set_variable_tag( database, tag, tagged_entity, values );
        check( database_inventory( database ) != baseline, std::string( description ) + " changes the inventory digest" );
        values.back() = original;
        set_variable_tag( database, tag, tagged_entity, values );
        check( database_inventory( database ) == baseline, std::string( description ) + " restores the inventory digest" );
    };
    check_mutation( double_tag, root, doubles, 4.5, "trailing variable double root value mutation" );
    check_mutation( handle_tag, entity, handles, vertices[3], "trailing variable handle value mutation" );
}

void canonical_scalar_encoding()
{
    if constexpr( sizeof( PetscReal ) == 8 )
    {
        const auto encoded = KARMA::detail::encode_checkpoint_scalar( static_cast< PetscScalar >( 1.0 ) );
        const std::array< unsigned char, 8 > expected{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f };
        check( encoded.size() >= expected.size() && std::equal( expected.begin(), expected.end(), encoded.begin() ),
               "checkpoint scalar uses canonical little-endian IEEE-754 bytes" );
        const std::array< unsigned char, 8 > synthetic{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xc0 };
        check( PetscRealPart( KARMA::detail::decode_checkpoint_scalar( synthetic ) ) == -2.5,
               "checkpoint scalar decoder interprets explicit little-endian bytes" );
    }
    else if constexpr( sizeof( PetscReal ) == 4 )
    {
        const auto encoded = KARMA::detail::encode_checkpoint_scalar( static_cast< PetscScalar >( 1.0 ) );
        const std::array< unsigned char, 4 > expected{ 0x00, 0x00, 0x80, 0x3f };
        check( encoded.size() >= expected.size() && std::equal( expected.begin(), expected.end(), encoded.begin() ),
               "checkpoint scalar uses canonical little-endian IEEE-754 bytes" );
        const std::array< unsigned char, 4 > synthetic{ 0x00, 0x00, 0x20, 0xc0 };
        check( PetscRealPart( KARMA::detail::decode_checkpoint_scalar( synthetic ) ) == -2.5f,
               "checkpoint scalar decoder interprets explicit little-endian bytes" );
    }
}

void checkpoint_failure_does_not_mutate_source()
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 2, 0 );
    KARMASystem system( mesh, "checkpoint-rollback" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    system.init();
    fill_solution( system.solution() );
    const DatabaseInventory before = collective_database_inventory( *mesh.moab() );
    const std::filesystem::path destination = "karma_checkpoint_publish_target.h5m";
    std::error_code ignored;
    std::filesystem::remove_all( destination, ignored );
    for( const auto& entry : std::filesystem::directory_iterator( "." ) )
        if( entry.path().filename().string().starts_with(
                ".karma_checkpoint_publish_target.h5m.karma-checkpoint-" ) )
            std::filesystem::remove_all( entry.path(), ignored );
    std::filesystem::create_directory( destination );
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).write_checkpoint( destination.string(), system, system.solution() );
           } ),
           "checkpoint publication failure is reported" );
    check( collective_database_inventory( *mesh.moab() ) == before,
           "failed checkpoint publication leaves the complete source digest unchanged" );
    bool reservation_found = false;
    for( const auto& entry : std::filesystem::directory_iterator( "." ) )
        reservation_found = reservation_found ||
                            entry.path().filename().string().starts_with(
                                ".karma_checkpoint_publish_target.h5m.karma-checkpoint-" );
    check( !reservation_found, "failed checkpoint publication removes its temporary reservation directory" );
    std::filesystem::remove_all( destination, ignored );
}

std::vector< unsigned char > root_manifest( moab::Interface& database )
{
    moab::Tag tag = nullptr;
    moab_check( database.tag_get_handle( "KARMA_FIELD_MANIFEST", tag ), "find checkpoint manifest" );
    const moab::EntityHandle root = 0;
    const void* data = nullptr;
    int size = 0;
    moab_check( database.tag_get_by_ptr( tag, &root, 1, &data, &size ), "read checkpoint manifest" );
    const auto* bytes = static_cast< const unsigned char* >( data );
    return { bytes, bytes + size };
}

std::uint64_t byte_checksum( std::span< const unsigned char > bytes )
{
    std::uint64_t result = 1469598103934665603ULL;
    for( const auto byte : bytes )
    {
        result ^= byte;
        result *= 1099511628211ULL;
    }
    return result;
}

void set_root_manifest( moab::Interface& database, const std::vector< unsigned char >& bytes )
{
    moab::Tag tag = nullptr;
    moab_check( database.tag_get_handle( "KARMA_FIELD_MANIFEST", tag ), "find checkpoint manifest" );
    const moab::EntityHandle root = 0;
    const void* data = bytes.data();
    const int size = static_cast< int >( bytes.size() );
    moab_check( database.tag_set_by_ptr( tag, &root, 1, &data, &size ), "replace checkpoint manifest" );
}

template < class Integer >
void store_little_endian( unsigned char* destination, Integer value )
{
    for( std::size_t byte = 0; byte < sizeof( value ); ++byte )
        destination[byte] = static_cast< unsigned char >( value >> ( 8 * byte ) );
}

template < class Integer > Integer load_little_endian( const unsigned char* source )
{
    Integer value = 0;
    for( std::size_t byte = 0; byte < sizeof( value ); ++byte )
        value |= static_cast< Integer >( source[byte] ) << ( 8 * byte );
    return value;
}

std::uint32_t take_word( const std::vector< unsigned char >& bytes, std::size_t& offset )
{
    const auto value = load_little_endian< std::uint32_t >( bytes.data() + offset );
    offset += sizeof( std::uint32_t );
    return value;
}

void skip_string_or_bytes( const std::vector< unsigned char >& bytes, std::size_t& offset )
{
    offset += take_word( bytes, offset );
}

void skip_basis_record( const std::vector< unsigned char >& bytes, std::size_t& offset )
{
    offset += 2 * sizeof( std::uint32_t ) + sizeof( std::uint64_t );
    skip_string_or_bytes( bytes, offset );
}

void skip_layout_record( const std::vector< unsigned char >& bytes, std::size_t& offset )
{
    offset += 2 * sizeof( std::uint32_t );
    skip_string_or_bytes( bytes, offset );
    offset += sizeof( std::uint64_t );
    skip_string_or_bytes( bytes, offset );
    offset += 4 * sizeof( std::uint32_t ) + 2 * sizeof( std::uint64_t );
    const auto has_applicability = take_word( bytes, offset );
    if( has_applicability ) offset += sizeof( std::uint64_t );
    const auto permutation_size = take_word( bytes, offset );
    offset += permutation_size * sizeof( std::uint32_t );
    offset += sizeof( std::uint64_t );
    const auto chunk_count = take_word( bytes, offset );
    offset += chunk_count * ( 2 * sizeof( std::uint32_t ) + sizeof( std::uint64_t ) );
}

struct PublishedLayout
{
    std::string tag_name;
    std::uint32_t storage = 0;
    std::uint64_t entity_count = 0;
    std::uint64_t candidate_count = 0;
    bool has_applicability = false;
    std::uint64_t record_count = 0;
    std::uint32_t chunk_count = 0;
};

PublishedLayout published_layout( moab::Interface& database, KARMA::GeometryVariant target_variant )
{
    const auto manifest = root_manifest( database );
    std::size_t offset = 6 * sizeof( std::uint32_t );
    skip_string_or_bytes( manifest, offset );
    const auto variable_count = take_word( manifest, offset );
    if( variable_count != 1 ) throw std::runtime_error( "threshold fixture must have one variable" );
    skip_string_or_bytes( manifest, offset );
    offset += 4 * sizeof( std::uint32_t );
    const auto basis_count = take_word( manifest, offset );
    for( std::uint32_t i = 0; i < basis_count; ++i ) skip_basis_record( manifest, offset );
    const auto layout_count = take_word( manifest, offset );
    for( std::uint32_t i = 0; i < layout_count; ++i )
    {
        const auto variant = take_word( manifest, offset );
        (void)take_word( manifest, offset );
        const auto name_size = take_word( manifest, offset );
        const std::string name( reinterpret_cast< const char* >( manifest.data() + offset ), name_size );
        offset += name_size + sizeof( std::uint64_t );
        skip_string_or_bytes( manifest, offset );
        (void)take_word( manifest, offset );
        const auto storage = take_word( manifest, offset );
        offset += 2 * sizeof( std::uint32_t );
        const auto entity_count = load_little_endian< std::uint64_t >( manifest.data() + offset );
        offset += sizeof( std::uint64_t );
        const auto candidate_count = load_little_endian< std::uint64_t >( manifest.data() + offset );
        offset += sizeof( std::uint64_t );
        const bool has_applicability = take_word( manifest, offset ) != 0;
        if( has_applicability ) offset += sizeof( std::uint64_t );
        const auto permutation_size = take_word( manifest, offset );
        offset += permutation_size * sizeof( std::uint32_t );
        const auto record_count = load_little_endian< std::uint64_t >( manifest.data() + offset );
        offset += sizeof( std::uint64_t );
        const auto chunk_count = take_word( manifest, offset );
        offset += chunk_count * ( 2 * sizeof( std::uint32_t ) + sizeof( std::uint64_t ) );
        if( variant == static_cast< std::uint32_t >( target_variant ) )
            return { name, storage, entity_count, candidate_count, has_applicability,
                     record_count, chunk_count };
    }
    throw std::runtime_error( "threshold fixture layout not found" );
}

struct ManifestRecordRanges
{
    std::vector< std::pair< std::size_t, std::size_t > > bases;
    std::vector< std::pair< std::size_t, std::size_t > > layouts;
};

ManifestRecordRanges first_variable_record_ranges( const std::vector< unsigned char >& manifest )
{
    std::size_t offset = 6 * sizeof( std::uint32_t );
    skip_string_or_bytes( manifest, offset );
    (void)take_word( manifest, offset );
    skip_string_or_bytes( manifest, offset );
    offset += 4 * sizeof( std::uint32_t );
    ManifestRecordRanges result;
    const auto basis_count = take_word( manifest, offset );
    for( std::uint32_t i = 0; i < basis_count; ++i )
    {
        const auto begin = offset;
        skip_basis_record( manifest, offset );
        result.bases.emplace_back( begin, offset );
    }
    const auto layout_count = take_word( manifest, offset );
    for( std::uint32_t i = 0; i < layout_count; ++i )
    {
        const auto begin = offset;
        skip_layout_record( manifest, offset );
        result.layouts.emplace_back( begin, offset );
    }
    return result;
}

void swap_manifest_records( std::vector< unsigned char >& manifest,
                            std::pair< std::size_t, std::size_t > first,
                            std::pair< std::size_t, std::size_t > second )
{
    std::vector< unsigned char > replacement;
    replacement.insert( replacement.end(), manifest.begin() + static_cast< std::ptrdiff_t >( second.first ),
                         manifest.begin() + static_cast< std::ptrdiff_t >( second.second ) );
    replacement.insert( replacement.end(), manifest.begin() + static_cast< std::ptrdiff_t >( first.first ),
                         manifest.begin() + static_cast< std::ptrdiff_t >( first.second ) );
    manifest.erase( manifest.begin() + static_cast< std::ptrdiff_t >( first.first ),
                    manifest.begin() + static_cast< std::ptrdiff_t >( second.second ) );
    manifest.insert( manifest.begin() + static_cast< std::ptrdiff_t >( first.first ),
                     replacement.begin(), replacement.end() );
}

void update_manifest_chunk( moab::Interface& database, std::uint64_t old_checksum,
                            std::uint32_t old_record_count, std::uint32_t record_count,
                            std::uint64_t new_checksum )
{
    auto manifest = root_manifest( database );
    std::array< unsigned char, 8 > old_bytes{};
    store_little_endian( old_bytes.data(), old_checksum );
    const auto position = std::search( manifest.begin(), manifest.end(), old_bytes.begin(), old_bytes.end() );
    if( position == manifest.end() || position - manifest.begin() < 20 )
        throw std::runtime_error( "chunk checksum not found in manifest" );
    auto* checksum = manifest.data() + ( position - manifest.begin() );
    const std::uint64_t total_records_on_disk = load_little_endian< std::uint64_t >( checksum - 20 );
    const std::uint64_t total_records = total_records_on_disk - old_record_count + record_count;
    store_little_endian( checksum - 4, record_count );
    store_little_endian( checksum - 20, total_records );
    store_little_endian( checksum, new_checksum );
    set_root_manifest( database, manifest );
}

void replace_manifest_checksum( moab::Interface& database, std::uint64_t old_checksum,
                                std::uint64_t new_checksum )
{
    auto manifest = root_manifest( database );
    std::array< unsigned char, 8 > old_bytes{};
    std::array< unsigned char, 8 > new_bytes{};
    store_little_endian( old_bytes.data(), old_checksum );
    store_little_endian( new_bytes.data(), new_checksum );
    const auto position = std::search( manifest.begin(), manifest.end(), old_bytes.begin(), old_bytes.end() );
    if( position == manifest.end() ) throw std::runtime_error( "chunk checksum not found in manifest" );
    std::copy( new_bytes.begin(), new_bytes.end(), position );
    set_root_manifest( database, manifest );
}

bool has_tag_prefix( moab::Interface& database, std::string_view prefix )
{
    std::vector< moab::Tag > tags;
    moab_check( database.tag_get_tags( tags ), "enumerate checkpoint tags" );
    return std::any_of( tags.begin(), tags.end(), [&]( moab::Tag tag ) {
        std::string name;
        moab_check( database.tag_get_name( tag, name ), "read checkpoint tag name" );
        return name.starts_with( prefix );
    } );
}

struct GlobalChunk
{
    moab::Tag tag = nullptr;
    moab::EntityHandle meshset = 0;
    std::string name;
    std::vector< unsigned char > bytes;
};

void make_global_layout_storage_noncanonical( moab::Interface& database, GlobalChunk& chunk )
{
    auto manifest = root_manifest( database );
    const std::string base_name = chunk.name.substr( 0, chunk.name.rfind( "_C" ) );
    const auto name = std::search( manifest.begin(), manifest.end(), base_name.begin(), base_name.end() );
    if( name == manifest.end() ) throw std::runtime_error( "global layout name not found in manifest" );
    auto* hash = manifest.data() + ( name - manifest.begin() ) + base_name.size();
    const std::uint32_t descriptor_size = load_little_endian< std::uint32_t >( hash + 8 );
    auto* descriptor = hash + 12;
    const std::uint32_t variable_name_size = load_little_endian< std::uint32_t >( descriptor );
    auto* descriptor_storage = descriptor + 4 + variable_name_size + 7 * sizeof( std::uint32_t );
    store_little_endian( descriptor_storage, std::uint32_t{ 0 } );
    auto* outer_storage = descriptor + descriptor_size + sizeof( std::uint32_t );
    store_little_endian( outer_storage, std::uint32_t{ 0 } );
    const std::uint64_t new_hash = byte_checksum(
        std::span< const unsigned char >( descriptor, descriptor_size ) );
    const std::string old_hash = base_name.substr( base_name.rfind( '_' ) + 1 );
    const std::string new_hash_text = std::to_string( new_hash );
    if( new_hash_text.size() != old_hash.size() )
        throw std::runtime_error( "noncanonical test hash width changed" );
    std::copy( new_hash_text.begin(), new_hash_text.end(),
               name + static_cast< std::ptrdiff_t >( base_name.size() - old_hash.size() ) );
    store_little_endian( hash, new_hash );
    set_root_manifest( database, manifest );

    moab_check( database.tag_delete( chunk.tag ), "delete canonical chunk payload tag" );
    const std::string prefix = base_name.substr( 0, base_name.size() - old_hash.size() );
    chunk.name = prefix + new_hash_text + "_C0";
    moab_check( database.tag_get_handle( chunk.name.c_str(), static_cast< int >( chunk.bytes.size() ),
                                         moab::MB_TYPE_OPAQUE, chunk.tag,
                                         moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                "create noncanonical chunk payload tag" );
    moab_check( database.tag_set_data( chunk.tag, &chunk.meshset, 1, chunk.bytes.data() ),
                "write noncanonical chunk payload" );
}

void replace_chunk_payload( moab::Interface& database, GlobalChunk& chunk,
                            const std::vector< unsigned char >& bytes )
{
    moab_check( database.tag_delete( chunk.tag ), "delete old global chunk tag" );
    moab_check( database.tag_get_handle( chunk.name.c_str(), static_cast< int >( bytes.size() ),
                                         moab::MB_TYPE_OPAQUE, chunk.tag,
                                         moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                "recreate global chunk tag" );
    moab_check( database.tag_set_data( chunk.tag, &chunk.meshset, 1, bytes.data() ),
                "replace global chunk payload" );
}

GlobalChunk global_chunk( moab::Interface& database, std::string_view prefix, unsigned chunk_id = 0 )
{
    std::vector< moab::Tag > tags;
    moab_check( database.tag_get_tags( tags ), "enumerate global chunk tags" );
    for( const auto tag : tags )
    {
        std::string name;
        moab_check( database.tag_get_name( tag, name ), "read global chunk tag name" );
        if( !name.starts_with( prefix ) || !name.ends_with( "_C" + std::to_string( chunk_id ) ) ) continue;
        moab::Range meshsets;
        moab_check( database.get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &tag, nullptr, 1, meshsets ),
                    "find global chunk meshset" );
        if( meshsets.size() != 1 ) throw std::runtime_error( "global chunk does not have one meshset" );
        int size = 0;
        moab_check( database.tag_get_bytes( tag, size ), "read global chunk size" );
        GlobalChunk result{ tag, meshsets.front(), name,
                            std::vector< unsigned char >( static_cast< std::size_t >( size ) ) };
        moab_check( database.tag_get_data( tag, &result.meshset, 1, result.bytes.data() ), "read global chunk" );
        return result;
    }
    throw std::runtime_error( "global chunk not found" );
}

void add_disjoint_quads( KARMA::KARMAMesh& mesh, KARMA::GeometryVariant variant, unsigned count,
                         double x_offset )
{
    const KARMA::KARMAReferenceTopology topology( variant );
    for( unsigned element_number = 0; element_number < count; ++element_number )
    {
        std::vector< double > coordinates;
        for( const auto& point : topology.reference_nodes() )
        {
            coordinates.push_back( point( 0 ) + x_offset + 2.0 * element_number );
            coordinates.push_back( point( 1 ) );
            coordinates.push_back( point( 2 ) );
        }
        moab::Range vertices;
        moab_check( mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( topology.node_count() ),
                                                   vertices ),
                    "create mixed-layout vertices" );
        std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
        moab::EntityHandle element = 0;
        moab_check( mesh.moab()->create_element( topology.topology(), connectivity.data(),
                                                 static_cast< int >( connectivity.size() ), element ),
                    "create mixed-layout element" );
    }
}

void repeated_checkpoint_replaces_sparse_storage_with_dense()
{
    using namespace KARMA;
    const std::string file = "karma_moab_repeated_threshold.h5m";
    const auto make_source = []( unsigned participants ) {
        auto mesh = std::make_unique< KARMAMesh >( 2, PETSC_COMM_WORLD );
        std::vector< double > coordinates;
        coordinates.reserve( 300 );
        for( unsigned vertex = 0; vertex < 100; ++vertex )
        {
            coordinates.push_back( static_cast< double >( vertex ) );
            coordinates.push_back( static_cast< double >( vertex % 7 ) );
            coordinates.push_back( 0.0 );
        }
        moab::Range vertices;
        moab_check( mesh->moab()->create_vertices( coordinates.data(), 100, vertices ),
                    "create threshold candidate vertices" );
        const std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
        for( unsigned element = 0; element < 5; ++element )
        {
            const auto* connectivity = handles.data() + 4 * element;
            moab::EntityHandle quad = 0;
            moab_check( mesh->moab()->create_element( moab::MBQUAD, connectivity, 4, quad ),
                        "create threshold participant quad" );
        }
        if( participants == 21 )
        {
            const std::array< moab::EntityHandle, 4 > connectivity{
                handles[0], handles[1], handles[2], handles[20]
            };
            moab::EntityHandle quad = 0;
            moab_check( mesh->moab()->create_element( moab::MBQUAD, connectivity.data(), connectivity.size(), quad ),
                        "create twenty-first threshold participant" );
        }
        mesh->assign_ids();
        mesh->update();
        auto system = std::make_unique< KARMASystem >( *mesh, "repeated-threshold" );
        system->add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
        system->init();
        fill_semantic_solution( *system, system->solution() );
        return std::pair{ std::move( mesh ), std::move( system ) };
    };

    auto [sparse_mesh, sparse_system] = make_source( 20 );
    const auto sparse_source = collective_database_inventory( *sparse_mesh->moab() );
    const auto sparse_values = vector_bytes( sparse_system->solution() );
    KARMAMoabIO( *sparse_mesh ).write_checkpoint( file, *sparse_system, sparse_system->solution() );
    check( collective_database_inventory( *sparse_mesh->moab() ) == sparse_source,
           "20/100 checkpoint write leaves the sparse source database unchanged" );
    check( vector_bytes( sparse_system->solution() ) == sparse_values,
           "20/100 checkpoint write leaves the sparse source vector unchanged" );
    std::string sparse_tag_name;
    {
        auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
        const auto layout = published_layout( *checkpoint.mesh().moab(), GeometryVariant::Quad4 );
        sparse_tag_name = layout.tag_name;
        check( layout.storage == 1 && layout.entity_count == 20 && layout.candidate_count == 100,
               "20/100 checkpoint uses sparse storage at the threshold" );
        compare_vectors( checkpoint.system().solution(), sparse_system->solution(),
                         "first publication restores exact 20/100 values and participants" );
    }

    auto [dense_mesh, dense_system] = make_source( 21 );
    const auto dense_source = collective_database_inventory( *dense_mesh->moab() );
    const auto dense_values = vector_bytes( dense_system->solution() );
    KARMAMoabIO( *dense_mesh ).write_checkpoint( file, *dense_system, dense_system->solution() );
    check( collective_database_inventory( *sparse_mesh->moab() ) == sparse_source,
           "second checkpoint write leaves the first source database unchanged" );
    check( vector_bytes( sparse_system->solution() ) == sparse_values,
           "second checkpoint write leaves the first source vector unchanged" );
    check( collective_database_inventory( *dense_mesh->moab() ) == dense_source,
           "21/100 checkpoint write leaves the dense source database unchanged" );
    check( vector_bytes( dense_system->solution() ) == dense_values,
           "21/100 checkpoint write leaves the dense source vector unchanged" );

    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    auto& database = *checkpoint.mesh().moab();
    const auto layout = published_layout( database, GeometryVariant::Quad4 );
    check( layout.storage == 0 && layout.entity_count == 21 && layout.candidate_count == 100 &&
               layout.has_applicability && layout.record_count == 0 && layout.chunk_count == 0,
           "replacement checkpoint has canonical dense partial storage without chunk metadata" );
    moab::Tag field = nullptr;
    moab_check( database.tag_get_handle( layout.tag_name.c_str(), field ), "find replacement field tag" );
    moab::TagType storage = moab::MB_TAG_ANY;
    moab_check( database.tag_get_type( field, storage ), "read replacement field tag storage" );
    check( storage == moab::MB_TAG_DENSE, "replacement checkpoint field tag is dense" );
    moab::Tag stale_sparse = nullptr;
    check( database.tag_get_handle( sparse_tag_name.c_str(), stale_sparse ) == moab::MB_TAG_NOT_FOUND,
           "replacement checkpoint does not retain the first sparse payload tag" );
    moab::Tag applicability = nullptr;
    moab_check( database.tag_get_handle( "KARMA_V6_APPLICABILITY_ID", applicability ),
                "find replacement applicability tag" );
    moab::Range selectors;
    moab_check( database.get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &applicability, nullptr, 1, selectors ),
                "find replacement applicability meshsets" );
    check( selectors.size() == 1, "replacement checkpoint has exactly one applicability meshset" );
    if( selectors.size() == 1 )
    {
        moab::Range participants;
        moab_check( database.get_entities_by_handle( selectors.front(), participants ),
                    "read replacement applicability participants" );
        check( participants.size() == 21, "replacement applicability contains exactly 21 participants" );
    }
    check( database.tag_get_handle( "KARMA_V6_CHUNK_ID", field ) == moab::MB_TAG_NOT_FOUND,
           "replacement checkpoint has no stale chunk identifier metadata" );
    compare_vectors( checkpoint.system().solution(), dense_system->solution(),
                     "second publication restores exact 21/100 values and participants" );
    std::remove( file.c_str() );
}

void partial_dense_checkpoint_survives_handle_reordering()
{
    using namespace KARMA;
    const std::string file = "karma_moab_shifted_partial_dense.h5m";
    KARMAMesh source( 2, PETSC_COMM_WORLD );
    add_disjoint_quads( source, GeometryVariant::Quad4, 21, 0.0 );
    add_disjoint_quads( source, GeometryVariant::Quad9, 79, 1000.0 );
    source.assign_ids();
    source.update();
    KARMASystem source_system( source, "shifted-applicability" );
    source_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    source_system.init();
    fill_semantic_solution( source_system, source_system.solution() );
    KARMAMoabIO( source ).write_checkpoint( file, source_system, source_system.solution() );

    KARMAMesh destination( 2, PETSC_COMM_WORLD );
    add_disjoint_quads( destination, GeometryVariant::Quad9, 3, -100.0 );
    destination.update();
    const moab::Range preexisting_elements = destination.elements();
    moab::Range preexisting_vertices;
    moab_check( destination.moab()->get_adjacencies( preexisting_elements, 0, false, preexisting_vertices,
                                                      moab::Interface::UNION ),
                "find pre-existing partial-dense vertices" );
    KARMAMoabIO( destination ).read_mesh( file );
    moab_check( destination.moab()->delete_entities( preexisting_elements ),
                "delete pre-existing partial-dense elements" );
    moab_check( destination.moab()->delete_entities( preexisting_vertices ),
                "delete pre-existing partial-dense vertices" );
    destination.update();
    check( destination.n_elements() == source.n_elements() && destination.n_vertices() == source.n_vertices(),
           "pre-existing entity removal preserves checkpoint candidate counts" );

    moab::Tag applicability = nullptr;
    moab_check( destination.moab()->tag_get_handle( "KARMA_V6_APPLICABILITY_ID", applicability ),
                "find shifted applicability identifier" );
    const int identifier = static_cast< int >( GeometryVariant::Quad9 ) + 1;
    const void* identifier_value = &identifier;
    moab::Range selectors;
    moab_check( destination.moab()->get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &applicability,
                                                                   &identifier_value, 1, selectors ),
                "find shifted applicability selector" );
    if( selectors.size() != 1 ) throw std::runtime_error( "partial-dense selector is not unique" );
    moab::Range original_participants;
    moab_check( destination.moab()->get_entities_by_handle( selectors.front(), original_participants ),
                "read shifted applicability participants" );
    moab::Range candidates = destination.vertices();
    moab::Range shifted_participants;
    auto candidate = candidates.begin();
    for( std::size_t i = 0; i < original_participants.size(); ++i, ++candidate )
        shifted_participants.insert( *candidate );
    moab_check( destination.moab()->remove_entities( selectors.front(), original_participants ),
                "remove raw applicability handles" );
    moab_check( destination.moab()->add_entities( selectors.front(), shifted_participants ),
                "install shifted raw applicability handles" );

    KARMASystem restored_system( destination, "shifted-applicability" );
    restored_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    restored_system.init();
    KARMAMoabIO( destination ).restore_solution( restored_system, restored_system.solution() );
    fill_semantic_solution( restored_system, restored_system.temporary_vector() );
    compare_vectors( restored_system.solution(), restored_system.temporary_vector(),
                     "partial dense applicability follows stable IDs after handle reordering" );
    std::remove( file.c_str() );
}

void noncanonical_manifest_order_is_rejected()
{
    using namespace KARMA;
    const std::string file = "karma_moab_noncanonical_order.h5m";
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    add_reference_element( mesh, GeometryVariant::Tri3 );
    add_disjoint_quads( mesh, GeometryVariant::Quad4, 1, 10.0 );
    mesh.assign_ids();
    mesh.update();
    KARMASystem system( mesh, "manifest-order" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::dG, 1 } );
    system.init();
    fill_solution( system.solution() );
    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );

    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    auto manifest = root_manifest( *checkpoint.mesh().moab() );
    const auto ranges = first_variable_record_ranges( manifest );
    if( ranges.bases.size() != 2 || ranges.layouts.size() != 2 )
        throw std::runtime_error( "manifest ordering fixture does not have two records" );
    swap_manifest_records( manifest, ranges.bases[0], ranges.bases[1] );
    set_root_manifest( *checkpoint.mesh().moab(), manifest );
    check( throws_invalid_containing( [&] {
               KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(),
                                                                  checkpoint.system().temporary_vector() );
           }, "basis descriptors are not canonical" ),
           "restore rejects reordered basis descriptors" );

    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    manifest = root_manifest( *checkpoint.mesh().moab() );
    const auto layout_ranges = first_variable_record_ranges( manifest ).layouts;
    swap_manifest_records( manifest, layout_ranges[0], layout_ranges[1] );
    set_root_manifest( *checkpoint.mesh().moab(), manifest );
    check( throws_invalid_containing( [&] {
               KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(),
                                                                  checkpoint.system().temporary_vector() );
           }, "layouts are not canonical" ),
           "restore rejects reordered layout records" );
    std::remove( file.c_str() );
}

void noncanonical_applicability_is_rejected()
{
    using namespace KARMA;
    const std::string file = "karma_moab_noncanonical_applicability.h5m";
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    add_disjoint_quads( mesh, GeometryVariant::Quad4, 21, 0.0 );
    add_disjoint_quads( mesh, GeometryVariant::Quad9, 79, 1000.0 );
    mesh.assign_ids();
    mesh.update();
    KARMASystem system( mesh, "applicability" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    system.init();
    fill_solution( system.solution() );
    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    auto manifest = root_manifest( *checkpoint.mesh().moab() );
    const std::string marker = "KARMA_V6_F0_G6_V_";
    const auto name = std::search( manifest.begin(), manifest.end(), marker.begin(), marker.end() );
    if( name == manifest.end() ) throw std::runtime_error( "dense partial layout not found" );
    const auto name_offset = static_cast< std::size_t >( name - manifest.begin() );
    const auto name_size = load_little_endian< std::uint32_t >( manifest.data() + name_offset - 4 );
    auto* hash = manifest.data() + ( name - manifest.begin() ) + name_size;
    const auto descriptor_size = load_little_endian< std::uint32_t >( hash + 8 );
    auto* descriptor = hash + 12;
    const auto variable_name_size = load_little_endian< std::uint32_t >( descriptor );
    auto* descriptor_applicability = descriptor + 4 + variable_name_size + 10 * sizeof( std::uint32_t ) +
                                     2 * sizeof( std::uint64_t );
    auto* outer_applicability = descriptor + descriptor_size + 4 * sizeof( std::uint32_t ) +
                                2 * sizeof( std::uint64_t );
    const auto original_id = load_little_endian< std::uint64_t >( outer_applicability + 4 );
    store_little_endian( descriptor_applicability + 4, original_id + 1 );
    store_little_endian( outer_applicability + 4, original_id + 1 );
    const auto new_hash = byte_checksum( std::span< const unsigned char >( descriptor, descriptor_size ) );
    const std::string old_name( reinterpret_cast< const char* >( manifest.data() + name_offset ), name_size );
    const std::string old_hash = old_name.substr( old_name.rfind( '_' ) + 1 );
    const std::string new_hash_text = std::to_string( new_hash );
    if( old_hash.size() != new_hash_text.size() ) throw std::runtime_error( "applicability hash width changed" );
    std::copy( new_hash_text.begin(), new_hash_text.end(), name + name_size - old_hash.size() );
    store_little_endian( hash, new_hash );
    set_root_manifest( *checkpoint.mesh().moab(), manifest );
    check( throws_invalid_containing( [&] {
               KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(),
                                                                  checkpoint.system().temporary_vector() );
           }, "not canonical" ),
           "restore rejects a noncanonical applicability identifier" );
    std::remove( file.c_str() );
}

void bounded_global_chunks()
{
    using namespace KARMA;
    const std::string file = "karma_moab_bounded_chunks.h5m";
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 40, 40, 0 );
    KARMASystem system( mesh, "bounded-chunks" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    system.init();
    fill_solution( system.solution() );
    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    std::vector< moab::Tag > tags;
    moab_check( checkpoint.mesh().moab()->tag_get_tags( tags ), "enumerate bounded chunk tags" );
    std::size_t chunks = 0;
    constexpr int maximum_chunk_bytes = 16 + 1024 *
        ( KARMA::DofKey::width * static_cast< int >( sizeof( std::uint64_t ) ) +
          static_cast< int >( sizeof( PetscScalar ) ) );
    for( const auto tag : tags )
    {
        std::string name;
        moab_check( checkpoint.mesh().moab()->tag_get_name( tag, name ), "read bounded chunk tag name" );
        if( !name.starts_with( "KARMA_V6_F0_G4_A_" ) || name.find( "_C" ) == std::string::npos ) continue;
        int bytes = 0;
        moab_check( checkpoint.mesh().moab()->tag_get_bytes( tag, bytes ), "read bounded chunk width" );
        check( bytes <= maximum_chunk_bytes, "global DoF chunk obeys the record bound" );
        ++chunks;
    }
    check( chunks > 1, "large global DoF arrays are split into multiple chunks" );
    compare_vectors( checkpoint.system().solution(), system.solution(), "multi-chunk global DoF array round-trips" );

    auto first = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_", 0 );
    auto second = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_", 1 );
    auto last = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_",
                              static_cast< unsigned >( chunks - 1 ) );
#if defined(PETSC_USE_COMPLEX)
    constexpr std::size_t scalar_components = 2;
#else
    constexpr std::size_t scalar_components = 1;
#endif
    constexpr std::size_t record_size = KARMA::DofKey::width * sizeof( std::uint64_t ) +
                                        scalar_components * sizeof( PetscReal );
    const std::uint32_t first_count = load_little_endian< std::uint32_t >( first.bytes.data() + 8 );
    const std::uint32_t last_count = load_little_endian< std::uint32_t >( last.bytes.data() + 8 );
    check( first_count == 1024 && last_count <= 512, "chunk split fixture has movable records" );
    auto reordered_first = first.bytes;
    auto reordered_second = second.bytes;
    std::swap_ranges( reordered_first.end() - static_cast< std::ptrdiff_t >( record_size ),
                      reordered_first.end(), reordered_second.begin() + 16 );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( first.bytes ),
                               byte_checksum( reordered_first ) );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( second.bytes ),
                               byte_checksum( reordered_second ) );
    replace_chunk_payload( *checkpoint.mesh().moab(), first, reordered_first );
    replace_chunk_payload( *checkpoint.mesh().moab(), second, reordered_second );
    check( throws_invalid_containing( [&] {
               KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(),
                                                                  checkpoint.system().temporary_vector() );
           }, "not strictly increasing" ),
           "restore rejects global DoF record disorder across chunks" );

    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    first = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_", 0 );
    last = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_",
                         static_cast< unsigned >( chunks - 1 ) );
    auto moved = first.bytes;
    moved.erase( moved.begin() + static_cast< std::ptrdiff_t >( 16 + 512 * record_size ), moved.end() );
    store_little_endian( moved.data() + 8, std::uint32_t{ 512 } );
    auto enlarged = last.bytes;
    enlarged.insert( enlarged.begin() + 16, first.bytes.begin() + static_cast< std::ptrdiff_t >( 16 + 512 * record_size ),
                     first.bytes.end() );
    store_little_endian( enlarged.data() + 8, last_count + 512 );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( first.bytes ), first_count, 512,
                           byte_checksum( moved ) );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( last.bytes ), last_count, last_count + 512,
                           byte_checksum( enlarged ) );
    replace_chunk_payload( *checkpoint.mesh().moab(), first, moved );
    replace_chunk_payload( *checkpoint.mesh().moab(), last, enlarged );
    check( throws_invalid_containing( [&] {
               KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(),
                                                                  checkpoint.system().temporary_vector() );
           }, "chunk layout is invalid" ),
           "restore rejects a noncanonical split of valid global DoF records" );
    std::remove( file.c_str() );
}

void exact_checkpoint_case( KARMA::GeometryVariant variant, unsigned ordinal )
{
    using namespace KARMA;
    const KARMAReferenceTopology topology( variant );
    KARMAMesh mesh( topology.dimension(), PETSC_COMM_WORLD );
    add_reference_element( mesh, variant );
    KARMASystem system( mesh, "topology-" + std::to_string( ordinal ) );
    system.add_variable( { "cg1", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.add_variable( { "cg2", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 } );
    for( const auto continuity : { BasisContinuity::dG, BasisContinuity::SE } )
        for( unsigned order = 0; order <= 2; ++order )
            system.add_variable( { "local-" + std::to_string( static_cast< unsigned >( continuity ) ) + "-" +
                                       std::to_string( order ),
                                   FieldBasisFamily::LagrangeNodal, order, continuity, 1 } );
    system.init();
    fill_solution( system.solution() );

    std::array< std::size_t, 3 > lower_entities{};
    for( unsigned dimension = 1; dimension < topology.dimension(); ++dimension )
    {
        moab::Range entities;
        moab_check( mesh.moab()->get_entities_by_dimension( 0, static_cast< int >( dimension ), entities ),
                    "count lower-dimensional entities before checkpoint" );
        lower_entities[dimension] = entities.size();
    }
    const std::string file = "karma_moab_topology_" + std::to_string( ordinal ) + ".h5m";
    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    for( unsigned dimension = 1; dimension < topology.dimension(); ++dimension )
    {
        moab::Range entities;
        moab_check( mesh.moab()->get_entities_by_dimension( 0, static_cast< int >( dimension ), entities ),
                    "count lower-dimensional entities after checkpoint" );
        check( entities.size() == lower_entities[dimension], "checkpoint does not create edges or faces" );
    }
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    if( rank == 0 )
    {
        auto& database = *checkpoint.mesh().moab();
        const auto manifest = root_manifest( database );
        check( manifest.size() >= 4 && manifest[0] == 6 && manifest[1] == 0 && manifest[2] == 0 && manifest[3] == 0,
               "checkpoint uses schema v6" );
        const std::string geometry = "_G" + std::to_string( static_cast< unsigned >( variant ) );
        const bool cg1_vertex = has_tag_prefix( database, "KARMA_V6_F0" + geometry + "_V_" );
        const bool cg1_global = has_tag_prefix( database, "KARMA_V6_F0" + geometry + "_A_" );
        const bool cg2_vertex = has_tag_prefix( database, "KARMA_V6_F1" + geometry + "_V_" );
        const bool cg2_global = has_tag_prefix( database, "KARMA_V6_F1" + geometry + "_A_" );
        const bool cg1_bijection = variant == GeometryVariant::Edge2 || variant == GeometryVariant::Tri3 ||
                                   variant == GeometryVariant::Quad4;
        const bool cg2_bijection = variant == GeometryVariant::Quad9;
        check( cg1_vertex == cg1_bijection && cg1_global == !cg1_bijection,
               "P1 cG uses global arrays unless it is a complete geometry-vertex bijection" );
        check( cg2_vertex == cg2_bijection && cg2_global == !cg2_bijection,
               "P2 cG uses global arrays unless it is a complete geometry-vertex bijection" );
        check( has_tag_prefix( database, "KARMA_V6_F2" + geometry + "_P_" ),
               "dG remains element-packed" );
        check( has_tag_prefix( database, "KARMA_V6_F5" + geometry + "_P_" ),
               "SE remains element-packed" );
    }
    compare_vectors( checkpoint.system().solution(), system.solution(), "all topology DoFs round-trip" );
    std::remove( file.c_str() );
}

void run_exact_tests()
{
    using namespace KARMA;
    checkpoint_failure_does_not_mutate_source();
    const std::vector< GeometryVariant > variants{
        GeometryVariant::Edge2, GeometryVariant::Tri3, GeometryVariant::Quad4, GeometryVariant::Quad9,
        GeometryVariant::Tet4, GeometryVariant::Hex8, GeometryVariant::Prism6, GeometryVariant::Pyramid5
    };
    for( unsigned i = 0; i < variants.size(); ++i ) exact_checkpoint_case( variants[i], i );
    bounded_global_chunks();
    repeated_checkpoint_replaces_sparse_storage_with_dense();
    partial_dense_checkpoint_survives_handle_reordering();
    noncanonical_manifest_order_is_rejected();
    noncanonical_applicability_is_rejected();
    const std::string file = "karma_moab_io_test.h5m";
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 2, 0 );
    KARMASystem system( mesh, "mixed" );
    system.add_variable( { "constant", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
    system.add_variable( { "nodal", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 } );
    system.add_variable( { "modal", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::dG, 1 } );
    system.add_variable( { "spectral", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 } );
    system.init();
    fill_solution( system.solution() );

    KARMAMoabIO output( mesh );
    const DatabaseInventory before_checkpoint = collective_database_inventory( *mesh.moab() );
    output.write_checkpoint( file, system, system.solution() );
    check( collective_database_inventory( *mesh.moab() ) == before_checkpoint,
           "successful checkpoint publication leaves the complete source digest unchanged" );
    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    check( checkpoint.system().name() == "mixed", "checkpoint restores the system name" );
    check( checkpoint.system().n_variables() == 4, "checkpoint restores all variables" );
    compare_vectors( checkpoint.system().solution(), system.solution(), "mixed checkpoint restores every DoF" );

    Vec restored = checkpoint.system().create_vector();
    KARMAMoabIO input( checkpoint.mesh() );
    input.restore_solution( checkpoint.system(), restored );
    compare_vectors( restored, checkpoint.system().solution(), "existing-system restore matches constructed restore" );
    VecDestroy( &restored );

    KARMASystem incompatible( checkpoint.mesh(), "wrong-name" );
    incompatible.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    incompatible.init();
    check( throws_invalid( [&] { input.restore_solution( incompatible, incompatible.solution() ); } ),
           "incompatible checkpoint metadata is rejected" );
    check( throws_invalid( [&] { output.write_mesh( "karma_moab_io_test.exo" ); } ),
           "ExodusII is rejected before MOAB fallback" );
    check( throws_invalid( [&] { output.write_checkpoint( "karma_moab_io_test.vtk", system, system.solution() ); } ),
           "lossy formats are rejected for exact checkpoints" );

    const std::string custom_file = "karma_moab_custom_basis.h5m";
    KARMABasisRegistry custom_registry;
    register_oriented_basis( custom_registry );
    KARMASystem custom( mesh, "custom", custom_registry );
    custom.add_variable( { "edge-modes", FieldBasisFamily::UserDefined, 3, BasisContinuity::cG, 2 } );
    custom.init();
    fill_solution( custom.solution() );
    const std::size_t edges_before = [&] {
        moab::Range edges;
        moab_check( mesh.moab()->get_entities_by_dimension( 0, 1, edges ), "count edges before checkpoint" );
        return edges.size();
    }();
    output.write_checkpoint( custom_file, custom, custom.solution() );
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    if( rank == 0 )
    {
        moab::Range edges_after;
        moab_check( mesh.moab()->get_entities_by_dimension( 0, 1, edges_after ), "count edges after checkpoint" );
        check( edges_after.size() == edges_before, "checkpoint storage does not create edges" );
    }
    check( throws_invalid( [&] { (void)KARMAMoabIO::load_checkpoint( custom_file, PETSC_COMM_WORLD ); } ),
           "checkpoint construction rejects a missing custom basis factory" );
    KARMABasisRegistry wrong_registry;
    register_different_basis( wrong_registry );
    check( throws_invalid( [&] {
               (void)KARMAMoabIO::load_checkpoint( custom_file, PETSC_COMM_WORLD, std::move( wrong_registry ) );
           } ),
           "checkpoint construction rejects different custom basis semantics under the same key" );
    KARMABasisRegistry restore_registry;
    register_oriented_basis( restore_registry );
    auto custom_checkpoint = KARMAMoabIO::load_checkpoint( custom_file, PETSC_COMM_WORLD, std::move( restore_registry ) );
    compare_vectors( custom_checkpoint.system().solution(), custom.solution(),
                     "custom oriented basis round-trips with a supplied registry" );

    Vec unchanged = checkpoint.system().create_vector();
    Vec expected_unchanged = checkpoint.system().create_vector();
    petsc_check( VecSet( unchanged, 73.0 ), "VecSet(corruption destination)" );
    petsc_check( VecSet( expected_unchanged, 73.0 ), "VecSet(corruption sentinel)" );
    moab::Tag field_tag = nullptr;
    std::vector< moab::Tag > tags;
    moab_check( checkpoint.mesh().moab()->tag_get_tags( tags ), "enumerate checkpoint tags" );
    for( const auto tag : tags )
    {
        std::string name;
        moab_check( checkpoint.mesh().moab()->tag_get_name( tag, name ), "read checkpoint tag name" );
        if( name.starts_with( "KARMA_V6_F1_G4_A_" ) && name.ends_with( "_C0" ) )
        {
            field_tag = tag;
            break;
        }
    }
    if( field_tag == nullptr ) throw std::runtime_error( "find global cG chunk for corruption" );
    moab::Range chunks;
    moab_check( checkpoint.mesh().moab()->get_entities_by_type_and_tag( 0, moab::MBENTITYSET, &field_tag,
                                                                         nullptr, 1, chunks ),
                "find global cG chunk meshset" );
    check( chunks.size() == 1, "global cG chunk has one meshset" );
    int bytes = 0;
    moab_check( checkpoint.mesh().moab()->tag_get_bytes( field_tag, bytes ), "read global cG chunk width" );
    std::vector< unsigned char > corrupt_payload( static_cast< std::size_t >( bytes ) );
    const moab::EntityHandle chunk = chunks.front();
    moab_check( checkpoint.mesh().moab()->tag_get_data( field_tag, &chunk, 1, corrupt_payload.data() ),
                "read global cG chunk for corruption" );
    corrupt_payload.back() ^= 1;
    moab_check( checkpoint.mesh().moab()->tag_set_data( field_tag, &chunk, 1, corrupt_payload.data() ),
                "corrupt global cG chunk" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects a global cG chunk checksum mismatch" );
    compare_vectors( unchanged, expected_unchanged, "failed global-cG restore leaves the destination unchanged" );
    corrupt_payload.back() ^= 1;
    moab_check( checkpoint.mesh().moab()->tag_set_data( field_tag, &chunk, 1, corrupt_payload.data() ),
                "restore global cG chunk" );

    const auto original = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F1_G4_A_" );
    auto reordered = original.bytes;
#if defined(PETSC_USE_COMPLEX)
    constexpr std::size_t scalar_components = 2;
#else
    constexpr std::size_t scalar_components = 1;
#endif
    constexpr std::size_t record_size = KARMA::DofKey::width * sizeof( std::uint64_t ) +
                                        scalar_components * sizeof( PetscReal );
    std::swap_ranges( reordered.begin() + 16,
                      reordered.begin() + static_cast< std::ptrdiff_t >( 16 + record_size ),
                      reordered.begin() + static_cast< std::ptrdiff_t >( 16 + record_size ) );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( original.bytes ), byte_checksum( reordered ) );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, reordered.data() ),
                "write reordered global cG records" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "not strictly increasing" ),
           "restore rejects noncanonical global DoF record order" );
    compare_vectors( unchanged, expected_unchanged, "reordered-record restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, original.bytes.data() ),
                "restore global cG record order" );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( reordered ), byte_checksum( original.bytes ) );

    auto malformed = original.bytes;
    std::fill_n( malformed.begin() + 16 + 5 * sizeof( std::uint64_t ), sizeof( std::uint64_t ), 0 );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( original.bytes ), byte_checksum( malformed ) );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, malformed.data() ),
                "write malformed global cG key" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects a malformed global DoF key" );
    compare_vectors( unchanged, expected_unchanged, "malformed-key restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, original.bytes.data() ),
                "restore valid global cG key" );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( malformed ), byte_checksum( original.bytes ) );

    auto nonfinite = original.bytes;
    const std::size_t scalar_offset = 16 + KARMA::DofKey::width * sizeof( std::uint64_t );
    const PetscScalar nan = static_cast< PetscScalar >( std::numeric_limits< PetscReal >::quiet_NaN() );
    std::memcpy( nonfinite.data() + scalar_offset, &nan, sizeof( nan ) );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( original.bytes ), byte_checksum( nonfinite ) );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, nonfinite.data() ),
                "write nonfinite global cG value" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects a nonfinite global DoF value" );
    compare_vectors( unchanged, expected_unchanged, "nonfinite restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( original.tag, &original.meshset, 1, original.bytes.data() ),
                "restore finite global cG value" );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( nonfinite ), byte_checksum( original.bytes ) );

    const std::uint32_t original_count = load_little_endian< std::uint32_t >( original.bytes.data() + 8 );

    auto missing_record = original.bytes;
    missing_record.erase( missing_record.begin() + 16,
                          missing_record.begin() + static_cast< std::ptrdiff_t >( 16 + record_size ) );
    store_little_endian( missing_record.data() + 8, original_count - 1 );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( original.bytes ), original_count,
                           original_count - 1, byte_checksum( missing_record ) );
    auto mutable_chunk = original;
    replace_chunk_payload( *checkpoint.mesh().moab(), mutable_chunk, missing_record );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "missing a destination global DoF key" ),
           "restore reaches exact coverage validation for a missing global DoF record" );
    compare_vectors( unchanged, expected_unchanged, "missing-record restore leaves the destination unchanged" );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( missing_record ), original_count - 1,
                           original_count, byte_checksum( original.bytes ) );
    replace_chunk_payload( *checkpoint.mesh().moab(), mutable_chunk, original.bytes );

    auto extra_record = original.bytes;
    std::vector< unsigned char > unknown(
        original.bytes.end() - static_cast< std::ptrdiff_t >( record_size ), original.bytes.end() );
    const auto last_slot = load_little_endian< std::uint64_t >(
        unknown.data() + 4 * sizeof( std::uint64_t ) );
    store_little_endian( unknown.data() + 4 * sizeof( std::uint64_t ), last_slot + 1 );
    extra_record.insert( extra_record.end(), unknown.begin(), unknown.end() );
    store_little_endian( extra_record.data() + 8, original_count + 1 );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( original.bytes ), original_count,
                           original_count + 1, byte_checksum( extra_record ) );
    replace_chunk_payload( *checkpoint.mesh().moab(), mutable_chunk, extra_record );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "extra global DoF record" ),
           "restore reaches exact coverage validation for an extra global DoF record" );
    compare_vectors( unchanged, expected_unchanged, "extra-record restore leaves the destination unchanged" );
    update_manifest_chunk( *checkpoint.mesh().moab(), byte_checksum( extra_record ), original_count + 1,
                           original_count, byte_checksum( original.bytes ) );
    replace_chunk_payload( *checkpoint.mesh().moab(), mutable_chunk, original.bytes );

    auto duplicate_record = original.bytes;
    std::copy_n( duplicate_record.begin() + 16, KARMA::DofKey::width * sizeof( std::uint64_t ),
                 duplicate_record.begin() + 16 + record_size );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( original.bytes ),
                               byte_checksum( duplicate_record ) );
    moab_check( checkpoint.mesh().moab()->tag_set_data( mutable_chunk.tag, &original.meshset, 1,
                                                        duplicate_record.data() ),
                "write duplicate global DoF record" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects duplicate global DoF records" );
    compare_vectors( unchanged, expected_unchanged, "duplicate-record restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( mutable_chunk.tag, &original.meshset, 1,
                                                        original.bytes.data() ),
                "restore unique global DoF records" );
    replace_manifest_checksum( *checkpoint.mesh().moab(), byte_checksum( duplicate_record ),
                               byte_checksum( original.bytes ) );

    moab::EntityHandle duplicate_chunk = 0;
    moab_check( checkpoint.mesh().moab()->create_meshset( moab::MESHSET_SET, duplicate_chunk ),
                "create duplicate global chunk meshset" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( mutable_chunk.tag, &duplicate_chunk, 1,
                                                        original.bytes.data() ),
                "write duplicate global chunk" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects duplicate chunks" );
    compare_vectors( unchanged, expected_unchanged, "duplicate-chunk restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->delete_entities( &duplicate_chunk, 1 ), "delete duplicate chunk meshset" );

    moab::Tag chunk_id_tag = nullptr;
    moab_check( checkpoint.mesh().moab()->tag_get_handle( "KARMA_V6_CHUNK_ID", chunk_id_tag ),
                "find chunk identifier tag" );
    moab::EntityHandle orphan_chunk = 0;
    moab_check( checkpoint.mesh().moab()->create_meshset( moab::MESHSET_SET, orphan_chunk ),
                "create orphan chunk meshset" );
    const int orphan_id = std::numeric_limits< int >::max();
    moab_check( checkpoint.mesh().moab()->tag_set_data( chunk_id_tag, &orphan_chunk, 1, &orphan_id ),
                "set orphan chunk identifier" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "chunk namespace" ),
           "restore rejects an orphan chunk identifier" );
    compare_vectors( unchanged, expected_unchanged, "orphan-chunk restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->delete_entities( &orphan_chunk, 1 ), "delete orphan chunk meshset" );

    moab::EntityHandle duplicate_id_chunk = 0;
    moab_check( checkpoint.mesh().moab()->create_meshset( moab::MESHSET_SET, duplicate_id_chunk ),
                "create duplicate-ID chunk meshset" );
    const int duplicate_id = 0;
    moab_check( checkpoint.mesh().moab()->tag_set_data( chunk_id_tag, &duplicate_id_chunk, 1, &duplicate_id ),
                "set duplicate chunk identifier" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "chunk namespace" ),
           "restore rejects a duplicate chunk identifier" );
    compare_vectors( unchanged, expected_unchanged, "duplicate-ID restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->delete_entities( &duplicate_id_chunk, 1 ),
                "delete duplicate-ID chunk meshset" );

    auto noncanonical = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F1_G4_A_" );
    make_global_layout_storage_noncanonical( *checkpoint.mesh().moab(), noncanonical );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "global array basis is invalid" ),
           "restore rejects a recomputed-hash noncanonical global layout" );
    compare_vectors( unchanged, expected_unchanged, "noncanonical-layout restore leaves the destination unchanged" );
    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    input = KARMAMoabIO( checkpoint.mesh() );
    mutable_chunk = global_chunk( *checkpoint.mesh().moab(), "KARMA_V6_F1_G4_A_" );

    moab::Tag orphan_payload = nullptr;
    const std::string orphan_name = original.name + "_C999999";
    moab_check( checkpoint.mesh().moab()->tag_get_handle( orphan_name.c_str(), 1, moab::MB_TYPE_OPAQUE,
                                                          orphan_payload,
                                                          moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ),
                "create undeclared chunk payload tag" );
    const unsigned char orphan_byte = 0;
    moab_check( checkpoint.mesh().moab()->tag_set_data( orphan_payload, &original.meshset, 1, &orphan_byte ),
                "set undeclared chunk payload" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "undeclared payload tags" ),
           "restore rejects an undeclared chunk payload tag" );
    compare_vectors( unchanged, expected_unchanged, "undeclared-payload restore leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_delete( orphan_payload ), "delete undeclared chunk payload tag" );

    moab_check( checkpoint.mesh().moab()->tag_get_handle( "KARMA_V6_CHUNK_ID", chunk_id_tag ),
                "refresh chunk identifier tag" );
    const moab::EntityHandle wrong_type_chunk = checkpoint.mesh().vertices().front();
    moab_check( checkpoint.mesh().moab()->tag_set_data( chunk_id_tag, &wrong_type_chunk, 1, &orphan_id ),
                "attach chunk identifier to a vertex" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "chunk namespace" ),
           "restore rejects a chunk identifier on a non-meshset" );
    compare_vectors( unchanged, expected_unchanged,
                     "non-meshset chunk identifier leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_delete_data( chunk_id_tag, &wrong_type_chunk, 1 ),
                "remove chunk identifier from vertex" );
    moab_check( checkpoint.mesh().moab()->tag_set_data( mutable_chunk.tag, &wrong_type_chunk, 1,
                                                        original.bytes.data() ),
                "attach chunk payload to a vertex" );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), unchanged ); },
                                      "attached to a non-meshset" ),
           "restore rejects a chunk payload on a non-meshset" );
    compare_vectors( unchanged, expected_unchanged,
                     "non-meshset chunk payload leaves the destination unchanged" );
    moab_check( checkpoint.mesh().moab()->tag_delete_data( mutable_chunk.tag, &wrong_type_chunk, 1 ),
                "remove chunk payload from vertex" );

    moab_check( checkpoint.mesh().moab()->tag_delete_data( mutable_chunk.tag, &original.meshset, 1 ),
                "remove global chunk payload" );
    check( throws_invalid( [&] { input.restore_solution( checkpoint.system(), unchanged ); } ),
           "restore rejects a missing chunk" );
    compare_vectors( unchanged, expected_unchanged, "missing-chunk restore leaves the destination unchanged" );
    VecDestroy( &unchanged );
    VecDestroy( &expected_unchanged );

    output.write_mesh( "karma_moab_io_test.vtk" );
    KARMAMesh vtk( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( vtk ).read_mesh( "karma_moab_io_test.vtk" );
    check( vtk.n_elements() == mesh.n_elements(), "VTK mesh interchange preserves elements" );
    output.write_mesh( "karma_moab_io_test.msh" );
    KARMAMesh gmsh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( gmsh ).read_mesh( "karma_moab_io_test.msh" );
    check( gmsh.n_elements() == mesh.n_elements(), "Gmsh mesh interchange preserves elements" );

    std::remove( file.c_str() );
    std::remove( custom_file.c_str() );
    std::remove( "karma_moab_io_test.vtk" );
    std::remove( "karma_moab_io_test.msh" );
}

void write_partition_fixture( const std::string& filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 4, 4, 0 );
    moab::Tag field = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "KARMA_PARTITION_TEST_FIELD", 1, moab::MB_TYPE_DOUBLE, field,
                                             moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ),
                "create partition fixture field" );
    std::vector< double > values;
    values.reserve( mesh.elements().size() );
    for( const auto element : mesh.elements() ) values.push_back( mesh.centroid( element )( 0 ) + 2 * mesh.centroid( element )( 1 ) );
    moab_check( mesh.moab()->tag_set_data( field, mesh.elements(), values.data() ), "set partition fixture field" );

    KARMASystem system( mesh, "partition-checkpoint" );
    system.add_variable( { "continuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 } );
    system.add_variable( { "discontinuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::dG, 1 } );
    system.add_variable( { "spectral", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 } );
    system.init();
    fill_semantic_solution( system, system.solution() );
    KARMAMoabIO( mesh ).write_checkpoint( filename, system, system.solution() );
}

void verify_partition_file( const std::string& filename )
{
    using namespace KARMA;
    auto checkpoint = KARMAMoabIO::load_checkpoint( filename, PETSC_COMM_WORLD );
    check( checkpoint.system().name() == "partition-checkpoint",
           "parallel output preserves checkpoint system metadata" );
    check( checkpoint.system().n_variables() == 3,
           "parallel output preserves checkpoint variable count" );
    const std::vector< VariableSpec > expected_variables{
        { "continuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 },
        { "discontinuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::dG, 1 },
        { "spectral", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 }
    };
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    if( rank == 0 )
        check( has_tag_prefix( *checkpoint.mesh().moab(), "KARMA_V6_F0_G4_A_" ),
               "changed-rank checkpoint contains a Q2 cG global DoF array" );
    for( unsigned variable = 0; variable < expected_variables.size(); ++variable )
    {
        const auto& actual = checkpoint.system().variable( variable );
        const auto& expected = expected_variables[variable];
        check( actual.name == expected.name && actual.family == expected.family && actual.order == expected.order &&
                   actual.continuity == expected.continuity && actual.components == expected.components,
               "parallel output preserves complete variable metadata" );
    }
    fill_semantic_solution( checkpoint.system(), checkpoint.system().temporary_vector() );
    compare_vectors( checkpoint.system().solution(), checkpoint.system().temporary_vector(),
                     "parallel output restores exact checkpoint values" );

    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh( filename );
    check( mesh.n_elements() == 16, "parallel output serial reread preserves all elements" );
    moab::Tag field = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "KARMA_PARTITION_TEST_FIELD", field ), "find partition output field" );
    std::vector< double > values( mesh.elements().size() );
    moab_check( mesh.moab()->tag_get_data( field, mesh.elements(), values.data() ), "read partition output field" );
    std::size_t offset = 0;
    for( const auto element : mesh.elements() )
    {
        const auto center = mesh.centroid( element );
        check( values[offset++] == center( 0 ) + 2 * center( 1 ),
               "parallel output preserves exact field values" );
    }
}

void asymmetric_read_filename()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).read_mesh( rank == 0 ? "rank-zero.h5m" : "rank-one.h5m",
                                              "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION" );
           } ),
           "asymmetric mesh-read filenames fail collectively" );
}

void replicated_parallel_write_rejected()
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 2, 0 );
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).write_mesh( "replicated-parallel-write.h5m", "PARALLEL=WRITE_PART" );
           } ),
           "replicated mesh rejects collective parallel-write options" );
    KARMASystem system( mesh, "replicated-write" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.init();
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).write_checkpoint( "replicated-parallel-checkpoint.h5m", system,
                                                     system.solution(), "PARALLEL=WRITE_PART" );
           } ),
           "replicated checkpoint rejects parallel options during preflight" );
    moab::Tag metadata = nullptr;
    check( mesh.moab()->tag_get_handle( "KARMA_CHECKPOINT_METADATA", metadata ) == moab::MB_TAG_NOT_FOUND,
           "failed checkpoint preflight does not mutate mesh tags" );
}

void owner_only_checkpoint_gather()
{
    using namespace KARMA;
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
    const std::string file = "karma_owner_only_checkpoint_" + std::to_string( ranks ) + ".h5m";
    KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    mesh.generate_box( 1, 0, 0 );
    KARMASystem system( mesh, "owner-only-gather" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
    system.init();
    fill_solution( system.solution() );

    KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() );
    const auto stats = detail::checkpoint_gather_stats();
    check( stats.local_records == static_cast< std::uint64_t >( system.n_local_dofs() ),
           "checkpoint gather serializes only locally owned DoFs" );
    const int local_empty_owner = system.n_local_dofs() == 0 ? 1 : 0;
    int empty_owners = 0;
    MPI_Allreduce( &local_empty_owner, &empty_owners, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    check( system.n_dofs() == 1 && empty_owners == ranks - 1,
           "checkpoint owner gather has actual zero-owned ranks" );
    check( stats.received_records == ( rank == 0 ? static_cast< std::uint64_t >( system.n_dofs() ) : 0 ),
           "only rank zero receives the global checkpoint records" );

    auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
    compare_vectors( checkpoint.system().solution(), system.solution(),
                     "owner-only checkpoint gather preserves schema-v6 values" );
    Vec destination = checkpoint.system().create_vector();
    petsc_check( VecSet( destination, static_cast< PetscScalar >( std::numeric_limits< PetscReal >::quiet_NaN() ) ),
                 "set owner-only restore failure destination" );
    KARMAMoabIO input( checkpoint.mesh() );
    check( throws_invalid_containing( [&] { input.restore_solution( checkpoint.system(), destination ); }, "nonfinite" ),
           "zero-owned-rank restore rejects a nonfinite destination collectively" );
    petsc_check( VecSet( destination, 0.0 ), "reset owner-only restore destination" );
    input.restore_solution( checkpoint.system(), destination );
    compare_vectors( destination, system.solution(), "zero-owned-rank restore retry succeeds" );
    const PetscScalar* read_values = nullptr;
    petsc_check( VecGetArrayRead( destination, &read_values ), "reacquire owner-only restore destination read array" );
    petsc_check( VecRestoreArrayRead( destination, &read_values ), "release owner-only restore destination read array" );
    PetscScalar* write_values = nullptr;
    petsc_check( VecGetArray( destination, &write_values ), "reacquire owner-only restore destination write array" );
    petsc_check( VecRestoreArray( destination, &write_values ), "release owner-only restore destination write array" );
    VecDestroy( &destination );

    if( rank == 0 )
    {
        const auto manifest = root_manifest( *checkpoint.mesh().moab() );
        check( manifest.size() >= 4 && manifest[0] == 6 && manifest[1] == 0 && manifest[2] == 0 && manifest[3] == 0,
               "owner-only gather keeps schema v6" );
        std::remove( file.c_str() );
    }
}

void asymmetric_checkpoint_filename()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 0, 0 );
    KARMASystem system( mesh, "asymmetric" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.init();
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).write_checkpoint( rank == 0 ? "rank-zero.h5m" : "rank-one.h5m",
                                                     system, system.solution() );
           } ),
           "asymmetric checkpoint filenames fail collectively" );
}

void asymmetric_checkpoint_system_name()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    const std::string file = "asymmetric-system-name.h5m";
    const std::string sentinel = "unchanged checkpoint output";
    if( rank == 0 )
    {
        std::ofstream output( file, std::ios::binary | std::ios::trunc );
        output.write( sentinel.data(), static_cast< std::streamsize >( sentinel.size() ) );
    }
    MPI_Barrier( PETSC_COMM_WORLD );

    KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 0, 0 );
    KARMASystem system( mesh, rank == 0 ? "rank-zero-system" : "rank-one-system" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.init();
    check( throws_invalid( [&] { KARMAMoabIO( mesh ).write_checkpoint( file, system, system.solution() ); } ),
           "asymmetric checkpoint system names fail collectively" );
    MPI_Barrier( PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        std::ifstream input( file, std::ios::binary );
        const std::string bytes{ std::istreambuf_iterator< char >( input ), std::istreambuf_iterator< char >() };
        check( bytes == sentinel, "asymmetric checkpoint system names leave output unchanged" );
        std::filesystem::remove( file );
    }
}

void asymmetric_restore_vector_validation( std::string_view failure )
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    const std::string file = "karma_asymmetric_restore.h5m";
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 2, 0 );
    KARMASystem system( mesh, "asymmetric-restore" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    system.init();
    fill_solution( system.solution() );
    KARMAMoabIO io( mesh );
    io.write_checkpoint( file, system, system.solution() );

    Vec valid = system.create_vector();
    petsc_check( VecSet( valid, 73.0 ), "set valid restore sentinel" );
    Vec invalid = nullptr;
    if( failure == "size" )
        petsc_check( VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs() + ( rank == 1 ? 1 : 0 ),
                                  system.n_dofs() + 1, &invalid ),
                     "create wrong-size restore vector" );
    else if( failure == "layout" )
        petsc_check( VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs(), system.n_dofs(), &invalid ),
                     "create wrong-layout restore vector" );
    Vec destination = rank == 1 && invalid ? invalid : valid;
    if( failure == "state" && rank == 1 )
        petsc_check( VecLockReadPush( destination ), "lock restore destination" );
    if( failure == "nonfinite" && rank == 1 )
    {
        PetscScalar* values = nullptr;
        petsc_check( VecGetArray( destination, &values ), "acquire nonfinite restore destination" );
        values[0] = static_cast< PetscScalar >( std::numeric_limits< PetscReal >::quiet_NaN() );
        petsc_check( VecRestoreArray( destination, &values ), "release nonfinite restore destination" );
    }

    PetscInt local_size = 0;
    petsc_check( VecGetLocalSize( destination, &local_size ), "read restore destination local size" );
    const PetscScalar* before_data = nullptr;
    petsc_check( VecGetArrayRead( destination, &before_data ), "read restore destination before failure" );
    std::vector< unsigned char > before( static_cast< std::size_t >( local_size ) * sizeof( PetscScalar ) );
    std::memcpy( before.data(), before_data, before.size() );
    petsc_check( VecRestoreArrayRead( destination, &before_data ), "release restore destination before failure" );

    check( throws_invalid( [&] { io.restore_solution( system, destination ); } ),
           "asymmetric invalid restore vector fails collectively" );

    const PetscScalar* after_data = nullptr;
    petsc_check( VecGetArrayRead( destination, &after_data ), "read restore destination after failure" );
    check( std::memcmp( before.data(), after_data, before.size() ) == 0,
           "asymmetric invalid restore leaves every destination unchanged" );
    petsc_check( VecRestoreArrayRead( destination, &after_data ), "release restore destination after failure" );
    if( failure == "state" && rank == 1 )
        petsc_check( VecLockReadPop( destination ), "unlock restore destination" );
    if( failure == "state" )
    {
        auto checkpoint = KARMAMoabIO::load_checkpoint( file, PETSC_COMM_WORLD );
        check( !throws_invalid( [&] {
                   KARMAMoabIO( checkpoint.mesh() ).restore_solution( checkpoint.system(), destination );
               } ),
               "restore remains retryable after asymmetric destination state failure" );
        compare_vectors( destination, system.solution(),
                         "restore retry succeeds after asymmetric destination state failure" );
    }
    VecDestroy( &invalid );
    VecDestroy( &valid );
    if( rank == 0 ) std::remove( file.c_str() );
}

void invalid_checkpoint_temporary_path()
{
    using namespace KARMA;
    KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 0, 0 );
    KARMASystem system( mesh, "invalid-temporary-path" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.init();
    check( throws_invalid( [&] {
               KARMAMoabIO( mesh ).write_checkpoint( "/dev/null/checkpoint.h5m", system, system.solution() );
           } ),
           "invalid rank-zero temporary path fails collectively before name broadcast" );
}

void check_conventional_ids( KARMA::KARMAMesh& mesh, int dimension )
{
    moab::Range entities;
    moab_check( mesh.moab()->get_entities_by_dimension( mesh.root_set(), dimension, entities ),
                "get entities for GLOBAL_ID validation" );
    moab::Range owned = entities;
    moab_check( mesh.parallel_comm()->filter_pstatus( owned, PSTATUS_NOT_OWNED, PSTATUS_NOT ),
                "filter owned GLOBAL_ID entities" );

    moab::Tag global_id = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ), "find assigned GLOBAL_ID" );
    std::vector< int > local_ids( owned.size() );
    if( !owned.empty() )
        moab_check( mesh.moab()->tag_get_data( global_id, owned, local_ids.data() ), "read owned GLOBAL_IDs" );
    check( std::all_of( local_ids.begin(), local_ids.end(), []( int id ) { return id > 0; } ),
           "owned GLOBAL_IDs are positive" );

    int rank_count = 1;
    MPI_Comm_size( PETSC_COMM_WORLD, &rank_count );
    const int local_count = static_cast< int >( local_ids.size() );
    std::vector< int > counts( static_cast< std::size_t >( rank_count ) );
    MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD );
    std::vector< int > displacements( counts.size() );
    for( std::size_t i = 1; i < counts.size(); ++i ) displacements[i] = displacements[i - 1] + counts[i - 1];
    std::vector< int > all_ids( static_cast< std::size_t >( std::accumulate( counts.begin(), counts.end(), 0 ) ) );
    MPI_Allgatherv( local_ids.data(), local_count, MPI_INT, all_ids.data(), counts.data(), displacements.data(), MPI_INT,
                    PETSC_COMM_WORLD );
    std::sort( all_ids.begin(), all_ids.end() );
    check( std::adjacent_find( all_ids.begin(), all_ids.end() ) == all_ids.end(),
           "owned GLOBAL_IDs are globally unique within their dimension" );

    {
        std::vector< int > originals( entities.size() );
        if( !entities.empty() )
            moab_check( mesh.moab()->tag_get_data( global_id, entities, originals.data() ),
                        "read visible GLOBAL_ID copies" );
        const int unset = 0;
        moab::Tag validation = nullptr;
        moab_check( mesh.moab()->tag_get_handle( "__karma_test_global_id_validation__", 1,
                                                 moab::MB_TYPE_INTEGER, validation,
                                                 moab::MB_TAG_DENSE | moab::MB_TAG_CREAT, &unset ),
                    "create test GLOBAL_ID validation tag" );
        if( !entities.empty() )
            moab_check( mesh.moab()->tag_set_data( validation, entities, originals.data() ),
                        "copy visible GLOBAL_ID values" );
        moab_check( mesh.parallel_comm()->exchange_tags( validation, entities ),
                    "exchange owner GLOBAL_ID values" );
        std::vector< int > owner_values( entities.size() );
        if( !entities.empty() )
            moab_check( mesh.moab()->tag_get_data( validation, entities, owner_values.data() ),
                        "read owner GLOBAL_ID values" );
        check( originals == owner_values, "shared and ghost copies agree on GLOBAL_ID" );
        moab_check( mesh.moab()->tag_delete( validation ), "delete test GLOBAL_ID validation tag" );
    }
}

void asymmetric_replicated_global_ids()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 2, 0 );
    moab::Tag global_id = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id,
                                             moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ),
                "create replicated GLOBAL_ID" );
    std::vector< int > ids( mesh.vertices().size(), rank == 0 ? 0 : 17 );
    moab_check( mesh.moab()->tag_set_data( global_id, mesh.vertices(), ids.data() ),
                "set asymmetric replicated GLOBAL_ID" );
    mesh.ensure_global_ids();
    for( const moab::Range* entities : { &mesh.vertices(), &mesh.elements() } )
    {
        ids.resize( entities->size() );
        moab_check( mesh.moab()->tag_get_data( global_id, *entities, ids.data() ),
                    "read repaired replicated GLOBAL_ID" );
        std::vector< int > expected( ids.size() );
        std::iota( expected.begin(), expected.end(), 1 );
        check( ids == expected, "replicated ranks receive identical deterministic GLOBAL_IDs" );
    }
}

void parallel_partition_roundtrip( const std::string& input, const std::string& output )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO io( mesh );
    io.read_mesh( input, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    check( !mesh.owned_elements().empty(), "partitioned load gives every rank owned elements" );
    check( mesh.owned_elements().size() < 16, "partitioned load does not replicate all elements" );
    unsigned long long local_owned = mesh.owned_elements().size();
    unsigned long long global_owned = 0;
    MPI_Allreduce( &local_owned, &global_owned, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, PETSC_COMM_WORLD );
    check( global_owned == 16, "partitioned load owns every element exactly once" );
    moab::Tag global_id = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ), "find partition GLOBAL_ID" );
    std::vector< int > local_ids( mesh.owned_elements().size() );
    moab_check( mesh.moab()->tag_get_data( global_id, mesh.owned_elements(), local_ids.data() ),
                "read owned element GLOBAL_IDs" );
    int rank_count = 1;
    MPI_Comm_size( PETSC_COMM_WORLD, &rank_count );
    const int local_count = static_cast< int >( local_ids.size() );
    std::vector< int > counts( static_cast< std::size_t >( rank_count ) );
    MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD );
    std::vector< int > displacements( counts.size() );
    for( std::size_t i = 1; i < counts.size(); ++i ) displacements[i] = displacements[i - 1] + counts[i - 1];
    std::vector< int > owned_ids( static_cast< std::size_t >( std::accumulate( counts.begin(), counts.end(), 0 ) ) );
    MPI_Allgatherv( local_ids.data(), local_count, MPI_INT, owned_ids.data(), counts.data(), displacements.data(),
                    MPI_INT, PETSC_COMM_WORLD );
    std::sort( owned_ids.begin(), owned_ids.end() );
    check( owned_ids.size() == 16 && std::adjacent_find( owned_ids.begin(), owned_ids.end() ) == owned_ids.end(),
           "partitioned load has unique element ownership" );

    moab::Range local_edges;
    moab_check( mesh.moab()->get_adjacencies( mesh.elements(), 1, true, local_edges, moab::Interface::UNION ),
                "materialize partitioned checkpoint edges" );
    moab::Range local_elements = mesh.elements();
    moab_check( mesh.parallel_comm()->resolve_shared_ents( mesh.root_set(), local_elements, 2, 1, nullptr,
                                                           &global_id ),
                "resolve partitioned checkpoint edges" );
    moab_check( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ),
                "add ghost elements and edges for GLOBAL_ID validation" );
    mesh.update();
    mesh.assign_ids();
    for( const auto element : mesh.elements() )
        for( unsigned side = 0; side < 4; ++side )
        {
            moab::EntityHandle edge = 0;
            moab_check( mesh.moab()->side_element( element, 1, side, edge ),
                        "query partitioned checkpoint edge" );
            check( edge != 0, "every partitioned Q2 side has an explicit edge" );
            int owner = -1;
            moab_check( mesh.parallel_comm()->get_owner( edge, owner ),
                        "query partitioned checkpoint edge owner" );
            check( owner >= 0 && owner < rank_count,
                   "every partitioned Q2 edge has a valid owner" );
        }
    moab::Range ghost_elements;
    moab_check( mesh.parallel_comm()->get_pstatus_entities( 2, PSTATUS_GHOST, ghost_elements ),
                "query ghost elements" );
    check( !ghost_elements.empty(), "partition fixture contains ghost elements" );

    std::vector< GlobalId > private_vertex_ids;
    std::vector< GlobalId > private_element_ids;
    std::vector< LocalId > private_vertex_local_ids;
    std::vector< LocalId > private_element_local_ids;
    mesh.global_ids( mesh.vertices(), private_vertex_ids );
    mesh.global_ids( mesh.elements(), private_element_ids );
    mesh.local_ids( mesh.vertices(), private_vertex_local_ids );
    mesh.local_ids( mesh.elements(), private_element_local_ids );

    moab::Range nonowned_vertices = mesh.vertices();
    moab_check( mesh.parallel_comm()->filter_pstatus( nonowned_vertices, PSTATUS_NOT_OWNED, PSTATUS_AND ),
                "find nonowned vertices to corrupt" );
    check( !nonowned_vertices.empty(), "partition fixture contains nonowned vertices" );
    if( !nonowned_vertices.empty() )
    {
        const moab::EntityHandle corrupt = *nonowned_vertices.begin();
        const int invalid = 0;
        moab_check( mesh.moab()->tag_set_data( global_id, &corrupt, 1, &invalid ),
                    "corrupt nonowner GLOBAL_ID copy" );
    }

    KARMASystem identity_system( mesh, "global-id-identity" );
    identity_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    identity_system.init();
    check_conventional_ids( mesh, 0 );
    check_conventional_ids( mesh, 2 );

    moab::Range owned_vertices = mesh.vertices();
    moab_check( mesh.parallel_comm()->filter_pstatus( owned_vertices, PSTATUS_NOT_OWNED, PSTATUS_NOT ),
                "find owned vertices to invalidate" );
    check( !owned_vertices.empty(), "partition fixture gives every rank owned vertices" );
    std::vector< int > nonpositive_vertex_ids( owned_vertices.size(), 0 );
    if( !owned_vertices.empty() )
        moab_check( mesh.moab()->tag_set_data( global_id, owned_vertices, nonpositive_vertex_ids.data() ),
                    "set non-positive owned vertex GLOBAL_IDs" );
    mesh.ensure_global_ids();
    check_conventional_ids( mesh, 0 );
    check_conventional_ids( mesh, 2 );

    std::vector< int > duplicate_element_ids( mesh.owned_elements().size(), 1 );
    if( !mesh.owned_elements().empty() )
        moab_check( mesh.moab()->tag_set_data( global_id, mesh.owned_elements(), duplicate_element_ids.data() ),
                    "set duplicate owned element GLOBAL_IDs" );
    mesh.ensure_global_ids();
    check_conventional_ids( mesh, 0 );
    check_conventional_ids( mesh, 2 );

    const std::vector< moab::EntityHandle > owned_elements( mesh.owned_elements().begin(), mesh.owned_elements().end() );
    check( identity_system.dof_map().assembly_elements() == owned_elements,
           "partitioned assembly elements are exactly MOAB-owned elements" );
    for( const auto element : mesh.elements() )
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        moab_check( mesh.moab()->get_connectivity( element, connectivity, count ),
                    "read partition element connectivity for vertex_dof" );
        std::vector< PetscInt > indices;
        identity_system.dof_map().dof_indices( element, indices, 0 );
        for( int vertex = 0; vertex < count; ++vertex )
            check( identity_system.dof_map().vertex_dof( connectivity[vertex], 0 ) == indices[static_cast< std::size_t >( vertex )],
                   "partitioned vertex_dof uses authoritative GLOBAL_ID identity" );
    }
    check_conventional_ids( mesh, 0 );
    check_conventional_ids( mesh, 2 );

    std::vector< GlobalId > private_vertex_ids_after;
    std::vector< GlobalId > private_element_ids_after;
    std::vector< LocalId > private_vertex_local_ids_after;
    std::vector< LocalId > private_element_local_ids_after;
    mesh.global_ids( mesh.vertices(), private_vertex_ids_after );
    mesh.global_ids( mesh.elements(), private_element_ids_after );
    mesh.local_ids( mesh.vertices(), private_vertex_local_ids_after );
    mesh.local_ids( mesh.elements(), private_element_local_ids_after );
    check( private_vertex_ids_after == private_vertex_ids && private_element_ids_after == private_element_ids,
           "GLOBAL_ID repair leaves private global IDs unchanged" );
    check( private_vertex_local_ids_after == private_vertex_local_ids &&
               private_element_local_ids_after == private_element_local_ids,
           "GLOBAL_ID repair leaves private local IDs unchanged" );

    moab::Range shared;
    moab_check( mesh.parallel_comm()->get_pstatus_entities( 0, PSTATUS_SHARED, shared ), "query shared vertices" );
    check( !shared.empty(), "partitioned load resolves shared vertices" );
    moab::Tag field = nullptr;
    moab_check( mesh.moab()->tag_get_handle( "KARMA_PARTITION_TEST_FIELD", field ), "find partition fixture field" );
    std::vector< double > values( mesh.owned_elements().size() );
    moab_check( mesh.moab()->tag_get_data( field, mesh.owned_elements(), values.data() ), "read partition fixture field" );
    std::size_t offset = 0;
    for( const auto element : mesh.owned_elements() )
    {
        const auto center = mesh.centroid( element );
        check( values[offset++] == center( 0 ) + 2 * center( 1 ),
               "partitioned load preserves exact field values" );
    }

    KARMASystem checkpoint_system( mesh, "partition-checkpoint" );
    checkpoint_system.add_variable( { "continuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 } );
    checkpoint_system.add_variable( { "discontinuous", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::dG, 1 } );
    checkpoint_system.add_variable( { "spectral", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 } );
    checkpoint_system.init();
    fill_semantic_solution( checkpoint_system, checkpoint_system.solution() );
    const DatabaseInventory checkpoint_source = collective_database_inventory( *mesh.moab() );
    const std::string& checkpoint_output = output;
    const std::filesystem::path failed_checkpoint = output + ".failed.h5m";
    std::error_code ignored;
    std::filesystem::remove_all( failed_checkpoint, ignored );
    std::filesystem::create_directory( failed_checkpoint );
    check( throws_invalid( [&] {
               io.write_checkpoint( failed_checkpoint.string(), checkpoint_system, checkpoint_system.solution(),
                                    "PARALLEL=WRITE_PART" );
           } ),
           "partitioned checkpoint publication failure is reported" );
    check( collective_database_inventory( *mesh.moab() ) == checkpoint_source,
           "failed partitioned checkpoint publication leaves every rank's source digest unchanged" );
    std::filesystem::remove_all( failed_checkpoint, ignored );
    io.write_checkpoint( checkpoint_output, checkpoint_system, checkpoint_system.solution(), "PARALLEL=WRITE_PART" );
    check( collective_database_inventory( *mesh.moab() ) == checkpoint_source,
           "successful partitioned checkpoint publication leaves every rank's source digest unchanged" );
    moab::EntityHandle mutated_entity = 0;
    double original_value = 0.0;
    if( !mesh.owned_elements().empty() )
    {
        mutated_entity = mesh.owned_elements().front();
        moab_check( mesh.moab()->tag_get_data( field, &mutated_entity, 1, &original_value ),
                    "read partition source field for inventory mutation" );
        const double mutated_value = original_value + 1.0;
        moab_check( mesh.moab()->tag_set_data( field, &mutated_entity, 1, &mutated_value ),
                    "mutate partition source field for inventory test" );
    }
    check( collective_database_inventory( *mesh.moab() ) != checkpoint_source,
           "partitioned source inventory detects an existing tag payload mutation" );
    if( mutated_entity != 0 )
        moab_check( mesh.moab()->tag_set_data( field, &mutated_entity, 1, &original_value ),
                    "restore partition source field after inventory mutation" );
    check( collective_database_inventory( *mesh.moab() ) == checkpoint_source,
           "partitioned source inventory returns to baseline after restoring tag payload" );
    auto checkpoint = KARMAMoabIO::load_checkpoint( checkpoint_output, PETSC_COMM_WORLD );
    fill_semantic_solution( checkpoint.system(), checkpoint.system().temporary_vector() );
    compare_vectors( checkpoint.system().solution(), checkpoint.system().temporary_vector(),
                     "partitioned checkpoint restores exact distributed values" );
    MPI_Barrier( PETSC_COMM_WORLD );
}
}  // namespace

int main( int argc, char** argv )
{
    const PetscErrorCode error = PetscInitialize( &argc, &argv, nullptr, nullptr );
    if( error != PETSC_SUCCESS ) return error;

    try
    {
        const auto mpi_layout = KARMA::detail::checked_mpi_displacements(
            std::array{ std::numeric_limits< int >::max() - 1, 1 }, 1 );
        check( mpi_layout.displacements == std::vector< int >( { 0, std::numeric_limits< int >::max() - 1 } ) &&
                   mpi_layout.total_count == static_cast< std::size_t >( std::numeric_limits< int >::max() ),
               "checkpoint gather layout accepts an exact INT_MAX total" );
        check( throws_invalid( [] {
                   (void)KARMA::detail::checked_mpi_displacements(
                       std::array{ std::numeric_limits< int >::max(), 1 }, 1 );
               } ),
               "checkpoint gather layout rejects a total above INT_MAX without allocation" );
        check( throws_invalid_containing(
                   [] { KARMA::detail::sync_checkpoint_file( "missing-checkpoint-sync-file" ); }, "open" ),
               "checkpoint file sync rejects a missing file" );
        check( throws_invalid_containing(
                   [] { KARMA::detail::sync_checkpoint_directory( "missing-checkpoint-sync-directory" ); }, "open" ),
               "checkpoint directory sync rejects a missing directory" );
        check( throws_invalid_containing(
                   [] { KARMA::detail::sync_checkpoint_directory( __FILE__ ); }, "directory" ),
               "checkpoint directory sync rejects a regular file" );
        check( throws_invalid_containing(
                   [] { KARMA::detail::sync_checkpoint_file( "." ); }, "regular file" ),
               "checkpoint file sync rejects a directory" );
        int cleanup_rank = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &cleanup_rank );
        if( cleanup_rank == 0 )
        {
            const std::filesystem::path cleanup_directory = "karma_checkpoint_cleanup_test";
            const std::filesystem::path cleanup_snapshot = cleanup_directory / "snapshot.h5m";
            const std::filesystem::path cleanup_staged = cleanup_directory / "staged.h5m";
            std::error_code cleanup_error;
            std::filesystem::remove_all( cleanup_directory, cleanup_error );
            std::filesystem::create_directories( cleanup_snapshot );
            check( throws_invalid_containing(
                       [&] {
                           KARMA::detail::remove_checkpoint_temporaries(
                               cleanup_directory.string(), cleanup_snapshot.string(), cleanup_staged.string() );
                       },
                       "remove checkpoint temporary file" ),
                   "checkpoint cleanup reports a temporary file with the wrong type" );
            std::filesystem::remove_all( cleanup_directory, cleanup_error );
            std::filesystem::create_directories( cleanup_directory );
            std::ofstream( cleanup_directory / "unexpected" ) << "occupied";
            check( throws_invalid_containing(
                       [&] {
                           KARMA::detail::remove_checkpoint_temporaries(
                               cleanup_directory.string(), cleanup_snapshot.string(), cleanup_staged.string() );
                       },
                       "remove checkpoint temporary reservation directory" ),
                   "checkpoint cleanup reports a nonempty reservation directory" );
            std::filesystem::remove_all( cleanup_directory, cleanup_error );
            check( !throws_invalid( [&] {
                       KARMA::detail::remove_checkpoint_temporaries(
                           cleanup_directory.string(), cleanup_snapshot.string(), cleanup_staged.string() );
                   } ),
                   "checkpoint cleanup is idempotent for missing paths" );
        }
        MPI_Barrier( PETSC_COMM_WORLD );
        check( throws_invalid_containing(
                   [] { KARMA::detail::check_checkpoint_barrier( MPI_ERR_OTHER, "test barrier" ); },
                   "test barrier failed with MPI error" ),
               "failed checkpoint barrier reports a direct local error" );
        check( throws_invalid_containing(
                   [] { KARMA::detail::throw_checkpoint_publication_durability_error( "injected sync failure" ); },
                   "destination has already been atomically replaced but crash durability could not be confirmed: "
                   "injected sync failure" ),
               "post-rename sync failure reports irreversible publication state" );
        using KARMA::detail::CheckpointPublicationState;
        check( KARMA::detail::checkpoint_result_collective_failed( CheckpointPublicationState::Published ) ==
                   CheckpointPublicationState::Published,
               "rank zero retains published state when rename-result propagation fails" );
        check( KARMA::detail::checkpoint_result_collective_failed( CheckpointPublicationState::Unpublished ) ==
                   CheckpointPublicationState::Indeterminate,
               "other ranks transition to indeterminate publication when rename-result propagation fails" );
        check( KARMA::detail::checkpoint_publication_result_error(
                   CheckpointPublicationState::Published, "injected collective failure" ) ==
                   "checkpoint destination was published, but collectively propagating the rename result failed: "
                   "injected collective failure",
               "rank zero publication-result failure reports the destination as published" );
        check( KARMA::detail::checkpoint_publication_result_error(
                   CheckpointPublicationState::Indeterminate, "injected collective failure" ) ==
                   "checkpoint publication is indeterminate; the destination may have been published because collectively "
                   "propagating the rename result failed: injected collective failure",
               "peer publication-result failure reports indeterminate publication" );
        check( KARMA::detail::checkpoint_values_are_finite( nullptr, 0 ),
               "zero-length checkpoint finite check accepts a null array" );
        KARMA::detail::copy_checkpoint_values( nullptr, nullptr, 0 );
        canonical_scalar_encoding();
        variable_typed_tag_inventory_tracks_trailing_values();
        if( argc == 3 && std::string_view( argv[1] ) == "--write-partition-fixture" )
            write_partition_fixture( argv[2] );
        else if( argc == 3 && std::string_view( argv[1] ) == "--verify-partition-file" )
            verify_partition_file( argv[2] );
        else if( argc == 2 && std::string_view( argv[1] ) == "--replicated-parallel-write" )
            replicated_parallel_write_rejected();
        else if( argc == 2 && std::string_view( argv[1] ) == "--owner-only-checkpoint-gather" )
            owner_only_checkpoint_gather();
        else if( argc == 2 && std::string_view( argv[1] ) == "--asymmetric-read-filename" )
            asymmetric_read_filename();
        else if( argc == 2 && std::string_view( argv[1] ) == "--asymmetric-checkpoint-filename" )
            asymmetric_checkpoint_filename();
        else if( argc == 2 && std::string_view( argv[1] ) == "--asymmetric-checkpoint-system-name" )
            asymmetric_checkpoint_system_name();
        else if( argc == 2 && std::string_view( argv[1] ) == "--invalid-checkpoint-temporary-path" )
            invalid_checkpoint_temporary_path();
        else if( argc == 2 && std::string_view( argv[1] ).starts_with( "--asymmetric-restore-vector-" ) )
            asymmetric_restore_vector_validation(
                std::string_view( argv[1] ).substr( std::string_view( "--asymmetric-restore-vector-" ).size() ) );
        else if( argc == 2 && std::string_view( argv[1] ) == "--asymmetric-replicated-global-ids" )
            asymmetric_replicated_global_ids();
        else if( argc == 2 && std::string_view( argv[1] ) == "--repeated-checkpoint-threshold" )
            repeated_checkpoint_replaces_sparse_storage_with_dense();
        else if( argc == 4 && std::string_view( argv[1] ) == "--parallel-partition-roundtrip" )
            parallel_partition_roundtrip( argv[2], argv[3] );
        else
            run_exact_tests();
    }
    catch( const std::exception& exception )
    {
        std::cerr << "FAIL: unexpected exception: " << exception.what() << '\n';
        ++failures;
    }

    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
