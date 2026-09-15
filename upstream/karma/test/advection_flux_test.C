#include "advection_reaction_assembly.h"
#include "advection_reaction_system.h"
#include "advection_reaction_mass_operator.h"
#include "advection_reaction_preconditioner_operator.h"
#include "advection_reaction_system_operator.h"
#include "karma_fe_base.h"
#include "karma_moab_io.h"
#include "karma_system.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef KARMA_DG_FACE_MUTATION_TEST
namespace KARMA
{
void assemble_advection_reaction_with_duplicate_interior_faces(
    const KARMASystem&, Vec, Vec, Mat, const AdvectionReactionCoefficients&, int );
std::size_t duplicate_interior_face_traversal_count();
}
#endif

namespace KARMA::detail
{
void set_advection_local_element_map_fault( bool enabled );
}

namespace
{
using AdvectionAssemblyFunction = void ( * )( const KARMA::KARMASystem&, Vec, Vec, Mat,
                                               const KARMA::AdvectionReactionCoefficients&, int );
static_assert( std::is_same_v< decltype( static_cast< AdvectionAssemblyFunction >(
                                  &KARMA::assemble_advection_reaction ) ),
                               AdvectionAssemblyFunction > );

int failures = 0;
void check( bool condition, const char* message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

bool throws_with( auto&& function, std::string_view text )
{
    try
    {
        function();
    }
    catch( const std::logic_error& error )
    {
        return std::string_view( error.what() ).find( text ) != std::string_view::npos;
    }
    return false;
}

void set_material( KARMA::KARMAMesh& mesh, moab::EntityHandle element, int material )
{
    moab::Tag tag = nullptr;
    moab::EntityHandle set = 0;
    check( mesh.moab()->create_meshset( moab::MESHSET_SET, set ) == moab::MB_SUCCESS, "create material set" );
    check( mesh.moab()->add_entities( set, &element, 1 ) == moab::MB_SUCCESS, "add element to material set" );
    check( mesh.moab()->tag_get_handle( "MATERIAL_SET", 1, moab::MB_TYPE_INTEGER, tag,
                                        moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT ) == moab::MB_SUCCESS,
           "get material tag" );
    check( mesh.moab()->tag_set_data( tag, &set, 1, &material ) == moab::MB_SUCCESS, "set material id" );
}

Real vector_entry( Vec vector, PetscInt index )
{
    PetscScalar value = 0;
    VecGetValues( vector, 1, &index, &value );
    return PetscRealPart( value );
}

Real matrix_entry( Mat matrix, PetscInt row, PetscInt column )
{
    PetscScalar value = 0;
    MatGetValues( matrix, 1, &row, 1, &column, &value );
    return PetscRealPart( value );
}

void require( bool condition, const char* message )
{
    if( !condition ) throw std::runtime_error( message );
}

struct AdvectionReference
{
    std::size_t interior_faces = 0;
    std::vector< KARMA::DofKey > keys;
    std::vector< PetscReal > residual;
    std::vector< PetscReal > matrix;
};

std::map< KARMA::DofKey, PetscInt > visible_dof_indices( const KARMA::KARMASystem& system )
{
    std::map< KARMA::DofKey, PetscInt > result;
    for( const auto element : system.mesh().elements() )
    {
        std::vector< PetscInt > indices;
        system.dof_map().dof_indices( element, indices );
        const auto& keys = system.dof_map().dof_keys( element );
        require( keys.size() == indices.size(), "advection keys and indices differ in size" );
        for( std::size_t i = 0; i < keys.size(); ++i )
        {
            const auto [found, inserted] = result.emplace( keys[i], indices[i] );
            require( inserted || found->second == indices[i], "stable advection key has inconsistent index" );
        }
    }
    return result;
}

std::map< KARMA::DofKey, PetscInt > global_dof_indices( const KARMA::KARMASystem& system )
{
    const auto visible = visible_dof_indices( system );
    std::vector< std::uint64_t > local_words;
    local_words.reserve( visible.size() * ( KARMA::DofKey::width + 1 ) );
    for( const auto& [key, index] : visible )
    {
        local_words.insert( local_words.end(), key.begin(), key.end() );
        local_words.push_back( static_cast< std::uint64_t >( index ) );
    }
    int ranks = 1;
    require( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ) == MPI_SUCCESS, "query advection rank count" );
    const int local_count = static_cast< int >( local_words.size() );
    std::vector< int > counts( static_cast< std::size_t >( ranks ) ), offsets( static_cast< std::size_t >( ranks ) );
    require( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather advection key counts" );
    for( int rank = 1; rank < ranks; ++rank ) offsets[rank] = offsets[rank - 1] + counts[rank - 1];
    std::vector< std::uint64_t > gathered( static_cast< std::size_t >( offsets.back() + counts.back() ) );
    require( MPI_Allgatherv( local_words.empty() ? nullptr : local_words.data(), local_count, MPI_UINT64_T,
                             gathered.empty() ? nullptr : gathered.data(), counts.data(), offsets.data(), MPI_UINT64_T,
                             PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather advection keys and indices" );
    std::map< KARMA::DofKey, PetscInt > result;
    constexpr std::size_t record_width = KARMA::DofKey::width + 1;
    require( gathered.size() % record_width == 0, "gathered advection key records are complete" );
    for( std::size_t offset = 0; offset < gathered.size(); offset += record_width )
    {
        KARMA::DofKey key;
        std::copy_n( gathered.begin() + static_cast< std::ptrdiff_t >( offset ), KARMA::DofKey::width, key.begin() );
        const auto index = static_cast< PetscInt >( gathered[offset + KARMA::DofKey::width] );
        const auto [found, inserted] = result.emplace( key, index );
        require( inserted || found->second == index, "partitioned stable advection key maps to one index" );
    }
    return result;
}

std::size_t canonical_interior_face_count( const KARMA::KARMASystem& system )
{
    std::set< std::pair< std::uint64_t, std::uint64_t > > faces;
    for( const auto element : system.dof_map().assembly_elements() )
        for( unsigned side = 0; side < system.mesh().n_sides( element ); ++side )
            if( const auto neighbor = system.mesh().neighbor( element, side ); neighbor )
            {
                const auto first = system.dof_map().canonical_element_ordinal( element );
                const auto second = system.dof_map().canonical_element_ordinal( *neighbor );
                require( first != second, "interior advection face cannot connect one global element" );
                faces.emplace( std::min( first, second ), std::max( first, second ) );
            }
    return faces.size();
}

std::size_t global_canonical_interior_face_count( const KARMA::KARMASystem& system )
{
    std::set< std::pair< std::uint64_t, std::uint64_t > > local_faces;
    for( const auto element : system.dof_map().assembly_elements() )
        for( unsigned side = 0; side < system.mesh().n_sides( element ); ++side )
            if( const auto neighbor = system.mesh().neighbor( element, side ); neighbor )
            {
                const auto first = system.dof_map().canonical_element_ordinal( element );
                const auto second = system.dof_map().canonical_element_ordinal( *neighbor );
                local_faces.emplace( std::min( first, second ), std::max( first, second ) );
            }
    std::vector< std::uint64_t > local_words;
    local_words.reserve( 2 * local_faces.size() );
    for( const auto& [first, second] : local_faces ) { local_words.push_back( first ); local_words.push_back( second ); }
    int ranks = 1;
    require( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ) == MPI_SUCCESS, "query advection face rank count" );
    const int local_count = static_cast< int >( local_words.size() );
    std::vector< int > counts( static_cast< std::size_t >( ranks ) ), offsets( static_cast< std::size_t >( ranks ) );
    require( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather canonical advection face counts" );
    for( int rank = 1; rank < ranks; ++rank ) offsets[rank] = offsets[rank - 1] + counts[rank - 1];
    std::vector< std::uint64_t > gathered( static_cast< std::size_t >( offsets.back() + counts.back() ) );
    require( MPI_Allgatherv( local_words.empty() ? nullptr : local_words.data(), local_count, MPI_UINT64_T,
                             gathered.empty() ? nullptr : gathered.data(), counts.data(), offsets.data(), MPI_UINT64_T,
                             PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather canonical advection face pairs" );
    require( gathered.size() % 2 == 0, "gathered canonical advection face pairs are complete" );
    std::set< std::pair< std::uint64_t, std::uint64_t > > global_faces;
    for( std::size_t i = 0; i < gathered.size(); i += 2 ) global_faces.emplace( gathered[i], gathered[i + 1] );
    return global_faces.size();
}

KARMA::AdvectionReactionCoefficients partitioned_advection_coefficients()
{
    KARMA::AdvectionReactionCoefficients coefficients;
    coefficients.velocity = []( const KARMA::Point& point, int ) -> KARMA::Vector3 {
        if( point( 0 ) > 1.e-12 && point( 0 ) < 1 - 1.e-12 && point( 1 ) > 1.e-12 && point( 1 ) < 1 - 1.e-12 )
            return KARMA::Vector3( 1, 0.375, 0 );
        return KARMA::Vector3::Zero();
    };
    coefficients.inflow = []( const KARMA::Point&, int ) { return 0.0; };
    return coefficients;
}

PetscScalar stable_advection_value( const KARMA::DofKey& key )
{
    std::uint64_t value = 0;
    for( const auto word : key ) value = value * 131 + word;
    return static_cast< PetscScalar >( value % 17 + 1 );
}

AdvectionReference read_advection_reference( const char* filename )
{
    std::ifstream input( filename );
    require( input.good(), "open advection reference" );
    AdvectionReference reference;
    std::size_t dof_count = 0;
    require( static_cast< bool >( input >> reference.interior_faces >> dof_count ), "read advection reference header" );
    reference.keys.resize( dof_count );
    for( auto& key : reference.keys )
        for( auto& word : key.words ) require( static_cast< bool >( input >> word ), "read advection DoF key" );
    reference.residual.resize( dof_count );
    for( auto& value : reference.residual ) require( static_cast< bool >( input >> value ), "read advection residual" );
    reference.matrix.resize( dof_count * dof_count );
    for( auto& value : reference.matrix ) require( static_cast< bool >( input >> value ), "read advection matrix" );
    return reference;
}

void initialize_advection_with_distinct_ghost_handles( const char* mesh_filename )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        mesh_filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    require( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
             "exchange distinct-handle fixture ghosts" );
    mesh.update();
    mesh.ensure_global_ids();

    std::vector< std::uint64_t > local;
    for( const auto owned : mesh.owned_elements() )
        for( unsigned side = 0; side < mesh.n_sides( owned ); ++side )
            if( const auto neighbor = mesh.neighbor( owned, side );
                neighbor && mesh.owned_elements().find( *neighbor ) == mesh.owned_elements().end() )
            {
                const auto identity = mesh.canonical_side_vertex_ids( owned, side );
                unsigned reciprocal = mesh.n_sides( *neighbor );
                for( unsigned candidate = 0; candidate < mesh.n_sides( *neighbor ); ++candidate )
                    if( mesh.canonical_side_vertex_ids( *neighbor, candidate ) == identity ) reciprocal = candidate;
                check( reciprocal < mesh.n_sides( *neighbor ),
                       "GLOBAL_ID side identity resolves reciprocal side across rank-local handles" );
                check( mesh.side_id( owned, side ).side == side,
                       "canonical face record preserves the owned side ordinal" );
                const auto handles = mesh.side_vertices( owned, side );
                local.push_back( identity.size() );
                local.insert( local.end(), identity.begin(), identity.end() );
                local.insert( local.end(), handles.begin(), handles.end() );
            }
    int ranks = 0;
    require( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ) == MPI_SUCCESS, "query distinct-handle rank count" );
    const int local_count = static_cast< int >( local.size() );
    std::vector< int > counts( static_cast< std::size_t >( ranks ) ), offsets( static_cast< std::size_t >( ranks ) );
    require( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather distinct-handle report counts" );
    for( int rank = 1; rank < ranks; ++rank ) offsets[rank] = offsets[rank - 1] + counts[rank - 1];
    std::vector< std::uint64_t > gathered( static_cast< std::size_t >( offsets.back() + counts.back() ) );
    require( MPI_Allgatherv( local.data(), local_count, MPI_UINT64_T, gathered.data(), counts.data(), offsets.data(),
                             MPI_UINT64_T, PETSC_COMM_WORLD ) == MPI_SUCCESS,
             "gather distinct-handle reports" );
    bool saw_distinct_handles = false;
    for( int first_rank = 0; first_rank < ranks; ++first_rank )
        for( int second_rank = first_rank + 1; second_rank < ranks; ++second_rank )
            for( int first = offsets[first_rank]; first < offsets[first_rank] + counts[first_rank]; )
            {
                const auto size = static_cast< int >( gathered[static_cast< std::size_t >( first )] );
                for( int second = offsets[second_rank]; second < offsets[second_rank] + counts[second_rank]; )
                {
                    const auto second_size = static_cast< int >( gathered[static_cast< std::size_t >( second )] );
                    if( size == second_size &&
                        std::equal( gathered.begin() + first + 1, gathered.begin() + first + 1 + size,
                                    gathered.begin() + second + 1 ) &&
                        !std::equal( gathered.begin() + first + 1 + size, gathered.begin() + first + 1 + 2 * size,
                                     gathered.begin() + second + 1 + size ) )
                        saw_distinct_handles = true;
                    second += 1 + 2 * second_size;
                }
                first += 1 + 2 * size;
            }
    check( saw_distinct_handles,
           "distributed copies share canonical GLOBAL_ID identity but use different rank-local handles" );

    KARMA::AdvectionReactionSystemPropertyManager properties;
    properties.register_model( 0, partitioned_advection_coefficients() );
    KARMA::AdvectionReactionSystem physics( mesh, std::move( properties ) );
    physics.init();
    Vec residual = physics.system().create_vector();
    physics.assemble_residual( physics.solution(), residual );
    VecDestroy( &residual );
    check( physics.is_initialized(), "advection lifecycle assembles with rank-distinct shared vertex handles" );
}

void initialize_partitioned_advection_nonmanifold()
{
    int rank = 0;
    require( MPI_Comm_rank( PETSC_COMM_WORLD, &rank ) == MPI_SUCCESS, "query nonmanifold fixture rank" );
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    const double rank_zero_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
    const double rank_one_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.5, 1, 0 };
    moab::Range vertices;
    require( mesh.moab()->create_vertices( rank == 0 ? rank_zero_coordinates : rank_one_coordinates,
                                            rank == 0 ? 3 : 4, vertices ) == moab::MB_SUCCESS,
             "create advection nonmanifold vertices" );
    moab::Tag global_id = nullptr;
    require( mesh.moab()->tag_get_handle( "GLOBAL_ID", 1, moab::MB_TYPE_INTEGER, global_id,
                                          moab::MB_TAG_DENSE | moab::MB_TAG_CREAT ) == moab::MB_SUCCESS,
             "create advection nonmanifold GLOBAL_ID tag" );
    const std::vector< int > ids = rank == 0 ? std::vector< int >{ 1, 2, 3 } : std::vector< int >{ 1, 2, 4, 5 };
    require( mesh.moab()->tag_set_data( global_id, vertices, ids.data() ) == moab::MB_SUCCESS,
             "assign advection nonmanifold vertex identities" );
    const std::vector< moab::EntityHandle > local_vertices( vertices.begin(), vertices.end() );
    moab::Range elements;
    int next_element_id = rank == 0 ? 10 : 11;
    const auto add_triangle = [&]( const std::array< moab::EntityHandle, 3 >& connectivity ) {
        moab::EntityHandle triangle = 0;
        require( mesh.moab()->create_element( moab::MBTRI, connectivity.data(), 3, triangle ) == moab::MB_SUCCESS,
                 "create advection nonmanifold triangle" );
        require( mesh.moab()->tag_set_data( global_id, &triangle, 1, &next_element_id ) == moab::MB_SUCCESS,
                 "assign advection nonmanifold element identity" );
        ++next_element_id;
        elements.insert( triangle );
    };
    if( rank == 0 ) add_triangle( { local_vertices[0], local_vertices[1], local_vertices[2] } );
    else
    {
        add_triangle( { local_vertices[1], local_vertices[0], local_vertices[2] } );
        add_triangle( { local_vertices[0], local_vertices[1], local_vertices[3] } );
    }
    moab::EntityHandle partition = 0;
    require( mesh.moab()->create_meshset( moab::MESHSET_SET, partition ) == moab::MB_SUCCESS &&
                 mesh.moab()->add_entities( partition, elements ) == moab::MB_SUCCESS,
             "create advection nonmanifold partition" );
    require( mesh.parallel_comm()->resolve_shared_ents( partition, 2, -1, &global_id ) == moab::MB_SUCCESS &&
                 mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 0, true ) == moab::MB_SUCCESS,
             "exchange advection nonmanifold ghost triangles" );
    mesh.update();
    require( mesh.parallel_comm()->exchange_tags( global_id, mesh.vertices() ) == moab::MB_SUCCESS,
             "exchange advection nonmanifold vertex identities" );
    mesh.assign_ids();
    KARMA::AdvectionReactionSystemPropertyManager properties;
    properties.register_model( 0, partitioned_advection_coefficients() );
    KARMA::AdvectionReactionSystem physics( mesh, std::move( properties ) );
    check( throws_with( [&] { physics.init(); }, "non-manifold side has multiple neighbors" ) &&
               !physics.is_initialized(),
           "advection lifecycle rejects distributed nonmanifold topology collectively" );
}

void initialize_partitioned_advection_without_ghosts( const char* mesh_filename )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        mesh_filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    mesh.update();
    KARMA::AdvectionReactionSystemPropertyManager properties;
    properties.register_model( 0, partitioned_advection_coefficients() );
    KARMA::AdvectionReactionSystem physics( mesh, std::move( properties ) );
    check( throws_with( [&] { physics.init(); }, "ghost" ) && !physics.is_initialized(),
           "partitioned advection lifecycle rejects a missing interface ghost collectively" );
}

void assemble_partitioned_advection( const char* mesh_filename, bool partitioned, const char* reference_filename,
                                      unsigned order, bool mutate = false )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        mesh_filename, partitioned ? "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" : "" );
    if( partitioned )
        require( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
                 "partitioned advection fixture exchanges ghost elements" );
    mesh.update();
    KARMA::AdvectionReactionSystemPropertyManager properties;
    properties.register_model( 0, partitioned_advection_coefficients() );
    KARMA::AdvectionReactionSystem::Configuration configuration;
    configuration.order = static_cast< KARMA::Order >( order );
    KARMA::AdvectionReactionSystem physics( mesh, std::move( properties ), configuration );
    physics.init();
    auto& system = physics.system();
    const auto visible = visible_dof_indices( system );
    Vec solution = physics.solution(), residual = system.create_vector(), mutant_residual = nullptr;
    Mat jacobian = system.create_matrix(), mutant_jacobian = nullptr;
    for( const auto& [key, index] : visible )
        require( VecSetValue( solution, index, stable_advection_value( key ), INSERT_VALUES ) == PETSC_SUCCESS,
                 "set advection solution by stable key" );
    require( VecAssemblyBegin( solution ) == PETSC_SUCCESS && VecAssemblyEnd( solution ) == PETSC_SUCCESS,
             "assemble advection stable-key solution" );
    physics.assemble_residual( solution, residual );
    physics.assemble_jacobian( solution, jacobian );
    if( !partitioned )
    {
        std::ofstream output( reference_filename );
        require( output.good(), "open advection reference output" );
        output << canonical_interior_face_count( system ) << ' ' << visible.size() << '\n';
        output << std::setprecision( 17 );
        for( const auto& [key, index] : visible )
        {
            (void)index;
            for( const auto word : key.words ) output << word << ' ';
            output << '\n';
        }
        for( const auto& [key, index] : visible )
        {
            (void)key;
            PetscScalar value = 0;
            require( VecGetValues( residual, 1, &index, &value ) == PETSC_SUCCESS, "read serial advection residual" );
            output << PetscRealPart( value ) << '\n';
        }
        for( const auto& [row_key, row] : visible )
            for( const auto& [column_key, column] : visible )
            {
                (void)row_key; (void)column_key;
                PetscScalar value = 0;
                require( MatGetValues( jacobian, 1, &row, 1, &column, &value ) == PETSC_SUCCESS,
                         "read serial advection matrix" );
                output << PetscRealPart( value ) << '\n';
            }
    }
    else
    {
        const AdvectionReference reference = read_advection_reference( reference_filename );
        std::map< KARMA::DofKey, PetscInt > reference_indices;
        for( std::size_t i = 0; i < reference.keys.size(); ++i )
            require( reference_indices.emplace( reference.keys[i], static_cast< PetscInt >( i ) ).second,
                     "serial advection reference keys are unique" );
        const auto partitioned_indices = global_dof_indices( system );
        require( partitioned_indices.size() == reference.keys.size(),
                 "partitioned advection exposes every serial stable DoF" );
        PetscInt first = 0, last = 0;
        require( VecGetOwnershipRange( residual, &first, &last ) == PETSC_SUCCESS, "get advection residual ownership" );
        std::vector< PetscInt > columns;
        for( const auto& key : reference.keys ) columns.push_back( partitioned_indices.at( key ) );
        std::set< PetscInt > face_dofs;
        std::set< std::pair< PetscInt, PetscInt > > face_couplings;
        for( const auto element : system.dof_map().assembly_elements() )
            for( unsigned side = 0; side < system.mesh().n_sides( element ); ++side )
                if( const auto neighbor = system.mesh().neighbor( element, side );
                    neighbor && system.dof_map().canonical_element_ordinal( element ) <
                                    system.dof_map().canonical_element_ordinal( *neighbor ) )
                {
                    std::vector< PetscInt > element_dofs, neighbor_dofs;
                    system.dof_map().dof_indices( element, element_dofs );
                    system.dof_map().dof_indices( *neighbor, neighbor_dofs );
                    face_dofs.insert( element_dofs.begin(), element_dofs.end() );
                    face_dofs.insert( neighbor_dofs.begin(), neighbor_dofs.end() );
                    for( const auto row : element_dofs )
                        for( const auto column : neighbor_dofs )
                        {
                            face_couplings.emplace( row, column );
                            face_couplings.emplace( column, row );
                        }
                }

        bool local_mutant_residual_difference = false;
        bool local_mutant_jacobian_difference = false;
        bool local_mutant_residual_reference_mismatch = false;
        bool local_mutant_jacobian_reference_mismatch = false;
        for( const auto& [key, row] : partitioned_indices )
        {
            if( row < first || row >= last ) continue;
            const auto reference_row = static_cast< std::size_t >( reference_indices.at( key ) );
            PetscScalar residual_value = 0;
            require( VecGetValues( residual, 1, &row, &residual_value ) == PETSC_SUCCESS,
                     "read owned partitioned advection residual" );
            require( !PetscIsInfOrNanScalar( residual_value ),
                     "production partitioned advection residual is finite" );
            require( std::abs( PetscRealPart( residual_value ) - reference.residual[reference_row] ) < 1.e-12,
                     "production partitioned advection residual equals serial stable-key reference" );
            std::vector< PetscScalar > values( columns.size() );
            require( MatGetValues( jacobian, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(), values.data() ) == PETSC_SUCCESS,
                     "read owned partitioned advection matrix row" );
            for( std::size_t column = 0; column < values.size(); ++column )
            {
                require( !PetscIsInfOrNanScalar( values[column] ),
                         "production partitioned advection Jacobian is finite" );
                require( std::abs( PetscRealPart( values[column] ) -
                                   reference.matrix[reference_row * columns.size() + column] ) < 1.e-12,
                         "production partitioned advection Jacobian equals serial stable-key reference" );
            }
        }
        const auto canonical_faces = global_canonical_interior_face_count( system );
        require( canonical_faces == reference.interior_faces,
                 "production partitioned advection has the serial canonical interior-face count" );
        if( mutate )
        {
#ifdef KARMA_DG_FACE_MUTATION_TEST
            mutant_residual = system.create_vector();
            mutant_jacobian = system.create_matrix();
            KARMA::assemble_advection_reaction_with_duplicate_interior_faces(
                system, solution, mutant_residual, mutant_jacobian, partitioned_advection_coefficients(), 5 );
#else
            require( false, "DG face mutation mode requires the mutation-only test executable" );
#endif
        }

        for( const auto& [key, row] : partitioned_indices )
        {
            if( !mutate || row < first || row >= last ) continue;
            const auto reference_row = static_cast< std::size_t >( reference_indices.at( key ) );
            PetscScalar residual_value = 0;
            require( VecGetValues( residual, 1, &row, &residual_value ) == PETSC_SUCCESS,
                     "reread owned production advection residual" );
            PetscScalar mutant_residual_value = 0;
            require( VecGetValues( mutant_residual, 1, &row, &mutant_residual_value ) == PETSC_SUCCESS,
                     "read owned duplicate-traversal advection residual" );
            require( !PetscIsInfOrNanScalar( mutant_residual_value ),
                     "duplicate-traversal advection residual is finite" );
            if( face_dofs.contains( row ) &&
                std::abs( PetscRealPart( mutant_residual_value - residual_value ) ) >= 1.e-12 )
                local_mutant_residual_difference = true;
            if( std::abs( PetscRealPart( mutant_residual_value ) - reference.residual[reference_row] ) >= 1.e-12 )
                local_mutant_residual_reference_mismatch = true;

            std::vector< PetscScalar > values( columns.size() ), mutant_values( columns.size() );
            require( MatGetValues( jacobian, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(),
                                   values.data() ) == PETSC_SUCCESS,
                     "reread owned production advection matrix row" );
            require( MatGetValues( mutant_jacobian, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(),
                                   mutant_values.data() ) == PETSC_SUCCESS,
                     "read owned duplicate-traversal advection matrix row" );
            for( std::size_t column = 0; column < mutant_values.size(); ++column )
            {
                require( !PetscIsInfOrNanScalar( mutant_values[column] ),
                         "duplicate-traversal advection Jacobian is finite" );
                const PetscReal mutant_value = PetscRealPart( mutant_values[column] );
                if( face_couplings.contains( { row, columns[column] } ) &&
                    std::abs( mutant_value - PetscRealPart( values[column] ) ) >= 1.e-12 )
                    local_mutant_jacobian_difference = true;
                if( std::abs( mutant_value - reference.matrix[reference_row * columns.size() + column] ) >= 1.e-12 )
                    local_mutant_jacobian_reference_mismatch = true;
            }
        }
        if( mutate )
        {
#ifdef KARMA_DG_FACE_MUTATION_TEST
            require( KARMA::duplicate_interior_face_traversal_count() == 2 * canonical_faces,
                     "duplicate traversal visits each canonical interior face exactly twice" );
#endif
            int discrepancies[] = { local_mutant_residual_difference ? 1 : 0,
                                    local_mutant_jacobian_difference ? 1 : 0,
                                    local_mutant_residual_reference_mismatch ? 1 : 0,
                                    local_mutant_jacobian_reference_mismatch ? 1 : 0 };
            require( MPI_Allreduce( MPI_IN_PLACE, discrepancies, 4, MPI_INT, MPI_MAX,
                                    PETSC_COMM_WORLD ) == MPI_SUCCESS,
                     "reduce duplicate-face mutation discrepancies" );
            check( discrepancies[0] == 1,
                   "duplicate traversal changes an interior-face residual entry from production" );
            check( discrepancies[1] == 1,
                   "duplicate traversal changes an interior-face Jacobian coupling from production" );
            check( discrepancies[2] == 1,
                   "duplicate-traversal residual mismatches the serial stable-key reference" );
            check( discrepancies[3] == 1,
                   "duplicate-traversal Jacobian mismatches the serial stable-key reference" );
        }
        else
        {
            PetscScalar sum = 0;
            require( VecSum( residual, &sum ) == PETSC_SUCCESS, "sum partitioned advection residual" );
            check( std::abs( PetscRealPart( sum ) ) < 1.e-12,
                   "closed partitioned advection flux is globally conservative" );
        }
    }
    MatDestroy( &mutant_jacobian );
    VecDestroy( &mutant_residual );
    MatDestroy( &jacobian );
    VecDestroy( &residual );
}

void check_mixed_tet_prism( unsigned order )
{
    KARMA::KARMAMesh mesh( 3, PETSC_COMM_WORLD );
    const double coordinates[] = { 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, -1,
                                   0, 0, 1, 1, 0, 1, 0, 1, 1 };
    moab::Range vertices;
    check( mesh.moab()->create_vertices( coordinates, 7, vertices ) == moab::MB_SUCCESS,
           "create mixed tet/prism vertices" );
    const std::vector< moab::EntityHandle > v( vertices.begin(), vertices.end() );
    const moab::EntityHandle tet_connectivity[] = { v[0], v[1], v[2], v[3] };
    const moab::EntityHandle prism_connectivity[] = { v[0], v[2], v[1], v[4], v[5], v[6] };
    moab::EntityHandle tet = 0, prism = 0;
    check( mesh.moab()->create_element( moab::MBTET, tet_connectivity, 4, tet ) == moab::MB_SUCCESS,
           "create mixed tetrahedron" );
    check( mesh.moab()->create_element( moab::MBPRISM, prism_connectivity, 6, prism ) == moab::MB_SUCCESS,
           "create mixed prism" );
    mesh.assign_ids();
    mesh.update();

    unsigned tet_side = mesh.n_sides( tet ), prism_side = mesh.n_sides( prism );
    for( unsigned side = 0; side < mesh.n_sides( tet ); ++side )
        if( mesh.neighbor( tet, side ) == prism ) tet_side = side;
    for( unsigned side = 0; side < mesh.n_sides( prism ); ++side )
        if( mesh.neighbor( prism, side ) == tet ) prism_side = side;
    check( tet_side < mesh.n_sides( tet ) && prism_side < mesh.n_sides( prism ),
           "mixed tetrahedron/prism discover common triangular face" );

    KARMA::KARMAFEBase tet_face( mesh, KARMA::GeometryVariant::Tet4,
        KARMA::KARMABasisRegistry().create(
            { KARMA::ReferenceCell::Tetrahedron, KARMA::FieldBasisFamily::LagrangeNodal, order,
              KARMA::BasisContinuity::dG } ), 4, KARMA::FEIntegrationDomain::Face );
    KARMA::KARMAFEBase prism_face( mesh, KARMA::GeometryVariant::Prism6,
        KARMA::KARMABasisRegistry().create(
            { KARMA::ReferenceCell::Prism, KARMA::FieldBasisFamily::LagrangeNodal, order,
              KARMA::BasisContinuity::dG } ), 4, KARMA::FEIntegrationDomain::Face );
    tet_face.reinit( tet, tet_side );
    const auto points = tet_face.get_xyz();
    const auto normals = tet_face.get_normals();
    prism_face.reinit_at_physical_points( prism, points );
    for( std::size_t q = 0; q < points.size(); ++q )
        check( ( prism_face.get_xyz()[q] - points[q] ).norm() < 1.e-12,
               "mixed tet/prism transferred quadrature point matches source physical point" );
    prism_face.reinit( prism, prism_side );
    check( normals.front().dot( prism_face.get_normals().front() ) < -1 + 1.e-12,
           "mixed 3D shared-face normals have opposite orientation" );

    KARMA::KARMASystem system( mesh, "mixed-tet-prism" );
    system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, order,
                           KARMA::BasisContinuity::dG, 1 } );
    system.init();
    system.add_element_adjacency( tet, prism );
    Vec solution = system.create_vector();
    Vec residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    VecSet( solution, 2 );
    KARMA::AdvectionReactionCoefficients coefficients;
    coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitZ(); };
    coefficients.inflow = []( const KARMA::Point&, int ) { return 2; };
    KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
    PetscReal norm = 0;
    VecNorm( residual, NORM_INFINITY, &norm );
    check( norm < 2.e-11, "mixed tet/prism preserves a compatible constant trace" );
    std::vector< PetscInt > tet_dofs, prism_dofs;
    system.dof_map().dof_indices( tet, tet_dofs );
    system.dof_map().dof_indices( prism, prism_dofs );
    bool coupled = false;
    for( const auto row : prism_dofs )
        for( const auto column : tet_dofs )
            coupled = coupled || std::abs( matrix_entry( jacobian, row, column ) ) > 1.e-13;
    check( coupled, "mixed tet/prism P0/P1 assembly contains shared-face off-diagonal coupling" );
    MatDestroy( &jacobian );
    VecDestroy( &residual );
    VecDestroy( &solution );
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if( mode == "--write-partitioned-advection-reference" ||
            mode == "--write-partitioned-advection-p1-reference" )
        {
            check( argc == 4, "serial advection reference mode requires mesh and reference paths" );
            if( argc == 4 )
                assemble_partitioned_advection( argv[2], false, argv[3],
                                                mode == "--write-partitioned-advection-p1-reference" ? 1U : 0U );
        }
        else if( mode == "--verify-partitioned-advection" || mode == "--verify-partitioned-advection-p1" ||
                 mode == "--verify-partitioned-advection-mutation" )
        {
            check( argc == 4, "partitioned advection mode requires mesh and reference paths" );
            if( argc == 4 )
                assemble_partitioned_advection( argv[2], true, argv[3],
                                                mode == "--verify-partitioned-advection-p1" ? 1U : 0U,
                                                mode == "--verify-partitioned-advection-mutation" );
        }
        else if( mode == "--partitioned-advection-without-ghosts" )
        {
            check( argc == 3, "missing-ghost advection mode requires a mesh path" );
            if( argc == 3 ) initialize_partitioned_advection_without_ghosts( argv[2] );
        }
        else if( mode == "--partitioned-advection-distinct-handles" )
        {
            check( argc == 3, "distinct-handle advection mode requires a mesh path" );
            if( argc == 3 ) initialize_advection_with_distinct_ghost_handles( argv[2] );
        }
        else if( mode == "--partitioned-advection-nonmanifold" )
            initialize_partitioned_advection_nonmanifold();
        else if( mode.rfind( "--asymmetric-object-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0; MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh object_mesh( 1, PETSC_COMM_WORLD );
            object_mesh.generate_box( 2, 0, 0 );
            KARMA::KARMASystem object_system( object_mesh, "object-validation" );
            object_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                                          KARMA::BasisContinuity::dG, 1 } );
            object_system.init();
            Vec solution = object_system.create_vector(), residual = object_system.create_vector(), wrong_vector = nullptr;
            Mat jacobian = object_system.create_matrix(), wrong_matrix = nullptr, self_matrix = nullptr;
            Vec self_vector = nullptr, plain_vector = nullptr;
            Mat plain_matrix = nullptr;
            VecCreateMPI( PETSC_COMM_WORLD, PETSC_DECIDE, 3, &wrong_vector );
            VecCreateMPI( PETSC_COMM_WORLD, object_system.n_local_dofs(), object_system.n_dofs(), &plain_vector );
            VecCreateSeq( PETSC_COMM_SELF, object_system.n_dofs(), &self_vector );
            MatCreateAIJ( PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 3, 3, 1, nullptr, 0, nullptr, &wrong_matrix );
            MatSetUp( wrong_matrix );
            MatCreateAIJ( PETSC_COMM_WORLD, object_system.n_local_dofs(), object_system.n_local_dofs(),
                          object_system.n_dofs(), object_system.n_dofs(), 1, nullptr, 0, nullptr, &plain_matrix );
            MatSetUp( plain_matrix );
            MatCreateSeqAIJ( PETSC_COMM_SELF, object_system.n_dofs(), object_system.n_dofs(), 1, nullptr, &self_matrix );
            VecSet( residual, 61 );
            MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ); MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY );
            MatShift( jacobian, 67 );
            Vec passed_solution = solution, passed_residual = residual;
            Mat passed_jacobian = jacobian;
            if( rank == 1 && mode == "--asymmetric-object-solution-null" ) passed_solution = nullptr;
            if( rank == 1 && mode == "--asymmetric-object-residual-null" ) passed_residual = nullptr;
            if( rank == 1 && mode == "--asymmetric-object-solution-size" ) passed_solution = wrong_vector;
            if( rank == 1 && mode == "--asymmetric-object-solution-comm" ) passed_solution = self_vector;
            if( rank == 1 && mode == "--asymmetric-object-solution-layout" ) passed_solution = plain_vector;
            if( rank == 1 && mode == "--asymmetric-object-residual-size" ) passed_residual = wrong_vector;
            if( rank == 1 && mode == "--asymmetric-object-residual-comm" ) passed_residual = self_vector;
            if( rank == 1 && mode == "--asymmetric-object-residual-layout" ) passed_residual = plain_vector;
            if( rank == 1 && mode == "--asymmetric-object-jacobian-size" ) passed_jacobian = wrong_matrix;
            if( rank == 1 && mode == "--asymmetric-object-jacobian-comm" ) passed_jacobian = self_matrix;
            if( rank == 1 && mode == "--asymmetric-object-jacobian-layout" ) passed_jacobian = plain_matrix;
            if( rank == 1 && mode == "--asymmetric-object-jacobian-presence" ) passed_jacobian = nullptr;
            KARMA::AdvectionReactionCoefficients object_coefficients;
            const bool mass_mode = mode.rfind( "--asymmetric-object-mass-", 0 ) == 0;
            if( mass_mode )
            {
                KARMA::AdvectionReactionMassOperator mass_operator( object_system );
                Mat passed_mass = jacobian;
                Vec passed_input = solution, passed_output = residual;
                if( rank == 1 && mode == "--asymmetric-object-mass-size" ) passed_mass = wrong_matrix;
                if( rank == 1 && mode == "--asymmetric-object-mass-comm" ) passed_mass = self_matrix;
                if( rank == 1 && mode == "--asymmetric-object-mass-layout" ) passed_mass = plain_matrix;
                if( rank == 1 && mode == "--asymmetric-object-mass-input-layout" ) passed_input = plain_vector;
                if( rank == 1 && mode == "--asymmetric-object-mass-output-layout" ) passed_output = plain_vector;
                check( throws_with( [&] { mass_operator.multiply( passed_mass, passed_input, passed_output ); }, "rank 1" ),
                       "asymmetric advection mass object is rejected collectively" );
            }
            else
            {
                check( throws_with( [&] { KARMA::assemble_advection_reaction( object_system, passed_solution,
                                                                              passed_residual, passed_jacobian,
                                                                              object_coefficients ); }, "rank 1" ),
                       "asymmetric advection object is rejected collectively" );
                KARMA::AdvectionReactionMassOperator mass_operator( object_system );
                Mat passed_mass = jacobian;
                Vec passed_input = solution;
                if( rank == 1 && mode == "--asymmetric-object-jacobian-size" ) passed_mass = wrong_matrix;
                else if( rank == 1 && mode == "--asymmetric-object-jacobian-comm" ) passed_mass = self_matrix;
                else if( rank == 1 && mode == "--asymmetric-object-jacobian-presence" ) passed_mass = plain_matrix;
                else if( rank == 1 && mode.find( "size" ) != std::string_view::npos ) passed_input = wrong_vector;
                else if( rank == 1 && mode.find( "comm" ) != std::string_view::npos ) passed_input = self_vector;
                else if( rank == 1 ) passed_input = plain_vector;
                check( throws_with( [&] { mass_operator.multiply( passed_mass, passed_input, residual ); }, "rank 1" ),
                       "asymmetric advection mass vector is rejected collectively" );
            }
            PetscReal residual_norm = 0, jacobian_norm = 0;
            VecNorm( residual, NORM_INFINITY, &residual_norm ); MatNorm( jacobian, NORM_INFINITY, &jacobian_norm );
            check( std::abs( residual_norm - 61 ) < 1.e-12 && std::abs( jacobian_norm - 67 ) < 1.e-12,
                   "invalid advection objects leave outputs unchanged" );
            MatDestroy( &self_matrix ); MatDestroy( &plain_matrix ); MatDestroy( &wrong_matrix );
            VecDestroy( &self_vector ); VecDestroy( &plain_vector ); VecDestroy( &wrong_vector );
            MatDestroy( &jacobian ); VecDestroy( &residual ); VecDestroy( &solution );
        }
        else if( mode == "--asymmetric-local-element-map-failure" )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh asymmetric_mesh( 1, PETSC_COMM_WORLD );
            asymmetric_mesh.generate_box( 2, 0, 0 );
            KARMA::AdvectionReactionSystemPropertyManager properties;
            properties.register_model( 0, partitioned_advection_coefficients() );
            KARMA::AdvectionReactionSystem physics( asymmetric_mesh, std::move( properties ) );
            KARMA::detail::set_advection_local_element_map_fault( rank == 1 );
            check( throws_with( [&] { physics.init(); }, "rank 1" ) && !physics.is_initialized(),
                   "rank-local advection element map failure is reported collectively" );
            KARMA::detail::set_advection_local_element_map_fault( false );
        }
        else if( mode.rfind( "--asymmetric-init-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0; MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh asymmetric_mesh( 1, PETSC_COMM_WORLD );
            const double asymmetric_coordinates[] = { 0, 0, 0, 1, 0, 0, 2, 0, 0 };
            moab::Range asymmetric_vertices;
            check( asymmetric_mesh.moab()->create_vertices( asymmetric_coordinates, 3, asymmetric_vertices ) ==
                       moab::MB_SUCCESS,
                   "asymmetric advection test creates vertices" );
            const std::vector< moab::EntityHandle > vertices( asymmetric_vertices.begin(), asymmetric_vertices.end() );
            const moab::EntityHandle first_connectivity[] = {
                vertices[0], rank == 1 && mode == "--asymmetric-init-connectivity" ? vertices[2] : vertices[1] };
            const moab::EntityHandle second_connectivity[] = { vertices[1], vertices[2] };
            moab::EntityHandle asymmetric_elements[] = { 0, 0 };
            check( asymmetric_mesh.moab()->create_element( moab::MBEDGE, first_connectivity, 2,
                                                            asymmetric_elements[0] ) == moab::MB_SUCCESS,
                   "asymmetric advection test creates the first element" );
            check( asymmetric_mesh.moab()->create_element( moab::MBEDGE, second_connectivity, 2,
                                                            asymmetric_elements[1] ) == moab::MB_SUCCESS,
                   "asymmetric advection test creates the second element" );
            const moab::EntityHandle* installed_connectivity = nullptr;
            int installed_count = 0;
            const bool read_back = asymmetric_mesh.moab()->get_connectivity(
                                       asymmetric_elements[0], installed_connectivity, installed_count ) ==
                                   moab::MB_SUCCESS;
            check( read_back && installed_count == 2 && installed_connectivity[0] == first_connectivity[0] &&
                       installed_connectivity[1] == first_connectivity[1],
                   "asymmetric advection test reads back the exact installed connectivity" );
            const bool connectivity_changed = read_back && installed_count == 2 &&
                                              installed_connectivity[0] == vertices[0] &&
                                              installed_connectivity[1] == vertices[2];
            check( connectivity_changed == ( rank == 1 && mode == "--asymmetric-init-connectivity" ),
                   "asymmetric advection connectivity role sequence differs only on rank 1" );
            asymmetric_mesh.assign_ids();
            asymmetric_mesh.update();
            KARMA::AdvectionReactionSystemPropertyManager asymmetric_properties;
            KARMA::AdvectionReactionCoefficients asymmetric_coefficients;
            asymmetric_properties.register_model( rank == 1 && mode == "--asymmetric-init-model" ? 1 : 0,
                                                  asymmetric_coefficients );
            KARMA::AdvectionReactionSystem::Configuration configuration;
            if( rank == 1 && mode == "--asymmetric-init-config" ) configuration.quadrature_degree = 3;
            KARMA::AdvectionReactionSystem asymmetric( asymmetric_mesh, std::move( asymmetric_properties ),
                                                       configuration );
            const std::string_view expected_error = mode == "--asymmetric-init-connectivity"
                                                        ? "partitioned DoF numbering requires a stable partition identity"
                                                        : "rank 1";
            check( throws_with( [&] { asymmetric.init(); }, expected_error ) && !asymmetric.is_initialized(),
                   "asymmetric advection initialization fails collectively before PETSc creation" );
        }
        else if( mode == "--mixed-tet-prism" )
        {
            PetscOptionsClearValue( nullptr, "--mixed-tet-prism" );
            for( const unsigned order : { 0u, 1u } ) check_mixed_tet_prism( order );
        }
        else
        {
        const KARMA::Vector3 normal = KARMA::Vector3::UnitX();
        check( KARMA::upwind_flux( 2, 7, KARMA::Vector3::UnitX(), normal ) == 2,
               "positive normal velocity selects interior trace" );
        check( KARMA::upwind_flux( 2, 7, -KARMA::Vector3::UnitX(), normal ) == -7,
               "negative normal velocity selects exterior trace" );
        check( KARMA::rusanov_flux( 2, 7, 3 * KARMA::Vector3::UnitX(),
                                    -KARMA::Vector3::UnitX(), normal ) == -8,
               "Rusanov flux uses both one-sided material velocities" );
        check( KARMA::rusanov_flux( 2, 7, KARMA::Vector3( 2, 4, 0 ),
                                    KARMA::Vector3( 2, -9, 0 ), normal ) == 4,
               "Rusanov flux accepts discontinuous tangential velocity" );

        KARMA::KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 0, 0 );
        KARMA::KARMASystem system( mesh, "advection" );
        system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                               KARMA::BasisContinuity::dG, 1 } );
        system.init();
        const std::vector< moab::EntityHandle > elements( mesh.elements().begin(), mesh.elements().end() );
        system.add_element_adjacency( elements[0], elements[1] );

        KARMA::AdvectionReactionCoefficients coefficients;
        coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitX(); };
        coefficients.inflow = []( const KARMA::Point&, int ) { return 3.0; };
        Vec solution = system.create_vector();
        Vec residual = system.create_vector();
        Mat jacobian = system.create_matrix();
        VecSet( solution, 3 );
        KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
        PetscReal norm = 0;
        VecNorm( residual, NORM_INFINITY, &norm );
        check( norm < 1.e-12, "constant state with compatible inflow is preserved" );

        std::vector< PetscInt > left, right;
        system.dof_map().dof_indices( elements[0], left );
        system.dof_map().dof_indices( elements[1], right );
        PetscInt first_owned = 0, last_owned = 0;
        MatGetOwnershipRange( jacobian, &first_owned, &last_owned );
        int local_off_diagonal = 0;
        for( PetscInt row = first_owned; row < last_owned; ++row )
        {
            PetscInt count = 0;
            const PetscInt* columns = nullptr;
            const PetscScalar* entries = nullptr;
            MatGetRow( jacobian, row, &count, &columns, &entries );
            for( PetscInt i = 0; i < count; ++i )
                if( columns[i] != row && std::abs( PetscRealPart( entries[i] ) ) > 0 ) local_off_diagonal = 1;
            MatRestoreRow( jacobian, row, &count, &columns, &entries );
        }
        int global_off_diagonal = 0;
        MPI_Allreduce( &local_off_diagonal, &global_off_diagonal, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD );
        check( global_off_diagonal != 0, "upwind Jacobian contains neighbor block" );

        Vec direction = system.create_vector();
        Vec perturbed = system.create_vector();
        Vec perturbed_residual = system.create_vector();
        Vec jacobian_direction = system.create_vector();
        VecSetRandom( direction, nullptr );
        VecWAXPY( perturbed, 1.e-7, direction, solution );
        KARMA::assemble_advection_reaction( system, perturbed, perturbed_residual, nullptr, coefficients );
        VecAXPY( perturbed_residual, -1, residual );
        VecScale( perturbed_residual, 1.e7 );
        MatMult( jacobian, direction, jacobian_direction );
        VecAXPY( perturbed_residual, -1, jacobian_direction );
        VecNorm( perturbed_residual, NORM_2, &norm );
        check( norm < 1.e-7, "DG Jacobian matches a residual finite difference" );

        coefficients.reaction_derivative = []( const KARMA::Point&, Real, int ) {
            return std::numeric_limits< Real >::infinity();
        };
        bool nonfinite_derivative = false;
        try
        {
            KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
        }
        catch( const std::logic_error& error )
        {
            nonfinite_derivative = std::string_view( error.what() ).find( "reaction derivative" ) != std::string_view::npos;
        }
        check( nonfinite_derivative, "non-finite explicit reaction derivative is rejected" );

        const Real infinity = std::numeric_limits< Real >::infinity();
        const auto rejects_nonfinite = [&]( KARMA::AdvectionReactionCoefficients invalid, const char* text ) {
            return throws_with( [&] {
                KARMA::assemble_advection_reaction( system, solution, residual, jacobian, invalid );
            }, text );
        };
        coefficients = {};
        coefficients.velocity = [infinity]( const KARMA::Point&, int ) { return KARMA::Vector3( infinity, 0, 0 ); };
        check( rejects_nonfinite( coefficients, "velocity" ), "non-finite velocity callback is rejected" );
        coefficients = {};
        coefficients.reaction = [infinity]( const KARMA::Point&, Real, int ) { return infinity; };
        check( rejects_nonfinite( coefficients, "reaction" ), "non-finite reaction callback is rejected" );
        coefficients = {};
        coefficients.source = [infinity]( const KARMA::Point&, Real, int ) { return infinity; };
        check( rejects_nonfinite( coefficients, "source" ), "non-finite source callback is rejected" );
        coefficients = {};
        coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitX(); };
        coefficients.inflow = [infinity]( const KARMA::Point&, int ) { return infinity; };
        check( rejects_nonfinite( coefficients, "inflow" ), "non-finite inflow callback is rejected" );
        coefficients = {};
        coefficients.source_derivative = [infinity]( const KARMA::Point&, Real, int ) { return infinity; };
        check( rejects_nonfinite( coefficients, "source derivative" ), "non-finite explicit source derivative is rejected" );
        coefficients = {};
        coefficients.reaction = [infinity]( const KARMA::Point&, Real field, int ) {
            return field == 3 ? 0.0 : infinity;
        };
        check( rejects_nonfinite( coefficients, "reaction derivative" ),
               "non-finite fallback derivative samples are rejected" );

        coefficients = {};
        coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitX(); };
        coefficients.reaction = []( const KARMA::Point&, Real field, int ) { return field * field; };
        coefficients.reaction_derivative = []( const KARMA::Point&, Real field, int ) { return 2 * field; };
        coefficients.source = []( const KARMA::Point&, Real field, int ) { return field * field * field; };
        coefficients.source_derivative = []( const KARMA::Point&, Real field, int ) { return 3 * field * field; };
        coefficients.inflow = []( const KARMA::Point&, int ) { return 3; };
        VecSet( solution, 3 );
        KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
        VecWAXPY( perturbed, 1.e-7, direction, solution );
        KARMA::assemble_advection_reaction( system, perturbed, perturbed_residual, nullptr, coefficients );
        VecAXPY( perturbed_residual, -1, residual );
        VecScale( perturbed_residual, 1.e7 );
        MatMult( jacobian, direction, jacobian_direction );
        VecAXPY( perturbed_residual, -1, jacobian_direction );
        VecNorm( perturbed_residual, NORM_2, &norm );
        check( norm < 2.e-5, "analytic nonlinear reaction/source Jacobian matches finite difference" );
        coefficients.reaction_derivative = {};
        coefficients.source_derivative = {};
        KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
        VecWAXPY( perturbed, 1.e-7, direction, solution );
        KARMA::assemble_advection_reaction( system, perturbed, perturbed_residual, nullptr, coefficients );
        VecAXPY( perturbed_residual, -1, residual );
        VecScale( perturbed_residual, 1.e7 );
        MatMult( jacobian, direction, jacobian_direction );
        VecAXPY( perturbed_residual, -1, jacobian_direction );
        VecNorm( perturbed_residual, NORM_2, &norm );
        check( norm < 2.e-5, "scale-aware fallback reaction/source Jacobian matches finite difference" );

        coefficients = {};
        VecSet( residual, 17 );
        MatZeroEntries( jacobian );
        MatShift( jacobian, 19 );
        coefficients.source = []( const KARMA::Point&, Real, int ) -> Real {
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            if( rank == 1 ) throw std::runtime_error( "asymmetric advection callback" );
            return 0;
        };
        if( mesh.n_ranks() == 2 )
        {
            bool collective_failure = false;
            try
            {
                KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
            }
            catch( const std::logic_error& error )
            {
                collective_failure = std::string_view( error.what() ).find( "rank 1" ) != std::string_view::npos &&
                                     std::string_view( error.what() ).find( "asymmetric advection callback" ) != std::string_view::npos;
            }
            check( collective_failure, "asymmetric callback failure reaches every rank" );
            PetscReal retained_residual = 0, retained_jacobian = 0;
            VecNorm( residual, NORM_INFINITY, &retained_residual );
            MatNorm( jacobian, NORM_INFINITY, &retained_jacobian );
            check( std::abs( retained_residual - 17 ) < 1.e-12 && std::abs( retained_jacobian - 19 ) < 1.e-12,
                   "callback failure occurs before any Mat/Vec output mutation" );

            coefficients = {};
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            if( rank == 1 )
            {
                moab::Tag material_tag = nullptr;
                moab::EntityHandle conflicting_set = 0;
                mesh.moab()->create_meshset( moab::MESHSET_SET, conflicting_set );
                const auto owned_element = system.dof_map().assembly_elements().front();
                mesh.moab()->add_entities( conflicting_set, &owned_element, 1 );
                int conflicting_material = 99;
                mesh.moab()->tag_get_handle( "MATERIAL_SET", 1, moab::MB_TYPE_INTEGER, material_tag,
                                             moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT );
                mesh.moab()->tag_set_data( material_tag, &conflicting_set, 1, &conflicting_material );
                moab::EntityHandle another_set = 0;
                mesh.moab()->create_meshset( moab::MESHSET_SET, another_set );
                mesh.moab()->add_entities( another_set, &owned_element, 1 );
                conflicting_material = 100;
                mesh.moab()->tag_set_data( material_tag, &another_set, 1, &conflicting_material );
            }
            coefficients = {};
            bool collective_material_failure = false;
            try
            {
                KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients );
            }
            catch( const std::logic_error& error )
            {
                collective_material_failure = std::string_view( error.what() ).find( "rank 1" ) != std::string_view::npos &&
                                              std::string_view( error.what() ).find( "multiple material sets" ) != std::string_view::npos;
            }
            check( collective_material_failure, "asymmetric material failure reaches every rank" );
        }
        VecDestroy( &jacobian_direction );
        VecDestroy( &perturbed_residual );
        VecDestroy( &perturbed );
        VecDestroy( &direction );
        MatDestroy( &jacobian );
        VecDestroy( &residual );
        VecDestroy( &solution );

        if( mesh.n_ranks() == 1 )
        {
            KARMA::KARMAMesh boundary_mesh( 1, PETSC_COMM_WORLD );
            boundary_mesh.generate_box( 1, 0, 0 );
            KARMA::KARMASystem boundary_system( boundary_mesh, "boundary-entries" );
            boundary_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                                            KARMA::BasisContinuity::dG, 1 } );
            boundary_system.init();
            Vec boundary_solution = boundary_system.create_vector();
            Vec boundary_residual = boundary_system.create_vector();
            Mat boundary_jacobian = boundary_system.create_matrix();
            VecSet( boundary_solution, 2 );
            KARMA::AdvectionReactionCoefficients boundary_coefficients;
            boundary_coefficients.velocity = []( const KARMA::Point& point, int ) -> KARMA::Vector3 {
                return point( 0 ) < 0.5 ? KARMA::Vector3( 1, 0, 0 ) : KARMA::Vector3( 0, 0, 0 );
            };
            boundary_coefficients.inflow = []( const KARMA::Point&, int ) { return 3; };
            KARMA::assemble_advection_reaction( boundary_system, boundary_solution, boundary_residual,
                                                boundary_jacobian, boundary_coefficients );
            check( std::abs( vector_entry( boundary_residual, 0 ) + 3 ) < 1.e-12,
                   "P0 inflow residual entry is exact" );
            check( std::abs( matrix_entry( boundary_jacobian, 0, 0 ) ) < 1.e-12,
                   "P0 inflow Jacobian entry is exactly zero" );
            boundary_coefficients.velocity = []( const KARMA::Point& point, int ) -> KARMA::Vector3 {
                return point( 0 ) > 0.5 ? KARMA::Vector3( 1, 0, 0 ) : KARMA::Vector3( 0, 0, 0 );
            };
            KARMA::assemble_advection_reaction( boundary_system, boundary_solution, boundary_residual,
                                                boundary_jacobian, boundary_coefficients );
            check( std::abs( vector_entry( boundary_residual, 0 ) - 2 ) < 1.e-12,
                   "P0 outflow residual entry is exact" );
            check( std::abs( matrix_entry( boundary_jacobian, 0, 0 ) - 1 ) < 1.e-12,
                   "P0 outflow Jacobian entry is exact" );
            MatDestroy( &boundary_jacobian );
            VecDestroy( &boundary_residual );
            VecDestroy( &boundary_solution );

            KARMA::KARMAMesh interface_mesh( 1, PETSC_COMM_WORLD );
            interface_mesh.generate_box( 2, 0, 0 );
            const std::vector< moab::EntityHandle > interface_elements( interface_mesh.elements().begin(),
                                                                         interface_mesh.elements().end() );
            set_material( interface_mesh, interface_elements[0], 11 );
            set_material( interface_mesh, interface_elements[1], 22 );
            KARMA::KARMASystem interface_system( interface_mesh, "rusanov-material" );
            interface_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                                             KARMA::BasisContinuity::dG, 1 } );
            interface_system.init();
            interface_system.add_element_adjacency( interface_elements[0], interface_elements[1] );
            std::vector< PetscInt > left_dof, right_dof;
            interface_system.dof_map().dof_indices( interface_elements[0], left_dof );
            interface_system.dof_map().dof_indices( interface_elements[1], right_dof );
            Vec interface_solution = interface_system.create_vector();
            Vec interface_residual = interface_system.create_vector();
            Mat interface_jacobian = interface_system.create_matrix();
            VecSet( interface_solution, 0 );
            VecSetValue( interface_solution, left_dof[0], 2, INSERT_VALUES );
            VecSetValue( interface_solution, right_dof[0], 7, INSERT_VALUES );
            VecAssemblyBegin( interface_solution );
            VecAssemblyEnd( interface_solution );
            KARMA::AdvectionReactionCoefficients interface_coefficients;
            interface_coefficients.velocity = []( const KARMA::Point&, int material ) -> KARMA::Vector3 {
                return material == 11 ? KARMA::Vector3( 3, 0, 0 ) : KARMA::Vector3( -1, 0, 0 );
            };
            interface_coefficients.inflow = []( const KARMA::Point&, int ) { return 0; };
            KARMA::assemble_advection_reaction( interface_system, interface_solution, interface_residual,
                                                interface_jacobian, interface_coefficients );
            check( std::abs( matrix_entry( interface_jacobian, left_dof[0], right_dof[0] ) + 2 ) < 1.e-12 &&
                       std::abs( matrix_entry( interface_jacobian, right_dof[0], left_dof[0] ) + 3 ) < 1.e-12,
                   "Rusanov assembly has exact nonzero off-diagonal blocks in both directions" );
            Vec interface_direction = interface_system.create_vector();
            Vec interface_perturbed = interface_system.create_vector();
            Vec interface_perturbed_residual = interface_system.create_vector();
            Vec interface_jacobian_direction = interface_system.create_vector();
            VecSetValue( interface_direction, left_dof[0], 0.4, INSERT_VALUES );
            VecSetValue( interface_direction, right_dof[0], -0.7, INSERT_VALUES );
            VecAssemblyBegin( interface_direction );
            VecAssemblyEnd( interface_direction );
            VecWAXPY( interface_perturbed, 1.e-7, interface_direction, interface_solution );
            KARMA::assemble_advection_reaction( interface_system, interface_perturbed,
                                                interface_perturbed_residual, nullptr, interface_coefficients );
            VecAXPY( interface_perturbed_residual, -1, interface_residual );
            VecScale( interface_perturbed_residual, 1.e7 );
            MatMult( interface_jacobian, interface_direction, interface_jacobian_direction );
            VecAXPY( interface_perturbed_residual, -1, interface_jacobian_direction );
            PetscReal interface_fd_norm = 0;
            VecNorm( interface_perturbed_residual, NORM_2, &interface_fd_norm );
            check( interface_fd_norm < 1.e-7, "Rusanov finite difference covers all four interface blocks" );
            VecDestroy( &interface_jacobian_direction );
            VecDestroy( &interface_perturbed_residual );
            VecDestroy( &interface_perturbed );
            VecDestroy( &interface_direction );
            const Real interface_pair_sum = vector_entry( interface_residual, left_dof[0] ) +
                                            vector_entry( interface_residual, right_dof[0] );
            check( std::isfinite( interface_pair_sum ), "discontinuous-normal two-material flux assembles finitely" );

            interface_coefficients.velocity = []( const KARMA::Point& point, int material ) -> KARMA::Vector3 {
                if( point( 0 ) < 0.25 || point( 0 ) > 0.75 ) return KARMA::Vector3::Zero();
                return material == 11 ? KARMA::Vector3( 3, 0, 0 ) : KARMA::Vector3( -1, 0, 0 );
            };
            KARMA::assemble_advection_reaction( interface_system, interface_solution, interface_residual,
                                                interface_jacobian, interface_coefficients );
            const Real closed_sum = vector_entry( interface_residual, left_dof[0] ) +
                                    vector_entry( interface_residual, right_dof[0] );
            check( std::abs( closed_sum ) < 1.e-12,
                   "nonconstant state has exact global conservation with closed boundary flux" );

            Vec ordered_residual = interface_system.create_vector();
            VecCopy( interface_residual, ordered_residual );
            const KARMA::GlobalId first_id = interface_mesh.global_id( interface_elements[0] );
            const KARMA::GlobalId second_id = interface_mesh.global_id( interface_elements[1] );
            interface_mesh.moab()->tag_set_data( interface_mesh.global_id_tag(), &interface_elements[0], 1, &second_id );
            interface_mesh.moab()->tag_set_data( interface_mesh.global_id_tag(), &interface_elements[1], 1, &first_id );
            KARMA::assemble_advection_reaction( interface_system, interface_solution, interface_residual,
                                                interface_jacobian, interface_coefficients );
            VecAXPY( ordered_residual, -1, interface_residual );
            PetscReal ordering_norm = 0;
            VecNorm( ordered_residual, NORM_INFINITY, &ordering_norm );
            check( ordering_norm < 1.e-12, "Rusanov residual is independent of element global-ID ordering" );
            VecDestroy( &ordered_residual );

            interface_coefficients.velocity = []( const KARMA::Point&, int material ) {
                return KARMA::Vector3( 2, material == 11 ? 4 : -9, 0 );
            };
            KARMA::assemble_advection_reaction( interface_system, interface_solution, interface_residual,
                                                interface_jacobian, interface_coefficients );
            check( std::isfinite( vector_entry( interface_residual, left_dof[0] ) ),
                   "continuous normal with differing tangential velocity is accepted" );

            KARMA::GlobalId duplicate = interface_mesh.global_id( interface_elements[0] );
            check( interface_mesh.moab()->tag_set_data( interface_mesh.global_id_tag(), &interface_elements[1], 1,
                                                        &duplicate ) == moab::MB_SUCCESS,
                   "install duplicate element global id" );
            check( throws_with( [&] {
                       KARMA::assemble_advection_reaction( interface_system, interface_solution, interface_residual,
                                                           interface_jacobian, interface_coefficients );
                   }, "duplicate global element id" ),
                   "duplicate global element ids are rejected before face ordering" );
            MatDestroy( &interface_jacobian );
            VecDestroy( &interface_residual );
            VecDestroy( &interface_solution );
        }

        KARMA::KARMAMesh lifecycle_mesh( 1, PETSC_COMM_WORLD );
        lifecycle_mesh.generate_box( 2, 0, 0 );
        if( lifecycle_mesh.n_ranks() > 1 )
        {
            const KARMA::GlobalId local_first = lifecycle_mesh.global_id( *lifecycle_mesh.elements().begin() );
            std::vector< KARMA::GlobalId > first_ids( static_cast< std::size_t >( lifecycle_mesh.n_ranks() ) );
            MPI_Allgather( &local_first, 1, MPI_INT, first_ids.data(), 1, MPI_INT, PETSC_COMM_WORLD );
            check( std::adjacent_find( first_ids.begin(), first_ids.end() ) == first_ids.end(),
                   "generated replicated meshes retain rank-private element IDs" );
        }
        KARMA::AdvectionReactionSystemPropertyManager properties;
        properties.register_model( 0, coefficients );
        KARMA::AdvectionReactionSystem physics( lifecycle_mesh, properties );
        physics.init();
        physics.init();
        check( physics.is_initialized() && physics.system().n_dofs() == 2 && physics.mass() && physics.jacobian(),
               "advection lifecycle is idempotent and owns PETSc operators" );
        physics.clear();
        check( !physics.is_initialized(), "advection clear releases initialized state" );
        physics.init();
        check( physics.is_initialized(), "advection can reinitialize after clear" );
        KARMA::AdvectionReactionSystemOperator physics_operator( physics );
        KARMA::AdvectionReactionPreconditionerOperator preconditioner_operator( physics );
        KARMA::AdvectionReactionMassOperator mass_operator( physics.system() );
        Vec delegated_residual = physics.system().create_vector();
        Mat delegated_jacobian = physics.system().create_matrix();
        Mat delegated_mass = physics.system().create_matrix();
        if( mesh.n_ranks() == 1 )
        {
            Vec plain = nullptr, wrong_size = nullptr, self_vector = nullptr;
            Vec remapped = physics.system().create_vector();
            Mat plain_matrix = nullptr, wrong_size_matrix = nullptr, self_matrix = nullptr;
            Mat remapped_matrix = physics.system().create_matrix();
            VecCreateMPI( PETSC_COMM_WORLD, physics.system().n_local_dofs(), physics.system().n_dofs(), &plain );
            VecCreateMPI( PETSC_COMM_WORLD, physics.system().n_local_dofs() + 1, physics.system().n_dofs() + 1,
                          &wrong_size );
            VecCreateSeq( PETSC_COMM_SELF, physics.system().n_dofs(), &self_vector );
            MatCreateAIJ( PETSC_COMM_WORLD, physics.system().n_local_dofs(), physics.system().n_local_dofs(),
                          physics.system().n_dofs(), physics.system().n_dofs(), 2, nullptr, 0, nullptr, &plain_matrix );
            MatSetUp( plain_matrix );
            MatCreateAIJ( PETSC_COMM_WORLD, physics.system().n_local_dofs() + 1, physics.system().n_local_dofs() + 1,
                          physics.system().n_dofs() + 1, physics.system().n_dofs() + 1, 1, nullptr, 0, nullptr,
                          &wrong_size_matrix );
            MatSetUp( wrong_size_matrix );
            MatCreateSeqAIJ( PETSC_COMM_SELF, physics.system().n_dofs(), physics.system().n_dofs(), 1, nullptr,
                             &self_matrix );
            std::vector< PetscInt > reversed = physics.system().dof_map().local_to_global_indices();
            std::reverse( reversed.begin(), reversed.end() );
            ISLocalToGlobalMapping wrong_mapping = nullptr;
            ISLocalToGlobalMappingCreate( PETSC_COMM_WORLD, 1, static_cast< PetscInt >( reversed.size() ),
                                          reversed.data(), PETSC_COPY_VALUES, &wrong_mapping );
            VecSetLocalToGlobalMapping( remapped, wrong_mapping );
            MatSetLocalToGlobalMapping( remapped_matrix, wrong_mapping, wrong_mapping );
            physics.assemble_jacobian( physics.solution(), delegated_jacobian );
            mass_operator.assemble( delegated_mass );
            VecSet( delegated_residual, 71 );
            MatZeroEntries( delegated_jacobian ); MatZeroEntries( delegated_mass );
            MatShift( delegated_jacobian, 73 ); MatShift( delegated_mass, 79 );
            check( throws_with( [&] { KARMA::assemble_advection_reaction( physics.system(), plain,
                                                                           delegated_residual, nullptr, coefficients ); },
                                "mapping" ),
                   "low-level assembly rejects a same-sized plain solution vector" );
            check( throws_with( [&] { physics.assemble_residual( wrong_size, delegated_residual ); }, "size" ),
                   "wrong-sized solution vector is rejected" );
            check( throws_with( [&] { physics.assemble_residual( remapped, delegated_residual ); }, "mapping" ),
                   "wrong solution mapping is rejected" );
            check( throws_with( [&] { physics.assemble_residual( physics.solution(), plain ); }, "mapping" ),
                   "same-sized plain residual vector is rejected" );
            check( throws_with( [&] { physics.assemble_jacobian( physics.solution(), plain_matrix ); }, "mapping" ),
                   "matrix without system mapping is rejected" );
            check( throws_with( [&] { physics.assemble_jacobian( physics.solution(), wrong_size_matrix ); }, "dimensions" ),
                   "wrong-sized matrix is rejected" );
            check( throws_with( [&] { physics.assemble_jacobian( physics.solution(), remapped_matrix ); }, "mapping" ),
                   "wrong matrix mapping is rejected" );
            check( throws_with( [&] { mass_operator.multiply( delegated_mass, plain, delegated_residual ); }, "mapping" ),
                   "mass adapter rejects a plain vector" );
            check( throws_with( [&] { mass_operator.multiply( delegated_mass, wrong_size, delegated_residual ); }, "size" ),
                   "mass adapter rejects a wrong-sized vector" );
            check( throws_with( [&] { mass_operator.solve( delegated_mass, delegated_residual, remapped ); }, "mapping" ),
                   "mass adapter rejects a remapped vector" );
            check( throws_with( [&] { mass_operator.assemble( plain_matrix ); }, "mapping" ),
                   "mass adapter rejects a matrix without mappings" );
            check( throws_with( [&] { mass_operator.assemble( wrong_size_matrix ); }, "dimensions" ),
                   "mass adapter rejects a wrong-sized matrix" );
            PetscReal retained_residual = 0, retained_jacobian = 0, retained_mass = 0;
            VecNorm( delegated_residual, NORM_INFINITY, &retained_residual );
            MatNorm( delegated_jacobian, NORM_INFINITY, &retained_jacobian );
            MatNorm( delegated_mass, NORM_INFINITY, &retained_mass );
            check( std::abs( retained_residual - 71 ) < 1.e-12 && std::abs( retained_jacobian - 73 ) < 1.e-12 &&
                       std::abs( retained_mass - 79 ) < 1.e-12,
                   "layout validation leaves advection outputs unchanged" );
            ISLocalToGlobalMappingDestroy( &wrong_mapping );
            MatDestroy( &remapped_matrix ); MatDestroy( &self_matrix ); MatDestroy( &wrong_size_matrix );
            MatDestroy( &plain_matrix );
            VecDestroy( &remapped ); VecDestroy( &self_vector ); VecDestroy( &wrong_size ); VecDestroy( &plain );
        }
        physics_operator.residual( physics.solution(), delegated_residual );
        physics_operator.jacobian( physics.solution(), delegated_jacobian );
        preconditioner_operator.assemble( physics.solution(), delegated_jacobian );
        mass_operator.assemble( delegated_mass );
        PetscReal delegated_mass_norm = 0; MatNorm( delegated_mass, NORM_INFINITY, &delegated_mass_norm );
        check( delegated_mass_norm > 0, "advection mass/operator/preconditioner adapters delegate to owned system" );
        MatDestroy( &delegated_mass ); MatDestroy( &delegated_jacobian ); VecDestroy( &delegated_residual );

        KARMA::AdvectionReactionSystemPropertyManager invalid_properties;
        KARMA::AdvectionReactionSystem invalid( lifecycle_mesh, std::move( invalid_properties ) );
        check( throws_with( [&] { invalid.init(); }, "material model" ) && !invalid.is_initialized(),
               "failed advection initialization rolls back" );
        invalid.properties().register_model( 0, coefficients );
        invalid.init();
        check( invalid.is_initialized(), "failed advection initialization remains retryable" );

        if( mesh.n_ranks() == 1 )
        {
            KARMA::KARMAMesh solve_mesh( 1, PETSC_COMM_WORLD );
            solve_mesh.generate_box( 1, 0, 0 );
            KARMA::AdvectionReactionCoefficients solve_coefficients;
            solve_coefficients.reaction = []( const KARMA::Point&, Real field, int ) { return field; };
            solve_coefficients.reaction_derivative = []( const KARMA::Point&, Real, int ) { return 1; };
            solve_coefficients.source = []( const KARMA::Point&, Real, int ) { return 2; };
            KARMA::AdvectionReactionSystemPropertyManager solve_properties;
            solve_properties.register_model( 0, solve_coefficients );
            KARMA::AdvectionReactionSystem solved( solve_mesh, std::move( solve_properties ) );
            solved.init(); solved.solve();
            check( std::abs( vector_entry( solved.solution(), 0 ) - 2 ) < 1.e-10,
                   "advection-reaction system solve finds the simple reaction equilibrium" );
        }

        if( mesh.n_ranks() == 1 )
            for( const unsigned order : { 0u, 1u } ) check_mixed_tet_prism( order );

        for( const unsigned dimension : { 1u, 2u } )
            for( const unsigned order : { 0u, 1u, 2u } )
            {
                KARMA::KARMAMesh order_mesh( dimension, PETSC_COMM_WORLD );
                order_mesh.generate_box( 1, dimension == 2 ? 1 : 0, 0 );
                KARMA::KARMASystem order_system( order_mesh, "orders" );
                order_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, order,
                                             KARMA::BasisContinuity::dG, 1 } );
                order_system.init();
                Vec order_solution = order_system.create_vector();
                Vec order_residual = order_system.create_vector();
                Mat order_jacobian = order_system.create_matrix();
                VecSet( order_solution, 2 );
                KARMA::AdvectionReactionCoefficients order_coefficients;
                order_coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitX(); };
                order_coefficients.reaction = []( const KARMA::Point&, Real field, int ) { return 4 * field; };
                order_coefficients.reaction_derivative = []( const KARMA::Point&, Real, int ) { return 4; };
                order_coefficients.source = []( const KARMA::Point&, Real, int ) { return 8; };
                order_coefficients.inflow = []( const KARMA::Point&, int ) { return 2; };
                KARMA::assemble_advection_reaction( order_system, order_solution, order_residual, order_jacobian,
                                                    order_coefficients );
                PetscReal order_norm = 0;
                VecNorm( order_residual, NORM_INFINITY, &order_norm );
                check( order_norm < 1.e-11, "P0/P1/P2 in 1D/2D preserve a balanced constant state" );
                MatDestroy( &order_jacobian );
                VecDestroy( &order_residual );
                VecDestroy( &order_solution );
            }
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
