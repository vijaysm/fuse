#include "karma_basis.h"
#include "karma_dof_map.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_moab_io.h"
#include "karma_parameters.h"
#include "karma_physics_bc_manager.h"
#include "karma_system.h"

#include <petsc.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>

namespace
{
std::unique_ptr< KARMABoundaryCondition > dirichlet( const KARMAParameters& parameters )
{
    const unsigned side = parameters.get< unsigned >( "side_id" );
    if( parameters.get< int >( "material" ) != 4 || ( side != 0 && side != 3 ) ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet, true );
    condition->set( parameters.get< Real >( "time" ) );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > all_dirichlet( const KARMAParameters& parameters )
{
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
    condition->set( parameters.get< Real >( "time" ) );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > rank_dirichlet( const KARMAParameters& parameters )
{
    (void)parameters;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    condition->set( static_cast< Real >( rank ) );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > throwing_rank_dirichlet( const KARMAParameters& parameters )
{
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    if( rank == 1 ) throw std::runtime_error( "asymmetric boundary callback failure" );
    return all_dirichlet( parameters );
}
std::unique_ptr< KARMABoundaryCondition > time_dirichlet( const KARMAParameters& parameters )
{
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet, true );
    condition->set( parameters.get< Real >( "time" ) );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > conflicting_dirichlet( const KARMAParameters& parameters )
{
    const unsigned side = parameters.get< unsigned >( "side_id" );
    if( side != 0 && side != 3 ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet, true );
    condition->set( 0 );
    condition->set( 1, static_cast< Real >( side ) );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > neumann( const KARMAParameters& parameters )
{
    if( parameters.get< unsigned >( "side_id" ) != 1 ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Neumann );
    condition->set( 2 );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > robin( const KARMAParameters& parameters )
{
    if( parameters.get< unsigned >( "side_id" ) != 2 ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Robin );
    condition->set( 0, 1 );
    condition->set( 1, 3 );
    condition->set( 2, 5 );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > all_neumann( const KARMAParameters& parameters )
{
    auto condition = neumann( parameters );
    if( condition ) return condition;
    condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Neumann );
    condition->set( 2 );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > all_robin( const KARMAParameters& parameters )
{
    auto condition = robin( parameters );
    if( condition ) return condition;
    condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Robin );
    condition->set( 0, 1 );
    condition->set( 1, 3 );
    condition->set( 2, 5 );
    return condition;
}
std::unique_ptr< KARMABoundaryCondition > wrong_dirichlet( const KARMAParameters& parameters )
{ return neumann( parameters ); }
std::unique_ptr< KARMABoundaryCondition > wrong_neumann( const KARMAParameters& parameters )
{ return robin( parameters ); }
std::unique_ptr< KARMABoundaryCondition > wrong_robin( const KARMAParameters& parameters )
{ return dirichlet( parameters ); }
bool throws_partition_identity( auto&& function )
{
    try
    {
        function();
    }
    catch( const std::logic_error& error )
    {
        return std::string_view( error.what() ).find( "stable partition identity" ) != std::string_view::npos;
    }
    return false;
}

bool throws_invalid( auto&& function )
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

template < typename Value > void append_bytes( std::string& bytes, const Value& value )
{
    bytes.append( reinterpret_cast< const char* >( &value ), sizeof( value ) );
}

class TrackedBoundaryCondition : public KARMABoundaryCondition
{
  public:
    TrackedBoundaryCondition( KARMA::PHYSICS::BCType type, int& destructions )
        : KARMABoundaryCondition( type ), _destructions( destructions )
    {
    }

    ~TrackedBoundaryCondition() override
    { ++_destructions; }

  private:
    int& _destructions;
};

int partitioned_boundary_test( const char* filename, bool conflicting_values, bool time_dependent = false,
                               bool callback_failure = false, bool asymmetric_continuity = false )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true );
    mesh.update();
    KARMA::KARMASystem system( mesh, "partitioned-bc" );
    system.add_variable( "u" );
    system.init();
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAPhysicsBCManager manager( 0, !asymmetric_continuity || rank == 0 );
    if( callback_failure ) manager.attach_function( throwing_rank_dirichlet, KARMA::PHYSICS::Dirichlet );
    else if( conflicting_values ) manager.attach_function( rank_dirichlet, KARMA::PHYSICS::Dirichlet );
    else if( time_dependent ) manager.attach_function( time_dirichlet, KARMA::PHYSICS::Dirichlet );
    else manager.attach_function( all_neumann, KARMA::PHYSICS::Neumann );
    KARMAParameters parameters;
    parameters.set< Real >( "time" ) = 0;
    bool rejected = false;
    try { manager.refresh( mesh, system, parameters ); }
    catch( const KARMA::InvalidArgument& error )
    {
        const std::string_view message( error.what() );
        rejected = asymmetric_continuity ? message.find( "boundary manager collective configuration mismatch" ) != std::string_view::npos :
                   callback_failure ? message.find( "rank 1:" ) != std::string_view::npos &&
                                         message.find( "asymmetric boundary callback failure" ) != std::string_view::npos
                                    : message.find( "Dirichlet" ) != std::string_view::npos;
    }
    int failures = 0;
    if( callback_failure || asymmetric_continuity )
        failures = rejected ? 0 : 1;
    else if( conflicting_values )
        failures = rejected ? 0 : 1;
    else if( time_dependent )
    {
        failures = rejected || manager.dirichlet_dofs().empty() ? 1 : 0;
        if( !manager.dirichlet_dofs().empty() &&
            !manager.get_dirichlet_condition( manager.dirichlet_dofs().front() )->is_time_dependent_bcs() ) ++failures;
        parameters.set< Real >( "time" ) = 7;
        try { manager.refresh( mesh, system, parameters ); }
        catch( const KARMA::InvalidArgument& ) { ++failures; }
        if( !manager.dirichlet_dofs().empty() )
        {
            const auto* condition = manager.get_dirichlet_condition( manager.dirichlet_dofs().front() );
            if( !condition->is_time_dependent_bcs() || std::abs( ( *condition )[0] - 7 ) > 1.e-12 ) ++failures;
        }
    }
    else
    {
        failures = rejected ? 1 : 0;
        unsigned local_natural = manager.n_neumann_nodes();
        unsigned global_natural = 0;
        MPI_Allreduce( &local_natural, &global_natural, 1, MPI_UNSIGNED, MPI_SUM, PETSC_COMM_WORLD );
        if( global_natural != 16 ) ++failures;
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    return global_failures == 0 ? 0 : 1;
}

int empty_partition_boundary_test()
{
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMA::KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const double coordinates[] = { 0, 0, 0, 1, 0, 0 };
        moab::Range vertices;
        mesh.moab()->create_vertices( coordinates, 2, vertices );
        const std::array< moab::EntityHandle, 2 > connectivity{ vertices[0], vertices[1] };
        moab::EntityHandle element = 0;
        mesh.moab()->create_element( moab::MBEDGE, connectivity.data(), 2, element );
    }
    mesh.assign_ids();
    mesh.update();
    mesh.resolve_shared_entities( 0 );
    KARMA::KARMASystem system( mesh, "empty-boundary-partition" );
    system.add_variable( "u" );
    system.init();
    KARMAParameters parameters;
    parameters.set< Real >( "time" ) = 0;
    KARMAPhysicsBCManager natural( 0 );
    natural.attach_function( all_neumann, KARMA::PHYSICS::Neumann );
    KARMAPhysicsBCManager essential( 0 );
    essential.attach_function( all_dirichlet, KARMA::PHYSICS::Dirichlet );
    int failures = 0;
    try
    {
        natural.refresh( mesh, system, parameters );
        essential.refresh( mesh, system, parameters );
    }
    catch( const KARMA::InvalidArgument& )
    {
        ++failures;
    }
    unsigned local_natural = natural.n_neumann_nodes();
    unsigned global_natural = 0;
    MPI_Allreduce( &local_natural, &global_natural, 1, MPI_UNSIGNED, MPI_SUM, PETSC_COMM_WORLD );
    if( global_natural != 2 || essential.dirichlet_dofs().size() != 2 ) ++failures;
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    return global_failures == 0 ? 0 : 1;
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-boundaries" )
    {
        const int result = partitioned_boundary_test( argv[2], false );
        PetscFinalize();
        return result;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-conflicting-dirichlet" )
    {
        const int result = partitioned_boundary_test( argv[2], true );
        PetscFinalize();
        return result;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-time-dirichlet" )
    {
        const int result = partitioned_boundary_test( argv[2], false, true );
        PetscFinalize();
        return result;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-callback-failure" )
    {
        const int result = partitioned_boundary_test( argv[2], false, false, true );
        PetscFinalize();
        return result;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-asymmetric-continuity" )
    {
        const int result = partitioned_boundary_test( argv[2], false, false, false, true );
        PetscFinalize();
        return result;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--empty-partition-boundaries" )
    {
        const int result = empty_partition_boundary_test();
        PetscFinalize();
        return result;
    }
    MPI_Comm_dup( PETSC_COMM_WORLD, &KARMA_COMM_WORLD );
    int failures = 0, ranks = 1;
    MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
    auto check = [&]( bool value, const char* message ) {
        if( !value )
        {
            std::cerr << "FAIL: " << message << '\n';
            ++failures;
        }
    };

    {
        const auto layout = KARMA::detail::checked_mpi_displacements(
            std::array{ std::numeric_limits< int >::max() - 1, 1 }, 1 );
        check( layout.displacements == std::vector< int >( { 0, std::numeric_limits< int >::max() - 1 } ) &&
                   layout.total_count == static_cast< std::size_t >( std::numeric_limits< int >::max() ),
               "boundary gather layout accepts an exact INT_MAX total" );
        check( throws_invalid( [] {
                   (void)KARMA::detail::checked_mpi_displacements(
                       std::array{ std::numeric_limits< int >::max(), 1 }, 1 );
               } ),
               "boundary gather layout rejects a total above INT_MAX without allocation" );

        constexpr std::size_t header_size = sizeof( PetscInt ) + sizeof( std::uint8_t ) + sizeof( std::uint64_t );
        check( throws_invalid( [] { (void)KARMA::detail::decode_dirichlet_conditions( std::string( 1, '\0' ) ); } ),
               "Dirichlet decoder rejects a truncated fixed header" );
        std::string oversized_values;
        const PetscInt dof = 3;
        const std::uint8_t time_dependent = 0;
        const std::uint64_t count = std::numeric_limits< std::uint64_t >::max();
        append_bytes( oversized_values, dof );
        append_bytes( oversized_values, time_dependent );
        append_bytes( oversized_values, count );
        check( oversized_values.size() == header_size &&
                   throws_invalid( [&] { (void)KARMA::detail::decode_dirichlet_conditions( oversized_values ); } ),
               "Dirichlet decoder rejects an overflowing value extent" );
        std::string truncated_values;
        const std::uint64_t one_value = 1;
        append_bytes( truncated_values, dof );
        append_bytes( truncated_values, time_dependent );
        append_bytes( truncated_values, one_value );
        check( throws_invalid( [&] { (void)KARMA::detail::decode_dirichlet_conditions( truncated_values ); } ),
               "Dirichlet decoder rejects truncated values" );
        const std::uint64_t no_values = 0;
        std::string invalid_flag;
        const std::uint8_t invalid_time_dependent = 2;
        append_bytes( invalid_flag, dof );
        append_bytes( invalid_flag, invalid_time_dependent );
        append_bytes( invalid_flag, no_values );
        check( throws_invalid( [&] { (void)KARMA::detail::decode_dirichlet_conditions( invalid_flag ); } ),
               "Dirichlet decoder rejects a non-boolean time-dependent flag" );
        std::string trailing;
        append_bytes( trailing, dof );
        append_bytes( trailing, time_dependent );
        append_bytes( trailing, no_values );
        trailing.push_back( '\0' );
        check( throws_invalid( [&] { (void)KARMA::detail::decode_dirichlet_conditions( trailing ); } ),
               "Dirichlet decoder rejects trailing bytes" );
    }

    {
        int destructions = 0;
        KARMAPhysicsBCManager legacy( 0 );
        auto* dirichlet_condition = new TrackedBoundaryCondition( KARMA::PHYSICS::Dirichlet, destructions );
        auto* neumann_condition   = new TrackedBoundaryCondition( KARMA::PHYSICS::Neumann, destructions );
        auto* robin_condition     = new TrackedBoundaryCondition( KARMA::PHYSICS::Robin, destructions );
        check( legacy.add_dirichlet_condition( 7u, dirichlet_condition ) &&
                   legacy.add_neumann_condition( 8u, 2u, neumann_condition ) &&
                   legacy.add_robin_condition( 9u, 3u, robin_condition ),
               "legacy raw conditions can be added" );
        check( legacy.n_dirichlet_nodes() == 1 && legacy.n_neumann_nodes() == 1 && legacy.n_robin_nodes() == 1,
               "legacy conditions contribute to public counts" );
        check( legacy.get_dirichlet_condition( PetscInt{ 7 } ) == dirichlet_condition &&
                   legacy.get_neumann_condition( 8u, 2u ) == neumann_condition &&
                   legacy.get_robin_condition( 9u, 3u ) == robin_condition,
               "legacy conditions are found by old kernel lookups" );
        std::string dirichlet_key = "D;7";
        std::string neumann_key   = "N;8;2";
        std::string robin_key     = "R;9;3";
        check( legacy.is_dirichlet_condition( dirichlet_key ) && legacy.is_neumann_condition( neumann_key ) &&
                   legacy.is_robin_condition( robin_key ) &&
                   legacy.get_dirichlet_condition().at( dirichlet_key ) == dirichlet_condition &&
                   legacy.get_neumann_condition().at( neumann_key ) == neumann_condition &&
                   legacy.get_robin_condition().at( robin_key ) == robin_condition,
               "legacy raw containers retain their keys and pointers" );
        legacy.print_conditions();

        KARMAPhysicsBCManager copied( legacy );
        KARMAPhysicsBCManager assigned( 1 );
        assigned = legacy;
        check( copied.n_dirichlet_nodes() == 0 && copied.n_neumann_nodes() == 0 && copied.n_robin_nodes() == 0 &&
                   assigned.n_dirichlet_nodes() == 0 && assigned.n_neumann_nodes() == 0 &&
                   assigned.n_robin_nodes() == 0,
               "copy and assignment copy callbacks but not owned legacy conditions" );
        check( legacy.delete_neumann_condition( 8u, 2u ) && !legacy.delete_neumann_condition( 8u, 2u ) &&
                   destructions == 1,
               "legacy deletion releases exactly one owned condition" );
        legacy.clear();
        check( destructions == 3, "legacy clear releases each remaining owned condition exactly once" );
    }

    {
        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        const auto element              = *mesh.elements().begin();
        int material                    = 4;
        moab::EntityHandle material_set = 0;
        moab::Tag material_tag          = nullptr;
        mesh.moab()->create_meshset( moab::MESHSET_SET, material_set );
        mesh.moab()->add_entities( material_set, &element, 1 );
        mesh.moab()->tag_get_handle( "MATERIAL_SET", 1, moab::MB_TYPE_INTEGER, material_tag,
                                     moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT );
        mesh.moab()->tag_set_data( material_tag, &material_set, 1, &material );
        KARMA::KARMASystem system( mesh, "bc" );
        system.add_variable( "u" );
        system.init();
        KARMAPhysicsBCManager manager( 0 );
        manager.attach_function( dirichlet, KARMA::PHYSICS::Dirichlet );
        manager.attach_function( neumann, KARMA::PHYSICS::Neumann );
        manager.attach_function( robin, KARMA::PHYSICS::Robin );
        KARMAParameters parameters;
        parameters.set< Real >( "time" ) = 1;

        if( ranks > 1 )
            check( !throws_partition_identity( [&] { manager.refresh( mesh, system, parameters ); } ),
                   "replicated multi-rank boundary discovery remains supported" );
        else
        {
            manager.refresh( mesh, system, parameters );
            check( manager.n_neumann_nodes() == 1 && manager.n_robin_nodes() == 1,
                   "disjoint serial natural boundary sides are discovered" );
            check( manager.n_dirichlet_nodes() == 3 && manager.dirichlet_dofs().size() == 3,
                   "Dirichlet global PetscInt DoFs are deduplicated across a side seam" );
            const auto id             = mesh.side_id( element, 1 );
            auto* overlapping_neumann = new KARMABoundaryCondition( KARMA::PHYSICS::Neumann );
            auto* extra_neumann       = new KARMABoundaryCondition( KARMA::PHYSICS::Neumann );
            manager.add_neumann_condition( static_cast< unsigned >( id.element ), id.side, overlapping_neumann );
            manager.add_neumann_condition( 999u, 1u, extra_neumann );
            check( manager.n_neumann_nodes() == 2,
                   "modern and legacy counts form a key union without double counting overlaps" );
            check( manager.get_neumann_condition( id.element, id.side ) != overlapping_neumann &&
                       manager.get_neumann_condition( 999u, 1u ) == extra_neumann &&
                       manager.get_neumann_condition( KARMA::GlobalId{ 999 }, 1u ) == extra_neumann,
                   "modern lookups take precedence and fall back to compatible legacy keys" );

            check( manager.get_neumann_condition( id.element, id.side ) != nullptr,
                    "side lookup uses private global element id and CN side" );
            KARMAPhysicsBCManager signature_manager( 0 );
            signature_manager.attach_function( neumann, KARMA::PHYSICS::Neumann );
            signature_manager.refresh( mesh, system, parameters );
            const std::string canonical_signature = signature_manager.assembly_signature( system );
            std::uint64_t serialized_ordinal = std::numeric_limits< std::uint64_t >::max();
            if( canonical_signature.size() >= 1 + sizeof( serialized_ordinal ) )
                std::memcpy( &serialized_ordinal, canonical_signature.data() + 1, sizeof( serialized_ordinal ) );
            check( !canonical_signature.empty() && canonical_signature.front() == 'N' &&
                       serialized_ordinal == system.dof_map().canonical_element_ordinal( element ),
                    "modern side signature serializes the system canonical element ordinal" );

            const KARMA::GlobalId duplicate_id = mesh.global_id( element );
            KARMA::GlobalId missing_id          = duplicate_id + 1000;
            check( mesh.moab()->tag_set_data( mesh.global_id_tag(), &element, 1, &missing_id ) == moab::MB_SUCCESS,
                    "test can replace the private element ID" );
            bool missing_rejected = false;
            try
            {
                (void)manager.assembly_signature( system );
            }
            catch( const std::logic_error& error )
            {
                missing_rejected = std::string_view( error.what() ).find( "missing" ) != std::string_view::npos;
            }
            check( missing_rejected, "modern side signature rejects missing private element ID resolution" );
            check( mesh.moab()->tag_set_data( mesh.global_id_tag(), &element, 1, &duplicate_id ) == moab::MB_SUCCESS,
                    "test restores the private element ID" );

            KARMA::KARMAMesh duplicate_mesh( 2, PETSC_COMM_WORLD );
            duplicate_mesh.generate_box( 2, 1, 0 );
            KARMA::KARMASystem duplicate_system( duplicate_mesh, "duplicate-bc-signature" );
            duplicate_system.add_variable( "u" );
            duplicate_system.init();
            KARMAPhysicsBCManager duplicate_manager( 0 );
            duplicate_manager.attach_function( all_neumann, KARMA::PHYSICS::Neumann );
            duplicate_manager.refresh( duplicate_mesh, duplicate_system, parameters );
            const auto first_duplicate_element = *duplicate_mesh.elements().begin();
            const auto second_duplicate_element = *std::next( duplicate_mesh.elements().begin() );
            const KARMA::GlobalId first_duplicate_id = duplicate_mesh.global_id( first_duplicate_element );
            check( duplicate_mesh.moab()->tag_set_data( duplicate_mesh.global_id_tag(), &second_duplicate_element, 1,
                                                        &first_duplicate_id ) == moab::MB_SUCCESS,
                    "test can create duplicate private element IDs" );
            bool duplicate_rejected = false;
            try
            {
                (void)duplicate_manager.assembly_signature( duplicate_system );
            }
            catch( const std::logic_error& error )
            {
                duplicate_rejected = std::string_view( error.what() ).find( "duplicate" ) != std::string_view::npos;
            }
            check( duplicate_rejected, "modern side signature rejects duplicate private element ID resolution" );

            parameters.set< Real >( "time" ) = 9;

            manager.refresh( mesh, system, parameters );
            check( std::abs( ( *manager.get_dirichlet_condition( manager.dirichlet_dofs().front() ) )[0] - 9 ) < 1.e-12,
                   "time refresh overwrites stale values" );

            const auto preserved_dof   = manager.dirichlet_dofs().front();
            const auto preserved_value = ( *manager.get_dirichlet_condition( preserved_dof ) )[0];
            const std::array< std::pair< modern_boundary_function, KARMA::PHYSICS::BCType >, 3 > mismatches{
                std::pair{ wrong_dirichlet, KARMA::PHYSICS::Dirichlet },
                std::pair{ wrong_neumann, KARMA::PHYSICS::Neumann }, std::pair{ wrong_robin, KARMA::PHYSICS::Robin }
            };
            for( const auto& [callback, type] : mismatches )
            {
                manager.attach_function( callback, type );
                bool mismatch_rejected = false;
                try
                {
                    manager.refresh( mesh, system, parameters );
                }
                catch( const std::logic_error& error )
                {
                    mismatch_rejected = std::string_view( error.what() ).find( "type" ) != std::string_view::npos;
                }
                check( mismatch_rejected && manager.get_dirichlet_condition( preserved_dof ) != nullptr &&
                           manager.n_neumann_nodes() == 2 && manager.n_robin_nodes() == 1 &&
                           std::abs( ( *manager.get_dirichlet_condition( preserved_dof ) )[0] - preserved_value ) <
                               1.e-12,
                       "callback type mismatch rejects refresh and preserves old state" );
                switch( type )
                {
                    case KARMA::PHYSICS::Dirichlet:
                        manager.attach_function( dirichlet, type );
                        break;
                    case KARMA::PHYSICS::Neumann:
                        manager.attach_function( neumann, type );
                        break;
                    case KARMA::PHYSICS::Robin:
                        manager.attach_function( robin, type );
                        break;
                    case KARMA::PHYSICS::UserBC:
                        break;
                }
            }

            KARMAPhysicsBCManager conflicting( 0 );
            conflicting.attach_function( conflicting_dirichlet, KARMA::PHYSICS::Dirichlet );
            bool conflict_rejected = false;
            try
            {
                conflicting.refresh( mesh, system, parameters );
            }
            catch( const std::logic_error& error )
            {
                conflict_rejected =
                    std::string_view( error.what() ).find( "conflicting Dirichlet" ) != std::string_view::npos;
            }
            check( conflict_rejected && conflicting.dirichlet_dofs().empty(),
                   "conflicting shared Dirichlet values reject transactionally" );

            KARMAPhysicsBCManager natural_overlap( 0 );
            natural_overlap.attach_function( all_neumann, KARMA::PHYSICS::Neumann );
            natural_overlap.attach_function( all_robin, KARMA::PHYSICS::Robin );
            bool natural_overlap_rejected = false;
            try
            {
                natural_overlap.refresh( mesh, system, parameters );
            }
            catch( const std::logic_error& error )
            {
                natural_overlap_rejected = std::string_view( error.what() ).find( "natural" ) != std::string_view::npos;
            }
            check( natural_overlap_rejected && natural_overlap.n_neumann_nodes() == 0 &&
                       natural_overlap.n_robin_nodes() == 0,
                   "Neumann and Robin overlap rejects refresh transactionally" );

            for( const auto [natural, type] :
                 { std::pair{ all_neumann, KARMA::PHYSICS::Neumann }, std::pair{ all_robin, KARMA::PHYSICS::Robin } } )
            {
                KARMAPhysicsBCManager overlap( 0 );
                overlap.attach_function( all_dirichlet, KARMA::PHYSICS::Dirichlet );
                overlap.attach_function( natural, type );
                bool overlap_rejected = false;
                try
                {
                    overlap.refresh( mesh, system, parameters );
                }
                catch( const std::logic_error& error )
                {
                    overlap_rejected = std::string_view( error.what() ).find( "overlapping" ) != std::string_view::npos;
                }
                check( overlap_rejected && overlap.dirichlet_dofs().empty() && overlap.n_neumann_nodes() == 0 &&
                           overlap.n_robin_nodes() == 0,
                       "Dirichlet and natural overlap rejects refresh transactionally" );
            }

            KARMA::KARMASystem p2_system( mesh, "p2-bc" );
            p2_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 2, KARMA::BasisContinuity::cG, 2 } );
            p2_system.init();
            KARMAPhysicsBCManager p2_manager( 0 );
            p2_manager.attach_function( dirichlet, KARMA::PHYSICS::Dirichlet );
            p2_manager.refresh( mesh, p2_system, parameters );
            check( p2_manager.dirichlet_dofs().size() == 10,
                   "P2 side discovery constrains vertices and edge DoFs for all components" );
            KARMA::KARMASystem p1_system( mesh, "p1-bc" );
            p1_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 2 } );
            p1_system.init();
            KARMAPhysicsBCManager p1_manager( 0 );
            p1_manager.attach_function( dirichlet, KARMA::PHYSICS::Dirichlet );
            p1_manager.refresh( mesh, p1_system, parameters );
            check( p1_manager.dirichlet_dofs().size() == 6, "P1 side discovery constrains all requested components" );
            KARMAPhysicsBCManager component_manager( 0, true, 1 );
            component_manager.attach_function( dirichlet, KARMA::PHYSICS::Dirichlet );
            component_manager.refresh( mesh, p2_system, parameters );
            check( component_manager.dirichlet_dofs().size() == 5, "component selection constrains one P2 component" );

            KARMA::KARMAMesh volume_mesh( 3, PETSC_COMM_WORLD );
            volume_mesh.generate_box( 1, 1, 1 );
            KARMA::KARMASystem volume_system( volume_mesh, "p2-face-bc" );
            volume_system.add_variable(
                { "u", KARMA::FieldBasisFamily::LagrangeNodal, 2, KARMA::BasisContinuity::cG, 1 } );
            volume_system.init();
            KARMAPhysicsBCManager volume_manager( 0 );
            volume_manager.attach_function( all_dirichlet, KARMA::PHYSICS::Dirichlet );
            volume_manager.refresh( volume_mesh, volume_system, parameters );
            check( volume_manager.dirichlet_dofs().size() == 26,
                   "P2 hex faces constrain vertices, edges, and face DoFs without the cell interior" );

            KARMA::KARMASystem dg_system( mesh, "dg-bc" );
            dg_system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::dG, 1 } );
            dg_system.init();
            KARMAPhysicsBCManager dg_manager( 0, false );
            dg_manager.attach_function( all_dirichlet, KARMA::PHYSICS::Dirichlet );
            dg_manager.refresh( mesh, dg_system, parameters );
            const auto dg_id = mesh.side_id( element, 0 );
            check( dg_manager.dirichlet_dofs().empty() &&
                       dg_manager.get_dirichlet_condition( dg_id.element, dg_id.side ) != nullptr,
                   "dG Dirichlet conditions remain weak and side-based" );

            KARMA::KARMABasisRegistry registry;
            auto basis = registry.create( { KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::LagrangeNodal,
                                            1, KARMA::BasisContinuity::cG } );
            KARMA::KARMAFEBase fe( mesh, KARMA::GeometryVariant::Quad4, std::move( basis ), 3,
                                   KARMA::FEIntegrationDomain::Face );
            fe.reinit( element, 0 );
            KARMABoundaryCondition neumann_condition( KARMA::PHYSICS::Neumann );
            neumann_condition.set( 2 );
            KARMABoundaryCondition robin_condition( KARMA::PHYSICS::Robin );
            robin_condition.set( 0, 2 );
            robin_condition.set( 1, 6 );
            robin_condition.set( 2, 10 );
            const auto load        = KARMAPhysicsBCManager::integrate_neumann_load( fe, neumann_condition );
            const auto robin_terms = KARMAPhysicsBCManager::integrate_robin( fe, robin_condition );
            check( std::abs( load( 0 ) - 1 ) < 1.e-12 && std::abs( load( 1 ) - 1 ) < 1.e-12,
                   "outward Neumann flux g contributes positive integral phi_i g" );
            check( std::abs( robin_terms.load( 0 ) - 2.5 ) < 1.e-12 &&
                       std::abs( robin_terms.load( 1 ) - 2.5 ) < 1.e-12 &&
                       std::abs( robin_terms.matrix( 0, 0 ) - 1 ) < 1.e-12 &&
                       std::abs( robin_terms.matrix( 0, 1 ) - 0.5 ) < 1.e-12 &&
                       std::abs( robin_terms.matrix( 1, 0 ) - 0.5 ) < 1.e-12 &&
                       std::abs( robin_terms.matrix( 1, 1 ) - 1 ) < 1.e-12,
                   "flux-form Robin a*(k grad(u).n)+b*u=c contributes b/a matrix and c/a load" );
            robin_condition.set( 0, 0 );
            bool zero_a_rejected = false;
            try
            {
                (void)KARMAPhysicsBCManager::integrate_robin( fe, robin_condition );
            }
            catch( const std::logic_error& )
            {
                zero_a_rejected = true;
            }
            check( zero_a_rejected, "Robin condition rejects a=0" );

            KARMA::KARMABasisRegistry p2_registry;
            auto p2_basis =
                p2_registry.create( { KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::LagrangeNodal, 2,
                                      KARMA::BasisContinuity::cG } );
            KARMA::KARMAFEBase p2_fe( mesh, KARMA::GeometryVariant::Quad4, std::move( p2_basis ), 5,
                                      KARMA::FEIntegrationDomain::Face );
            p2_fe.reinit( element, 0 );
            neumann_condition.set( 2 );
            robin_condition.set( 0, 2 );
            robin_condition.set( 1, 6 );
            robin_condition.set( 2, 10 );
            const auto p2_load  = KARMAPhysicsBCManager::integrate_neumann_load( p2_fe, neumann_condition );
            const auto p2_robin = KARMAPhysicsBCManager::integrate_robin( p2_fe, robin_condition );
            const std::array< Real, 9 > expected_load{ 1.0 / 3, 1.0 / 3, 0, 0, 4.0 / 3, 0, 0, 0, 0 };
            const std::array< std::array< Real, 3 >, 3 > edge_mass{
                std::array< Real, 3 >{ 2.0 / 5, -1.0 / 10, 1.0 / 5 },
                std::array< Real, 3 >{ -1.0 / 10, 2.0 / 5, 1.0 / 5 }, std::array< Real, 3 >{ 1.0 / 5, 1.0 / 5, 8.0 / 5 }
            };
            const std::array< std::size_t, 3 > side_basis{ 0, 1, 4 };
            for( std::size_t i = 0; i < expected_load.size(); ++i )
            {
                check( std::abs( p2_load( i ) - expected_load[i] ) < 1.e-12,
                       "P2 Neumann load has the expected value in every entry" );
                check( std::abs( p2_robin.load( i ) - 2.5 * expected_load[i] ) < 1.e-12,
                       "P2 Robin load has the expected value in every entry" );
                for( std::size_t j = 0; j < expected_load.size(); ++j )
                {
                    Real expected     = 0;
                    const auto row    = std::find( side_basis.begin(), side_basis.end(), i );
                    const auto column = std::find( side_basis.begin(), side_basis.end(), j );
                    if( row != side_basis.end() && column != side_basis.end() )
                        expected = edge_mass[static_cast< std::size_t >( row - side_basis.begin() )]
                                            [static_cast< std::size_t >( column - side_basis.begin() )];
                    check( std::abs( p2_robin.matrix( i, j ) - expected ) < 1.e-12,
                           "P2 Robin matrix has the expected value in every entry" );
                }
            }

            KARMA::KARMAMesh nonmanifold( 2, PETSC_COMM_WORLD );
            const double coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0.5, 1, 0 };
            moab::Range vertices;
            nonmanifold.moab()->create_vertices( coordinates, 5, vertices );
            const std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
            const moab::EntityHandle connectivity[][3] = { { handles[0], handles[1], handles[2] },
                                                           { handles[1], handles[0], handles[3] },
                                                           { handles[0], handles[1], handles[4] } };
            for( const auto& nodes : connectivity )
            {
                moab::EntityHandle triangle = 0;
                nonmanifold.moab()->create_element( moab::MBTRI, nodes, 3, triangle );
            }
            nonmanifold.assign_ids();
            nonmanifold.update();
            KARMA::KARMASystem nonmanifold_system( nonmanifold, "nonmanifold" );
            nonmanifold_system.add_variable( "u" );
            nonmanifold_system.init();
            check( throws_partition_identity( [&] {
                       KARMAParameters p;
                       p.set< Real >( "time" ) = 0;
                       manager.refresh( nonmanifold, nonmanifold_system, p );
                   } ) == false,
                   "non-manifold rejection is distinct from partition-identity rejection" );
            bool rejected = false;
            try
            {
                KARMAParameters p;
                p.set< Real >( "time" ) = 0;
                manager.refresh( nonmanifold, nonmanifold_system, p );
            }
            catch( const std::logic_error& error )
            {
                rejected = std::string_view( error.what() ).find( "non-manifold" ) != std::string_view::npos;
            }
            check( rejected, "boundary discovery rejects non-manifold sides" );
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    MPI_Comm_free( &KARMA_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
