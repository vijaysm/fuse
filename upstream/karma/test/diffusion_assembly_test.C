#include "diffusion_reaction_assembly.h"
#include "karma_fe_base.h"
#include "karma_moab_io.h"
#include "karma_parameters.h"
#include "karma_physics_bc_manager.h"
#include "karma_reference_topology.h"
#include "karma_system.h"
#include "petsc_native_scatter.h"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace
{
int failures = 0;
void check( bool condition, const char* message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void require( bool condition, const char* message )
{
    check( condition, message );
    if( !condition ) throw std::runtime_error( message );
}

std::unique_ptr< KARMABoundaryCondition > boundary_conditions( const KARMAParameters& parameters )
{
    const auto type = parameters.get< unsigned >( "side_id" );
    if( type == 0 )
    {
        auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
        condition->set( 2 );
        return condition;
    }
    return nullptr;
}

std::unique_ptr< KARMABoundaryCondition > neumann_condition( const KARMAParameters& parameters )
{
    if( parameters.get< unsigned >( "side_id" ) != 1 ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Neumann );
    condition->set( 2 );
    return condition;
}

std::unique_ptr< KARMABoundaryCondition > replicated_dirichlet_condition( const KARMAParameters& )
{
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
    condition->set( 2 );
    return condition;
}

std::unique_ptr< KARMABoundaryCondition > asymmetric_dirichlet_condition( const KARMAParameters& )
{
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet, true );
    condition->set( 0, rank == 1 ? 3 : 2 );
    condition->set( 1, 4 );
    return condition;
}

std::unique_ptr< KARMABoundaryCondition > robin_condition( const KARMAParameters& parameters )
{
    if( parameters.get< unsigned >( "side_id" ) != 2 ) return nullptr;
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Robin );
    condition->set( 0, 2 );
    condition->set( 1, 6 );
    condition->set( 2, 10 );
    return condition;
}

void discover_replicated_boundary( KARMAPhysicsBCManager& boundary, const KARMA::KARMAMesh& mesh,
                                   const KARMA::KARMASystem& system, KARMAParameters& parameters )
{
    for( const auto element : mesh.elements() )
        boundary.update_boundary_conditions( mesh, element, system, parameters );
}

bool throws_with( auto&& function, std::string_view text, std::string_view additional_text = {} )
{
    try
    {
        function();
    }
    catch( const std::logic_error& error )
    {
        const std::string_view message( error.what() );
        return message.find( text ) != std::string_view::npos &&
               ( additional_text.empty() || message.find( additional_text ) != std::string_view::npos );
    }
    return false;
}

void check_low_level_continuity_rejection( KARMA::KARMAMesh& mesh, KARMA::BasisContinuity continuity )
{
    KARMA::KARMASystem system( mesh, "unsupported-diffusion-continuity" );
    system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 1, continuity, 1 } );
    system.init();
    const auto element = *mesh.elements().begin();
    const KARMA::DiffusionReactionCoefficients coefficients;

    check( throws_with( [&] { (void)KARMA::element_diffusion_reaction( system, element, coefficients ); },
                        "interior diffusion fluxes are not implemented" ),
           "element diffusion assembly rejects dG/SE without interior fluxes" );

    Vec solution = system.create_vector();
    Vec residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY );
    check( throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients ); },
                        "interior diffusion fluxes are not implemented" ),
           "global diffusion assembly rejects dG/SE without interior fluxes" );
    PetscReal residual_norm = -1;
    PetscReal jacobian_norm = -1;
    const PetscErrorCode residual_error = VecNorm( residual, NORM_2, &residual_norm );
    const PetscErrorCode jacobian_error = MatNorm( jacobian, NORM_FROBENIUS, &jacobian_norm );
    const PetscErrorCode residual_usable = VecSet( residual, 1 );
    const PetscErrorCode jacobian_usable = MatZeroEntries( jacobian );
    check( residual_error == PETSC_SUCCESS && jacobian_error == PETSC_SUCCESS && residual_usable == PETSC_SUCCESS &&
               jacobian_usable == PETSC_SUCCESS && residual_norm == 0 && jacobian_norm == 0,
           "rejected dG/SE assembly leaves matrix and vector zero and usable" );
    MatDestroy( &jacobian );
    VecDestroy( &residual );
    VecDestroy( &solution );
}

struct NaturalBoundaryTotals
{
    unsigned local_records = 0;
    unsigned local_physical_sides = 0;
    std::string signature;
};

struct NaturalBoundaryReference
{
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
        check( keys.size() == indices.size(), "diffusion keys and indices have equal sizes" );
        for( std::size_t i = 0; i < keys.size(); ++i )
        {
            const auto [found, inserted] = result.emplace( keys[i], indices[i] );
            check( inserted || found->second == indices[i], "stable diffusion key has one index" );
        }
    }
    return result;
}

std::map< KARMA::DofKey, PetscInt > global_dof_indices( const KARMA::KARMASystem& system )
{
    const auto visible = visible_dof_indices( system );
    std::vector< std::uint64_t > local_words;
    for( const auto& [key, index] : visible )
    {
        local_words.insert( local_words.end(), key.begin(), key.end() );
        local_words.push_back( static_cast< std::uint64_t >( index ) );
    }
    int ranks = 1;
    check( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ) == MPI_SUCCESS, "query diffusion rank count" );
    const int local_count = static_cast< int >( local_words.size() );
    std::vector< int > counts( static_cast< std::size_t >( ranks ) ), offsets( static_cast< std::size_t >( ranks ) );
    check( MPI_Allgather( &local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather diffusion stable-key counts" );
    for( int rank = 1; rank < ranks; ++rank ) offsets[rank] = offsets[rank - 1] + counts[rank - 1];
    std::vector< std::uint64_t > gathered( static_cast< std::size_t >( offsets.back() + counts.back() ) );
    check( MPI_Allgatherv( local_words.empty() ? nullptr : local_words.data(), local_count, MPI_UINT64_T,
                           gathered.empty() ? nullptr : gathered.data(), counts.data(), offsets.data(), MPI_UINT64_T,
                           PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather diffusion stable-key records" );
    std::map< KARMA::DofKey, PetscInt > result;
    constexpr std::size_t record_width = KARMA::DofKey::width + 1;
    check( gathered.size() % record_width == 0, "diffusion stable-key records are complete" );
    for( std::size_t offset = 0; offset < gathered.size(); offset += record_width )
    {
        KARMA::DofKey key;
        std::copy_n( gathered.begin() + static_cast< std::ptrdiff_t >( offset ), KARMA::DofKey::width, key.begin() );
        const auto index = static_cast< PetscInt >( gathered[offset + KARMA::DofKey::width] );
        const auto [found, inserted] = result.emplace( key, index );
        check( inserted || found->second == index, "partitioned diffusion key maps to one index" );
    }
    return result;
}

NaturalBoundaryReference read_natural_boundary_reference( const char* filename )
{
    std::ifstream input( filename );
    NaturalBoundaryReference reference;
    std::size_t dof_count = 0;
    check( input.good() && static_cast< bool >( input >> dof_count ), "open diffusion natural reference" );
    reference.keys.resize( dof_count );
    for( auto& key : reference.keys )
        for( auto& word : key.words ) check( static_cast< bool >( input >> word ), "read diffusion DoF key" );
    reference.residual.resize( dof_count );
    for( auto& value : reference.residual ) check( static_cast< bool >( input >> value ), "read diffusion residual" );
    reference.matrix.resize( dof_count * dof_count );
    for( auto& value : reference.matrix ) check( static_cast< bool >( input >> value ), "read diffusion Jacobian" );
    return reference;
}

NaturalBoundaryTotals assemble_natural_boundary( const char* filename, bool partitioned,
                                                  const char* reference_filename = nullptr, bool mutate = false )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh(
        filename, partitioned ? "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" : "" );
    if( partitioned )
        require( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
                 "partitioned diffusion fixture exchanges ghost elements" );
    mesh.update();
    KARMA::KARMASystem system( mesh, "partitioned-natural-diffusion" );
    system.add_variable( "u" );
    system.init();
    KARMAPhysicsBCManager boundary( 0 );
    boundary.attach_function( neumann_condition, KARMA::PHYSICS::Neumann );
    boundary.attach_function( robin_condition, KARMA::PHYSICS::Robin );
    KARMAParameters parameters;
    parameters.set< Real >( "time" ) = 0;
    boundary.refresh( mesh, system, parameters );
    NaturalBoundaryTotals result;
    result.local_records = boundary.n_neumann_nodes() + boundary.n_robin_nodes();
    for( const auto element : mesh.owned_elements() )
        for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
            if( mesh.is_boundary_side( element, side ) ) ++result.local_physical_sides;
    require( result.local_records <= result.local_physical_sides,
             "natural boundary records are restricted to physical sides, not partition interfaces" );
    result.signature = boundary.assembly_signature( system );
    Vec solution = system.create_vector(), residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    require( VecSet( solution, 2 ) == PETSC_SUCCESS, "partitioned diffusion initializes solution" );
    KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian,
                                        KARMA::DiffusionReactionCoefficients{}, &boundary );
    if( reference_filename )
    {
        const NaturalBoundaryReference reference = read_natural_boundary_reference( reference_filename );
        std::map< KARMA::DofKey, PetscInt > reference_indices;
        for( std::size_t i = 0; i < reference.keys.size(); ++i )
            require( reference_indices.emplace( reference.keys[i], static_cast< PetscInt >( i ) ).second,
                     "serial diffusion reference keys are unique" );
        const auto partitioned_indices = global_dof_indices( system );
        require( partitioned_indices.size() == reference.keys.size(),
                 "partitioned diffusion exposes every serial stable DoF" );
        PetscInt first = 0, last = 0;
        require( VecGetOwnershipRange( residual, &first, &last ) == PETSC_SUCCESS,
                 "get owned partitioned diffusion residual rows" );
        std::vector< PetscInt > columns;
        for( const auto& key : reference.keys ) columns.push_back( partitioned_indices.at( key ) );
        for( const auto& [key, row] : partitioned_indices )
        {
            if( row < first || row >= last ) continue;
            const auto reference_row = static_cast< std::size_t >( reference_indices.at( key ) );
            PetscScalar residual_value = 0;
            require( VecGetValues( residual, 1, &row, &residual_value ) == PETSC_SUCCESS,
                     "read owned partitioned diffusion residual" );
            require( !PetscIsInfOrNanScalar( residual_value ), "partitioned diffusion residual is finite" );
            require( std::abs( PetscRealPart( residual_value ) - reference.residual[reference_row] ) < 1.e-12,
                     "partitioned diffusion residual equals serial stable-key reference" );
            std::vector< PetscScalar > values( columns.size() );
            require( MatGetValues( jacobian, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(), values.data() ) ==
                         PETSC_SUCCESS,
                     "read owned partitioned diffusion Jacobian row" );
            for( std::size_t column = 0; column < values.size(); ++column )
            {
                require( !PetscIsInfOrNanScalar( values[column] ), "partitioned diffusion Jacobian is finite" );
                require( std::abs( PetscRealPart( values[column] ) -
                                   reference.matrix[reference_row * columns.size() + column] ) < 1.e-12,
                         "partitioned diffusion Jacobian equals serial stable-key reference" );
            }
        }

        if( mutate )
        {
            Vec expected_neumann = system.create_vector();
            for( const auto element : system.dof_map().assembly_elements() )
            {
                std::vector< PetscInt > dofs;
                system.dof_map().dof_indices( element, dofs );
                const moab::EntityHandle* connectivity = nullptr;
                int connectivity_size = 0;
                require( mesh.moab()->get_connectivity( element, connectivity, connectivity_size ) == moab::MB_SUCCESS,
                         "read diffusion element connectivity for expected Neumann load" );
                const auto geometry = KARMA::KARMAReferenceTopology::from_moab(
                    mesh.moab()->type_from_handle( element ), static_cast< std::size_t >( connectivity_size ) );
                const auto& spec = system.variable( 0 );
                KARMA::KARMAFEBase face(
                    mesh, geometry,
                    system.basis_registry().create(
                        { KARMA::reference_cell( geometry ), spec.family, spec.order, spec.continuity } ),
                    5, KARMA::FEIntegrationDomain::Face );
                for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
                {
                    if( !mesh.is_boundary_side( element, side ) ) continue;
                    const auto id = mesh.side_id( element, side );
                    const auto* condition = boundary.get_neumann_condition( id.element, id.side );
                    if( !condition ) continue;
                    face.reinit( element, side );
                    KARMA::scatter( expected_neumann,
                                    KARMAPhysicsBCManager::integrate_neumann_load( face, *condition ), dofs,
                                    ADD_VALUES );
                }
            }
            require( VecAssemblyBegin( expected_neumann ) == PETSC_SUCCESS &&
                         VecAssemblyEnd( expected_neumann ) == PETSC_SUCCESS,
                     "assemble expected Neumann contribution" );

            PetscReal expected_neumann_norm = 0;
            require( VecNorm( expected_neumann, NORM_INFINITY, &expected_neumann_norm ) == PETSC_SUCCESS &&
                         std::isfinite( expected_neumann_norm ) && expected_neumann_norm > 1.e-12,
                     "expected Neumann contribution is finite and nonzero" );
            unsigned global_neumann_records = boundary.n_neumann_nodes();
            require( MPI_Allreduce( MPI_IN_PLACE, &global_neumann_records, 1, MPI_UNSIGNED, MPI_SUM,
                                    PETSC_COMM_WORLD ) == MPI_SUCCESS,
                     "count discovered Neumann records" );

            KARMAPhysicsBCManager mutant_boundary( 0 );
            mutant_boundary.attach_function( robin_condition, KARMA::PHYSICS::Robin );
            mutant_boundary.refresh( mesh, system, parameters );
            require( mutant_boundary.n_neumann_nodes() == 0 && global_neumann_records > 0,
                     "test-only mutation suppresses discovered Neumann data before assembly" );
            Vec mutant_residual = system.create_vector();
            Mat mutant_jacobian = system.create_matrix();
            KARMA::assemble_diffusion_reaction( system, solution, mutant_residual, mutant_jacobian,
                                                KARMA::DiffusionReactionCoefficients{}, &mutant_boundary );

            bool local_affected_difference = false;
            bool local_reference_mismatch = false;
            for( const auto& [key, row] : partitioned_indices )
            {
                if( row < first || row >= last ) continue;
                const auto reference_row = static_cast< std::size_t >( reference_indices.at( key ) );
                PetscScalar normal_value = 0, mutant_value = 0, neumann_value = 0;
                require( VecGetValues( residual, 1, &row, &normal_value ) == PETSC_SUCCESS &&
                             VecGetValues( mutant_residual, 1, &row, &mutant_value ) == PETSC_SUCCESS &&
                             VecGetValues( expected_neumann, 1, &row, &neumann_value ) == PETSC_SUCCESS,
                         "read normal, mutant, and expected Neumann residual entries" );
                require( !PetscIsInfOrNanScalar( mutant_value ), "Neumann-suppressed residual is finite" );
                const PetscReal difference = PetscRealPart( mutant_value - normal_value );
                const PetscReal expected = PetscRealPart( neumann_value );
                require( std::abs( difference - expected ) < 1.e-12,
                         "mutant residual difference equals the suppressed Neumann contribution" );
                if( std::abs( expected ) >= 1.e-12 )
                {
                    local_affected_difference = local_affected_difference || std::abs( difference ) >= 1.e-12;
                    local_reference_mismatch = local_reference_mismatch ||
                        std::abs( PetscRealPart( mutant_value ) - reference.residual[reference_row] ) >= 1.e-12;
                }

                std::vector< PetscScalar > normal_values( columns.size() ), mutant_values( columns.size() );
                require( MatGetValues( jacobian, 1, &row, static_cast< PetscInt >( columns.size() ), columns.data(),
                                       normal_values.data() ) == PETSC_SUCCESS &&
                             MatGetValues( mutant_jacobian, 1, &row, static_cast< PetscInt >( columns.size() ),
                                           columns.data(), mutant_values.data() ) == PETSC_SUCCESS,
                         "read normal and Neumann-suppressed Jacobian rows" );
                for( std::size_t column = 0; column < mutant_values.size(); ++column )
                {
                    require( !PetscIsInfOrNanScalar( mutant_values[column] ),
                             "Neumann-suppressed Jacobian is finite" );
                    require( std::abs( PetscRealPart( mutant_values[column] - normal_values[column] ) ) < 1.e-12,
                             "suppressing a pure Neumann load does not change the Jacobian" );
                }
            }
            int discrepancies[] = { local_affected_difference ? 1 : 0, local_reference_mismatch ? 1 : 0 };
            require( MPI_Allreduce( MPI_IN_PLACE, discrepancies, 2, MPI_INT, MPI_MAX, PETSC_COMM_WORLD ) == MPI_SUCCESS,
                     "reduce Neumann-suppression mutation discrepancies" );
            require( discrepancies[0] == 1,
                     "Neumann suppression changes an affected boundary residual DoF from normal" );
            require( discrepancies[1] == 1,
                     "Neumann-suppressed residual specifically mismatches the serial reference" );
            MatDestroy( &mutant_jacobian );
            VecDestroy( &mutant_residual );
            VecDestroy( &expected_neumann );
        }
    }
    MatDestroy( &jacobian );
    VecDestroy( &residual );
    VecDestroy( &solution );
    return result;
}

void write_natural_boundary_reference( const char* filename, const char* reference_filename )
{
    std::ofstream output( reference_filename );
    check( output.good(), "open serial natural boundary reference" );
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMA::KARMAMoabIO( mesh ).read_mesh( filename );
    mesh.update();
    KARMA::KARMASystem system( mesh, "serial-natural-diffusion-reference" );
    system.add_variable( "u" );
    system.init();
    KARMAPhysicsBCManager boundary( 0 );
    boundary.attach_function( neumann_condition, KARMA::PHYSICS::Neumann );
    boundary.attach_function( robin_condition, KARMA::PHYSICS::Robin );
    KARMAParameters parameters;
    parameters.set< Real >( "time" ) = 0;
    boundary.refresh( mesh, system, parameters );
    Vec solution = system.create_vector(), residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    VecSet( solution, 2 );
    KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian,
                                        KARMA::DiffusionReactionCoefficients{}, &boundary );
    PetscReal residual_norm = 0, jacobian_norm = 0;
    check( VecNorm( residual, NORM_INFINITY, &residual_norm ) == PETSC_SUCCESS &&
               MatNorm( jacobian, NORM_INFINITY, &jacobian_norm ) == PETSC_SUCCESS && residual_norm > 1.e-12 &&
               jacobian_norm > 1.e-12,
           "serial natural boundary fixture has nonzero analytic contributions" );
    const auto indices = visible_dof_indices( system );
    output << indices.size() << '\n' << std::setprecision( 17 );
    for( const auto& [key, index] : indices )
    {
        (void)index;
        for( const auto word : key.words ) output << word << ' ';
        output << '\n';
    }
    for( const auto& [key, index] : indices )
    {
        (void)key;
        PetscScalar value = 0;
        check( VecGetValues( residual, 1, &index, &value ) == PETSC_SUCCESS, "read serial diffusion residual" );
        output << PetscRealPart( value ) << '\n';
    }
    for( const auto& [row_key, row] : indices )
        for( const auto& [column_key, column] : indices )
        {
            (void)row_key;
            (void)column_key;
            PetscScalar value = 0;
            check( MatGetValues( jacobian, 1, &row, 1, &column, &value ) == PETSC_SUCCESS,
                   "read serial diffusion Jacobian" );
            output << PetscRealPart( value ) << '\n';
        }
    MatDestroy( &jacobian );
    VecDestroy( &residual );
    VecDestroy( &solution );
}

void verify_partitioned_natural_boundary( const char* filename, const char* reference_filename, bool mutate = false )
{
    const auto totals = assemble_natural_boundary( filename, true, reference_filename, mutate );
    int ranks = 1;
    MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
    const int local_size = static_cast< int >( totals.signature.size() );
    std::vector< int > sizes( static_cast< std::size_t >( ranks ) ), offsets( static_cast< std::size_t >( ranks ) );
    check( MPI_Allgather( &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather owner-local natural boundary signature sizes" );
    for( int rank = 1; rank < ranks; ++rank ) offsets[rank] = offsets[rank - 1] + sizes[rank - 1];
    std::string signatures( static_cast< std::size_t >( offsets.back() + sizes.back() ), '\0' );
    check( MPI_Allgatherv( totals.signature.empty() ? nullptr : totals.signature.data(), local_size, MPI_CHAR,
                           signatures.empty() ? nullptr : signatures.data(), sizes.data(), offsets.data(), MPI_CHAR,
                           PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather owner-local natural boundary signatures" );
    bool unequal = false;
    for( int rank = 1; rank < ranks; ++rank )
        unequal = unequal || sizes[rank] != sizes[0] ||
                  signatures.compare( offsets[rank], sizes[rank], signatures, 0, sizes[0] ) != 0;
    check( unequal, "partitioned physical natural boundaries produce unequal owner-local records across ranks" );
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if( mode == "--replicated-dirichlet-boundary" )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
            mesh.generate_box( 1, 1, 0 );
            KARMA::KARMASystem system( mesh, "replicated-dirichlet-diffusion" );
            system.add_variable( "u" );
            system.init();

            KARMAPhysicsBCManager boundary( 0 );
            boundary.attach_function( replicated_dirichlet_condition, KARMA::PHYSICS::Dirichlet );
            KARMAParameters parameters;
            parameters.set< Real >( "time" ) = 0;
            discover_replicated_boundary( boundary, mesh, system, parameters );

            Vec solution = system.create_vector(), residual = system.create_vector();
            Mat jacobian = system.create_matrix();
            VecSet( solution, 3 );
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian,
                                                KARMA::DiffusionReactionCoefficients{}, &boundary );

            PetscInt first = 0, last = 0;
            check( VecGetOwnershipRange( solution, &first, &last ) == PETSC_SUCCESS,
                   "replicated Dirichlet test gets solution ownership range" );
            int local_nonlocal = 0, local_checked = 0;
            for( const PetscInt dof : boundary.dirichlet_dofs() )
            {
                if( dof < first || dof >= last )
                {
                    local_nonlocal = 1;
                    continue;
                }
                PetscScalar constrained_residual = 0;
                check( VecGetValues( residual, 1, &dof, &constrained_residual ) == PETSC_SUCCESS,
                       "owner reads constrained residual" );
                check( std::abs( PetscRealPart( constrained_residual ) - 1 ) < 1.e-12,
                       "constrained residual globally equals u-g" );
                for( PetscInt column = 0; column < system.n_dofs(); ++column )
                {
                    PetscScalar entry = 0;
                    check( MatGetValues( jacobian, 1, &dof, 1, &column, &entry ) == PETSC_SUCCESS,
                           "owner reads constrained Jacobian row" );
                    const Real expected = column == dof ? 1 : 0;
                    check( std::abs( PetscRealPart( entry ) - expected ) < 1.e-12,
                           "constrained Jacobian rows are identity" );
                }
                ++local_checked;
            }
            int ranks_with_nonlocal = 0, globally_checked = 0;
            MPI_Allreduce( &local_nonlocal, &ranks_with_nonlocal, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
            MPI_Allreduce( &local_checked, &globally_checked, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
            check( ranks_with_nonlocal > 0, "at least one rank observes a nonlocal constrained DoF" );
            check( globally_checked == static_cast< int >( boundary.dirichlet_dofs().size() ),
                   "every constrained residual and Jacobian row is checked by its owner" );

            MatDestroy( &jacobian );
            VecDestroy( &residual );
            VecDestroy( &solution );
        }
        else if( mode == "--write-partitioned-natural-reference" )
        {
            check( argc == 4, "serial natural boundary reference requires mesh and reference paths" );
            if( argc == 4 ) write_natural_boundary_reference( argv[2], argv[3] );
        }
        else if( mode == "--partitioned-natural-boundary" || mode == "--partitioned-natural-boundary-mutation" )
        {
            check( argc == 4, "partitioned diffusion mode requires mesh and serial reference paths" );
            if( argc == 4 )
                verify_partitioned_natural_boundary( argv[2], argv[3], mode == "--partitioned-natural-boundary-mutation" );
        }
        else if( mode == "--replicated-natural-boundary" )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
            mesh.generate_box( 1, 1, 0 );
            KARMA::KARMASystem system( mesh, "replicated-natural-diffusion" );
            system.add_variable( "u" );
            system.init();

            KARMAPhysicsBCManager boundary( 0 );
            boundary.attach_function( neumann_condition, KARMA::PHYSICS::Neumann );
            boundary.attach_function( robin_condition, KARMA::PHYSICS::Robin );
            KARMAParameters parameters;
            parameters.set< Real >( "time" ) = 0;
            discover_replicated_boundary( boundary, mesh, system, parameters );

            const auto element = *mesh.elements().begin();
            const KARMA::GlobalId private_id = mesh.global_id( element );
            KARMA::GlobalId minimum_id = 0, maximum_id = 0;
            MPI_Allreduce( &private_id, &minimum_id, 1, MPI_INT, MPI_MIN, PETSC_COMM_WORLD );
            MPI_Allreduce( &private_id, &maximum_id, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD );
            check( minimum_id != maximum_id, "replicated ranks retain different private element IDs" );

            const std::string signature = boundary.assembly_signature( system );
            const int signature_size = static_cast< int >( signature.size() );
            int reference_size = signature_size;
            MPI_Bcast( &reference_size, 1, MPI_INT, 0, PETSC_COMM_WORLD );
            std::string reference( static_cast< std::size_t >( reference_size ), '\0' );
            if( rank == 0 ) reference = signature;
            MPI_Bcast( reference.data(), reference_size, MPI_CHAR, 0, PETSC_COMM_WORLD );
            check( signature == reference, "replicated natural boundary signatures use canonical element ordinals" );

            Vec solution = system.create_vector(), residual = system.create_vector();
            Mat jacobian = system.create_matrix();
            VecSet( solution, 2 );
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian,
                                                KARMA::DiffusionReactionCoefficients{}, &boundary );
            MatDestroy( &jacobian );
            VecDestroy( &residual );
            VecDestroy( &solution );
        }
        else if( mode.rfind( "--asymmetric-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
            const bool boundary_id_failure = mode == "--asymmetric-boundary-missing-id" ||
                                             mode == "--asymmetric-boundary-duplicate-id";
            mesh.generate_box( boundary_id_failure ? 2 : 1, 1, 0 );
            KARMA::KARMASystem system( mesh, "asymmetric-diffusion" );
            system.add_variable( "u" );
            system.init();
            Vec solution = system.create_vector(), residual = system.create_vector();
            Mat jacobian = system.create_matrix();
            Vec bad_vector = nullptr;
            Mat bad_matrix = nullptr;
            const bool vector_failure = mode.find( "solution" ) != std::string_view::npos ||
                                        mode.find( "residual" ) != std::string_view::npos;
            if( vector_failure && mode.find( "null" ) == std::string_view::npos )
            {
                const PetscInt extra = mode.find( "size" ) != std::string_view::npos && rank == 1 ? 1 : 0;
                const MPI_Comm communicator = mode.find( "comm" ) != std::string_view::npos ? PETSC_COMM_SELF
                                                                                             : PETSC_COMM_WORLD;
                VecCreateMPI( communicator, system.n_local_dofs() + extra, PETSC_DECIDE, &bad_vector );
            }
            if( mode.find( "jacobian" ) != std::string_view::npos && mode.find( "presence" ) == std::string_view::npos )
            {
                const PetscInt extra = mode.find( "size" ) != std::string_view::npos && rank == 1 ? 1 : 0;
                const MPI_Comm communicator = mode.find( "comm" ) != std::string_view::npos ? PETSC_COMM_SELF
                                                                                             : PETSC_COMM_WORLD;
                MatCreateAIJ( communicator, system.n_local_dofs() + extra, system.n_local_dofs() + extra,
                              PETSC_DECIDE, PETSC_DECIDE, 1, nullptr, 0, nullptr, &bad_matrix );
                MatSetUp( bad_matrix );
            }
            constexpr PetscReal residual_sentinel = 37;
            constexpr PetscReal jacobian_sentinel = 41;
            check( VecSet( residual, residual_sentinel ) == PETSC_SUCCESS,
                   "asymmetric diffusion test initializes the residual sentinel" );
            check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
                   "asymmetric diffusion test begins initial Jacobian assembly" );
            check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
                   "asymmetric diffusion test ends initial Jacobian assembly" );
            check( MatShift( jacobian, jacobian_sentinel ) == PETSC_SUCCESS,
                   "asymmetric diffusion test initializes the Jacobian sentinel" );
            PetscReal initial_jacobian_norm = 0;
            check( MatNorm( jacobian, NORM_INFINITY, &initial_jacobian_norm ) == PETSC_SUCCESS &&
                       initial_jacobian_norm == jacobian_sentinel,
                   "asymmetric diffusion test observes the exact Jacobian sentinel before validation" );
            KARMAPhysicsBCManager boundary( 0 );
            const KARMAPhysicsBCManager* call_boundary = nullptr;
            KARMAParameters parameters;
            parameters.set< Real >( "time" ) = 0;
            if( mode == "--asymmetric-boundary-presence" )
            {
                boundary.attach_function( boundary_conditions, KARMA::PHYSICS::Dirichlet );
                discover_replicated_boundary( boundary, mesh, system, parameters );
                if( rank == 1 ) call_boundary = &boundary;
            }
            if( mode == "--asymmetric-boundary-dirichlet" )
            {
                boundary.attach_function( asymmetric_dirichlet_condition, KARMA::PHYSICS::Dirichlet );
                discover_replicated_boundary( boundary, mesh, system, parameters );
                call_boundary = &boundary;
            }
            if( mode == "--asymmetric-boundary-natural" )
            {
                boundary.attach_function( rank == 1 ? robin_condition : neumann_condition,
                                          rank == 1 ? KARMA::PHYSICS::Robin : KARMA::PHYSICS::Neumann );
                discover_replicated_boundary( boundary, mesh, system, parameters );
                call_boundary = &boundary;
            }
            if( boundary_id_failure )
            {
                boundary.attach_function( neumann_condition, KARMA::PHYSICS::Neumann );
                discover_replicated_boundary( boundary, mesh, system, parameters );
                call_boundary = &boundary;
                if( rank == 1 )
                {
                    const auto first = *mesh.elements().begin();
                    const auto second = *std::next( mesh.elements().begin() );
                    const KARMA::GlobalId replacement = mode == "--asymmetric-boundary-missing-id"
                                                            ? mesh.global_id( second ) + 1000
                                                            : mesh.global_id( first );
                    const auto changed = second;
                    check( mesh.moab()->tag_set_data( mesh.global_id_tag(), &changed, 1, &replacement ) ==
                               moab::MB_SUCCESS,
                           "asymmetric boundary test can corrupt rank-1 private element IDs" );
                }
            }
            Vec call_solution = solution, call_residual = residual;
            Mat call_jacobian = jacobian;
            if( rank == 1 && mode.find( "solution" ) != std::string_view::npos ) call_solution = bad_vector;
            if( rank == 1 && mode.find( "residual" ) != std::string_view::npos ) call_residual = bad_vector;
            if( rank == 1 && mode.find( "jacobian" ) != std::string_view::npos )
                call_jacobian = mode.find( "presence" ) != std::string_view::npos ? nullptr : bad_matrix;
            const std::string expected_error = mode == "--asymmetric-boundary-missing-id"
                                                   ? "missing private element ID"
                                               : mode == "--asymmetric-boundary-duplicate-id"
                                                   ? "duplicate private element ID"
                                                   : "rank";
            const std::string rank_error = boundary_id_failure ? "rank 1" : "";
            check( throws_with( [&] { KARMA::assemble_diffusion_reaction(
                                         system, call_solution, call_residual, call_jacobian,
                                         KARMA::DiffusionReactionCoefficients{}, call_boundary ); }, expected_error,
                                rank_error ),
                   "asymmetric diffusion object validation fails collectively" );
            PetscReal residual_norm = 0, jacobian_norm = 0;
            const PetscErrorCode residual_norm_error = VecNorm( residual, NORM_INFINITY, &residual_norm );
            const PetscErrorCode jacobian_norm_error = MatNorm( jacobian, NORM_INFINITY, &jacobian_norm );
            check( residual_norm_error == PETSC_SUCCESS && jacobian_norm_error == PETSC_SUCCESS &&
                       residual_norm == residual_sentinel && jacobian_norm == jacobian_sentinel,
                   "asymmetric diffusion validation leaves exact sentinels unchanged" );
            MatDestroy( &bad_matrix );
            VecDestroy( &bad_vector );
            MatDestroy( &jacobian );
            VecDestroy( &residual );
            VecDestroy( &solution );
        }
        else
        {
        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        check_low_level_continuity_rejection( mesh, KARMA::BasisContinuity::dG );
        check_low_level_continuity_rejection( mesh, KARMA::BasisContinuity::SE );

        KARMA::KARMASystem vector_system( mesh, "vector-diffusion" );
        vector_system.add_variable(
            { "u", KARMA::FieldBasisFamily::LagrangeNodal, 1, KARMA::BasisContinuity::cG, 2 } );
        vector_system.init();
        check( throws_with(
                   [&] {
                       (void)KARMA::element_diffusion_reaction( vector_system, *mesh.elements().begin(),
                                                               KARMA::DiffusionReactionCoefficients{} );
                   },
                   "one scalar variable" ),
               "element diffusion assembly validates variable and component bounds" );
        Vec vector_solution = vector_system.create_vector();
        Vec vector_residual = vector_system.create_vector();
        check( throws_with(
                   [&] {
                       KARMA::assemble_diffusion_reaction( vector_system, vector_solution, vector_residual, nullptr,
                                                          KARMA::DiffusionReactionCoefficients{} );
                   },
                   "one scalar variable" ),
               "global diffusion assembly validates variable and component bounds" );
        VecDestroy( &vector_residual );
        VecDestroy( &vector_solution );

        KARMA::KARMASystem system( mesh, "diffusion" );
        system.add_variable( "u" );
        system.init();
        const auto element = *mesh.elements().begin();

        KARMA::DiffusionReactionCoefficients coefficients;
        coefficients.diffusion = []( const KARMA::Point&, Real, int material ) { return material == 7 ? 2.0 : 1.0; };
        coefficients.reaction  = []( const KARMA::Point&, Real, int ) { return 0.0; };
        coefficients.source    = []( const KARMA::Point&, Real, int ) { return 0.0; };

        int material           = 7;
        moab::EntityHandle set = 0;
        moab::Tag tag          = nullptr;
        mesh.moab()->create_meshset( moab::MESHSET_SET, set );
        mesh.moab()->add_entities( set, &element, 1 );
        mesh.moab()->tag_get_handle( "MATERIAL_SET", 1, moab::MB_TYPE_INTEGER, tag,
                                     moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT );
        mesh.moab()->tag_set_data( tag, &set, 1, &material );

        const auto terms          = KARMA::element_diffusion_reaction( system, element, coefficients );
        const Real expected[4][4] = { { 4, -1, -2, -1 }, { -1, 4, -1, -2 }, { -2, -1, 4, -1 }, { -1, -2, -1, 4 } };
        for( std::size_t i = 0; i < 4; ++i )
            for( std::size_t j = 0; j < 4; ++j )
                check( std::abs( terms.jacobian( i, j ) - expected[i][j] / 3.0 ) < 1.e-12,
                       "material coefficient scales analytic unit-quad stiffness" );

        Vec solution = system.create_vector();
        Vec residual = system.create_vector();
        Mat jacobian = system.create_matrix();
        if( mesh.n_ranks() == 1 )
        {
            VecSet( solution, 3 );
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
            Vec plain = nullptr, wrong_size = nullptr;
            Mat plain_matrix = nullptr, wrong_size_matrix = nullptr;
            VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs(), system.n_dofs(), &plain );
            VecCreateMPI( PETSC_COMM_WORLD, system.n_local_dofs() + 1, system.n_dofs() + 1, &wrong_size );
            MatCreateAIJ( PETSC_COMM_WORLD, system.n_local_dofs(), system.n_local_dofs(), system.n_dofs(),
                          system.n_dofs(), 1, nullptr, 0, nullptr, &plain_matrix );
            MatSetUp( plain_matrix );
            MatCreateAIJ( PETSC_COMM_WORLD, system.n_local_dofs() + 1, system.n_local_dofs() + 1,
                          system.n_dofs() + 1, system.n_dofs() + 1, 1, nullptr, 0, nullptr, &wrong_size_matrix );
            MatSetUp( wrong_size_matrix );
            VecSet( residual, 37 );
            MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY );
            MatShift( jacobian, 41 );
            PetscReal expected_jacobian = 0;
            MatNorm( jacobian, NORM_INFINITY, &expected_jacobian );
            check( throws_with( [&] { KARMA::assemble_diffusion_reaction( system, nullptr, residual, jacobian,
                                                                          coefficients ); }, "null" ) &&
                       throws_with( [&] { KARMA::assemble_diffusion_reaction( system, wrong_size, residual, jacobian,
                                                                            coefficients ); }, "size" ) &&
                       throws_with( [&] { KARMA::assemble_diffusion_reaction( system, plain, residual, jacobian,
                                                                            coefficients ); }, "mapping" ),
                   "diffusion assembly rejects null, wrong-sized, and plain solution vectors" );
            check( throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, nullptr, jacobian,
                                                                          coefficients ); }, "null" ) &&
                       throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, wrong_size, jacobian,
                                                                            coefficients ); }, "size" ) &&
                       throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, plain, jacobian,
                                                                            coefficients ); }, "mapping" ),
                   "diffusion assembly rejects null, wrong-sized, and plain residual vectors" );
            check( throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, residual,
                                                                          wrong_size_matrix, coefficients ); },
                                "dimensions" ) &&
                       throws_with( [&] { KARMA::assemble_diffusion_reaction( system, solution, residual, plain_matrix,
                                                                            coefficients ); }, "mapping" ),
                   "diffusion assembly rejects wrong-sized and plain Jacobian matrices" );
            PetscReal retained_residual = 0, retained_jacobian = 0;
            VecNorm( residual, NORM_INFINITY, &retained_residual );
            MatNorm( jacobian, NORM_INFINITY, &retained_jacobian );
            check( std::abs( retained_residual - 37 ) < 1.e-12 &&
                       std::abs( retained_jacobian - expected_jacobian ) < 1.e-12,
                   "invalid diffusion objects leave outputs unchanged" );
            MatDestroy( &wrong_size_matrix );
            MatDestroy( &plain_matrix );
            VecDestroy( &wrong_size );
            VecDestroy( &plain );
        }
        VecSet( solution, 3 );
        KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
        PetscReal norm = 0;
        VecNorm( residual, NORM_2, &norm );
        check( norm < 1.e-12, "constant field has zero residual for zero source and reaction" );

        Vec direction          = system.create_vector();
        Vec perturbed          = system.create_vector();
        Vec perturbed_residual = system.create_vector();
        Vec jacobian_direction = system.create_vector();
        VecSetRandom( direction, nullptr );
        VecWAXPY( perturbed, 1.e-7, direction, solution );
        KARMA::assemble_diffusion_reaction( system, perturbed, perturbed_residual, nullptr, coefficients );
        VecAXPY( perturbed_residual, -1, residual );
        VecScale( perturbed_residual, 1.e7 );
        MatMult( jacobian, direction, jacobian_direction );
        VecAXPY( perturbed_residual, -1, jacobian_direction );
        VecNorm( perturbed_residual, NORM_2, &norm );
        check( norm < 1.e-7, "assembled Jacobian matches finite difference of residual" );

        coefficients.reaction = []( const KARMA::Point&, Real field, int ) { return field; };
        KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
        VecWAXPY( perturbed, 1.e-7, direction, solution );
        KARMA::assemble_diffusion_reaction( system, perturbed, perturbed_residual, nullptr, coefficients );
        VecAXPY( perturbed_residual, -1, residual );
        VecScale( perturbed_residual, 1.e7 );
        MatMult( jacobian, direction, jacobian_direction );
        VecAXPY( perturbed_residual, -1, jacobian_direction );
        VecNorm( perturbed_residual, NORM_2, &norm );
        check( norm < 1.e-5, "field-dependent reaction Jacobian matches residual finite difference" );

        coefficients.diffusion            = []( const KARMA::Point&, Real field, int ) { return field * field + 2; };
        coefficients.diffusion_derivative = []( const KARMA::Point&, Real field, int ) { return 2 * field; };
        coefficients.reaction             = []( const KARMA::Point&, Real field, int ) { return field * field; };
        coefficients.reaction_derivative  = []( const KARMA::Point&, Real field, int ) { return 2 * field; };
        coefficients.source               = []( const KARMA::Point&, Real field, int ) { return 3 * field; };
        coefficients.source_derivative    = []( const KARMA::Point&, Real, int ) { return 3; };
        for( const Real field : { Real{ 1.e-12 }, Real{ 1.e6 } } )
        {
            KARMA::RealDenseVector values( 4 );
            values.native().setConstant( field );
            const auto explicit_terms = KARMA::element_diffusion_reaction( system, element, coefficients, &values );
            coefficients.diffusion_derivative = {};
            coefficients.reaction_derivative  = {};
            coefficients.source_derivative    = {};
            const auto finite_difference_terms =
                KARMA::element_diffusion_reaction( system, element, coefficients, &values );
            check( explicit_terms.jacobian.native().isApprox( finite_difference_terms.jacobian.native(), 1.e-8 ),
                   "scale-aware central derivatives agree with explicit derivatives at small and large fields" );
            coefficients.diffusion_derivative = []( const KARMA::Point&, Real value, int ) { return 2 * value; };
            coefficients.reaction_derivative  = []( const KARMA::Point&, Real value, int ) { return 2 * value; };
            coefficients.source_derivative    = []( const KARMA::Point&, Real, int ) { return 3; };
        }
        coefficients.diffusion_derivative = []( const KARMA::Point&, Real, int ) -> Real {
            throw std::runtime_error( "derivative callback" );
        };
        coefficients.reaction_derivative = coefficients.diffusion_derivative;
        coefficients.source_derivative   = coefficients.diffusion_derivative;
        coefficients.source              = []( const KARMA::Point&, Real, int ) { return 0.0; };
        bool residual_without_derivatives = true;
        try
        {
            KARMA::assemble_diffusion_reaction( system, solution, residual, nullptr, coefficients );
        }
        catch( ... )
        {
            residual_without_derivatives = false;
        }
        check( residual_without_derivatives, "residual-only assembly does not evaluate derivative callbacks" );
        bool jacobian_uses_derivatives = false;
        try
        {
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
        }
        catch( const std::logic_error& error )
        {
            jacobian_uses_derivatives = std::string_view( error.what() ).find( "derivative callback" ) != std::string_view::npos;
        }
        check( jacobian_uses_derivatives, "Jacobian assembly reports derivative callback failure" );

        coefficients                    = {};
        coefficients.source = []( const KARMA::Point&, Real, int ) { return std::numeric_limits< Real >::infinity(); };
        bool nonfinite_rejected = false;
        try
        {
            (void)KARMA::element_diffusion_reaction( system, element, coefficients );
        }
        catch( const std::logic_error& )
        {
            nonfinite_rejected = true;
        }
        check( nonfinite_rejected, "non-finite coefficient callbacks are rejected" );

        coefficients        = {};
        int rank = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
        const int local_assembly_rank = system.dof_map().assembly_elements().empty() ? -1 : rank;
        int assembly_rank = -1;
        MPI_Allreduce( &local_assembly_rank, &assembly_rank, 1, MPI_INT, MPI_MAX, PETSC_COMM_WORLD );
        const bool inject_callback_failure = mesh.n_ranks() == 2 && rank == assembly_rank;
        coefficients.source = [inject_callback_failure]( const KARMA::Point&, Real, int ) -> Real {
            if( inject_callback_failure ) throw std::runtime_error( "asymmetric callback" );
            return 0;
        };
        if( mesh.n_ranks() == 2 )
        {
            bool collective_failure = false;
            try
            {
                KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
            }
            catch( const std::logic_error& error )
            {
                collective_failure = std::string_view( error.what() ).find(
                                         "rank " + std::to_string( assembly_rank ) ) != std::string_view::npos &&
                                     std::string_view( error.what() ).find( "asymmetric callback" ) != std::string_view::npos;
            }
            check( collective_failure, "asymmetric callback failure reaches every rank with failing-rank context" );
        }
        if( mesh.n_ranks() == 2 )
        {
            if( rank == assembly_rank )
            {
                moab::EntityHandle conflicting_set = 0;
                mesh.moab()->create_meshset( moab::MESHSET_SET, conflicting_set );
                const auto owned_element = system.dof_map().assembly_elements().front();
                mesh.moab()->add_entities( conflicting_set, &owned_element, 1 );
                int conflicting_material = 8;
                mesh.moab()->tag_set_data( tag, &conflicting_set, 1, &conflicting_material );
            }
            coefficients = {};
            bool collective_material_failure = false;
            try
            {
                KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
            }
            catch( const std::logic_error& error )
            {
                collective_material_failure =
                    std::string_view( error.what() ).find( "rank " + std::to_string( assembly_rank ) ) !=
                        std::string_view::npos &&
                    std::string_view( error.what() ).find( "multiple material sets" ) != std::string_view::npos;
            }
            check( collective_material_failure,
                   "asymmetric material failure reaches every rank with failing-rank context" );
        }
        if( mesh.n_ranks() == 1 )
        {
            coefficients.source = []( const KARMA::Point&, Real, int ) -> Real {
                throw std::runtime_error( "callback" );
            };
            VecSet( residual, 37 );
            MatZeroEntries( jacobian );
            MatShift( jacobian, 41 );
            PetscReal expected_callback_jacobian = 0;
            MatNorm( jacobian, NORM_INFINITY, &expected_callback_jacobian );
            bool callback_threw = false;
            try
            {
                KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
            }
            catch( const std::logic_error& error )
            {
                callback_threw = std::string_view( error.what() ).find( "rank 0" ) != std::string_view::npos &&
                                 std::string_view( error.what() ).find( "callback" ) != std::string_view::npos;
            }
            PetscScalar usable_sum            = 0;
            PetscReal retained_residual = 0, retained_jacobian = 0;
            VecNorm( residual, NORM_INFINITY, &retained_residual );
            MatNorm( jacobian, NORM_INFINITY, &retained_jacobian );
            const PetscErrorCode usable_error = VecSet( solution, 1 );
            const PetscErrorCode sum_error    = VecSum( solution, &usable_sum );
            check( callback_threw && usable_error == PETSC_SUCCESS && sum_error == PETSC_SUCCESS &&
                       std::isfinite( PetscRealPart( usable_sum ) ) && std::abs( retained_residual - 37 ) < 1.e-12 &&
                       std::abs( retained_jacobian - expected_callback_jacobian ) < 1.e-12,
                    "callback exceptions restore PETSc access and leave outputs unchanged" );

            coefficients = {};
            coefficients.diffusion = []( const KARMA::Point&, Real, int ) -> Real {
                throw std::runtime_error( "material callback" );
            };
            VecSet( residual, 43 );
            MatZeroEntries( jacobian );
            MatShift( jacobian, 47 );
            PetscReal expected_material_jacobian = 0;
            MatNorm( jacobian, NORM_INFINITY, &expected_material_jacobian );
            check( throws_with( [&] {
                       KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients );
                   }, "material callback" ),
                   "material callback failure is reported" );
            VecNorm( residual, NORM_INFINITY, &retained_residual );
            MatNorm( jacobian, NORM_INFINITY, &retained_jacobian );
            check( std::abs( retained_residual - 43 ) < 1.e-12 &&
                       std::abs( retained_jacobian - expected_material_jacobian ) < 1.e-12,
                   "material callback failure leaves outputs unchanged" );

        }

        coefficients           = {};
        coefficients.diffusion = []( const KARMA::Point&, Real, int ) { return 7; };
        KARMAPhysicsBCManager natural_boundary( 0 );
        natural_boundary.attach_function( neumann_condition, KARMA::PHYSICS::Neumann );
        natural_boundary.attach_function( robin_condition, KARMA::PHYSICS::Robin );
        KARMAParameters parameters;
        parameters.set< Real >( "time" ) = 0;
        if( mesh.n_ranks() == 1 )
        {
            natural_boundary.refresh( mesh, system, parameters );
            VecSet( solution, 2 );
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients, &natural_boundary );
            PetscScalar residual_sum = 0;
            VecSum( residual, &residual_sum );
            check( std::abs( PetscRealPart( residual_sum ) + 1.0 ) < 1.e-12,
                   "Neumann and flux-form Robin terms use documented outward-flux signs" );

            KARMAPhysicsBCManager essential_boundary( 0 );
            essential_boundary.attach_function( boundary_conditions, KARMA::PHYSICS::Dirichlet );
            essential_boundary.refresh( mesh, system, parameters );
            KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients,
                                                &essential_boundary );
            const PetscInt constrained = essential_boundary.dirichlet_dofs().front();
            PetscScalar diagonal = 0, constrained_residual = 1;
            MatGetValues( jacobian, 1, &constrained, 1, &constrained, &diagonal );
            VecGetValues( residual, 1, &constrained, &constrained_residual );
            check( std::abs( PetscRealPart( diagonal ) - 1 ) < 1.e-12 &&
                       std::abs( PetscRealPart( constrained_residual ) ) < 1.e-12,
                   "strong Dirichlet elimination sets unit row and u-g residual" );
            VecCopy( solution, perturbed );
            VecSetValue( perturbed, constrained, 2 + 1.e-7, INSERT_VALUES );
            VecAssemblyBegin( perturbed );
            VecAssemblyEnd( perturbed );
            KARMA::assemble_diffusion_reaction( system, perturbed, perturbed_residual, nullptr, coefficients,
                                                &essential_boundary );
            VecAXPY( perturbed_residual, -1, residual );
            VecScale( perturbed_residual, 1.e7 );
            VecSet( direction, 0 );
            VecSetValue( direction, constrained, 1, INSERT_VALUES );
            VecAssemblyBegin( direction );
            VecAssemblyEnd( direction );
            MatMult( jacobian, direction, jacobian_direction );
            VecAXPY( perturbed_residual, -1, jacobian_direction );
            VecNorm( perturbed_residual, NORM_2, &norm );
            check( norm < 1.e-7, "strong Dirichlet residual and eliminated Jacobian are consistent" );
        }

        VecDestroy( &jacobian_direction );
        VecDestroy( &perturbed_residual );
        VecDestroy( &perturbed );
        VecDestroy( &direction );
        MatDestroy( &jacobian );
        VecDestroy( &residual );
        VecDestroy( &solution );
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
