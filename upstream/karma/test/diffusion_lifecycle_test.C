#include "diffusion_reaction_mass_operator.h"
#include "diffusion_reaction_preconditioner_operator.h"
#include "diffusion_reaction_system.h"
#include "diffusion_reaction_system_operator.h"
#include "diffusion_reaction_system_property_manager.h"
#include "karma_exceptions.h"
#include "karma_parameters.h"
#include "karma_physics_bc_manager.h"

#include <petsc.h>

#include <cmath>
#include <iostream>
#include <limits>
#include <string_view>

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

template < class Function >
bool rejects( Function&& function, std::string_view text )
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

KARMA::DiffusionReactionCoefficients model( Real diffusion, Real reaction = 0, Real source = 0 )
{
    KARMA::DiffusionReactionCoefficients result;
    result.diffusion = [=]( const KARMA::Point&, Real, int ) { return diffusion; };
    result.reaction  = [=]( const KARMA::Point&, Real, int ) { return reaction; };
    result.source    = [=]( const KARMA::Point&, Real, int ) { return source; };
    return result;
}

std::unique_ptr< KARMABoundaryCondition > all_dirichlet( const KARMAParameters& )
{
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
    condition->set( 1 );
    return condition;
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if( mode.rfind( "--asymmetric-init-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, std::string( mode ).c_str() );
            int rank = 0;
            MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
            KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
            mesh.generate_box( 1, 1, 0 );
            KARMA::DiffusionReactionSystemPropertyManager properties;
            if( rank != 1 || mode != "--asymmetric-init-registry" ) properties.register_model( 0, model( 1 ) );
            if( rank == 1 && mode == "--asymmetric-init-material" ) properties.register_model( 7, model( 1 ) );
            KARMA::DiffusionReactionSystem::Configuration configuration;
            if( rank == 1 && mode == "--asymmetric-init-name" ) configuration.variable_name.clear();
            if( rank == 1 && mode == "--asymmetric-init-quadrature" ) configuration.quadrature_degree = 0;
            if( rank == 1 && mode == "--asymmetric-init-continuity" )
                configuration.continuity = KARMA::BasisContinuity::dG;
            if( rank == 1 && mode == "--asymmetric-init-config" ) configuration.order = KARMA::Order::Second;
            KARMA::DiffusionReactionSystem physics( mesh, std::move( properties ), configuration );
            check( rejects( [&] { physics.init(); }, "rank 1" ) && !physics.is_initialized(),
                   "asymmetric diffusion initialization fails collectively before PETSc creation" );
            if( mode == "--asymmetric-init-registry" )
            {
                if( rank == 1 ) physics.properties().register_model( 0, model( 1 ) );
                physics.init();
                check( physics.is_initialized(), "diffusion initialization remains retryable after collective validation" );
            }
        }
        else
        {
        KARMA::DiffusionReactionSystemPropertyManager properties;
        properties.register_model( 0, model( 2, 1, 3 ) );
        properties.register_model( 7, model( 5 ) );
        check( properties.model( 7 ).diffusion( KARMA::Point::Zero(), 0, 7 ) == 5,
               "model registry dispatches by material" );
        check( rejects( [&] { properties.register_model( 7, model( 1 ) ); }, "already registered" ),
               "model registry rejects duplicate materials" );
        check( rejects( [&] { (void)properties.model( 9 ); }, "material 9" ),
               "model registry rejects missing materials" );
        const auto dispatch = properties.coefficients();
        check( dispatch.diffusion( KARMA::Point::Zero(), 0, 7 ) == 5,
               "coefficient adapter dispatches by material" );

        KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 1, 0 );
        KARMA::DiffusionReactionSystem physics( mesh, properties );
        physics.init();
        physics.init();
        check( physics.is_initialized() && physics.system().n_variables() == 1 &&
                   physics.system().variable( 0 ).name == "u",
               "initialization is idempotent and configures one scalar variable" );

        KARMA::DiffusionReactionSystem::Configuration invalid;
        invalid.variable_name = "";
        KARMA::DiffusionReactionSystem invalid_system( mesh, properties, invalid );
        check( rejects( [&] { invalid_system.init(); }, "variable name" ),
                "initialization rejects invalid configuration" );

        for( const auto continuity : { KARMA::BasisContinuity::dG, KARMA::BasisContinuity::SE } )
        {
            KARMA::DiffusionReactionSystem::Configuration unsupported;
            unsupported.continuity = continuity;
            KARMA::DiffusionReactionSystem retryable( mesh, properties, unsupported );
            check( rejects( [&] { retryable.init(); }, "continuous Galerkin" ) && !retryable.is_initialized(),
                   "unsupported diffusion continuity is rejected before state commit" );
            unsupported.continuity = KARMA::BasisContinuity::cG;
            KARMA::DiffusionReactionSystem valid_retry( mesh, properties, unsupported );
            valid_retry.init();
            check( valid_retry.is_initialized(), "valid continuity initializes after unsupported configuration rejection" );
        }

        Vec x = physics.system().create_vector();
        Vec y = physics.system().create_vector();
        Vec residual = physics.system().create_vector();
        Mat mass = physics.system().create_matrix();
        Mat jacobian = physics.system().create_matrix();
        VecSet( x, 1 );

        KARMA::DiffusionReactionMassOperator mass_operator( physics.system() );
        mass_operator.assemble( mass );
        mass_operator.multiply( mass, x, y );
        if( mesh.n_ranks() == 1 )
        {
            Vec plain = nullptr;
            Mat plain_mass = nullptr;
            VecCreateMPI( PETSC_COMM_WORLD, physics.system().n_local_dofs(), physics.system().n_dofs(), &plain );
            MatCreateAIJ( PETSC_COMM_WORLD, physics.system().n_local_dofs(), physics.system().n_local_dofs(),
                          physics.system().n_dofs(), physics.system().n_dofs(), 1, nullptr, 0, nullptr, &plain_mass );
            MatSetUp( plain_mass );
            VecSet( y, 37 );
            check( rejects( [&] { mass_operator.multiply( mass, plain, y ); }, "mapping" ) &&
                       rejects( [&] { mass_operator.solve( mass, x, plain ); }, "mapping" ) &&
                       rejects( [&] { mass_operator.assemble( plain_mass ); }, "mapping" ) &&
                       rejects( [&] { physics.assemble_mass( plain_mass ); }, "mapping" ),
                   "diffusion mass adapters and system enforce system mappings" );
            PetscReal retained = 0;
            VecNorm( y, NORM_INFINITY, &retained );
            check( std::abs( retained - 37 ) < 1.e-12,
                   "rejected diffusion mass adapter calls leave output unchanged" );
            MatDestroy( &plain_mass );
            VecDestroy( &plain );
            mass_operator.multiply( mass, x, y );
        }
        PetscScalar mass_sum = 0;
        VecSum( y, &mass_sum );
        check( std::abs( PetscRealPart( mass_sum ) - 1 ) < 1.e-12,
                "mass adapter delegates assembly and multiply" );
        if( mesh.n_ranks() == 1 )
        {
            physics.boundary_conditions().attach_function( all_dirichlet, KARMA::PHYSICS::Dirichlet );
            KARMAParameters parameters;
            parameters.set< Real >( "time" ) = 0;
            physics.boundary_conditions().refresh( mesh, physics.system(), parameters );
            physics.assemble_mass( mass );
            MatAXPY( mass, -1, physics.mass(), SAME_NONZERO_PATTERN );
            PetscReal mass_difference = 0;
            MatNorm( mass, NORM_FROBENIUS, &mass_difference );
            check( mass_difference < 1.e-12,
                   "system mass assembly remains unconstrained and independent of refreshed boundary conditions" );
            physics.boundary_conditions().clear();
        }

        KARMA::DiffusionReactionSystemOperator system_operator( physics );
        system_operator.residual( x, residual );
        system_operator.jacobian( x, jacobian );
        PetscReal direct_norm = 0;
        PetscReal adapter_norm = 0;
        Vec direct = physics.system().create_vector();
        KARMA::assemble_diffusion_reaction( physics.system(), x, direct, nullptr, properties.coefficients() );
        VecAXPY( direct, -1, residual );
        VecNorm( direct, NORM_2, &direct_norm );
        MatNorm( jacobian, NORM_FROBENIUS, &adapter_norm );
        check( direct_norm < 1.e-12 && adapter_norm > 0,
               "system operator delegates residual and Jacobian assembly" );

        KARMA::DiffusionReactionPreconditionerOperator preconditioner( physics );
        Mat preconditioner_matrix = physics.system().create_matrix();
        preconditioner.assemble( x, preconditioner_matrix );
        MatAXPY( preconditioner_matrix, -1, jacobian, SAME_NONZERO_PATTERN );
        MatNorm( preconditioner_matrix, NORM_FROBENIUS, &adapter_norm );
        check( adapter_norm < 1.e-12, "preconditioner delegates verified Jacobian assembly" );

        Mat diagonal = physics.system().create_matrix();
        MatZeroEntries( diagonal );
        PetscInt first = 0, last = 0;
        MatGetOwnershipRange( diagonal, &first, &last );
        for( PetscInt row = first; row < last; ++row )
        {
            const PetscScalar value = row + 2;
            MatSetValue( diagonal, row, row, value, INSERT_VALUES );
            VecSetValue( x, row, 2 * value, INSERT_VALUES );
        }
        MatAssemblyBegin( diagonal, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( diagonal, MAT_FINAL_ASSEMBLY );
        VecAssemblyBegin( x );
        VecAssemblyEnd( x );
        KARMA::point_jacobi_solve( diagonal, x, y );
        VecShift( y, -2 );
        VecNorm( y, NORM_INFINITY, &adapter_norm );
        check( adapter_norm < 1.e-12, "point Jacobi gives the known diagonal solution" );

        for( const PetscScalar scale : { PetscScalar{ 1.e-20 }, PetscScalar{ 1.e20 } } )
        {
            MatZeroEntries( diagonal );
            VecSet( x, 2 * scale );
            for( PetscInt row = first; row < last; ++row )
                MatSetValue( diagonal, row, row, scale, INSERT_VALUES );
            MatAssemblyBegin( diagonal, MAT_FINAL_ASSEMBLY );
            MatAssemblyEnd( diagonal, MAT_FINAL_ASSEMBLY );
            KARMA::point_jacobi_solve( diagonal, x, y );
            VecShift( y, -2 );
            VecNorm( y, NORM_INFINITY, &adapter_norm );
            check( adapter_norm < 1.e-12, "point Jacobi validation is invariant under uniform matrix scaling" );
        }

        MatSetValue( diagonal, first, first, 0, INSERT_VALUES );
        MatAssemblyBegin( diagonal, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( diagonal, MAT_FINAL_ASSEMBLY );
        check( rejects( [&] { KARMA::point_jacobi_solve( diagonal, x, y ); }, "diagonal" ),
               "point Jacobi rejects a zero diagonal" );
        MatSetValue( diagonal, first, first, std::numeric_limits< PetscScalar >::infinity(), INSERT_VALUES );
        MatAssemblyBegin( diagonal, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( diagonal, MAT_FINAL_ASSEMBLY );
        check( rejects( [&] { KARMA::point_jacobi_solve( diagonal, x, y ); }, "diagonal" ),
               "point Jacobi rejects a nonfinite diagonal" );

        VecSet( physics.solution(), 0 );
        physics.solve();
        PetscReal solution_norm = 0;
        VecNorm( physics.solution(), NORM_INFINITY, &solution_norm );
        VecShift( physics.solution(), -3 );
        PetscReal solve_error = 0;
        VecNorm( physics.solution(), NORM_INFINITY, &solve_error );
        check( std::isfinite( solution_norm ) && solve_error < 1.e-10,
               "PETSc nonlinear solve integration recovers the constant reaction-source solution" );

        MatDestroy( &diagonal );
        MatDestroy( &preconditioner_matrix );
        VecDestroy( &direct );
        MatDestroy( &jacobian );
        MatDestroy( &mass );
        VecDestroy( &residual );
        VecDestroy( &y );
        VecDestroy( &x );
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    const PetscErrorCode finalize_error = PetscFinalize();
    return finalize_error ? finalize_error : global_failures == 0 ? 0 : 1;
}
