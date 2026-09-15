#include "diffusion_reaction_assembly.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_geometry_map.h"
#include "karma_parameters.h"
#include "karma_physics_bc_manager.h"
#include "karma_system.h"

#include <petsc.h>
#include <petscksp.h>

#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <string>
#include <vector>

namespace
{
int failures = 0;
void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error )
        KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in MMS " ) + operation ) );
}

Real exact( const KARMA::Point& point )
{ return std::sin( std::numbers::pi * point( 0 ) ) * std::sin( std::numbers::pi * point( 1 ) ); }

std::unique_ptr< KARMABoundaryCondition > zero_dirichlet( const KARMAParameters& )
{
    auto condition = std::make_unique< KARMABoundaryCondition >( KARMA::PHYSICS::Dirichlet );
    condition->set( 0 );
    return condition;
}

Real l2_error( const KARMA::KARMASystem& system, Vec solution )
{
    Real squared = 0;
    std::vector< PetscInt > dofs;
    for( const auto element : system.mesh().elements() )
    {
        const KARMA::KARMAGeometryMap geometry( system.mesh(), element );
        const auto& spec = system.variable( 0 );
        auto basis       = system.basis_registry().create(
            { KARMA::reference_cell( geometry.variant() ), spec.family, spec.order, spec.continuity } );
        KARMA::KARMAFEBase fe( system.mesh(), geometry.variant(), std::move( basis ), 5 );
        fe.reinit( element );
        system.dof_map().dof_indices( element, dofs, 0 );
        std::vector< PetscScalar > values( dofs.size() );
        VecGetValues( solution, static_cast< PetscInt >( dofs.size() ), dofs.data(), values.data() );
        for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
        {
            Real numerical = 0;
            for( std::size_t i = 0; i < dofs.size(); ++i )
                numerical += PetscRealPart( values[i] ) * fe.get_phi()[i][q];
            const Real difference = numerical - exact( fe.get_xyz()[q] );
            squared += fe.get_JxW()[q] * difference * difference;
        }
    }
    return std::sqrt( squared );
}

Real solve( unsigned cells )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( cells, cells, 0 );
    KARMA::KARMASystem system( mesh, "mms" );
    system.add_variable( "u" );
    system.init();
    KARMAPhysicsBCManager boundary( 0 );
    boundary.attach_function( zero_dirichlet, KARMA::PHYSICS::Dirichlet );
    KARMAParameters parameters;
    parameters.set< Real >( "time" ) = 0;
    boundary.refresh( mesh, system, parameters );

    KARMA::DiffusionReactionCoefficients coefficients;
    coefficients.source = []( const KARMA::Point& point, Real, int ) {
        return 2 * std::numbers::pi * std::numbers::pi * exact( point );
    };
    Vec solution = system.create_vector();
    Vec residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    VecSet( solution, 0 );
    KARMA::assemble_diffusion_reaction( system, solution, residual, jacobian, coefficients, &boundary );
    VecScale( residual, -1 );
    KSP solver = nullptr;
    petsc_check( KSPCreate( PETSC_COMM_WORLD, &solver ), "KSPCreate" );
    petsc_check( KSPSetOperators( solver, jacobian, jacobian ), "KSPSetOperators" );
    petsc_check( KSPSetType( solver, KSPCG ), "KSPSetType" );
    petsc_check( KSPSetTolerances( solver, 1.e-12, PETSC_DEFAULT, PETSC_DEFAULT, 1000 ), "KSPSetTolerances" );
    petsc_check( KSPSolve( solver, residual, solution ), "KSPSolve" );
    KSPConvergedReason reason         = KSP_CONVERGED_ITERATING;
    const PetscErrorCode reason_error = KSPGetConvergedReason( solver, &reason );
    if( reason_error || reason <= 0 )
    {
        std::cerr << "FAIL: KSP reason query=" << reason_error << ", reason=" << reason << '\n';
        ++failures;
    }
    Vec linear_residual = system.create_vector();
    petsc_check( MatMult( jacobian, solution, linear_residual ), "MatMult(linear residual)" );
    petsc_check( VecAXPY( linear_residual, -1, residual ), "VecAXPY(linear residual)" );
    PetscReal residual_norm = 0;
    petsc_check( VecNorm( linear_residual, NORM_2, &residual_norm ), "VecNorm(linear residual)" );
    if( !std::isfinite( residual_norm ) || residual_norm > 1.e-10 )
    {
        std::cerr << "FAIL: KSP residual " << residual_norm << " exceeds tolerance\n";
        ++failures;
    }
    VecDestroy( &linear_residual );
    const Real error = l2_error( system, solution );
    KSPDestroy( &solver );
    MatDestroy( &jacobian );
    VecDestroy( &residual );
    VecDestroy( &solution );
    return error;
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    int ranks = 1;
    MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
    if( ranks == 1 )
    {
        const std::array< Real, 3 > errors{ solve( 8 ), solve( 16 ), solve( 32 ) };
        for( std::size_t i = 0; i < errors.size(); ++i )
            if( !std::isfinite( errors[i] ) || errors[i] <= 0 || ( i > 0 && errors[i] >= errors[i - 1] ) )
            {
                std::cerr << "FAIL: MMS error[" << i << "]=" << errors[i]
                          << " is not finite, positive, and strictly decreasing\n";
                ++failures;
            }
        for( std::size_t i = 0; i < 2; ++i )
        {
            const Real rate = std::log( errors[i] / errors[i + 1] ) / std::log( 2.0 );
            std::cout << "MMS " << 8 * ( 1 << i ) << "->" << 16 * ( 1 << i ) << ": errors " << errors[i] << ", "
                      << errors[i + 1] << ", rate " << rate << '\n';
            if( !std::isfinite( rate ) || rate < 1.8 || rate > 2.2 )
            {
                std::cerr << "FAIL: P1 L2 convergence rate " << rate << " from errors " << errors[i] << " and "
                          << errors[i + 1] << '\n';
                ++failures;
            }
        }
        if( errors.back() >= 1.e-3 )
        {
            std::cerr << "FAIL: finest MMS error " << errors.back() << " exceeds 1e-3\n";
            ++failures;
        }
    }
    PetscFinalize();
    return failures == 0 ? 0 : 1;
}
