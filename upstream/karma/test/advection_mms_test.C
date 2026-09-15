#include "advection_reaction_assembly.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_geometry_map.h"
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
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in advection MMS " ) + operation ) );
}

Real exact( const KARMA::Point& point )
{
    return std::sin( std::numbers::pi * point( 0 ) );
}

Real l2_error( const KARMA::KARMASystem& system, Vec solution )
{
    Real squared = 0;
    std::vector< PetscInt > dofs;
    for( const auto element : system.mesh().elements() )
    {
        const KARMA::KARMAGeometryMap geometry( system.mesh(), element );
        const auto& spec = system.variable( 0 );
        auto basis = system.basis_registry().create(
            { KARMA::reference_cell( geometry.variant() ), spec.family, spec.order, spec.continuity } );
        KARMA::KARMAFEBase fe( system.mesh(), geometry.variant(), std::move( basis ), 8 );
        fe.reinit( element );
        system.dof_map().dof_indices( element, dofs );
        std::vector< PetscScalar > values( dofs.size() );
        petsc_check( VecGetValues( solution, static_cast< PetscInt >( dofs.size() ), dofs.data(), values.data() ),
                     "VecGetValues(error)" );
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
    KARMA::KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    mesh.generate_box( cells, 0, 0 );
    KARMA::KARMASystem system( mesh, "advection-mms" );
    system.add_variable( { "u", KARMA::FieldBasisFamily::LagrangeNodal, 1,
                           KARMA::BasisContinuity::dG, 1 } );
    system.init();
    const std::vector< moab::EntityHandle > elements( mesh.elements().begin(), mesh.elements().end() );
    for( std::size_t i = 1; i < elements.size(); ++i )
        system.add_element_adjacency( elements[i - 1], elements[i] );

    KARMA::AdvectionReactionCoefficients coefficients;
    coefficients.velocity = []( const KARMA::Point&, int ) { return KARMA::Vector3::UnitX(); };
    coefficients.reaction = []( const KARMA::Point&, Real field, int ) { return field; };
    coefficients.reaction_derivative = []( const KARMA::Point&, Real, int ) { return 1; };
    coefficients.source = []( const KARMA::Point& point, Real, int ) {
        return std::numbers::pi * std::cos( std::numbers::pi * point( 0 ) ) + exact( point );
    };
    coefficients.source_derivative = []( const KARMA::Point&, Real, int ) { return 0; };
    coefficients.inflow = []( const KARMA::Point&, int ) { return 0; };

    Vec solution = system.create_vector();
    Vec residual = system.create_vector();
    Mat jacobian = system.create_matrix();
    petsc_check( VecSet( solution, 0 ), "VecSet(solution)" );
    KARMA::assemble_advection_reaction( system, solution, residual, jacobian, coefficients, 8 );
    petsc_check( VecScale( residual, -1 ), "VecScale(rhs)" );

    KSP solver = nullptr;
    petsc_check( KSPCreate( PETSC_COMM_WORLD, &solver ), "KSPCreate" );
    petsc_check( KSPSetOperators( solver, jacobian, jacobian ), "KSPSetOperators" );
    petsc_check( KSPSetType( solver, KSPPREONLY ), "KSPSetType" );
    PC preconditioner = nullptr;
    petsc_check( KSPGetPC( solver, &preconditioner ), "KSPGetPC" );
    petsc_check( PCSetType( preconditioner, PCLU ), "PCSetType" );
    petsc_check( KSPSolve( solver, residual, solution ), "KSPSolve" );
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    petsc_check( KSPGetConvergedReason( solver, &reason ), "KSPGetConvergedReason" );
    if( reason <= 0 )
    {
        std::cerr << "FAIL: advection MMS KSP reason " << reason << '\n';
        ++failures;
    }
    Vec linear_residual = system.create_vector();
    petsc_check( MatMult( jacobian, solution, linear_residual ), "MatMult(linear residual)" );
    petsc_check( VecAXPY( linear_residual, -1, residual ), "VecAXPY(linear residual)" );
    PetscReal residual_norm = 0;
    petsc_check( VecNorm( linear_residual, NORM_2, &residual_norm ), "VecNorm(linear residual)" );
    if( !std::isfinite( residual_norm ) || residual_norm > 1.e-10 )
    {
        std::cerr << "FAIL: advection MMS algebraic residual " << residual_norm << '\n';
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
        constexpr std::array< unsigned, 3 > meshes{ 16, 32, 64 };
        const std::array< Real, 3 > errors{ solve( meshes[0] ), solve( meshes[1] ), solve( meshes[2] ) };
        for( std::size_t i = 0; i < errors.size(); ++i )
            if( !std::isfinite( errors[i] ) || errors[i] <= 0 || ( i > 0 && errors[i] >= errors[i - 1] ) )
            {
                std::cerr << "FAIL: advection MMS error[" << i << "]=" << errors[i]
                          << " is not finite, positive, and decreasing\n";
                ++failures;
            }
        for( std::size_t i = 0; i + 1 < errors.size(); ++i )
        {
            const Real rate = std::log( errors[i] / errors[i + 1] ) / std::log( 2.0 );
            std::cout << "Advection MMS " << meshes[i] << "->" << meshes[i + 1] << ": errors "
                      << errors[i] << ", " << errors[i + 1] << ", rate " << rate << '\n';
            if( !std::isfinite( rate ) || rate < 1.7 || rate > 2.3 )
            {
                std::cerr << "FAIL: P1 DG L2 convergence rate " << rate << '\n';
                ++failures;
            }
        }
        if( errors.back() >= 1.e-3 )
        {
            std::cerr << "FAIL: finest advection MMS error " << errors.back() << " exceeds 1e-3\n";
            ++failures;
        }
    }
    PetscFinalize();
    return failures == 0 ? 0 : 1;
}
