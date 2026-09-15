#include "karma_fe_base.h"
#include "karma_geometry_map.h"
#include "karma_mesh.h"
#include "karma_system.h"
#include "petsc_raii.h"
#include "transient_reaction_diffusion_assembly.h"
#include "transient_reaction_diffusion_component.h"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using FieldFunction = std::function<Real( const KARMA::Point& )>;

constexpr Real diffusion = 0.2;
constexpr Real pi = std::numbers::pi_v<Real>;

void require( bool condition, const std::string& message )
{
    if( !condition ) throw std::runtime_error( message );
}

void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error != PETSC_SUCCESS ) throw std::runtime_error( operation );
}

Real exact( const KARMA::Point& point, Real time )
{
    return std::exp( -time ) * std::cos( pi * point( 0 ) ) * std::cos( pi * point( 1 ) );
}

Real exact_time_derivative( const KARMA::Point& point, Real time ) { return -exact( point, time ); }

Real exact_laplacian( const KARMA::Point& point, Real time )
{
    return -2 * pi * pi * exact( point, time );
}

// The assembled residual is
//   (w_right-w_left)/dt * phi + d grad(avg(w)).grad(phi) - avg(source) * phi.
// With the natural boundary term omitted, integration by parts gives the strong
// equation w_t - d Laplacian(w) = source and homogeneous Neumann flux. Hence
// for w=exp(-t) cos(pi*x) cos(pi*y), Laplacian(w)=-2*pi^2*w and
//   source=(-1+2*d*pi^2)*w.
Real manufactured_source( const KARMA::Point& point, Real time )
{
    return ( -1 + 2 * diffusion * pi * pi ) * exact( point, time );
}

KARMA::TransientReactionDiffusionModel manufactured_model()
{
    KARMA::TransientReactionDiffusionModel model;
    model.diffusion = diffusion;
    model.reaction = []( const KARMA::Point& point, Real, Real, Real time ) {
        return manufactured_source( point, time );
    };
    model.state_derivative = []( const KARMA::Point&, Real, Real, Real ) { return 0.; };
    return model;
}

void interpolate( const KARMA::KARMASystem& system, Vec vector, const FieldFunction& field )
{
    petsc_check( VecSet( vector, 0 ), "zero interpolation vector" );
    for( const auto vertex : system.mesh().vertices() )
    {
        double coordinates[3]{};
        require( system.mesh().moab()->get_coords( &vertex, 1, coordinates ) == moab::MB_SUCCESS,
                 "read interpolation coordinates" );
        const PetscInt dof = system.dof_map().vertex_dof( vertex, 0, 0 );
        petsc_check( VecSetValue( vector, dof, field( KARMA::Point( coordinates ) ), INSERT_VALUES ),
                     "set interpolation value" );
    }
    petsc_check( VecAssemblyBegin( vector ), "begin interpolation assembly" );
    petsc_check( VecAssemblyEnd( vector ), "end interpolation assembly" );
}

Real l2_error( const KARMA::KARMASystem& system, Vec values, const FieldFunction& field )
{
    petsc_check( VecGhostUpdateBegin( values, INSERT_VALUES, SCATTER_FORWARD ), "begin L2 ghost update" );
    petsc_check( VecGhostUpdateEnd( values, INSERT_VALUES, SCATTER_FORWARD ), "end L2 ghost update" );
    KARMA::PetscVecLocalForm local_form( values );
    KARMA::PetscVecArrayRead local_values( local_form.get() );
    Real squared = 0;
    for( const auto element : system.dof_map().assembly_elements() )
    {
        const KARMA::KARMAGeometryMap geometry( system.mesh(), element );
        const auto& specification = system.variable( 0 );
        auto basis = system.basis_registry().create(
            { KARMA::reference_cell( geometry.variant() ), specification.family, specification.order,
              specification.continuity } );
        KARMA::KARMAFEBase fe( system.mesh(), geometry.variant(), std::move( basis ), 6 );
        fe.reinit( element );
        std::vector<PetscInt> local_dofs;
        system.dof_map().local_dof_indices( element, local_dofs );
        for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
        {
            Real numerical = 0;
            for( std::size_t i = 0; i < local_dofs.size(); ++i )
                numerical += PetscRealPart( local_values.get()[local_dofs[i]] ) * fe.get_phi()[i][q];
            const Real difference = numerical - field( fe.get_xyz()[q] );
            squared += fe.get_JxW()[q] * difference * difference;
        }
    }
    return std::sqrt( squared );
}

struct SolveResult
{
    Real error = 0;
    Real residual = 0;
    Real residual_scale = 0;
};

SolveResult solve_manufactured( unsigned cells, Real time_step, Real final_time )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( cells, cells, 0 );
    KARMA::TransientReactionDiffusionComponent component( mesh, manufactured_model() );
    component.init();
    KARMA::PetscOwner<Vec, VecDestroy> input;
    *input.put() = component.system().create_vector();
    petsc_check( VecSet( input.get(), 0 ), "zero manufactured input" );
    component.bind_input( input.get() );
    interpolate( component.system(), component.solution(), []( const KARMA::Point& point ) { return exact( point, 0 ); } );
    petsc_check( VecCopy( component.solution(), component.trial_solution() ), "initialize manufactured trial" );

    const int steps = static_cast<int>( std::llround( final_time / time_step ) );
    require( std::abs( steps * time_step - final_time ) < 64 * std::numeric_limits<Real>::epsilon(),
             "manufactured final time is not divisible by the time step" );
    KARMA::PetscOwner<Vec, VecDestroy> previous;
    *previous.put() = component.system().create_vector();
    Real time = 0;
    for( int step = 0; step < steps; ++step )
    {
        petsc_check( VecCopy( component.solution(), previous.get() ), "save previous manufactured state" );
        component.checkpoint();
        component.advance( time, time + time_step );
        component.commit();
        component.release_checkpoint();
        time += time_step;
    }

    KARMA::PetscOwner<Vec, VecDestroy> residual;
    *residual.put() = component.system().create_vector();
    KARMA::assemble_transient_reaction_diffusion(
        component.system(), previous.get(), component.solution(), component.input(), final_time - time_step,
        final_time, manufactured_model(), residual.get(), nullptr, 5 );
    PetscReal residual_norm = 0, old_norm = 0, solution_norm = 0, input_norm = 0;
    petsc_check( VecNorm( residual.get(), NORM_INFINITY, &residual_norm ), "measure manufactured residual" );
    petsc_check( VecNorm( previous.get(), NORM_INFINITY, &old_norm ), "measure manufactured old scale" );
    petsc_check( VecNorm( component.solution(), NORM_INFINITY, &solution_norm ), "measure manufactured state scale" );
    petsc_check( VecNorm( component.input(), NORM_INFINITY, &input_norm ), "measure manufactured input scale" );
    const Real scale = std::max<Real>( 1, std::max( { old_norm, solution_norm, input_norm } ) );
    require( std::isfinite( residual_norm ) && residual_norm <= 1.e-8 * scale,
             "independent scale-aware manufactured residual check failed" );

    return { l2_error( component.system(), component.solution(), [final_time]( const KARMA::Point& point ) {
                 return exact( point, final_time );
             } ),
             residual_norm, scale };
}

void verify_strong_form()
{
    const std::array<KARMA::Point, 4> points{
        KARMA::Point( 0.13, 0.27, 0 ), KARMA::Point( 0.41, 0.73, 0 ),
        KARMA::Point( 0.82, 0.19, 0 ), KARMA::Point( 0.57, 0.64, 0 ) };
    const std::array<Real, 3> times{ 0, 0.17, 0.63 };
    for( const auto& point : points )
        for( const Real time : times )
        {
            const Real residual = exact_time_derivative( point, time ) -
                                  diffusion * exact_laplacian( point, time ) - manufactured_source( point, time );
            require( std::abs( residual ) <= 32 * std::numeric_limits<Real>::epsilon(),
                     "manufactured strong-form residual is nonzero" );
        }
}

void verify_boundary_flux()
{
    const std::array<Real, 3> coordinates{ 0.11, 0.43, 0.89 };
    const std::array<Real, 3> times{ 0, 0.23, 0.71 };
    for( const Real coordinate : coordinates )
        for( const Real time : times )
        {
            const Real factor = std::exp( -time );
            const Real left = factor * pi * std::sin( 0 ) * std::cos( pi * coordinate );
            const Real right = -factor * pi * std::sin( pi ) * std::cos( pi * coordinate );
            const Real bottom = factor * pi * std::cos( pi * coordinate ) * std::sin( 0 );
            const Real top = -factor * pi * std::cos( pi * coordinate ) * std::sin( pi );
            const Real tolerance = 8 * pi * std::numeric_limits<Real>::epsilon();
            require( std::abs( left ) <= tolerance && std::abs( right ) <= tolerance &&
                         std::abs( bottom ) <= tolerance && std::abs( top ) <= tolerance,
                     "manufactured outward boundary flux is nonzero" );
        }
}

void verify_stiffness_energy()
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 1, 1, 0 );
    KARMA::KARMASystem system( mesh, "transient stiffness energy" );
    system.add_variable( "u" );
    system.init();
    const auto element = *mesh.elements().begin();
    KARMA::RealDenseVector left( 4 ), right( 4 ), input( 4 );
    const Real right_values[4]{ -0.7, 0.2, 1.1, -0.4 };
    for( std::size_t i = 0; i < right.size(); ++i ) right( i ) = right_values[i];
    KARMA::TransientReactionDiffusionModel with_diffusion;
    with_diffusion.diffusion = diffusion;
    with_diffusion.reaction = []( const KARMA::Point&, Real, Real, Real ) { return 0.; };
    with_diffusion.state_derivative = []( const KARMA::Point&, Real, Real, Real ) { return 0.; };
    auto without_diffusion = with_diffusion;
    without_diffusion.diffusion = 0;
    const auto full = KARMA::element_transient_reaction_diffusion(
        system, element, with_diffusion, { 0, 1, left, right, input }, 6 );
    const auto mass_only = KARMA::element_transient_reaction_diffusion(
        system, element, without_diffusion, { 0, 1, left, right, input }, 6 );
    const auto stiffness = 2 * ( full.jacobian.native() - mass_only.jacobian.native() ) / diffusion;
    const Real energy = right.native().dot( stiffness * right.native() );
    const auto constant = KARMA::RealDenseVector::Storage::Ones( 4 );
    const Real constant_energy = constant.dot( stiffness * constant );
    require( energy > 1.e-2, "nonconstant diffusion stiffness energy is not positive" );
    require( std::abs( constant_energy ) <= 1.e-12 * stiffness.norm(),
             "constant mode has nonzero diffusion stiffness energy" );
    require( ( stiffness - stiffness.transpose() ).norm() <= 1.e-13 * stiffness.norm(),
             "diffusion stiffness is not symmetric" );
}

void verify_cn_amplification()
{
    constexpr Real lambda = 1.7;
    constexpr Real step = 0.2;
    const Real amplification = ( 1 - lambda * step / 2 ) / ( 1 + lambda * step / 2 );
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 1, 0 );
    KARMA::TransientReactionDiffusionModel model;
    model.diffusion = 0;
    model.reaction = []( const KARMA::Point&, Real state, Real, Real ) { return -lambda * state; };
    model.state_derivative = []( const KARMA::Point&, Real, Real, Real ) { return -lambda; };
    KARMA::TransientReactionDiffusionComponent component( mesh, model );
    component.init();
    KARMA::PetscOwner<Vec, VecDestroy> input;
    KARMA::PetscOwner<Vec, VecDestroy> initial;
    *input.put() = component.system().create_vector();
    *initial.put() = component.system().create_vector();
    petsc_check( VecSet( input.get(), 0 ), "zero amplification input" );
    component.bind_input( input.get() );
    interpolate( component.system(), component.solution(), []( const KARMA::Point& point ) {
        return 0.4 + 0.7 * point( 0 ) - 0.3 * point( 1 );
    } );
    petsc_check( VecCopy( component.solution(), initial.get() ), "copy amplification initial state" );
    component.checkpoint();
    component.advance( 0, step );

    // With d=0 and reaction=-lambda*w, the consistent mass matrix multiplies
    // both CN terms: M[(u1-u0)/h + lambda*(u0+u1)/2]=0. Since M is invertible,
    // every nodal coefficient has exactly the scalar CN amplification; no mass
    // lumping or special geometry is required. The 2x1 mesh supplies six
    // independently checked, nonuniform coefficients.
    PetscInt first = 0, last = 0;
    petsc_check( VecGetOwnershipRange( initial.get(), &first, &last ), "read amplification ownership" );
    std::vector<PetscInt> indices( static_cast<std::size_t>( last - first ) );
    std::vector<PetscScalar> before( indices.size() ), after( indices.size() );
    for( PetscInt i = first; i < last; ++i ) indices[static_cast<std::size_t>( i - first )] = i;
    petsc_check( VecGetValues( initial.get(), last - first, indices.data(), before.data() ),
                 "read amplification initial coefficients" );
    petsc_check( VecGetValues( component.trial_solution(), last - first, indices.data(), after.data() ),
                 "read amplification final coefficients" );
    for( std::size_t i = 0; i < indices.size(); ++i )
    {
        const Real expected = amplification * PetscRealPart( before[i] );
        const Real tolerance = 2.e-11 * std::max<Real>( 1, std::abs( expected ) );
        require( std::abs( PetscRealPart( after[i] ) - expected ) <= tolerance,
                 "CN coefficient amplification does not match the exact factor" );
    }
}

void verify_options_prefix()
{
    const auto run = []( const char* option ) {
        petsc_check( PetscOptionsSetValue( nullptr, option, "1e6" ), "set SNES prefix test option" );
        bool succeeded = true;
        try { (void)solve_manufactured( 2, 0.1, 0.1 ); }
        catch( const std::exception& ) { succeeded = false; }
        petsc_check( PetscOptionsClearValue( nullptr, option ), "clear SNES prefix test option" );
        return succeeded;
    };
    require( run( "-snes_atol" ), "unprefixed ambient SNES option changed the transient solve" );
    require( !run( "-karma_transient_reaction_diffusion_snes_atol" ),
             "prefixed SNES option did not reach the transient solve" );
}

Real temporal_nodal_error( Real time_step, Real final_time )
{
    KARMA::KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    mesh.generate_box( 4, 4, 0 );
    KARMA::TransientReactionDiffusionModel model;
    model.diffusion = 0;
    model.reaction = []( const KARMA::Point&, Real state, Real, Real ) { return -state; };
    model.state_derivative = []( const KARMA::Point&, Real, Real, Real ) { return -1.; };
    KARMA::TransientReactionDiffusionComponent component( mesh, model );
    component.init();
    KARMA::PetscOwner<Vec, VecDestroy> input;
    *input.put() = component.system().create_vector();
    petsc_check( VecSet( input.get(), 0 ), "zero temporal input" );
    component.bind_input( input.get() );
    interpolate( component.system(), component.solution(), []( const KARMA::Point& point ) { return exact( point, 0 ); } );
    petsc_check( VecCopy( component.solution(), component.trial_solution() ), "initialize temporal trial" );
    const int steps = static_cast<int>( std::llround( final_time / time_step ) );
    Real time = 0;
    for( int step = 0; step < steps; ++step )
    {
        component.checkpoint();
        component.advance( time, time + time_step );
        component.commit();
        component.release_checkpoint();
        time += time_step;
    }
    PetscReal error = 0;
    for( const auto vertex : mesh.vertices() )
    {
        double coordinates[3]{};
        require( mesh.moab()->get_coords( &vertex, 1, coordinates ) == moab::MB_SUCCESS,
                 "read temporal error coordinates" );
        const PetscInt dof = component.system().dof_map().vertex_dof( vertex, 0, 0 );
        PetscScalar value = 0;
        petsc_check( VecGetValues( component.solution(), 1, &dof, &value ), "read temporal solution coefficient" );
        error = std::max( error, std::abs( PetscRealPart( value ) - exact( KARMA::Point( coordinates ), final_time ) ) );
    }
    return error;
}

void verify_temporal_convergence()
{
    constexpr Real final_time = 0.4;
    const std::array<Real, 3> steps{ 0.2, 0.1, 0.05 };
    std::array<Real, 3> errors;
    for( std::size_t i = 0; i < steps.size(); ++i ) errors[i] = temporal_nodal_error( steps[i], final_time );
    for( std::size_t i = 0; i + 1 < errors.size(); ++i )
    {
        const Real rate = std::log( errors[i] / errors[i + 1] ) / std::log( 2. );
        std::cout << "transient temporal " << steps[i] << "->" << steps[i + 1] << ": errors "
                  << errors[i] << ", " << errors[i + 1] << ", rate " << rate << '\n';
        require( std::isfinite( rate ) && rate >= 1.8 && rate <= 2.2,
                 "Crank-Nicolson temporal convergence rate is outside [1.8, 2.2]" );
    }
}

void verify_spatial_convergence_and_solver_plateau()
{
    constexpr Real final_time = 0.01;
    constexpr Real spatial_step = 0.01;
    const std::array<unsigned, 3> cells{ 4, 8, 16 };
    std::array<SolveResult, 3> results;
    for( std::size_t i = 0; i < cells.size(); ++i ) results[i] = solve_manufactured( cells[i], spatial_step, final_time );
    for( std::size_t i = 0; i + 1 < results.size(); ++i )
    {
        const Real rate = std::log( results[i].error / results[i + 1].error ) / std::log( 2. );
        std::cout << "transient spatial " << cells[i] << "->" << cells[i + 1] << ": errors "
                  << results[i].error << ", " << results[i + 1].error << ", rate " << rate << '\n';
        require( std::isfinite( rate ) && rate >= 1.8 && rate <= 2.2,
                 "P1 spatial L2 convergence rate is outside [1.8, 2.2]" );
    }

    const auto temporal_reference = solve_manufactured( cells.back(), spatial_step / 2, final_time );
    const Real temporal_estimate = std::abs( results.back().error - temporal_reference.error ) / 3;
    std::cout << "finest spatial error=" << results.back().error
              << ", Richardson temporal estimate=" << temporal_estimate
              << ", ratio=" << temporal_estimate / results.back().error
              << ", scaled residual=" << results.back().residual / results.back().residual_scale << '\n';
    require( temporal_estimate < 0.1 * results.back().error,
             "temporal error is not below ten percent of the finest spatial error" );

    petsc_check( PetscOptionsSetValue( nullptr, "-karma_transient_reaction_diffusion_snes_rtol", "1e-12" ),
                 "set strict transient SNES tolerance" );
    const auto strict = solve_manufactured( cells.back(), spatial_step, final_time );
    petsc_check( PetscOptionsClearValue( nullptr, "-karma_transient_reaction_diffusion_snes_rtol" ),
                 "clear strict transient SNES tolerance" );
    const Real solver_difference = std::abs( strict.error - results.back().error );
    std::cout << "solver tolerance plateau: default error=" << results.back().error
              << ", strict error=" << strict.error << ", difference=" << solver_difference << '\n';
    require( solver_difference < 0.01 * results.back().error,
             "solver tolerance change is not below the discretization-error plateau" );
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) != PETSC_SUCCESS ) return 1;
    int status = 0;
    try
    {
        int ranks = 1;
        petsc_check( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ), "query MMS communicator size" );
        require( ranks == 1, "transient reaction-diffusion MMS is a serial test" );
        std::cout << "checking exact identities\n";
        verify_strong_form();
        verify_boundary_flux();
        std::cout << "checking stiffness energy\n";
        verify_stiffness_energy();
        std::cout << "checking CN amplification\n";
        verify_cn_amplification();
        std::cout << "checking temporal convergence\n";
        verify_temporal_convergence();
        std::cout << "checking spatial convergence\n";
        verify_spatial_convergence_and_solver_plateau();
        std::cout << "checking options prefix\n";
        verify_options_prefix();
    }
    catch( const std::exception& error )
    {
        std::cerr << "transient reaction-diffusion MMS failure: " << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
