#include "karma_fe_base.h"
#include "karma_fixed_point.h"
#include "karma_geometry_map.h"
#include "karma_mesh.h"
#include "karma_operator_split_graph.h"
#include "karma_system.h"
#include "karma_transient_reaction_diffusion_adapter.h"
#include "petsc_raii.h"

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
using namespace KARMA;
using FieldFunction = std::function<Real( const Point& )>;

constexpr Real pi = std::numbers::pi_v<Real>;
constexpr Real du = 0.01;
constexpr Real dv = 0.005;
constexpr Real feed = 0.04;
constexpr Real kill = 0.06;
constexpr Real temporal_contamination_fraction = 0.1;
constexpr Real mutation_final_time = 0.05;

void require( bool condition, const std::string& message )
{
    if( !condition ) throw std::runtime_error( message );
}

void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error != PETSC_SUCCESS ) throw std::runtime_error( operation );
}

Real u_exact( const Point& point, Real time )
{
    return 0.75 + 0.15 * std::exp( -0.5 * time ) * std::cos( pi * point( 0 ) ) * std::cos( pi * point( 1 ) );
}

Real v_exact( const Point& point, Real time )
{
    return 0.25 + 0.08 * std::exp( -0.4 * time ) * std::cos( 2 * pi * point( 0 ) ) * std::cos( pi * point( 1 ) );
}

Real u_time_derivative( const Point& point, Real time ) { return -0.075 * std::exp( -0.5 * time ) * std::cos( pi * point( 0 ) ) * std::cos( pi * point( 1 ) ); }
Real v_time_derivative( const Point& point, Real time ) { return -0.032 * std::exp( -0.4 * time ) * std::cos( 2 * pi * point( 0 ) ) * std::cos( pi * point( 1 ) ); }
Real u_laplacian( const Point& point, Real time ) { return -2 * pi * pi * ( u_exact( point, time ) - 0.75 ); }
Real v_laplacian( const Point& point, Real time ) { return -5 * pi * pi * ( v_exact( point, time ) - 0.25 ); }

// The assembled weak residual is w_t*phi + D grad(w).grad(phi) - R*phi.
// Thus its strong equation is w_t-D Laplacian(w)=R.  With the Gray--Scott
// reactions R_u=-UV^2+F(1-U)+s_u and R_v=UV^2-(F+k)V+s_v, the independent
// manufactured sources are exactly those below.  These functions never call
// production Gray--Scott reaction helpers.
Real u_source( const Point& point, Real time, bool wrong_diffusion_sign = false )
{
    const Real diffusion_term = ( wrong_diffusion_sign ? du : -du ) * u_laplacian( point, time );
    const Real u = u_exact( point, time );
    const Real v = v_exact( point, time );
    return u_time_derivative( point, time ) + diffusion_term + u * v * v - feed * ( 1 - u );
}

Real v_source( const Point& point, Real time, bool wrong_diffusion_sign = false )
{
    const Real diffusion_term = ( wrong_diffusion_sign ? dv : -dv ) * v_laplacian( point, time );
    const Real u = u_exact( point, time );
    const Real v = v_exact( point, time );
    return v_time_derivative( point, time ) + diffusion_term - u * v * v + ( feed + kill ) * v;
}

GrayScottParameters parameters( bool wrong_diffusion_sign = false )
{
    GrayScottParameters result;
    result.diffusion_u = du;
    result.diffusion_v = dv;
    result.feed_rate = feed;
    result.kill_rate = kill;
    result.source_u = [wrong_diffusion_sign]( const Point& point, Real time ) { return u_source( point, time, wrong_diffusion_sign ); };
    result.source_v = [wrong_diffusion_sign]( const Point& point, Real time ) { return v_source( point, time, wrong_diffusion_sign ); };
    return result;
}

OperatorSplitGraphSpec graph( SplitScheme scheme )
{
    const auto component = []( const std::string& id ) {
        return ComponentSpec{ id,
                              { { "state", PortDirection::Output, false, { id + ".state", "1", 1, FieldRole::Prognostic } },
                                { "coupled", PortDirection::Input, true, { id + ".coupled", "1", 1, FieldRole::Input } } },
                              true, true };
    };
    return { "gray-scott-spatial-mms", { { { "U", component( "U" ) }, { "V", component( "V" ) } } },
             { { { "U", "state", "V", "coupled" }, { "V", "state", "U", "coupled" } } }, scheme };
}

void interpolate( const KARMASystem& system, Vec vector, const FieldFunction& field )
{
    petsc_check( VecSet( vector, 0 ), "zero interpolation vector" );
    for( const auto vertex : system.mesh().vertices() )
    {
        double coordinates[3]{};
        require( system.mesh().moab()->get_coords( &vertex, 1, coordinates ) == moab::MB_SUCCESS, "read interpolation coordinates" );
        const PetscInt dof = system.dof_map().vertex_dof( vertex, 0, 0 );
        petsc_check( VecSetValue( vector, dof, field( Point( coordinates ) ), INSERT_VALUES ), "set interpolation value" );
    }
    petsc_check( VecAssemblyBegin( vector ), "begin interpolation assembly" );
    petsc_check( VecAssemblyEnd( vector ), "end interpolation assembly" );
}

Real l2_error( const KARMASystem& system, Vec values, const FieldFunction& field )
{
    petsc_check( VecGhostUpdateBegin( values, INSERT_VALUES, SCATTER_FORWARD ), "begin L2 ghost update" );
    petsc_check( VecGhostUpdateEnd( values, INSERT_VALUES, SCATTER_FORWARD ), "end L2 ghost update" );
    PetscVecLocalForm local_form( values );
    PetscVecArrayRead local_values( local_form.get() );
    Real squared = 0;
    for( const auto element : system.dof_map().assembly_elements() )
    {
        const KARMAGeometryMap geometry( system.mesh(), element );
        const auto& specification = system.variable( 0 );
        auto basis = system.basis_registry().create( { reference_cell( geometry.variant() ), specification.family,
                                                        specification.order, specification.continuity } );
        KARMAFEBase fe( system.mesh(), geometry.variant(), std::move( basis ), 6 );
        fe.reinit( element );
        std::vector<PetscInt> local_dofs;
        system.dof_map().local_dof_indices( element, local_dofs );
        for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
        {
            Real numerical = 0;
            for( std::size_t i = 0; i < local_dofs.size(); ++i ) numerical += PetscRealPart( local_values.get()[local_dofs[i]] ) * fe.get_phi()[i][q];
            const Real difference = numerical - field( fe.get_xyz()[q] );
            squared += fe.get_JxW()[q] * difference * difference;
        }
    }
    return std::sqrt( squared );
}

struct Errors
{
    Real u;
    Real v;
    [[nodiscard]] Real combined() const { return std::hypot( u, v ); }
};

struct TrajectoryValues
{
    std::vector<Real> u;
    std::vector<Real> v;
};

Errors trajectory_difference( const TrajectoryValues& left, const TrajectoryValues& right );

void dropped_transfer( const ConnectionSpec&, KARMANativeComponent&, KARMANativeComponent&, FieldVisibility, Vec staging )
{
    petsc_check( VecSet( staging, 0 ), "drop transferred coupling field" );
}

Errors solve( unsigned u_cells, unsigned v_cells, SplitScheme scheme, Real final_time, Real step,
               bool wrong_diffusion_sign = false, bool drop_transfer = false, TrajectoryValues* values = nullptr )

{
    KARMAMesh u_mesh( 2, PETSC_COMM_WORLD );
    KARMAMesh v_mesh( 2, PETSC_COMM_WORLD );
    u_mesh.generate_box( u_cells, u_cells, 0 );
    v_mesh.generate_box( v_cells, v_cells, 0 );
    TransientReactionDiffusionNativeAdapter u( "U", u_mesh, GrayScottSpecies::U, parameters( wrong_diffusion_sign ) );
    TransientReactionDiffusionNativeAdapter v( "V", v_mesh, GrayScottSpecies::V, parameters( wrong_diffusion_sign ) );
    const auto compiled = compile_operator_split_graph( graph( scheme ), PETSC_COMM_WORLD );
    SchedulerOptions options;
    if( drop_transfer ) options.exchange_hook = dropped_transfer;
    KARMACouplingScheduler scheduler( compiled, { &u, &v }, options );
    interpolate( *std::get<PetscFieldBinding>( u.output_field( "state", FieldVisibility::Committed ).binding() ).system,
                 std::get<PetscFieldBinding>( u.output_field( "state", FieldVisibility::Committed ).binding() ).vector,
                 []( const Point& point ) { return u_exact( point, 0 ); } );
    interpolate( *std::get<PetscFieldBinding>( v.output_field( "state", FieldVisibility::Committed ).binding() ).system,
                 std::get<PetscFieldBinding>( v.output_field( "state", FieldVisibility::Committed ).binding() ).vector,
                 []( const Point& point ) { return v_exact( point, 0 ); } );
    const unsigned steps = static_cast<unsigned>( std::ceil( final_time / step ) );
    const Real dt = final_time / steps;
    for( unsigned n = 0; n < steps; ++n ) scheduler.advance( n * dt, ( n + 1 ) * dt );
    const auto u_field = std::get<PetscFieldBinding>( u.output_field( "state", FieldVisibility::Committed ).binding() );
    const auto v_field = std::get<PetscFieldBinding>( v.output_field( "state", FieldVisibility::Committed ).binding() );
    if( values != nullptr )
    {
        PetscInt u_size = 0;
        PetscInt v_size = 0;
        petsc_check( VecGetSize( u_field.vector, &u_size ), "read U trajectory size" );
        petsc_check( VecGetSize( v_field.vector, &v_size ), "read V trajectory size" );
        values->u.resize( static_cast<std::size_t>( u_size ) );
        values->v.resize( static_cast<std::size_t>( v_size ) );
        std::vector<PetscInt> u_indices( values->u.size() );
        std::vector<PetscInt> v_indices( values->v.size() );
        for( PetscInt i = 0; i < u_size; ++i ) u_indices[static_cast<std::size_t>( i )] = i;
        for( PetscInt i = 0; i < v_size; ++i ) v_indices[static_cast<std::size_t>( i )] = i;
        std::vector<PetscScalar> u_values( values->u.size() );
        std::vector<PetscScalar> v_values( values->v.size() );
        petsc_check( VecGetValues( u_field.vector, u_size, u_indices.data(), u_values.data() ), "read U trajectory" );
        petsc_check( VecGetValues( v_field.vector, v_size, v_indices.data(), v_values.data() ), "read V trajectory" );
        for( std::size_t i = 0; i < values->u.size(); ++i ) values->u[i] = PetscRealPart( u_values[i] );
        for( std::size_t i = 0; i < values->v.size(); ++i ) values->v[i] = PetscRealPart( v_values[i] );
    }
    return { l2_error( *u_field.system, u_field.vector, [final_time]( const Point& point ) { return u_exact( point, final_time ); } ),
              l2_error( *v_field.system, v_field.vector, [final_time]( const Point& point ) { return v_exact( point, final_time ); } ) };

}

void verify_strong_form_and_flux()
{
    const std::array<Point, 3> points{ Point( 0.13, 0.27, 0 ), Point( 0.41, 0.73, 0 ), Point( 0.82, 0.19, 0 ) };
    for( const auto& point : points )
        for( const Real time : { 0., 0.17, 0.63 } )
        {
            const Real u = u_exact( point, time );
            const Real v = v_exact( point, time );
            require( std::abs( u_time_derivative( point, time ) - du * u_laplacian( point, time ) + u * v * v - feed * ( 1 - u ) - u_source( point, time ) ) < 128 * std::numeric_limits<Real>::epsilon(), "U strong-form source/sign identity failed" );
            require( std::abs( v_time_derivative( point, time ) - dv * v_laplacian( point, time ) - u * v * v + ( feed + kill ) * v - v_source( point, time ) ) < 128 * std::numeric_limits<Real>::epsilon(), "V strong-form source/sign identity failed" );
        }
    for( const Real coordinate : { 0.11, 0.43, 0.89 } )
    {
        const Real tolerance = 16 * pi * std::numeric_limits<Real>::epsilon();
        require( std::abs( std::sin( pi ) * std::cos( pi * coordinate ) ) <= tolerance && std::abs( std::sin( 2 * pi ) * std::cos( pi * coordinate ) ) <= tolerance, "manufactured outward flux is nonzero" );
    }
}

/** The refinement study alone.
 *
 * This cannot be split further: an observed convergence rate is a property
 * of the whole sequence of meshes, so all three levels have to be solved in
 * one process. The mutation checks that used to live here did not share that
 * constraint and are now verify_mutation_detection().
 */
void verify_spatial_convergence()
{
    const std::array<std::array<unsigned, 2>, 3> meshes{ { { 8, 11 }, { 16, 19 }, { 32, 37 } } };
    std::array<Errors, 3> errors;
    for( std::size_t i = 0; i < meshes.size(); ++i )
    {
        const Real h_u = 1. / meshes[i][0];
        const Real h_v = 1. / meshes[i][1];
        const Real dt = std::min( h_u, h_v ) / 16;
        TrajectoryValues one_step;
        TrajectoryValues two_half_steps;
        errors[i] = solve( meshes[i][0], meshes[i][1], SplitScheme::Strang, dt, dt, false, false, &one_step );
        (void)solve( meshes[i][0], meshes[i][1], SplitScheme::Strang, dt, dt / 2, false, false, &two_half_steps );
        const Errors temporal_difference = trajectory_difference( one_step, two_half_steps );
        const Errors temporal_indicator{ h_u * temporal_difference.u, h_v * temporal_difference.v };
        const Real u_ratio = temporal_indicator.u / errors[i].u;
        const Real v_ratio = temporal_indicator.v / errors[i].v;
        std::cout << "spatial " << meshes[i][0] << ',' << meshes[i][1] << ": T=" << dt
                  << " U=" << errors[i].u << " V=" << errors[i].v << " combined=" << errors[i].combined()
                  << " temporal-indicator U=" << temporal_indicator.u << " V=" << temporal_indicator.v
                  << " ratios U=" << u_ratio << " V=" << v_ratio << '\n';
        require( std::isfinite( errors[i].u ) && std::isfinite( errors[i].v ) && errors[i].u > 0 && errors[i].v > 0,
                 "spatial MMS errors must be finite and positive" );
        require( std::isfinite( u_ratio ) && std::isfinite( v_ratio ) && u_ratio < temporal_contamination_fraction &&
                     v_ratio < temporal_contamination_fraction,
                 "one-step versus two-half-step temporal contamination exceeds ten percent of the spatial error" );
        if( i > 0 ) require( errors[i].u < errors[i - 1].u && errors[i].v < errors[i - 1].v && errors[i].combined() < errors[i - 1].combined(), "spatial MMS errors are not strictly decreasing" );
    }
    for( std::size_t i = 1; i < errors.size(); ++i )
    {
        const Real u_denominator = std::log( static_cast<Real>( meshes[i][0] ) / meshes[i - 1][0] );
        const Real v_denominator = std::log( static_cast<Real>( meshes[i][1] ) / meshes[i - 1][1] );
        const Real combined_denominator = std::log( std::min( 1. / static_cast<Real>( meshes[i - 1][0] ), 1. / static_cast<Real>( meshes[i - 1][1] ) ) /
                                                    std::min( 1. / static_cast<Real>( meshes[i][0] ), 1. / static_cast<Real>( meshes[i][1] ) ) );
        const Real u_rate = std::log( errors[i - 1].u / errors[i].u ) / u_denominator;
        const Real v_rate = std::log( errors[i - 1].v / errors[i].v ) / v_denominator;
        const Real combined_rate = std::log( errors[i - 1].combined() / errors[i].combined() ) / combined_denominator;
        std::cout << "spatial rates " << meshes[i - 1][0] << ',' << meshes[i - 1][1] << "->"
                  << meshes[i][0] << ',' << meshes[i][1] << ": denominators U=" << u_denominator
                  << " V=" << v_denominator << " combined=" << combined_denominator << "; rates U=" << u_rate
                  << " V=" << v_rate << " combined=" << combined_rate << '\n';
        require( u_rate >= 1.8 && v_rate >= 1.8 && combined_rate >= 1.8, "spatial MMS rate below 1.8" );
    }
}

/** Deliberately wrong physics must produce visibly worse error than the
 * baseline; independent of the refinement study, and cheaper, because it
 * runs at one medium resolution rather than up to 32x37. */
void verify_mutation_detection()
{
    const Real mutation_dt = std::min( 1. / 16., 1. / 19. ) / 16;
    const Errors mutation_baseline = solve( 16, 19, SplitScheme::Strang, mutation_final_time, mutation_dt );
    const Errors wrong_sign = solve( 16, 19, SplitScheme::Strang, mutation_final_time, mutation_dt, true );
    const Errors dropped = solve( 16, 19, SplitScheme::Strang, mutation_final_time, mutation_dt, false, true );
    std::cout << "mutations at T=" << mutation_final_time << ": baseline=" << mutation_baseline.combined()
              << " wrong-sign=" << wrong_sign.combined() << " dropped-transfer=" << dropped.combined() << '\n';
    const Real wrong_sign_excess = wrong_sign.combined() - mutation_baseline.combined();
    const Real dropped_transfer_excess = dropped.combined() - mutation_baseline.combined();
    require( std::isfinite( wrong_sign_excess ) && wrong_sign_excess > mutation_baseline.combined(),
             "wrong diffusion-sign mutation was not detected" );
    require( std::isfinite( dropped_transfer_excess ) && dropped_transfer_excess > mutation_baseline.combined(),
             "dropped-transfer mutation was not detected" );
}


Errors trajectory_difference( const TrajectoryValues& left, const TrajectoryValues& right )
{
    require( left.u.size() == right.u.size() && left.v.size() == right.v.size(),
             "temporal trajectories have incompatible layouts" );
    Real u_squared = 0;
    Real v_squared = 0;
    for( std::size_t i = 0; i < left.u.size(); ++i ) u_squared += std::pow( left.u[i] - right.u[i], 2 );
    for( std::size_t i = 0; i < left.v.size(); ++i ) v_squared += std::pow( left.v[i] - right.v[i], 2 );
    return { std::sqrt( u_squared ), std::sqrt( v_squared ) };
}

void verify_mesh_time_cross_product()
{
    constexpr unsigned u_cells = 4;
    constexpr unsigned v_cells = 5;
    const std::array<Real, 3> steps{ 0.025, 0.0125, 0.00625 };
    for( const auto scheme : { SplitScheme::Lie, SplitScheme::Strang } )
    {
        std::array<TrajectoryValues, 3> trajectories;
        for( std::size_t i = 0; i < steps.size(); ++i )
            (void)solve( u_cells, v_cells, scheme, 0.05, steps[i], false, false, &trajectories[i] );
        const Real coarse_difference = trajectory_difference( trajectories[0], trajectories[1] ).combined();
        const Real fine_difference = trajectory_difference( trajectories[1], trajectories[2] ).combined();
        const Real rate = std::log( coarse_difference / fine_difference ) / std::log( 2. );
        std::cout << ( scheme == SplitScheme::Lie ? "Lie" : "Strang" )
                  << " temporal self-differences=" << coarse_difference << ',' << fine_difference
                  << " rate=" << rate << '\n';
        require( std::isfinite( coarse_difference ) && std::isfinite( fine_difference ) &&
                     coarse_difference > fine_difference && fine_difference > 0,
                 "temporal self-convergence differences are not positive and decreasing" );
        const Real minimum_rate = scheme == SplitScheme::Lie ? 0.85 : 1.8;
        require( rate >= minimum_rate, "operator-split temporal self-convergence rate is below its regime threshold" );
    }
}

void verify_fixed_point_runtime()
{
    constexpr Real step = 0.05;
    KARMAMesh u_mesh( 2, PETSC_COMM_WORLD );
    KARMAMesh v_mesh( 2, PETSC_COMM_WORLD );
    u_mesh.generate_box( 4, 4, 0 );
    v_mesh.generate_box( 5, 5, 0 );
    TransientReactionDiffusionNativeAdapter u( "U", u_mesh, GrayScottSpecies::U, parameters() );
    TransientReactionDiffusionNativeAdapter v( "V", v_mesh, GrayScottSpecies::V, parameters() );
    KARMACouplingScheduler scheduler( compile_operator_split_graph( graph( SplitScheme::Lie ), PETSC_COMM_WORLD ),
                                      { &u, &v } );
    u.set_initial_condition( []( const Point& point ) { return u_exact( point, 0 ); } );
    v.set_initial_condition( []( const Point& point ) { return v_exact( point, 0 ); } );
    FixedPointOptions options;
    options.maximum_iterations = 30;
    options.absolute_tolerance = 1e-11;
    options.relative_tolerance = 1e-9;
    options.initial_relaxation = 0.8;
    options.use_aitken = true;
    const auto result = scheduler.advance_fixed_point( 0, step, options );
    if( result.status != FixedPointStatus::Converged )
    {
        std::cerr << "fixed-point status=" << static_cast<int>( result.status )
                  << " iterations=" << result.iterations << " residual=" << result.residual
                  << " message=" << result.message << '\n';
        for( const auto& metric : result.fields )
            std::cerr << metric.id << " residual=" << metric.residual << " scale=" << metric.scale
                      << " tolerance=" << metric.tolerance << '\n';
    }
    require( result.status == FixedPointStatus::Converged, "Gray-Scott strong-coupling fixed point did not converge" );
    require( result.fields.size() == 2 && result.fields[0].residual <= result.fields[0].tolerance &&
                 result.fields[1].residual <= result.fields[1].tolerance,
             "Gray-Scott fixed-point field metrics did not satisfy their tolerances" );
    const auto setup_counts = scheduler.connection_setup_counts();
    require( setup_counts[0] == 1 && setup_counts[1] == 1,
             "fixed-point sweeps rebuilt a compiled connection" );
    require( scheduler.position().step == 1 && std::abs( scheduler.position().time - step ) < 1e-14,
             "fixed-point scheduler did not publish one macro step" );
    const auto u_field = std::get<PetscFieldBinding>( u.output_field( "state", FieldVisibility::Committed ).binding() );
    const auto v_field = std::get<PetscFieldBinding>( v.output_field( "state", FieldVisibility::Committed ).binding() );
    const Errors strong{
        l2_error( *u_field.system, u_field.vector, []( const Point& point ) { return u_exact( point, step ); } ),
        l2_error( *v_field.system, v_field.vector, []( const Point& point ) { return v_exact( point, step ); } ) };
    const Errors weak = solve( 4, 5, SplitScheme::Lie, step, step );
    require( std::isfinite( strong.combined() ) && std::isfinite( weak.combined() ),
             "Gray-Scott weak/strong MMS errors are nonfinite" );
    std::cout << "fixed-point Gray-Scott MMS: iterations=" << result.iterations
              << " relaxation=" << result.relaxation << " weighted-residuals="
              << result.fields[0].residual << ',' << result.fields[1].residual
              << " weak-error=" << weak.combined() << " strong-error=" << strong.combined() << '\n';
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) != PETSC_SUCCESS ) return 1;
    int status = 0;
    try
    {
        int ranks = 0;
        petsc_check( MPI_Comm_size( PETSC_COMM_WORLD, &ranks ), "query MMS communicator size" );
        const std::string section = argc == 2 ? std::string( argv[1] ) : std::string( "--all" );
        require( argc <= 2, "usage: karma_gray_scott_spatial_mms_test [section]" );

        if( section == "--fixed-point" ) verify_fixed_point_runtime();
        else
        {
            require( ranks == 1, "Gray-Scott spatial MMS is serial; independently partitioned trajectories are tested separately" );

            // --strong-form and --spatial-convergence split what used to be
            // one ~195s serial test into two independent checks.
            // They are independent checks, so CTest can run them concurrently. They share no
            // state: each builds its own meshes and systems. --all preserves
            // the original single-run behavior for interactive use.
            //
            // --temporal-self-convergence keeps its historical meaning (the
            // strong-form and spatial-convergence checks first, then the
            // mesh/time cross product) so an existing invocation of it does
            // not silently start checking less than it did.
            if( section == "--all" || section == "--strong-form" || section == "--temporal-self-convergence" )
                verify_strong_form_and_flux();
            if( section == "--all" || section == "--spatial-convergence" ||
                section == "--temporal-self-convergence" )
                verify_spatial_convergence();
            if( section == "--all" || section == "--mutation-detection" ||
                section == "--temporal-self-convergence" )
                verify_mutation_detection();
            if( section == "--temporal-self-convergence" ) verify_mesh_time_cross_product();

            require( section == "--all" || section == "--strong-form" || section == "--spatial-convergence" ||
                         section == "--mutation-detection" || section == "--temporal-self-convergence",
                     "usage: karma_gray_scott_spatial_mms_test "
                     "[--strong-form|--spatial-convergence|--mutation-detection"
                     "|--temporal-self-convergence|--fixed-point]" );
        }
    }
    catch( const std::exception& error )
    {
        std::cerr << "Gray-Scott spatial MMS failure: " << error.what() << '\n';
        status = 1;
    }
    PetscFinalize();
    return status;
}
