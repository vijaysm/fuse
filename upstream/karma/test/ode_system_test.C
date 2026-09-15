#include "ode_system.h"
#include "ode_mass_operator.h"
#include "ode_system_operator.h"
#include "karma_exceptions.h"

#include <petsc.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
int failures = 0;
void check( bool condition, const char* message )
{
    if( !condition ) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}

template < class Function > bool throws_exception( Function&& function )
{
    try { function(); } catch( const std::exception& ) { return true; }
    return false;
}

void set_values( Vec vector, const std::vector< PetscScalar >& values )
{
    std::vector< PetscInt > indices( values.size() );
    for( std::size_t i = 0; i < values.size(); ++i ) indices[i] = static_cast< PetscInt >( i );
    VecSetValues( vector, static_cast< PetscInt >( values.size() ), indices.data(), values.data(), INSERT_VALUES );
    VecAssemblyBegin( vector ); VecAssemblyEnd( vector );
}

std::vector< PetscScalar > get_values( Vec vector, PetscInt size )
{
    PetscInt first = 0, last = 0; VecGetOwnershipRange( vector, &first, &last );
    std::vector< PetscScalar > local( static_cast< std::size_t >( size ), 0 ), global( local.size(), 0 );
    for( PetscInt i = first; i < last; ++i ) VecGetValues( vector, 1, &i, &local[static_cast< std::size_t >( i )] );
    MPI_Allreduce( local.data(), global.data(), size, MPIU_SCALAR, MPI_SUM,
                   PetscObjectComm( reinterpret_cast< PetscObject >( vector ) ) );
    return global;
}

std::vector< PetscScalar > get_matrix( Mat matrix, PetscInt size )
{
    PetscInt first = 0, last = 0; MatGetOwnershipRange( matrix, &first, &last );
    std::vector< PetscScalar > local( static_cast< std::size_t >( size * size ), 0 ), global( local.size(), 0 );
    for( PetscInt i = first; i < last; ++i )
        for( PetscInt j = 0; j < size; ++j )
            MatGetValues( matrix, 1, &i, 1, &j, &local[static_cast< std::size_t >( i * size + j )] );
    MPI_Allreduce( local.data(), global.data(), size * size, MPIU_SCALAR, MPI_SUM,
                   PetscObjectComm( reinterpret_cast< PetscObject >( matrix ) ) );
    return global;
}

KARMA::ODEModel decay_model()
{
    KARMA::ODEModel model;
    model.rhs = []( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) { rhs( 0 ) = -2 * state( 0 ); };
    model.jacobian = []( Real, const KARMA::RealDenseVector&, KARMA::RealDenseMatrix& jacobian ) { jacobian( 0, 0 ) = -2; };
    return model;
}

double solve_decay( double step )
{
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, decay_model() );
    system.init(); set_values( system.solution(), { 1 } ); system.solve( 0, 1, step );
    return PetscRealPart( get_values( system.solution(), 1 )[0] );
}

void test_decay_refinement()
{
    PetscOptionsSetValue( nullptr, "-ts_type", "beuler" );
    PetscOptionsSetValue( nullptr, "-ts_exact_final_time", "stepover" );
    const std::array< double, 4 > steps{ .2, .1, .05, .025 };
    std::array< double, 4 > errors{};
    for( std::size_t i = 0; i < steps.size(); ++i )
    {
        errors[i] = std::abs( solve_decay( steps[i] ) - std::exp( -2.0 ) );
        check( std::isfinite( errors[i] ), "Crank-Nicolson decay error is finite" );
        if( i ) check( errors[i] < errors[i - 1], "Crank-Nicolson decay error decreases under refinement" );
    }
    for( std::size_t i = 1; i < steps.size(); ++i )
    {
        const double rate = std::log( errors[i - 1] / errors[i] ) / std::log( 2.0 );
        check( rate >= 1.8 && rate <= 2.2, "Crank-Nicolson decay has second-order convergence" );
    }
    check( errors.back() < 6.e-5, "finest scalar decay error meets the absolute bound" );
    check( std::abs( solve_decay( .002 ) - std::exp( -2.0 ) ) < 1.e-6, "scalar decay retains the accurate endpoint check" );
    PetscOptionsClearValue( nullptr, "-ts_type" );
    PetscOptionsClearValue( nullptr, "-ts_exact_final_time" );
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    if( rank == 0 )
    {
        std::cout << "CN errors:";
        for( const double error : errors ) std::cout << ' ' << error;
        std::cout << "; rates:";
        for( std::size_t i = 1; i < steps.size(); ++i )
            std::cout << ' ' << std::log( errors[i - 1] / errors[i] ) / std::log( 2.0 );
        std::cout << '\n';
    }
}

void test_repeated_solve()
{
    PetscOptionsSetValue( nullptr, "-ts_max_steps", "5" );
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, decay_model() );
    system.init(); PetscOptionsClearValue( nullptr, "-ts_max_steps" );
    set_values( system.solution(), { 1 } ); system.solve( 0, 1, .2 );
    const double first = PetscRealPart( get_values( system.solution(), 1 )[0] );
    set_values( system.solution(), { 1 } );
    const bool succeeded = !throws_exception( [&] { system.solve( 0, 1, .2 ); } );
    const double second = PetscRealPart( get_values( system.solution(), 1 )[0] );
    check( succeeded && std::abs( first - second ) < 1.e-14,
           "repeated solve restarts at step zero and preserves max_steps" );
}

KARMA::ODEModel nonlinear_model( bool analytic )
{
    KARMA::ODEModel model;
    model.rhs = []( Real time, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) {
        rhs( 0 ) = state( 0 ) * state( 1 ) + time;
        rhs( 1 ) = std::sin( state( 0 ) ) - state( 1 ) * state( 1 );
    };
    if( analytic ) model.jacobian = []( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseMatrix& jacobian ) {
        jacobian( 0, 0 ) = state( 1 ); jacobian( 0, 1 ) = state( 0 );
        jacobian( 1, 0 ) = std::cos( state( 0 ) ); jacobian( 1, 1 ) = -2 * state( 1 );
    };
    model.mass = []( Real time, KARMA::RealDenseMatrix& mass ) {
        mass( 0, 0 ) = 2 + time; mass( 0, 1 ) = .5; mass( 1, 0 ) = -.25; mass( 1, 1 ) = 1 + time;
    };
    return model;
}

void test_dense_entries()
{
    constexpr double time = .3, shift = 1.7;
    KARMA::ODESystem analytic( PETSC_COMM_WORLD, { "u", "v" }, nonlinear_model( true ) );
    analytic.init(); set_values( analytic.solution(), { 1.2, -.7 } );
    Vec derivative = analytic.system().create_vector(), residual = analytic.system().create_vector();
    set_values( derivative, { .4, -.2 } ); analytic.residual( time, analytic.solution(), derivative, residual );
    const auto result = get_values( residual, 2 );
    check( std::abs( PetscRealPart( result[0] ) - ( 2.3 * .4 + .5 * -.2 - ( 1.2 * -.7 + .3 ) ) ) < 1.e-12 &&
           std::abs( PetscRealPart( result[1] ) - ( -.25 * .4 + 1.3 * -.2 - ( std::sin( 1.2 ) - .49 ) ) ) < 1.e-12,
           "residual equals M(t) udot minus f(t,u) entry by entry" );
    analytic.jacobian( time, analytic.solution(), shift, analytic.operator_matrix() );
    const std::array< double, 4 > expected{ shift * 2.3 - -.7, shift * .5 - 1.2,
                                            shift * -.25 - std::cos( 1.2 ), shift * 1.3 - 1.4 };
    const auto exact = get_matrix( analytic.operator_matrix(), 2 );
    for( std::size_t i = 0; i < expected.size(); ++i )
        check( std::abs( PetscRealPart( exact[i] ) - expected[i] ) < 1.e-12, "analytic Jacobian entry is exact" );
    KARMA::ODESystem fallback( PETSC_COMM_WORLD, { "u", "v" }, nonlinear_model( false ) );
    fallback.init(); set_values( fallback.solution(), { 1.2, -.7 } );
    fallback.jacobian( time, fallback.solution(), shift, fallback.operator_matrix() );
    const auto finite_difference = get_matrix( fallback.operator_matrix(), 2 );
    for( std::size_t i = 0; i < expected.size(); ++i )
        check( std::abs( PetscRealPart( finite_difference[i] ) - expected[i] ) < 2.e-9,
               "fallback coupled Jacobian agrees with exact values" );
    VecDestroy( &residual ); VecDestroy( &derivative );
}

void test_argument_validation_and_adapters()
{
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, decay_model() );
    system.init();
    KARMA::ODESystemOperator system_operator( system );
    KARMA::ODEMassOperator mass_operator( system );
    Vec derivative = system.system().create_vector(), output = system.system().create_vector();
    Vec wrong = nullptr;
    VecCreateMPI( PETSC_COMM_WORLD, PETSC_DECIDE, 2, &wrong );
    VecSet( output, 37 );
    check( throws_exception( [&] { system_operator.residual( 0, wrong, derivative, output ); } ),
           "oversized ODE state is rejected" );
    PetscReal norm = 0; VecNorm( output, NORM_INFINITY, &norm );
    check( std::abs( norm - 37 ) < 1.e-12, "invalid residual arguments do not mutate output" );
    Mat wrong_matrix = nullptr;
    MatCreateAIJ( PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, 2, 2, 2, nullptr, 0, nullptr, &wrong_matrix );
    MatSetUp( wrong_matrix ); MatAssemblyBegin( wrong_matrix, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( wrong_matrix, MAT_FINAL_ASSEMBLY ); MatShift( wrong_matrix, 29 );
    check( throws_exception( [&] { system_operator.jacobian( 0, system.solution(), 1, wrong_matrix ); } ),
           "wrong-sized ODE Jacobian matrix is rejected" );
    MatNorm( wrong_matrix, NORM_INFINITY, &norm );
    check( std::abs( norm - 29 ) < 1.e-12, "invalid Jacobian arguments do not mutate output" );
    const double nan = std::numeric_limits< double >::quiet_NaN();
    set_values( system.solution(), { nan } ); VecSet( output, 31 );
    check( throws_exception( [&] { system.residual( 0, system.solution(), derivative, output ); } ),
           "nonfinite ODE state is rejected" );
    check( throws_exception( [&] { system.jacobian( nan, system.solution(), 1, system.operator_matrix() ); } ),
           "nonfinite ODE time is rejected" );
    check( throws_exception( [&] { system.jacobian( 0, system.solution(), nan, system.operator_matrix() ); } ),
           "nonfinite ODE shift is rejected" );
    check( throws_exception( [&] { system.solve( 0, std::numeric_limits< double >::infinity(), .1 ); } ),
           "nonfinite ODE final time is rejected" );
    set_values( system.solution(), { 1 } );
    mass_operator.assemble( 0, system.solution(), system.mass() );
    mass_operator.multiply( system.mass(), system.solution(), output );
    check( std::abs( PetscRealPart( get_values( output, 1 )[0] ) - 1 ) < 1.e-12,
           "ODE mass adapter delegates assembly and multiply" );
    mass_operator.solve( system.mass(), output, derivative );
    check( std::abs( PetscRealPart( get_values( derivative, 1 )[0] ) - 1 ) < 1.e-12,
            "ODE mass adapter delegates solve" );
    VecSet( output, 43 );
    check( throws_exception( [&] { mass_operator.multiply( system.mass(), wrong, output ); } ),
           "ODE mass multiply rejects a wrong-sized input" );
    VecNorm( output, NORM_INFINITY, &norm );
    check( std::abs( norm - 43 ) < 1.e-12, "invalid mass multiply leaves output unchanged" );
    VecSet( derivative, 47 );
    check( throws_exception( [&] { mass_operator.solve( wrong_matrix, output, derivative ); } ),
           "ODE mass solve rejects a wrong-sized matrix" );
    VecNorm( derivative, NORM_INFINITY, &norm );
    check( std::abs( norm - 47 ) < 1.e-12, "invalid mass solve leaves output unchanged" );
    system.clear();
    check( !system.is_initialized(), "ODE clear releases initialized state" );
    system.init();
    check( system.is_initialized(), "ODE can reinitialize after clear" );
    MatDestroy( &wrong_matrix ); VecDestroy( &wrong ); VecDestroy( &output ); VecDestroy( &derivative );
}

void test_callback_validation()
{
    const double nan = std::numeric_limits< double >::quiet_NaN();
    auto expect_rhs_failure = [&]( KARMA::ODERightHandSide rhs, const char* message ) {
        KARMA::ODEModel model; model.rhs = std::move( rhs );
        KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, std::move( model ) ); system.init();
        Vec derivative = system.system().create_vector(), residual = system.system().create_vector();
        check( throws_exception( [&] { system.residual( 0, system.solution(), derivative, residual ); } ), message );
        PetscReal norm = -1; check( VecSet( residual, 0 ) == PETSC_SUCCESS && VecNorm( residual, NORM_INFINITY, &norm ) == PETSC_SUCCESS,
                                    "RHS failure leaves output usable" );
        VecDestroy( &residual ); VecDestroy( &derivative );
    };
    expect_rhs_failure( [nan]( Real, const KARMA::RealDenseVector&, KARMA::RealDenseVector& rhs ) { rhs( 0 ) = nan; }, "nonfinite RHS is rejected" );
    expect_rhs_failure( []( Real, const KARMA::RealDenseVector&, KARMA::RealDenseVector& ) { throw std::runtime_error( "rhs" ); }, "throwing RHS propagates" );
    expect_rhs_failure( []( Real, const KARMA::RealDenseVector&, KARMA::RealDenseVector& rhs ) { rhs.resize( 2 ); }, "wrong-sized RHS is rejected" );

    auto expect_jacobian_failure = [&]( KARMA::ODEModel model, const char* message ) {
        KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, std::move( model ) ); system.init(); set_values( system.solution(), { 2 } );
        check( throws_exception( [&] { system.jacobian( 0, system.solution(), 1, system.operator_matrix() ); } ), message );
        PetscReal norm = -1;
        const PetscErrorCode zero_error = MatZeroEntries( system.operator_matrix() );
        const PetscErrorCode begin_error = MatAssemblyBegin( system.operator_matrix(), MAT_FINAL_ASSEMBLY );
        const PetscErrorCode end_error = MatAssemblyEnd( system.operator_matrix(), MAT_FINAL_ASSEMBLY );
        check( zero_error == PETSC_SUCCESS && begin_error == PETSC_SUCCESS && end_error == PETSC_SUCCESS &&
                   MatNorm( system.operator_matrix(), NORM_INFINITY, &norm ) == PETSC_SUCCESS,
               "Jacobian failure leaves output usable" );
    };
    auto model = decay_model(); model.jacobian = [nan]( Real, const KARMA::RealDenseVector&, KARMA::RealDenseMatrix& value ) { value( 0, 0 ) = nan; };
    expect_jacobian_failure( std::move( model ), "nonfinite analytic Jacobian is rejected" );
    model = decay_model(); model.jacobian = []( Real, const KARMA::RealDenseVector&, KARMA::RealDenseMatrix& ) { throw std::runtime_error( "jacobian" ); };
    expect_jacobian_failure( std::move( model ), "throwing analytic Jacobian propagates" );
    model = decay_model(); model.jacobian = []( Real, const KARMA::RealDenseVector&, KARMA::RealDenseMatrix& value ) { value.resize( 2, 2 ); };
    expect_jacobian_failure( std::move( model ), "wrong-sized analytic Jacobian is rejected" );
    model = {}; model.rhs = [nan]( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) { rhs( 0 ) = state( 0 ) == 2 ? nan : state( 0 ); };
    expect_jacobian_failure( std::move( model ), "fallback validates base RHS" );
    model = {}; model.rhs = [nan]( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) { rhs( 0 ) = state( 0 ) == 2 ? 2 : nan; };
    expect_jacobian_failure( std::move( model ), "fallback validates probe RHS" );
    model = {}; model.rhs = []( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) {
        rhs( 0 ) = state( 0 ) == 2 ? 0 : std::copysign( std::numeric_limits< double >::max(), state( 0 ) - 2 );
    };
    expect_jacobian_failure( std::move( model ), "fallback validates computed derivative entries" );

    std::vector< double > probes;
    model = {};
    model.rhs = [&probes]( Real, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) {
        probes.push_back( state( 0 ) ); rhs( 0 ) = state( 0 );
    };
    KARMA::ODESystem scaled( PETSC_COMM_WORLD, { "y" }, std::move( model ) );
    scaled.init(); set_values( scaled.solution(), { 1.e6 } );
    scaled.jacobian( 0, scaled.solution(), 1, scaled.operator_matrix() );
    const double expected_step = std::cbrt( std::numeric_limits< Real >::epsilon() ) * 1.e6;
    check( probes.size() == 3 && std::abs( probes[1] - 1.e6 - expected_step ) < expected_step * 1.e-6 &&
               std::abs( probes[2] - 1.e6 + expected_step ) < expected_step * 1.e-6,
           "fallback uses component-scaled cbrt-epsilon central probes" );

    auto expect_mass_failure = [&]( KARMA::ODEMass mass, const char* message ) {
        auto mass_model = decay_model(); mass_model.mass = [mass = std::move( mass )]( Real time, KARMA::RealDenseMatrix& value ) {
            if( time == 0 ) value( 0, 0 ) = 1; else mass( time, value );
        };
        KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, std::move( mass_model ) ); system.init();
        check( throws_exception( [&] { system.assemble_mass( 1, system.solution(), system.mass() ); } ), message );
        PetscReal norm = -1; check( MatZeroEntries( system.mass() ) == PETSC_SUCCESS && MatNorm( system.mass(), NORM_INFINITY, &norm ) == PETSC_SUCCESS,
                                    "mass failure leaves output usable" );
    };
    expect_mass_failure( [nan]( Real, KARMA::RealDenseMatrix& value ) { value( 0, 0 ) = nan; }, "nonfinite mass is rejected" );
    expect_mass_failure( []( Real, KARMA::RealDenseMatrix& ) { throw std::runtime_error( "mass" ); }, "throwing mass propagates" );
    expect_mass_failure( []( Real, KARMA::RealDenseMatrix& value ) { value.resize( 2, 2 ); }, "wrong-sized mass is rejected" );

    auto invalid_init_model = decay_model();
    invalid_init_model.mass = []( Real, KARMA::RealDenseMatrix& value ) {
        value( 0, 0 ) = std::numeric_limits< double >::quiet_NaN();
    };
    KARMA::ODESystem invalid_init( PETSC_COMM_WORLD, { "y" }, std::move( invalid_init_model ) );
    check( throws_exception( [&] { invalid_init.init(); } ) && !invalid_init.is_initialized(),
           "mass callback failure rolls back ODE initialization" );
}

void test_mpi_object_validation()
{
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, decay_model() );
    system.init();
    Vec self_state = nullptr, output = system.system().create_vector(), derivative = system.system().create_vector();
    VecCreateSeq( PETSC_COMM_SELF, 1, &self_state );
    VecSet( output, 41 );
    check( throws_exception( [&] { system.residual( 0, self_state, derivative, output ); } ),
           "PETSC_COMM_SELF state is collectively rejected by WORLD ODE system" );
    PetscReal norm = 0; VecNorm( output, NORM_INFINITY, &norm );
    check( std::abs( norm - 41 ) < 1.e-12, "communicator rejection leaves ODE output unchanged" );
    KARMA::ODEMassOperator mass_operator( system );
    VecSet( output, 43 );
    check( throws_exception( [&] { mass_operator.multiply( system.mass(), self_state, output ); } ),
           "ODE mass multiply rejects a wrong-communicator input" );
    VecNorm( output, NORM_INFINITY, &norm );
    check( std::abs( norm - 43 ) < 1.e-12, "wrong-communicator mass multiply leaves output unchanged" );
    VecSet( derivative, 47 );
    check( throws_exception( [&] { mass_operator.solve( system.mass(), self_state, derivative ); } ),
           "ODE mass solve rejects a wrong-communicator right-hand side" );
    VecNorm( derivative, NORM_INFINITY, &norm );
    check( std::abs( norm - 47 ) < 1.e-12, "wrong-communicator mass solve leaves output unchanged" );
    VecDestroy( &derivative ); VecDestroy( &output ); VecDestroy( &self_state );
}

void test_asymmetric_init( const std::string& mode )
{
    int rank = 0; MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    auto model = decay_model();
    std::vector< std::string > unknowns{ "y" };
    if( rank == 1 && mode == "empty" ) unknowns.clear();
    if( rank == 1 && mode == "name" ) unknowns[0] = "z";
    if( rank == 1 && mode == "model" ) model.rhs = {};
    KARMA::ODESystem system( PETSC_COMM_WORLD, std::move( unknowns ), std::move( model ) );
    check( throws_exception( [&] { system.init(); } ) && !system.is_initialized(),
           "asymmetric ODE initialization fails collectively before PETSc creation" );
}

void test_asymmetric_scalar( const std::string& mode )
{
    int rank = 0; MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    const bool nonfinite = mode.ends_with( "-nonfinite" );
    const std::string kind = nonfinite ? mode.substr( 0, mode.size() - 10 ) : mode;
    const Real altered = nonfinite ? std::numeric_limits< Real >::quiet_NaN() : 0.25;
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, decay_model() );
    system.init(); set_values( system.solution(), { 1 } );
    Vec derivative = system.system().create_vector(), output = system.system().create_vector();
    constexpr PetscReal output_sentinel = 53;
    constexpr PetscReal jacobian_sentinel = 59;
    check( VecSet( output, output_sentinel ) == PETSC_SUCCESS,
           "asymmetric ODE test initializes the residual sentinel" );
    if( kind == "jacobian-time" || kind == "shift" )
    {
        check( MatAssemblyBegin( system.operator_matrix(), MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
               "asymmetric ODE test begins initial Jacobian assembly" );
        check( MatAssemblyEnd( system.operator_matrix(), MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
               "asymmetric ODE test ends initial Jacobian assembly" );
        check( MatShift( system.operator_matrix(), jacobian_sentinel ) == PETSC_SUCCESS,
               "asymmetric ODE test initializes the Jacobian sentinel" );
        PetscReal initial_jacobian_norm = 0;
        check( MatNorm( system.operator_matrix(), NORM_INFINITY, &initial_jacobian_norm ) == PETSC_SUCCESS &&
                   initial_jacobian_norm == jacobian_sentinel,
               "asymmetric ODE test observes the exact Jacobian sentinel before validation" );
    }
    bool rejected = false;
    if( kind == "mass-time" )
        rejected = throws_exception( [&] { system.assemble_mass( rank == 1 ? altered : 0, system.solution(), system.mass() ); } );
    else if( kind == "residual-time" )
        rejected = throws_exception( [&] { system.residual( rank == 1 ? altered : 0, system.solution(), derivative, output ); } );
    else if( kind == "jacobian-time" )
        rejected = throws_exception( [&] { system.jacobian( rank == 1 ? altered : 0, system.solution(), 1, system.operator_matrix() ); } );
    else if( kind == "shift" )
        rejected = throws_exception( [&] { system.jacobian( 0, system.solution(), rank == 1 ? altered : 1, system.operator_matrix() ); } );
    else if( kind == "initial-time" )
        rejected = throws_exception( [&] { system.solve( rank == 1 ? altered : 0, 1, .1 ); } );
    else if( kind == "final-time" )
        rejected = throws_exception( [&] { system.solve( 0, rank == 1 ? altered : 1, .1 ); } );
    else if( kind == "dt" )
        rejected = throws_exception( [&] { system.solve( 0, 1, rank == 1 ? altered : .1 ); } );
    check( rejected, "asymmetric ODE scalar is rejected collectively" );
    PetscReal retained = 0;
    if( kind == "residual-time" )
    {
        VecNorm( output, NORM_INFINITY, &retained );
        check( std::abs( retained - 53 ) < 1.e-12, "scalar validation leaves residual output unchanged" );
    }
    else if( kind == "jacobian-time" || kind == "shift" )
    {
        const PetscErrorCode norm_error = MatNorm( system.operator_matrix(), NORM_INFINITY, &retained );
        check( norm_error == PETSC_SUCCESS && retained == jacobian_sentinel,
               "scalar validation leaves the exact Jacobian sentinel unchanged" );
    }
    else if( kind == "initial-time" || kind == "final-time" || kind == "dt" )
    {
        VecNorm( system.solution(), NORM_INFINITY, &retained );
        check( std::abs( retained - 1 ) < 1.e-12, "scalar validation leaves ODE solution unchanged" );
    }
    VecDestroy( &output ); VecDestroy( &derivative );
}

void test_asymmetric_failure( const std::string& mode )
{
    int rank = 0; MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    const bool nonfinite = mode.ends_with( "-nonfinite" );
    const std::string kind = nonfinite ? mode.substr( 0, mode.size() - 10 ) : mode;
    auto model = decay_model();
    if( kind == "rhs" ) model.rhs = [rank, nonfinite]( Real time, const KARMA::RealDenseVector& state, KARMA::RealDenseVector& rhs ) { if( rank == 1 && time > 0 ) { if( !nonfinite ) throw std::runtime_error( "asymmetric rhs" ); rhs( 0 ) = std::numeric_limits< double >::quiet_NaN(); } else rhs( 0 ) = -state( 0 ); };
    else if( kind == "jacobian" ) model.jacobian = [rank, nonfinite]( Real time, const KARMA::RealDenseVector&, KARMA::RealDenseMatrix& jacobian ) { if( rank == 1 && time > 0 ) { if( !nonfinite ) throw std::runtime_error( "asymmetric jacobian" ); jacobian( 0, 0 ) = std::numeric_limits< double >::quiet_NaN(); } else jacobian( 0, 0 ) = -1; };
    else model.mass = [rank, nonfinite]( Real time, KARMA::RealDenseMatrix& mass ) { if( rank == 1 && time > 0 ) { if( !nonfinite ) throw std::runtime_error( "asymmetric mass" ); mass( 0, 0 ) = std::numeric_limits< double >::quiet_NaN(); } else mass( 0, 0 ) = 1; };
    KARMA::ODESystem system( PETSC_COMM_WORLD, { "y" }, std::move( model ) ); system.init(); set_values( system.solution(), { 1 } );
    check( throws_exception( [&] { system.solve( 0, 1, .1 ); } ),
           "asymmetric TS callback failure is reported collectively" );
    PetscReal norm = -1; check( VecNorm( system.solution(), NORM_INFINITY, &norm ) == PETSC_SUCCESS, "asymmetric failure releases borrows" );
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string mode = argc > 1 ? argv[1] : "";
        if( mode.rfind( "--asymmetric-init-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, mode.c_str() );
            test_asymmetric_init( mode.substr( 18 ) );
        }
        else if( mode.rfind( "--asymmetric-scalar-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, mode.c_str() );
            test_asymmetric_scalar( mode.substr( 20 ) );
        }
        else if( mode.rfind( "--asymmetric-", 0 ) == 0 )
        {
            PetscOptionsClearValue( nullptr, mode.c_str() );
            test_asymmetric_failure( mode.substr( 13 ) );
        }
        else
        {
            int ranks = 1; MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
            test_decay_refinement(); test_repeated_solve(); test_dense_entries();
            if( ranks == 1 ) test_argument_validation_and_adapters();
            else test_mpi_object_validation();
            test_callback_validation();
        }
    }
    int global_failures = 0; MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize(); return global_failures == 0 ? 0 : 1;
}
