#include "ode_system.h"
#include "../common/impls/collective_signature.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

KARMA::RealDenseVector read_vector( Vec vector, PetscInt size )
{
    PetscInt first = 0, last = 0;
    check( VecGetOwnershipRange( vector, &first, &last ), "VecGetOwnershipRange(ODE read)" );
    KARMA::PetscVecArrayRead local( vector );
    std::vector< PetscScalar > values( static_cast< std::size_t >( size ) );
    for( PetscInt i = first; i < last; ++i ) values[static_cast< std::size_t >( i )] = local.get()[i - first];
    std::vector< PetscScalar > global( static_cast< std::size_t >( size ) );
    check( MPI_Allreduce( values.data(), global.data(), size, MPIU_SCALAR, MPI_SUM,
                          PetscObjectComm( reinterpret_cast< PetscObject >( vector ) ) ),
           "MPI_Allreduce(ODE state)" );
    KARMA::RealDenseVector result( static_cast< std::size_t >( size ) );
    for( PetscInt i = 0; i < size; ++i ) result( static_cast< std::size_t >( i ) ) = PetscRealPart( global[i] );
    return result;
}

void write_vector( Vec vector, const KARMA::RealDenseVector& values )
{
    PetscInt first = 0, last = 0;
    check( VecGetOwnershipRange( vector, &first, &last ), "VecGetOwnershipRange(ODE)" );
    for( PetscInt i = first; i < last; ++i )
        check( VecSetValue( vector, i, values( static_cast< std::size_t >( i ) ), INSERT_VALUES ),
               "VecSetValue(ODE)" );
    check( VecAssemblyBegin( vector ), "VecAssemblyBegin(ODE)" );
    check( VecAssemblyEnd( vector ), "VecAssemblyEnd(ODE)" );
}

void validate_vector( const KARMA::RealDenseVector& value, PetscInt size, const char* name )
{
    if( value.size() != static_cast< std::size_t >( size ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned the wrong vector dimension" ) );
    for( PetscInt i = 0; i < size; ++i )
        if( !std::isfinite( value( static_cast< std::size_t >( i ) ) ) )
            KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned a nonfinite entry" ) );
}

void validate_matrix( const KARMA::RealDenseMatrix& value, PetscInt size, const char* name )
{
    if( value.rows() != static_cast< std::size_t >( size ) || value.cols() != static_cast< std::size_t >( size ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned the wrong matrix dimensions" ) );
    for( PetscInt i = 0; i < size; ++i )
        for( PetscInt j = 0; j < size; ++j )
            if( !std::isfinite( value( static_cast< std::size_t >( i ), static_cast< std::size_t >( j ) ) ) )
                KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned a nonfinite entry" ) );
}

void collective_check( MPI_Comm communicator, const std::string& local_error, std::exception_ptr local_exception );

void validate_communicator( PetscObject object, MPI_Comm expected, const char* name )
{
    int relation = MPI_UNEQUAL;
    check( MPI_Comm_compare( PetscObjectComm( object ), expected, &relation ), "MPI_Comm_compare(ODE)" );
    if( relation != MPI_IDENT && relation != MPI_CONGRUENT )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " communicator is not congruent with the ODE system" ) );
}

void validate_vector_object( Vec vector, MPI_Comm communicator, PetscInt global_size, PetscInt local_size,
                             const char* name, bool finite )
{
    std::string local_error;
    try
    {
        if( !vector ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " must not be null" ) );
        validate_communicator( reinterpret_cast< PetscObject >( vector ), communicator, name );
        PetscInt actual_global = 0, actual_local = 0;
        check( VecGetSize( vector, &actual_global ), "VecGetSize(ODE validation)" );
        check( VecGetLocalSize( vector, &actual_local ), "VecGetLocalSize(ODE validation)" );
        if( actual_global != global_size || actual_local != local_size )
            KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " has incompatible global or local size" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown ODE vector validation failure"; }
    collective_check( communicator, local_error, nullptr );
    if( finite )
    {
        PetscReal norm = 0;
        check( VecNorm( vector, NORM_INFINITY, &norm ), "VecNorm(ODE validation)" );
        if( !std::isfinite( norm ) ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " contains a nonfinite entry" ) );
    }
}

void validate_matrix_object( Mat matrix, MPI_Comm communicator, PetscInt global_size, PetscInt local_size,
                             const char* name )
{
    std::string local_error;
    try
    {
        if( !matrix ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " must not be null" ) );
        validate_communicator( reinterpret_cast< PetscObject >( matrix ), communicator, name );
        PetscInt rows = 0, columns = 0, local_rows = 0, local_columns = 0;
        check( MatGetSize( matrix, &rows, &columns ), "MatGetSize(ODE validation)" );
        check( MatGetLocalSize( matrix, &local_rows, &local_columns ), "MatGetLocalSize(ODE validation)" );
        if( rows != global_size || columns != global_size || local_rows != local_size || local_columns != local_size )
            KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " has incompatible global or local dimensions" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown ODE matrix validation failure"; }
    collective_check( communicator, local_error, nullptr );
}

void validate_scalar( MPI_Comm communicator, KARMA::Real value, const char* name )
{
    int ranks = 1;
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(ODE scalar)" );
    std::vector< KARMA::Real > values;
    KARMA::detail::prepare_collectively( communicator, [&] { values.resize( static_cast< std::size_t >( ranks ) ); },
                                         "ODE scalar gather allocation failed" );
    check( MPI_Allgather( &value, 1, MPIU_REAL, values.data(), 1, MPIU_REAL, communicator ),
           "MPI_Allgather(ODE scalar)" );
    for( int rank = 0; rank < ranks; ++rank )
    {
        if( !std::isfinite( values[static_cast< std::size_t >( rank )] ) )
            KARMA_THROW( KARMA::InvalidArgument( "rank " + std::to_string( rank ) + ": " + name + " must be finite" ) );
        if( values[static_cast< std::size_t >( rank )] != values[0] )
            KARMA_THROW( KARMA::InvalidArgument( "rank " + std::to_string( rank ) + ": " + name +
                                                 " differs from rank 0" ) );
    }
}

void collective_check( MPI_Comm communicator, const std::string& local_error, std::exception_ptr local_exception )
{
    int ranks = 1;
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(ODE callback)" );
    if( ranks == 1 && local_exception ) std::rethrow_exception( local_exception );
    KARMA::detail::collective_signature_error( communicator, local_error );
}

template < class Function >
void evaluate_collectively( MPI_Comm communicator, Function&& function, const char* unknown_error )
{
    std::string local_error;
    std::exception_ptr local_exception;
    try
    {
        function();
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
        local_exception = std::current_exception();
    }
    catch( ... )
    {
        local_error = unknown_error;
        local_exception = std::current_exception();
    }
    collective_check( communicator, local_error, local_exception );
}

void require_matching_signature( MPI_Comm communicator, const std::string& signature )
{
    KARMA::detail::require_collective_signature(
        communicator, {}, [&] { return signature; }, "ODE signature preparation failed",
        "ODE unknown/model signature differs from rank 0" );
}
}  // namespace

namespace KARMA
{
ODESystem::ODESystem( MPI_Comm communicator, std::vector< std::string > unknowns, ODEModel model )
    : _communicator( communicator ), _unknowns( std::move( unknowns ) ), _properties( std::move( model ) )
{}

void ODESystem::init()
{
    if( _initialized ) return;
    std::string local_error;
    if( _unknowns.empty() ) local_error = "ODE system requires at least one unknown";
    else if( !_properties.is_ready() ) local_error = "ODE model requires a right-hand-side callback";
    std::string signature = _properties.is_ready() ? "rhs;" : "no-rhs;";
    signature += _properties.model().jacobian ? "jacobian;" : "no-jacobian;";
    signature += _properties.model().mass ? "mass;" : "no-mass;";
    for( const auto& unknown : _unknowns )
    {
        if( unknown.empty() && local_error.empty() ) local_error = "ODE unknown name must not be empty";
        signature += std::to_string( unknown.size() ) + ":" + unknown + ";";
    }
    evaluate_collectively( _communicator, [&] {
        if( !local_error.empty() ) KARMA_THROW( InvalidArgument( local_error ) );
    }, "unknown ODE initialization validation failure" );
    require_matching_signature( _communicator, signature );
    auto system = std::make_unique< KARMAAlgebraicSystem >( _communicator, "ODESystem" );
    for( const auto& name : _unknowns )
        system->add_variable( { name, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
    system->init();
    PetscOwner< Vec, VecDestroy > residual;
    PetscOwner< Mat, MatDestroy > mass;
    PetscOwner< Mat, MatDestroy > matrix;
    PetscOwner< TS, TSDestroy > solver;
    *residual.put() = system->create_vector();
    *mass.put() = system->create_matrix();
    *matrix.put() = system->create_matrix();
    check( TSCreate( _communicator, solver.put() ), "TSCreate(ODE)" );
    check( TSSetProblemType( solver.get(), TS_NONLINEAR ), "TSSetProblemType(ODE)" );
    check( TSSetIFunction( solver.get(), residual.get(), ts_residual, this ), "TSSetIFunction(ODE)" );
    check( TSSetIJacobian( solver.get(), matrix.get(), matrix.get(), ts_jacobian, this ), "TSSetIJacobian(ODE)" );
    check( TSSetFromOptions( solver.get() ), "TSSetFromOptions(ODE)" );
    check( TSSetType( solver.get(), TSCN ), "TSSetType(ODE)" );
    check( TSSetExactFinalTime( solver.get(), TS_EXACTFINALTIME_MATCHSTEP ), "TSSetExactFinalTime(ODE)" );
    _system = std::move( system );
    _residual = std::move( residual );
    _mass = std::move( mass );
    _operator = std::move( matrix );
    _solver = std::move( solver );
    _initialized = true;
    try
    {
        assemble_mass( 0, _system->solution(), _mass.get() );
    }
    catch( ... )
    {
        clear();
        throw;
    }
}

void ODESystem::clear() noexcept
{
    _solver.reset();
    _operator.reset();
    _mass.reset();
    _residual.reset();
    _system.reset();
    _initialized = false;
}

void ODESystem::require_initialized() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "ODE system is not initialized" ) );
}

KARMAAlgebraicSystem& ODESystem::system() { require_initialized(); return *_system; }
const KARMAAlgebraicSystem& ODESystem::system() const { require_initialized(); return *_system; }
Vec ODESystem::solution() const { require_initialized(); return _system->solution(); }
Mat ODESystem::mass() const { require_initialized(); return _mass.get(); }
Mat ODESystem::operator_matrix() const { require_initialized(); return _operator.get(); }

void ODESystem::validate_petsc_vector( Vec value, const char* name, bool finite ) const
{
    require_initialized();
    validate_vector_object( value, _communicator, _system->n_dofs(), _system->n_local_dofs(), name, finite );
}

void ODESystem::validate_petsc_matrix( Mat value, const char* name ) const
{
    require_initialized();
    validate_matrix_object( value, _communicator, _system->n_dofs(), _system->n_local_dofs(), name );
}

void ODESystem::assemble_mass( Real time, Vec state, Mat value ) const
{
    require_initialized();
    const PetscInt size = _system->n_dofs();
    validate_scalar( _communicator, time, "ODE mass time" );
    validate_vector_object( state, _communicator, size, _system->n_local_dofs(), "ODE mass state", true );
    validate_matrix_object( value, _communicator, size, _system->n_local_dofs(), "ODE mass matrix" );
    RealDenseMatrix dense( static_cast< std::size_t >( size ), static_cast< std::size_t >( size ) );
    evaluate_collectively( _communicator, [&] {
        if( _properties.model().mass ) _properties.model().mass( time, dense );
        else dense.native().setIdentity();
        validate_matrix( dense, size, "ODE mass" );
    }, "unknown ODE mass callback failure" );
    check( MatZeroEntries( value ), "MatZeroEntries(ODE mass)" );
    PetscInt first = 0, last = 0;
    check( MatGetOwnershipRange( value, &first, &last ), "MatGetOwnershipRange(ODE mass)" );
    for( PetscInt i = first; i < last; ++i )
        for( PetscInt j = 0; j < size; ++j )
            check( MatSetValue( value, i, j, dense( i, j ), INSERT_VALUES ), "MatSetValue(ODE mass)" );
    check( MatAssemblyBegin( value, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(ODE mass)" );
    check( MatAssemblyEnd( value, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(ODE mass)" );
}

void ODESystem::residual( Real time, Vec state, Vec time_derivative, Vec value ) const
{
    require_initialized();
    const PetscInt size = _system->n_dofs();
    validate_scalar( _communicator, time, "ODE residual time" );
    validate_vector_object( state, _communicator, size, _system->n_local_dofs(), "ODE state", true );
    validate_vector_object( time_derivative, _communicator, size, _system->n_local_dofs(), "ODE time derivative", true );
    validate_vector_object( value, _communicator, size, _system->n_local_dofs(), "ODE residual output", false );
    const auto state_values = read_vector( state, size );
    RealDenseVector rhs( static_cast< std::size_t >( size ) );
    evaluate_collectively( _communicator, [&] {
        _properties.model().rhs( time, state_values, rhs );
        validate_vector( rhs, size, "ODE right-hand side" );
    }, "unknown ODE right-hand-side callback failure" );
    assemble_mass( time, state, _mass.get() );
    check( MatMult( _mass.get(), time_derivative, value ), "MatMult(ODE residual)" );
    RealDenseVector negative_rhs = rhs;
    negative_rhs.native() *= -1;
    PetscOwner< Vec, VecDestroy > forcing;
    *forcing.put() = _system->create_vector();
    write_vector( forcing.get(), negative_rhs );
    check( VecAXPY( value, 1, forcing.get() ), "VecAXPY(ODE residual)" );
}

void ODESystem::jacobian( Real time, Vec state, Real shift, Mat value ) const
{
    require_initialized();
    const PetscInt size = _system->n_dofs();
    validate_scalar( _communicator, time, "ODE Jacobian time" );
    validate_scalar( _communicator, shift, "ODE Jacobian shift" );
    validate_vector_object( state, _communicator, size, _system->n_local_dofs(), "ODE state", true );
    validate_matrix_object( value, _communicator, size, _system->n_local_dofs(), "ODE Jacobian matrix" );
    const auto state_values = read_vector( state, size );
    RealDenseMatrix rhs_jacobian( static_cast< std::size_t >( size ), static_cast< std::size_t >( size ) );
    evaluate_collectively( _communicator, [&] {
        if( _properties.model().jacobian )
        {
            _properties.model().jacobian( time, state_values, rhs_jacobian );
            validate_matrix( rhs_jacobian, size, "ODE Jacobian" );
            return;
        }
        RealDenseVector base( static_cast< std::size_t >( size ) );
        _properties.model().rhs( time, state_values, base );
        validate_vector( base, size, "ODE fallback base right-hand side" );
        RealDenseVector plus( static_cast< std::size_t >( size ) ), minus( static_cast< std::size_t >( size ) );
        for( PetscInt j = 0; j < size; ++j )
        {
            const Real step = std::cbrt( std::numeric_limits< Real >::epsilon() ) *
                              std::max( Real{ 1 }, std::abs( state_values( static_cast< std::size_t >( j ) ) ) );
            auto perturbed = state_values;
            perturbed( static_cast< std::size_t >( j ) ) += step;
            _properties.model().rhs( time, perturbed, plus );
            validate_vector( plus, size, "ODE fallback positive probe right-hand side" );
            perturbed( static_cast< std::size_t >( j ) ) -= 2 * step;
            _properties.model().rhs( time, perturbed, minus );
            validate_vector( minus, size, "ODE fallback negative probe right-hand side" );
            for( PetscInt i = 0; i < size; ++i )
            {
                const Real derivative = ( plus( static_cast< std::size_t >( i ) ) -
                                          minus( static_cast< std::size_t >( i ) ) ) / ( 2 * step );
                if( !std::isfinite( derivative ) )
                    KARMA_THROW( InvalidArgument( "ODE fallback Jacobian produced a nonfinite derivative" ) );
                rhs_jacobian( static_cast< std::size_t >( i ), static_cast< std::size_t >( j ) ) = derivative;
            }
        }
    }, "unknown ODE Jacobian callback failure" );
    assemble_mass( time, state, value );
    check( MatScale( value, shift ), "MatScale(ODE Jacobian)" );
    PetscInt first = 0, last = 0;
    check( MatGetOwnershipRange( value, &first, &last ), "MatGetOwnershipRange(ODE Jacobian)" );
    for( PetscInt i = first; i < last; ++i )
        for( PetscInt j = 0; j < size; ++j )
            check( MatSetValue( value, i, j, -rhs_jacobian( i, j ), ADD_VALUES ), "MatSetValue(ODE Jacobian)" );
    check( MatAssemblyBegin( value, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(ODE Jacobian)" );
    check( MatAssemblyEnd( value, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(ODE Jacobian)" );
}

void ODESystem::solve( Real initial_time, Real final_time, Real initial_step )
{
    require_initialized();
    validate_scalar( _communicator, initial_time, "ODE initial time" );
    validate_scalar( _communicator, final_time, "ODE final time" );
    validate_scalar( _communicator, initial_step, "ODE initial step" );
    validate_vector_object( solution(), _communicator, _system->n_dofs(), _system->n_local_dofs(),
                            "ODE solution", true );
    if( final_time <= initial_time || initial_step <= 0 ) KARMA_THROW( InvalidArgument( "invalid ODE time interval" ) );
    check( TSSetStepNumber( _solver.get(), 0 ), "TSSetStepNumber(ODE)" );
    check( TSSetTime( _solver.get(), initial_time ), "TSSetTime(ODE)" );
    check( TSSetTimeStep( _solver.get(), initial_step ), "TSSetTimeStep(ODE)" );
    check( TSSetMaxTime( _solver.get(), final_time ), "TSSetMaxTime(ODE)" );
    check( TSSetSolution( _solver.get(), solution() ), "TSSetSolution(ODE)" );
    check( TSSolve( _solver.get(), solution() ), "TSSolve(ODE)" );
    TSConvergedReason reason = TS_CONVERGED_ITERATING;
    check( TSGetConvergedReason( _solver.get(), &reason ), "TSGetConvergedReason(ODE)" );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "ODE integration did not converge" ) );
}

PetscErrorCode ODESystem::ts_residual( TS, PetscReal time, Vec state, Vec derivative, Vec value, void* context ) noexcept
{
    try
    {
        static_cast< ODESystem* >( context )->residual( time, state, derivative, value );
        return PETSC_SUCCESS;
    }
    catch( const std::exception& error )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", error.what() );
    }
    catch( ... )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", "unknown ODE residual failure" );
    }
}

PetscErrorCode ODESystem::ts_jacobian( TS, PetscReal time, Vec state, Vec, PetscReal shift, Mat matrix,
                                       Mat preconditioner, void* context ) noexcept
{
    try
    {
        auto* system = static_cast< ODESystem* >( context );
        system->jacobian( time, state, shift, matrix );
        if( preconditioner != matrix ) system->jacobian( time, state, shift, preconditioner );
        return PETSC_SUCCESS;
    }
    catch( const std::exception& error )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", error.what() );
    }
    catch( ... )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", "unknown ODE Jacobian failure" );
    }
}
}  // namespace KARMA
