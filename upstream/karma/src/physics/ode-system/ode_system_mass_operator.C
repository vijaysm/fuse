#include "ode_mass_operator.h"

#include "karma_exceptions.h"
#include "ode_system.h"
#include "petsc_raii.h"

#include <petscksp.h>

#include <string>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}
}  // namespace

namespace KARMA
{
void ODEMassOperator::assemble( Real time, Vec state, Mat mass ) const
{
    _system.assemble_mass( time, state, mass );
}

void ODEMassOperator::multiply( Mat mass, Vec input, Vec output ) const
{
    _system.validate_petsc_matrix( mass, "ODE mass multiply matrix" );
    _system.validate_petsc_vector( input, "ODE mass multiply input" );
    _system.validate_petsc_vector( output, "ODE mass multiply output" );
    check( MatMult( mass, input, output ), "MatMult(ODE mass)" );
}

void ODEMassOperator::solve( Mat mass, Vec rhs, Vec solution ) const
{
    _system.validate_petsc_matrix( mass, "ODE mass solve matrix" );
    _system.validate_petsc_vector( rhs, "ODE mass solve right-hand side" );
    _system.validate_petsc_vector( solution, "ODE mass solve output" );
    PetscOwner< KSP, KSPDestroy > solver;
    check( KSPCreate( _system.system().communicator(), solver.put() ), "KSPCreate(ODE mass)" );
    check( KSPSetOperators( solver.get(), mass, mass ), "KSPSetOperators(ODE mass)" );
    check( KSPSetFromOptions( solver.get() ), "KSPSetFromOptions(ODE mass)" );
    check( KSPSolve( solver.get(), rhs, solution ), "KSPSolve(ODE mass)" );
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    check( KSPGetConvergedReason( solver.get(), &reason ), "KSPGetConvergedReason(ODE mass)" );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "ODE mass solve did not converge" ) );
}
}  // namespace KARMA
