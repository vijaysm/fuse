#include "advection_reaction_mass_operator.h"

#include "generic_mass_operator.h"
#include "karma_system.h"

namespace KARMA
{
void AdvectionReactionMassOperator::assemble( Mat mass ) const
{
    _system.validate_matrix_layout( mass, "advection mass matrix" );
    assemble_mass_matrix( _system, mass );
}
void AdvectionReactionMassOperator::multiply( Mat mass, Vec input, Vec output ) const
{
    _system.validate_matrix_layout( mass, "advection mass multiply matrix" );
    _system.validate_vector_layout( input, "advection mass multiply input" );
    _system.validate_vector_layout( output, "advection mass multiply output" );
    mass_multiply( _system, mass, input, output );
}
void AdvectionReactionMassOperator::solve( Mat mass, Vec rhs, Vec solution, bool lumped ) const
{
    _system.validate_matrix_layout( mass, "advection mass solve matrix" );
    _system.validate_vector_layout( rhs, "advection mass solve right-hand side" );
    _system.validate_vector_layout( solution, "advection mass solve output" );
    mass_solve( _system, mass, rhs, solution, lumped );
}
}  // namespace KARMA
