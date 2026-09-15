#include "advection_reaction_preconditioner_operator.h"

#include "advection_reaction_system.h"
#include "diffusion_reaction_preconditioner_operator.h"

namespace KARMA
{
void AdvectionReactionPreconditionerOperator::assemble( Vec solution, Mat preconditioner ) const
{ _physics.assemble_jacobian( solution, preconditioner ); }
void AdvectionReactionPreconditionerOperator::solve( Mat preconditioner, Vec rhs, Vec solution ) const
{ point_jacobi_solve( preconditioner, rhs, solution ); }
}  // namespace KARMA
