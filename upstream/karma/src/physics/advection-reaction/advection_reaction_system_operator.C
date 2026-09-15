#include "advection_reaction_system_operator.h"
#include "advection_reaction_system.h"

namespace KARMA
{
void AdvectionReactionSystemOperator::residual( Vec solution, Vec residual ) const
{ _physics.assemble_residual( solution, residual ); }
void AdvectionReactionSystemOperator::jacobian( Vec solution, Mat jacobian ) const
{ _physics.assemble_jacobian( solution, jacobian ); }
}  // namespace KARMA
