#include "diffusion_reaction_system_operator.h"

#include "diffusion_reaction_system.h"

namespace KARMA
{
void DiffusionReactionSystemOperator::residual( Vec solution, Vec residual ) const
{
    _physics.assemble_residual( solution, residual );
}

void DiffusionReactionSystemOperator::jacobian( Vec solution, Mat jacobian ) const
{
    _physics.assemble_jacobian( solution, jacobian );
}

void DiffusionReactionSystemOperator::assemble( Vec solution, Vec residual, Mat jacobian ) const
{
    _physics.assemble_residual( solution, residual );
    _physics.assemble_jacobian( solution, jacobian );
}
}  // namespace KARMA
