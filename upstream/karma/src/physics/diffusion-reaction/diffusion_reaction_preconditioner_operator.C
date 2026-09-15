#include "diffusion_reaction_preconditioner_operator.h"

#include "diffusion_reaction_system.h"

namespace KARMA
{
void DiffusionReactionPreconditionerOperator::assemble( Vec solution, Mat preconditioner ) const
{
    _physics.assemble_jacobian( solution, preconditioner );
}

void DiffusionReactionPreconditionerOperator::solve( Mat preconditioner, Vec rhs, Vec solution ) const
{
    point_jacobi_solve( preconditioner, rhs, solution );
}
}  // namespace KARMA
