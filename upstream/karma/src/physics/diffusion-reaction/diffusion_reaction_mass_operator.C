#include "diffusion_reaction_mass_operator.h"

#include "generic_mass_operator.h"
#include "karma_system.h"

namespace KARMA
{
void DiffusionReactionMassOperator::assemble( Mat mass ) const
{
    assemble_mass_matrix( _system, mass );
}

void DiffusionReactionMassOperator::multiply( Mat mass, Vec input, Vec output ) const
{
    mass_multiply( _system, mass, input, output );
}

void DiffusionReactionMassOperator::solve( Mat mass, Vec rhs, Vec solution, bool lumped ) const
{
    mass_solve( _system, mass, rhs, solution, lumped );
}
}  // namespace KARMA
