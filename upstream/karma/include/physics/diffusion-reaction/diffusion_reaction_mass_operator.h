#ifndef __diffusion_reaction_mass_operator_h__
#define __diffusion_reaction_mass_operator_h__

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class KARMASystem;

class DiffusionReactionMassOperator
{
  public:
    explicit DiffusionReactionMassOperator( const KARMASystem& system ) noexcept : _system( system ) {}

    void assemble( Mat mass ) const;
    void multiply( Mat mass, Vec input, Vec output ) const;
    void solve( Mat mass, Vec rhs, Vec solution, bool lumped = false ) const;

  private:
    const KARMASystem& _system;
};
}  // namespace KARMA

#endif
