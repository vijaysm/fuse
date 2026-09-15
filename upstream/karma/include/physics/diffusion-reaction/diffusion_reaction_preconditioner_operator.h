#ifndef __diffusion_reaction_preconditioner_operator_h__
#define __diffusion_reaction_preconditioner_operator_h__

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class DiffusionReactionSystem;

void point_jacobi_solve( Mat matrix, Vec rhs, Vec solution );

class DiffusionReactionPreconditionerOperator
{
  public:
    explicit DiffusionReactionPreconditionerOperator( const DiffusionReactionSystem& physics ) noexcept
        : _physics( physics )
    {}

    void assemble( Vec solution, Mat preconditioner ) const;
    void solve( Mat preconditioner, Vec rhs, Vec solution ) const;

  private:
    const DiffusionReactionSystem& _physics;
};
}  // namespace KARMA

#endif
