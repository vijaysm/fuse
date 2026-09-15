#ifndef __advection_reaction_preconditioner_operator_h__
#define __advection_reaction_preconditioner_operator_h__

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class AdvectionReactionSystem;
class AdvectionReactionPreconditionerOperator
{
  public:
    explicit AdvectionReactionPreconditionerOperator( const AdvectionReactionSystem& physics ) noexcept
        : _physics( physics ) {}
    void assemble( Vec solution, Mat preconditioner ) const;
    void solve( Mat preconditioner, Vec rhs, Vec solution ) const;
  private:
    const AdvectionReactionSystem& _physics;
};
}  // namespace KARMA
#endif
