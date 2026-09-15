#ifndef __advection_reaction_system_operator_h__
#define __advection_reaction_system_operator_h__

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class AdvectionReactionSystem;
class AdvectionReactionSystemOperator
{
  public:
    explicit AdvectionReactionSystemOperator( const AdvectionReactionSystem& physics ) noexcept : _physics( physics ) {}
    void residual( Vec solution, Vec residual ) const;
    void jacobian( Vec solution, Mat jacobian ) const;
  private:
    const AdvectionReactionSystem& _physics;
};
}  // namespace KARMA
#endif
