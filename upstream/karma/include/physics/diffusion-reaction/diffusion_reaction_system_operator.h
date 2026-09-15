#ifndef __diffusion_reaction_system_operator_h__
#define __diffusion_reaction_system_operator_h__

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class DiffusionReactionSystem;

class DiffusionReactionSystemOperator
{
  public:
    explicit DiffusionReactionSystemOperator( const DiffusionReactionSystem& physics ) noexcept : _physics( physics ) {}

    void residual( Vec solution, Vec residual ) const;
    void jacobian( Vec solution, Mat jacobian ) const;
    void assemble( Vec solution, Vec residual, Mat jacobian ) const;

  private:
    const DiffusionReactionSystem& _physics;
};
}  // namespace KARMA

#endif
