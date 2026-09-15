#ifndef __ode_system_operator_h__
#define __ode_system_operator_h__

#include "karma_geometry.h"

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class ODESystem;

class ODESystemOperator
{
  public:
    explicit ODESystemOperator( const ODESystem& system ) noexcept : _system( system ) {}
    void residual( Real time, Vec state, Vec time_derivative, Vec value ) const;
    void jacobian( Real time, Vec state, Real shift, Mat value ) const;

  private:
    const ODESystem& _system;
};
}  // namespace KARMA

#endif
