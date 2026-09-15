#ifndef __ode_mass_operator_h__
#define __ode_mass_operator_h__

#include "karma_geometry.h"

#include <petscmat.h>
#include <petscvec.h>

namespace KARMA
{
class ODESystem;

class ODEMassOperator
{
  public:
    explicit ODEMassOperator( const ODESystem& system ) noexcept : _system( system ) {}
    void assemble( Real time, Vec state, Mat mass ) const;
    void multiply( Mat mass, Vec input, Vec output ) const;
    void solve( Mat mass, Vec rhs, Vec solution ) const;

  private:
    const ODESystem& _system;
};
}  // namespace KARMA

#endif
