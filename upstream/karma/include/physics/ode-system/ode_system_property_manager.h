#ifndef __ode_system_property_manager_h__
#define __ode_system_property_manager_h__

#include "karma_dense.h"

#include <functional>

namespace KARMA
{
using ODERightHandSide = std::function< void( Real time, const RealDenseVector& state, RealDenseVector& value ) >;
using ODEJacobian = std::function< void( Real time, const RealDenseVector& state, RealDenseMatrix& value ) >;
using ODEMass = std::function< void( Real time, RealDenseMatrix& value ) >;

struct ODEModel
{
    ODERightHandSide rhs;
    ODEJacobian jacobian;
    ODEMass mass;
};

class ODESystemPropertyManager
{
  public:
    explicit ODESystemPropertyManager( ODEModel model = {} );
    [[nodiscard]] const ODEModel& model() const noexcept { return _model; }
    [[nodiscard]] bool is_ready() const noexcept { return static_cast< bool >( _model.rhs ); }

  private:
    ODEModel _model;
};
}  // namespace KARMA

#endif
