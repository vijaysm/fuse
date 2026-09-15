#include "ode_system_operator.h"

#include "ode_system.h"

namespace KARMA
{
void ODESystemOperator::residual( Real time, Vec state, Vec time_derivative, Vec value ) const
{
    _system.residual( time, state, time_derivative, value );
}

void ODESystemOperator::jacobian( Real time, Vec state, Real shift, Mat value ) const
{
    _system.jacobian( time, state, shift, value );
}
}  // namespace KARMA
