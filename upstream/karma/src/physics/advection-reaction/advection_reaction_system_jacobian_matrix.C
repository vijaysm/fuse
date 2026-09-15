#include "advection_reaction_assembly.h"

#include <algorithm>
#include <cmath>

namespace KARMA
{
Real upwind_flux( Real interior, Real exterior, const Vector3& velocity, const Vector3& outward_normal )
{
    const Real normal_velocity = velocity.dot( outward_normal );
    return normal_velocity * ( normal_velocity >= 0 ? interior : exterior );
}

Real rusanov_flux( Real interior, Real exterior, const Vector3& interior_velocity,
                   const Vector3& exterior_velocity, const Vector3& outward_normal )
{
    const Real interior_speed = interior_velocity.dot( outward_normal );
    const Real exterior_speed = exterior_velocity.dot( outward_normal );
    const Real alpha          = std::max( std::abs( interior_speed ), std::abs( exterior_speed ) );
    return 0.5 * ( interior_speed * interior + exterior_speed * exterior ) -
           0.5 * alpha * ( exterior - interior );
}
}  // namespace KARMA
