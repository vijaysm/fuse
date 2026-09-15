#include "ode_system_property_manager.h"

#include <utility>

namespace KARMA
{
ODESystemPropertyManager::ODESystemPropertyManager( ODEModel model ) : _model( std::move( model ) ) {}
}  // namespace KARMA
