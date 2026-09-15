#ifndef __advection_reaction_system_property_manager_h__
#define __advection_reaction_system_property_manager_h__

#include "advection_reaction_assembly.h"

#include <map>
#include <vector>

namespace KARMA
{
class AdvectionReactionSystemPropertyManager
{
  public:
    void register_model( int material_id, AdvectionReactionCoefficients model );
    [[nodiscard]] bool has_model( int material_id ) const noexcept;
    [[nodiscard]] const AdvectionReactionCoefficients& model( int material_id ) const;
    [[nodiscard]] AdvectionReactionCoefficients coefficients() const;
    [[nodiscard]] std::vector< int > material_ids() const;
    [[nodiscard]] bool empty() const noexcept { return _models.empty(); }

  private:
    std::map< int, AdvectionReactionCoefficients > _models;
};
}  // namespace KARMA

#endif
