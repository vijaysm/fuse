#ifndef __diffusion_reaction_system_property_manager_h__
#define __diffusion_reaction_system_property_manager_h__

#include "diffusion_reaction_assembly.h"

#include <map>
#include <vector>

namespace KARMA
{
class DiffusionReactionSystemPropertyManager
{
  public:
    void register_model( int material_id, DiffusionReactionCoefficients model );
    [[nodiscard]] bool has_model( int material_id ) const noexcept;
    [[nodiscard]] const DiffusionReactionCoefficients& model( int material_id ) const;
    [[nodiscard]] DiffusionReactionCoefficients coefficients() const;
    [[nodiscard]] std::vector< int > material_ids() const;
    [[nodiscard]] bool empty() const noexcept { return _models.empty(); }

  private:
    std::map< int, DiffusionReactionCoefficients > _models;
};
}  // namespace KARMA

#endif
