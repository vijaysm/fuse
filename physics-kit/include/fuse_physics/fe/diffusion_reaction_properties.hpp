#pragma once

/// Material properties for fuse_physics::fe::DiffusionReactionSystem.
/// Adapted from KARMA's include/physics/diffusion-reaction/
/// diffusion_reaction_assembly.h's own DiffusionReactionCoefficients and
/// diffusion_reaction_system_property_manager.h's own
/// DiffusionReactionSystemPropertyManager.
///
/// REDUCED, not a literal port. KARMA's own property manager keys
/// DiffusionReactionCoefficients by an integer material_id taken from a
/// MOAB mesh's own block sets; its own DiffusionReactionCoefficients
/// additionally carries base-class machinery through
/// karma_physics_base_property_manager.h's generic function-pointer
/// registry, a named-property lookup table shared by every KARMA physics
/// module, most of whose entries no diffusion-reaction model ever uses.
///
/// fuse_physics::spatial::Mesh carries no material/block id at all (its
/// own header has no material_id, block_id or region_id accessor --
/// generate_box/generate_2d_box produce a single homogeneous box with no
/// sub-region tagging whatsoever), so a material_id-keyed map has
/// nothing on the mesh side to key against: DiffusionReactionSystem
/// owns exactly ONE DiffusionReactionCoefficients for the whole domain
/// below -- still a function of PHYSICAL POSITION (and the field value,
/// for the reaction term), which is everything a manufactured-solution
/// test on one box needs, since a coefficient that must vary by region
/// can already vary by evaluating `position` against that region's own
/// geometric test inside the one callable.
///
/// The load term is a SEPARATE concern, `DiffusionReactionSystem::
/// set_forcing` (a (position, time) callable, diffusion_reaction_
/// system.hpp's own header comment): a forcing/source term is a
/// property of the PARTICULAR problem being solved (and, for a
/// manufactured transient solution, generally time-dependent), while
/// `diffusion` and `reaction` below are properties of the MATERIAL
/// (time-independent, evaluated at the current field value). Keeping
/// the two apart means every member of this struct is read by
/// DiffusionReactionSystem's own assembly; neither holds data nothing
/// consumes.
///
/// The two derivative members are kept (not dropped): the steady and
/// transient DiffusionReactionSystem residuals are nonlinear in the
/// field whenever `reaction` or `diffusion` depend on it, and the
/// steady solve's own Jacobi/SSOR preconditioner is built from an
/// ANALYTIC Jacobian -- K + M diag(reaction_derivative(u)), the reaction
/// being ADDED to the residual (below), the same
/// pattern generic_operators.hpp's own TransientReactionDiffusionModel::
/// reaction_derivative plays for its own (fixed quadratic) reaction law.
/// A coefficient whose derivative is left unset (std::function's own
/// empty state) means that term's own contribution is CONSTANT in the
/// field (derivative identically zero) -- true for `diffusion`/
/// `reaction`'s own defaults (1, 0, both field-independent).

#include <functional>

#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

/// g(position, field_value): a material property as a function of
/// physical position and the current field value (constant in the
/// field for a linear coefficient).
using ScalarCoefficient = std::function<double(const Point& position, double field_value)>;

struct DiffusionReactionCoefficients {
  /// k(x, u) > 0 in `-div(k(x,u) grad(u))`. Default: homogeneous,
  /// field-independent, k = 1.
  ScalarCoefficient diffusion = [](const Point&, double) { return 1.0; };
  /// R(x, u) in the reaction term `+ R(x,u)` added to the residual.
  /// Default: zero.
  ScalarCoefficient reaction = [](const Point&, double) { return 0.0; };

  /// d(diffusion)/d(field_value). Empty (unset) means 0 identically.
  ScalarCoefficient diffusion_derivative;
  /// d(reaction)/d(field_value). Empty (unset) means 0 identically.
  ScalarCoefficient reaction_derivative;
};

}  // namespace fuse::physics::fe
