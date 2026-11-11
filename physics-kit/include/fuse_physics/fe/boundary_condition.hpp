#pragma once

/// Boundary conditions on a box-shaped fuse_physics::spatial::Domain:
/// Dirichlet (value of the unknown), Neumann (prescribed outward flux)
/// and Robin (k du/dn + robin_coefficient * u = value, the normalised
/// form of a typed a*(k grad(u).n)+b*u=c with a=1 fixed -- dividing the
/// general form through by a is always valid when a is not zero, and
/// a=0 degenerates Robin into a condition on u alone, which is
/// Dirichlet under a different name; fixing a=1 removes a redundant
/// degree of freedom rather than hiding a case).
///
/// Adapted from KARMA's include/physics/common/karma_boundary_condition.h
/// and karma_physics_bc_manager.h. NOT a literal port: KARMA's own
/// KARMABoundaryCondition stores a vector of PETSc Real values indexed
/// by position and KARMAPhysicsBCManager keys conditions by (element,
/// side) or node id discovered through a MOAB mesh's side sets. This
/// port's boundary identity is geometric instead: fuse_physics::
/// spatial::Mesh's self-generated box meshes carry no side-set tags at
/// all, so a boundary names one of the SIX faces of the box domain
/// (BoundaryId below) and a vertex, side, or any other dof is
/// classified onto it by comparing its own physical position against
/// the mesh's own measured bounding box.
///
/// BOUNDING BOX MEASURED, NOT ASSUMED. The box's own lower corner is
/// read from the mesh's own vertices (the componentwise minimum over
/// `mesh.vertex_positions()`), not assumed at the origin: a box
/// generated from a different corner (`generate_box`/`generate_2d_box`
/// both take an explicit `lo` point) still classifies correctly. The
/// upper corner is the lower corner plus `domain`'s own length_x/y/z.
///
/// SINGLE RANK ONLY. Resolution walks every vertex, every side, and
/// every dof of the WHOLE mesh directly, with no distinction between
/// an owned and a ghost entity, and a BoundaryConditionManager is
/// therefore built only from a DofMap with no communicator -- the same
/// restriction AlgebraicSystem::eliminate_dirichlet_rows already
/// carries today. A caller that resolves against a multi-rank DofMap is
/// refused by name (E-PHYSICS-BC-MPI-NOT-SUPPORTED) rather than
/// silently resolving an incomplete or duplicated boundary set across
/// ranks.

#include <cstdint>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/dof_map.hpp"
#include "fuse_physics/geometry.hpp"
#include "fuse_physics/spatial/domain.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::fe {

/// One of the six faces of a box domain. A 2-D domain (dimension() == 2)
/// has no ZMin/ZMax face; registering a condition there is refused
/// (E-PHYSICS-BC-UNKNOWN-BOUNDARY) rather than silently accepted and
/// never matched by any vertex or side.
enum class BoundaryId { XMin, XMax, YMin, YMax, ZMin, ZMax };

enum class BoundaryConditionKind { Dirichlet, Neumann, Robin };

/// g(p, t): the boundary data as a function of physical position and
/// time -- Dirichlet: u = g. Neumann: k du/dn = g (g is the prescribed
/// outward flux, with the convention that it contributes +integral(phi_i
/// g) to the load). Robin: k du/dn + robin_coefficient * u = g.
using BoundaryValue = std::function<double(const Point&, double)>;

struct BoundaryCondition {
  BoundaryConditionKind kind = BoundaryConditionKind::Dirichlet;
  BoundaryValue value;
  /// Meaningful only for Robin. Refused (E-PHYSICS-BC-ROBIN-COEFFICIENT)
  /// unless strictly positive: with k > 0, the Robin bilinear form's own
  /// boundary contribution is robin_coefficient * integral(phi_i phi_j)
  /// over the face, a POSITIVE SEMI-DEFINITE addition to the element
  /// stiffness exactly when robin_coefficient >= 0; requiring it
  /// STRICTLY positive is what lets a Robin condition alone (with no
  /// Dirichlet boundary anywhere else on the domain) make the discrete
  /// operator coercive and the steady problem well-posed -- a Robin
  /// condition with robin_coefficient == 0 is a Neumann condition under
  /// a different name and does not supply that coercivity.
  double robin_coefficient = 0.0;
  /// d(value)/dt, for a Dirichlet condition in a transient problem:
  /// DiffusionReactionSystem::rhs reads it at a Dirichlet dof, so the
  /// semi-discrete equation there is `du/dt = d(value)/dt`, which
  /// integrates to `u(t) = value(t)` to the integrator's own accuracy
  /// given `u(0) = value(0)`. A transient solve refuses a Dirichlet
  /// condition that leaves this unset; `steady_dirichlet` supplies the
  /// zero rate of boundary data that does not depend on time. A steady
  /// solve does not read it.
  BoundaryValue value_time_derivative;

  /// A Dirichlet condition whose data does not depend on time: `value` is
  /// evaluated as given and its rate is identically zero.
  [[nodiscard]] static BoundaryCondition steady_dirichlet(BoundaryValue value) {
    BoundaryCondition condition;
    condition.kind = BoundaryConditionKind::Dirichlet;
    condition.value = std::move(value);
    condition.value_time_derivative = [](const Point&, double) { return 0.0; };
    return condition;
  }
};

/// One Dirichlet dof, resolved against a DofMap: its global index, the
/// position its value is evaluated at (the dof's own interpolation
/// point, mapped through its element's own geometry map -- the nodal
/// convention a Lagrange basis's own dof is naturally evaluated at, for
/// any basis order), and which BoundaryCondition supplied the value.
struct ResolvedDirichlet {
  std::int64_t dof = 0;
  Point position;
  const BoundaryCondition* condition = nullptr;
};

/// One Neumann or Robin face, resolved against a Mesh: the element and
/// side FEBase's own reinit(element, side) (fe_base.hpp) integrates,
/// and which BoundaryCondition supplies the face data.
struct ResolvedFace {
  spatial::EntityHandle element = 0;
  unsigned side = 0;
  const BoundaryCondition* condition = nullptr;
};

/// Classifies a single position against a box with lower corner `lo`
/// and upper corner `lo + (length_x, length_y, length_z)`, considering
/// only the first `dimension` axes. Returns every BoundaryId `position`
/// lies on (a corner or edge of the box lies on more than one face at
/// once); empty if `position` is not on any face of the box within
/// `tolerance`.
///
/// TOLERANCE, derived not chosen: `tolerance` is meant to be
/// `box_length_scale * 4 * std::numeric_limits<double>::epsilon()`, the
/// same "a few ULPs over the operation's own length scale" derivation
/// physics-kit's other geometric comparisons use -- computed ONCE by
/// BoundaryConditionManager's own constructor from the mesh's own
/// measured extent, not re-derived at every call, and passed in here so
/// this free function stays a pure, directly testable geometric
/// predicate.
[[nodiscard]] std::vector<BoundaryId> classify_position(const Point& position, const Point& lo, double length_x,
                                                         double length_y, double length_z, unsigned dimension,
                                                         double tolerance);

/// The physical position of `dof_indices(element, ..., variable)`'s own
/// k-th entry, for every owned element, written into `out[dof]` --
/// `basis`'s own interpolation point for that dof (`Basis::
/// interpolation_points()`, in the SAME order `storage()`'s own records
/// and `dof_indices()`'s own global indices already agree on), mapped
/// through THIS element's own geometry map (its physical vertex
/// positions). Correct for any dof, not only a vertex one: an edge or
/// face dof's own interpolation point maps to that edge or face's own
/// physical location exactly as a vertex dof's own interpolation point
/// maps to the vertex itself. `out` is sized `dof_map.n_dofs()` and a
/// dof touched by more than one element is written more than once, with
/// the same value each time for a conforming (continuous) basis.
void dof_physical_positions(const spatial::Mesh& mesh, const DofMap& dof_map, unsigned variable, const Basis& basis,
                             std::vector<Point>& out);

/// Resolves and holds the boundary conditions for ONE scalar variable
/// of a DofMap built over `mesh`/`domain`. `set()` registers one
/// condition per BoundaryId (refusing E-PHYSICS-BC-DUPLICATE on a
/// second registration for the same id, and E-PHYSICS-BC-UNKNOWN-BOUNDARY
/// for a ZMin/ZMax id on a domain with dimension() < 3); `resolve()`
/// walks the mesh once and classifies every dof and every boundary side
/// against the registered conditions, refusing E-PHYSICS-BC-UNMATCHED
/// (naming the boundary) if a registered condition matched neither a
/// dof nor a face -- a condition on a face the mesh does not have (a
/// mesh shifted or sized so that face never geometrically occurs) is a
/// caller error caught here rather than a silently inert registration.
///
/// A box corner or edge lying on two registered Dirichlet boundaries at
/// once (e.g. the domain's own lower corner, on both XMin and YMin)
/// takes the FIRST matching id in BoundaryId's own declaration order
/// (XMin, XMax, YMin, YMax, ZMin, ZMax) -- a documented convention, not
/// a disagreement check: every test built against this manager uses a
/// single manufactured solution shared by every boundary condition, so
/// the two candidate values are IDENTICAL at such a corner by
/// construction, and picking either changes nothing a test could
/// observe.
class BoundaryConditionManager {
 public:
  BoundaryConditionManager(const spatial::Mesh& mesh, const spatial::Domain& domain);

  void set(BoundaryId boundary_id, BoundaryCondition condition);

  /// `dof_map` must have no communicator (n_ranks() == 1, this class's
  /// own header comment). `basis` is the SAME basis `dof_map` was built
  /// with for `variable` (0 for a single-variable DofMap) -- needed to
  /// resolve a non-vertex (edge/face) dof's own side membership and
  /// physical position for a higher-order basis. Refuses
  /// E-PHYSICS-BC-MPI-NOT-SUPPORTED, E-PHYSICS-BC-NO-CONDITIONS (this
  /// class's own header comment), and E-PHYSICS-BC-UNMATCHED.
  void resolve(const DofMap& dof_map, const Basis& basis, unsigned variable = 0);

  [[nodiscard]] const std::vector<ResolvedDirichlet>& dirichlet_dofs() const noexcept { return dirichlet_; }
  [[nodiscard]] const std::vector<ResolvedFace>& neumann_faces() const noexcept { return neumann_; }
  [[nodiscard]] const std::vector<ResolvedFace>& robin_faces() const noexcept { return robin_; }

 private:
  const spatial::Mesh* mesh_;
  const spatial::Domain* domain_;
  Point lo_;  ///< the box's own measured lower corner (mesh's own vertex bounding box)
  double tolerance_;
  std::map<BoundaryId, BoundaryCondition> conditions_;
  std::vector<ResolvedDirichlet> dirichlet_;
  std::vector<ResolvedFace> neumann_;
  std::vector<ResolvedFace> robin_;
};

}  // namespace fuse::physics::fe
