#pragma once

/// Mesh-bound finite-element evaluator: basis values, physical gradients
/// and quadrature weights at an element's quadrature points, adapted from
/// KARMA's karma_fe_base.h/.C (port-adapted; see
/// docs/dev/physics-kit.md).
///
/// DECOUPLED FROM PETSc's PetscQuadrature (quadrature.hpp's own note) and
/// bound to fuse_physics::spatial::Mesh rather than KARMA's MOAB-backed
/// KARMAMesh (spatial/mesh.hpp's own note on why that mesh is native). The public surface -- reinit(element), reinit(element, side),
/// reinit_at_physical_points, and the getters -- mirrors KARMAFEBase's
/// contract exactly: output storage is allocated at construction and stays
/// address-stable across successful and failed reinitialisation, so a
/// caller that holds references to get_phi()/get_dphi() across a failed
/// reinit still sees valid (if now NaN-filled) storage rather than a
/// dangling one.
///
/// SCOPE: FEIntegrationDomain::Volume and Face are both ported
/// and tested, including Prism and Pyramid, whose sides have MIXED
/// shapes (Prism: 3 quadrilateral + 2 triangular; Pyramid: 4 triangular
/// + 1 quadrilateral) -- a single `quadrature_` member is not enough
/// once one parent cell's own sides are not all the same shape.
/// `quadrature_` holds the Quadrilateral-side rule (every supported
/// family has at most one side SHAPE that is not Quadrilateral: Tet's
/// sides are uniformly Triangle); `triangle_quadrature_` holds the
/// matching Triangle-side rule for Prism and Pyramid's triangular
/// sides, selected per side by fe_base.cpp's own quadrature_for_side().

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "fuse_physics/fe/basis.hpp"
#include "fuse_physics/fe/geometry_map.hpp"
#include "fuse_physics/fe/quadrature.hpp"
#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"
#include "fuse_physics/spatial/mesh.hpp"

// fe/ including spatial/ (rather than only forward-declaring KARMAMesh, as
// KARMA's own karma_fe_base.h does) is a deliberate simplification: both
// live inside physics-kit, so this is not the FUSE-vs-physics-kit boundary
// tools/check-layering.py's check_physics_kit_boundary enforces (that gate
// walks include/fuse/, src/, python/, examples/ -- never physics-kit/
// itself), and physics-kit/CMakeLists.txt already builds fe/ and spatial/
// into the SAME target, so there is no separate-library ordering to
// preserve either. A future
// FUSE_HAVE_MOAB-bound mesh type would still satisfy this same
// #include, since spatial::Mesh is physics-kit's own type either way.

namespace fuse::physics::fe {

enum class FEIntegrationDomain { Volume, Face };

/// Geometry-independent scalar field evaluation on a fixed reference cell
/// and geometry variant, bound to one mesh. Output storage is allocated at
/// construction and remains address-stable across successful and failed
/// reinitialisation (same contract KARMA's own class states).
class FEBase {
 public:
  FEBase(const spatial::Mesh& mesh, GeometryVariant geometry_variant, Basis basis, int quadrature_degree,
         FEIntegrationDomain domain = FEIntegrationDomain::Volume);

  void reinit(spatial::EntityHandle element);
  void reinit(spatial::EntityHandle element, unsigned side);
  void reinit_at_physical_points(spatial::EntityHandle element, std::span<const Point> points);

  [[nodiscard]] bool is_valid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t n_quadrature_points() const noexcept { return active_qpoints_; }

  [[nodiscard]] const std::vector<double>& get_JxW() const noexcept { return jxw_; }
  [[nodiscard]] const std::vector<std::vector<double>>& get_phi() const noexcept { return phi_; }
  [[nodiscard]] const std::vector<std::vector<Vector3>>& get_dphi() const noexcept { return dphi_; }
  [[nodiscard]] const std::vector<Point>& get_xyz() const noexcept { return xyz_; }
  [[nodiscard]] const std::vector<Vector3>& get_normals() const noexcept { return normals_; }

  [[nodiscard]] GeometryVariant geometry_variant() const noexcept { return geometry_variant_; }
  [[nodiscard]] const Basis& basis() const noexcept { return basis_; }

 private:
  void invalidate() noexcept;
  void set_active_size(std::size_t count);
  void evaluate_reference_points(const GeometryMap& geometry, std::span<const Point> points,
                                  std::span<const double> weights, bool compute_weights);
  /// The quadrature rule to use on `side`: quadrature_ for a quadrilateral
  /// side, triangle_quadrature_ (always non-null when any side is
  /// triangular -- constructor's own invariant) for a triangular one.
  [[nodiscard]] const Quadrature& quadrature_for_side(unsigned side) const;

  const spatial::Mesh& mesh_;
  GeometryVariant geometry_variant_;
  ReferenceTopology geometry_reference_;
  Basis basis_;
  GeometryMap geometry_;
  Quadrature quadrature_;
  std::unique_ptr<Quadrature> triangle_quadrature_;  // non-null only for a Face FEBase on a Prism or Pyramid
  FEIntegrationDomain domain_;
  std::size_t n_basis_;
  std::size_t capacity_;
  std::size_t active_qpoints_;
  bool valid_ = false;

  std::vector<double> jxw_;
  std::vector<std::vector<double>> phi_;
  std::vector<std::vector<Vector3>> dphi_;
  std::vector<Point> xyz_;
  std::vector<Vector3> normals_;

  std::vector<Point> geometry_nodes_scratch_;
  std::vector<double> reference_phi_;
  std::vector<Vector3> reference_gradient_;
  std::vector<Point> side_parent_points_;
  std::vector<Matrix3> side_parent_jacobians_;
  std::vector<Point> physical_point_scratch_;
  std::vector<Point> reference_point_scratch_;
};

}  // namespace fuse::physics::fe
