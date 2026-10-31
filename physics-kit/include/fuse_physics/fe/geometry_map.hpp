#pragma once

/// Isoparametric element geometry (reference-to-physical map, Jacobian,
/// inverse map), adapted from KARMA's karma_geometry_map.h/.C (docs/dev/
/// physics-kit-disposition.md Appendix A, port-adapt, owner decision D-A).
///
/// DECOUPLED FROM A MESH (docs/dev/physics-kit-disposition.md Appendix A:
/// "no fuse::backend equivalent"). KARMA's KARMAGeometryMap has a
/// mesh-bound constructor (KARMAGeometryMap(const KARMAMesh&,
/// moab::EntityHandle)) alongside its explicit-nodes constructor. This port
/// keeps only the explicit-nodes form in the public header: the mesh-bound
/// construction is a thin convenience (fetch node coordinates, then call
/// the explicit-nodes constructor) that belongs with whichever concrete
/// mesh type supplies the coordinates -- fuse_physics::fe::FEBase
/// (fe_base.hpp) does exactly that against fuse_physics::spatial::Mesh.
///
/// WAVE 1 SCOPE: the five CellFamily values reference_topology.hpp
/// implements. The quad4 affine fast path and the MBPYRAMID-specific
/// collapsed-coordinate validation sampling KARMA's own geometry_map.C
/// carries are both pyramid/prism-adjacent optimisations this wave's
/// scope does not need; the quad4 affine cache IS ported (sees real use:
/// every structured-grid quad element is affine) but the pyramid sampling
/// path is not (no pyramid variant exists to sample).

#include <array>
#include <cstddef>
#include <span>
#include <vector>

#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

enum class InverseMapStatus { Located, NotContained, InvalidGeometry };

struct InverseMapResult {
  InverseMapStatus status = InverseMapStatus::NotContained;
  Point reference_point = Point::zero();
  double residual = 0.0;
};

struct SideGeometry {
  Point point = Point::zero();
  Matrix3 tangents = Matrix3::zero();
  Vector3 normal = Vector3::zero();
  double measure = 0.0;
};

/// A reusable geometry evaluator for one element's node coordinates.
/// Const evaluation updates internal scratch state; an instance must not be
/// shared concurrently (same contract KARMA's own class states).
class GeometryMap {
 public:
  GeometryMap(ReferenceTopology reference, std::span<const Point> nodes);
  explicit GeometryMap(ReferenceTopology reference);
  void reinit(std::span<const Point> nodes);

  [[nodiscard]] GeometryVariant variant() const noexcept { return reference_.variant(); }
  [[nodiscard]] unsigned reference_dimension() const noexcept { return reference_.dimension(); }
  [[nodiscard]] std::size_t node_count() const noexcept { return node_count_; }
  [[nodiscard]] const ReferenceTopology& reference_topology() const noexcept { return reference_; }
  [[nodiscard]] std::span<const Point> nodes() const noexcept { return {nodes_.data(), node_count_}; }

  [[nodiscard]] Point map(const Point& reference_point) const;
  [[nodiscard]] Matrix3 jacobian(const Point& reference_point) const;
  [[nodiscard]] Matrix3 metric(const Point& reference_point) const;
  [[nodiscard]] double measure(const Point& reference_point) const;
  [[nodiscard]] Point inverse_map(const Point& physical_point) const;
  [[nodiscard]] InverseMapResult try_inverse_map(const Point& physical_point, double reference_tolerance,
                                                  double residual_tolerance) const noexcept;
  [[nodiscard]] SideGeometry side(unsigned side, const Point& side_point) const;
  [[nodiscard]] SideGeometry side_from_parent(unsigned side, const Point& parent_point,
                                               const Matrix3& parent_side_jacobian) const;

 private:
  void update_affine_quad4_cache();
  void validate() const;

  ReferenceTopology reference_;
  std::array<Point, 27> nodes_{};
  std::size_t node_count_ = 0;
  bool has_affine_quad4_cache_ = false;
  Point affine_quad4_origin_ = Point::zero();
  Vector3 affine_quad4_r_ = Vector3::zero();
  Vector3 affine_quad4_s_ = Vector3::zero();
  Matrix3 affine_quad4_jacobian_ = Matrix3::zero();
  Matrix3 affine_quad4_metric_ = Matrix3::zero();
  double affine_quad4_measure_ = 0.0;
  mutable std::array<double, 27> values_{};
  mutable std::array<Point, 27> derivatives_{};
};

}  // namespace fuse::physics::fe
