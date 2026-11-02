#pragma once

/// A box-shaped computational domain descriptor, adapted from KARMA's
/// karma_domain.h/.C and enums/domain_shape.h (docs/dev/physics-kit-
/// disposition.md Appendix A, port-adapt, owner decision D-A).
///
/// I6 FIX (docs/dev/physics-kit-disposition.md Section 1.1 / Appendix A row
/// for src/numerics/spatial/karma_domain.C): the upstream constructor
/// asserts `Type != Hexagon || Type != Unstructured`, three times
/// (karma_domain.C lines 32, 54, 77). That is vacuous: for ANY value of
/// Type, at most one of the two equalities can hold, so the OR is true
/// unconditionally and the assert never fires, in Debug or Release. The
/// surrounding comment ("currently only cubic shapes are supported...this
/// restriction will be lifted off") states the INTENDED invariant plainly:
/// Hexagon and Unstructured are not yet supported shapes and construction
/// should refuse them, i.e. the condition needed AND, not OR. Ported here
/// as a real refusal (fuse's exception-based error convention, matching
/// this port's own throw-on-invalid-argument style elsewhere in fe/): a
/// Domain constructed with Shape::Hexagon or Shape::Unstructured throws,
/// verified by Spatial.DomainRefusesUnsupportedShapes
/// (physics-kit/test/spatial_domain_test.cpp) actually triggering it --
/// the task's own rule that "a refusal counts as verified only when a test
/// makes it fire".

#include <stdexcept>

namespace fuse::physics::spatial {

/// The shape of a computational domain. Renamed from KARMA's DomainShape
/// (enums/domain_shape.h) only by namespace; the enumerators are unchanged
/// since they are already descriptive, literature-standard shape names.
enum class Shape { Edge, Square, Cube, Hexagon, Unstructured };

/// A box-shaped domain: dimension, shape, extents and per-axis symmetry
/// flags. Structured-grid generation (spatial::Mesh::generate_box) reads
/// extents directly; Shape::Hexagon and Shape::Unstructured are accepted by
/// the enum (a later wave's unstructured-mesh path needs to name them) but
/// refused by this constructor until that support exists -- see this
/// header's own comment above for the I6 fix this corrects.
class Domain {
 public:
  Domain(unsigned dimension, Shape shape, double length_x, double length_y, double length_z, bool symmetric);
  Domain(unsigned dimension, Shape shape, double length_x, double length_y, double length_z, bool symmetric_x,
         bool symmetric_y, bool symmetric_z);

  [[nodiscard]] unsigned dimension() const noexcept { return dimension_; }
  [[nodiscard]] Shape shape() const noexcept { return shape_; }
  [[nodiscard]] double length_x() const noexcept { return length_x_; }
  [[nodiscard]] double length_y() const noexcept { return length_y_; }
  [[nodiscard]] double length_z() const noexcept { return length_z_; }
  [[nodiscard]] bool is_symmetric() const noexcept { return symmetric_x_ && symmetric_y_ && symmetric_z_; }
  [[nodiscard]] bool is_symmetric_x() const noexcept { return symmetric_x_; }
  [[nodiscard]] bool is_symmetric_y() const noexcept { return symmetric_y_; }
  [[nodiscard]] bool is_symmetric_z() const noexcept { return symmetric_z_; }

 private:
  static void check_shape_supported(Shape shape);

  unsigned dimension_;
  Shape shape_;
  double length_x_;
  double length_y_;
  double length_z_;
  bool symmetric_x_;
  bool symmetric_y_;
  bool symmetric_z_;
};

}  // namespace fuse::physics::spatial
