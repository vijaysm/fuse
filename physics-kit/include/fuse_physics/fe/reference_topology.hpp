#pragma once

/// Reference cells and their analytic shape-function interpolation, adapted
/// from KARMA's karma_reference_topology.h/.C (docs/dev/physics-kit-
/// disposition.md Appendix A, port-adapt, owner decision D-A).
///
/// WAVE 1 SCOPE. This port covers the five element families KARMA calls
/// Edge, Tri, Quad, Tet and Hex at Lagrange order 1 and 2 (GeometryVariant
/// below), i.e. the simplex and tensor-product cells whose shape functions
/// are the textbook barycentric/tensor-Lagrange forms. Prism and pyramid
/// (KARMA's GeometryVariant::Prism6/15/18, Pyramid5/13/14) are NOT ported in
/// this wave: the pyramid family needs a rational (non-polynomial) basis
/// construction (karma_reference_topology.C's pyramid_values(), built from a
/// Vandermonde solve over a collapsed-coordinate candidate set) that is a
/// substantial separate numerical undertaking, and upstream/karma/
/// KNOWN_ISSUES.md item 1 already flags the pyramid's quadrature accuracy as
/// an unverified risk in KARMA itself -- see this wave's KNOWN_ISSUES.md
/// hunk, which carries that note forward, and design_notes/open_questions
/// for the deferral. A request for one of the three Prism/Pyramid variants
/// is refused with a clear message (Cell::Unsupported... never constructed
/// for them); see physics.fe_reference_topology_test's
/// "RefusesUnimplementedCellFamilies" case.
///
/// DECOUPLING FROM MOAB (docs/dev/layering.md's public-header third-party
/// rule). KARMA's KARMAReferenceTopology::topology()/from_moab() return and
/// accept moab::EntityType; tools/check-layering.py's FORBIDDEN_PUBLIC bans
/// "moab/" in any physics-kit public header. GeometryVariant plays the same
/// role KARMA's (GeometryVariant, moab::EntityType) pair did, as a plain
/// physics-kit-own enum; a moab::EntityType <-> GeometryVariant conversion,
/// where physics-kit eventually needs one for a MOAB-backed mesh binding, is
/// left to that (not-yet-written, FUSE_HAVE_MOAB-gated) translation unit,
/// named as an open gap rather than guessed at here.

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

/// The geometric element families this wave ports, Lagrange order 1 (linear)
/// and 2 (quadratic, full tensor/complete simplex -- not the Quad8
/// serendipity family, which KARMA itself treats as a distinct variant and
/// this wave defers along with Prism/Pyramid).
enum class GeometryVariant { Edge2, Edge3, Tri3, Tri6, Quad4, Quad9, Tet4, Tet10, Hex8, Hex27 };

/// Which reference cell a GeometryVariant belongs to, irrespective of order.
enum class CellFamily { Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron };

[[nodiscard]] CellFamily cell_family(GeometryVariant variant) noexcept;
[[nodiscard]] unsigned cell_dimension(CellFamily family) noexcept;
[[nodiscard]] unsigned geometry_order(GeometryVariant variant) noexcept;

/// Shape values and derivatives at one reference point, parallel to the
/// node ordering of the ReferenceTopology that produced them.
struct ReferenceValues {
  std::vector<double> values;
  std::vector<Point> derivatives;
};

/// A fixed reference cell (family + order) and its analytic Lagrange shape
/// functions, ported from KARMAReferenceTopology.
class ReferenceTopology {
 public:
  explicit ReferenceTopology(GeometryVariant variant);

  [[nodiscard]] GeometryVariant variant() const noexcept { return variant_; }
  [[nodiscard]] CellFamily family() const noexcept { return cell_family(variant_); }
  [[nodiscard]] unsigned dimension() const noexcept { return cell_dimension(family()); }
  [[nodiscard]] unsigned order() const noexcept { return geometry_order(variant_); }
  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
  [[nodiscard]] const std::vector<Point>& reference_nodes() const noexcept { return nodes_; }

  [[nodiscard]] Point center() const noexcept;

  /// Shape values only.
  [[nodiscard]] std::vector<double> evaluate_values(const Point& point) const;
  void evaluate_values(const Point& point, std::span<double> values) const;

  /// Shape values and reference-space derivatives.
  [[nodiscard]] ReferenceValues evaluate(const Point& point) const;
  void evaluate(const Point& point, std::span<double> values, std::span<Point> derivatives) const;

  [[nodiscard]] unsigned side_count() const noexcept;
  /// Node indices (into reference_nodes()) of one codimension-1 side, CN
  /// convention: the side's own linear corners first, any midside nodes
  /// after -- see fe/topology_tables.hpp's side tables.
  [[nodiscard]] std::vector<std::size_t> side_nodes(unsigned side) const;
  [[nodiscard]] ReferenceTopology side_topology(unsigned side) const;
  [[nodiscard]] Point side_reference_map(unsigned side, const Point& point) const;
  [[nodiscard]] Matrix3 side_reference_jacobian(unsigned side, const Point& point) const;

 private:
  GeometryVariant variant_;
  std::vector<Point> nodes_;
};

}  // namespace fuse::physics::fe
