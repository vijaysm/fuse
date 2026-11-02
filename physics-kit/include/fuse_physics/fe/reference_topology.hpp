#pragma once

/// Reference cells and their analytic shape-function interpolation, adapted
/// from KARMA's karma_reference_topology.h/.C (docs/dev/physics-kit-
/// disposition.md Appendix A, port-adapt, owner decision D-A).
///
/// WAVE 1b SCOPE (owner instruction 2026-10-31, "execute FE wave 1 first
/// completely"): this port now covers all seven element families KARMA's
/// own GeometryVariant names -- Edge, Triangle, Quadrilateral, Tetrahedron,
/// Hexahedron (wave 1) plus Prism and Pyramid (wave 1b) -- at Lagrange
/// order 1 and 2 (the complete tensor/simplex family, e.g. Quad9/Tet10/
/// Hex27/Prism18/Pyramid14, not the Quad8 serendipity variant).
///
/// THE PYRAMID IS RATIONAL, NOT POLYNOMIAL. Its five linear (and 14
/// quadratic) shape functions are built from a collapsed-coordinate
/// candidate set containing terms like x*y/(1-z), which are finite
/// everywhere except the exact apex (1-z = 0) where they are removable
/// singularities evaluated by direct limit, not by the rational formula --
/// see basis.cpp's pyramid_candidates() and its own header comment, ported
/// from karma_reference_topology.C's pyramid_values()/pyramid_candidates().
/// The construction solves a square linear system (candidate values at
/// each reference node) whose conditioning is measured and refused above a
/// derived threshold -- see basis.cpp's pyramid_nodal_coefficients().
///
/// DECOUPLING FROM MOAB (docs/dev/layering.md's public-header third-party
/// rule). KARMA's KARMAReferenceTopology::topology()/from_moab() return and
/// accept moab::EntityType; tools/check-layering.py's FORBIDDEN_PUBLIC bans
/// "moab/" in any physics-kit public header. GeometryVariant plays the same
/// role KARMA's (GeometryVariant, moab::EntityType) pair did, as a plain
/// physics-kit-own enum; a moab::EntityType <-> GeometryVariant conversion,
/// where physics-kit eventually needs one for a MOAB-backed mesh binding, is
/// left to that translation unit.

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

/// The geometric element families this port implements, Lagrange order 1
/// (linear) and 2 (quadratic, full tensor/complete simplex -- not the
/// serendipity Quad8).
enum class GeometryVariant {
  Edge2,
  Edge3,
  Tri3,
  Tri6,
  Quad4,
  // Quad8 (serendipity, 8 nodes) exists ONLY as a face shape -- Prism15's
  // and Pyramid13's one quadrilateral side (reference_topology.cpp's
  // side_topology()) -- and is never registered as a stand-alone basis
  // family (basis.cpp's variant_for() always selects Quad9 for a
  // Quadrilateral cell's own order-2 basis).
  Quad8,
  Quad9,
  Tet4,
  Tet10,
  Hex8,
  Hex27,
  Prism6,
  Prism15,
  Prism18,
  Pyramid5,
  Pyramid13,
  Pyramid14
};

/// Which reference cell a GeometryVariant belongs to, irrespective of
/// order. Appended (Prism, Pyramid) rather than inserted, so the five
/// wave-1 enumerators keep their original integral values.
enum class CellFamily { Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid };

[[nodiscard]] CellFamily cell_family(GeometryVariant variant) noexcept;
[[nodiscard]] unsigned cell_dimension(CellFamily family) noexcept;
[[nodiscard]] unsigned geometry_order(GeometryVariant variant) noexcept;

/// How many EDGES (1-D sub-entities, distinct from a codimension-1
/// "side") `family`'s reference cell has, and which two of its own P1
/// CORNER indices bound edge `edge` -- the SAME tables (by content and
/// by order) KARMA's own nodes_for() uses to place an order-2 basis's
/// edge-midpoint nodes (karma_reference_topology.C), confirmed here by
/// matching this port's OWN nodes_for() (reference_topology.cpp) edge
/// midpoint coordinates against each table entry, so a caller indexing
/// by `edge` here and a caller indexing by `DofAssociation::local_
/// entity_index` into the order-2 node list (basis.cpp's own
/// dof_associations(), which pushes EXACTLY `edge_count(family)` edge
/// associations, one per edge, in this same order) agree on which edge
/// is which. For `Edge`, `Triangle`, and `Quadrilateral` an "edge" is
/// the SAME sub-entity `ReferenceTopology::side_nodes`/`side_topology`
/// already name a "side" (a 2-D or 1-D cell's own codimension-1
/// facets); `edge_vertices` still answers for those families (reading
/// the corner pair directly, no side machinery involved) so a caller
/// never needs to branch on dimension to find an edge's two corners.
[[nodiscard]] unsigned edge_count(CellFamily family) noexcept;
[[nodiscard]] std::array<std::size_t, 2> edge_vertices(CellFamily family, unsigned edge);

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

  /// Shape values only. Throws for Pyramid at the exact apex (0,0,1): the
  /// shape VALUES are well-defined there (basis.cpp evaluates the removable
  /// limit directly), so this is not that case -- see evaluate_values's
  /// own apex special-case in reference_topology.cpp, ported unchanged from
  /// KARMA's identical carve-out.
  [[nodiscard]] std::vector<double> evaluate_values(const Point& point) const;
  void evaluate_values(const Point& point, std::span<double> values) const;

  /// Shape values and reference-space derivatives. Throws for Pyramid at
  /// the exact apex: the DERIVATIVES are genuinely path-dependent there
  /// (KARMA's own documented carve-out, karma_geometry_map.h's jacobian()
  /// comment), so this is refused rather than returning a direction-
  /// dependent answer silently.
  [[nodiscard]] ReferenceValues evaluate(const Point& point) const;
  void evaluate(const Point& point, std::span<double> values, std::span<Point> derivatives) const;

  [[nodiscard]] unsigned side_count() const noexcept;
  /// Node indices (into reference_nodes()) of one codimension-1 side, CN
  /// convention: the side's own linear corners first, any midside nodes
  /// after.
  [[nodiscard]] std::vector<std::size_t> side_nodes(unsigned side) const;
  /// The side's own cell shape. A Hexahedron's sides are uniformly
  /// Quadrilateral and a Tetrahedron's uniformly Triangle, but Prism and
  /// Pyramid have MIXED face shapes (prism: 3 quad + 2 tri; pyramid: 4 tri
  /// + 1 quad) -- callers that need to know WHICH shape a given side is
  /// (fe_base.cpp's own per-side quadrature-rule selection) read
  /// side_topology(side).family().
  [[nodiscard]] ReferenceTopology side_topology(unsigned side) const;
  [[nodiscard]] Point side_reference_map(unsigned side, const Point& point) const;
  [[nodiscard]] Matrix3 side_reference_jacobian(unsigned side, const Point& point) const;

 private:
  GeometryVariant variant_;
  std::vector<Point> nodes_;
};

}  // namespace fuse::physics::fe
