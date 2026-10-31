// fuse_physics::fe::ReferenceTopology -- adapted from KARMA's
// karma_reference_topology.h/.C (see reference_topology.hpp's header
// comment for the wave's scope and the MOAB-decoupling rationale).

#include "fuse_physics/fe/reference_topology.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace fuse::physics::fe {

namespace {

Point p(double x, double y = 0.0, double z = 0.0) { return Point(x, y, z); }

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

std::vector<Point> nodes_for(GeometryVariant variant) {
  switch (variant) {
    case GeometryVariant::Edge2:
      return {p(0), p(1)};
    case GeometryVariant::Edge3:
      return {p(0), p(1), p(.5)};
    case GeometryVariant::Tri3:
      return {p(0, 0), p(1, 0), p(0, 1)};
    case GeometryVariant::Tri6:
      return {p(0, 0), p(1, 0), p(0, 1), p(.5, 0), p(.5, .5), p(0, .5)};
    case GeometryVariant::Quad4:
      return {p(0, 0), p(1, 0), p(1, 1), p(0, 1)};
    case GeometryVariant::Quad9:
      return {p(0, 0), p(1, 0), p(1, 1), p(0, 1), p(.5, 0), p(1, .5), p(.5, 1), p(0, .5), p(.5, .5)};
    case GeometryVariant::Tet4:
      return {p(0, 0, 0), p(1, 0, 0), p(0, 1, 0), p(0, 0, 1)};
    case GeometryVariant::Tet10:
      return {p(0, 0, 0), p(1, 0, 0), p(0, 1, 0), p(0, 0, 1), p(.5, 0, 0), p(.5, .5, 0),
              p(0, .5, 0),  p(0, 0, .5), p(.5, 0, .5), p(0, .5, .5)};
    case GeometryVariant::Hex8:
    case GeometryVariant::Hex27: {
      std::vector<Point> result{p(0, 0, 0), p(1, 0, 0), p(1, 1, 0), p(0, 1, 0),
                                 p(0, 0, 1), p(1, 0, 1), p(1, 1, 1), p(0, 1, 1)};
      if (variant == GeometryVariant::Hex8) return result;
      const int edges[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5},
                               {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}};
      for (const auto& edge : edges) result.push_back(.5 * (result[edge[0]] + result[edge[1]]));
      const int faces[][4] = {{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6},
                               {3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};
      for (const auto& face : faces)
        result.push_back(.25 * (result[face[0]] + result[face[1]] + result[face[2]] + result[face[3]]));
      result.push_back(p(.5, .5, .5));
      return result;
    }
  }
  refuse("unknown geometry variant");
}

using SideList = std::vector<std::vector<std::size_t>>;

const SideList& side_table(GeometryVariant variant) {
  static const SideList edge{{0}, {1}};
  static const SideList tri3{{0, 1}, {1, 2}, {2, 0}};
  static const SideList tri6{{0, 1, 3}, {1, 2, 4}, {2, 0, 5}};
  static const SideList quad4{{0, 1}, {1, 2}, {2, 3}, {3, 0}};
  static const SideList quad9{{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}};
  static const SideList tet4{{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  static const SideList tet10{{0, 1, 3, 4, 8, 7}, {1, 2, 3, 5, 9, 8}, {0, 3, 2, 7, 9, 6}, {0, 2, 1, 6, 5, 4}};
  static const SideList hex8{{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};
  static const SideList hex27{{0, 1, 5, 4, 8, 13, 16, 12, 20},  {1, 2, 6, 5, 9, 14, 17, 13, 21},
                               {2, 3, 7, 6, 10, 15, 18, 14, 22}, {3, 0, 4, 7, 11, 12, 19, 15, 23},
                               {0, 3, 2, 1, 11, 10, 9, 8, 24},   {4, 5, 6, 7, 16, 17, 18, 19, 25}};
  switch (variant) {
    case GeometryVariant::Edge2:
    case GeometryVariant::Edge3:
      return edge;
    case GeometryVariant::Tri3:
      return tri3;
    case GeometryVariant::Tri6:
      return tri6;
    case GeometryVariant::Quad4:
      return quad4;
    case GeometryVariant::Quad9:
      return quad9;
    case GeometryVariant::Tet4:
      return tet4;
    case GeometryVariant::Tet10:
      return tet10;
    case GeometryVariant::Hex8:
      return hex8;
    case GeometryVariant::Hex27:
      return hex27;
  }
  refuse("unknown geometry variant");
}

/// Linear Lagrange basis on [0,1] (order 1) or the three-node quadratic
/// Lagrange basis on [0,1] (order 2), value and derivative, in the same
/// {node0, node1, node2} ordering KARMA's own "line" lambda uses
/// (karma_reference_topology.C). node2 is unused at order 1.
std::array<std::pair<double, double>, 3> line_basis(double x, unsigned order) {
  if (order == 1) return {std::pair{1 - x, -1.0}, {x, 1.0}, {0.0, 0.0}};
  return {std::pair{2 * (x - .5) * (x - 1), 4 * x - 3}, {2 * x * (x - .5), 4 * x - 1}, {4 * x * (1 - x), 4 - 8 * x}};
}

}  // namespace

CellFamily cell_family(GeometryVariant variant) noexcept {
  switch (variant) {
    case GeometryVariant::Edge2:
    case GeometryVariant::Edge3:
      return CellFamily::Edge;
    case GeometryVariant::Tri3:
    case GeometryVariant::Tri6:
      return CellFamily::Triangle;
    case GeometryVariant::Quad4:
    case GeometryVariant::Quad9:
      return CellFamily::Quadrilateral;
    case GeometryVariant::Tet4:
    case GeometryVariant::Tet10:
      return CellFamily::Tetrahedron;
    case GeometryVariant::Hex8:
    case GeometryVariant::Hex27:
      return CellFamily::Hexahedron;
  }
  return CellFamily::Edge;
}

unsigned cell_dimension(CellFamily family) noexcept {
  switch (family) {
    case CellFamily::Edge:
      return 1;
    case CellFamily::Triangle:
    case CellFamily::Quadrilateral:
      return 2;
    case CellFamily::Tetrahedron:
    case CellFamily::Hexahedron:
      return 3;
  }
  return 0;
}

unsigned geometry_order(GeometryVariant variant) noexcept {
  switch (variant) {
    case GeometryVariant::Edge2:
    case GeometryVariant::Tri3:
    case GeometryVariant::Quad4:
    case GeometryVariant::Tet4:
    case GeometryVariant::Hex8:
      return 1;
    default:
      return 2;
  }
}

ReferenceTopology::ReferenceTopology(GeometryVariant variant) : variant_(variant), nodes_(nodes_for(variant)) {}

Point ReferenceTopology::center() const noexcept {
  switch (family()) {
    case CellFamily::Edge:
      return p(.5);
    case CellFamily::Triangle:
      return p(1.0 / 3.0, 1.0 / 3.0);
    case CellFamily::Quadrilateral:
      return p(.5, .5);
    case CellFamily::Tetrahedron:
      return p(.25, .25, .25);
    case CellFamily::Hexahedron:
      return p(.5, .5, .5);
  }
  return Point::zero();
}

std::vector<double> ReferenceTopology::evaluate_values(const Point& point) const {
  std::vector<double> values(node_count());
  evaluate_values(point, values);
  return values;
}

void ReferenceTopology::evaluate_values(const Point& point, std::span<double> values) const {
  if (values.size() != node_count()) refuse("reference value storage has wrong size");
  std::vector<Point> derivatives(node_count());
  evaluate(point, values, derivatives);
}

ReferenceValues ReferenceTopology::evaluate(const Point& point) const {
  ReferenceValues result{std::vector<double>(node_count()), std::vector<Point>(node_count())};
  evaluate(point, result.values, result.derivatives);
  return result;
}

void ReferenceTopology::evaluate(const Point& point, std::span<double> output_values,
                                  std::span<Point> output_derivatives) const {
  if (output_values.size() != node_count() || output_derivatives.size() != node_count())
    refuse("reference evaluation storage has wrong size");
  if (!point.all_finite()) refuse("reference point must be finite");
  std::fill(output_values.begin(), output_values.end(), 0.0);
  std::fill(output_derivatives.begin(), output_derivatives.end(), Point::zero());
  const unsigned ord = order();

  if (family() == CellFamily::Edge) {
    const auto basis = line_basis(point(0), ord);
    for (std::size_t i = 0; i < node_count(); ++i) {
      output_values[i] = basis[i].first;
      output_derivatives[i](0) = basis[i].second;
    }
    return;
  }

  if (family() == CellFamily::Triangle || family() == CellFamily::Tetrahedron) {
    const bool tet = family() == CellFamily::Tetrahedron;
    const std::array<double, 4> lambda{1 - point(0) - point(1) - (tet ? point(2) : 0.0), point(0), point(1),
                                        point(2)};
    const std::size_t count = tet ? 4 : 3;
    const Point dl[] = {p(-1, -1, tet ? -1 : 0), p(1, 0, 0), p(0, 1, 0), p(0, 0, 1)};
    for (std::size_t i = 0; i < count; ++i) {
      output_values[i] = ord == 1 ? lambda[i] : lambda[i] * (2 * lambda[i] - 1);
      output_derivatives[i] = (ord == 1 ? 1.0 : 4 * lambda[i] - 1) * dl[i];
    }
    constexpr std::array<std::pair<std::size_t, std::size_t>, 6> edges{
        std::pair{0u, 1u}, {1u, 2u}, {2u, 0u}, {0u, 3u}, {1u, 3u}, {2u, 3u}};
    const std::size_t edge_count = tet ? 6 : 3;
    if (ord == 2)
      for (std::size_t e = 0; e < edge_count; ++e) {
        const auto [i, j] = edges[e];
        const std::size_t n = count + e;
        output_values[n] = 4 * lambda[i] * lambda[j];
        output_derivatives[n] = 4 * (lambda[j] * dl[i] + lambda[i] * dl[j]);
      }
    return;
  }

  // Quadrilateral and hexahedron: tensor product of 1-D line_basis along
  // each axis a reference node's coordinate selects (0, 1, or 1/2).
  const auto bx = line_basis(point(0), ord), by = line_basis(point(1), ord), bz = line_basis(point(2), ord);
  const bool hex = family() == CellFamily::Hexahedron;
  for (std::size_t n = 0; n < node_count(); ++n) {
    const Point& q = nodes_[n];
    const auto index = [](double v) { return v == 0.0 ? 0u : v == 1.0 ? 1u : 2u; };
    const unsigned i = index(q(0)), j = index(q(1)), k = index(q(2));
    const double bzk = hex ? bz[k].first : 1.0;
    output_values[n] = bx[i].first * by[j].first * bzk;
    output_derivatives[n] = p(bx[i].second * by[j].first * bzk, bx[i].first * by[j].second * bzk,
                               hex ? bx[i].first * by[j].first * bz[k].second : 0.0);
  }
}

unsigned ReferenceTopology::side_count() const noexcept {
  return static_cast<unsigned>(side_table(variant_).size());
}

std::vector<std::size_t> ReferenceTopology::side_nodes(unsigned side) const {
  const auto& sides = side_table(variant_);
  if (side >= sides.size()) refuse("side index is out of range");
  return sides[side];
}

ReferenceTopology ReferenceTopology::side_topology(unsigned side) const {
  if (side >= side_count()) refuse("side index is out of range");
  switch (family()) {
    case CellFamily::Triangle:
    case CellFamily::Quadrilateral:
      return ReferenceTopology(order() == 1 ? GeometryVariant::Edge2 : GeometryVariant::Edge3);
    case CellFamily::Tetrahedron:
      return ReferenceTopology(order() == 1 ? GeometryVariant::Tri3 : GeometryVariant::Tri6);
    case CellFamily::Hexahedron:
      return ReferenceTopology(order() == 1 ? GeometryVariant::Quad4 : GeometryVariant::Quad9);
    case CellFamily::Edge:
      refuse("an edge's sides are vertices, which have no ReferenceTopology of their own");
  }
  refuse("unknown cell family");
}

namespace {
/// The side's linear corner count (first entries of its side_nodes()
/// list, by the same CN convention karma_reference_topology.C relies on
/// for moab::CN::SubEntityVertexIndices: primary corners first, midside
/// nodes appended after).
unsigned linear_corner_count(CellFamily parent_family) {
  switch (parent_family) {
    case CellFamily::Triangle:
    case CellFamily::Quadrilateral:
      return 2;  // an edge side
    case CellFamily::Tetrahedron:
      return 3;  // a triangle side
    case CellFamily::Hexahedron:
      return 4;  // a quadrilateral side
    case CellFamily::Edge:
      return 1;
  }
  return 0;
}
}  // namespace

Point ReferenceTopology::side_reference_map(unsigned side, const Point& point) const {
  if (side >= side_count()) refuse("side index is out of range");
  const auto side_node_indices = side_nodes(side);
  const unsigned corners = linear_corner_count(family());
  const GeometryVariant linear = family() == CellFamily::Triangle || family() == CellFamily::Quadrilateral
                                      ? GeometryVariant::Edge2
                                  : family() == CellFamily::Tetrahedron ? GeometryVariant::Tri3
                                                                        : GeometryVariant::Quad4;
  const auto values = ReferenceTopology(linear).evaluate(point);
  Point result = Point::zero();
  for (unsigned i = 0; i < corners; ++i) result += values.values[i] * nodes_[side_node_indices[i]];
  return result;
}

Matrix3 ReferenceTopology::side_reference_jacobian(unsigned side, const Point& point) const {
  if (side >= side_count()) refuse("side index is out of range");
  const auto side_node_indices = side_nodes(side);
  const unsigned corners = linear_corner_count(family());
  const GeometryVariant linear = family() == CellFamily::Triangle || family() == CellFamily::Quadrilateral
                                      ? GeometryVariant::Edge2
                                  : family() == CellFamily::Tetrahedron ? GeometryVariant::Tri3
                                                                        : GeometryVariant::Quad4;
  const auto values = ReferenceTopology(linear).evaluate(point);
  Matrix3 result = Matrix3::zero();
  for (unsigned i = 0; i < corners; ++i)
    for (unsigned d = 0; d < dimension() - 1; ++d)
      result.set_col(d, result.col(d) + values.derivatives[i](d) * nodes_[side_node_indices[i]]);
  return result;
}

}  // namespace fuse::physics::fe
