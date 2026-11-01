// fuse_physics::fe::ReferenceTopology -- adapted from KARMA's
// karma_reference_topology.h/.C (see reference_topology.hpp's header
// comment for the MOAB-decoupling rationale).
//
// WAVE 1b (owner instruction 2026-10-31): adds Prism6/15/18 and
// Pyramid5/13/14 (node tables, side tables, and the wedge/pyramid branches
// of evaluate()), plus the serendipity GeometryVariant::Quad8 -- needed
// only as the face shape of Prism15's and Pyramid13's one quadrilateral
// side (KARMA's own side_topology() names Quad8 for exactly those two
// cases; see this file's side_topology() below), not registered as a
// stand-alone basis family (basis.cpp's variant_for() never selects it).

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
    case GeometryVariant::Quad8:
      return {p(0, 0), p(1, 0), p(1, 1), p(0, 1), p(.5, 0), p(1, .5), p(.5, 1), p(0, .5)};
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
    case GeometryVariant::Prism6:
    case GeometryVariant::Prism15:
    case GeometryVariant::Prism18: {
      std::vector<Point> result{p(0, 0, 0), p(1, 0, 0), p(0, 1, 0), p(0, 0, 1), p(1, 0, 1), p(0, 1, 1)};
      if (variant == GeometryVariant::Prism6) return result;
      const int edges[][2] = {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {5, 3}};
      for (const auto& edge : edges) result.push_back(.5 * (result[edge[0]] + result[edge[1]]));
      if (variant == GeometryVariant::Prism18) {
        result.push_back(p(.5, 0, .5));
        result.push_back(p(.5, .5, .5));
        result.push_back(p(0, .5, .5));
      }
      return result;
    }
    case GeometryVariant::Pyramid5:
    case GeometryVariant::Pyramid13:
    case GeometryVariant::Pyramid14: {
      std::vector<Point> result{p(-1, -1, 0), p(1, -1, 0), p(1, 1, 0), p(-1, 1, 0), p(0, 0, 1)};
      if (variant == GeometryVariant::Pyramid5) return result;
      const int edges[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}};
      for (const auto& edge : edges) result.push_back(.5 * (result[edge[0]] + result[edge[1]]));
      if (variant == GeometryVariant::Pyramid14) result.push_back(p(0, 0, 0));
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
  static const SideList quad_quadratic{{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}};
  static const SideList tet4{{0, 1, 3}, {1, 2, 3}, {0, 3, 2}, {0, 2, 1}};
  static const SideList tet10{{0, 1, 3, 4, 8, 7}, {1, 2, 3, 5, 9, 8}, {0, 3, 2, 7, 9, 6}, {0, 2, 1, 6, 5, 4}};
  static const SideList hex8{{0, 1, 5, 4}, {1, 2, 6, 5}, {2, 3, 7, 6}, {3, 0, 4, 7}, {0, 3, 2, 1}, {4, 5, 6, 7}};
  static const SideList hex27{{0, 1, 5, 4, 8, 13, 16, 12, 20},  {1, 2, 6, 5, 9, 14, 17, 13, 21},
                               {2, 3, 7, 6, 10, 15, 18, 14, 22}, {3, 0, 4, 7, 11, 12, 19, 15, 23},
                               {0, 3, 2, 1, 11, 10, 9, 8, 24},   {4, 5, 6, 7, 16, 17, 18, 19, 25}};
  static const SideList prism6{{0, 1, 4, 3}, {1, 2, 5, 4}, {0, 3, 5, 2}, {0, 2, 1}, {3, 4, 5}};
  static const SideList prism15{{0, 1, 4, 3, 6, 10, 12, 9}, {1, 2, 5, 4, 7, 11, 13, 10},
                                 {0, 3, 5, 2, 9, 14, 11, 8}, {0, 2, 1, 8, 7, 6}, {3, 4, 5, 12, 13, 14}};
  static const SideList prism18{{0, 1, 4, 3, 6, 10, 12, 9, 15}, {1, 2, 5, 4, 7, 11, 13, 10, 16},
                                 {0, 3, 5, 2, 9, 14, 11, 8, 17}, {0, 2, 1, 8, 7, 6}, {3, 4, 5, 12, 13, 14}};
  static const SideList pyramid5{{0, 1, 4}, {1, 2, 4}, {2, 3, 4}, {3, 0, 4}, {0, 3, 2, 1}};
  static const SideList pyramid13{{0, 1, 4, 5, 10, 9}, {1, 2, 4, 6, 11, 10}, {2, 3, 4, 7, 12, 11},
                                   {3, 0, 4, 8, 9, 12}, {0, 3, 2, 1, 8, 7, 6, 5}};
  static const SideList pyramid14{{0, 1, 4, 5, 10, 9}, {1, 2, 4, 6, 11, 10}, {2, 3, 4, 7, 12, 11},
                                   {3, 0, 4, 8, 9, 12}, {0, 3, 2, 1, 8, 7, 6, 5, 13}};
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
    case GeometryVariant::Quad8:
    case GeometryVariant::Quad9:
      return quad_quadratic;
    case GeometryVariant::Tet4:
      return tet4;
    case GeometryVariant::Tet10:
      return tet10;
    case GeometryVariant::Hex8:
      return hex8;
    case GeometryVariant::Hex27:
      return hex27;
    case GeometryVariant::Prism6:
      return prism6;
    case GeometryVariant::Prism15:
      return prism15;
    case GeometryVariant::Prism18:
      return prism18;
    case GeometryVariant::Pyramid5:
      return pyramid5;
    case GeometryVariant::Pyramid13:
      return pyramid13;
    case GeometryVariant::Pyramid14:
      return pyramid14;
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

/// The serendipity Quad8 shape functions, ported verbatim from
/// karma_reference_topology.C's own `if (_variant == GeometryVariant::Quad8)`
/// branch. Used only as a face shape (this file's side_topology()), never
/// registered as a stand-alone basis family.
void evaluate_quad8(const Point& point, std::span<double> values, std::span<Point> derivatives) {
  const double x = 2 * point(0) - 1, y = 2 * point(1) - 1;
  const double sx[] = {-1, 1, 1, -1}, sy[] = {-1, -1, 1, 1};
  for (std::size_t i = 0; i < 4; ++i) {
    values[i] = .25 * (1 + sx[i] * x) * (1 + sy[i] * y) * (sx[i] * x + sy[i] * y - 1);
    derivatives[i](0) = .5 * sx[i] * (1 + sy[i] * y) * (2 * sx[i] * x + sy[i] * y);
    derivatives[i](1) = .5 * sy[i] * (1 + sx[i] * x) * (sx[i] * x + 2 * sy[i] * y);
  }
  values[4] = .5 * (1 - x * x) * (1 - y);
  values[5] = .5 * (1 + x) * (1 - y * y);
  values[6] = .5 * (1 - x * x) * (1 + y);
  values[7] = .5 * (1 - x) * (1 - y * y);
  derivatives[4] = p(-2 * x * (1 - y), -(1 - x * x));
  derivatives[5] = p(1 - y * y, -2 * y * (1 + x));
  derivatives[6] = p(-2 * x * (1 + y), 1 - x * x);
  derivatives[7] = p(-(1 - y * y), -2 * y * (1 - x));
}

/// One rational candidate term (karma_reference_topology.C's
/// pyramid_candidates/Candidate) for the pyramid's collapsed-coordinate
/// basis: value and reference-space derivative of
/// x^i y^j (1-z)^(m-i-j) z^k, m = max(i,j), evaluated directly at the
/// removable limit (1-z) -> 0 when it occurs (the apex, where x and y are
/// also forced to 0 by the pyramid's own geometry), rather than through
/// the rational formula itself.
struct Candidate {
  double value;
  Point derivative;
};
struct PyramidCandidates {
  std::array<Candidate, 14> values{};
  std::size_t size = 0;
};

PyramidCandidates pyramid_candidates(const Point& point, unsigned order) {
  const double x = point(0), y = point(1), z = point(2), w = 1.0 - z;
  PyramidCandidates result;
  for (unsigned i = 0; i <= order; ++i)
    for (unsigned j = 0; j <= order; ++j) {
      const unsigned m = std::max(i, j);
      for (unsigned k = 0; k + m <= order; ++k) {
        if (w == 0.0) {
          const double value = (i == 0 && j == 0) ? std::pow(z, static_cast<int>(k)) : 0.0;
          double dx = 0.0, dy = 0.0, dz = 0.0;
          if (i == 1 && j == 0 && m == 1) dx = std::pow(z, static_cast<int>(k));
          if (j == 1 && i == 0 && m == 1) dy = std::pow(z, static_cast<int>(k));
          if (i == 0 && j == 0 && k > 0) dz = k * std::pow(z, static_cast<int>(k - 1));
          result.values[result.size++] = {value, p(dx, dy, dz)};
          continue;
        }
        const double value = std::pow(x, static_cast<int>(i)) * std::pow(y, static_cast<int>(j)) *
                              std::pow(w, static_cast<int>(m - i - j)) * std::pow(z, static_cast<int>(k));
        const int w_power = static_cast<int>(m) - static_cast<int>(i) - static_cast<int>(j);
        auto direct = [&](unsigned di, unsigned dj, unsigned dk, int dw) {
          if ((di && x == 0.0) || (dj && y == 0.0) || (dk && z == 0.0)) return 0.0;
          return std::pow(x, static_cast<int>(di)) * std::pow(y, static_cast<int>(dj)) * std::pow(w, dw) *
                 std::pow(z, static_cast<int>(dk));
        };
        result.values[result.size++] = {
            value, p(i ? i * direct(i - 1, j, k, w_power) : 0.0, j ? j * direct(i, j - 1, k, w_power) : 0.0,
                     (k ? k * direct(i, j, k - 1, w_power) : 0.0) -
                         (w_power ? w_power * direct(i, j, k, w_power - 1) : 0.0))};
      }
    }
  return result;
}

/// Inverts an n x n matrix by Gauss-Jordan elimination with partial
/// (largest-in-column) row pivoting -- Golub, G.H. and Van Loan, C.F.,
/// "Matrix Computations", 4th ed., 2013, Algorithm 3.2.1 (LU with partial
/// pivoting), specialised here to explicit matrix inversion by
/// back-substituting each unit column. Returns the inverse and the
/// infinity-norm condition number estimate cond = ||A||_inf * ||A^-1||_inf
/// (Golub & Van Loan Section 2.3.2's standard definition), computed
/// directly from the two matrices now that both are in hand -- an exact
/// number for THIS A, not a cheap approximation.
///
/// Refuses, naming the measured condition number, when it exceeds
/// `1.0 / sqrt(eps)` (~1.49e8 in double precision): a linear solve loses
/// roughly log10(cond) decimal digits of accuracy to rounding (Golub & Van
/// Loan Section 2.3.3's standard rule of thumb, `Ax=b`'s relative error
/// bound is O(cond * eps)), so cond >= 1/sqrt(eps) is the exact point at
/// which fewer than the eps-to-sqrt(eps) crossover's ~8 significant digits
/// survive -- below this wave's own 1e-10 pointwise tolerance floor
/// (fe_reference_topology_test.cpp's Kronecker/partition-of-unity checks),
/// so a coefficient computed past this threshold could not be trusted to
/// that floor regardless of which test exercises it.
struct SquareInverse {
  std::vector<double> inverse;  // row-major, n*n
  double condition_number = 0.0;
};

SquareInverse invert_square_or_refuse(std::vector<double> a, std::size_t n) {
  std::vector<double> original = a;
  std::vector<double> inv(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) inv[i * n + i] = 1.0;

  for (std::size_t col = 0; col < n; ++col) {
    std::size_t pivot_row = col;
    double pivot_value = std::abs(a[col * n + col]);
    for (std::size_t row = col + 1; row < n; ++row) {
      if (std::abs(a[row * n + col]) > pivot_value) {
        pivot_value = std::abs(a[row * n + col]);
        pivot_row = row;
      }
    }
    if (pivot_value == 0.0) refuse("pyramid interpolation matrix is exactly singular");
    if (pivot_row != col) {
      for (std::size_t c = 0; c < n; ++c) std::swap(a[col * n + c], a[pivot_row * n + c]);
      for (std::size_t c = 0; c < n; ++c) std::swap(inv[col * n + c], inv[pivot_row * n + c]);
    }
    const double pivot = a[col * n + col];
    for (std::size_t c = 0; c < n; ++c) {
      a[col * n + c] /= pivot;
      inv[col * n + c] /= pivot;
    }
    for (std::size_t row = 0; row < n; ++row) {
      if (row == col) continue;
      const double factor = a[row * n + col];
      if (factor == 0.0) continue;
      for (std::size_t c = 0; c < n; ++c) {
        a[row * n + c] -= factor * a[col * n + c];
        inv[row * n + c] -= factor * inv[col * n + c];
      }
    }
  }

  const auto row_sum_abs = [n](const std::vector<double>& m) {
    double worst = 0.0;
    for (std::size_t r = 0; r < n; ++r) {
      double s = 0.0;
      for (std::size_t c = 0; c < n; ++c) s += std::abs(m[r * n + c]);
      worst = std::max(worst, s);
    }
    return worst;
  };
  const double norm_a = row_sum_abs(original);
  const double norm_inv = row_sum_abs(inv);
  const double condition_number = norm_a * norm_inv;
  const double threshold = 1.0 / std::sqrt(std::numeric_limits<double>::epsilon());
  if (!std::isfinite(condition_number) || condition_number > threshold)
    refuse(("pyramid interpolation matrix is ill-conditioned: infinity-norm condition number " +
             std::to_string(condition_number) + " exceeds the derived threshold 1/sqrt(eps) = " +
             std::to_string(threshold))
                .c_str());
  return {inv, condition_number};
}

/// Builds the nodal (Lagrange) basis coefficients for the pyramid's
/// collapsed-coordinate candidate set at the given order: the n x n matrix
/// of each candidate evaluated at each reference node (n = 5 for order 1,
/// 14 for order 2 -- the candidate count equals the node count exactly for
/// both orders, verified by this file's own
/// PyramidCandidateCountMatchesNodeCount test, so this is a genuine square
/// solve, not a rank-reducing selection), inverted once and cached (the
/// nodes are fixed reference data, read-only after the first call).
const std::vector<double>& pyramid_nodal_coefficients(unsigned order) {
  static const std::vector<double> linear = [] {
    const auto nodes = nodes_for(GeometryVariant::Pyramid5);
    const std::size_t n = nodes.size();
    std::vector<double> matrix(n * n);
    for (std::size_t r = 0; r < n; ++r) {
      const auto c = pyramid_candidates(nodes[r], 1);
      if (c.size != n) refuse("pyramid order-1 candidate count does not match node count");
      for (std::size_t j = 0; j < n; ++j) matrix[r * n + j] = c.values[j].value;
    }
    return invert_square_or_refuse(matrix, n).inverse;
  }();
  static const std::vector<double> quadratic = [] {
    const auto nodes = nodes_for(GeometryVariant::Pyramid14);
    const std::size_t n = nodes.size();
    std::vector<double> matrix(n * n);
    for (std::size_t r = 0; r < n; ++r) {
      const auto c = pyramid_candidates(nodes[r], 2);
      if (c.size != n) refuse("pyramid order-2 candidate count does not match node count");
      for (std::size_t j = 0; j < n; ++j) matrix[r * n + j] = c.values[j].value;
    }
    return invert_square_or_refuse(matrix, n).inverse;
  }();
  return order == 1 ? linear : quadratic;
}

/// Ported from karma_reference_topology.C's pyramid_values(): evaluates
/// the nodal pyramid basis at `point` by combining the candidate values
/// there with the cached inverse coefficients, then (for Pyramid13 only)
/// folding the 14th (Pyramid14-only) node's contribution into the 13
/// actually-present nodes via the fixed linear combination KARMA's own
/// code states (`weights` below).
void pyramid_values(GeometryVariant variant, const Point& point, const std::vector<Point>& nodes,
                     std::span<double> output_values, std::span<Point> output_derivatives) {
  const unsigned order = variant == GeometryVariant::Pyramid5 ? 1 : 2;
  const auto& coefficients = pyramid_nodal_coefficients(order);
  const std::size_t n = order == 1 ? 5 : 14;
  const auto candidates = pyramid_candidates(point, order);

  std::array<double, 14> nodal_values{};
  std::array<Point, 14> nodal_derivatives{};
  for (std::size_t column = 0; column < n; ++column)
    for (std::size_t row = 0; row < n; ++row) {
      const double coefficient = coefficients[row * n + column];
      nodal_values[column] += candidates.values[row].value * coefficient;
      nodal_derivatives[column] += candidates.values[row].derivative * coefficient;
    }
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    output_values[i] = nodal_values[i];
    output_derivatives[i] = nodal_derivatives[i];
  }
  if (variant == GeometryVariant::Pyramid13) {
    constexpr double weights[] = {-.25, -.25, -.25, -.25, 0, .5, .5, .5, .5, 0, 0, 0, 0};
    for (std::size_t i = 0; i < nodes.size(); ++i) {
      output_values[i] += weights[i] * nodal_values[13];
      output_derivatives[i] += weights[i] * nodal_derivatives[13];
    }
  }
}

unsigned linear_corner_count_for_shape(CellFamily shape) {
  switch (shape) {
    case CellFamily::Edge:
      return 2;
    case CellFamily::Triangle:
      return 3;
    case CellFamily::Quadrilateral:
      return 4;
    default:
      refuse("a side's own shape must be Edge, Triangle, or Quadrilateral");
  }
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
    case GeometryVariant::Quad8:
    case GeometryVariant::Quad9:
      return CellFamily::Quadrilateral;
    case GeometryVariant::Tet4:
    case GeometryVariant::Tet10:
      return CellFamily::Tetrahedron;
    case GeometryVariant::Hex8:
    case GeometryVariant::Hex27:
      return CellFamily::Hexahedron;
    case GeometryVariant::Prism6:
    case GeometryVariant::Prism15:
    case GeometryVariant::Prism18:
      return CellFamily::Prism;
    case GeometryVariant::Pyramid5:
    case GeometryVariant::Pyramid13:
    case GeometryVariant::Pyramid14:
      return CellFamily::Pyramid;
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
    case CellFamily::Prism:
    case CellFamily::Pyramid:
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
    case GeometryVariant::Prism6:
    case GeometryVariant::Pyramid5:
      return 1;
    default:
      return 2;
  }
}

unsigned edge_count(CellFamily family) noexcept {
  // Edge, Triangle, Quadrilateral, Tetrahedron, Hexahedron, Prism, Pyramid --
  // the SAME table basis.cpp's own (private) edge_count_of duplicates;
  // kept in sync by this file's own test (fe_reference_topology_test.cpp's
  // EdgeVerticesMatchNodesForMidpoints cross-checks every entry against
  // nodes_for()'s own edge-midpoint construction above).
  constexpr std::array<unsigned, 7> counts{1, 3, 4, 6, 12, 9, 8};
  return counts[static_cast<std::size_t>(family)];
}

std::array<std::size_t, 2> edge_vertices(CellFamily family, unsigned edge) {
  // Edge/Triangle/Quadrilateral: an "edge" is that family's own side
  // (ReferenceTopology::side_nodes' own corner-pair/triple/quad), read
  // directly here rather than through side machinery so a caller never
  // branches on dimension. Tetrahedron/Hexahedron/Prism/Pyramid: the
  // SAME {corner, corner} tables this file's own nodes_for() uses to
  // place each family's order-2 edge-midpoint nodes (KARMA's
  // karma_reference_topology.C has the identical tables, by content and
  // by order -- this port's own nodes_for() comment already notes the
  // corner/side ordering agrees with KARMA's MOAB/Exodus convention).
  switch (family) {
    case CellFamily::Edge:
      if (edge >= 1) break;
      return {0, 1};
    case CellFamily::Triangle: {
      constexpr std::array<std::array<std::size_t, 2>, 3> edges{{{0, 1}, {1, 2}, {2, 0}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
    case CellFamily::Quadrilateral: {
      constexpr std::array<std::array<std::size_t, 2>, 4> edges{{{0, 1}, {1, 2}, {2, 3}, {3, 0}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
    case CellFamily::Tetrahedron: {
      constexpr std::array<std::array<std::size_t, 2>, 6> edges{
          {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 3}, {2, 3}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
    case CellFamily::Hexahedron: {
      constexpr std::array<std::array<std::size_t, 2>, 12> edges{{{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 5},
                                                                     {2, 6}, {3, 7}, {4, 5}, {5, 6}, {6, 7}, {7, 4}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
    case CellFamily::Prism: {
      constexpr std::array<std::array<std::size_t, 2>, 9> edges{
          {{0, 1}, {1, 2}, {2, 0}, {0, 3}, {1, 4}, {2, 5}, {3, 4}, {4, 5}, {5, 3}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
    case CellFamily::Pyramid: {
      constexpr std::array<std::array<std::size_t, 2>, 8> edges{
          {{0, 1}, {1, 2}, {2, 3}, {3, 0}, {0, 4}, {1, 4}, {2, 4}, {3, 4}}};
      if (edge >= edges.size()) break;
      return edges[edge];
    }
  }
  refuse("edge index is out of range");
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
    case CellFamily::Prism:
      return p(1.0 / 3.0, 1.0 / 3.0, .5);
    case CellFamily::Pyramid:
      return p(0, 0, .25);
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
  if (family() == CellFamily::Pyramid && point == Point(0, 0, 1)) {
    // The apex: shape VALUES have a well-defined limit (only the apex
    // node's own function is nonzero there, by construction of the
    // candidate set at w == 0), ported from KARMA's identical carve-out.
    std::fill(values.begin(), values.end(), 0.0);
    values[4] = 1.0;
    return;
  }
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
  if (family() == CellFamily::Pyramid && point == Point(0, 0, 1))
    refuse("pyramid basis derivatives are path-dependent at the exact apex");
  std::fill(output_values.begin(), output_values.end(), 0.0);
  std::fill(output_derivatives.begin(), output_derivatives.end(), Point::zero());
  const unsigned ord = order();

  if (family() == CellFamily::Pyramid) {
    pyramid_values(variant_, point, nodes_, output_values, output_derivatives);
    return;
  }

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

  if (variant_ == GeometryVariant::Quad8) {
    evaluate_quad8(point, output_values, output_derivatives);
    return;
  }

  if (family() == CellFamily::Quadrilateral || family() == CellFamily::Hexahedron) {
    // Tensor product of 1-D line_basis along each axis a reference node's
    // coordinate selects (0, 1, or 1/2).
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
    return;
  }

  // Wedges (Prism family): Prism18 is Tri6 x Edge3 (full tensor); Prism15 is
  // the standard edge-serendipity wedge; Prism6 is the trilinear wedge.
  // Ported unchanged from karma_reference_topology.C's own wedge branch.
  const double lambda[] = {1 - point(0) - point(1), point(0), point(1)};
  const Point dl[] = {p(-1, -1, 0), p(1, 0, 0), p(0, 1, 0)};
  const double tau = 2 * point(2) - 1;
  if (variant_ == GeometryVariant::Prism6) {
    for (unsigned layer = 0; layer < 2; ++layer)
      for (unsigned i = 0; i < 3; ++i) {
        const double z_value = layer == 0 ? 1 - point(2) : point(2);
        const double z_derivative = layer == 0 ? -1 : 1;
        const std::size_t n = 3 * layer + i;
        output_values[n] = lambda[i] * z_value;
        output_derivatives[n] = z_value * dl[i];
        output_derivatives[n](2) = lambda[i] * z_derivative;
      }
    return;
  }
  if (variant_ == GeometryVariant::Prism15) {
    for (unsigned layer = 0; layer < 2; ++layer)
      for (unsigned i = 0; i < 3; ++i) {
        const double sign = layer == 0 ? -1.0 : 1.0;
        const std::size_t n = 3 * layer + i;
        output_values[n] = .5 * lambda[i] * (1 + sign * tau) * (2 * lambda[i] - 2 + sign * tau);
        output_derivatives[n] = .5 * (1 + sign * tau) * (4 * lambda[i] - 2 + sign * tau) * dl[i];
        output_derivatives[n](2) = sign * lambda[i] * (2 * lambda[i] - 1 + 2 * sign * tau);
      }
    const unsigned pairs[][2] = {{0, 1}, {1, 2}, {2, 0}};
    for (unsigned layer = 0; layer < 2; ++layer)
      for (unsigned e = 0; e < 3; ++e) {
        const double sign = layer == 0 ? -1.0 : 1.0;
        const auto i = pairs[e][0], j = pairs[e][1];
        const std::size_t n = (layer == 0 ? 6 : 12) + e;
        output_values[n] = 2 * lambda[i] * lambda[j] * (1 + sign * tau);
        output_derivatives[n] = 2 * (1 + sign * tau) * (lambda[j] * dl[i] + lambda[i] * dl[j]);
        output_derivatives[n](2) = 4 * sign * lambda[i] * lambda[j];
      }
    for (unsigned i = 0; i < 3; ++i) {
      const std::size_t n = 9 + i;
      output_values[n] = lambda[i] * (1 - tau * tau);
      output_derivatives[n] = (1 - tau * tau) * dl[i];
      output_derivatives[n](2) = -4 * tau * lambda[i];
    }
    return;
  }
  // Prism18.
  const auto bz = line_basis(point(2), ord);
  const double tri_values[] = {lambda[0] * (2 * lambda[0] - 1), lambda[1] * (2 * lambda[1] - 1),
                                lambda[2] * (2 * lambda[2] - 1), 4 * lambda[0] * lambda[1], 4 * lambda[1] * lambda[2],
                                4 * lambda[2] * lambda[0]};
  const Point tri_derivatives[] = {(4 * lambda[0] - 1) * dl[0],
                                    (4 * lambda[1] - 1) * dl[1],
                                    (4 * lambda[2] - 1) * dl[2],
                                    4 * (lambda[1] * dl[0] + lambda[0] * dl[1]),
                                    4 * (lambda[2] * dl[1] + lambda[1] * dl[2]),
                                    4 * (lambda[0] * dl[2] + lambda[2] * dl[0])};
  const unsigned tri_index[] = {0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5};
  const unsigned z_index[] = {0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2};
  for (std::size_t n = 0; n < node_count(); ++n) {
    const auto i = tri_index[n], k = z_index[n];
    output_values[n] = tri_values[i] * bz[k].first;
    output_derivatives[n] = bz[k].first * tri_derivatives[i];
    output_derivatives[n](2) = tri_values[i] * bz[k].second;
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
  const bool quadratic = order() == 2;
  switch (family()) {
    case CellFamily::Triangle:
    case CellFamily::Quadrilateral:
      return ReferenceTopology(quadratic ? GeometryVariant::Edge3 : GeometryVariant::Edge2);
    case CellFamily::Tetrahedron:
      return ReferenceTopology(quadratic ? GeometryVariant::Tri6 : GeometryVariant::Tri3);
    case CellFamily::Hexahedron:
      return ReferenceTopology(quadratic ? GeometryVariant::Quad9 : GeometryVariant::Quad4);
    case CellFamily::Prism: {
      const bool quad_side = side < 3;  // sides 0-2 are the three quad faces, 3-4 the two tri caps
      if (quad_side)
        return ReferenceTopology(!quadratic ? GeometryVariant::Quad4
                                  : variant_ == GeometryVariant::Prism15 ? GeometryVariant::Quad8
                                                                         : GeometryVariant::Quad9);
      return ReferenceTopology(quadratic ? GeometryVariant::Tri6 : GeometryVariant::Tri3);
    }
    case CellFamily::Pyramid: {
      const bool quad_side = side == 4;  // sides 0-3 are the four tri faces, 4 the one quad base
      if (quad_side)
        return ReferenceTopology(!quadratic ? GeometryVariant::Quad4
                                  : variant_ == GeometryVariant::Pyramid13 ? GeometryVariant::Quad8
                                                                           : GeometryVariant::Quad9);
      return ReferenceTopology(quadratic ? GeometryVariant::Tri6 : GeometryVariant::Tri3);
    }
    case CellFamily::Edge:
      refuse("an edge's sides are vertices, which have no ReferenceTopology of their own");
  }
  refuse("unknown cell family");
}

Point ReferenceTopology::side_reference_map(unsigned side, const Point& point) const {
  if (side >= side_count()) refuse("side index is out of range");
  const auto side_node_indices = side_nodes(side);
  const unsigned corners = linear_corner_count_for_shape(side_topology(side).family());
  const GeometryVariant linear = corners == 2   ? GeometryVariant::Edge2
                                  : corners == 3 ? GeometryVariant::Tri3
                                                 : GeometryVariant::Quad4;
  const auto values = ReferenceTopology(linear).evaluate(point);
  Point result = Point::zero();
  for (unsigned i = 0; i < corners; ++i) result += values.values[i] * nodes_[side_node_indices[i]];
  return result;
}

Matrix3 ReferenceTopology::side_reference_jacobian(unsigned side, const Point& point) const {
  if (side >= side_count()) refuse("side index is out of range");
  const auto side_node_indices = side_nodes(side);
  const unsigned corners = linear_corner_count_for_shape(side_topology(side).family());
  const GeometryVariant linear = corners == 2   ? GeometryVariant::Edge2
                                  : corners == 3 ? GeometryVariant::Tri3
                                                 : GeometryVariant::Quad4;
  const auto values = ReferenceTopology(linear).evaluate(point);
  Matrix3 result = Matrix3::zero();
  for (unsigned i = 0; i < corners; ++i)
    for (unsigned d = 0; d < dimension() - 1; ++d)
      result.set_col(d, result.col(d) + values.derivatives[i](d) * nodes_[side_node_indices[i]]);
  return result;
}

}  // namespace fuse::physics::fe
