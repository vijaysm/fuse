// fuse_physics::fe::Quadrature -- native Gauss-Legendre/Gauss-Jacobi
// quadrature via the Golub-Welsch eigenvalue algorithm, and collapsed-
// coordinate (Duffy) conical products for the triangle and tetrahedron.
// See quadrature.hpp's header comment for the literature sources.
//
// RECURRENCE COEFFICIENTS. For the Jacobi weight w(x) = (1-x)^alpha on
// [-1,1] (beta = 0, the only case this file needs), the monic orthogonal
// polynomials' three-term recurrence coefficients are (Abramowitz, M. and
// Stegun, I.A., "Handbook of Mathematical Functions", 1964, 22.7.1; Gautschi,
// W., "Orthogonal Polynomials: Computation and Approximation", 2004, table
// 1.1, specialised to beta = 0):
//
//   a_0 = -alpha / (alpha + 2)
//   a_n = -alpha^2 / ((2n+alpha)(2n+alpha+2)),                       n >= 1
//   b_0 = 2^(alpha+1) / (alpha + 1)     (the zeroth moment, i.e. total mass)
//   b_n = 4 n^2 (n+alpha)^2 / ((2n+alpha)^2 ((2n+alpha)^2 - 1)),     n >= 1
//
// alpha = 0 reduces to the classical Gauss-Legendre recurrence a_n = 0,
// b_n = n^2/(4n^2-1) -- checked by this file's own GaussLegendreMatchesClassicalRecurrence
// unit test. The quadrature nodes are the eigenvalues of the symmetric
// tridiagonal Jacobi matrix (diagonal a_n, off-diagonal sqrt(b_n)); the
// weights are b_0 times the squared first component of each normalised
// eigenvector (Golub-Welsch 1969, theorem stated in the paper's section 2).

#include "fuse_physics/fe/quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace fuse::physics::fe {

namespace {

[[noreturn]] void refuse(const char* message) { throw std::invalid_argument(message); }

/// Eigenvalues (ascending) and first-row eigenvector components of a real
/// symmetric tridiagonal matrix with diagonal `diag` and off-diagonal
/// `offdiag` (size n-1), via the classical cyclic Jacobi eigenvalue
/// algorithm (Jacobi, C.G.J., 1846; see e.g. Golub, G.H. and Van Loan, C.F.,
/// "Matrix Computations", 4th ed., 2013, section 8.4). Operates on the
/// full dense symmetric matrix (n is a handful of quadrature points per
/// axis, so O(n^3) per sweep is immaterial); eigenvectors are accumulated
/// from the identity so the first row is available for the Golub-Welsch
/// weight formula.
struct EigenDecomposition {
  std::vector<double> eigenvalues;
  std::vector<double> first_row;  // first_row[k] = (eigenvector k)[0]
  int sweeps = 0;                 // Jacobi sweeps performed before the stopping test held
};

EigenDecomposition symmetric_tridiagonal_eigen(const std::vector<double>& diag,
                                                const std::vector<double>& offdiag) {
  const std::size_t n = diag.size();
  std::vector<std::vector<double>> m(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) m[i][i] = diag[i];
  for (std::size_t i = 0; i + 1 < n; ++i) {
    m[i][i + 1] = offdiag[i];
    m[i + 1][i] = offdiag[i];
  }
  std::vector<std::vector<double>> v(n, std::vector<double>(n, 0.0));
  for (std::size_t i = 0; i < n; ++i) v[i][i] = 1.0;

  constexpr int max_sweeps = 100;
  int sweeps_done = 0;
  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double off_norm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = i + 1; j < n; ++j) off_norm += m[i][j] * m[i][j];
    if (off_norm <= std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon() * n * n) break;
    ++sweeps_done;

    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = p + 1; q < n; ++q) {
        if (std::abs(m[p][q]) <= std::numeric_limits<double>::epsilon() * (std::abs(m[p][p]) + std::abs(m[q][q])))
          continue;
        const double theta = (m[q][q] - m[p][p]) / (2.0 * m[p][q]);
        const double sign = theta >= 0.0 ? 1.0 : -1.0;
        const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0);
        const double s = t * c;
        const double mpp = m[p][p], mqq = m[q][q], mpq = m[p][q];
        m[p][p] = mpp - t * mpq;
        m[q][q] = mqq + t * mpq;
        m[p][q] = 0.0;
        m[q][p] = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
          if (i == p || i == q) continue;
          const double mip = m[i][p], miq = m[i][q];
          m[i][p] = c * mip - s * miq;
          m[p][i] = m[i][p];
          m[i][q] = s * mip + c * miq;
          m[q][i] = m[i][q];
        }
        for (std::size_t i = 0; i < n; ++i) {
          const double vip = v[i][p], viq = v[i][q];
          v[i][p] = c * vip - s * viq;
          v[i][q] = s * vip + c * viq;
        }
      }
  }

  EigenDecomposition result;
  result.eigenvalues.resize(n);
  result.first_row.resize(n);
  for (std::size_t i = 0; i < n; ++i) result.eigenvalues[i] = m[i][i];
  std::vector<std::size_t> order(n);
  for (std::size_t i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) { return result.eigenvalues[a] < result.eigenvalues[b]; });
  EigenDecomposition sorted;
  sorted.sweeps = sweeps_done;
  sorted.eigenvalues.resize(n);
  sorted.first_row.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    sorted.eigenvalues[i] = result.eigenvalues[order[i]];
    sorted.first_row[i] = v[0][order[i]];
  }
  return sorted;
}

/// Gauss-Jacobi nodes and weights for weight (1-x)^alpha on [-1,1], via the
/// recurrence coefficients in this file's header comment and the Golub-
/// Welsch eigenvalue construction. `alpha` is a non-negative integer in
/// this file's use (0 for plain Gauss-Legendre, 1 and 2 for the triangle's
/// and tetrahedron's collapsed axes).
struct Rule1D {
  std::vector<double> nodes;    // on [-1,1]
  std::vector<double> weights;  // sum to the weight function's total mass
  int sweeps = 0;               // from the Golub-Welsch eigen-solve (see Quadrature::round_off_bound)
};

Rule1D gauss_jacobi(unsigned n_points, double alpha) {
  if (n_points == 0) refuse("a quadrature rule needs at least one point");
  std::vector<double> diag(n_points), offdiag(n_points > 0 ? n_points - 1 : 0);
  diag[0] = -alpha / (alpha + 2.0);
  for (std::size_t n = 1; n < n_points; ++n) {
    const double two_n_alpha = 2.0 * static_cast<double>(n) + alpha;
    diag[n] = -(alpha * alpha) / (two_n_alpha * (two_n_alpha + 2.0));
  }
  for (std::size_t n = 1; n < n_points; ++n) {
    const double two_n_alpha = 2.0 * static_cast<double>(n) + alpha;
    const double b_n = 4.0 * static_cast<double>(n) * static_cast<double>(n) * (static_cast<double>(n) + alpha) *
                        (static_cast<double>(n) + alpha) / (two_n_alpha * two_n_alpha * (two_n_alpha * two_n_alpha - 1.0));
    offdiag[n - 1] = std::sqrt(b_n);
  }
  const double b0 = std::pow(2.0, alpha + 1.0) / (alpha + 1.0);

  const auto decomposition = symmetric_tridiagonal_eigen(diag, offdiag);
  Rule1D rule;
  rule.sweeps = decomposition.sweeps;
  rule.nodes = decomposition.eigenvalues;
  rule.weights.resize(n_points);
  for (std::size_t i = 0; i < n_points; ++i) rule.weights[i] = b0 * decomposition.first_row[i] * decomposition.first_row[i];
  return rule;
}

/// Points per axis for exactness to `degree`: an n-point Gauss rule is
/// exact to polynomial degree 2n-1, so n = ceil((degree+1)/2) suffices; the
/// "+2 then halve" form matches KARMA's own points_per_axis formula
/// (karma_quadrature.h) so the two agree on every degree.
unsigned points_per_axis(int degree) { return static_cast<unsigned>(std::max(1, (degree + 2) / 2)); }

/// Gauss-Legendre on [0,1] (alpha = 0, rescaled from [-1,1]).
Rule1D gauss_legendre_01(unsigned n_points) {
  Rule1D rule = gauss_jacobi(n_points, 0.0);
  for (auto& x : rule.nodes) x = 0.5 * (x + 1.0);
  for (auto& w : rule.weights) w *= 0.5;
  return rule;
}

/// Gauss-Jacobi(alpha) on [0,1] (weight (1-x)^alpha), rescaled from [-1,1]:
/// x' = (x+1)/2 in [0,1], (1-x) = 2(1-x'), so the weight picks up a factor
/// 2^alpha on top of the usual 1/2 Jacobian -- folded into the returned
/// weights so callers need not know the rescaling happened.
Rule1D gauss_jacobi_01(unsigned n_points, double alpha) {
  Rule1D rule = gauss_jacobi(n_points, alpha);
  for (auto& x : rule.nodes) x = 0.5 * (x + 1.0);
  const double scale = std::pow(2.0, alpha + 1.0);
  for (auto& w : rule.weights) w /= scale;
  return rule;
}

}  // namespace

Quadrature::Quadrature(CellFamily family, int degree) : family_(family), degree_(degree) {
  if (degree < 0 || degree > 64) refuse("quadrature degree must be in [0, 64]");
  const unsigned n = points_per_axis(degree);

  if (family == CellFamily::Edge) {
    const auto rule = gauss_legendre_01(n);
    sweeps_ = rule.sweeps;
    for (std::size_t i = 0; i < n; ++i) {
      points_.push_back(Point(rule.nodes[i]));
      weights_.push_back(rule.weights[i]);
    }
    return;
  }
  if (family == CellFamily::Quadrilateral) {
    const auto rx = gauss_legendre_01(n), ry = gauss_legendre_01(n);
    sweeps_ = std::max(rx.sweeps, ry.sweeps);
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i) {
        points_.push_back(Point(rx.nodes[i], ry.nodes[j]));
        weights_.push_back(rx.weights[i] * ry.weights[j]);
      }
    return;
  }
  if (family == CellFamily::Hexahedron) {
    const auto rx = gauss_legendre_01(n), ry = gauss_legendre_01(n), rz = gauss_legendre_01(n);
    sweeps_ = std::max({rx.sweeps, ry.sweeps, rz.sweeps});
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) {
          points_.push_back(Point(rx.nodes[i], ry.nodes[j], rz.nodes[k]));
          weights_.push_back(rx.weights[i] * ry.weights[j] * rz.weights[k]);
        }
    return;
  }
  if (family == CellFamily::Triangle) {
    // Collapsed coordinates (a,b) in [0,1]^2 -> (x,y) = (a(1-b), b),
    // Jacobian (1-b). "a" needs no weight (Gauss-Legendre); "b" carries the
    // Jacobian via Gauss-Jacobi(alpha=1).
    const auto ra = gauss_legendre_01(n), rb = gauss_jacobi_01(n, 1.0);
    sweeps_ = std::max(ra.sweeps, rb.sweeps);
    for (std::size_t j = 0; j < n; ++j)
      for (std::size_t i = 0; i < n; ++i) {
        const double a = ra.nodes[i], b = rb.nodes[j];
        points_.push_back(Point(a * (1.0 - b), b));
        weights_.push_back(ra.weights[i] * rb.weights[j]);
      }
    return;
  }
  if (family == CellFamily::Tetrahedron) {
    // Double collapse (a,b,c) in [0,1]^3 -> (x,y,z) = (a(1-b)(1-c),
    // b(1-c), c), Jacobian (1-b)(1-c)^2: "a" plain, "b" Gauss-Jacobi(1),
    // "c" Gauss-Jacobi(2) -- the same alpha-absorbs-the-Jacobian-power
    // technique karma_quadrature.h's own comment derives for the
    // pyramid's single collapsed axis, applied here to two.
    const auto ra = gauss_legendre_01(n), rb = gauss_jacobi_01(n, 1.0), rc = gauss_jacobi_01(n, 2.0);
    sweeps_ = std::max({ra.sweeps, rb.sweeps, rc.sweeps});
    for (std::size_t k = 0; k < n; ++k)
      for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = 0; i < n; ++i) {
          const double a = ra.nodes[i], b = rb.nodes[j], c = rc.nodes[k];
          points_.push_back(Point(a * (1.0 - b) * (1.0 - c), b * (1.0 - c), c));
          weights_.push_back(ra.weights[i] * rb.weights[j] * rc.weights[k]);
        }
    return;
  }
  refuse("unsupported quadrature cell family");
}

double Quadrature::round_off_bound(double sum_abs_terms, unsigned polynomial_degree,
                                   double cell_measure) const noexcept {
  // See the header for the derivation. n1d is the largest 1-D rule size this
  // rule was assembled from (points_per_axis), the dimension of the
  // eigenproblem whose stopping test and rotations bound the node and weight
  // errors; the per-term operation count is 2*dim (Duffy map and weight
  // products) + polynomial_degree (monomial powers) + 1 (the product) + 1
  // (the accumulation).
  const double eps = std::numeric_limits<double>::epsilon();
  const double n1d = static_cast<double>(points_per_axis(degree_));
  const double dim = static_cast<double>(dimension());
  const double ops = 2.0 * dim + static_cast<double>(polynomial_degree) + 2.0;
  const double arithmetic = (static_cast<double>(n_points()) + ops) * sum_abs_terms;
  const double solver = (static_cast<double>(sweeps_) + 1.0) * n1d *
                        (dim * sum_abs_terms + static_cast<double>(polynomial_degree) * cell_measure);
  return eps * (arithmetic + solver);
}

}  // namespace fuse::physics::fe
