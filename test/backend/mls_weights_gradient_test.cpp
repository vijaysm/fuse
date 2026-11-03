// Shape gradients of the moving-least-squares weights: fuse::device::solve_mls_on_space<Space, T>
// (include/fuse/backend/mls_weights_device.hpp) with T = fuse::num::dual<S, N>
// returns, for each weight, its gradient with respect to the coordinates of the
// source and target points. The stage's gate is a Taylor-remainder test of
// order 2 in the coordinates (docs/dev/derivative-gate.md,
// fuse::linalg::taylor_remainder_order); the rest of this file is what makes that
// number mean something:
//
//   1. the value part of a dual-number run equals the double run BITWISE, so the
//      gradient is the gradient of the weights the library already produces;
//   2. identities that hold exactly and are differentiated by hand: the weights
//      sum to 1 (gradient 0 in every direction), and the weights reproduce every
//      polynomial of the fit degree, sum_j w_j q(x_j) = q(x_t), so the gradient of
//      the left side (product rule over the weights AND the source coordinates)
//      equals the gradient of q at the target;
//   3. the gradient equals a fourth-order central difference of the double run,
//      parameter by parameter (a check on each partial, where the Taylor test
//      checks one direction);
//   4. the Taylor remainder of the weights as a function of ALL coordinates has
//      order 2 along a fixed direction, and a gradient that leaves out the motion
//      of the target has order 1 (the test of the test);
//   5. a symmetric stencil, whose elimination multipliers are zero in value and not
//      in gradient, and a neighbour that coincides with the target, which is a
//      branch point of the distance;
//   6. a single-precision gradient through the same kernel.
//
// Every case runs on Kokkos::HostSpace and, in a build with a device backend, on
// the device memory space (typed test).

#include "fuse/backend/mls_weights_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/mls_weights.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/num/dual.hpp"

namespace {

using fuse::MlsRadial;

using D1 = fuse::num::dual<double, 1>;
using D27 = fuse::num::dual<double, 27>;  // up to 27 coordinates: 3-D, a stencil of 8 and one target
using F1 = fuse::num::dual<float, 1>;

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif
using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

constexpr double kEps = std::numeric_limits<double>::epsilon();

// ---------------------------------------------------------------------------
// Constants of the tolerances, each MEASURED (docs/dev/mls-shape-gradient.md
// records the measurement and the headroom).
// ---------------------------------------------------------------------------

// Identity defects (partition of unity, polynomial reproduction) are bounded by
// c * eps * (the sum of the absolute values of the terms that were added), the
// summation-error form of fuse/linalg/derivative_gate.hpp.
constexpr double kIdentityConstant = 1.0e4;
// Agreement with the fourth-order central difference of the double run:
// |gradient - difference| <= kDifferenceTolerance * (1 + |gradient|).
constexpr double kDifferenceTolerance = 1.0e-6;
// A single-precision gradient against the double one:
// |single - double| <= kSingleTolerance * (1 + |double|).
constexpr double kSingleTolerance = 1.0e-5;

// ---------------------------------------------------------------------------
// Point clouds. The standard random distributions differ between libraries, so
// the clouds use a small linear congruential generator and are identical on every
// platform (Knuth, TAOCP vol. 2, 3.2.1, the 64-bit multiplier of MMIX).
// ---------------------------------------------------------------------------

struct Lcg {
  std::uint64_t s;
  double next() {  // uniform in [0, 1)
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) * (1.0 / 9007199254740992.0);
  }
  double symmetric() { return 2.0 * next() - 1.0; }  // uniform in [-1, 1)
};

struct Cloud {
  int dim = 2;
  int degree = 1;
  int k = 6;
  MlsRadial radial = MlsRadial::Wendland2;
  std::vector<double> src;  // n_src x 3, row-major
  std::vector<double> tgt;  // n_tgt x 3, row-major
  std::vector<int> nbr;     // n_tgt x k source indices
  [[nodiscard]] std::size_t n_src() const { return src.size() / 3; }
  [[nodiscard]] std::size_t n_tgt() const { return tgt.size() / 3; }
  /// Number of coordinates that carry a parameter: `dim` per point.
  [[nodiscard]] std::size_t n_params() const {
    return static_cast<std::size_t>(dim) * (n_src() + n_tgt());
  }
};

double distance(const Cloud& c, std::size_t t, std::size_t s) {
  double d2 = 0.0;
  for (int a = 0; a < 3; ++a) {
    const double e = c.src[3 * s + a] - c.tgt[3 * t + a];
    d2 += e * e;
  }
  return std::sqrt(d2);
}

/// The k nearest sources of every target, nearest first, ties broken by index.
void fill_neighbours(Cloud& c) {
  c.nbr.clear();
  for (std::size_t t = 0; t < c.n_tgt(); ++t) {
    std::vector<std::pair<double, int>> order;
    for (std::size_t s = 0; s < c.n_src(); ++s) order.emplace_back(distance(c, t, s), int(s));
    std::sort(order.begin(), order.end());
    for (int j = 0; j < c.k; ++j) c.nbr.push_back(order[static_cast<std::size_t>(j)].second);
  }
}

/// A jittered lattice of side `n` in `dim` dimensions (spacing 1, jitter +-0.3) and
/// `n_tgt` targets inside it.
Cloud grid_cloud(int dim, int degree, int k, MlsRadial radial, int n, std::size_t n_tgt,
                 std::uint64_t seed) {
  Cloud c;
  c.dim = dim;
  c.degree = degree;
  c.k = k;
  c.radial = radial;
  Lcg rng{seed};
  const int nz = dim == 3 ? n : 1;
  const int ny = dim >= 2 ? n : 1;
  for (int iz = 0; iz < nz; ++iz) {
    for (int iy = 0; iy < ny; ++iy) {
      for (int ix = 0; ix < n; ++ix) {
        c.src.push_back(ix + 0.3 * rng.symmetric());
        c.src.push_back(dim >= 2 ? iy + 0.3 * rng.symmetric() : 0.0);
        c.src.push_back(dim == 3 ? iz + 0.3 * rng.symmetric() : 0.0);
      }
    }
  }
  const double lo = 0.35 * (n - 1);
  const double span = 0.3 * (n - 1);
  for (std::size_t t = 0; t < n_tgt; ++t) {
    c.tgt.push_back(lo + span * rng.next());
    c.tgt.push_back(dim >= 2 ? lo + span * rng.next() : 0.0);
    c.tgt.push_back(dim == 3 ? lo + span * rng.next() : 0.0);
  }
  fill_neighbours(c);
  return c;
}

/// The stencil of target 0 of `c` as a cloud of its own: only the k neighbours as
/// sources, in the order of `c.nbr`, with the single target. Every coordinate of it
/// then fits in one `D27` (dim * (k + 1) <= 27).
Cloud local_cloud(const Cloud& c) {
  Cloud l;
  l.dim = c.dim;
  l.degree = c.degree;
  l.k = c.k;
  l.radial = c.radial;
  for (int j = 0; j < c.k; ++j) {
    for (int a = 0; a < 3; ++a) l.src.push_back(c.src[3 * std::size_t(c.nbr[std::size_t(j)]) + a]);
    l.nbr.push_back(j);
  }
  for (int a = 0; a < 3; ++a) l.tgt.push_back(c.tgt[std::size_t(a)]);
  return l;
}

/// The gap between the largest and the second largest neighbour distance of every
/// target: the support radius is 1.1 times the largest distance, so a coordinate
/// motion that closes this gap swaps the farthest neighbour and puts a kink in the
/// weights (mls_weights_device.hpp, "WHAT IS DIFFERENTIATED").
double smallest_radius_gap(const Cloud& c) {
  double gap = std::numeric_limits<double>::infinity();
  for (std::size_t t = 0; t < c.n_tgt(); ++t) {
    const double last = distance(c, t, std::size_t(c.nbr[t * std::size_t(c.k) + std::size_t(c.k) - 1]));
    const double prev = distance(c, t, std::size_t(c.nbr[t * std::size_t(c.k) + std::size_t(c.k) - 2]));
    gap = std::min(gap, last - prev);
  }
  return gap;
}

// ---------------------------------------------------------------------------
// Running the kernel on `Space` from host vectors.
// ---------------------------------------------------------------------------

template <class T>
struct Solved {
  std::vector<T> val;    // n_tgt * k weights
  std::vector<int> col;  // n_tgt * k source indices
  int n_fail = 0;
};

/// `src` and `tgt` hold 3 numbers per point (x, y, z), of the scalar type `T`.
template <class Space, class T>
Solved<T> solve(const Cloud& c, const std::vector<T>& src, const std::vector<T>& tgt) {
  Kokkos::View<T * [3], Space> src_v("mls_grad_src", c.n_src());
  Kokkos::View<T * [3], Space> tgt_v("mls_grad_tgt", c.n_tgt());
  Kokkos::View<int*, Space> nbr_v("mls_grad_nbr", c.nbr.size());
  {
    auto s = Kokkos::create_mirror_view(src_v);
    for (std::size_t i = 0; i < c.n_src(); ++i) {
      for (std::size_t a = 0; a < 3; ++a) s(i, a) = src[3 * i + a];
    }
    Kokkos::deep_copy(src_v, s);
    auto t = Kokkos::create_mirror_view(tgt_v);
    for (std::size_t i = 0; i < c.n_tgt(); ++i) {
      for (std::size_t a = 0; a < 3; ++a) t(i, a) = tgt[3 * i + a];
    }
    Kokkos::deep_copy(tgt_v, t);
    auto n = Kokkos::create_mirror_view(nbr_v);
    for (std::size_t i = 0; i < c.nbr.size(); ++i) n(i) = c.nbr[i];
    Kokkos::deep_copy(nbr_v, n);
  }
  auto res = fuse::device::solve_mls_on_space<Space>(c.dim, c.degree, c.k, c.radial, src_v,
                                                      tgt_v, nbr_v);
  Solved<T> out;
  out.n_fail = res.n_fail;
  auto val_h = Kokkos::create_mirror_view(res.val);
  auto col_h = Kokkos::create_mirror_view(res.col);
  Kokkos::deep_copy(val_h, res.val);
  Kokkos::deep_copy(col_h, res.col);
  out.val.assign(val_h.data(), val_h.data() + val_h.extent(0));
  out.col.assign(col_h.data(), col_h.data() + col_h.extent(0));
  return out;
}

/// The coordinates of `c` seeded as parameters: coordinate `a < dim` of source `s`
/// is parameter `dim * s + a`, of target `t` parameter `dim * (n_src + t) + a`; the
/// unused coordinates are constants. Parameters beyond `dim * (n_src + n_tgt)` stay
/// zero in every gradient.
std::vector<D27> seeded(const Cloud& c, const std::vector<double>& pts, std::size_t first_point) {
  std::vector<D27> out(pts.size());
  for (std::size_t i = 0; i < pts.size() / 3; ++i) {
    for (std::size_t a = 0; a < 3; ++a) {
      const double x = pts[3 * i + a];
      out[3 * i + a] =
          a < std::size_t(c.dim)
              ? D27::variable(x, static_cast<int>(std::size_t(c.dim) * (first_point + i) + a))
              : D27(x);
    }
  }
  return out;
}

bool bitwise_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

// ---------------------------------------------------------------------------
// Polynomials of the fit degree, evaluated in the scalar type `T` so that the
// gradient of q(x_j) with respect to the coordinates of x_j comes with it.
// ---------------------------------------------------------------------------

struct Term {
  int a, b, c;
  double coef;
};

std::vector<Term> polynomial_terms(int dim, int degree) {
  std::vector<Term> terms;
  int index = 0;
  for (int total = 0; total <= degree; ++total) {
    for (int a = total; a >= 0; --a) {
      for (int b = total - a; b >= 0; --b) {
        const int cc = total - a - b;
        if (dim < 2 && b > 0) continue;
        if (dim < 3 && cc > 0) continue;
        terms.push_back(Term{a, b, cc, 1.0 + 0.37 * index});
        ++index;
      }
    }
  }
  return terms;
}

template <class T>
T evaluate(const std::vector<Term>& terms, const T& x, const T& y, const T& z) {
  T sum(0.0);
  for (const Term& t : terms) {
    T m(t.coef);
    for (int i = 0; i < t.a; ++i) m *= x;
    for (int i = 0; i < t.b; ++i) m *= y;
    for (int i = 0; i < t.c; ++i) m *= z;
    sum += m;
  }
  return sum;
}

// ---------------------------------------------------------------------------
// The typed fixture: every case on the host space and on the device space.
// ---------------------------------------------------------------------------

using Spaces = std::conditional_t<kHasDeviceBackend,
                                  ::testing::Types<Kokkos::HostSpace, DeviceMemSpace>,
                                  ::testing::Types<Kokkos::HostSpace>>;

template <class Space>
class MlsShapeGradient : public ::testing::Test {};
TYPED_TEST_SUITE(MlsShapeGradient, Spaces);

/// Identities 1-3 above on one stencil (the cloud must have a single target).
template <class Space>
void check_stencil(const Cloud& c, const std::string& label) {
  ASSERT_EQ(c.n_tgt(), 1U);
  const std::size_t n_params = c.n_params();
  ASSERT_LE(n_params, 27U) << label;
  const std::vector<D27> src_d = seeded(c, c.src, 0);
  const std::vector<D27> tgt_d = seeded(c, c.tgt, c.n_src());
  const Solved<D27> g = solve<Space, D27>(c, src_d, tgt_d);
  const Solved<double> w = solve<Space, double>(c, c.src, c.tgt);
  ASSERT_EQ(g.n_fail, 0) << label;
  ASSERT_EQ(w.n_fail, 0) << label;
  ASSERT_EQ(g.val.size(), std::size_t(c.k));

  // 1. The value part of the dual run is the double run, bit for bit.
  for (std::size_t j = 0; j < g.val.size(); ++j) {
    EXPECT_TRUE(bitwise_equal(g.val[j].value(), w.val[j]))
        << label << " weight " << j << ": " << g.val[j].value() << " vs " << w.val[j];
  }

  // 2a. Partition of unity: the weights sum to 1 and the sum has gradient 0.
  D27 sum(0.0);
  double abs_sum = 0.0;
  std::vector<double> abs_grad(n_params, 0.0);
  for (const D27& v : g.val) {
    sum += v;
    abs_sum += std::fabs(v.value());
    for (std::size_t p = 0; p < n_params; ++p) abs_grad[p] += std::fabs(v.gradient(int(p)));
  }
  EXPECT_LE(std::fabs(sum.value() - 1.0), kIdentityConstant * kEps * abs_sum) << label;
  double worst_unity = 0.0;
  for (std::size_t p = 0; p < n_params; ++p) {
    const double ratio = std::fabs(sum.gradient(int(p))) / (kEps * (abs_grad[p] + 1e-300));
    worst_unity = std::max(worst_unity, ratio);
    EXPECT_LE(std::fabs(sum.gradient(int(p))), kIdentityConstant * kEps * abs_grad[p])
        << label << " parameter " << p;
  }

  // 2b. Polynomial reproduction, differentiated: F = sum_j w_j q(x_j), q_t = q(x_t),
  // F = q_t in value and in every gradient entry.
  const std::vector<Term> terms = polynomial_terms(c.dim, c.degree);
  const D27 q_t = evaluate<D27>(terms, tgt_d[0], tgt_d[1], tgt_d[2]);
  D27 f(0.0);
  std::vector<double> scale(n_params + 1, 0.0);  // sum of |terms| for value (slot n) and gradients
  for (std::size_t j = 0; j < g.val.size(); ++j) {
    const std::size_t s = std::size_t(c.nbr[j]);
    const D27 q_j = evaluate<D27>(terms, src_d[3 * s], src_d[3 * s + 1], src_d[3 * s + 2]);
    f += g.val[j] * q_j;
    scale[n_params] += std::fabs(g.val[j].value() * q_j.value());
    for (std::size_t p = 0; p < n_params; ++p) {
      scale[p] += std::fabs(g.val[j].gradient(int(p)) * q_j.value()) +
                  std::fabs(g.val[j].value() * q_j.gradient(int(p)));
    }
  }
  EXPECT_LE(std::fabs(f.value() - q_t.value()), kIdentityConstant * kEps * scale[n_params]) << label;
  double worst_poly = std::fabs(f.value() - q_t.value()) / (kEps * scale[n_params]);
  for (std::size_t p = 0; p < n_params; ++p) {
    const double defect = std::fabs(f.gradient(int(p)) - q_t.gradient(int(p)));
    const double bound = kIdentityConstant * kEps * (scale[p] + std::fabs(q_t.gradient(int(p))));
    worst_poly = std::max(worst_poly, defect / (kEps * (scale[p] + std::fabs(q_t.gradient(int(p))))));
    EXPECT_LE(defect, bound) << label << " parameter " << p << ": " << f.gradient(int(p))
                             << " vs " << q_t.gradient(int(p));
  }

  // 3. Each gradient entry against the fourth-order central difference
  // (8 (f(+h) - f(-h)) - (f(+2h) - f(-2h))) / (12 h) of the double run.
  const double h = 1.0e-3;
  double worst_diff = 0.0;
  for (std::size_t p = 0; p < n_params; ++p) {
    const bool is_src = p < std::size_t(c.dim) * c.n_src();
    const std::size_t point = p / std::size_t(c.dim);
    const std::size_t a = p % std::size_t(c.dim);
    auto at = [&](double shift) {
      std::vector<double> s = c.src, t = c.tgt;
      (is_src ? s[3 * point + a] : t[3 * (point - c.n_src()) + a]) += shift;
      return solve<Space, double>(c, s, t).val;
    };
    const std::vector<double> fp1 = at(h), fm1 = at(-h), fp2 = at(2 * h), fm2 = at(-2 * h);
    for (std::size_t j = 0; j < g.val.size(); ++j) {
      const double fd = (8.0 * (fp1[j] - fm1[j]) - (fp2[j] - fm2[j])) / (12.0 * h);
      const double gr = g.val[j].gradient(int(p));
      const double err = std::fabs(gr - fd) / (1.0 + std::fabs(gr));
      worst_diff = std::max(worst_diff, err);
      EXPECT_LE(err, kDifferenceTolerance)
          << label << " weight " << j << " parameter " << p << ": gradient " << gr
          << " difference " << fd;
    }
  }
  std::cerr << "[mls-gradient] " << label << ": worst partition-of-unity ratio " << worst_unity
            << " eps, worst polynomial ratio " << worst_poly << " eps, worst difference error "
            << worst_diff << "\n";
}

TYPED_TEST(MlsShapeGradient, IdentitiesAndDifferencesOnATwoDimensionalLinearFit) {
  const Cloud c = local_cloud(grid_cloud(2, 1, 6, MlsRadial::Wendland2, 5, 1, 11));
  check_stencil<TypeParam>(c, "2D degree 1 k 6 wendland2");
}

TYPED_TEST(MlsShapeGradient, IdentitiesAndDifferencesForEveryRadialFunction) {
  for (MlsRadial r : {MlsRadial::Wendland0, MlsRadial::Wendland2, MlsRadial::Wendland4,
                      MlsRadial::Wendland6}) {
    const Cloud c = local_cloud(grid_cloud(2, 2, 10, r, 5, 1, 23));
    check_stencil<TypeParam>(c, std::string("2D degree 2 k 10 ") + fuse::to_cstring(r));
  }
}

TYPED_TEST(MlsShapeGradient, IdentitiesAndDifferencesInThreeDimensions) {
  const Cloud c = local_cloud(grid_cloud(3, 1, 8, MlsRadial::Wendland2, 4, 1, 37));
  check_stencil<TypeParam>(c, "3D degree 1 k 8 wendland2");
}

TYPED_TEST(MlsShapeGradient, IdentitiesAndDifferencesInOneDimension) {
  const Cloud c = local_cloud(grid_cloud(1, 2, 6, MlsRadial::Wendland4, 9, 1, 5));
  check_stencil<TypeParam>(c, "1D degree 2 k 6 wendland4");
}

// A stencil symmetric about the target in x: sources at (+-1, 0), (0, +-1.5) and
// (0, 2.2) around the target at the origin. The moment-matrix entry
// sum_j phi_j dx_j is then EXACTLY zero in value (the two terms are equal and
// opposite, the rest have dx = 0) while its gradient is not, so the elimination
// multipliers of that column are zero in value and nonzero in gradient: a
// kernel that skipped them on the value alone would lose their contribution
// (mls_is_zero). The farthest source is unique, so the support radius is
// differentiable here and the central difference is a valid reference.
TYPED_TEST(MlsShapeGradient, SymmetricStencilKeepsTheGradientOfAZeroMultiplier) {
  Cloud c;
  c.dim = 2;
  c.degree = 1;
  c.k = 5;
  c.radial = MlsRadial::Wendland2;
  c.src = {1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, -1.5, 0.0, 0.0, 2.2, 0.0};
  c.tgt = {0.0, 0.0, 0.0};
  c.nbr = {0, 1, 2, 3, 4};
  check_stencil<TypeParam>(c, "symmetric stencil, degree 1");
}

// A neighbour that coincides with the target has distance 0, where the square
// root of the distance has no gradient. mls_norm_of_squares gives that distance
// the gradient 0, which is the gradient of the weight for Wendland2/4/6 (their
// radial function has zero slope at the origin); the weights are then smooth
// there and agree with the central difference. For Wendland0 (a cone at the
// origin) the value is checked for finiteness and the identities, which hold for
// any choice of the gradient of the distances (the weights satisfy the moment
// equations for every positive radial values).
TYPED_TEST(MlsShapeGradient, NeighbourCoincidingWithTheTargetHasAFiniteGradient) {
  for (MlsRadial r : {MlsRadial::Wendland2, MlsRadial::Wendland4, MlsRadial::Wendland6}) {
    Cloud c = local_cloud(grid_cloud(2, 1, 6, r, 5, 1, 11));
    for (int a = 0; a < 3; ++a) c.tgt[std::size_t(a)] = c.src[std::size_t(a)];  // source 0 = target
    check_stencil<TypeParam>(c, std::string("coincident neighbour ") + fuse::to_cstring(r));
  }
  Cloud c0 = local_cloud(grid_cloud(2, 1, 6, MlsRadial::Wendland0, 5, 1, 11));
  for (int a = 0; a < 3; ++a) c0.tgt[std::size_t(a)] = c0.src[std::size_t(a)];
  const std::vector<D27> src_d = seeded(c0, c0.src, 0);
  const std::vector<D27> tgt_d = seeded(c0, c0.tgt, c0.n_src());
  const Solved<D27> g = solve<TypeParam, D27>(c0, src_d, tgt_d);
  ASSERT_EQ(g.n_fail, 0);
  for (const D27& v : g.val) {
    EXPECT_TRUE(std::isfinite(v.value()));
    for (std::size_t p = 0; p < c0.n_params(); ++p) EXPECT_TRUE(std::isfinite(v.gradient(int(p))));
  }
}

// ---------------------------------------------------------------------------
// The gate of stage U8: the Taylor remainder of the weights, as a function of ALL
// the coordinates (every source and every target), has order 2 along a fixed
// direction when the claimed gradient is the dual-number one, and order 1 when
// the gradient is wrong. docs/dev/derivative-gate.md: the sweep h = h0 10^-k, the
// round-off floor identified rather than fitted through, the order read in
// (1.8, 2.3) for a correct gradient and (0.7, 1.3) for a wrong one.
// ---------------------------------------------------------------------------

/// The parameter vector of `c`: `dim` coordinates per source, then per target.
std::vector<double> pack(const Cloud& c) {
  std::vector<double> x;
  for (std::size_t i = 0; i < c.n_src(); ++i) {
    for (int a = 0; a < c.dim; ++a) x.push_back(c.src[3 * i + std::size_t(a)]);
  }
  for (std::size_t i = 0; i < c.n_tgt(); ++i) {
    for (int a = 0; a < c.dim; ++a) x.push_back(c.tgt[3 * i + std::size_t(a)]);
  }
  return x;
}

/// Point arrays (3 per point) from a parameter vector of the same layout.
template <class T>
void unpack(const Cloud& c, const std::vector<T>& x, std::vector<T>& s, std::vector<T>& t) {
  s.assign(c.src.size(), T(0.0));
  t.assign(c.tgt.size(), T(0.0));
  const std::size_t dim = std::size_t(c.dim);
  for (std::size_t i = 0; i < c.n_src(); ++i) {
    for (std::size_t a = 0; a < dim; ++a) s[3 * i + a] = x[dim * i + a];
  }
  for (std::size_t i = 0; i < c.n_tgt(); ++i) {
    for (std::size_t a = 0; a < dim; ++a) t[3 * i + a] = x[dim * (c.n_src() + i) + a];
  }
}

template <class Space>
fuse::linalg::TaylorRemainderResult weights_taylor(const Cloud& c, const std::vector<double>& v,
                                                   bool include_target_motion, double h0,
                                                   int ndecades) {
  const std::size_t n_moving = std::size_t(c.dim) * c.n_src();  // parameters of the sources
  const fuse::linalg::HostVectorFn F = [&](const std::vector<double>& x) {
    std::vector<double> s, t;
    unpack<double>(c, x, s, t);
    return solve<Space, double>(c, s, t).val;
  };
  const fuse::linalg::HostJacobianActionFn Jv = [&](const std::vector<double>& x,
                                                    const std::vector<double>& dir) {
    std::vector<D1> xd(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      xd[i] = D1(x[i]);
      xd[i].d[1] = (include_target_motion || i < n_moving) ? dir[i] : 0.0;
    }
    std::vector<D1> s, t;
    unpack<D1>(c, xd, s, t);
    const Solved<D1> g = solve<Space, D1>(c, s, t);
    std::vector<double> out;
    for (const D1& w : g.val) out.push_back(w.gradient(0));
    return out;
  };
  return fuse::linalg::taylor_remainder_order(F, Jv, pack(c), v, h0, ndecades);
}

std::vector<double> unit_direction(std::size_t n, std::uint64_t seed) {
  Lcg rng{seed};
  std::vector<double> v(n);
  double vmax = 0.0;
  for (double& e : v) {
    e = rng.symmetric();
    vmax = std::max(vmax, std::fabs(e));
  }
  for (double& e : v) e /= vmax;  // max-norm 1
  return v;
}

/// A cloud whose every target has a gap between the two farthest neighbour
/// distances above `min_gap`, found by trying seeds in order (the generator is
/// the same on every platform, so the cloud is too).
Cloud cloud_with_radius_gap(int dim, int degree, int k, MlsRadial radial, int n,
                            std::size_t n_tgt, double min_gap) {
  for (std::uint64_t seed = 1; seed < 2000; ++seed) {
    Cloud c = grid_cloud(dim, degree, k, radial, n, n_tgt, seed);
    if (smallest_radius_gap(c) > min_gap) return c;
  }
  ADD_FAILURE() << "no cloud with a radius gap above " << min_gap;
  return grid_cloud(dim, degree, k, radial, n, n_tgt, 1);
}

template <class Space>
void check_taylor(const Cloud& c, std::uint64_t direction_seed, const std::string& label) {
  const double h0 = 1.0e-2;
  const int ndecades = 9;
  // Along h v every coordinate moves by at most h, so two distances close by at
  // most 4 sqrt(dim) h: below that gap no neighbour swaps the farthest place and
  // the weights stay on one smooth branch for the whole sweep.
  ASSERT_GT(smallest_radius_gap(c), 4.0 * std::sqrt(double(c.dim)) * h0) << label;
  const std::vector<double> v = unit_direction(pack(c).size(), direction_seed);
  const auto good = weights_taylor<Space>(c, v, /*include_target_motion=*/true, h0, ndecades);
  ASSERT_FALSE(good.at_floor_everywhere) << label;
  const auto wrong = weights_taylor<Space>(c, v, /*include_target_motion=*/false, h0, ndecades);
  ASSERT_FALSE(wrong.at_floor_everywhere) << label;
  std::cerr << "[mls-gradient] taylor " << label << ": order " << good.order << " (floor "
            << good.floor << ", " << good.fit_points << " points), target motion left out: order "
            << wrong.order << "\n";
  EXPECT_GT(good.order, 1.8) << label;
  EXPECT_LT(good.order, 2.3) << label;
  EXPECT_GT(wrong.order, 0.7) << label;
  EXPECT_LT(wrong.order, 1.3) << label;
}

TYPED_TEST(MlsShapeGradient, TaylorRemainderHasOrderTwoInTwoDimensionsForEveryRadialFunction) {
  for (MlsRadial r : {MlsRadial::Wendland0, MlsRadial::Wendland2, MlsRadial::Wendland4,
                      MlsRadial::Wendland6}) {
    const Cloud c = cloud_with_radius_gap(2, 2, 10, r, 6, 3, 0.2);
    check_taylor<TypeParam>(c, 101, std::string("2D degree 2 k 10 ") + fuse::to_cstring(r));
  }
}

TYPED_TEST(MlsShapeGradient, TaylorRemainderHasOrderTwoInOneAndThreeDimensions) {
  check_taylor<TypeParam>(cloud_with_radius_gap(1, 2, 6, MlsRadial::Wendland4, 12, 3, 0.2), 7,
                          "1D degree 2 k 6 wendland4");
  check_taylor<TypeParam>(cloud_with_radius_gap(3, 1, 10, MlsRadial::Wendland2, 4, 2, 0.1), 9,
                          "3D degree 1 k 10 wendland2");
}

// ---------------------------------------------------------------------------
// Mixed precision: the same kernel on dual<float, 1>. The reference is the
// double run on the SAME coordinates (rounded to single first), so the only
// difference is the single-precision arithmetic of the kernel.
// ---------------------------------------------------------------------------

TYPED_TEST(MlsShapeGradient, SinglePrecisionGradientAgreesWithDoubleToSinglePrecision) {
  const Cloud c = local_cloud(grid_cloud(2, 1, 6, MlsRadial::Wendland2, 5, 1, 11));
  const std::vector<double> x = pack(c);
  const std::vector<double> dir = unit_direction(x.size(), 3);
  std::vector<F1> xf(x.size());
  std::vector<D1> xd(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float xr = static_cast<float>(x[i]);
    const float dr = static_cast<float>(dir[i]);
    xf[i] = F1(xr);
    xf[i].d[1] = dr;
    xd[i] = D1(static_cast<double>(xr));
    xd[i].d[1] = static_cast<double>(dr);
  }
  std::vector<F1> sf, tf;
  std::vector<D1> sd, td;
  unpack<F1>(c, xf, sf, tf);
  unpack<D1>(c, xd, sd, td);
  const Solved<F1> gf = solve<TypeParam, F1>(c, sf, tf);
  const Solved<D1> gd = solve<TypeParam, D1>(c, sd, td);
  ASSERT_EQ(gf.n_fail, 0);
  ASSERT_EQ(gd.n_fail, 0);
  double worst = 0.0;
  for (std::size_t j = 0; j < gd.val.size(); ++j) {
    for (int slot = 0; slot < 2; ++slot) {
      const double ref = gd.val[j].d[slot];
      const double got = static_cast<double>(gf.val[j].d[slot]);
      const double err = std::fabs(got - ref) / (1.0 + std::fabs(ref));
      worst = std::max(worst, err);
      EXPECT_LE(err, kSingleTolerance) << "weight " << j << " slot " << slot;
    }
  }
  std::cerr << "[mls-gradient] single precision: worst relative difference from double " << worst
            << "\n";
}

}  // namespace
