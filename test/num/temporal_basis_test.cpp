#include <gtest/gtest.h>
#include <fuse/num/temporal_basis.hpp>
#include <fuse/base/error.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

using fuse::num::cubic_spline_moments;
using fuse::num::eval_cubic_spline;
using fuse::num::eval_polynomial;
using fuse::num::fit_polynomial;
using fuse::num::gauss_rule;
using fuse::num::GaussRule;
using fuse::num::integrate_cubic_spline;
using fuse::num::integrate_polynomial;

namespace {

/// Apply an n-point Gauss rule on [a, b].
template <class F>
double quad(const GaussRule& g, double a, double b, F f) {
  const double mid  = 0.5 * (a + b);
  const double half = 0.5 * (b - a);
  double       s    = 0.0;
  for (int q = 0; q < g.count; ++q) s += g.weights[q] * half * f(mid + half * g.nodes[q]);
  return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Gauss-Legendre
// ---------------------------------------------------------------------------

TEST(TemporalBasis, GaussRuleIsExactThroughDegreeTwoNMinusOne) {
  for (int n = 1; n <= 4; ++n) {
    const GaussRule g = gauss_rule(n);
    ASSERT_EQ(g.count, n);
    const int top = 2 * n - 1;
    for (int p = 0; p <= top; ++p) {
      // int_0^2 x^p dx = 2^(p+1) / (p+1)
      const double got = quad(g, 0.0, 2.0, [p](double x) {
        return std::pow(x, static_cast<double>(p));
      });
      const double want = std::pow(2.0, p + 1) / (p + 1);
      EXPECT_NEAR(got, want, 1e-12 * std::max(1.0, std::fabs(want)))
          << "n=" << n << " degree=" << p;
    }
    // Weights sum to the interval length.
    double wsum = 0.0;
    for (int q = 0; q < n; ++q) wsum += g.weights[q];
    EXPECT_NEAR(wsum, 2.0, 1e-15) << "n=" << n;
  }
}

// The rule is a pair of raw pointers, so an unsupported count must throw
// rather than yield a null-pointer struct that only crashes in Release.
TEST(TemporalBasis, GaussRuleRejectsUnsupportedPointCounts) {
  EXPECT_THROW((void)gauss_rule(0), fuse::InvalidArgument);
  EXPECT_THROW((void)gauss_rule(-1), fuse::InvalidArgument);
  EXPECT_THROW((void)gauss_rule(5), fuse::InvalidArgument);
}

// ---------------------------------------------------------------------------
// Polynomial fit
// ---------------------------------------------------------------------------

TEST(TemporalBasis, FitRecoversTheExactCoefficientsOfAPolynomial) {
  const std::vector<double> t = {0.0, 0.25, 0.5, 0.75, 1.0};
  std::vector<double>       f(t.size());
  const auto p = [](double x) { return 3.0 - 2.0 * x + 5.0 * x * x; };
  for (std::size_t i = 0; i < t.size(); ++i) f[i] = p(t[i]);

  const std::vector<double> c = fit_polynomial(t, f, 2, 0.0);
  ASSERT_EQ(c.size(), 3u);
  EXPECT_NEAR(c[0], 3.0, 1e-12);
  EXPECT_NEAR(c[1], -2.0, 1e-12);
  EXPECT_NEAR(c[2], 5.0, 1e-12);
  EXPECT_NEAR(eval_polynomial(c, 0.4, 0.0), p(0.4), 1e-12);
  // int_0^1 p = 3 - 1 + 5/3
  EXPECT_NEAR(integrate_polynomial(c, 0.0, 1.0, 0.0), 3.0 - 1.0 + 5.0 / 3.0,
              1e-12);
}

TEST(TemporalBasis, FitIsExpandedAboutT0) {
  const std::vector<double> t = {10.0, 10.5, 11.0};
  const auto   p  = [](double x) { return 1.0 + 2.0 * x + 3.0 * x * x; };
  std::vector<double> f(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) f[i] = p(t[i]);

  const std::vector<double> c = fit_polynomial(t, f, 2, 10.0);
  // p(10 + s) = p(10) + p'(10) s + 3 s^2 = 321 + 62 s + 3 s^2
  EXPECT_NEAR(c[0], 321.0, 1e-9);
  EXPECT_NEAR(c[1], 62.0, 1e-9);
  EXPECT_NEAR(c[2], 3.0, 1e-9);
  EXPECT_NEAR(eval_polynomial(c, 10.3, 10.0), p(10.3), 1e-9);
}

// The requested degree is clamped to n-1: above that the normal equations are
// singular. Waveform relies on this agreeing with its own clamp.
TEST(TemporalBasis, FitClampsTheDegreeToTheSampleCount) {
  const std::vector<double> t = {0.0, 1.0};
  const std::vector<double> f = {1.0, 3.0};
  const std::vector<double> c = fit_polynomial(t, f, 4, 0.0);
  ASSERT_EQ(c.size(), 2u) << "degree must clamp to n-1 = 1";
  EXPECT_NEAR(c[0], 1.0, 1e-14);
  EXPECT_NEAR(c[1], 2.0, 1e-14);

  const std::vector<double> one_t = {2.0};
  const std::vector<double> one_f = {7.0};
  const std::vector<double> c1 = fit_polynomial(one_t, one_f, 3, 2.0);
  ASSERT_EQ(c1.size(), 1u);
  EXPECT_NEAR(c1[0], 7.0, 1e-14);
}

// A short coupling window is not exotic -- it is what a small time step looks
// like, and `Waveform` passes window-LOCAL times, so h here is literally the
// window length. The fit must be as accurate at h = 1e-6 as at h = 1.
//
// (Checked separately, and recorded in temporal_basis.hpp: this passes with
// and without the internal s = (t - t0)/h scaling, because the window length
// enters the normal equations as a pure diagonal similarity that partial
// pivoting absorbs. The test earns its keep as a property test, not as a
// regression test for the scaling.)
TEST(TemporalBasis, FitIsAccurateOnAVeryShortWindow) {
  constexpr double h = 1e-6;
  const std::vector<double> t = {0.0, 0.5 * h, h};
  const auto p = [](double x) { return 3.0 - 2.0 * x + 5.0 * x * x; };
  std::vector<double> f(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) f[i] = p(t[i]);

  const std::vector<double> c = fit_polynomial(t, f, 2, 0.0);
  for (const double x : {0.1 * h, 0.37 * h, 0.9 * h}) {
    EXPECT_NEAR(eval_polynomial(c, x, 0.0), p(x), 1e-12) << "x/h=" << x / h;
  }
  // int_0^h p dt = 3h - h^2 + 5h^3/3
  EXPECT_NEAR(integrate_polynomial(c, 0.0, h, 0.0),
              3.0 * h - h * h + 5.0 * h * h * h / 3.0, 1e-18);
}

// ...and the other extreme. A multi-day climate coupling window measured in
// seconds is order 1e6, so the long end is as real as the short one. The field
// VALUES stay O(1) across the window -- that is the physically meaningful
// case, and the one the returned monomial coefficients (which are then
// O(1/h^k), down to 5e-19 here) have to survive.
TEST(TemporalBasis, FitIsAccurateOnAVeryLongWindow) {
  constexpr double h = 1e6;
  const std::vector<double> t = {0.0, h / 3.0, 2.0 * h / 3.0, h};
  const auto p = [](double x) {
    const double u = x / h;
    return 3.0 - 2.0 * u + 5.0 * u * u - 0.5 * u * u * u;
  };
  std::vector<double> fv(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) fv[i] = p(t[i]);

  const std::vector<double> c = fit_polynomial(t, fv, 3, 0.0);
  ASSERT_EQ(c.size(), 4u);
  EXPECT_NEAR(c[3] * h * h * h, -0.5, 1e-11);
  for (const double s : {0.1, 0.37, 0.9}) {
    const double x = s * h;
    EXPECT_NEAR(eval_polynomial(c, x, 0.0), p(x), 1e-12) << "s=" << s;
  }
  // int_0^h p dt / h = 3 - 1 + 5/3 - 1/8
  EXPECT_NEAR(integrate_polynomial(c, 0.0, h, 0.0) / h,
              3.0 - 1.0 + 5.0 / 3.0 - 0.125, 1e-12);
}

TEST(TemporalBasis, FitRejectsMalformedInput) {
  const std::vector<double> t = {0.0, 1.0};
  const std::vector<double> f = {1.0};
  EXPECT_THROW((void)fit_polynomial(t, f, 1, 0.0), fuse::InvalidArgument);
  EXPECT_THROW((void)fit_polynomial({}, {}, 1, 0.0), fuse::InvalidArgument);
  EXPECT_THROW((void)fit_polynomial(t, {1.0, 2.0}, -1, 0.0),
               fuse::InvalidArgument);
  // Duplicate abscissae make the normal equations singular for degree 1.
  EXPECT_THROW((void)fit_polynomial({1.0, 1.0}, {2.0, 3.0}, 1, 1.0),
               std::runtime_error);
}

TEST(TemporalBasis, EvalPolynomialOfAnEmptyFitIsZero) {
  EXPECT_DOUBLE_EQ(eval_polynomial({}, 3.0, 0.0), 0.0);
  EXPECT_DOUBLE_EQ(integrate_polynomial({}, 0.0, 3.0, 0.0), 0.0);
}

// ---------------------------------------------------------------------------
// Natural cubic spline
// ---------------------------------------------------------------------------

TEST(TemporalBasis, SplineMomentsVanishBelowThreeSamples) {
  EXPECT_TRUE(cubic_spline_moments({}, {}).empty());
  EXPECT_EQ(cubic_spline_moments({1.0}, {2.0}).size(), 1u);
  const std::vector<double> m = cubic_spline_moments({0.0, 1.0}, {2.0, 4.0});
  ASSERT_EQ(m.size(), 2u);
  EXPECT_DOUBLE_EQ(m[0], 0.0);
  EXPECT_DOUBLE_EQ(m[1], 0.0);
  // Two samples therefore degenerate to linear interpolation.
  EXPECT_NEAR(eval_cubic_spline({0.0, 1.0}, {2.0, 4.0}, m, 0.25), 2.5, 1e-14);
}

TEST(TemporalBasis, SplineInterpolatesEverySampleAndIsC2AtTheEnds) {
  const std::vector<double> t = {0.0, 0.3, 0.55, 0.8, 1.0};
  std::vector<double>       f(t.size());
  for (std::size_t i = 0; i < t.size(); ++i) f[i] = std::exp(std::sin(3.0 * t[i]));
  const std::vector<double> m = cubic_spline_moments(t, f);
  ASSERT_EQ(m.size(), t.size());
  // Natural end condition.
  EXPECT_DOUBLE_EQ(m.front(), 0.0);
  EXPECT_DOUBLE_EQ(m.back(), 0.0);
  for (std::size_t i = 0; i < t.size(); ++i) {
    EXPECT_NEAR(eval_cubic_spline(t, f, m, t[i]), f[i], 1e-13) << "i=" << i;
  }
}

// Two-point Gauss is exact through degree 3 and the spline is a cubic on each
// piece, so the piecewise integral carries no quadrature error at all.
TEST(TemporalBasis, SplineIntegralIsExactPiecewise) {
  const std::vector<double> t = {0.0, 1.0, 2.0};
  const std::vector<double> f = {0.0, 1.0, 8.0};
  const std::vector<double> m = cubic_spline_moments(t, f);
  // Compare against a very fine composite Simpson of the same spline.
  const auto s = [&](double x) { return eval_cubic_spline(t, f, m, x); };
  constexpr int kN = 20000;
  const double  dx = 2.0 / kN;
  double        ref = 0.0;
  for (int i = 0; i < kN; ++i) {
    const double a = i * dx, b = a + dx;
    ref += (b - a) / 6.0 * (s(a) + 4.0 * s(0.5 * (a + b)) + s(b));
  }
  EXPECT_NEAR(integrate_cubic_spline(t, f, m, 0.0, 2.0), ref, 1e-10);
  // Clipped sub-interval, and the additivity that space-time conservation
  // accounting depends on.
  const double left  = integrate_cubic_spline(t, f, m, 0.0, 0.7);
  const double right = integrate_cubic_spline(t, f, m, 0.7, 2.0);
  EXPECT_NEAR(left + right, integrate_cubic_spline(t, f, m, 0.0, 2.0), 1e-13);
  EXPECT_DOUBLE_EQ(integrate_cubic_spline(t, f, m, 1.0, 1.0), 0.0);
  EXPECT_DOUBLE_EQ(integrate_cubic_spline(t, f, m, 1.5, 0.5), 0.0);
}

TEST(TemporalBasis, SplineRejectsMalformedInput) {
  EXPECT_THROW((void)cubic_spline_moments({0.0, 1.0}, {1.0}),
               fuse::InvalidArgument);
  EXPECT_THROW((void)cubic_spline_moments({0.0, 1.0, 1.0}, {1.0, 2.0, 3.0}),
               fuse::InvalidArgument);
  EXPECT_THROW((void)eval_cubic_spline({}, {}, {}, 0.0), fuse::InvalidArgument);
  EXPECT_THROW((void)integrate_cubic_spline({}, {}, {}, 0.0, 1.0),
               fuse::InvalidArgument);
  EXPECT_THROW((void)eval_cubic_spline({0.0, 1.0}, {1.0, 2.0}, {0.0}, 0.5),
               fuse::InvalidArgument);
}
