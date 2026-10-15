// Numerical order-of-accuracy study for the temporal bases.
//
// Whatever order a basis claims, Phase 6.5's convergence study will measure it,
// and a basis that claims p and delivers p-1 has to fail HERE, not there. Every
// order stated in waveform.hpp's header comment is measured below.
//
// The refinement being performed matters and is easy to get wrong:
//
//   * For a *waveform*, refinement means SHORTENING THE time window of a waveform iteration while
//     keeping the interpolatory sample count (p+1 samples for a degree-p
//     representation). That is what actually happens when a component reduces
//     its coupling interval. Over a window of length h the error is O(h^(p+1)).
//
//   * Adding samples to a fixed window at fixed p does NOT converge for DG --
//     the least-squares fit tends to the best degree-p approximation and the
//     error saturates. `DgDoesNotConvergeUnderSampleRefinement` pins that so
//     nobody designs a multirate scheme around a refinement that does nothing.

#include <gtest/gtest.h>
#include <fuse/num/temporal_basis.hpp>
#include <fuse/num/waveform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using fuse::Rational;
using fuse::TemporalBasis;
using fuse::Waveform;

namespace {

/// The test function for every order study below: a pure exponential.
///
/// The choice is not cosmetic. An order study only measures the intended order
/// if the leading error term dominates at the h values used, and for
/// f = exp(a t) the k-th derivative is a^k exp(a t) for every k, so the
/// relative variation of the error coefficient across a window is (e^(a h) - 1)
/// regardless of the order being measured. Every basis here is therefore
/// equally close to its asymptotic range on one shared h ladder, and a = 4
/// keeps the finest DG(3) error near 1e-8 -- six orders of magnitude above the
/// measured 1e-14 round-off floor of the degree-3 fit.
///
/// Two functions were tried first and rejected, and the reasons are worth
/// keeping because they are the standard ways an order study lies:
///
///  * `exp(sin 3t)`: d^3/dx^3 exp(sin x) = -exp(sin x) cos x sin x (3 + sin x)
///    VANISHES at x = 0, and every window here starts at t = 0. The DG(2)
///    leading error term is identically zero and the study reports order 4.
///    It would have accepted a false claim of 4 and rejected the true 3.
///  * `exp(sin(3t + phi))` for a generic phi: no zero derivative, but the
///    successive derivatives of exp(sin) differ by more than an order of
///    magnitude and pass through zeros nearby, so at any single h the
///    different bases sit at very different distances from their asymptotic
///    ranges (measured: DG(2) 2.99 but DG(3) 4.50 at phi = 1; Linear 1.89 but
///    DG(3) 4.01 at phi = 0.4).
double f(double t) { return std::exp(4.0 * t); }

/// Reference integral of f over [0, b], by composite 4-point Gauss-Legendre.
/// 400 panels puts the quadrature error far below any interpolation error
/// measured here (a 4-point rule is exact through degree 7).
double exact_integral(double b) {
  const fuse::num::GaussRule g = fuse::num::gauss_rule(4);
  constexpr int              kPanels = 400;
  const double               dx      = b / kPanels;
  double                     sum     = 0.0;
  for (int p = 0; p < kPanels; ++p) {
    const double lo   = p * dx;
    const double mid  = lo + 0.5 * dx;
    const double half = 0.5 * dx;
    for (int q = 0; q < g.count; ++q) {
      sum += g.weights[q] * half * f(mid + half * g.nodes[q]);
    }
  }
  return sum;
}

/// Build a 1-element waveform on the window [0, 1/den] with `samples` equally
/// spaced sample points, the first at 0 and the last at 1/den (or just the
/// first, when samples == 1 -- the explicit/Constant case).
Waveform on_window(TemporalBasis basis, int degree, std::int64_t den,
                   int samples) {
  Waveform w{1, static_cast<std::size_t>(samples), basis, degree};
  w.reset(Rational{0}, Rational{1, den});
  for (int i = 0; i < samples; ++i) {
    // t_i = (i / (samples-1)) * (1/den), exactly.
    const Rational t = (samples == 1)
                           ? Rational{0}
                           : Rational{i, static_cast<std::int64_t>(samples - 1) * den};
    const double v = f(t.to_double());
    EXPECT_TRUE(w.push(t, &v).ok());
  }
  return w;
}

/// Max pointwise error over 22 points inside the window, chosen at k/23 of the
/// window so they never coincide with a sample node for any of the sample
/// counts used here (23 is prime).
double max_eval_error(const Waveform& w, std::int64_t den) {
  double worst = 0.0;
  for (std::int64_t k = 1; k <= 22; ++k) {
    const Rational t{k, 23 * den};
    double         out = 0.0;
    EXPECT_TRUE(w.evaluate(t, &out).ok()) << t.to_string();
    worst = std::max(worst, std::fabs(out - f(t.to_double())));
  }
  return worst;
}

/// Tolerance on every measured order. The observed deviations are all <= 0.10
/// (the next term of the exponential's error expansion, which decays as the
/// ladder refines and always makes the order come out slightly HIGH), so 0.15
/// is loose enough to be stable and more than six times tighter than the 1.0
/// that a genuine loss of one order would produce.
constexpr double kTol = 0.15;

double order_of(double coarse_err, double fine_err) {
  return std::log2(coarse_err / fine_err);
}

/// Halve the window three times and report the observed order at each step.
/// The finest pair is the one asserted on; the coarser ones are printed so a
/// regression shows the whole trend, not just a single number.
void expect_window_order(const char* label, TemporalBasis basis, int degree,
                         int samples, double expected, double tol) {
  const std::int64_t dens[] = {8, 16, 32, 64};
  double             err[4];
  for (int i = 0; i < 4; ++i) {
    err[i] = max_eval_error(on_window(basis, degree, dens[i], samples), dens[i]);
    ASSERT_GT(err[i], 0.0) << label << " h=1/" << dens[i]
                           << ": exactly zero error means the study is not "
                              "measuring anything";
  }
  std::string trace;
  for (int i = 1; i < 4; ++i) {
    trace += "  h=1/" + std::to_string(dens[i - 1]) + "->1/" +
             std::to_string(dens[i]) + "  err " + std::to_string(err[i - 1]) +
             " -> " + std::to_string(err[i]) + "  order " +
             std::to_string(order_of(err[i - 1], err[i])) + "\n";
  }
  const double observed = order_of(err[2], err[3]);
  testing::Test::RecordProperty(std::string(label) + "_order", std::to_string(observed));
  EXPECT_NEAR(observed, expected, tol)
      << label << " claims order " << expected << "\n" << trace;
  std::cout << "[ ORDER    ] " << label << " expected " << expected
            << ", observed " << observed << "\n"
            << trace;
}

}  // namespace

// ---------------------------------------------------------------------------
// Pointwise interpolation order under window refinement
// ---------------------------------------------------------------------------

TEST(WaveformOrder, ConstantIsFirstOrder) {
  expect_window_order("Constant(1 sample)", TemporalBasis::Constant, 0, 1, 1.0,
                      kTol);
}

TEST(WaveformOrder, LinearIsSecondOrder) {
  expect_window_order("Linear(2 samples)", TemporalBasis::Linear, 1, 2, 2.0,
                      kTol);
}

TEST(WaveformOrder, DgOneIsSecondOrder) {
  expect_window_order("DG(1), 2 samples", TemporalBasis::DG, 1, 2, 2.0, kTol);
}

TEST(WaveformOrder, DgTwoIsThirdOrder) {
  expect_window_order("DG(2), 3 samples", TemporalBasis::DG, 2, 3, 3.0, kTol);
}

TEST(WaveformOrder, DgThreeIsFourthOrder) {
  expect_window_order("DG(3), 4 samples", TemporalBasis::DG, 3, 4, 4.0, kTol);
}

// A natural cubic spline over a window it does not extend beyond is limited by
// its end condition S'' == 0, which is wrong here, so it is SECOND order --
// not fourth. Measured, not assumed.
TEST(WaveformOrder, NaturalSplineOverAWindowIsSecondOrder) {
  expect_window_order("Spline(3), 4 samples", TemporalBasis::Spline, 3, 4, 2.0,
                      kTol);
}

// ---------------------------------------------------------------------------
// The trap: refining samples inside a fixed window does not refine DG(p)
// ---------------------------------------------------------------------------

TEST(WaveformOrder, DgDoesNotConvergeUnderSampleRefinement) {
  const auto err_with = [](int samples) {
    Waveform w{1, static_cast<std::size_t>(samples), TemporalBasis::DG, 1};
    w.reset(Rational{0}, Rational{1});
    for (int i = 0; i < samples; ++i) {
      const Rational t{i, samples - 1};
      const double   v = f(t.to_double());
      EXPECT_TRUE(w.push(t, &v).ok());
    }
    return max_eval_error(w, 1);
  };
  const double e3  = err_with(3);
  const double e17 = err_with(17);
  std::cout << "[ ORDER    ] DG(1) fixed window: 3 samples err " << e3
            << ", 17 samples err " << e17 << "\n";
  // Not merely "slow": the error is essentially unchanged, because both are
  // approximating the same best-degree-1 fit.
  EXPECT_GT(e17, 0.5 * e3)
      << "DG(p) appears to converge under sample refinement; it does not, and "
         "a scheme that relies on that would silently lose order";
  EXPECT_GT(e17, 0.05) << "still O(1) error, as a degree-1 fit to a curved "
                          "function over a unit window must be";
}

namespace {

/// A Spline waveform over [0, 1] with `intervals` equal sample intervals.
Waveform spline_on_unit_window(std::int64_t intervals) {
  const int samples = static_cast<int>(intervals) + 1;
  Waveform  w{1, static_cast<std::size_t>(samples), TemporalBasis::Spline, 3};
  w.reset(Rational{0}, Rational{1});
  for (std::int64_t i = 0; i <= intervals; ++i) {
    const Rational t{i, intervals};
    const double   v = f(t.to_double());
    EXPECT_TRUE(w.push(t, &v).ok());
  }
  return w;
}

double eval_at(const Waveform& w, Rational t) {
  double out = 0.0;
  EXPECT_TRUE(w.evaluate(t, &out).ok()) << t.to_string();
  return out;
}

}  // namespace

// A spline, unlike DG, IS interpolatory and does converge under sample
// refinement. Away from the ends it is fourth order: the wrong natural end
// condition pollutes the solution, but that pollution decays geometrically
// (ratio sqrt(3) - 2 per interval), so at a FIXED interior point the number of
// intervals separating it from the boundary grows as h shrinks and the
// pollution disappears faster than any power of h.
TEST(WaveformOrder, SplineRefinementIsFourthOrderAwayFromTheEnds) {
  // Measured at the midpoints of the interior half of the intervals, where
  // the O(h^4) interpolation error peaks and where the end-condition
  // pollution (which decays like |sqrt(3) - 2|^j per interval j from the
  // boundary) is already below it.
  const auto err_with = [](std::int64_t intervals) {
    const Waveform w     = spline_on_unit_window(intervals);
    double         worst = 0.0;
    for (std::int64_t j = intervals / 4; j < 3 * intervals / 4; ++j) {
      const Rational t{2 * j + 1, 2 * intervals};
      worst = std::max(worst, std::fabs(eval_at(w, t) - f(t.to_double())));
    }
    return worst;
  };
  const double e32  = err_with(32);
  const double e64  = err_with(64);
  const double e128 = err_with(128);
  const double observed = order_of(e64, e128);
  std::cout << "[ ORDER    ] Spline(3) interior refinement: " << e32 << " -> "
            << e64 << " -> " << e128 << ", order " << observed << "\n";
  testing::Test::RecordProperty("spline_interior_order",
                                std::to_string(observed));
  EXPECT_NEAR(observed, 4.0, kTol);
}

// ...but inside the boundary layer -- the first sample interval, where the
// natural condition S'' == 0 is imposed on a function whose f'' is not zero --
// it is only SECOND order. This is why waveform.hpp does not claim 4, and why
// the 4-sample window-refinement case above measures 2: there, every interval
// is a boundary interval.
TEST(WaveformOrder, SplineBoundaryLayerIsOnlySecondOrder) {
  const auto err_at_first_midpoint = [](std::int64_t intervals) {
    const Waveform w = spline_on_unit_window(intervals);
    const Rational t{1, 2 * intervals};  // midpoint of the first interval
    return std::fabs(eval_at(w, t) - f(t.to_double()));
  };
  const double e16 = err_at_first_midpoint(16);
  const double e32 = err_at_first_midpoint(32);
  const double e64 = err_at_first_midpoint(64);
  const double observed = order_of(e32, e64);
  std::cout << "[ ORDER    ] Spline(3) boundary layer: " << e16 << " -> " << e32
            << " -> " << e64 << ", order " << observed << "\n";
  testing::Test::RecordProperty("spline_boundary_order",
                                std::to_string(observed));
  EXPECT_NEAR(observed, 2.0, kTol);
}

// ---------------------------------------------------------------------------
// Integration order under window refinement
// ---------------------------------------------------------------------------

namespace {

void expect_integral_order(const char* label, TemporalBasis basis, int degree,
                           int samples, double expected, double tol) {
  const std::int64_t dens[] = {4, 8, 16, 32};
  double             err[4];
  for (int i = 0; i < 4; ++i) {
    const Waveform w = on_window(basis, degree, dens[i], samples);
    const Rational b{1, dens[i]};
    double         out = 0.0;
    ASSERT_TRUE(w.integrate(Rational{0}, b, &out).ok());
    err[i] = std::fabs(out - exact_integral(b.to_double()));
    ASSERT_GT(err[i], 0.0) << label;
  }
  const double observed = order_of(err[2], err[3]);
  std::cout << "[ ORDER    ] integral " << label << " expected " << expected
            << ", observed " << observed << "  (" << err[0] << " " << err[1]
            << " " << err[2] << " " << err[3] << ")\n";
  testing::Test::RecordProperty(std::string(label) + "_integral_order",
                 std::to_string(observed));
  EXPECT_NEAR(observed, expected, tol) << label;
}

}  // namespace

// Left-rectangle rule: error O(h^2) on a window of length h.
TEST(WaveformOrder, ConstantIntegralIsSecondOrder) {
  expect_integral_order("Constant(1 sample)", TemporalBasis::Constant, 0, 1,
                        2.0, kTol);
}

// Trapezoid: O(h^3).
TEST(WaveformOrder, LinearIntegralIsThirdOrder) {
  expect_integral_order("Linear(2 samples)", TemporalBasis::Linear, 1, 2, 3.0,
                        kTol);
}

// The integral of the interpolating quadratic is Simpson's rule: O(h^5).
// This is the primitive the space-time conservation check (spec Section 7.4) rides
// on, so its order is the one that must not silently drop.
TEST(WaveformOrder, DgTwoIntegralIsFifthOrder) {
  expect_integral_order("DG(2), 3 samples", TemporalBasis::DG, 2, 3, 5.0, kTol);
}
