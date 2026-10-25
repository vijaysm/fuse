#include <gtest/gtest.h>
#include <fuse/num/waveform.hpp>
#include <fuse/base/error.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

using fuse::Rational;
using fuse::TemporalBasis;
using fuse::Waveform;

// ---------------------------------------------------------------------------
// Allocation counting, as in test/num/accelerator_test.cpp.
// `WaveformOrder.DgAndSplineEvaluateAllocationIsIndependentOfElement
// Count` below is the fence that MEASURES the per-element solve buffers
// being hoisted, rather than asserting it in a comment. This is the only
// translation unit in this test binary, so defining global operator
// new/delete here is safe (one definition per binary, not per file).
// ---------------------------------------------------------------------------
namespace {
std::size_t g_allocations = 0;
bool        g_counting    = false;

class AllocationCounter {
 public:
  AllocationCounter() {
    g_allocations = 0;
    g_counting    = true;
  }
  ~AllocationCounter() { g_counting = false; }
  AllocationCounter(const AllocationCounter&)            = delete;
  AllocationCounter& operator=(const AllocationCounter&) = delete;
  [[nodiscard]] std::size_t count() const { return g_allocations; }
};
}  // namespace

void* operator new(std::size_t n) {
  // #ifndef/#ifdef __CUDA_ARCH__ around the counting AND the failure path: a
  // CUDA-enabled Kokkos dual-compiles every global operator new/delete
  // override for device too (nothing here is reachable from a kernel; nvcc
  // does this unconditionally), and the device side can neither support
  // exceptions nor see g_counting/g_allocations, host globals.
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
  void* p = std::malloc(n == 0 ? 1 : n);
#ifdef __CUDA_ARCH__
  if (p == nullptr) __trap();  // the device pass cannot throw (and never runs this)
#else
  if (p == nullptr) throw std::bad_alloc();  // the standard contract, on the host
#endif
  return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
#ifndef __CUDA_ARCH__
  if (g_counting) ++g_allocations;
#endif
  return std::malloc(n == 0 ? 1 : n);
}
void* operator new[](std::size_t n, const std::nothrow_t& t) noexcept {
  return ::operator new(n, t);
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept {
  std::free(p);
}

namespace {

/// Fill a 1-element waveform with f(t) at the given times over an explicit
/// window. The window is a separate argument from the sample times because
/// they are genuinely different things for the `Constant` basis: the last
/// sample holds to `window_end`, not to the last sample time.
template <class F>
Waveform sampled_in(TemporalBasis basis, int degree, Rational window_begin,
                    Rational window_end, const std::vector<Rational>& times,
                    F f) {
  Waveform w{1, times.size(), basis, degree};
  w.reset(window_begin, window_end);
  for (const Rational t : times) {
    const double v = f(t.to_double());
    EXPECT_TRUE(w.push(t, &v).ok());
  }
  return w;
}

/// The common case: the window is exactly the sampled range.
template <class F>
Waveform sampled(TemporalBasis basis, int degree,
                 const std::vector<Rational>& times, F f) {
  return sampled_in(basis, degree, times.front(), times.back(), times, f);
}

double eval(const Waveform& w, Rational t) {
  double out = 0.0;
  EXPECT_TRUE(w.evaluate(t, &out).ok());
  return out;
}

double integral(const Waveform& w, Rational a, Rational b) {
  double out = 0.0;
  EXPECT_TRUE(w.integrate(a, b, &out).ok());
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction and sample bookkeeping
// ---------------------------------------------------------------------------

TEST(Waveform, StartsEmpty) {
  Waveform w{4, 8, TemporalBasis::Linear, 1};
  EXPECT_EQ(w.sample_count(), 0u);
  EXPECT_EQ(w.elements(), 4u);
  EXPECT_EQ(w.capacity(), 8u);
  EXPECT_EQ(w.basis(), TemporalBasis::Linear);
  EXPECT_EQ(w.degree(), 1);
  // Empty is checked before the buffer, so the null `out` is never touched.
  const auto r = w.evaluate(Rational{0}, nullptr);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-EMPTY");
}

TEST(Waveform, RecordsSamplesInOrder) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double a = 1.0, b = 2.0;
  ASSERT_TRUE(w.push(Rational{0}, &a).ok());
  ASSERT_TRUE(w.push(Rational{1, 2}, &b).ok());
  EXPECT_EQ(w.sample_count(), 2u);
  EXPECT_EQ(w.sample_time(1), Rational(1, 2));
  EXPECT_DOUBLE_EQ(w.sample_values(1)[0], 2.0);
}

TEST(Waveform, RejectsOutOfOrderSamples) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{1, 2}, &v).ok());
  const auto r = w.push(Rational{1, 4}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-ORDER");
}

// Strictly increasing, not merely non-decreasing: two samples at the same time
// would make the Linear bracket divide by zero and the DG normal equations
// singular.
TEST(Waveform, RejectsRepeatedSampleTimes) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{1, 2}, &v).ok());
  const auto r = w.push(Rational{1, 2}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-ORDER");
}

TEST(Waveform, RejectsSamplesOutsideTheWindow) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  const auto   r = w.push(Rational{2}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-WINDOW");
  EXPECT_FALSE(w.push(Rational{-1}, &v).ok());
  EXPECT_EQ(w.sample_count(), 0u);
}

TEST(Waveform, RejectsOverfill) {
  Waveform w{1, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  ASSERT_TRUE(w.push(Rational{1, 2}, &v).ok());
  const auto r = w.push(Rational{1}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-FULL");
}

TEST(Waveform, RejectsNullBuffers) {
  Waveform w{2, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  {
    const auto r = w.push(Rational{0}, nullptr);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-NULL");
  }
  const double v[2] = {1.0, 2.0};
  ASSERT_TRUE(w.push(Rational{0}, v).ok());
  {
    const auto r = w.evaluate(Rational{0}, nullptr);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-NULL");
  }
  {
    const auto r = w.integrate(Rational{0}, Rational{0}, nullptr);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-NULL");
  }
  {
    const auto r = w.mean(nullptr);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-NULL");
  }
}

// An empty mesh partition on a rank is legal, so a zero-element waveform must
// work rather than trip a null-buffer check.
TEST(Waveform, ZeroElementsIsLegal) {
  Waveform w{0, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  ASSERT_TRUE(w.push(Rational{0}, nullptr).ok());
  ASSERT_TRUE(w.push(Rational{1}, nullptr).ok());
  EXPECT_TRUE(w.evaluate(Rational{1, 2}, nullptr).ok());
  EXPECT_TRUE(w.integrate(Rational{0}, Rational{1}, nullptr).ok());
  EXPECT_TRUE(w.mean(nullptr).ok());
}

TEST(Waveform, SampleAccessorsAreRangeChecked) {
  Waveform w{1, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  EXPECT_THROW((void)w.sample_time(1), fuse::InvalidArgument);
  EXPECT_THROW((void)w.sample_values(1), fuse::InvalidArgument);
}

// Standing rule: never hold a pointer into a vector across something that can
// grow it. The value buffer is sized once in the constructor, so it cannot
// grow, and this pins that.
TEST(Waveform, SampleValuesPointerSurvivesFurtherPushes) {
  Waveform w{1, 8, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{8});
  const double first = 11.0;
  ASSERT_TRUE(w.push(Rational{0}, &first).ok());
  const double* const p = w.sample_values(0);
  for (int i = 1; i < 8; ++i) {
    const double v = static_cast<double>(i);
    ASSERT_TRUE(w.push(Rational{i}, &v).ok());
  }
  EXPECT_EQ(p, w.sample_values(0));
  EXPECT_DOUBLE_EQ(*p, 11.0);
}

TEST(Waveform, ConstructorRejectsBasisDegreeMismatch) {
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::None, 0}), fuse::InvalidArgument);
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::Constant, 1}),
               fuse::InvalidArgument);
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::Linear, 2}),
               fuse::InvalidArgument);
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::DG, -1}), fuse::InvalidArgument);
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::DG, Waveform::kMaxDgDegree + 1}),
               fuse::InvalidArgument);
  // Only the natural cubic exists; silently building one when asked for a
  // quintic would be an unannounced loss of order.
  EXPECT_THROW((Waveform{1, 1, TemporalBasis::Spline, 5}),
               fuse::InvalidArgument);
  EXPECT_NO_THROW((Waveform{1, 1, TemporalBasis::Spline, 3}));
}

TEST(Waveform, ResetRejectsAnInvertedWindow) {
  Waveform w{1, 2, TemporalBasis::Linear, 1};
  EXPECT_THROW(w.reset(Rational{1}, Rational{0}), fuse::InvalidArgument);
  EXPECT_NO_THROW(w.reset(Rational{1}, Rational{1}));  // zero length is legal
}

TEST(Waveform, ResetClearsSamplesButKeepsCapacity) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  w.reset(Rational{1}, Rational{2});
  EXPECT_EQ(w.sample_count(), 0u);
  EXPECT_EQ(w.window_begin(), Rational{1});
  EXPECT_EQ(w.window_end(), Rational{2});
  EXPECT_EQ(w.capacity(), 4u);
  ASSERT_TRUE(w.push(Rational{1}, &v).ok());
}

// ---------------------------------------------------------------------------
// Interpolation
// ---------------------------------------------------------------------------

// Spec Section 4.4: explicit coupling is the degenerate waveform, not a separate path.
TEST(Waveform, SingleSampleConstantIsTheExplicitCase) {
  Waveform w{1, 1, TemporalBasis::Constant, 0};
  w.reset(Rational{0}, Rational{1});
  const double v = 7.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  EXPECT_TRUE(w.is_degenerate());
  EXPECT_DOUBLE_EQ(eval(w, Rational{0}), 7.0);
  // The single sample holds across the whole window: that is what makes the
  // explicit case usable at all by the other component.
  EXPECT_DOUBLE_EQ(eval(w, Rational{1, 2}), 7.0);
  EXPECT_DOUBLE_EQ(eval(w, Rational{1}), 7.0);
  double m = 0.0;
  ASSERT_TRUE(w.mean(&m).ok());
  EXPECT_DOUBLE_EQ(m, 7.0);
}

TEST(Waveform, IsDegenerateOnlyForOneConstantSample) {
  const auto two = sampled(TemporalBasis::Constant, 0,
                           {Rational{0}, Rational{1}}, [](double) { return 1.0; });
  EXPECT_FALSE(two.is_degenerate());
  const auto one_linear =
      sampled_in(TemporalBasis::Linear, 1, Rational{0}, Rational{1},
                 {Rational{0}}, [](double) { return 1.0; });
  EXPECT_FALSE(one_linear.is_degenerate());
}

TEST(Waveform, ConstantBasisHoldsThePrecedingSample) {
  const auto w = sampled(TemporalBasis::Constant, 0,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return t; });
  EXPECT_DOUBLE_EQ(eval(w, Rational{1, 4}), 0.0);  // still the t=0 sample
  EXPECT_DOUBLE_EQ(eval(w, Rational{3, 4}), 0.5);  // the t=1/2 sample
}

// The Constant basis is the one whose support runs to window_end, because the
// zero-order hold is the representation. Pin that explicitly.
TEST(Waveform, ConstantSupportRunsToTheEndOfTheWindow) {
  const auto w =
      sampled_in(TemporalBasis::Constant, 0, Rational{0}, Rational{2},
                 {Rational{0}, Rational{1}}, [](double t) { return t + 1.0; });
  EXPECT_EQ(w.support_end(), Rational{2});
  EXPECT_DOUBLE_EQ(eval(w, Rational{3, 2}), 2.0);
  EXPECT_DOUBLE_EQ(eval(w, Rational{2}), 2.0);
  double out = 0.0;
  const auto r = w.evaluate(Rational{5, 2}, &out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-OUTSIDE");
}

TEST(Waveform, LinearBasisIsExactOnALinearFunction) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 3.0 * t + 1.0; });
  EXPECT_NEAR(eval(w, Rational{1, 3}), 2.0, 1e-14);
  EXPECT_NEAR(eval(w, Rational{2, 3}), 3.0, 1e-14);
}

TEST(Waveform, DgDegreeTwoIsExactOnAQuadratic) {
  const auto w = sampled(TemporalBasis::DG, 2,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return 2.0 * t * t - t + 3.0; });
  for (const Rational t : {Rational{1, 4}, Rational{1, 3}, Rational{3, 4}}) {
    const double x = t.to_double();
    EXPECT_NEAR(eval(w, t), 2.0 * x * x - x + 3.0, 1e-12) << "at t=" << x;
  }
}

// min(degree, n-1): three samples cannot support a cubic, and the fit must
// degrade to the quadratic through them rather than solve a singular system.
TEST(Waveform, DgClampsItsDegreeToTheSampleCount) {
  const auto w = sampled(TemporalBasis::DG, 4,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return 2.0 * t * t - t + 3.0; });
  EXPECT_NEAR(eval(w, Rational{1, 3}), 2.0 / 9.0 - 1.0 / 3.0 + 3.0, 1e-12);
}

TEST(Waveform, SplineInterpolatesItsSamplesAndIsExactOnALinearFunction) {
  const std::vector<Rational> times = {Rational{0}, Rational{1, 4},
                                       Rational{1, 2}, Rational{3, 4},
                                       Rational{1}};
  const auto w = sampled(TemporalBasis::Spline, 3, times,
                         [](double t) { return 5.0 * t - 2.0; });
  for (const Rational t : times) {
    EXPECT_NEAR(eval(w, t), 5.0 * t.to_double() - 2.0, 1e-13);
  }
  // Natural end conditions are exactly right for a straight line, so the
  // spline reproduces it everywhere.
  EXPECT_NEAR(eval(w, Rational{1, 3}), 5.0 / 3.0 - 2.0, 1e-13);
  EXPECT_NEAR(integral(w, Rational{0}, Rational{1}), 0.5, 1e-13);
}

TEST(Waveform, EvaluateRefusesToExtrapolate) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1, 2}},
                         [](double t) { return t; });
  double     out = 0.0;
  const auto r = w.evaluate(Rational{3, 4}, &out);
  ASSERT_FALSE(r.ok()) << "extrapolation is Extrapolate's job (spec Section 5.1)";
  EXPECT_EQ(r.error().code, "E-WF-OUTSIDE");
}

// Even inside the window: the samples stop at t_last, so a Linear waveform is
// undefined past it. This is the failure mode standing rule 4 is about.
TEST(Waveform, LinearRefusesToExtrapolateInsideTheWindow) {
  const auto w =
      sampled_in(TemporalBasis::Linear, 1, Rational{0}, Rational{1},
                 {Rational{0}, Rational{1, 2}}, [](double t) { return t; });
  EXPECT_EQ(w.support_end(), Rational(1, 2));
  double     out = 0.0;
  const auto r = w.evaluate(Rational{3, 4}, &out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-OUTSIDE");
}

// No backward extrapolation either, for any basis.
TEST(Waveform, EvaluateRefusesTimesBeforeTheFirstSample) {
  for (const auto pair : {std::pair{TemporalBasis::Constant, 0},
                          std::pair{TemporalBasis::Linear, 1},
                          std::pair{TemporalBasis::DG, 1}}) {
    const auto w =
        sampled_in(pair.first, pair.second, Rational{0}, Rational{1},
                   {Rational{1, 2}, Rational{1}}, [](double t) { return t; });
    double     out = 0.0;
    const auto r = w.evaluate(Rational{1, 4}, &out);
    ASSERT_FALSE(r.ok()) << to_cstring(pair.first);
    EXPECT_EQ(r.error().code, "E-WF-OUTSIDE") << to_cstring(pair.first);
  }
}

TEST(Waveform, HandlesMultipleElements) {
  Waveform w{3, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double a[3] = {0.0, 10.0, 20.0};
  const double b[3] = {2.0, 12.0, 22.0};
  ASSERT_TRUE(w.push(Rational{0}, a).ok());
  ASSERT_TRUE(w.push(Rational{1}, b).ok());

  double out[3] = {0, 0, 0};
  ASSERT_TRUE(w.evaluate(Rational{1, 2}, out).ok());
  EXPECT_NEAR(out[0], 1.0, 1e-14);
  EXPECT_NEAR(out[1], 11.0, 1e-14);
  EXPECT_NEAR(out[2], 21.0, 1e-14);
}

// Each element gets its own fit; a shared one would smear them together.
TEST(Waveform, DgFitsEachElementIndependently) {
  Waveform w{2, 3, TemporalBasis::DG, 2};
  w.reset(Rational{0}, Rational{1});
  const auto f0 = [](double t) { return 2.0 * t * t - t + 3.0; };
  const auto f1 = [](double t) { return -t * t + 4.0 * t; };
  for (const Rational t : {Rational{0}, Rational{1, 2}, Rational{1}}) {
    const double v[2] = {f0(t.to_double()), f1(t.to_double())};
    ASSERT_TRUE(w.push(t, v).ok());
  }
  double out[2] = {0, 0};
  ASSERT_TRUE(w.evaluate(Rational{1, 3}, out).ok());
  EXPECT_NEAR(out[0], f0(1.0 / 3.0), 1e-12);
  EXPECT_NEAR(out[1], f1(1.0 / 3.0), 1e-12);
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

// The window is [0, 1] while the samples stop at 1/2: the Constant basis holds
// the last sample to the end of the window, so the integral over [0, 1] is
// 4 * 1/2 + 4 * 1/2 = 4.
TEST(Waveform, IntegratesAConstantExactly) {
  const auto w =
      sampled_in(TemporalBasis::Constant, 0, Rational{0}, Rational{1},
                 {Rational{0}, Rational{1, 2}}, [](double) { return 4.0; });
  EXPECT_NEAR(integral(w, Rational{0}, Rational{1}), 4.0, 1e-14);
}

// Unequal holds, so the answer depends on the piece boundaries being right.
TEST(Waveform, IntegratesAStaircaseExactly) {
  Waveform w{1, 3, TemporalBasis::Constant, 0};
  w.reset(Rational{0}, Rational{1});
  const double a = 1.0, b = 2.0, c = 4.0;
  ASSERT_TRUE(w.push(Rational{0}, &a).ok());
  ASSERT_TRUE(w.push(Rational{1, 4}, &b).ok());
  ASSERT_TRUE(w.push(Rational{3, 4}, &c).ok());
  // 1*(1/4) + 2*(1/2) + 4*(1/4)
  EXPECT_NEAR(integral(w, Rational{0}, Rational{1}), 0.25 + 1.0 + 1.0, 1e-14);
  // A sub-interval straddling two holds: 2*(1/4) + 4*(1/8).
  EXPECT_NEAR(integral(w, Rational{1, 2}, Rational{7, 8}), 0.5 + 0.5, 1e-14);
}

TEST(Waveform, IntegratesALinearFunctionExactly) {
  // int_0^1 (2t) dt = 1
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  EXPECT_NEAR(integral(w, Rational{0}, Rational{1}), 1.0, 1e-14);
}

// The primitive the space-time conservation check depends on.
TEST(Waveform, IntegratesAQuadraticExactlyWithDgTwo) {
  // int_0^1 (3t^2) dt = 1
  const auto w = sampled(TemporalBasis::DG, 2,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return 3.0 * t * t; });
  EXPECT_NEAR(integral(w, Rational{0}, Rational{1}), 1.0, 1e-12);
}

TEST(Waveform, IntegratesASubIntervalCorrectly) {
  // int_{1/4}^{3/4} (2t) dt = 1/2
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  EXPECT_NEAR(integral(w, Rational{1, 4}, Rational{3, 4}), 0.5, 1e-14);
}

// A sub-interval that straddles several sample intervals, clipped at both ends.
TEST(Waveform, IntegratesAClippedMultiPieceLinearInterval) {
  const auto w = sampled(TemporalBasis::Linear, 1,
                         {Rational{0}, Rational{1, 3}, Rational{2, 3},
                          Rational{1}},
                         [](double t) { return 2.0 * t; });
  // int_{1/6}^{5/6} 2t dt = (5/6)^2 - (1/6)^2 = 24/36 = 2/3
  EXPECT_NEAR(integral(w, Rational{1, 6}, Rational{5, 6}), 2.0 / 3.0, 1e-14);
}

TEST(Waveform, IntegrateOverAPointIsZero) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  EXPECT_DOUBLE_EQ(integral(w, Rational{1, 3}, Rational{1, 3}), 0.0);
}

TEST(Waveform, IntegrateRefusesToLeaveTheSupport) {
  const auto w =
      sampled_in(TemporalBasis::Linear, 1, Rational{0}, Rational{1},
                 {Rational{0}, Rational{1, 2}}, [](double t) { return t; });
  double out = 0.0;
  {
    const auto r = w.integrate(Rational{0}, Rational{1}, &out);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-RANGE");
  }
  {  // reversed interval
    const auto r = w.integrate(Rational{1, 2}, Rational{0}, &out);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-WF-RANGE");
  }
}

TEST(Waveform, IntegrateAndEvaluateOnAnEmptyWaveformAreEmptyErrors) {
  Waveform w{1, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  double out = 0.0;
  EXPECT_EQ(w.integrate(Rational{0}, Rational{1}, &out).error().code,
            "E-WF-EMPTY");
  EXPECT_EQ(w.mean(&out).error().code, "E-WF-EMPTY");
}

TEST(Waveform, MeanIsTheWindowIntegralOverItsLength) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  double out = 0.0;
  ASSERT_TRUE(w.mean(&out).ok());
  EXPECT_NEAR(out, 1.0, 1e-14);
}

TEST(Waveform, MeanRefusesAZeroLengthWindow) {
  Waveform w{1, 1, TemporalBasis::Constant, 0};
  w.reset(Rational{3}, Rational{3});
  const double v = 5.0;
  ASSERT_TRUE(w.push(Rational{3}, &v).ok());
  double     out = 0.0;
  const auto r = w.mean(&out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-WINDOW");
}

// You cannot average over time you did not sample.
TEST(Waveform, MeanRefusesWhenTheSamplesDoNotSpanTheWindow) {
  const auto w =
      sampled_in(TemporalBasis::Linear, 1, Rational{0}, Rational{1},
                 {Rational{0}, Rational{1, 2}}, [](double t) { return t; });
  double     out = 0.0;
  const auto r = w.mean(&out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-RANGE");
}

// Time is exact: a window offset by a huge integer must not degrade the fit,
// because every conversion to double happens after an exact Rational
// subtraction against window_begin.
TEST(Waveform, LargeAbsoluteTimesDoNotDegradeTheFit) {
  const std::int64_t t0 = 1'000'000'000;
  Waveform           w{1, 3, TemporalBasis::DG, 2};
  w.reset(Rational{t0}, Rational{t0 + 1});
  const auto f = [](double s) { return 2.0 * s * s - s + 3.0; };
  for (const Rational off : {Rational{0}, Rational{1, 2}, Rational{1}}) {
    const double v = f(off.to_double());
    ASSERT_TRUE(w.push(Rational{t0} + off, &v).ok()) << off.to_string();
  }
  EXPECT_NEAR(eval(w, Rational{t0} + Rational{1, 3}), f(1.0 / 3.0), 1e-12);
  EXPECT_NEAR(integral(w, Rational{t0}, Rational{t0 + 1}),
              2.0 / 3.0 - 0.5 + 3.0, 1e-12);
}

// ---------------------------------------------------------------------------
// The per-element allocation fence.
// ---------------------------------------------------------------------------
//
// The shared factorization (`PolynomialFitPlan`/`SplineMomentPlan`) is
// hoisted out of the per-element loop, and `solve_polynomial_fit`/
// `solve_cubic_spline_moments` use the `_into` entry points with caller
// scratch rather than allocating their own `rhs`/`c` (resp. `r`/`m`) on
// every element. The fence: one
// `evaluate()` call's allocation count must not depend on the element count,
// since every allocation left in the per-element loop would make it scale
// with `elements_`.
namespace {
std::size_t count_allocations_for_one_evaluate(TemporalBasis basis, int degree,
                                               std::size_t elements) {
  Waveform w{elements, 4, basis, degree};
  w.reset(Rational{0}, Rational{3});
  std::vector<double> vals(elements);
  for (int i = 0; i < 4; ++i) {
    for (std::size_t e = 0; e < elements; ++e) {
      vals[e] = static_cast<double>(i * i) + 0.1 * static_cast<double>(e);
    }
    EXPECT_TRUE(w.push(Rational{i}, vals.data()).ok());
  }
  std::vector<double> out(elements);
  const AllocationCounter counter;
  EXPECT_TRUE(w.evaluate(Rational{3, 2}, out.data()).ok());
  // integrate_dg / integrate_spline get the same hoist as the evaluate
  // pair, so they are under this fence too: a per-element buffer
  // reintroduced there would otherwise pass.
  EXPECT_TRUE(w.integrate(Rational{1, 2}, Rational{5, 2}, out.data()).ok());
  return counter.count();
}
}  // namespace

TEST(Waveform, DgAndSplineEvaluateAllocationIsIndependentOfElementCount) {
  for (const auto [basis, degree] :
       {std::pair{TemporalBasis::DG, 2}, std::pair{TemporalBasis::Spline, 3}}) {
    const std::size_t few  = count_allocations_for_one_evaluate(basis, degree, 2);
    const std::size_t many = count_allocations_for_one_evaluate(basis, degree, 400);
    EXPECT_GT(few, 0u) << "the counter itself is not wired up";
    EXPECT_EQ(few, many)
        << "basis=" << static_cast<int>(basis)
        << ": allocation count scales with element count, so a per-element "
           "buffer is still being allocated inside the loop";
  }
}
