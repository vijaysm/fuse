#include <gtest/gtest.h>
#include <fuse/linalg/least_squares.hpp>
#include <fuse/num/accelerator.hpp>
#include <fuse/base/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::Accelerator;
using fuse::FilterKind;
using fuse::make_accelerator;

// ---------------------------------------------------------------------------
// Allocation counting, as in test/verification/iteration_test.cpp and
// test/unit/runner_test.cpp. Spec Section 2 layer 3 requires the sweep loop to
// allocate nothing, and an accelerated implicit coupling group runs the accelerator inside that
// loop, so "the accelerator allocates nothing per update" is part of the same
// requirement. Replacing the global operator new is the only way to MEASURE
// that rather than assert it in a comment; SteadyStateUpdateAllocatesNothing
// is the test, and it counted 56 allocations over 56 Aitken updates before the
// buffers were hoisted.
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
  // exceptions nor see g_counting/g_allocations, host globals. Same shape
  // as waveform_test.cpp's overrides.
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

/// Fixed-point map x -> G(x) with a known contraction factor. Every
/// accelerator must converge on this; the interesting question is how fast.
struct Contraction {
  double factor;
  std::vector<double> fixed_point;

  std::vector<double> apply(const std::vector<double>& x) const {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      out[i] = fixed_point[i] + factor * (x[i] - fixed_point[i]);
    }
    return out;
  }
};

/// Iterate to convergence, returning the iteration count (-1 if it did not).
int iterate_to_convergence(fuse::Accelerator& accel, const Contraction& g,
                           double tol, int max_iters) {
  std::vector<double> x(g.fixed_point.size(), 0.0);
  for (int k = 0; k < max_iters; ++k) {
    const std::vector<double> gx = g.apply(x);
    double residual = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double d = gx[i] - x[i];
      residual += d * d;
    }
    if (std::sqrt(residual) < tol) return k;

    std::vector<double> next(x.size());
    EXPECT_TRUE(accel.update(x.data(), gx.data(), next.data(), x.size(), k).ok());
    x = next;
  }
  return -1;
}

/// A general affine fixed-point map G(x) = p + A (x - p).
///
/// The Contraction above is A = factor * I, which every method in the family
/// solves in two iterations because its error only ever occupies a single
/// direction. That hides the whole point of the ladder, so the ordering tests
/// below use this instead, with A dense and with a spread of eigenvalues.
struct LinearMap {
  std::size_t         n = 0;
  std::vector<double> a;  ///< n x n, row major
  std::vector<double> p;

  std::vector<double> apply(const std::vector<double>& x) const {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
      double s = p[i];
      for (std::size_t j = 0; j < n; ++j) s += a[i * n + j] * (x[j] - p[j]);
      out[i] = s;
    }
    return out;
  }
};

/// A = H D H with H = I - 2 v v^T / (v^T v) a Householder reflector, so A is
/// dense and symmetric with exactly the eigenvalues D. `v` is deliberately not
/// parallel to `p`: with v == p the reflector maps the initial error onto its
/// own negative and the "dense" problem degenerates into a mirror image of the
/// diagonal one.
LinearMap dense_map(const std::vector<double>& d, const std::vector<double>& p,
                    const std::vector<double>& v) {
  const std::size_t n = d.size();
  double vtv = 0.0;
  for (double e : v) vtv += e * e;
  std::vector<double> h(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      h[i * n + j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vtv;
    }
  }
  std::vector<double> hd(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) hd[i * n + j] = h[i * n + j] * d[j];
  }
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n; ++k) s += hd[i * n + k] * h[k * n + j];
      a[i * n + j] = s;
    }
  }
  return LinearMap{n, std::move(a), p};
}

LinearMap diagonal_map(const std::vector<double>& d,
                       const std::vector<double>& p) {
  const std::size_t n = d.size();
  std::vector<double> a(n * n, 0.0);
  for (std::size_t i = 0; i < n; ++i) a[i * n + i] = d[i];
  return LinearMap{n, std::move(a), p};
}

/// Iteration count on a LinearMap; -1 if it did not converge, -2 if update()
/// reported an error. Starts from x = 0 every time, so a second call on the
/// same accelerator is a second time window over the same problem.
int iterate_linear(Accelerator& accel, const LinearMap& g, double tol,
                   int max_iters) {
  std::vector<double> x(g.n, 0.0);
  for (int k = 0; k < max_iters; ++k) {
    const std::vector<double> gx = g.apply(x);
    double residual = 0.0;
    for (std::size_t i = 0; i < g.n; ++i) {
      const double d = gx[i] - x[i];
      residual += d * d;
    }
    if (std::sqrt(residual) < tol) return k;

    std::vector<double> next(g.n);
    const auto r = accel.update(x.data(), gx.data(), next.data(), g.n, k);
    if (!r.ok()) return -2;
    x = std::move(next);
  }
  return -1;
}

/// The one problem the ladder is measured on. Eigenvalues 0.95 (slow), 0.9,
/// 0.8, 0.5 and -0.7 (oscillating), in a basis where none of them is aligned
/// with a coordinate axis.
LinearMap ladder_problem() {
  return dense_map({0.95, 0.9, 0.8, 0.5, -0.7}, {1.0, 2.0, 3.0, 4.0, 5.0},
                   {1.0, -2.0, 0.5, 3.0, -1.5});
}

std::unique_ptr<Accelerator> must_make(const AccelConfig& cfg) {
  auto made = make_accelerator(cfg);
  EXPECT_TRUE(made.ok()) << (made.ok() ? std::string{} : made.error().message);
  return std::move(made.value());
}

constexpr double kTol      = 1e-10;
constexpr int    kMaxIters = 5000;

}  // namespace

// ---------------------------------------------------------------------------
// Semantics of each kind
// ---------------------------------------------------------------------------

TEST(Accelerator, ConstantRelaxationBlends) {
  AccelConfig cfg{.kind = AccelKind::Constant, .omega = 0.25};
  auto a = make_accelerator(cfg);
  ASSERT_TRUE(a.ok()) << a.error().message;

  const std::vector<double> old_x{0.0, 10.0};
  const std::vector<double> new_x{10.0, 0.0};
  std::vector<double> out(2, 0.0);
  ASSERT_TRUE(a.value()->update(old_x.data(), new_x.data(), out.data(), 2, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 2.5);
  EXPECT_DOUBLE_EQ(out[1], 7.5);
}

TEST(Accelerator, NoneKindIsAPassThrough) {
  auto a = must_make(AccelConfig{.kind = AccelKind::None});
  const std::vector<double> old_x{1.0};
  const std::vector<double> new_x{5.0};
  std::vector<double> out(1, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 1, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 5.0) << "plain Picard takes the new value";
}

TEST(Accelerator, EveryKindIsConstructible) {
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    const auto a = make_accelerator(AccelConfig{.kind = k});
    ASSERT_TRUE(a.ok()) << "kind " << static_cast<int>(k);
    EXPECT_FALSE(a.value()->name().empty());
    EXPECT_STREQ(fuse::to_cstring(k), a.value()->name().c_str());
  }
  for (const FilterKind f : {FilterKind::None, FilterKind::QR1,
                             FilterKind::QR2, FilterKind::QR3}) {
    EXPECT_STRNE(fuse::to_cstring(f), "?");
  }
}

TEST(Accelerator, ConstantRelaxationConvergesOnAContraction) {
  AccelConfig cfg{.kind = AccelKind::Constant, .omega = 1.0};
  auto a = must_make(cfg);
  const Contraction g{.factor = 0.5, .fixed_point = {1.0, 2.0, 3.0}};
  const int iters = iterate_to_convergence(*a, g, 1e-10, 200);
  // Pinned, not merely bounded: with omega == 1 this is plain Picard, whose
  // error is exactly 0.5^k * ||x_0 - p||, so the crossing iteration is a
  // property of the arithmetic and not of the machine.
  EXPECT_EQ(iters, 35);
}

// ---------------------------------------------------------------------------
// The ladder. Counting iterations is the only honest test here: a subtly wrong
// accelerator still converges, just slowly, and looks like a hard problem.
// ---------------------------------------------------------------------------

TEST(Accelerator, AitkenBeatsConstantRelaxation) {
  const Contraction g{.factor = 0.9, .fixed_point = {1.0, 2.0}};  // slow

  auto constant = must_make(AccelConfig{.kind = AccelKind::Constant, .omega = 1.0});
  auto aitken = must_make(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.5});

  const int n_const = iterate_to_convergence(*constant, g, 1e-10, 500);
  const int n_aitken = iterate_to_convergence(*aitken, g, 1e-10, 500);
  EXPECT_EQ(n_const, 205);
  // A(x) = 0.9 * I, so the error lives in one direction and Aitken's single
  // dynamic omega is exact after one secant pair.
  EXPECT_EQ(n_aitken, 2);
  EXPECT_LT(n_aitken, n_const)
      << "aitken " << n_aitken << " vs constant " << n_const;
}

TEST(Accelerator, AitkenRejectsANonPositiveCandidateAndRestartsToOmega0) {
  // Window seed track, Stage E (KNOWN_ISSUES.md D76): a crafted 2-sweep
  // scalar sequence whose UNGUARDED secant candidate is negative.
  // Sweep 0: no history, w = initial_relaxation = 0.5, r_prev = 1 - 0 = 1.
  // Sweep 1: r = 7 - 5 = 2, d = r - r_prev = 1, num = r_prev*d = 1,
  // den = d*d = 1, candidate = -omega_*num/den = -0.5*1/1 = -0.5 -- the
  // formula this track's own D76 entry measured AitkenAccel::update()
  // used to accept outright (the window-3 stagnation: a negative omega
  // at sweep 3, never recovering). The guarded formula rejects a
  // non-positive candidate and restarts to omega0_, so sweep 1's own
  // output is 5 + 0.5*2 = 6, not 5 + (-0.5)*2 = 4. Every value here is
  // exactly representable and every operation a multiply or divide by a
  // power of two, so this is checked bit-for-bit (EXPECT_EQ), not to a
  // tolerance.
  auto a = must_make(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.5});

  const double old0 = 0.0, new0 = 1.0;
  double out0 = 0.0;
  ASSERT_TRUE(a->update(&old0, &new0, &out0, 1, 0).ok());
  EXPECT_EQ(out0, 0.5);

  const double old1 = 5.0, new1 = 7.0;
  double out1 = 0.0;
  ASSERT_TRUE(a->update(&old1, &new1, &out1, 1, 1).ok());
  EXPECT_EQ(out1, 6.0)
      << "unguarded formula would give 5 + (-0.5)*2 = 4; guarded must "
         "reject the negative candidate and restart to omega0_ = 0.5, "
         "giving 5 + 0.5*2 = 6";
}

TEST(Accelerator, IqnIlsConvergesFastOnALinearProblem) {
  const Contraction g{.factor = 0.95, .fixed_point = {1.0, 2.0, 3.0, 4.0}};
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                 .initial_relaxation = 0.1,
                                 .reuse = 8,
                                 .filter = FilterKind::QR2,
                                 .limit = 1e-2});
  const int iters = iterate_to_convergence(*a, g, 1e-10, 100);
  // One under-relaxed step, one secant pair, exact. Far fewer than the
  // 1/(1 - 0.95) = 20 of the underlying contraction.
  EXPECT_EQ(iters, 2) << "IQN-ILS took " << iters << " iterations";
}

TEST(Accelerator, LadderIsMonotoneOnADenseLinearProblem) {
  const LinearMap g = ladder_problem();

  auto picard = must_make(AccelConfig{.kind = AccelKind::None});
  auto constant = must_make(AccelConfig{.kind = AccelKind::Constant, .omega = 1.0});
  auto secant = must_make(
      AccelConfig{.kind = AccelKind::Secant, .initial_relaxation = 0.5});
  auto aitken = must_make(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.5});
  auto ils = must_make(AccelConfig{
      .kind = AccelKind::IQN_ILS, .initial_relaxation = 0.5, .reuse = 8});
  auto imvj = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});

  const int n_picard   = iterate_linear(*picard, g, kTol, kMaxIters);
  const int n_constant = iterate_linear(*constant, g, kTol, kMaxIters);
  const int n_secant   = iterate_linear(*secant, g, kTol, kMaxIters);
  const int n_aitken   = iterate_linear(*aitken, g, kTol, kMaxIters);
  const int n_ils      = iterate_linear(*ils, g, kTol, kMaxIters);
  const int n_imvj     = iterate_linear(*imvj, g, kTol, kMaxIters);

  // Pinned counts for picard/constant/secant/ils/imvj. These were measured
  // to be identical under -O0, -O1, -O3 -DNDEBUG, -O2 -ffp-contract=fast and
  // ASan+UBSan; the kernels' ascending-index accumulation convention is what
  // makes them reproducible for a given buffer. MEASURED 2026-10-12
  // (FpPolicy track, KNOWN_ISSUES.md D30): that list does NOT cover
  // -ffp-contract=off (the flag did not exist when it was written) -- under
  // FUSE_FP_REPRODUCIBLE=ON these five counts were checked directly and are
  // unchanged (382, 382, 269, 7, 7); only n_aitken, below, moves between the
  // two contraction policies. The list above shows these five counts are
  // insensitive to several compiler/optimizer choices; it does not show, and
  // was never claimed to show, insensitivity to contraction policy
  // specifically -- that had to be measured separately, and was.
  EXPECT_EQ(n_picard, 382);
  EXPECT_EQ(n_constant, 382) << "omega == 1 is exactly Picard";
  EXPECT_EQ(n_secant, 269);
  // n_aitken is deliberately NOT pinned to a literal count. MEASURED
  // 2026-10-12 (FpPolicy track, KNOWN_ISSUES.md D30): the literal this test
  // used to assert (56) was a rounding-dependent number, not a robust
  // property of the method. At x0 = {0,0,0,0,0} exactly, n_aitken = 56 under
  // the compiler's default (fused) contraction and 57 under
  // FUSE_FP_REPRODUCIBLE=ON (-ffp-contract=off); the two residual histories
  // first diverge at iteration k=1, by 2 ulp (1.8748738283994892 vs
  // 1.8748738283994888). A 24-way perturbation of x0 by 1 ulp of 1.0
  // (2^-52, an independent +/- sign on each of the 5 components -- the
  // smallest perturbation for which "1 ulp" is physically meaningful here,
  // since x0=0 itself admits no such perturbation) gives, in BOTH builds, a
  // spread across the same four values:
  //   contracted (default):       {55: 3, 56: 12, 57: 7, 59: 2}, mean 56.42
  //   FUSE_FP_REPRODUCIBLE=ON:     {55: 3, 56: 14, 57: 6, 59: 1}, mean 56.25
  // 56 is the single most common value in BOTH distributions, and the
  // 56->57 shift between the two unperturbed baselines sits well inside
  // this natural, physically-irrelevant scatter -- not a systematic effect
  // of the contraction policy. The property this test is actually named
  // for, the ladder's monotonicity, is asserted below instead, and holds
  // regardless of which of {55,56,57,59} n_aitken lands on.
  // 7, not the 6 of an unfiltered rank-5 fit, and the extra iteration is the
  // QR2 filter doing exactly what it is for. These accelerators take
  // AccelConfig's default filter, QR2 at limit = 1e-2, so a column is dropped
  // once less than 1% of it is orthogonal to the newer columns. Measured
  // remainder fractions of the oldest column on the unfiltered waveform:
  // 3.5e-1 at iteration 2, 4.5e-2 at 3, 1.8e-3 at 4. At iteration 4 the first
  // secant pair is therefore dropped -- and on this exactly linear 5-dof
  // problem that 0.18% is precisely the fifth direction still missing from the
  // Jacobian, so the exact fit arrives one iteration later. Filtering costs
  // accuracy on a clean linear problem and buys conditioning on a dirty one;
  // this is the cost side, pinned so that it stays visible.
  EXPECT_EQ(n_ils, 7) << "5 unknowns: one under-relaxed step, a rank-5 fit, and "
                         "one iteration to replace the column QR2 dropped";
  EXPECT_EQ(n_imvj, 7) << "first window, so IMVJ has nothing retained yet";

  // Monotone as a ladder, asserted as an ordering rather than as six
  // independent facts. n_aitken's own value is not pinned (see above), but
  // its POSITION in this ordering is: across all 48 measurements behind the
  // comment above (24 perturbations x 2 contraction policies) n_aitken
  // stayed in {55,56,57,59} -- far below n_secant=269 and far above
  // n_ils=7 -- so these relations hold regardless of which rounding-
  // dependent value it lands on.
  EXPECT_LT(n_secant, n_picard);
  EXPECT_LT(n_aitken, n_secant);
  EXPECT_LT(n_ils, n_aitken);
  EXPECT_LE(n_imvj, n_ils);
}

TEST(Accelerator, SteffensenIsExactOnADiagonalProblemAndBasisDependent) {
  const std::vector<double> d{0.95, 0.9, 0.8, 0.5, -0.7};
  const std::vector<double> p{1.0, 2.0, 3.0, 4.0, 5.0};

  auto on_diagonal = must_make(
      AccelConfig{.kind = AccelKind::Steffensen, .initial_relaxation = 0.5});
  const int n_diag =
      iterate_linear(*on_diagonal, diagonal_map(d, p), kTol, kMaxIters);
  EXPECT_EQ(n_diag, 2) << "a diagonal secant is exact on a diagonal Jacobian";

  // Steffensen is the one member of the family that is not orthogonally
  // equivariant: it models the Jacobian by its diagonal, so on a dense
  // Jacobian with the same spectrum it is worse than plain Picard. That is a
  // property of the method, not a defect -- but MEASURED 2026-10-12
  // (FpPolicy track, KNOWN_ISSUES.md D30), the specific iteration count this
  // test used to pin (681) is NOT a robust property to assert: Steffensen's
  // iteration on this dense, basis-rotated problem sits at or very near a
  // chaotic basin boundary. Residual histories under the two contraction
  // policies track each other to 5-6 significant figures through k<=35, then
  // both show intermittent large residual bursts from k~60-165 (a signature
  // of a trajectory near a chaotic repeller/saddle); the default
  // (contracted) build's bursts happen to relax back into convergence by
  // k=681, while FUSE_FP_REPRODUCIBLE=ON's burst near k=165 does not, and
  // the residual instead diverges (to ~1e+137 by k=4400). A 24-way
  // perturbation of the initial vector by 1 ulp of 1.0 (same convention as
  // LadderIsMonotoneOnADenseLinearProblem, above) shows this is not specific
  // to the ON build: 4 of 24 (16.7%) perturbed runs ALSO diverge
  // (non-convergent, -1) in the DEFAULT contracted build, and 8 of 24
  // (33%) diverge under ON; the convergent runs ranged 488-2366 iterations
  // across both policies -- nowhere near a tight cluster around 681. So 681
  // was one outcome of a dynamical system whose qualitative behaviour
  // (converge vs. diverge) is controlled by perturbations far smaller than
  // any physically meaningful input uncertainty, not a measured, reproducible
  // property of the method. What IS robust, and is asserted below instead,
  // is the property the test is named for: basis dependence -- Steffensen on
  // this dense problem either fails to converge at all, or takes far more
  // than the n_diag=2 iterations the SAME spectrum costs it in a basis where
  // it is exact. K=20 (giving the n_diag=2 case a factor of 40) is chosen to
  // sit comfortably below the smallest observed convergent ratio across all
  // 36 finite measurements above (488/2=244, so K=20 carries more than a
  // 12x margin below the weakest case actually measured) while remaining
  // many times larger than any ratio a correctly-functioning, well-
  // conditioned secant-based method could plausibly reach by chance -- i.e.
  // it does not depend on the single value 681 that used to be pinned here.
  auto on_dense = must_make(
      AccelConfig{.kind = AccelKind::Steffensen, .initial_relaxation = 0.5});
  const int n_dense = iterate_linear(*on_dense, ladder_problem(), kTol, kMaxIters);
  constexpr int kBasisDependenceFactor = 20;  // see measured margin above
  EXPECT_TRUE(n_dense == -1 || n_dense >= kBasisDependenceFactor * n_diag)
      << "n_dense = " << n_dense << ", n_diag = " << n_diag
      << " -- Steffensen on a dense, basis-rotated problem should either not "
         "converge within the iteration budget, or take at least "
      << kBasisDependenceFactor << "x as long as it does when the basis is "
         "aligned with its own diagonal model, not a specific pinned count";
}

TEST(Accelerator, ImvjBeatsIqnIlsOnLaterWindows) {
  const LinearMap g = ladder_problem();
  auto imvj = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
  auto ils = must_make(AccelConfig{
      .kind = AccelKind::IQN_ILS, .initial_relaxation = 0.5, .reuse = 8});

  std::vector<int> imvj_counts;
  std::vector<int> ils_counts;
  for (int window = 0; window < 4; ++window) {
    imvj_counts.push_back(iterate_linear(*imvj, g, kTol, kMaxIters));
    ils_counts.push_back(iterate_linear(*ils, g, kTol, kMaxIters));
    imvj->reset();
    ils->reset();
  }

  // 7 for the first window under the default QR2 filter; see
  // LadderIsMonotoneOnADenseLinearProblem for the measured reason. The
  // retained Jacobian is unaffected: the column QR2 dropped is folded back in
  // through the window it was collected in, so later windows still cost one.
  EXPECT_EQ(imvj_counts[0], 7);
  EXPECT_EQ(imvj_counts[1], 1) << "the retained inverse Jacobian is exact for "
                                  "a linear problem, so one step suffices";
  EXPECT_EQ(imvj_counts[2], 1);
  EXPECT_EQ(imvj_counts[3], 1);
  // IQN-ILS discards everything at the window boundary, so every window costs
  // the same. That difference is the whole content of IMVJ.
  EXPECT_EQ(ils_counts[0], 7);
  EXPECT_EQ(ils_counts[1], 7);
  EXPECT_EQ(ils_counts[2], 7);
  EXPECT_EQ(ils_counts[3], 7);
}

// ---------------------------------------------------------------------------
// No history on the first iteration
// ---------------------------------------------------------------------------

TEST(Accelerator, FirstIterationIsUnderRelaxed) {
  // No Jacobian estimate exists yet, so iteration 0 must use
  // initial_relaxation rather than attempting a quasi-Newton step.
  AccelConfig cfg{.kind = AccelKind::IQN_ILS, .initial_relaxation = 0.25};
  auto a = must_make(cfg);
  const std::vector<double> old_x{0.0};
  const std::vector<double> new_x{4.0};
  std::vector<double> out(1, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 1, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 1.0);   // 0 + 0.25 * (4 - 0)
}

TEST(Accelerator, EveryHistoryBasedKindDegradesToRelaxationWithoutHistory) {
  // Iteration 0 of a window has no residual difference, so every method that
  // consumes one must fall back to initial_relaxation. Asserted for all of
  // them, not just IQN-ILS.
  for (const AccelKind k : {AccelKind::Aitken, AccelKind::Steffensen,
                            AccelKind::Secant, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k, .initial_relaxation = 0.25});
    const std::vector<double> old_x{2.0, -1.0};
    const std::vector<double> new_x{6.0, 3.0};
    std::vector<double> out(2, 0.0);
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 2, 0).ok())
        << fuse::to_cstring(k);
    EXPECT_DOUBLE_EQ(out[0], 3.0) << fuse::to_cstring(k);   //  2 + 0.25 * 4
    EXPECT_DOUBLE_EQ(out[1], 0.0) << fuse::to_cstring(k);   // -1 + 0.25 * 4
    EXPECT_EQ(a->columns(), 0u) << fuse::to_cstring(k);
  }
}

TEST(Accelerator, ResetIsEquivalentToPassingIterationZero) {
  const Contraction g{.factor = 0.9, .fixed_point = {1.0, 2.0, 3.0}};
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                 .initial_relaxation = 0.3, .reuse = 4});
  std::vector<double> x{0.0, 0.0, 0.0};
  for (int k = 0; k < 3; ++k) {
    const std::vector<double> gx = g.apply(x);
    std::vector<double> next(3);
    ASSERT_TRUE(a->update(x.data(), gx.data(), next.data(), 3, k).ok());
    x = next;
  }
  ASSERT_GT(a->columns(), 0u);

  // Passing iteration 0 again must clear the window exactly as reset() does,
  // otherwise a stale column from the previous window silently survives.
  const std::vector<double> old_x{0.0, 0.0, 0.0};
  const std::vector<double> new_x{4.0, 4.0, 4.0};
  std::vector<double> out(3, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 3, 0).ok());
  EXPECT_EQ(a->columns(), 0u);
  EXPECT_DOUBLE_EQ(out[0], 1.2);  // 0 + 0.3 * 4
}

// ---------------------------------------------------------------------------
// History depth and rank deficiency
// ---------------------------------------------------------------------------

TEST(Accelerator, ResetClearsHistory) {
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS});
  // The inputs must actually change from iteration to iteration. Feeding the
  // same (x_old, x_new) pair every time makes every residual difference
  // exactly zero, and a zero column carries no secant information -- see
  // AZeroResidualDifferenceIsNotStoredAsAColumn.
  std::vector<double> out(2, 0.0);
  for (int k = 0; k < 3; ++k) {
    const std::vector<double> old_x{0.1 * k, -0.2 * k};
    const std::vector<double> new_x{1.0 + 0.3 * k, 1.0 - 0.4 * k};
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 2, k).ok());
  }
  EXPECT_GT(a->columns(), 0u);
  a->reset();
  EXPECT_EQ(a->columns(), 0u);
}

// REVIEW.md R-45: `cols_` (bounded by `cfg.reuse`, default 8) is completely
// independent of `n` (the interface size, legitimately as small as 1), so any
// coupling with a small interface and the default reuse limit reaches
// `cols_ > n` on the very first window -- exactly the shape
// `linalg::least_squares_into` is called with, un-preconditioned by
// `check_shape`'s `m >= k`. This drives that shape at the Accelerator level
// (linalg-level coverage is `LeastSquares.IntoAgreesWithBuiltinWhenMIsLessThanK`)
// and asserts `update()` still succeeds and returns a finite iterate.
TEST(Accelerator, HandlesAnInterfaceSmallerThanTheReuseLimit) {
  AccelConfig cfg{.kind = AccelKind::IQN_ILS, .filter = FilterKind::None};
  auto a = must_make(cfg);  // default reuse = 8
  std::vector<double> old_x(1, 0.0), out(1, 0.0);
  for (int k = 0; k < 4; ++k) {
    // Distinct residuals each iteration, so the difference vectors stored as
    // columns are not the degenerate zero-column case.
    const std::vector<double> new_x = {
        1.0 - std::pow(0.3, k + 1) + 0.01 * static_cast<double>(k)};
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 1, k).ok())
        << "at iteration " << k;
    old_x = out;
    EXPECT_TRUE(std::isfinite(out[0])) << "at iteration " << k;
  }
  EXPECT_GT(a->columns(), 1u)
      << "the test must actually reach cols_ > n=1 to exercise R-45's path";
}

TEST(Accelerator, RespectsTheReuseLimit) {
  AccelConfig cfg{.kind = AccelKind::IQN_ILS, .reuse = 3,
                  .filter = FilterKind::None};
  auto a = must_make(cfg);
  std::vector<double> old_x(4, 0.0), out(4, 0.0);
  for (int k = 0; k < 10; ++k) {
    std::vector<double> new_x(4);
    for (std::size_t i = 0; i < 4; ++i) {
      new_x[i] = static_cast<double>(k) * 0.1 + static_cast<double>(i);
    }
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 4, k).ok());
    old_x = out;
    EXPECT_LE(a->columns(), 3u) << "at iteration " << k;
  }
  EXPECT_EQ(a->columns(), 3u) << "reuse limit must bound the column count";
}

TEST(Accelerator, ColumnCountIsBoundedOverAVeryLongIteration) {
  // The history depth policy is a memory policy: unbounded columns are both a
  // slow leak and a growing O(k^2) cost per iteration.
  const LinearMap g = ladder_problem();
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_ILS, .initial_relaxation = 0.5, .reuse = 4});
  std::vector<double> x(g.n, 0.0);
  for (int k = 0; k < 500; ++k) {
    const std::vector<double> gx = g.apply(x);
    std::vector<double> next(g.n);
    ASSERT_TRUE(a->update(x.data(), gx.data(), next.data(), g.n, k).ok())
        << "at iteration " << k;
    x = next;
    ASSERT_LE(a->columns(), 4u) << "at iteration " << k;
  }
}

TEST(Accelerator, AZeroResidualDifferenceIsNotStoredAsAColumn) {
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                 .initial_relaxation = 0.25});
  const std::vector<double> old_x{0.0, 0.0};
  const std::vector<double> new_x{1.0, 1.0};
  std::vector<double> out(2, 0.0);
  for (int k = 0; k < 5; ++k) {
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 2, k).ok());
    // A stalled iteration produces Delta r == 0. The column is information
    // free and is rejected rather than spending a history slot on something
    // the rank truncation is guaranteed to discard.
    EXPECT_EQ(a->columns(), 0u) << "at iteration " << k;
    EXPECT_DOUBLE_EQ(out[0], 0.25);
    EXPECT_TRUE(std::isfinite(out[0]));
    EXPECT_TRUE(std::isfinite(out[1]));
  }
}

TEST(Accelerator, IqnIlsStaysStableLongAfterConvergence) {
  // Rank deficiency is the normal case, not an edge case: once the iteration
  // has converged, every new column is numerically dependent on the old ones.
  // Continuing to iterate must neither fail nor move the iterate away.
  const LinearMap g = ladder_problem();
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_ILS, .initial_relaxation = 0.5, .reuse = 8});
  std::vector<double> x(g.n, 0.0);
  for (int k = 0; k < 200; ++k) {
    const std::vector<double> gx = g.apply(x);
    std::vector<double> next(g.n);
    ASSERT_TRUE(a->update(x.data(), gx.data(), next.data(), g.n, k).ok())
        << "at iteration " << k;
    x = std::move(next);
  }
  for (std::size_t i = 0; i < g.n; ++i) {
    EXPECT_NEAR(x[i], g.p[i], 1e-9) << "component " << i;
  }
}

TEST(Accelerator, ChangingTheVectorLengthClearsTheHistory) {
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                 .initial_relaxation = 0.25, .reuse = 8});
  std::vector<double> out3(3, 0.0);
  for (int k = 0; k < 3; ++k) {
    const std::vector<double> old_x{0.1 * k, 0.2 * k, 0.3 * k};
    const std::vector<double> new_x{1.0 + k, 2.0 - k, 3.0 * k};
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out3.data(), 3, k).ok());
  }
  ASSERT_GT(a->columns(), 0u);

  // A different length means a different space; the stored columns cannot be
  // reinterpreted, so they must go. iteration is deliberately non-zero here so
  // that the length check, not the window check, is what clears them.
  const std::vector<double> old_x{1.0, 1.0};
  const std::vector<double> new_x{5.0, 5.0};
  std::vector<double> out2(2, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out2.data(), 2, 7).ok());
  EXPECT_EQ(a->columns(), 0u);
  EXPECT_DOUBLE_EQ(out2[0], 2.0);  // 1 + 0.25 * 4
}

TEST(Accelerator, EveryKindHandlesAChangeOfVectorLengthBothWays) {
  // The same rule for the whole family, and in both directions. It is stated
  // separately from the test above because the buffers behind it are now
  // members that are GROWN AND NEVER SHRUNK: a shrinking n leaves a longer
  // buffer holding the previous window's numbers, so "is the stored history
  // usable" can no longer be answered by comparing a size against n, and a
  // kind that still did would silently mix two spaces. Aitken and Steffensen
  // did exactly that before the buffers were hoisted.
  for (const AccelKind k : {AccelKind::Aitken, AccelKind::Steffensen,
                            AccelKind::Secant, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{
        .kind = k, .initial_relaxation = 0.25, .reuse = 8});
    const char* who = fuse::to_cstring(k);

    std::vector<double> out3(3, 0.0);
    for (int step = 0; step < 4; ++step) {
      const std::vector<double> old_x{0.1 * step, 0.2 * step, 0.3 * step};
      const std::vector<double> new_x{1.0 + step, 2.0 - step, 3.0 * step};
      ASSERT_TRUE(
          a->update(old_x.data(), new_x.data(), out3.data(), 3, step).ok())
          << who;
    }
    // A window boundary at length 3, so IMVJ has something retained to lose.
    a->reset();
    if (k == AccelKind::IQN_IMVJ) ASSERT_EQ(a->retained_blocks(), 1u) << who;

    // Shrink, at a non-zero iteration so the length check and not the window
    // check is what has to fire. With no usable history every kind must fall
    // back to initial_relaxation exactly: 1 + 0.25 * (5 - 1).
    const std::vector<double> o2{1.0, 1.0}, n2{5.0, 5.0};
    std::vector<double>       out2(2, 0.0);
    ASSERT_TRUE(a->update(o2.data(), n2.data(), out2.data(), 2, 7).ok()) << who;
    EXPECT_EQ(a->columns(), 0u) << who;
    EXPECT_EQ(a->retained_blocks(), 0u)
        << who << ": an inverse Jacobian on a 3-vector says nothing about a "
                  "2-vector";
    for (const double v : out2) EXPECT_DOUBLE_EQ(v, 2.0) << who;

    // And grow again, still mid-window: 2 + 0.25 * (10 - 2).
    const std::vector<double> o3{2.0, 2.0, 2.0}, n3{10.0, 10.0, 10.0};
    ASSERT_TRUE(a->update(o3.data(), n3.data(), out3.data(), 3, 8).ok()) << who;
    EXPECT_EQ(a->columns(), 0u) << who;
    for (const double v : out3) EXPECT_DOUBLE_EQ(v, 4.0) << who;
  }
}

// ---------------------------------------------------------------------------
// Near-zero denominators
// ---------------------------------------------------------------------------

TEST(Accelerator, AitkenFallsBackWhenTheDenominatorVanishes) {
  // Identical residuals two iterations running: Delta r == 0, so the Aitken
  // quotient is 0/0. It must degrade to initial_relaxation, not produce a NaN
  // that then poisons every later iterate.
  auto a = must_make(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.25});
  const std::vector<double> old_x{0.0, 0.0};
  const std::vector<double> new_x{4.0, 8.0};
  std::vector<double> out(2, 0.0);
  for (int k = 0; k < 4; ++k) {
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 2, k).ok())
        << "at iteration " << k;
    EXPECT_DOUBLE_EQ(out[0], 1.0) << "at iteration " << k;
    EXPECT_DOUBLE_EQ(out[1], 2.0) << "at iteration " << k;
  }
}

TEST(Accelerator, AitkenFallsBackWhenTheNumeratorVanishes) {
  // Orthogonal residual change: the numerator is 0, so the natural formula
  // gives omega == 0 and the iterate freezes for ever (and every later
  // denominator is then 0 too). Rejected in favour of initial_relaxation.
  auto a = must_make(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.5});
  std::vector<double> out(2, 0.0);

  const std::vector<double> o0{0.0, 0.0}, n0{1.0, 0.0};   // r_0 = (1, 0)
  ASSERT_TRUE(a->update(o0.data(), n0.data(), out.data(), 2, 0).ok());
  // r_1 = (1, 1): Delta r = (0, 1) is orthogonal to r_0, so <r_0, dr> == 0.
  const std::vector<double> o1{0.0, 0.0}, n1{1.0, 1.0};
  ASSERT_TRUE(a->update(o1.data(), n1.data(), out.data(), 2, 1).ok());
  EXPECT_DOUBLE_EQ(out[0], 0.5);
  EXPECT_DOUBLE_EQ(out[1], 0.5);
}

TEST(Accelerator, SteffensenFallsBackPerComponent) {
  // Component 0 has a usable secant pair, component 1 does not (its residual
  // never changes). The good component must still take its secant step while
  // the degenerate one falls back -- not the whole vector giving up, and not
  // an inf in component 1 poisoning the iterate.
  auto a = must_make(
      AccelConfig{.kind = AccelKind::Steffensen, .initial_relaxation = 0.5});
  std::vector<double> out(2, 0.0);

  const std::vector<double> o0{0.0, 0.0}, n0{2.0, 3.0};   // r_0 = (2, 3)
  ASSERT_TRUE(a->update(o0.data(), n0.data(), out.data(), 2, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 1.0);
  EXPECT_DOUBLE_EQ(out[1], 1.5);

  // x_old = (1, 1.5); r_1 = (1, 3) so dr = (-1, 0), dx = (1, 1.5).
  const std::vector<double> o1{1.0, 1.5}, n1{2.0, 4.5};
  ASSERT_TRUE(a->update(o1.data(), n1.data(), out.data(), 2, 1).ok());
  // component 0: x - r * dx / dr = 1 - 1 * 1 / (-1) = 2
  EXPECT_DOUBLE_EQ(out[0], 2.0);
  // component 1: dr == 0, so 1.5 + 0.5 * 3 = 3
  EXPECT_DOUBLE_EQ(out[1], 3.0);
  EXPECT_TRUE(std::isfinite(out[1]));
}

// ---------------------------------------------------------------------------
// The least-squares solve
// ---------------------------------------------------------------------------

namespace {

/// Two columns spanning a plane, the second only `eps` away from the first,
/// which is exactly the geometry the quasi-Newton V matrix has: differences of
/// nearly equal residual vectors. Returns column-major m x 2.
void near_dependent_pair(std::size_t m, double eps, std::vector<double>& a) {
  std::vector<double> c1(m), u(m);
  for (std::size_t i = 0; i < m; ++i) {
    c1[i] = std::sin(0.37 * static_cast<double>(i) + 0.11);
    u[i]  = std::cos(0.71 * static_cast<double>(i) + 0.29);
  }
  double d = 0.0, n1 = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    d += c1[i] * u[i];
    n1 += c1[i] * c1[i];
  }
  for (std::size_t i = 0; i < m; ++i) u[i] -= d / n1 * c1[i];
  double nu = 0.0;
  for (double e : u) nu += e * e;
  nu = std::sqrt(nu);
  for (double& e : u) e *= std::sqrt(n1) / nu;

  a.assign(m * 2, 0.0);
  for (std::size_t i = 0; i < m; ++i) {
    a[i]     = c1[i];
    a[m + i] = c1[i] + eps * u[i];
  }
}

}  // namespace

TEST(QnLeastSquares, QrBeatsNormalEquationsOnNearDependentColumns) {
  constexpr std::size_t m = 64;
  constexpr double eps = 1e-8;   // cond(A) ~ 2e8, cond(A^T A) ~ 4e16
  std::vector<double> a;
  near_dependent_pair(m, eps, a);

  const double t0 = 1.0, t1 = -2.0;
  std::vector<double> b(m);
  for (std::size_t i = 0; i < m; ++i) b[i] = t0 * a[i] + t1 * a[m + i];

  std::vector<double> aq = a, bq = b, alpha;
  const std::size_t rank = fuse::linalg::least_squares_builtin(aq, m, 2, bq, alpha);
  ASSERT_EQ(rank, 2u) << "both columns carry information at eps = 1e-8";

  const double qr_err = std::hypot(alpha[0] - t0, alpha[1] - t1) / std::hypot(t0, t1);

  // Normal equations, solved exactly as a 2x2 system. The Gram matrix has
  // condition cond(A)^2 ~ 4e16, past what a double can resolve, so this loses
  // every digit even though the QR on the same data loses about half.
  double g00 = 0, g01 = 0, g11 = 0, r0 = 0, r1 = 0;
  for (std::size_t i = 0; i < m; ++i) {
    g00 += a[i] * a[i];
    g01 += a[i] * a[m + i];
    g11 += a[m + i] * a[m + i];
    r0  += a[i] * b[i];
    r1  += a[m + i] * b[i];
  }
  const double det = g00 * g11 - g01 * g01;
  const double ne0 = (r0 * g11 - g01 * r1) / det;
  const double ne1 = (g00 * r1 - g01 * r0) / det;
  const double ne_err = std::hypot(ne0 - t0, ne1 - t1) / std::hypot(t0, t1);

  EXPECT_LT(qr_err, 1e-6) << "pivoted Householder QR, relative error in alpha";
  EXPECT_GT(ne_err, 1e-2) << "normal equations should have lost every digit";
  EXPECT_GT(ne_err, 1e4 * qr_err)
      << "QR " << qr_err << " vs normal equations " << ne_err;

  // The least-squares residual is the quantity the accelerator actually uses,
  // and QR keeps it at rounding level even where alpha itself is inaccurate.
  double s = 0.0, bn = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double e = alpha[0] * a[i] + alpha[1] * a[m + i] - b[i];
    s += e * e;
    bn += b[i] * b[i];
  }
  EXPECT_LT(std::sqrt(s / bn), 1e-12);
}

TEST(QnLeastSquares, StaysUsableAsColumnsBecomeMoreDependent) {
  constexpr std::size_t m = 64;
  const double t0 = 1.0, t1 = -2.0;
  for (const double eps : {1e-6, 1e-8, 1e-10, 1e-12}) {
    std::vector<double> a;
    near_dependent_pair(m, eps, a);
    std::vector<double> b(m);
    for (std::size_t i = 0; i < m; ++i) b[i] = t0 * a[i] + t1 * a[m + i];
    std::vector<double> aq = a, bq = b, alpha;
    const std::size_t rank = fuse::linalg::least_squares_builtin(aq, m, 2, bq, alpha);
    ASSERT_EQ(rank, 2u) << "eps = " << eps;
    // Backward stability bound: the error grows like eps_machine / eps, and
    // stays far away from the total loss the normal equations suffer.
    const double err = std::hypot(alpha[0] - t0, alpha[1] - t1) / std::hypot(t0, t1);
    EXPECT_LT(err, 1e-3) << "eps = " << eps << ", relative error " << err;
    EXPECT_TRUE(std::isfinite(alpha[0]));
    EXPECT_TRUE(std::isfinite(alpha[1]));
  }
}

TEST(QnLeastSquares, DetectsExactRankDeficiency) {
  constexpr std::size_t m = 64;
  std::vector<double> a(m * 3), b(m);
  for (std::size_t i = 0; i < m; ++i) {
    a[i]         = std::sin(0.37 * static_cast<double>(i));
    a[m + i]     = 2.0 * a[i];                     // exactly dependent
    a[2 * m + i] = std::cos(0.5 * static_cast<double>(i));
    b[i]         = 3.0 * a[i] + 1.5 * a[2 * m + i];
  }
  std::vector<double> aq = a, bq = b, alpha;
  const std::size_t rank = fuse::linalg::least_squares_builtin(aq, m, 3, bq, alpha);
  EXPECT_EQ(rank, 2u) << "three columns spanning a plane";

  // A dropped column gets exactly zero, and the retained ones still reproduce
  // the right-hand side: rank deficiency costs accuracy in alpha, never in the
  // fit.
  double s = 0.0;
  for (std::size_t i = 0; i < m; ++i) {
    const double e = alpha[0] * a[i] + alpha[1] * a[m + i] +
                     alpha[2] * a[2 * m + i] - b[i];
    s += e * e;
  }
  EXPECT_LT(std::sqrt(s), 1e-12);
  EXPECT_TRUE(alpha[0] == 0.0 || alpha[1] == 0.0)
      << "one of the two dependent columns must be dropped outright";
}

TEST(QnLeastSquares, HandlesAZeroColumnAndAnEmptyProblem) {
  constexpr std::size_t m = 32;
  std::vector<double> a(m * 2, 0.0), b(m);
  for (std::size_t i = 0; i < m; ++i) {
    a[i] = std::sin(static_cast<double>(i));
    b[i] = 2.0 * a[i];
  }
  std::vector<double> aq = a, bq = b, alpha;
  EXPECT_EQ(fuse::linalg::least_squares_builtin(aq, m, 2, bq, alpha), 1u);
  EXPECT_NEAR(alpha[0], 2.0, 1e-12);
  EXPECT_DOUBLE_EQ(alpha[1], 0.0);

  std::vector<double> empty_a, empty_b, empty_out;
  EXPECT_EQ(fuse::linalg::least_squares_builtin(empty_a, 0, 0, empty_b, empty_out), 0u);
  EXPECT_TRUE(empty_out.empty());

  std::vector<double> all_zero(m * 2, 0.0), rhs = b, out;
  EXPECT_EQ(fuse::linalg::least_squares_builtin(all_zero, m, 2, rhs, out), 0u);
  ASSERT_EQ(out.size(), 2u);
  EXPECT_DOUBLE_EQ(out[0], 0.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
}

// ---------------------------------------------------------------------------
// Task 7.4: QR column filtering.
//
// Every test here is written to fail against the no-op filter it replaced:
// each one either compares a QR filter against FilterKind::None on the same
// input, or asserts a column count that the no-op cannot produce. A filter
// test that a no-op passes measures nothing.
// ---------------------------------------------------------------------------

namespace {

/// Drives an accelerator through a prescribed sequence of residuals and reports
/// the surviving column count after each call. `x_old` is held at zero and
/// `x_new` set to the wanted residual, so the stored secant columns are exactly
/// the successive differences of `residuals`. The point is to put a chosen V in
/// front of the filter, not to solve anything.
std::vector<std::size_t> columns_after_each(
    Accelerator& accel, const std::vector<std::vector<double>>& residuals) {
  const std::size_t         n = residuals.front().size();
  const std::vector<double> zero(n, 0.0);
  std::vector<std::size_t>  counts;
  std::vector<double>       out(n, 0.0);
  for (std::size_t k = 0; k < residuals.size(); ++k) {
    EXPECT_TRUE(accel
                    .update(zero.data(), residuals[k].data(), out.data(), n,
                            static_cast<int>(k))
                    .ok())
        << "at step " << k;
    counts.push_back(accel.columns());
  }
  return counts;
}

AccelConfig filter_cfg(FilterKind f, double limit) {
  return AccelConfig{.kind               = AccelKind::IQN_ILS,
                     .initial_relaxation = 0.25,
                     .reuse              = 8,
                     .filter             = f,
                     .limit              = limit};
}

/// Delta r_1 = (0, 1, 0, 0) and Delta r_2 = (0, 1, eps, 0): the second secant
/// column is `eps` away from being a multiple of the first, which is the
/// geometry a converging quasi-Newton window produces by construction.
std::vector<std::vector<double>> near_dependent_sequence(double eps) {
  return {{1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 2.0, eps, 0.0}};
}

}  // namespace

TEST(QnColumnFilter, Qr2DropsANearlyDependentColumnThatNoFilterKeeps) {
  const auto seq = near_dependent_sequence(1e-6);
  const std::vector<std::size_t> kept{0u, 1u, 2u};
  const std::vector<std::size_t> dropped{0u, 1u, 1u};

  auto none = must_make(filter_cfg(FilterKind::None, 1e-2));
  EXPECT_EQ(columns_after_each(*none, seq), kept)
      << "no filter keeps both columns; the rank truncation inside the solve "
         "does not touch the stored history";

  // Only 1e-6 of the older column is orthogonal to the newer one, far under
  // the 1% limit, so QR2 drops it. Columns are judged newest first, so it is
  // the older of the two that goes.
  auto qr2 = must_make(filter_cfg(FilterKind::QR2, 1e-2));
  EXPECT_EQ(columns_after_each(*qr2, seq), dropped);

  // QR1 agrees here: 1e-6 is also below 1e-2 * ||V||_F ~ 1.4e-2. The two
  // filters are separated by the next test, not this one.
  auto qr1 = must_make(filter_cfg(FilterKind::QR1, 1e-2));
  EXPECT_EQ(columns_after_each(*qr1, seq), dropped);
}

TEST(QnColumnFilter, Qr1AndQr2DisagreeOnASmallButIndependentColumn) {
  // Delta r_1 = (0, 1, 0, 0), Delta r_2 = (0, 0, 1e-6, 0). The newest column
  // is tiny but exactly orthogonal to the older one, which is the case that
  // tells the two threshold conventions apart:
  //   QR2 asks what fraction of the column is new -- all of it -- and keeps it;
  //   QR1 asks whether the column is large next to ||V||_F -- it is not, being
  //   1e-6 against a threshold of 1e-2 -- and drops it.
  // Getting the two the wrong way round makes this test fail in both
  // directions at once.
  const std::vector<std::vector<double>> seq = {
      {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0, 1e-6, 0.0}};

  auto qr2 = must_make(filter_cfg(FilterKind::QR2, 1e-2));
  EXPECT_EQ(columns_after_each(*qr2, seq), (std::vector<std::size_t>{0u, 1u, 2u}));

  auto qr1 = must_make(filter_cfg(FilterKind::QR1, 1e-2));
  EXPECT_EQ(columns_after_each(*qr1, seq), (std::vector<std::size_t>{0u, 1u, 1u}))
      << "QR1's threshold is not tied to the column being judged, so it can "
         "drop the newest column";
}

TEST(QnColumnFilter, Qr1CanEmptyTheHistoryAndTheStepThenRelaxes) {
  // Two orthogonal unit columns, so ||V||_F = sqrt(2) and a limit of 0.9 puts
  // the QR1 threshold at 1.27 -- above the whole length of either column, so
  // both go and `cols` reaches 0. That is a legal state, not a hole: it is
  // exactly the state at a window boundary, and the step must degrade to
  // relaxation rather than misbehave.
  const std::vector<double> zero{0.0, 0.0};
  const std::vector<double> r1{1.0, 0.0};   // Delta r_1 = (1, 0)
  const std::vector<double> r2{1.0, 1.0};   // Delta r_2 = (0, 1)
  std::vector<double>       out(2, 0.0);

  auto qr1 = must_make(filter_cfg(FilterKind::QR1, 0.9));
  ASSERT_TRUE(qr1->update(zero.data(), zero.data(), out.data(), 2, 0).ok());
  ASSERT_TRUE(qr1->update(zero.data(), r1.data(), out.data(), 2, 1).ok());
  EXPECT_EQ(qr1->columns(), 1u) << "a lone column is its own ||V||_F, and "
                                   "0.9 * ||v|| < ||v||, so it always survives";
  ASSERT_TRUE(qr1->update(zero.data(), r2.data(), out.data(), 2, 2).ok());
  EXPECT_EQ(qr1->columns(), 0u);
  // cols == 0 with no retained Jacobian is the relaxation case.
  EXPECT_DOUBLE_EQ(out[0], 0.25);  // 0 + 0.25 * (1 - 0)
  EXPECT_DOUBLE_EQ(out[1], 0.25);
  EXPECT_TRUE(std::isfinite(out[0]));
  EXPECT_TRUE(std::isfinite(out[1]));

  // The same input under QR2, whose threshold is the column's own norm, keeps
  // both: 0.9 * ||v_j|| < ||v_j||, for either column.
  auto qr2 = must_make(filter_cfg(FilterKind::QR2, 0.9));
  ASSERT_TRUE(qr2->update(zero.data(), zero.data(), out.data(), 2, 0).ok());
  ASSERT_TRUE(qr2->update(zero.data(), r1.data(), out.data(), 2, 1).ok());
  ASSERT_TRUE(qr2->update(zero.data(), r2.data(), out.data(), 2, 2).ok());
  EXPECT_EQ(qr2->columns(), 2u);
}

TEST(QnColumnFilter, Qr2NeverEmptiesANonEmptyHistory) {
  // The newest column is examined first and so is compared against nothing,
  // and limit < 1 by construction (make_accelerator enforces [0, 1)). One
  // column therefore always survives, however dependent the window has become
  // -- asserted at the most aggressive limit the configuration allows.
  auto a = must_make(filter_cfg(FilterKind::QR2, 0.999999));
  const std::vector<double> zero(3, 0.0);
  std::vector<double>       out(3, 0.0);
  for (int k = 0; k < 12; ++k) {
    // Successive residuals that differ by an ever smaller, always parallel
    // amount: every secant column after the first is a shrinking multiple of
    // its predecessor.
    const double s = std::pow(0.5, k);
    const std::vector<double> r{1.0 + s, 2.0 - s, 3.0 + 0.5 * s};
    ASSERT_TRUE(a->update(zero.data(), r.data(), out.data(), 3, k).ok())
        << "at step " << k;
    EXPECT_LE(a->columns(), 1u) << "at step " << k;
    if (k >= 2) EXPECT_EQ(a->columns(), 1u) << "at step " << k;
  }
}

TEST(QnColumnFilter, Qr3MatchesQr2) {
  // QR3 (Davis, Schulte & Uekermann 2022, Algorithm 3; preCICE QR3FILTER) is a
  // cost optimisation over QR2, not a different criterion: it runs QR2's test
  // on an incrementally maintained factorisation and falls through to a full
  // QR2 rebuild only when that test trips. This implementation rebuilds every
  // call, so the two read the same matrix and must retain the same columns.
  // Pinned on inputs where the answer is not trivially "keep everything", and
  // against QR1, which really is a different criterion.
  for (const double eps : {1e-6, 1e-3, 1e-1}) {
    const auto seq = near_dependent_sequence(eps);
    auto qr2 = must_make(filter_cfg(FilterKind::QR2, 1e-2));
    auto qr3 = must_make(filter_cfg(FilterKind::QR3, 1e-2));
    EXPECT_EQ(columns_after_each(*qr3, seq), columns_after_each(*qr2, seq))
        << "eps = " << eps;
  }
  const std::vector<std::vector<double>> independent = {
      {1.0, 0.0, 0.0, 0.0}, {1.0, 1.0, 0.0, 0.0}, {1.0, 1.0, 1e-6, 0.0}};
  auto qr3 = must_make(filter_cfg(FilterKind::QR3, 1e-2));
  auto qr1 = must_make(filter_cfg(FilterKind::QR1, 1e-2));
  EXPECT_EQ(columns_after_each(*qr3, independent),
            (std::vector<std::size_t>{0u, 1u, 2u}));
  EXPECT_EQ(columns_after_each(*qr1, independent),
            (std::vector<std::size_t>{0u, 1u, 1u}))
      << "QR3 follows QR2's relative test, not QR1's";
}

TEST(QnColumnFilter, AZeroLimitDisablesEveryFilter) {
  // limit == 0 is the lower end of the validated range. preCICE spells the
  // same rule `bool applyFilter = (singularityLimit > 0.0)`.
  const auto seq = near_dependent_sequence(1e-6);
  const std::vector<std::size_t> kept{0u, 1u, 2u};
  for (const FilterKind f : {FilterKind::QR1, FilterKind::QR2, FilterKind::QR3}) {
    auto a = must_make(filter_cfg(f, 0.0));
    EXPECT_EQ(columns_after_each(*a, seq), kept) << fuse::to_cstring(f);
  }
}

TEST(QnColumnFilter, FilteringChangesTheWindowOnARealIteration) {
  // End to end on the ladder problem rather than on a hand-built V, because a
  // filter that only ever fires on synthetic input is not doing its job.
  //
  // The counts are pinned for both settings and they differ: at iteration 4
  // the first secant pair has only 0.18% of itself left orthogonal to the
  // three newer ones, so QR2 drops it and the window is one shorter than the
  // unfiltered one. That single drop is the whole of the 6 -> 7 iteration
  // change pinned in LadderIsMonotoneOnADenseLinearProblem.
  //
  // After convergence the counts agree again, and deliberately so: the
  // residual differences are then rounding noise, and noise vectors in five
  // dimensions are not nearly parallel, so QR2 correctly keeps them. A filter
  // is not a converged-iteration detector.
  const LinearMap g = ladder_problem();
  const std::vector<std::vector<std::size_t>> expected{
      {0u, 1u, 2u, 3u, 4u, 4u, 4u, 4u},   // FilterKind::None
      {0u, 1u, 2u, 3u, 3u, 4u, 4u, 4u}};  // FilterKind::QR2
  int which = 0;
  for (const FilterKind f : {FilterKind::None, FilterKind::QR2}) {
    AccelConfig cfg{.kind               = AccelKind::IQN_ILS,
                    .initial_relaxation = 0.5,
                    .reuse              = 4,
                    .filter             = f,
                    .limit              = 1e-2};
    auto                     a = must_make(cfg);
    std::vector<double>      x(g.n, 0.0);
    std::vector<std::size_t> counts;
    for (int k = 0; k < 60; ++k) {
      const std::vector<double> gx = g.apply(x);
      std::vector<double>       next(g.n);
      ASSERT_TRUE(a->update(x.data(), gx.data(), next.data(), g.n, k).ok())
          << fuse::to_cstring(f) << " at iteration " << k;
      x = std::move(next);
      ASSERT_LE(a->columns(), 4u) << fuse::to_cstring(f) << " at iteration " << k;
      if (counts.size() < 8) counts.push_back(a->columns());
    }
    EXPECT_EQ(counts, expected[static_cast<std::size_t>(which)])
        << fuse::to_cstring(f);
    // Filtering must not cost correctness, only columns.
    for (std::size_t i = 0; i < g.n; ++i) {
      EXPECT_NEAR(x[i], g.p[i], 1e-9) << fuse::to_cstring(f) << " component " << i;
    }
    ++which;
  }
  EXPECT_NE(expected[0], expected[1]) << "the two settings must be "
                                         "distinguishable, or this test is "
                                         "measuring nothing";
}

// ---------------------------------------------------------------------------
// Non-finite values, null pointers, sizes
// ---------------------------------------------------------------------------

TEST(Accelerator, NonFiniteInputIsRejectedByEveryKind) {
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k});
    std::vector<double> out(2, -1.0);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    const std::vector<double> good{1.0, 2.0};
    for (const double bad_value : {nan, inf, -inf}) {
      const std::vector<double> bad{bad_value, 2.0};
      const auto r1 = a->update(bad.data(), good.data(), out.data(), 2, 0);
      EXPECT_FALSE(r1.ok()) << fuse::to_cstring(k);
      EXPECT_EQ(r1.error().code, "E-ACCEL-NONFINITE");
      const auto r2 = a->update(good.data(), bad.data(), out.data(), 2, 0);
      EXPECT_FALSE(r2.ok()) << fuse::to_cstring(k);
      EXPECT_EQ(r2.error().code, "E-ACCEL-NONFINITE");
    }
    // Rejected before anything was written, and before anything was recorded.
    EXPECT_DOUBLE_EQ(out[0], -1.0);
    EXPECT_EQ(a->columns(), 0u);
  }
}

TEST(Accelerator, NullPointersAndNegativeIterationsAreRejected) {
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS});
  const std::vector<double> x{1.0, 2.0};
  std::vector<double> out(2, 0.0);

  EXPECT_EQ(a->update(nullptr, x.data(), out.data(), 2, 0).error().code,
            "E-ACCEL-NULL");
  EXPECT_EQ(a->update(x.data(), nullptr, out.data(), 2, 0).error().code,
            "E-ACCEL-NULL");
  EXPECT_EQ(a->update(x.data(), x.data(), nullptr, 2, 0).error().code,
            "E-ACCEL-NULL");
  EXPECT_EQ(a->update(x.data(), x.data(), out.data(), 2, -1).error().code,
            "E-ACCEL-ITERATION");
}

TEST(Accelerator, WritingTheOutputOverAnInputIsRejected) {
  // Not a crash if allowed, which is what makes it worth rejecting: the
  // history is recorded from x_old and x_new after x_out has been written, so
  // an in-place call would store a secant pair that was never evaluated.
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k});
    std::vector<double> x{1.0, 2.0};
    std::vector<double> y{3.0, 4.0};
    EXPECT_EQ(a->update(x.data(), y.data(), x.data(), 2, 0).error().code,
              "E-ACCEL-ALIAS") << fuse::to_cstring(k);
    EXPECT_EQ(a->update(x.data(), y.data(), y.data(), 2, 0).error().code,
              "E-ACCEL-ALIAS") << fuse::to_cstring(k);
  }
}

TEST(Accelerator, PartiallyOverlappingOutputIsRejected) {
  // Exact aliasing is the easy half. An x_out that merely *overlaps* an input
  // is the dangerous half: it is perfectly well defined C++, so it is clean
  // under ASan and UBSan, and it produces a plausible-looking wrong answer --
  // IQN-ILS on the ladder problem drifts about 50% away from the fixed point
  // in the last component while reporting success on every call.
  //
  // std::less<const double*> is a strict total order over arbitrary pointers
  // and must agree with `<` inside a single array, so a range test cannot miss
  // a genuine overlap.
  constexpr std::size_t n = 3;
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k});
    // One buffer, three disjoint tiles: x_old = [0, n), x_new = [n, 2n),
    // x_out = [2n, 3n). Adjacent but disjoint, so it must be accepted -- a
    // range check that is off by one would reject this.
    std::vector<double> buf(3 * n);
    for (std::size_t i = 0; i < buf.size(); ++i) {
      buf[i] = 1.0 + static_cast<double>(i);
    }
    double* const old_p = buf.data();
    double* const new_p = buf.data() + n;
    double* const out_p = buf.data() + 2 * n;
    EXPECT_TRUE(a->update(old_p, new_p, out_p, n, 0).ok())
        << fuse::to_cstring(k) << ": abutting but disjoint tiles are legal";

    // Every length-n window of the buffer that starts before x_new ends
    // overlaps x_old or x_new; only the one starting at 2n does not. All the
    // offsets used here stay inside `buf`, so no out-of-range pointer is ever
    // formed.
    for (std::size_t off = 0; off < 2 * n; ++off) {
      const auto r = a->update(old_p, new_p, buf.data() + off, n, 0);
      ASSERT_FALSE(r.ok()) << fuse::to_cstring(k) << ": overlap at offset "
                           << off << " was accepted";
      EXPECT_EQ(r.error().code, "E-ACCEL-ALIAS") << fuse::to_cstring(k);
    }
    // Total containment of the output range inside an input range, for a
    // shorter vector, is overlap too.
    const auto inside = a->update(old_p, new_p, old_p + 1, n - 1, 0);
    ASSERT_FALSE(inside.ok()) << fuse::to_cstring(k);
    EXPECT_EQ(inside.error().code, "E-ACCEL-ALIAS") << fuse::to_cstring(k);
  }
}

// ---------------------------------------------------------------------------
// Overflow of the relaxation blend itself, and the E-ACCEL-DIVERGED contract
// ---------------------------------------------------------------------------

TEST(Accelerator, AWideBlendDoesNotOverflowOnItsOwnDifference) {
  // x_old + w * (x_new - x_old) overflows in the *difference* long before the
  // blend itself is unrepresentable: at w = 0.5 with the endpoints at the two
  // extremes of the double range the exact answer is 0 and the difference is
  // +inf. Nothing here is pathological -- both inputs are finite, omega is an
  // ordinary 0.5 -- so reporting E-ACCEL-DIVERGED and leaving inf in x_out
  // would be a spurious failure on a perfectly representable step.
  constexpr double kHuge = 1e308;
  for (const AccelKind k : {AccelKind::Constant, AccelKind::Aitken,
                            AccelKind::Steffensen, AccelKind::Secant,
                            AccelKind::IQN_ILS, AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{
        .kind = k, .initial_relaxation = 0.5, .omega = 0.5, .reuse = 8});
    const std::vector<double> old_x{-kHuge, 1.0};
    const std::vector<double> new_x{kHuge, 2.0};
    std::vector<double> out(2, -7.0);
    const auto r = a->update(old_x.data(), new_x.data(), out.data(), 2, 0);
    ASSERT_TRUE(r.ok()) << fuse::to_cstring(k) << ": " << r.error().code << " "
                        << r.error().message;
    EXPECT_DOUBLE_EQ(out[0], 0.0) << fuse::to_cstring(k);
    EXPECT_DOUBLE_EQ(out[1], 1.5) << fuse::to_cstring(k);
  }
}

TEST(Accelerator, EveryKindLeavesAnExactFixedPointBitForBit) {
  // The guard that fixes the overflow above must not be bought by switching
  // relax_into to the convex form (1 - w) * x_old + w * x_new, because
  // (1 - w) * x + w * x is not x in floating point. Once the coupled solvers
  // return exactly what they were given, every accelerator must return that
  // same vector to the last bit, at every history depth and -- for IMVJ --
  // with a retained Jacobian in place. An iterate that drifts by one ulp per
  // call turns a converged window into an endless one.
  const LinearMap g = ladder_problem();
  const std::vector<double> fixed{1.5, -2.25, 3.125, 0.5, -4.0};

  for (int warm = 0; warm <= 6; ++warm) {
    for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                              AccelKind::Aitken, AccelKind::Secant,
                              AccelKind::Steffensen, AccelKind::IQN_ILS,
                              AccelKind::IQN_IMVJ}) {
      auto a = must_make(AccelConfig{.kind = k,
                                     .initial_relaxation = 0.5,
                                     .omega = 0.7,
                                     .reuse = 8});
      if (k == AccelKind::IQN_IMVJ) {
        // Close one full window first, so the retained inverse Jacobian is
        // non-empty when the fixed point arrives.
        std::vector<double> x(g.n, 0.0);
        for (int it = 0; it < 8; ++it) {
          const std::vector<double> gx = g.apply(x);
          std::vector<double> nx(g.n);
          ASSERT_TRUE(a->update(x.data(), gx.data(), nx.data(), g.n, it).ok());
          x = std::move(nx);
        }
        a->reset();
        ASSERT_GT(a->retained_blocks(), 0u);
      }
      std::vector<double> x(g.n, 0.0);
      for (int it = 0; it < warm; ++it) {
        const std::vector<double> gx = g.apply(x);
        std::vector<double> nx(g.n);
        ASSERT_TRUE(a->update(x.data(), gx.data(), nx.data(), g.n, it).ok())
            << fuse::to_cstring(k);
        x = std::move(nx);
      }
      // A vector unrelated to the waveform, so nothing can cancel by luck.
      std::vector<double> out(g.n, 12345.0);
      const auto r =
          a->update(fixed.data(), fixed.data(), out.data(), g.n, warm);
      ASSERT_TRUE(r.ok()) << fuse::to_cstring(k) << " warm=" << warm << ": "
                          << r.error().message;
      for (std::size_t i = 0; i < g.n; ++i) {
        EXPECT_EQ(out[i], fixed[i])
            << fuse::to_cstring(k) << " warm=" << warm << " component " << i;
      }
    }
  }
}

TEST(Accelerator, DivergenceLeavesAFiniteIterateInXOut) {
  // The E-ACCEL-DIVERGED contract: the call fails, but x_out is finite, so a
  // caller that logs the failure and continues from the fallback cannot
  // propagate inf into the coupled solution.
  //
  // omega = 1e300 against a residual of 1e300 overflows the relaxed blend
  // itself, which is the hardest case -- there is no weaker relaxed step to
  // retreat to, so x_out must come back as x_old unchanged.
  constexpr double kBig = 1e300;
  for (const AccelKind k : {AccelKind::Constant, AccelKind::Aitken,
                            AccelKind::Steffensen, AccelKind::Secant,
                            AccelKind::IQN_ILS, AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k,
                                   .initial_relaxation = kBig,
                                   .omega = kBig,
                                   .reuse = 8});
    const std::vector<double> old_x{0.0, -3.0};
    const std::vector<double> new_x{kBig, -3.0};
    std::vector<double> out(2, -7.0);
    const auto r = a->update(old_x.data(), new_x.data(), out.data(), 2, 0);
    ASSERT_FALSE(r.ok()) << fuse::to_cstring(k);
    EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED") << fuse::to_cstring(k);
    for (std::size_t i = 0; i < 2; ++i) {
      EXPECT_TRUE(std::isfinite(out[i]))
          << fuse::to_cstring(k) << " component " << i << " = " << out[i];
      EXPECT_DOUBLE_EQ(out[i], old_x[i])
          << fuse::to_cstring(k) << ": the fallback overflowed too, so x_out "
                                    "must be x_old unchanged";
    }
    EXPECT_NE(r.error().message.find("x_out holds x_old unchanged"),
              std::string::npos)
        << fuse::to_cstring(k) << ": " << r.error().message;
    EXPECT_EQ(a->columns(), 0u) << fuse::to_cstring(k);
  }
}

TEST(Accelerator, DivergenceFallsBackToTheUnderRelaxedStepWhenItSurvives) {
  // The other half of the contract: when only the accelerated step overflows
  // and the plain under-relaxed step is representable, x_out must hold that
  // step, not the blown-up one. Both kinds below reach it through their own
  // dynamic factor rather than through initial_relaxation.
  {
    // Aitken. omega0 = 1e280, and one secant pair whose residual moves
    // DOWN by a single ulp (window seed track Stage E's own D76
    // safeguard rejects a non-positive candidate and restarts to
    // omega0_, so an UPWARD ulp here -- making num = r_prev*d > 0,
    // candidate = -omega_*num/den < 0 -- is rejected and never reaches
    // relax_into at all; a DOWNWARD ulp makes num < 0, candidate =
    // -omega_*num/den > 0, so the guard accepts it and the Irons-Tuck
    // factor is still amplified by ~1/eps to 6e295, and 6e295 * 1e20
    // overflows -- while 1e280 * 1e20 is a finite 1e300).
    auto a = must_make(
        AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 1e280});
    std::vector<double> out(1, 0.0);
    const std::vector<double> o0{0.0}, n0{1e20};
    ASSERT_TRUE(a->update(o0.data(), n0.data(), out.data(), 1, 0).ok());
    ASSERT_DOUBLE_EQ(out[0], 1e280 * 1e20);

    const std::vector<double> o1{0.0}, n1{1e20 - 16384.0};  // -1 ulp of 1e20
    ASSERT_NE(n1[0], n0[0]) << "the perturbation must be representable";
    const auto r = a->update(o1.data(), n1.data(), out.data(), 1, 1);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED");
    EXPECT_NE(r.error().message.find("x_out holds the under-relaxed fallback"),
              std::string::npos) << r.error().message;
    ASSERT_TRUE(std::isfinite(out[0])) << out[0];
    EXPECT_DOUBLE_EQ(out[0], 0.0 + 1e280 * (n1[0] - 0.0));
  }
  {
    // Steffensen. The residual itself overflows at iteration 1, so the
    // diagonal secant degrades to omega0 * inf, but the guarded blend of the
    // two finite endpoints is an exact 0.
    auto a = must_make(
        AccelConfig{.kind = AccelKind::Steffensen, .initial_relaxation = 0.5});
    std::vector<double> out(1, 0.0);
    const std::vector<double> o0{-1.0}, n0{1.0};
    ASSERT_TRUE(a->update(o0.data(), n0.data(), out.data(), 1, 0).ok());

    const std::vector<double> o1{-1e308}, n1{1e308};
    const auto r = a->update(o1.data(), n1.data(), out.data(), 1, 1);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED");
    EXPECT_NE(r.error().message.find("x_out holds the under-relaxed fallback"),
              std::string::npos) << r.error().message;
    ASSERT_TRUE(std::isfinite(out[0])) << out[0];
    EXPECT_DOUBLE_EQ(out[0], 0.0);
  }
}

TEST(Accelerator, ZeroLengthIsASuccessfulNoOp) {
  // A rank that owns no interface elements is ordinary in a parallel run.
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{.kind = k});
    EXPECT_TRUE(a->update(nullptr, nullptr, nullptr, 0, 0).ok())
        << fuse::to_cstring(k);
    EXPECT_EQ(a->columns(), 0u) << fuse::to_cstring(k);
  }
}

TEST(Accelerator, ConfigurationIsValidated) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  for (const double bad : {0.0, nan, inf}) {
    const auto r = make_accelerator(
        AccelConfig{.kind = AccelKind::IQN_ILS, .initial_relaxation = bad});
    ASSERT_FALSE(r.ok()) << "initial_relaxation " << bad;
    EXPECT_EQ(r.error().code, "E-ACCEL-CONFIG");
  }
  for (const double bad : {0.0, nan, inf}) {
    const auto r =
        make_accelerator(AccelConfig{.kind = AccelKind::Constant, .omega = bad});
    ASSERT_FALSE(r.ok()) << "omega " << bad;
    EXPECT_EQ(r.error().code, "E-ACCEL-CONFIG");
  }
  for (const int bad : {0, -1, Accelerator::kMaxReuse + 1}) {
    const auto r =
        make_accelerator(AccelConfig{.kind = AccelKind::IQN_ILS, .reuse = bad});
    ASSERT_FALSE(r.ok()) << "reuse " << bad;
    EXPECT_EQ(r.error().code, "E-ACCEL-CONFIG");
  }
  for (const double bad : {-1e-3, 1.0, 2.0, nan, inf}) {
    const auto r =
        make_accelerator(AccelConfig{.kind = AccelKind::IQN_ILS, .limit = bad});
    ASSERT_FALSE(r.ok()) << "limit " << bad;
    EXPECT_EQ(r.error().code, "E-ACCEL-CONFIG");
  }
  EXPECT_TRUE(make_accelerator(AccelConfig{.kind = AccelKind::IQN_ILS,
                                           .reuse = Accelerator::kMaxReuse,
                                           .limit = 0.0})
                  .ok());
}

// ---------------------------------------------------------------------------
// Multi-field support
// ---------------------------------------------------------------------------

TEST(Accelerator, FieldPartitionIsValidated) {
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS});
  EXPECT_THROW(a->set_field_partition({1, 4}), fuse::InvalidArgument);
  EXPECT_THROW(a->set_field_partition({0, 4, 2}), fuse::InvalidArgument)
      << "a decrease overlaps the blocks and stays an error";
  // An empty block is not an error: a rank can own none of one field of a
  // mesh entity set and some of another. Rejecting it threw out of resolve_regions()
  // on exactly the ranks that owned nothing (host run of the MPI-enabled build configuration, 5 ranks, 4
  // entities), hanging the ranks that did.
  EXPECT_NO_THROW(a->set_field_partition({0, 4, 4}));
  EXPECT_NO_THROW(a->set_field_partition({0, 0, 4}));
  EXPECT_NO_THROW(a->set_field_partition({0, 4}));
  EXPECT_EQ(a->field_partition().size(), 2u);
  EXPECT_NO_THROW(a->set_field_partition({}));
  EXPECT_TRUE(a->field_partition().empty());
}

TEST(Accelerator, FieldScalingIsANoOpForASingleBlock) {
  // The partition must be set explicitly. AccelConfig's default is an *empty*
  // partition, for which field_weights returns exactly 1.0 and no reciprocal
  // norm is ever formed, so without the set_field_partition({0}) below this
  // test would compare `scale_fields = true` against `scale_fields = false`
  // over two runs of literally the same arithmetic and could not fail.
  //
  // With one real block the row weights are a single global scalar 1/||r||.
  // It multiplies the least-squares matrix and its right-hand side alike, so
  // the minimiser is unchanged -- but not bit for bit: the QR sees a rescaled
  // matrix and its rank truncation is relative to the leading pivot, so once
  // the residual collapses the two runs can retain different column sets. The
  // measured worst relative difference over these iterations is 4.0e-12,
  // so 1e-9 is a real bound with margin rather than a rounding-level one.
  //
  // Seven iterations, not six: the default QR2 filter drops the oldest column
  // at iteration 4 (see LadderIsMonotoneOnADenseLinearProblem), so the exact
  // fit lands one step later and the convergence check below needs the extra
  // step to remain a check rather than a formality.
  constexpr double kTolRel = 1e-9;
  constexpr int    kSteps  = 7;
  const LinearMap g = ladder_problem();
  auto scaled = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                      .initial_relaxation = 0.5,
                                      .scale_fields = true});
  auto plain = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                     .initial_relaxation = 0.5,
                                     .scale_fields = false});
  scaled->set_field_partition({0});
  ASSERT_EQ(scaled->field_partition().size(), 1u);
  ASSERT_TRUE(plain->field_partition().empty());

  std::vector<double> xs(g.n, 0.0), xp(g.n, 0.0);
  for (int k = 0; k < kSteps; ++k) {
    const std::vector<double> gs = g.apply(xs);
    const std::vector<double> gp = g.apply(xp);
    std::vector<double> ns(g.n), np(g.n);
    ASSERT_TRUE(scaled->update(xs.data(), gs.data(), ns.data(), g.n, k).ok());
    ASSERT_TRUE(plain->update(xp.data(), gp.data(), np.data(), g.n, k).ok());
    for (std::size_t i = 0; i < g.n; ++i) {
      EXPECT_NEAR(ns[i], np[i], kTolRel * std::max(std::abs(np[i]), 1.0))
          << "iteration " << k << " component " << i;
    }
    xs = std::move(ns);
    xp = std::move(np);
  }
  // And the fit really did converge to the same place, which is the point the
  // tolerance above must not be allowed to hide.
  for (std::size_t i = 0; i < g.n; ++i) {
    EXPECT_NEAR(xs[i], g.p[i], 1e-8) << "component " << i;
    EXPECT_NEAR(xp[i], g.p[i], 1e-8) << "component " << i;
  }
}

namespace {

/// Two independent 5-dof fixed-point problems concatenated into one vector,
/// with the second block expressed in units `scale` times larger. `scale` is a
/// power of two so the change of units is exact in binary floating point and
/// any difference observed is the algorithm's, not the rounding's.
LinearMap two_field_problem(double scale) {
  const LinearMap a = ladder_problem();
  const LinearMap b = dense_map({0.85, 0.75, 0.6, -0.4, 0.3},
                                {2.0, -1.0, 4.0, 1.0, 3.0},
                                {2.0, 1.0, -3.0, 0.5, 1.0});
  const std::size_t nb = a.n;
  LinearMap g;
  g.n = 2 * nb;
  g.a.assign(g.n * g.n, 0.0);
  g.p.assign(g.n, 0.0);
  for (std::size_t i = 0; i < nb; ++i) {
    g.p[i]      = a.p[i];
    g.p[nb + i] = scale * b.p[i];
    for (std::size_t j = 0; j < nb; ++j) {
      g.a[i * g.n + j]                     = a.a[i * nb + j];
      g.a[(nb + i) * g.n + (nb + j)]       = b.a[i * nb + j];
    }
  }
  return g;
}

/// Runs `steps` accelerated iterations from x = 0 and returns the waveform
/// of the first field block only.
std::vector<std::vector<double>> first_block_trajectory(Accelerator& accel,
                                                        const LinearMap& g,
                                                        std::size_t nb,
                                                        int steps) {
  std::vector<std::vector<double>> trace;
  std::vector<double> x(g.n, 0.0);
  for (int k = 0; k < steps; ++k) {
    const std::vector<double> gx = g.apply(x);
    std::vector<double> next(g.n);
    EXPECT_TRUE(accel.update(x.data(), gx.data(), next.data(), g.n, k).ok())
        << "at iteration " << k;
    x = std::move(next);
    trace.emplace_back(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(nb));
  }
  return trace;
}

}  // namespace

TEST(Accelerator, PerFieldScalingMakesTheFitIndependentOfFieldUnits) {
  // Spec 5.3: per-field scaling is required for parallel multi-field
  // quasi-Newton. What it buys is precisely this -- the least-squares fit no
  // longer depends on the units each field happens to be expressed in, so a
  // field carrying 1e8-magnitude values cannot crowd a unit-magnitude field
  // out of the residual history.
  constexpr double kScale = 67108864.0;  // 2^26: an exact change of units
  constexpr int    kSteps = 8;
  const std::size_t nb = ladder_problem().n;

  const LinearMap plain_units  = two_field_problem(1.0);
  const LinearMap scaled_units = two_field_problem(kScale);

  auto a1 = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse = 8,
                                  .scale_fields = true});
  auto a2 = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse = 8,
                                  .scale_fields = true});
  a1->set_field_partition({0, nb});
  a2->set_field_partition({0, nb});
  const auto with_a = first_block_trajectory(*a1, plain_units, nb, kSteps);
  const auto with_b = first_block_trajectory(*a2, scaled_units, nb, kSteps);

  ASSERT_EQ(with_a.size(), static_cast<std::size_t>(kSteps));
  for (int k = 0; k < kSteps; ++k) {
    for (std::size_t i = 0; i < nb; ++i) {
      // Bit for bit: the row weights are exact powers of two apart, so the
      // scaled least-squares system is literally the same matrix.
      EXPECT_DOUBLE_EQ(with_a[k][i], with_b[k][i])
          << "iteration " << k << " component " << i;
    }
  }

  // Without it, the small field's own residual is invisible next to the large
  // one and is effectively fitted to rounding noise.
  auto b1 = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse = 8,
                                  .scale_fields = false});
  auto b2 = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                  .initial_relaxation = 0.5,
                                  .reuse = 8,
                                  .scale_fields = false});
  const auto without_a = first_block_trajectory(*b1, plain_units, nb, kSteps);
  const auto without_b = first_block_trajectory(*b2, scaled_units, nb, kSteps);

  double worst = 0.0;
  for (int k = 0; k < kSteps; ++k) {
    for (std::size_t i = 0; i < nb; ++i) {
      const double denom = std::abs(without_a[k][i]);
      if (denom > 1e-12) {
        worst = std::max(worst,
                         std::abs(without_a[k][i] - without_b[k][i]) / denom);
      }
    }
  }
  EXPECT_GT(worst, 1.0) << "unscaled fit should be badly unit dependent, worst "
                           "relative difference was " << worst;
}

TEST(QnFieldWeights, DegenerateBlockNormsFallBackToUnitWeight) {
  // The row weight is 1 / ||r_block||. A converged block whose sum of squares
  // underflows to 0 and a block whose sum of squares overflows to inf both
  // produce a weight that is not a usable number; each must fall back to 1
  // rather than inf or 0, because an inf weight multiplied by a zero row is a
  // NaN in the least-squares matrix.
  const double tiny = 1e-320;  // subnormal: its square underflows to exactly 0
  const double huge = 1e200;   // its square overflows to inf
  const std::vector<double> r{tiny, tiny, huge, huge, 3.0, 4.0};
  std::vector<double> w;
  fuse::detail::field_weights({0, 2, 4}, r.data(), r.size(), w);

  ASSERT_EQ(w.size(), 6u);
  EXPECT_DOUBLE_EQ(w[0], 1.0) << "underflowed block norm";
  EXPECT_DOUBLE_EQ(w[1], 1.0);
  EXPECT_DOUBLE_EQ(w[2], 1.0) << "overflowed block norm";
  EXPECT_DOUBLE_EQ(w[3], 1.0);
  EXPECT_DOUBLE_EQ(w[4], 0.2) << "1 / hypot(3, 4)";
  EXPECT_DOUBLE_EQ(w[5], 0.2);
  for (const double e : w) EXPECT_TRUE(std::isfinite(e));

  // No partition is not "one block": it is *no* scaling at all. The weights
  // come back as exactly 1.0 and no reciprocal norm is formed, which is why a
  // test that leaves the partition at its default cannot tell scale_fields
  // = true from scale_fields = false.
  fuse::detail::field_weights({}, r.data(), r.size(), w);
  for (const double e : w) EXPECT_DOUBLE_EQ(e, 1.0);

  // An explicit single block does form the scalar, and it is not 1.
  const std::vector<double> unit{3.0, 4.0, 12.0};
  fuse::detail::field_weights({0}, unit.data(), unit.size(), w);
  ASSERT_EQ(w.size(), 3u);
  for (const double e : w) EXPECT_DOUBLE_EQ(e, 1.0 / 13.0) << "1 / ||r||";
  fuse::detail::field_weights({}, unit.data(), unit.size(), w);
  for (const double e : w) EXPECT_DOUBLE_EQ(e, 1.0) << "the default partition";
}

TEST(Accelerator, AFieldBlockConvergedToUnderflowKeepsTheUpdateFinite) {
  // End to end: one field block has collapsed to subnormal magnitude while
  // the other is still O(1). Whichever layer catches it, the accelerator must
  // return a finite iterate and report success.
  auto a = must_make(AccelConfig{.kind = AccelKind::IQN_ILS,
                                 .initial_relaxation = 0.5,
                                 .reuse = 8,
                                 .scale_fields = true});
  a->set_field_partition({0, 2});

  const double tiny = 1e-320;
  std::vector<double> out(4, 0.0);
  const std::vector<double> o0{0.0, 0.0, 0.0, 0.0};
  const std::vector<double> n0{tiny, 2.0 * tiny, 1.0, 2.0};
  ASSERT_TRUE(a->update(o0.data(), n0.data(), out.data(), 4, 0).ok());

  const std::vector<double> o1{0.5 * tiny, tiny, 0.5, 1.0};
  const std::vector<double> n1{0.7 * tiny, 1.4 * tiny, 1.3, 2.6};
  const auto r = a->update(o1.data(), n1.data(), out.data(), 4, 1);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
  for (std::size_t i = 0; i < 4; ++i) {
    EXPECT_TRUE(std::isfinite(out[i])) << "component " << i << " = " << out[i];
  }
}

TEST(Accelerator, ImvjRetainedJacobianIsBounded) {
  // The history depth policy is only enforced if something enforces it. IMVJ
  // chains at most min(reuse, kMaxJacobianBlocks) past windows; without a cap
  // the retained Jacobian grows by one low-rank block per time window for the
  // whole life of the run, which is a slow leak and a growing per-iteration
  // cost. Each window here perturbs the spectrum, so every window really does
  // have something new to contribute and the cap is what stops the growth --
  // on a fixed linear problem the retained Jacobian becomes exact after the
  // first window and no further blocks are ever produced.
  const auto window_problem = [](int w) {
    const double e = 0.01 * static_cast<double>(w % 17);
    return dense_map({0.95 - e, 0.9 + e, 0.8 - e, 0.5 + e, -0.7 + e},
                     {1.0, 2.0, 3.0, 4.0, 5.0},
                     {1.0 + e, -2.0, 0.5, 3.0, -1.5});
  };

  const std::size_t cap8 = std::min<std::size_t>(
      8, static_cast<std::size_t>(Accelerator::kMaxJacobianBlocks));
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
  EXPECT_EQ(a->retained_blocks(), 0u);
  for (int window = 0; window < 200; ++window) {
    ASSERT_GT(iterate_linear(*a, window_problem(window), kTol, 500), 0)
        << "window " << window;
    a->reset();
    ASSERT_LE(a->retained_blocks(), cap8) << "window " << window;
  }
  EXPECT_EQ(a->retained_blocks(), cap8) << "the cap must actually bite";

  // `reuse` below the block cap is what limits it instead.
  auto small = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 2});
  for (int window = 0; window < 40; ++window) {
    ASSERT_GT(iterate_linear(*small, window_problem(window), kTol, 500), 0)
        << "window " << window;
    small->reset();
    ASSERT_LE(small->retained_blocks(), 2u) << "window " << window;
  }
  EXPECT_EQ(small->retained_blocks(), 2u);

  // Nothing is retained by the kinds that do not claim to.
  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Secant,
                            AccelKind::Steffensen, AccelKind::IQN_ILS}) {
    auto other = must_make(AccelConfig{.kind = k, .initial_relaxation = 0.5});
    for (int window = 0; window < 5; ++window) {
      iterate_linear(*other, window_problem(window), kTol, 2000);
      other->reset();
    }
    EXPECT_EQ(other->retained_blocks(), 0u) << fuse::to_cstring(k);
  }
}

// ---------------------------------------------------------------------------
// The hot loop. Spec Section 2 layer 3 requires the interpreted sweep loop to
// allocate nothing, and an accelerated implicit coupling group calls update() from inside that
// loop, so the accelerators are part of the same requirement -- measured, not
// asserted in a comment.
// ---------------------------------------------------------------------------

namespace {

/// The perturbed window sequence of ImvjRetainedJacobianIsBounded, at file
/// scope so both tests can use it. Every window really does have something new
/// to say, so IMVJ folds a block at every boundary and its chain reaches the
/// cap rather than going quiet after the first window.
LinearMap perturbed_problem(int w) {
  const double e = 0.01 * static_cast<double>(w % 17);
  return dense_map({0.95 - e, 0.9 + e, 0.8 - e, 0.5 + e, -0.7 + e},
                   {1.0, 2.0, 3.0, 4.0, 5.0}, {1.0 + e, -2.0, 0.5, 3.0, -1.5});
}

/// g.apply() into caller-owned storage. LinearMap::apply returns a vector by
/// value, which would be counted against the accelerator.
void apply_into(const LinearMap& g, const std::vector<double>& x,
                std::vector<double>& out) {
  for (std::size_t i = 0; i < g.n; ++i) {
    double s = g.p[i];
    for (std::size_t j = 0; j < g.n; ++j) s += g.a[i * g.n + j] * (x[j] - g.p[j]);
    out[i] = s;
  }
}

/// One window, with every buffer supplied by the caller and no assertion
/// inside: this runs under the allocation counter, where a failing EXPECT_ --
/// or anything else that formats a message -- would allocate and be blamed on
/// the accelerator. Returns false if update() reported an error.
bool run_window_no_alloc(Accelerator& accel, const LinearMap& g, int iters,
                         std::vector<double>& x, std::vector<double>& gx,
                         std::vector<double>& next) {
  for (std::size_t i = 0; i < g.n; ++i) x[i] = 0.0;
  for (int k = 0; k < iters; ++k) {
    apply_into(g, x, gx);
    double residual = 0.0;
    for (std::size_t i = 0; i < g.n; ++i) {
      const double d = gx[i] - x[i];
      residual += d * d;
    }
    if (std::sqrt(residual) < kTol) break;
    if (!accel.update(x.data(), gx.data(), next.data(), g.n, k).ok()) {
      return false;
    }
    for (std::size_t i = 0; i < g.n; ++i) x[i] = next[i];
  }
  accel.reset();
  return true;
}

}  // namespace

TEST(Accelerator, SteadyStateUpdateAllocatesNothing) {
  // MEASURED BEFORE THE FIX, so this is not a test that passes against the bug
  // it is named for: AitkenAccel::update() built a std::vector<double> r(n) per
  // call and the quasi-Newton members built their whole least-squares
  // workspace per call, which came to 56 allocations over 56 updates in
  // Iteration.TheAcceleratorAllocatesNothingPerUpdate and 5088 over these
  // eight windows for IQN-IMVJ. Every one of those buffers is now a member.
  //
  // The first 24 windows are warm-up and are not counted. That is not slack in
  // the assertion -- a buffer has to be allocated once before it can be reused,
  // "steady state" is the claim, and IMVJ in particular has to fill its chain
  // of eight retained blocks before a window boundary can recycle one. What is
  // counted is eight further windows of exactly the same shape, in which
  // nothing is new.
  std::vector<LinearMap> maps;
  maps.reserve(32);
  for (int w = 0; w < 32; ++w) maps.push_back(perturbed_problem(w));

  for (const AccelKind k : {AccelKind::None, AccelKind::Constant,
                            AccelKind::Aitken, AccelKind::Steffensen,
                            AccelKind::Secant, AccelKind::IQN_ILS,
                            AccelKind::IQN_IMVJ}) {
    auto a = must_make(AccelConfig{
        .kind = k, .initial_relaxation = 0.5, .omega = 0.5, .reuse = 8});
    const std::size_t   n = maps.front().n;
    std::vector<double> x(n, 0.0), gx(n, 0.0), next(n, 0.0);

    bool ok = true;
    for (int w = 0; w < 24; ++w) {
      ok = run_window_no_alloc(*a, maps[static_cast<std::size_t>(w)], 40, x, gx,
                               next) && ok;
    }
    ASSERT_TRUE(ok) << fuse::to_cstring(k) << ": the warm-up must not diverge, "
                       "or the counted phase measures a different code path";

    std::size_t allocations = 0;
    {
      AllocationCounter counter;
      for (int w = 24; w < 32; ++w) {
        ok = run_window_no_alloc(*a, maps[static_cast<std::size_t>(w)], 40, x,
                                 gx, next) && ok;
      }
      allocations = counter.count();
    }
    std::printf("%-10s 8 windows after warm-up: %zu allocations, %zu columns, "
                "%zu retained blocks\n",
                fuse::to_cstring(k), allocations, a->columns(),
                a->retained_blocks());
    EXPECT_TRUE(ok) << fuse::to_cstring(k);
    EXPECT_EQ(allocations, 0u) << fuse::to_cstring(k)
                               << ": update() must not reach the allocator in "
                                  "the sweep loop";
  }
}

// ---------------------------------------------------------------------------
// What a divergence costs IQN-IMVJ's retained Jacobian.
//
// The two tests below are the two halves of one decision, and each fails
// against the opposite blanket policy: the first fails if the chain is kept
// unconditionally (which is what the code did before), the second fails if it
// is cleared unconditionally. See the long comment in accelerated_step().
// ---------------------------------------------------------------------------

TEST(Accelerator, ImvjDropsARetainedJacobianThatIsWhatDiverged) {
  // A window in which the iterate moved by 1e308 while the residual moved by
  // 1 folds into a retained block with |Z| / |V| ~ 1e308: a legitimate window
  // by every test the accelerator applies, and an inverse Jacobian that maps
  // any residual bigger than about 2 to infinity. That is exactly the shape a
  // stalled solver produces, and it is the case the retained Jacobian -- not
  // the current window -- is what blows up.
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
  std::vector<double> out(1, 0.0);

  const double o0[1] = {-1.0}, n0[1] = {0.0};
  ASSERT_TRUE(a->update(o0, n0, out.data(), 1, 0).ok());
  const double o1[1] = {1e308}, n1[1] = {1e308};
  ASSERT_TRUE(a->update(o1, n1, out.data(), 1, 1).ok());
  ASSERT_EQ(a->columns(), 1u) << "the poisoning window must really be stored";
  a->reset();
  ASSERT_EQ(a->retained_blocks(), 1u) << "and must really be folded in";

  // Applying it to a residual of 10 overflows.
  const double o2[1] = {0.0}, n2[1] = {10.0};
  const auto r = a->update(o2, n2, out.data(), 1, 0);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED");
  EXPECT_TRUE(std::isfinite(out[0])) << out[0];
  EXPECT_DOUBLE_EQ(out[0], 5.0) << "the postcondition: the under-relaxed "
                                   "fallback, 0 + 0.5 * (10 - 0)";
  EXPECT_EQ(a->retained_blocks(), 0u)
      << "the retained Jacobian is what mapped a finite residual to infinity, "
         "so it must not survive the divergence";

  // The point of dropping it. With the chain still in place this repeats the
  // same overflow on every later call, for ever, and the caller has an
  // accelerator that can never recover.
  const auto again = a->update(o2, n2, out.data(), 1, 1);
  EXPECT_TRUE(again.ok()) << again.error().message;
  EXPECT_TRUE(std::isfinite(out[0])) << out[0];
}

TEST(Accelerator, ImvjKeepsARetainedJacobianTheDivergenceDidNotComeFrom) {
  // The other half. Here the retained block is ordinary and the blow-up is in
  // the current window's own least squares: component 0 moves by 1e300 between
  // the two iterates while contributing nothing to the residual, and component
  // 1's residual moves by 2^-40, so the fit's coefficient is ~1e12 and
  // Wn alpha overflows. Dropping the whole retained Jacobian here would throw
  // away every past window because of one bad extrapolation in this one --
  // which is IQN-ILS with extra steps, and the opposite of what IMVJ is for.
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
  std::vector<double> out(2, 0.0);

  const double h0o[2] = {0.0, 0.0}, h0n[2] = {1.0, 2.0};
  ASSERT_TRUE(a->update(h0o, h0n, out.data(), 2, 0).ok());
  const double h1o[2] = {0.5, 1.0}, h1n[2] = {1.25, 2.5};
  ASSERT_TRUE(a->update(h1o, h1n, out.data(), 2, 1).ok());
  a->reset();
  ASSERT_EQ(a->retained_blocks(), 1u);

  constexpr double kTiny = 1.0 / 1099511627776.0;  // 2^-40, exactly
  const double p0o[2] = {0.0, -1.0}, p0n[2] = {0.0, 0.0};
  ASSERT_TRUE(a->update(p0o, p0n, out.data(), 2, 0).ok());
  const double p1o[2] = {1e300, -1.0}, p1n[2] = {1e300, kTiny};
  const auto r = a->update(p1o, p1n, out.data(), 2, 1);
  ASSERT_FALSE(r.ok()) << "the window really must blow up, or this test is "
                          "asserting retention of an untested Jacobian";
  EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED");
  for (const double v : out) EXPECT_TRUE(std::isfinite(v)) << v;
  EXPECT_EQ(a->columns(), 0u) << "the window's columns go, as always";
  EXPECT_EQ(a->retained_blocks(), 1u)
      << "the retained Jacobian was not what diverged -- the step without it "
         "overflows too -- so it must survive";
}

TEST(Accelerator, ImvjDropsAPoisonedChainEvenWhenEveryProbeExoneratesIt) {
  // The two tests above, interleaved -- which is what a stalled solver
  // actually produces, and which defeats the probe. The chain is poisoned
  // exactly as in the first; every subsequent call is shaped like the second,
  // so the current window's own least squares overflows too and the probe
  // says "not the chain" EVERY TIME. The probe is not wrong about what it
  // measures; it is answering "would this step have been finite without the
  // chain", and the answer is no while the chain is also unusable.
  //
  // MEASURED against the first version of this fix, which had no reprieve
  // budget and argued instead that a second divergence would be judged at
  // cols_ == 0: 11 consecutive E-ACCEL-DIVERGED calls with the chain retained
  // throughout, and a fresh benign input still diverging afterwards. The
  // argument was false because `have_prev_` survives the diverged return, so
  // append_column() regenerates a column at the top of the next call, before
  // the probe runs. This test is that sequence.
  auto a = must_make(AccelConfig{
      .kind = AccelKind::IQN_IMVJ, .initial_relaxation = 0.5, .reuse = 8});
  std::vector<double> out(2, 0.0);

  // Poison: the iterate moves by 1e308 while the residual moves by 1.
  const double z0o[2] = {-1.0, 0.0}, z0n[2] = {0.0, 0.0};
  ASSERT_TRUE(a->update(z0o, z0n, out.data(), 2, 0).ok());
  const double z1o[2] = {1e308, 0.0}, z1n[2] = {1e308, 0.0};
  ASSERT_TRUE(a->update(z1o, z1n, out.data(), 2, 1).ok());
  a->reset();
  ASSERT_EQ(a->retained_blocks(), 1u);

  constexpr double kTiny = 1.0 / 1099511627776.0;  // 2^-40, exactly
  const double pa_o[2] = {0.0, -1.0}, pa_n[2] = {0.0, 0.0};
  const double pb_o[2] = {1e300, -1.0}, pb_n[2] = {1e300, kTiny};

  int  failures_holding_the_chain = 0;
  bool chain_gone                 = false;
  for (int k = 0; k < 12; ++k) {
    const bool even = (k % 2) == 0;
    const auto r    = a->update(even ? pa_o : pb_o, even ? pa_n : pb_n,
                                out.data(), 2, k);
    // The postcondition holds on every one of these, failed or not.
    for (const double v : out) EXPECT_TRUE(std::isfinite(v)) << "call " << k;
    if (!r.ok()) {
      EXPECT_EQ(r.error().code, "E-ACCEL-DIVERGED") << "call " << k;
      if (!chain_gone) ++failures_holding_the_chain;
    }
    if (a->retained_blocks() == 0) chain_gone = true;
  }

  EXPECT_TRUE(chain_gone)
      << "the retained Jacobian outlasted twelve failed calls: the probe "
         "exonerates it every time, so something other than the probe has to "
         "bound how long it can survive";
  EXPECT_GE(failures_holding_the_chain, 1)
      << "if the chain went at the very first divergence this test is not "
         "measuring the budget, it is measuring unconditional clearing";
  EXPECT_LE(failures_holding_the_chain, 2)
      << "at most kMaxChainReprieves + 1 divergences may be spent on one "
         "retained Jacobian";

  // The point of the bound: an ordinary input works again afterwards. Against
  // the unbudgeted version this call returns E-ACCEL-DIVERGED.
  const double f0[2] = {0.0, 0.0}, f1[2] = {10.0, 0.0};
  const auto   fresh = a->update(f0, f1, out.data(), 2, 0);
  EXPECT_TRUE(fresh.ok()) << fresh.error().message;
  EXPECT_DOUBLE_EQ(out[0], 5.0);
  EXPECT_DOUBLE_EQ(out[1], 0.0);
}
