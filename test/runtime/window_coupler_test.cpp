// Verification-tier tests for the "Time windows" runtime driver
// (fuse/runtime/window_coupler.hpp): a linear two-way coupled ODE with a
// closed-form exact solution, run at commensurate ratios (1, 2, 4) and the
// incommensurate 7:13 case, under Fixed/Fixed, Adjustable/Adjustable and a
// mixed pair, checking order, time conservation, m=1 equivalence with
// today's single-clock region, agreement with a monolithic solve of the
// same discretization, and iteration counts across accelerators.

#include <gtest/gtest.h>
#include <fuse/runtime/window_coupler.hpp>

#include <cmath>
#include <utility>
#include <memory>
#include <string>
#include <vector>

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::Component;
using fuse::CouplingGraph;
using fuse::DtPolicy;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Result;
using fuse::StepContext;
using fuse::Waveform;
using fuse::WindowChannel;
using fuse::WindowCoupler;
using fuse::WindowCouplerConfig;
using fuse::WindowMember;

namespace {

// dx/dt = -a*x + b*u, `u` the MEAN forcing over [t_begin, t_end) --
// window_coupler.hpp's class comment explains why a WindowCoupler-driven
// component receives a mean, not a two-point trajectory: it is what the
// real `Component::advance()` contract (one snapshot per call) allows. This
// makes `advance()` EXACT given that mean, isolating the coupling scheme's
// own error from any ODE-integrator truncation error.
class LinearOdeComponent : public Component {
 public:
  LinearOdeComponent(std::string name, double a, double b, double x0)
      : name_(std::move(name)), a_(a), b_(b), x_(x0), provided_(x0) {}

  [[nodiscard]] std::string name() const override { return name_; }
  Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
    return Result<void>::success();  // WindowCoupler binds fields directly
  }
  Result<void> advance(const StepContext& ctx) override {
    const double h = ctx.dt().to_double();
    if (std::abs(a_) < 1e-300) {
      x_ = x_ + b_ * required_ * h;
    } else {
      const double e = std::exp(-a_ * h);
      x_ = x_ * e + b_ * required_ * (1.0 - e) / a_;
    }
    provided_ = x_;
    return Result<void>::success();
  }
  Result<void> save_state() override {
    saved_ = x_;
    return Result<void>::success();
  }
  Result<void> restore_state() override {
    x_ = saved_;
    provided_ = x_;
    return Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

  [[nodiscard]] double* provided_ptr() { return &provided_; }
  [[nodiscard]] double* required_ptr() { return &required_; }
  [[nodiscard]] double x() const { return x_; }

 private:
  std::string name_;
  double      a_, b_;
  double      x_;
  double      provided_;
  double      required_ = 0.0;
  double      saved_ = 0.0;
};

// The test problem: dx/dt = -1.3x + 0.7y, dy/dt = -0.9y + 0.5x,
// x(0)=1.0, y(0)=0.2. Closed form via eigen-decomposition of
// M = [[-a,b],[d,-c]] (b != 0, real distinct eigenvalues for these
// coefficients -- checked once, offline, against scipy.linalg.expm to
// round-off).
constexpr double kA = 1.3, kB = 0.7, kC = 0.9, kD = 0.5;
constexpr double kX0 = 1.0, kY0 = 0.2;

struct ExactSolution {
  double x, y;
};

[[nodiscard]] ExactSolution exact_reference(double t) {
  const double trace = -kA - kC;
  const double det = kA * kC - kB * kD;
  const double disc = trace * trace - 4.0 * det;
  const double lam1 = (trace + std::sqrt(disc)) / 2.0;
  const double lam2 = (trace - std::sqrt(disc)) / 2.0;
  const double c2 = (kY0 - kX0 * (kA + lam1) / kB) / (lam2 - lam1);
  const double c1 = kX0 / kB - c2;
  const double e1 = std::exp(lam1 * t);
  const double e2 = std::exp(lam2 * t);
  return {kB * (c1 * e1 + c2 * e2),
          c1 * (kA + lam1) * e1 + c2 * (kA + lam2) * e2};
}

struct RunResult {
  double           xA = 0.0, xB = 0.0;
  std::vector<int> iterations;
};

// A <-> B with an EXPLICIT window length independent of either member's own
// interval -- the shape a window neither step divides needs (see
// `AWindowNeitherStepDividesConvergesAtOrderTwo` below). `run_two_member` (next) is
// the common case where the window length equals A's own interval, which is
// what makes A the window-definer and structurally incapable of
// straddling.
[[nodiscard]] RunResult run_two_member_general(
    Rational window_length, Rational intervalA, Rational intervalB,
    int n_windows, DtPolicy pol_a, DtPolicy pol_b,
    AccelKind accel = AccelKind::Aitken, int order = 2,
    std::size_t waveform_capacity = 4096,
    fuse::WaveformIteration mode = fuse::WaveformIteration::Jacobi,
    bool lag_b_to_a = false) {
  auto comp_a = std::make_unique<LinearOdeComponent>("A", kA, kB, kX0);
  auto comp_b = std::make_unique<LinearOdeComponent>("B", kC, kD, kY0);

  WindowMember mA;
  mA.component = comp_a.get();
  mA.name = "A";
  mA.interval = intervalA;
  mA.policy = pol_a;
  mA.provided = comp_a->provided_ptr();
  mA.required = comp_a->required_ptr();
  mA.length = 1;

  WindowMember mB;
  mB.component = comp_b.get();
  mB.name = "B";
  mB.interval = intervalB;
  mB.policy = pol_b;
  mB.provided = comp_b->provided_ptr();
  mB.required = comp_b->required_ptr();
  mB.length = 1;

  std::vector<WindowMember> members;
  members.push_back(std::move(mA));
  members.push_back(std::move(mB));
  // a2b (A -> B) is never lagged here: under WaveformIteration::GaussSeidel
  // with lag_b_to_a set, b2a (B -> A) is the one broken direction -- the
  // same "exactly one lagged, one not" shape every other window fixture in
  // this codebase uses (Unit B's own two-way IR/schedule golden fixture).
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .lagged = false},
      {.provider = 1, .consumer = 0, .lagged = lag_b_to_a}};

  WindowCouplerConfig cfg;
  cfg.window_length = window_length;
  cfg.max_iterations = 300;
  cfg.acceleration = AccelConfig{.kind = accel};
  cfg.order = order;
  cfg.waveform_capacity = waveform_capacity;
  cfg.iteration = mode;

  Result<std::unique_ptr<WindowCoupler>> created =
      WindowCoupler::create(members, channels, cfg);
  if (!created.ok()) {
    ADD_FAILURE() << "WindowCoupler::create failed: " << created.error().message;
    return {};
  }
  std::unique_ptr<WindowCoupler> coupler = std::move(created.value());

  RunResult out;
  for (int w = 0; w < n_windows; ++w) {
    const Result<void> r = coupler->run_window();
    if (!r.ok()) {
      ADD_FAILURE() << "run_window() failed at window " << w << ": "
                    << r.error().message;
      return out;
    }
    out.iterations.push_back(coupler->stats().iterations);
  }
  out.xA = comp_a->x();
  out.xB = comp_b->x();
  return out;
}

// A <-> B, window length `Tc` (A's own interval; A is the window-definer,
// and therefore structurally incapable of straddling regardless of its
// DtPolicy), B's interval `hB`. Runs `n_windows` back to back and returns
// the final state plus each window's iteration count.
[[nodiscard]] RunResult run_two_member(
    Rational Tc, Rational hB, int n_windows, DtPolicy pol_a, DtPolicy pol_b,
    AccelKind accel = AccelKind::Aitken, int order = 2,
    std::size_t waveform_capacity = 4096,
    fuse::WaveformIteration mode = fuse::WaveformIteration::Jacobi,
    bool lag_b_to_a = false) {
  RunResult out = run_two_member_general(Tc, Tc, hB, n_windows, pol_a, pol_b,
                                         accel, order, waveform_capacity, mode,
                                         lag_b_to_a);
  return out;
}

}  // namespace

namespace {

// advance() sets its OWN provided value equal to whatever mean it was
// just fed, ignoring its previous state entirely -- so a member's own
// output on ANY iterate is exactly, closed-form, whatever it was handed,
// with no ODE dynamics to fold in. Records every (iteration, required)
// pair it was ever called with, which is what
// GaussSeidelReadsThePartnersSameIterateOutput below inspects directly,
// rather than trying to back out "what delivered[] must have held" from
// an opaque final state.
class IdentityRecordingComponent : public Component {
 public:
  explicit IdentityRecordingComponent(double x0) : x_(x0), provided_(x0) {}
  [[nodiscard]] std::string name() const override { return "identity"; }
  Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext& ctx) override {
    seen.push_back({ctx.iteration, required_});
    x_ = required_;
    provided_ = x_;
    return Result<void>::success();
  }
  Result<void> save_state() override { saved_ = x_; return Result<void>::success(); }
  Result<void> restore_state() override {
    x_ = saved_; provided_ = x_; return Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }
  [[nodiscard]] double* provided_ptr() { return &provided_; }
  [[nodiscard]] double* required_ptr() { return &required_; }

  struct Seen { int iteration; double required; };
  std::vector<Seen> seen;

 private:
  double x_, provided_, required_ = 0.0, saved_ = 0.0;
};

// A (x0=5) -> B (x0=100) unlagged; B -> A lagged (so the graph has a lag
// somewhere, as WaveformIteration::GaussSeidel needs, without it being
// the direction under test). Window length 1, both intervals 1: exactly
// one substep each, no subcycling, so the mean channel_io_->mean() reads
// is the plain 2-knot linear average -- easy closed-form arithmetic, not
// an empirically-measured number.
struct TwoMemberIdentityRig {
  IdentityRecordingComponent a{5.0};
  IdentityRecordingComponent b{100.0};
  std::unique_ptr<WindowCoupler> coupler;
};

// `a_declared_first`: whether A (the UNLAGGED provider -- a2b -- the one
// Gauss-Seidel's own topological order must run BEFORE its consumer B)
// occupies `members[0]` or `members[1]`. The declared ORDER must not
// change the result: `WindowCoupler::create()` derives the actual sweep
// order from the unlagged-channel graph, not from this vector's own
// index order (fuse/runtime/window_coupler.cpp's own
// topological_sweep_order()). Channel indices are looked up by role
// (`a_idx`/`b_idx`), never hardcoded, so this genuinely exercises the
// OTHER declaration order rather than silently building the same one
// twice under a different name.
[[nodiscard]] std::unique_ptr<TwoMemberIdentityRig> make_identity_rig_ordered(
    fuse::WaveformIteration mode, bool a_declared_first,
    int max_iterations = 1,
    AccelConfig accel = AccelConfig{.kind = AccelKind::None}) {
  auto rig = std::make_unique<TwoMemberIdentityRig>();
  WindowMember mA;
  mA.component = &rig->a;
  mA.name = "A";
  mA.interval = Rational{1};
  mA.policy = DtPolicy::Adjustable;
  mA.provided = rig->a.provided_ptr();
  mA.required = rig->a.required_ptr();
  mA.length = 1;
  WindowMember mB = mA;
  mB.component = &rig->b;
  mB.name = "B";
  mB.provided = rig->b.provided_ptr();
  mB.required = rig->b.required_ptr();

  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.max_iterations = max_iterations;
  cfg.acceleration = accel;
  cfg.iteration = mode;

  const std::size_t a_idx = a_declared_first ? 0 : 1;
  const std::size_t b_idx = a_declared_first ? 1 : 0;
  std::vector<WindowMember> members(2);
  members[a_idx] = mA;
  members[b_idx] = mB;
  const std::vector<WindowChannel> channels = {
      {.provider = a_idx, .consumer = b_idx, .lagged = false},
      {.provider = b_idx, .consumer = a_idx, .lagged = true}};
  auto created = WindowCoupler::create(members, channels, cfg);
  if (!created.ok()) {
    ADD_FAILURE() << "WindowCoupler::create failed: " << created.error().message;
    return rig;
  }
  rig->coupler = std::move(created.value());
  return rig;
}

[[nodiscard]] std::unique_ptr<TwoMemberIdentityRig> make_identity_rig(
    fuse::WaveformIteration mode, int max_iterations = 1,
    AccelConfig accel = AccelConfig{.kind = AccelKind::None}) {
  return make_identity_rig_ordered(mode, /*a_declared_first=*/true,
                                   max_iterations, accel);
}

}  // namespace

// THE defining property: within one iterate, GaussSeidel's consumer reads
// the PROVIDER'S OWN OUTPUT from that SAME iterate, not the sweep-start
// snapshot Jacobi freezes for the whole sweep. Closed form: A's own
// window-start value is 5, B's is 100. A -> B is unlagged, so on iterate
// 0, B's fed mean is the 2-knot linear average of A's [window-start,
// window-end] values. Window-start is always 5 regardless of mode; what
// differs is window-end:
//   Jacobi:      window-end is STILL the seeded initial guess (5, flat-
//                extrapolated from A's own one-point history) -- A hasn't
//                produced anything yet from THIS mode's point of view, so
//                B's mean = (5+5)/2 = 5.
//   GaussSeidel: A ran first (the unlagged provider must precede its
//                consumer), and its own iterate-0 OUTPUT is exactly
//                whatever it was fed (identity dynamics) -- A's own
//                required on iterate 0 is B's seeded value (100, by the
//                same flat-extrapolation argument, since B->A is lagged
//                and B has produced nothing yet either). So A's
//                window-end value is 100, and B's mean = (5+100)/2 = 52.5.
// Acceleration acts on the LAGGED direction only: B provides only the
// lagged channel (b2a), so B's own value going into iterate 1 is what
// Aitken's `initial_relaxation` (default 0.1, AccelConfig's own default
// -- deliberately not overridden, so this exercises the SAME factor
// every other WindowCoupler test in this file already runs under)
// relaxes; A provides only the unlagged channel (a2b), so A's own value
// is copied straight through, never relaxed, whatever the omega.
//
// Closed form for iterate 0 -> 1, continuing GaussSeidelReadsThePartners
// SameIterateOutput's own iterate-0 arithmetic (A=100, B=52.5 there):
//   B's relaxed value feeding iterate 1: x_old=100 (B's own seed), x_new=
//     52.5 (B's iterate-0 output), omega0=0.1 ->
//     100 + 0.1*(52.5-100) = 95.25.
//   A's required on iterate 1 (mean of B's window-start=100 and this
//     relaxed value): (100+95.25)/2 = 97.625 -- RELAXED, not the
//     unrelaxed (100+52.5)/2 = 76.25 an omega-blind implementation
//     would produce.
//   B's required on iterate 1 (mean of A's window-start=5 and A's iterate-
//     1 output, UNRELAXED since A is excluded from acceleration):
//     (5+97.625)/2 = 51.3125.
// Iterate 1 -> 2 is where "the accelerated vector is not restricted"
// (build-evidence/mutations/window-runner.toml's own row for this) would
// actually surface: Aitken's own omega is still just `initial_relaxation`
// on the FIRST accelerated call (`have_prev_` is false), so including A
// in that vector changes nothing observable YET -- but by the SECOND
// accelerated call (the one that produces iterate 2's feed), Aitken's own
// formula aggregates the PREVIOUS residual across every entry in the
// vector it was called with, so a wrongly-included A residual changes
// the omega applied to B too, and through it what A itself reads next.
// Measured (this build) rather than hand-carried through Aitken's own
// formula: iterate 2 and 3 both read 68.33333333333333 (A) / 36.
// 666666666666664 (B) -- the system reaches an exact fixed point by
// iterate 2, so iterate 3 repeating it is itself part of what this pins.
TEST(WindowCouplerVerification, GaussSeidelAccelerationActsOnlyOnTheLaggedDirection) {
  auto rig = make_identity_rig(fuse::WaveformIteration::GaussSeidel, /*max_iterations=*/4,
                               AccelConfig{.kind = AccelKind::Aitken});
  (void)rig->coupler->run_window();
  ASSERT_EQ(rig->a.seen.size(), 4u);
  ASSERT_EQ(rig->b.seen.size(), 4u);
  EXPECT_NEAR(rig->a.seen[0].required, 100.0, 1e-9);
  EXPECT_NEAR(rig->b.seen[0].required, 52.5, 1e-9);
  EXPECT_NEAR(rig->a.seen[1].required, 97.625, 1e-9)
      << "the LAGGED direction (A consuming B) should show Aitken's "
         "relaxation";
  EXPECT_NEAR(rig->b.seen[1].required, 51.3125, 1e-9)
      << "the UNLAGGED direction (B consuming A) must NOT be relaxed";
  EXPECT_NEAR(rig->a.seen[2].required, 68.33333333333333, 1e-9)
      << "iterate 2's feed depends on Aitken's SECOND accelerated call, "
         "which is where an unrestricted vector would pollute omega";
  EXPECT_NEAR(rig->b.seen[2].required, 36.666666666666664, 1e-9);
  EXPECT_NEAR(rig->a.seen[3].required, 68.33333333333333, 1e-9)
      << "reached an exact fixed point by iterate 2";
  EXPECT_NEAR(rig->b.seen[3].required, 36.666666666666664, 1e-9);
}

// The sweep order Gauss-Seidel actually uses is derived from the
// UNLAGGED-channel graph (topological_sweep_order(), src/runtime/
// window_coupler.cpp), never from `members`' own declaration order --
// so declaring the unlagged provider (A) SECOND instead of first must
// not change a single recorded value. A mutation that made the sweep
// iterate in bare declaration order instead would pass every OTHER test
// in this file (none of them declares A second), and would fail only
// here: with A declared second, that mutation sweeps B before A, so B
// would read A's STALE (not-yet-advanced-this-sweep) delivered value --
// Jacobi's own behaviour, silently substituted for Gauss-Seidel's.
TEST(WindowCouplerVerification, GaussSeidelResultIsIndependentOfDeclarationOrder) {
  auto standard = make_identity_rig_ordered(fuse::WaveformIteration::GaussSeidel,
                                            /*a_declared_first=*/true, 4,
                                            AccelConfig{.kind = AccelKind::Aitken});
  (void)standard->coupler->run_window();
  auto reordered = make_identity_rig_ordered(fuse::WaveformIteration::GaussSeidel,
                                             /*a_declared_first=*/false, 4,
                                             AccelConfig{.kind = AccelKind::Aitken});
  (void)reordered->coupler->run_window();

  ASSERT_EQ(standard->a.seen.size(), reordered->a.seen.size());
  ASSERT_EQ(standard->b.seen.size(), reordered->b.seen.size());
  for (std::size_t k = 0; k < standard->a.seen.size(); ++k) {
    EXPECT_EQ(standard->a.seen[k].iteration, reordered->a.seen[k].iteration) << "k=" << k;
    EXPECT_EQ(standard->a.seen[k].required, reordered->a.seen[k].required)
        << "k=" << k << " (A's own required, declared first vs second)";
  }
  for (std::size_t k = 0; k < standard->b.seen.size(); ++k) {
    EXPECT_EQ(standard->b.seen[k].iteration, reordered->b.seen[k].iteration) << "k=" << k;
    EXPECT_EQ(standard->b.seen[k].required, reordered->b.seen[k].required)
        << "k=" << k << " (B's own required, declared first vs second)";
  }
}

TEST(WindowCouplerVerification, GaussSeidelReadsThePartnersSameIterateOutput) {
  auto rig = make_identity_rig(fuse::WaveformIteration::GaussSeidel);
  (void)rig->coupler->run_window();  // NOCONVERGE at max_iterations=1 is fine
  ASSERT_FALSE(rig->b.seen.empty());
  EXPECT_EQ(rig->b.seen[0].iteration, 0);
  EXPECT_NEAR(rig->b.seen[0].required, 52.5, 1e-12);
}

TEST(WindowCouplerVerification, JacobiReadsThePartnersInitialGuessNotTheSameIterateOutput) {
  auto rig = make_identity_rig(fuse::WaveformIteration::Jacobi);
  (void)rig->coupler->run_window();
  ASSERT_FALSE(rig->b.seen.empty());
  EXPECT_EQ(rig->b.seen[0].iteration, 0);
  EXPECT_NEAR(rig->b.seen[0].required, 5.0, 1e-12);
}

// ---------------------------------------------------------------- (a)/(b)

TEST(WindowCouplerVerification, CommensurateRatiosConverge) {
  for (const int m : {1, 2, 4}) {
    const RunResult r = run_two_member(Rational{1}, Rational{1, m}, 5,
                                       DtPolicy::Adjustable, DtPolicy::Adjustable);
    const ExactSolution ex = exact_reference(5.0);
    EXPECT_NEAR(r.xA, ex.x, 0.05) << "m=" << m;
    EXPECT_NEAR(r.xB, ex.y, 0.05) << "m=" << m;
    for (const int it : r.iterations) EXPECT_GT(it, 0);
  }
}

TEST(WindowCouplerVerification, IncommensurateSevenThirteenAllPolicyPairs) {
  const struct { DtPolicy a, b; const char* label; } cases[] = {
      {DtPolicy::Fixed, DtPolicy::Fixed, "fixed/fixed"},
      {DtPolicy::Adjustable, DtPolicy::Adjustable, "adjustable/adjustable"},
      {DtPolicy::Fixed, DtPolicy::Adjustable, "mixed A-fixed,B-adjustable"},
      {DtPolicy::Adjustable, DtPolicy::Fixed, "mixed A-adjustable,B-fixed"},
  };
  // Tolerance derived from measured error, not an arbitrary round number
  // against a decayed solution: this window (Tc=13) is deliberately
  // coarse -- kA*Tc = 16.9, far outside the asymptotic regime the order
  // tests above require -- so it exercises the policy combinations, not
  // accuracy at scale. Measured (this build): fixed/fixed and mixed
  // A-adjustable,B-fixed both give errA=1.09e-2, errB=2.36e-2;
  // adjustable/adjustable and mixed A-fixed,B-adjustable both give
  // errA=1.51e-2, errB=1.37e-2. 0.03 is comfortably above the largest of
  // these (2.36e-2) without being the old, unexamined 0.05.
  for (const auto& c : cases) {
    const RunResult r = run_two_member(Rational{13}, Rational{7}, 3, c.a, c.b);
    const ExactSolution ex = exact_reference(39.0);
    EXPECT_NEAR(r.xA, ex.x, 0.03) << c.label;
    EXPECT_NEAR(r.xB, ex.y, 0.03) << c.label;
  }
}

// Asymptotic regime (a*h <= ~0.13 at the coarsest level used): the
// refinement set below is m in {2,4,8,16,32} (base shifted from m=1, see
// the comment on asymptotic_refinements() for why), so the coarsest
// scale is Tc=13/70=0.1857, kA*Tc=0.241 -- itself still coarser than the
// ~0.13 target, which is why the base is shifted rather than the target
// loosened. The finest scale (m=32, d=1120) reaches kA*Tc =
// 1.3*13/1120 = 0.0151, well inside the asymptotic regime. d = 35*m:
// n_windows = 7*m is always a multiple
// of the 7:13 straddle's own phase cycle (gcd(7,13)=1, so the straddle's
// position within its substep repeats every 7 windows) AND
// n_windows*Tc = 7m * 13/(35m) = 91/35 = 2.6 exactly for every m -- fixed
// T_total, fixed final-window phase, varying only the discretization.
// Skipping the phase alignment while holding T_total fixed produces a
// non-monotone, sometimes order-0-looking error sequence purely from
// measuring different runs' final windows at different phases of that
// cycle -- verified directly: the same coefficients and scales, refined by
// naive doubling instead of by this scheme, gave errors 3.8e-3, 1.6e-3,
// 4.9e-4, 5.0e-4, 2.1e-4, 6.3e-5, 6.3e-5, 2.6e-5 (two near-flat steps) for
// Fixed/Fixed, while THIS scheme's phase-aligned sequence is
// monotone to 2-3 significant figures at every step.
struct AsymptoticRefinement {
  std::int64_t d;
  Rational     Tc, hB;
  int          n_windows;
};
[[nodiscard]] std::vector<AsymptoticRefinement> asymptotic_refinements() {
  // Base shifted to m=2 (d=70), not m=1 (d=35): measured directly, the
  // mixed Adjustable/Fixed pair's coarsest step (d=35 -> d=70) has order
  // 2.137, above the [1.9, 2.1] bound the order tests below assert at
  // EVERY step -- the coarsest scale in this file's original 35*m base
  // still carries enough higher-order truncation content to push order
  // just past 2.1 before the asymptotic regime is fully entered.
  // Starting from d=70 keeps every step (measured for all four policy
  // pairs used below) inside [1.9, 2.1].
  std::vector<AsymptoticRefinement> out;
  for (const std::int64_t m : {2LL, 4LL, 8LL, 16LL, 32LL}) {
    const std::int64_t d = 35 * m;
    out.push_back({d, Rational{13, d}, Rational{7, d}, static_cast<int>(7 * m)});
  }
  return out;
}
// Every step's order must land in [1.9, 2.1]; the
// helper below applies that uniformly and reports every step's number on
// failure, not just the first.
void expect_order_two_at_every_step(const std::vector<double>& errs,
                                    const char* label) {
  ASSERT_GE(errs.size(), 2u);
  for (std::size_t i = 0; i + 1 < errs.size(); ++i) {
    const double order = std::log(errs[i] / errs[i + 1]) / std::log(2.0);
    EXPECT_GE(order, 1.9) << label << " refinement step " << i << ": order "
                         << order << " (errs " << errs[i] << " -> "
                         << errs[i + 1] << ")";
    EXPECT_LE(order, 2.1) << label << " refinement step " << i << ": order "
                         << order << " (errs " << errs[i] << " -> "
                         << errs[i + 1] << ")";
  }
}

TEST(WindowCouplerVerification, ErrorConvergesAtOrderTwoAdjustableAdjustable) {
  std::vector<double> errs;
  for (const auto& r : asymptotic_refinements()) {
    const RunResult res = run_two_member(r.Tc, r.hB, r.n_windows,
                                         DtPolicy::Adjustable, DtPolicy::Adjustable);
    const ExactSolution ex = exact_reference(2.6);
    errs.push_back(std::hypot(res.xA - ex.x, res.xB - ex.y));
  }
  expect_order_two_at_every_step(errs, "adjustable/adjustable");
}

TEST(WindowCouplerVerification, ErrorConvergesAtOrderTwoFixedFixedIncommensurate) {
  std::vector<double> errs;
  for (const auto& r : asymptotic_refinements()) {
    const RunResult res =
        run_two_member(r.Tc, r.hB, r.n_windows, DtPolicy::Fixed, DtPolicy::Fixed);
    const ExactSolution ex = exact_reference(2.6);
    errs.push_back(std::hypot(res.xA - ex.x, res.xB - ex.y));
  }
  expect_order_two_at_every_step(errs, "fixed/fixed");
}

TEST(WindowCouplerVerification, ErrorConvergesAtOrderTwoMixedFixedAdjustable) {
  std::vector<double> errs;
  for (const auto& r : asymptotic_refinements()) {
    const RunResult res =
        run_two_member(r.Tc, r.hB, r.n_windows, DtPolicy::Fixed, DtPolicy::Adjustable);
    const ExactSolution ex = exact_reference(2.6);
    errs.push_back(std::hypot(res.xA - ex.x, res.xB - ex.y));
  }
  expect_order_two_at_every_step(errs, "mixed A-fixed,B-adjustable");
}

TEST(WindowCouplerVerification, ErrorConvergesAtOrderTwoMixedAdjustableFixed) {
  std::vector<double> errs;
  for (const auto& r : asymptotic_refinements()) {
    const RunResult res =
        run_two_member(r.Tc, r.hB, r.n_windows, DtPolicy::Adjustable, DtPolicy::Fixed);
    const ExactSolution ex = exact_reference(2.6);
    errs.push_back(std::hypot(res.xA - ex.x, res.xB - ex.y));
  }
  expect_order_two_at_every_step(errs, "mixed A-adjustable,B-fixed");
}

// A genuinely mixed case where BOTH members straddle at once, using an
// explicit window that is a multiple of neither member's own step, would
// ideally carry an order-2 assertion. That assertion is NOT made here --
// two known gaps, unrelated to straddle_window_start, prevent it:
//
// 1. window=10/d: A's interval (13/d)
//    exceeds the window by 30%, so A's overshoot grows window over
//    window (13, 26, 39, 52 against window boundaries 10, 20, 30, 40)
//    until, at the 5th window, A's carried-over clock (52/d) is already
//    PAST that window's own end (50/d) -- a member whose interval so
//    exceeds the window that it must SKIP a window entirely.
//    WindowCoupler's bookkeeping does not handle that: `build_delivered`
//    constructs a Waveform with window_begin > window_end and throws.
//
// 2. window=12/d (which keeps A's per-window excess small, 8%, delaying
//    gap 1 well past any window count tested here -- verified directly by
//    simulating the substep grid alone, first skip at window 12): a
//    DIFFERENT failure appears much earlier. With BOTH A and B
//    straddle_pending simultaneously, `resolve_pending_straddles()`
//    replays them one at a time and one member's replay reads the OTHER
//    member's `last_window_waveform`/`history` -- measured directly,
//    Fixed/Fixed throws "Waveform::integrate: [range] is not inside the
//    support [range]" as early as window 2 (every d value tried).
//    Neither mixed case (one member Fixed, the other Adjustable, so only
//    the Fixed one actually straddles) throws, but both show
//    non-monotone or growing error with refinement, not order 2.
//    Adjustable/Adjustable (neither member straddles, both truncate)
//    comes closest to correct -- order 1.87 at the one step measured,
//    under the 1.9 floor but not crashing or growing. This is a second,
//    independent simultaneous-straddle interaction bug.
//
// The straddle envelope. Outside it the coupler would run and return a
// wrong answer with no diagnostic -- growing error in the mixed cases --
// so it is refused at create(). Inside it, a window
// that neither step divides is still verified to order 2.
namespace {

[[nodiscard]] Result<std::unique_ptr<WindowCoupler>> create_pair(
    LinearOdeComponent& a, LinearOdeComponent& b, Rational window,
    Rational ia, Rational ib, DtPolicy pa, DtPolicy pb) {
  WindowMember ma;
  ma.component = &a; ma.name = "A"; ma.interval = ia; ma.policy = pa;
  ma.provided = a.provided_ptr(); ma.required = a.required_ptr(); ma.length = 1;
  WindowMember mb;
  mb.component = &b; mb.name = "B"; mb.interval = ib; mb.policy = pb;
  mb.provided = b.provided_ptr(); mb.required = b.required_ptr(); mb.length = 1;
  WindowCouplerConfig cfg;
  cfg.window_length = window;
  return WindowCoupler::create({ma, mb}, {{0, 1}, {1, 0}}, cfg);
}

}  // namespace

TEST(WindowCouplerRefusals, AFixedStepLongerThanTheWindowIsRefused) {
  LinearOdeComponent a("A", kA, kB, kX0), b("B", kC, kD, kY0);
  const auto r = create_pair(a, b, Rational{12, 35}, Rational{13, 35},
                             Rational{7, 35}, DtPolicy::Fixed, DtPolicy::Adjustable);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-LONGSTEP") << r.error().message;
}

TEST(WindowCouplerRefusals, TwoFixedMembersThatCanStraddleTogetherAreRefused) {
  // 5 and 7 are both shorter than 12 and neither divides it.
  LinearOdeComponent a("A", kA, kB, kX0), b("B", kC, kD, kY0);
  const auto r = create_pair(a, b, Rational{12, 35}, Rational{5, 35},
                             Rational{7, 35}, DtPolicy::Fixed, DtPolicy::Fixed);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-STRADDLE") << r.error().message;
}

TEST(WindowCouplerRefusals, AFixedStepThatDividesTheWindowNeverStraddlesAndIsAccepted) {
  // The envelope counts only members that can straddle: 4 divides 12.
  LinearOdeComponent a("A", kA, kB, kX0), b("B", kC, kD, kY0);
  const auto r = create_pair(a, b, Rational{12, 35}, Rational{4, 35},
                             Rational{7, 35}, DtPolicy::Fixed, DtPolicy::Fixed);
  EXPECT_TRUE(r.ok()) << r.error().message;
}

// Inside the envelope with a window NEITHER step divides: A is Adjustable
// with a step longer than the window (so it truncates to it), B is Fixed
// and straddles. Phase-aligned like the other order tests: 12k mod 7 repeats
// every 7 windows, so n_windows = 7m at d = 35m holds T = 2.4 fixed.
TEST(WindowCouplerVerification, AWindowNeitherStepDividesConvergesAtOrderTwo) {
  const DtPolicy pol_a[] = {DtPolicy::Adjustable, DtPolicy::Adjustable};
  const DtPolicy pol_b[] = {DtPolicy::Fixed, DtPolicy::Adjustable};
  for (int c = 0; c < 2; ++c) {
    double errs[5];
    for (int k = 0; k < 5; ++k) {
      const std::int64_t m = std::int64_t{2} << k;
      const RunResult r = run_two_member_general(
          Rational{12, 35 * m}, Rational{13, 35 * m}, Rational{7, 35 * m},
          static_cast<int>(7 * m), pol_a[c], pol_b[c]);
      const ExactSolution ex = exact_reference(2.4);
      errs[k] = std::hypot(r.xA - ex.x, r.xB - ex.y);
    }
    for (int k = 0; k < 4; ++k) {
      const double order = std::log2(errs[k] / errs[k + 1]);
      EXPECT_GE(order, 1.9) << "B " << (c == 0 ? "Fixed" : "Adjustable")
                            << ", step " << k << ": " << errs[k] << " -> " << errs[k + 1];
      EXPECT_LE(order, 2.1) << "B " << (c == 0 ? "Fixed" : "Adjustable")
                            << ", step " << k << ": " << errs[k] << " -> " << errs[k + 1];
    }
  }
}

// Deliberately stiff: kA*h and kC*h are O(1) at the coarsest scale used
// elsewhere in this file (nowhere near the asymptotic regime the order
// tests require). This test checks STABILITY -- the coupled iteration
// converges within max_iterations and produces a finite, bounded answer --
// not accuracy; do not add an error/order assertion here.
TEST(WindowCouplerVerification, StiffRegimeStaysStableWithoutBlowingUp) {
  const RunResult r =
      run_two_member(Rational{13}, Rational{7}, 6, DtPolicy::Fixed, DtPolicy::Fixed);
  ASSERT_FALSE(r.iterations.empty());
  EXPECT_TRUE(std::isfinite(r.xA));
  EXPECT_TRUE(std::isfinite(r.xB));
  EXPECT_LT(std::abs(r.xA), 10.0) << "diverged instead of staying stable";
  EXPECT_LT(std::abs(r.xB), 10.0) << "diverged instead of staying stable";
  for (int it : r.iterations) {
    EXPECT_LT(it, 300) << "hit max_iterations instead of converging";
  }
}

// ---------------------------------------------------------------- (c)

// At m=1 both members take exactly one substep per window, on the SAME
// grid -- the window mechanism never engages truncation, extrapolation or
// straddling. This IS today's single-clock region's own sweep model
// (checkpoint once, Gauss-Seidel over the whole step, accelerate,
// converge), so its result must match a hand-rolled version of that model
// applied to the same test problem, to tight tolerance.
TEST(WindowCouplerVerification, RatioOneMatchesTodaysSingleClockRegion) {
  LinearOdeComponent a("A", kA, kB, kX0);
  LinearOdeComponent b("B", kC, kD, kY0);
  const double h = 1.0;
  double xA = kX0, xB = kY0;
  for (int w = 0; w < 5; ++w) {
    const double x0 = xA, y0 = xB;  // window-start (committed) values
    double y_fed = y0, x_fed = x0;  // seed: hold the window-start value
    double xn = x0, yn = y0;
    // Each sweep must re-advance from the WINDOW-START state, not from
    // wherever the previous sweep's advance() left it -- WindowCoupler's
    // own sweep loop calls restore_state() at the top of every iteration
    // for exactly this reason (src/runtime/window_coupler.cpp). Missing
    // this checkpoint here made an earlier version of this reference
    // compound x_/y_ across all 100 sweeps instead of resetting each one,
    // collapsing it to a near-zero fixed point unrelated to the window's
    // actual answer.
    (void)a.save_state();
    (void)b.save_state();
    for (int k = 0; k < 100; ++k) {
      (void)a.restore_state();
      (void)b.restore_state();
      StepContext ctx;
      ctx.t_begin = Rational{w};
      ctx.t_end = Rational{w + 1};
      ctx.iteration = k;
      // WindowCoupler feeds each component the MEAN of its partner's
      // Waveform over the whole substep, not the partner's raw iterate --
      // for a single full-window substep (m=1, exactly this case) that
      // Waveform has two knots, (T0, window-start value) and (T1, the
      // partner's current iterate), so its mean is their AVERAGE. Feeding
      // the raw iterate here (an earlier version of this reference did)
      // converges to a different, wrong fixed point.
      *a.required_ptr() = 0.5 * (y0 + y_fed);
      (void)a.advance(ctx);
      xn = a.x();
      *b.required_ptr() = 0.5 * (x0 + x_fed);
      (void)b.advance(ctx);
      yn = b.x();
      const double resid = std::hypot(xn - x_fed, yn - y_fed);
      // constant relaxation 0.5 -- a plain, deterministic reference, not
      // the production Aitken accelerator, so this check is independent of
      // WindowCoupler's own accelerator wiring.
      x_fed = x_fed + 0.5 * (xn - x_fed);
      y_fed = y_fed + 0.5 * (yn - y_fed);
      if (resid < 1e-13) break;
    }
    xA = xn;
    xB = yn;
  }

  const RunResult r = run_two_member(Rational{1}, Rational{1}, 5,
                                     DtPolicy::Adjustable, DtPolicy::Adjustable,
                                     AccelKind::Constant);
  EXPECT_NEAR(r.xA, xA, 1e-9);
  EXPECT_NEAR(r.xB, xB, 1e-9);
}

// ---------------------------------------------------------------- (d)

TEST(WindowCouplerVerification, TimeConservationOfTheExchangedIntegral) {
  // Waveform::integrate is exact for piecewise-linear data (its Linear
  // basis), so a partition of a window sums to the whole to round-off --
  // exactly the property WindowCoupler leans on for every mean-forcing
  // sample it hands a consumer. Exercised directly here on a Waveform built
  // the way WindowCoupler builds one, rather than assumed.
  Waveform wf(1, 16, fuse::TemporalBasis::Linear, 1);
  wf.reset(Rational{0}, Rational{14});
  const std::vector<std::pair<Rational, double>> samples = {
      {Rational{0}, 1.0}, {Rational{7}, 0.6}, {Rational{13}, 0.2}, {Rational{14}, 0.05}};
  for (const auto& [t, v] : samples) {
    double vv = v;
    ASSERT_TRUE(wf.push(t, &vv).ok());
  }
  double whole = 0.0;
  ASSERT_TRUE(wf.integrate(Rational{0}, Rational{14}, &whole).ok());
  double p0 = 0.0, p1 = 0.0, p2 = 0.0;
  ASSERT_TRUE(wf.integrate(Rational{0}, Rational{7}, &p0).ok());
  ASSERT_TRUE(wf.integrate(Rational{7}, Rational{13}, &p1).ok());
  ASSERT_TRUE(wf.integrate(Rational{13}, Rational{14}, &p2).ok());
  EXPECT_NEAR(whole, p0 + p1 + p2, 1e-12);

  // An arbitrary split not aligned with the sample knots sums exactly too.
  double q0 = 0.0, q1 = 0.0, q2 = 0.0;
  ASSERT_TRUE(wf.integrate(Rational{0}, Rational{5}, &q0).ok());
  ASSERT_TRUE(wf.integrate(Rational{5}, Rational{11}, &q1).ok());
  ASSERT_TRUE(wf.integrate(Rational{11}, Rational{14}, &q2).ok());
  EXPECT_NEAR(whole, q0 + q1 + q2, 1e-12);
}

// ---------------------------------------------------------------- (f)

TEST(WindowCouplerVerification, IterationCountsAcrossAccelerators) {
  int total_none = 0, total_aitken = 0, total_iqn = 0;
  double x_none = 0, x_aitken = 0, x_iqn = 0;
  for (const auto [accel, total, xv] :
       std::initializer_list<std::tuple<AccelKind, int*, double*>>{
           {AccelKind::None, &total_none, &x_none},
           {AccelKind::Aitken, &total_aitken, &x_aitken},
           {AccelKind::IQN_ILS, &total_iqn, &x_iqn}}) {
    const RunResult r = run_two_member(Rational{13}, Rational{7}, 4,
                                       DtPolicy::Fixed, DtPolicy::Adjustable, accel);
    for (const int it : r.iterations) *total += it;
    *xv = r.xA;
  }
  EXPECT_LT(total_aitken, total_none);
  EXPECT_NEAR(x_none, x_aitken, 1e-6);
  EXPECT_NEAR(x_iqn, x_aitken, 1e-6);
}

// ---------------------------------------------------------------- (g)

// WaveformIteration::GaussSeidel and the default Jacobi solve the SAME
// discrete fixed-point system (same substep grid, same channel means) by
// two different classical iterations; both are checked here against the
// SAME converged answer, not against each other's convergence path, which
// is the one thing that legitimately differs between them (iteration
// counts are exercised separately, IterationCountsAcrossAccelerators
// above).
//
// Tolerance derived from measured error, not an arbitrary round number
// (this file's own convention -- see IncommensurateSevenThirteenAllPolicy
// Pairs above): measured (this build) |xA_jacobi - xA_gauss_seidel| across
// three shapes (m=4 commensurate adjustable/adjustable, and the
// incommensurate 7:13 case at both fixed/fixed and adjustable/adjustable)
// tops out at 2.3e-11 -- both modes' own convergence criterion is the
// default 1e-10 relative residual, and the two iterations' different
// floating-point summation order costs another couple of decades on top
// of that. 1e-8 is comfortably above the largest measured value (2.3e-11)
// without being the old, unexamined round number.
// Measured (this build), 13:7 fixed/adjustable, 4 windows back to back:
// jacobi=[16,16,15,14], gauss_seidel=[9,10,9,9] -- Gauss-Seidel converges
// in noticeably fewer sweeps per window on this problem, which is exactly
// what "reads the partner's freshest available value instead of a
// sweep-stale one" should buy. Pinned as a strict inequality per window,
// not the exact counts (an accelerator or convergence-metric change
// elsewhere could shift both by the same amount without this test's own
// point -- Gauss-Seidel beats Jacobi here -- becoming false).
TEST(WindowCouplerVerification, GaussSeidelConvergesInFewerIterationsOnThisProblem) {
  const RunResult jacobi = run_two_member(Rational{13}, Rational{7}, 4,
                                          DtPolicy::Fixed, DtPolicy::Adjustable);
  const RunResult gauss_seidel = run_two_member(
      Rational{13}, Rational{7}, 4, DtPolicy::Fixed, DtPolicy::Adjustable,
      AccelKind::Aitken, 2, 4096, fuse::WaveformIteration::GaussSeidel, true);
  ASSERT_EQ(jacobi.iterations.size(), gauss_seidel.iterations.size());
  for (std::size_t w = 0; w < jacobi.iterations.size(); ++w) {
    EXPECT_LT(gauss_seidel.iterations[w], jacobi.iterations[w]) << "window " << w;
  }
}

TEST(WindowCouplerVerification, GaussSeidelConvergesToTheSameDiscreteSolutionAsJacobi) {
  const struct { Rational Tc, hB; int n_windows; DtPolicy a, b; const char* label; } cases[] = {
      {Rational{1}, Rational{1, 4}, 5, DtPolicy::Adjustable, DtPolicy::Adjustable,
       "m=4 adjustable/adjustable"},
      {Rational{13}, Rational{7}, 3, DtPolicy::Fixed, DtPolicy::Fixed,
       "7:13 fixed/fixed"},
      {Rational{13}, Rational{7}, 3, DtPolicy::Adjustable, DtPolicy::Adjustable,
       "7:13 adjustable/adjustable"},
  };
  for (const auto& c : cases) {
    const RunResult jacobi = run_two_member(c.Tc, c.hB, c.n_windows, c.a, c.b);
    const RunResult gauss_seidel = run_two_member(
        c.Tc, c.hB, c.n_windows, c.a, c.b, AccelKind::Aitken, /*order=*/2,
        /*waveform_capacity=*/4096, fuse::WaveformIteration::GaussSeidel,
        /*lag_b_to_a=*/true);
    EXPECT_NEAR(jacobi.xA, gauss_seidel.xA, 1e-8) << c.label;
    EXPECT_NEAR(jacobi.xB, gauss_seidel.xB, 1e-8) << c.label;
    for (const int it : gauss_seidel.iterations) EXPECT_GT(it, 0) << c.label;
  }
}

// GaussSeidel with NEITHER channel lagged is refused at construction, the
// exact runtime counterpart of validate_coupling_window()'s own
// E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG (fuse/schedule/window.hpp): with every
// channel unlagged, the unlagged-edge graph is the full two-way cycle
// (a2b: 0->1, b2a: 1->0), and no sweep order can run both members before
// each other. One method (a cycle check over the unlagged-edge graph),
// one meaning, at both the compiled-graph and the standalone-driver
// level -- so the driver refuses exactly the configuration the compiler
// would refuse to schedule.
TEST(WindowCouplerRefusals, GaussSeidelWithNoLaggedChannelAtAllIsRefused) {
  auto comp_a = std::make_unique<LinearOdeComponent>("A", kA, kB, kX0);
  auto comp_b = std::make_unique<LinearOdeComponent>("B", kC, kD, kY0);
  WindowMember mA;
  mA.component = comp_a.get();
  mA.name = "A";
  mA.interval = Rational{1};
  mA.policy = DtPolicy::Adjustable;
  mA.provided = comp_a->provided_ptr();
  mA.required = comp_a->required_ptr();
  mA.length = 1;
  WindowMember mB = mA;
  mB.component = comp_b.get();
  mB.name = "B";
  mB.provided = comp_b->provided_ptr();
  mB.required = comp_b->required_ptr();

  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.iteration = fuse::WaveformIteration::GaussSeidel;
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .lagged = false},
      {.provider = 1, .consumer = 0, .lagged = false}};
  const auto r = WindowCoupler::create({mA, mB}, channels, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-GAUSSSEIDEL-NOLAG");
}

// A provider with BOTH a lagged and an unlagged outgoing channel: refused
// rather than silently picking one lag status and getting the other
// consumer's semantics wrong (WindowCoupler::create()'s own docstring).
// Member A provides to both B (unlagged) and C (lagged).
TEST(WindowCouplerRefusals, MixedLagFanOutIsRefused) {
  auto comp_a = std::make_unique<LinearOdeComponent>("A", kA, kB, kX0);
  auto comp_b = std::make_unique<LinearOdeComponent>("B", kC, kD, kY0);
  auto comp_c = std::make_unique<LinearOdeComponent>("C", kC, kD, kY0);
  WindowMember mA;
  mA.component = comp_a.get();
  mA.name = "A";
  mA.interval = Rational{1};
  mA.policy = DtPolicy::Adjustable;
  mA.provided = comp_a->provided_ptr();
  mA.required = comp_a->required_ptr();
  mA.length = 1;
  WindowMember mB = mA;
  mB.component = comp_b.get();
  mB.name = "B";
  mB.provided = comp_b->provided_ptr();
  mB.required = comp_b->required_ptr();
  WindowMember mC = mA;
  mC.component = comp_c.get();
  mC.name = "C";
  mC.provided = comp_c->provided_ptr();
  mC.required = comp_c->required_ptr();

  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.iteration = fuse::WaveformIteration::GaussSeidel;
  // A -> B unlagged, A -> C lagged, C -> A lagged (so the graph is
  // otherwise legal -- the mixed fan-out on A is the ONLY thing this
  // configuration is refused for).
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .lagged = false},
      {.provider = 0, .consumer = 2, .lagged = true},
      {.provider = 2, .consumer = 0, .lagged = true}};
  const auto r = WindowCoupler::create({mA, mB, mC}, channels, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-GAUSSSEIDEL-MIXED-LAG");
}

TEST(WindowCouplerRefusals, EmptyMembersOrChannelsRefused) {
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  const auto r = WindowCoupler::create({}, {}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-EMPTY");
}

TEST(WindowCouplerRefusals, NullComponentRefused) {
  WindowMember m;
  m.name = "orphan";
  m.interval = Rational{1};
  m.length = 0;
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  const auto r = WindowCoupler::create({m}, {{0, 0}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-NULL");
}

TEST(WindowCouplerRefusals, NonPositiveIntervalRefused) {
  LinearOdeComponent c("C", 1.0, 1.0, 0.0);
  WindowMember m;
  m.component = &c;
  m.name = "C";
  m.interval = Rational{0};
  m.provided = c.provided_ptr();
  m.required = c.required_ptr();
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  const auto r = WindowCoupler::create({m}, {{0, 0}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-INTERVAL");
}

TEST(WindowCouplerRefusals, OrderBelowTwoRefused) {
  LinearOdeComponent c("C", 1.0, 1.0, 0.0);
  WindowMember m;
  m.component = &c;
  m.name = "C";
  m.interval = Rational{1};
  m.provided = c.provided_ptr();
  m.required = c.required_ptr();
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.order = 1;
  const auto r = WindowCoupler::create({m}, {{0, 0}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-ORDER");
}

TEST(WindowCouplerRefusals, FixedWithoutRollbackSupportRefused) {
  class NoRollback : public Component {
   public:
    [[nodiscard]] std::string name() const override { return "nr"; }
    Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
      return Result<void>::success();
    }
    Result<void> advance(const StepContext&) override {
      return Result<void>::success();
    }
    double p = 0.0, r = 0.0;
  } c;
  WindowMember m;
  m.component = &c;
  m.name = "nr";
  m.interval = Rational{1};
  m.policy = DtPolicy::Fixed;
  m.provided = &c.p;
  m.required = &c.r;
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  const auto r = WindowCoupler::create({m}, {{0, 0}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-ROLLBACK");
}

TEST(WindowCouplerRefusals, ChannelIndexOutOfRangeRefused) {
  LinearOdeComponent c("C", 1.0, 1.0, 0.0);
  WindowMember m;
  m.component = &c;
  m.name = "C";
  m.interval = Rational{1};
  m.provided = c.provided_ptr();
  m.required = c.required_ptr();
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  const auto r = WindowCoupler::create({m}, {{0, 5}}, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-CHANNEL");
}

// A convergence measure names a field TRANSFER (a WindowChannel, by its own
// `name`); one naming a transfer the coupler does not have is refused by name
// at create(), rather than silently falling back to some default at sweep time.
TEST(WindowCouplerRefusals, ConvergenceMeasureNamingAnUnknownChannelIsRefused) {
  LinearOdeComponent a("A", 1.0, 1.0, 0.0), b("B", 1.0, 1.0, 0.0);
  WindowMember mA;
  mA.component = &a; mA.name = "A"; mA.interval = Rational{1};
  mA.provided = a.provided_ptr(); mA.required = a.required_ptr();
  WindowMember mB;
  mB.component = &b; mB.name = "B"; mB.interval = Rational{1};
  mB.provided = b.provided_ptr(); mB.required = b.required_ptr();
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Relative, 1e-6, "not_a_channel"});
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .name = "a_out"},
      {.provider = 1, .consumer = 0, .name = "b_out"}};
  const auto r = WindowCoupler::create({mA, mB}, channels, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-WINDOW-CONVERGENCE-FIELD") << r.error().message;
}

// The companion positive case: every measure names a channel that exists
// -- create() must succeed, and each measure's own residual is checked
// against ONLY the channel it names (verified end to end by the driver-
// vs-Runner acceptance tests in window_runner_straddle_test.cpp, not
// re-verified numerically here).
TEST(WindowCouplerRefusals, ConvergenceMeasureNamingAKnownChannelIsAccepted) {
  LinearOdeComponent a("A", 1.0, 1.0, 0.0), b("B", 1.0, 1.0, 0.0);
  WindowMember mA;
  mA.component = &a; mA.name = "A"; mA.interval = Rational{1};
  mA.provided = a.provided_ptr(); mA.required = a.required_ptr();
  WindowMember mB;
  mB.component = &b; mB.name = "B"; mB.interval = Rational{1};
  mB.provided = b.provided_ptr(); mB.required = b.required_ptr();
  WindowCouplerConfig cfg;
  cfg.window_length = Rational{1};
  cfg.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Relative, 1e-6, "a_out"});
  cfg.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Relative, 1e-6, "b_out"});
  const std::vector<WindowChannel> channels = {
      {.provider = 0, .consumer = 1, .name = "a_out"},
      {.provider = 1, .consumer = 0, .name = "b_out"}};
  const auto r = WindowCoupler::create({mA, mB}, channels, cfg);
  ASSERT_TRUE(r.ok()) << r.error().message;
}

// ---------------------------------------------------------------- layout
//
// A field-valued member with several substeps per window. Two independent
// scalar problems packed into length-2 members must reproduce each problem run
// on its own. The elements never interact, so a difference beyond the
// iteration tolerance is the driver mixing (element, substep) entries. Not bit
// for bit: convergence is judged on the whole packed vector, so the packed run
// may stop one sweep apart from the scalar one (measured about 1e-11). The
// transposed write this test exists for was off by about 1e-2 (0.4136 against
// 0.4074); every scalar or single-substep test above is blind to it.
namespace {

class PackedOde final : public Component {
 public:
  PackedOde(std::vector<double> a, std::vector<double> b, std::vector<double> x0)
      : a_(std::move(a)), b_(std::move(b)), x_(std::move(x0)), saved_(x_),
        provided_(x_), required_(x_.size(), 0.0) {}
  [[nodiscard]] std::string name() const override { return "packed"; }
  Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext& ctx) override {
    const double h = ctx.dt().to_double();
    for (std::size_t e = 0; e < x_.size(); ++e) {
      const double ex = std::exp(-a_[e] * h);
      x_[e] = x_[e] * ex + b_[e] * required_[e] * (1.0 - ex) / a_[e];
    }
    provided_ = x_;
    return Result<void>::success();
  }
  Result<void> save_state() override { saved_ = x_; return Result<void>::success(); }
  Result<void> restore_state() override {
    x_ = saved_; provided_ = x_; return Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }
  double* provided_ptr() { return provided_.data(); }
  double* required_ptr() { return required_.data(); }
  [[nodiscard]] const std::vector<double>& x() const { return x_; }

 private:
  std::vector<double> a_, b_, x_, saved_, provided_, required_;
};

std::pair<std::vector<double>, std::vector<double>> run_packed(
    const std::vector<double>& aA, const std::vector<double>& bA, const std::vector<double>& xA,
    const std::vector<double>& aB, const std::vector<double>& bB, const std::vector<double>& xB,
    Rational window, Rational hB, int n_windows, DtPolicy pol_b) {
  PackedOde A(aA, bA, xA), B(aB, bB, xB);
  WindowMember mA;
  mA.component = &A; mA.name = "A"; mA.interval = window; mA.policy = DtPolicy::Adjustable;
  mA.provided = A.provided_ptr(); mA.required = A.required_ptr(); mA.length = xA.size();
  WindowMember mB;
  mB.component = &B; mB.name = "B"; mB.interval = hB; mB.policy = pol_b;
  mB.provided = B.provided_ptr(); mB.required = B.required_ptr(); mB.length = xB.size();
  WindowCouplerConfig cfg;
  cfg.window_length = window;
  cfg.max_iterations = 300;
  cfg.acceleration = AccelConfig{.kind = AccelKind::None};
  auto c = WindowCoupler::create({mA, mB}, {{0, 1}, {1, 0}}, cfg);
  if (!c.ok()) { ADD_FAILURE() << c.error().message; return {}; }
  for (int w = 0; w < n_windows; ++w) {
    const Result<void> r = c.value()->run_window();
    if (!r.ok()) { ADD_FAILURE() << r.error().message; return {}; }
  }
  return {A.x(), B.x()};
}

}  // namespace

TEST(WindowCouplerLayout, PackedElementsReproduceEachScalarProblemExactly) {
  // Element 0 and element 1 are different problems, so a transposed write
  // hands each element its neighbour's trajectory.
  const std::vector<double> aA{kA, 0.4}, bA{kB, 0.9}, xA{kX0, -0.3};
  const std::vector<double> aB{kC, 1.7}, bB{kD, 0.2}, xB{kY0, 0.8};
  for (const DtPolicy pol : {DtPolicy::Adjustable, DtPolicy::Fixed}) {
    // 13:7 at d = 100: B takes two substeps per window and straddles when Fixed.
    const auto packed = run_packed(aA, bA, xA, aB, bB, xB, Rational{13, 100},
                                   Rational{7, 100}, 7, pol);
    ASSERT_EQ(packed.first.size(), 2u);
    for (std::size_t e = 0; e < 2; ++e) {
      const auto alone = run_packed({aA[e]}, {bA[e]}, {xA[e]}, {aB[e]}, {bB[e]}, {xB[e]},
                                    Rational{13, 100}, Rational{7, 100}, 7, pol);
      ASSERT_EQ(alone.first.size(), 1u);
      EXPECT_NEAR(packed.first[e], alone.first[0], 1e-9) << "A element " << e;
      EXPECT_NEAR(packed.second[e], alone.second[0], 1e-9) << "B element " << e;
    }
  }
}

// ------------------------------------------------------- detail::advance_straddle_split

namespace {

// Records every advance() call (its StepContext and the `required` value
// seen at call time) and writes a value derived from `required` and the
// call's own ordinal into `provided`, so a test can tell exactly which
// call produced which `out_at_split` entry -- a real Component's
// advance() output could coincidentally look the same whichever call
// produced it, but this cannot.
class RecordingComponent : public Component {
 public:
  struct Call {
    Rational t_begin, t_end;
    std::int64_t step;
    int iteration;
    double required_seen;
  };

  [[nodiscard]] std::string name() const override { return "recording"; }
  Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext& ctx) override {
    calls.push_back({ctx.t_begin, ctx.t_end, ctx.step, ctx.iteration, required_});
    // 1000 marks which call produced this value (1 = first/in-phase call,
    // 2 = second/tail call); required_ is folded in so a wrong mean would
    // also be caught, not just a wrong call order.
    provided_ = static_cast<double>(calls.size()) * 1000.0 + required_;
    return Result<void>::success();
  }
  Result<void> save_state() override { return Result<void>::success(); }
  Result<void> restore_state() override { return Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return 0; }

  double* provided_ptr() { return &provided_; }
  double* required_ptr() { return &required_; }

  std::vector<Call> calls;

 private:
  double provided_ = 0.0;
  double required_ = 0.0;
};

[[nodiscard]] WindowMember make_member(RecordingComponent& c) {
  WindowMember m;
  m.component = &c;
  m.name = "R";
  m.provided = c.provided_ptr();
  m.required = c.required_ptr();
  m.length = 1;
  return m;
}

}  // namespace

TEST(AdvanceStraddleSplit, CallsBothAdvancesInOrderOverTheCorrectBounds) {
  RecordingComponent c;
  const WindowMember m = make_member(c);
  double out = 0.0;
  const Result<void> r = fuse::detail::advance_straddle_split(
      m, Rational{2}, Rational{5}, Rational{9}, /*step=*/3, /*iteration=*/4,
      [](std::size_t, Rational, Rational) -> Result<double> { return 11.0; },
      [](std::size_t) -> Result<double> { return 13.0; },
      [](std::size_t, Rational) -> double { return 17.0; }, &out);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(c.calls.size(), 2u);
  EXPECT_EQ(c.calls[0].t_begin, Rational{2});
  EXPECT_EQ(c.calls[0].t_end, Rational{5});
  EXPECT_EQ(c.calls[0].step, 3);
  EXPECT_EQ(c.calls[0].iteration, 4);
  EXPECT_EQ(c.calls[0].required_seen, 11.0);  // mean_in_window's value
  EXPECT_EQ(c.calls[1].t_begin, Rational{5});
  EXPECT_EQ(c.calls[1].t_end, Rational{9});
  EXPECT_EQ(c.calls[1].step, 3);
  EXPECT_EQ(c.calls[1].iteration, 4);
  EXPECT_EQ(c.calls[1].required_seen, 0.5 * (13.0 + 17.0));  // the blended tail mean
}

TEST(AdvanceStraddleSplit, WritesTheFirstCallsExactValueNeverAnInterpolant) {
  RecordingComponent c;
  const WindowMember m = make_member(c);
  double out = -1.0;
  const Result<void> r = fuse::detail::advance_straddle_split(
      m, Rational{0}, Rational{3}, Rational{8}, 0, 0,
      [](std::size_t, Rational, Rational) -> Result<double> { return 2.0; },
      [](std::size_t) -> Result<double> { return 100.0; },
      [](std::size_t, Rational) -> double { return 200.0; }, &out);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(c.calls.size(), 2u);
  // out_at_split is what the FIRST call wrote to `provided` -- 1000*1 + 2.0
  // -- not the second call's 1000*2 + 150.0, and not any blend of the two.
  EXPECT_EQ(out, 1000.0 + 2.0);
  EXPECT_NE(out, c.calls[1].required_seen + 2000.0);
}

TEST(AdvanceStraddleSplit, SkipsTheSecondAdvanceWhenTheSplitIsTheChunksOwnEnd) {
  RecordingComponent c;
  const WindowMember m = make_member(c);
  bool boundary_called = false, extrapolate_called = false;
  double out = -1.0;
  const Result<void> r = fuse::detail::advance_straddle_split(
      m, Rational{4}, Rational{7}, Rational{7}, 0, 0,
      [](std::size_t, Rational, Rational) -> Result<double> { return 5.0; },
      [&](std::size_t) -> Result<double> {
        boundary_called = true;
        return 0.0;
      },
      [&](std::size_t, Rational) -> double {
        extrapolate_called = true;
        return 0.0;
      },
      &out);
  ASSERT_TRUE(r.ok()) << r.error().message;
  ASSERT_EQ(c.calls.size(), 1u);  // no tail call when t_split == t_b
  EXPECT_FALSE(boundary_called);
  EXPECT_FALSE(extrapolate_called);
  EXPECT_EQ(out, 1000.0 + 5.0);  // still the (only) call's exact value
}

TEST(AdvanceStraddleSplit, MeanInWindowFailurePropagatesBeforeAnyAdvance) {
  RecordingComponent c;
  const WindowMember m = make_member(c);
  double out = 0.0;
  const Result<void> r = fuse::detail::advance_straddle_split(
      m, Rational{0}, Rational{2}, Rational{5}, 0, 0,
      [](std::size_t, Rational, Rational) -> Result<double> {
        return fuse::Failure{"E-TEST", "mean_in_window refused"};
      },
      [](std::size_t) -> Result<double> { return 0.0; },
      [](std::size_t, Rational) -> double { return 0.0; }, &out);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TEST");
  EXPECT_EQ(c.calls.size(), 0u);
}
