// KNOWN_ISSUES.md D31 -- window Newton over the region's REAL participant
// ranks, native substrate. Not a fuse_add_mpi_test() gtest binary run
// through ctest in THIS sandbox: it calls MPI_Init, must be launched by
// mpiexec, and this sandbox cannot start MPI at all (same constraint
// test/parallel/window_runner_mpi_test.cpp names) -- build-verified
// only here; registered as a host row (parallel.window_newton_mpi.{2,3}).
//
// THE FIXTURE: window_runner_mpi_test.cpp's own `LinearOdeComponent`/
// `build_window_graph()`, reproduced here (not shared via a header --
// the identical file-local-helper reasoning that file's own comment
// gives), with ONE line changed: `decl.acceleration.kind` is
// `AccelKind::Newton` instead of `AccelKind::None`. `LinearOdeComponent`
// is this track's own "small synthetic case where the exact answer is
// known" (the task's own phrase): `dx/dt = -a*x + b*u` integrated in
// closed form given the partner's own mean `u` over the substep, so F
// is AFFINE in the Newton iterate and the native solver's own quadratic-
// convergence theory predicts (and Sec. 8.2's own dn_heat measurement
// confirms for an identical affine seam) that Newton reaches its own
// convergence test in very few outer iterations regardless of rank
// count.
//
// THE COMPARISON. `iterations_per_window`/`gmres_iterations_per_window`
// are the Newton outer-iteration count (`IterationStats::iterations`,
// now `fuse::num::NewtonResult::iterations` for a Newton-selected
// region -- the SAME field `window_runner_mpi_test.cpp` already reads
// for Picard's own sweep count) and the summed GMRES iteration count
// over that solve's own trace (`Runner::debug_newton_gmres_iterations()`,
// TEST/DIAGNOSTIC-ONLY). Compared EXACTLY (`EXPECT_EQ`) against the
// single-process reference: both quantities are INTEGERS decided by a
// convergence TEST against a reduced norm, and the reduction itself
// (`ParticipantSetReducer::sum()`, an `MPI_Allreduce` over exactly two
// terms, this region's own two participants) is a SUM OF 2 TERMS IN
// A FIXED, RANK-INDEPENDENT ORDER (`ParticipantSetReducerResult::
// comm_f`'s own split orders by `R.participants`, ascending base rank,
// identically on every rank) -- Higham's own recursive-summation bound
// (`ToleranceBound` below) for `n = 2` partial sums is `gamma_2 =
// 2u/(1-2u)`, about `4.44e-16`, which this problem's own `rtol = 1e-8`
// swallows whole: the reduction's own round-off cannot move the
// CONVERGENCE DECISION (an integer) at all, only the many-more-orders-
// of-magnitude-larger last bit of the converged VALUE itself -- which
// is why the iteration COUNTS are asserted exactly and the FINAL VALUES
// (`x_a_per_window`/`x_b_per_window`) are asserted to the derived bound
// instead, both printed, never a bitwise comparison across rank counts.

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>
#include <fuse/transport/transport.hpp>

using fuse::AccelKind;
using fuse::Component;
using fuse::ComponentDecl;
using fuse::ComponentSet;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::CouplingWindowScope;
using fuse::DtPolicy;
using fuse::FieldId;
using fuse::FieldRef;
using fuse::FieldRegistry;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::IterationRegionDecl;
using fuse::Layout;
using fuse::MemorySpace;
using fuse::MeshDecl;
using fuse::NewtonConfig;
using fuse::Rational;
using fuse::Region;
using fuse::RankSet;
using fuse::Result;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

int world_size() {
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

int current_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

// window_runner_mpi_test.cpp's own LinearOdeComponent, unchanged.
class LinearOdeComponent final : public Component {
 public:
  LinearOdeComponent(std::string name, double a, double b, double x0)
      : name_(std::move(name)), a_(a), b_(b), x_(x0), provided_(x0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  Result<void> register_fields(FieldRegistry& reg,
                                const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      double* p = is_out ? &provided_ : &required_;
      auto r = reg.bind(
          f, FieldRef{p, Layout{.stride = 1, .count = 1}, MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return Result<void>::success();
  }

  Result<void> advance(const StepContext& ctx) override {
    ++advance_count;
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
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }
  [[nodiscard]] double x() const { return x_; }

  int advance_count = 0;

 private:
  std::string name_;
  double      a_, b_;
  double      x_;
  double      provided_;
  double      required_ = 0.0;
  double      saved_ = 0.0;
};

constexpr double kAA = 1.3, kAB = 0.7, kAC = 0.9, kAD = 0.5;
constexpr double kAX0 = 1.0, kAY0 = 0.2;

// window_runner_mpi_test.cpp's own build_window_graph(), with the
// accelerator switched to Newton and a default NewtonConfig --
// LinearOdeComponent's own capabilities() is kCapabilityRollback only,
// which is all a window-Newton region (NewtonResidualPath::Advance,
// no block-Jacobi preconditioner requested) needs.
CouplingGraph build_window_newton_graph(RankSet a_ranks, RankSet b_ranks) {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:line:1"});
  const auto r = g.add_region(Region{.name = "r", .mesh = m, .dimension = 1});

  const auto a = g.add_component(
      ComponentDecl{.name = "A", .ranks = a_ranks, .interval = Rational{13}});
  const auto b = g.add_component(
      ComponentDecl{.name = "B", .ranks = b_ranks, .interval = Rational{7}});

  const auto a_out = g.add_field(a, FieldSpec{.name = "a_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{.name = "b_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});
  const auto b_out = g.add_field(b, FieldSpec{.name = "b_out", .region = r,
                                              .units = u::kKelvin,
                                              .conservation = Conservation::Extensive,
                                              .intent = Intent::Provides});
  const auto a_in = g.add_field(a, FieldSpec{.name = "a_in", .region = r,
                                             .units = u::kKelvin,
                                             .conservation = Conservation::Extensive,
                                             .intent = Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep, not a real-time lag");

  CouplingWindowScope scope;
  scope.explicit_interval = Rational{91};
  scope.order = 2;
  scope.policies = {{"A", DtPolicy::Fixed}, {"B", DtPolicy::Fixed}};
  scope.iteration = WaveformIteration::Jacobi;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 300;
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "a_out"});
  decl.convergence.add(
      fuse::ConvergenceMeasure{fuse::MeasureKind::Absolute, 1e-12, "b_out"});
  decl.window_scope = scope;
  // KNOWN_ISSUES.md D31: the one difference from window_runner_mpi_
  // test.cpp's own fixture.
  decl.acceleration.kind = AccelKind::Newton;
  decl.newton_config = NewtonConfig{};
  g.add_iteration_region(decl);
  return g;
}

struct RunOutcome {
  bool ok = false;
  std::string error;
  std::vector<double> x_a_per_window, x_b_per_window;
  std::vector<int> newton_iterations_per_window;
  std::vector<std::size_t> gmres_iterations_per_window;
};

constexpr int kPeriods = 5;

RunOutcome run_window_newton(const CouplingGraph& g, const RunnerConfig& cfg) {
  RunOutcome out;
  const auto lowered = fuse::lower(g);
  if (!lowered.ok()) { out.error = lowered.diags.to_string(); return out; }
  fuse::SerialMeshBackend backend;
  if (!backend.load(g.mesh(fuse::MeshId{0})).ok()) {
    out.error = "mesh load failed";
    return out;
  }
  const fuse::ClockSet clocks{g};
  const auto sched =
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, lowered.dag, backend));
  if (!sched.ok()) { out.error = sched.diags.to_string(); return out; }

  auto a = std::make_unique<LinearOdeComponent>("A", kAA, kAB, kAX0);
  auto b = std::make_unique<LinearOdeComponent>("B", kAC, kAD, kAY0);
  LinearOdeComponent* pa = a.get();
  LinearOdeComponent* pb = b.get();
  ComponentSet components;
  components.add(std::move(a));
  components.add(std::move(b));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) { out.error = r.error().message; return out; }
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry, cfg);
  if (!init.ok()) { out.error = init.error().code + ": " + init.error().message; return out; }

  for (int p = 0; p < kPeriods; ++p) {
    const auto ran = runner.run_period(components);
    if (!ran.ok()) { out.error = ran.error().code + ": " + ran.error().message; return out; }
    if (runner.iteration_stats().empty()) { out.error = "no iteration stats"; return out; }
    const fuse::IterationStats& st = runner.iteration_stats().front();
    out.x_a_per_window.push_back(pa->x());
    out.x_b_per_window.push_back(pb->x());
    out.newton_iterations_per_window.push_back(st.iterations);
    const auto gmres = runner.debug_newton_gmres_iterations(0);
    if (!gmres.ok()) { out.error = gmres.error().code + ": " + gmres.error().message; return out; }
    std::size_t total_gmres = 0;
    for (const std::size_t it : gmres.value()) total_gmres += it;
    out.gmres_iterations_per_window.push_back(total_gmres);
  }
  out.ok = true;
  return out;
}

std::function<Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

/// Higham's own recursive-summation bound (Accuracy and Stability of
/// Numerical Algorithms, 2nd ed., Thm 4.2/8.1): a sum of `n` partials
/// computed in floating point differs from the exact sum by at most
/// `gamma_n * sum(|x_i|)`, `gamma_n = n*u/(1-n*u)`, `u` the unit
/// roundoff (`2^-53` for `double`). `ParticipantSetReducer::sum()` over
/// `R.participants` reduces exactly `n = participant count` terms per
/// call; this bound covers every such reduction inside one Newton
/// solve (the outer convergence norm, every GMRES inner product), not
/// just one of them -- a WORST-CASE, not a per-call figure, since a
/// Newton solve calls the reducer many times and errors do not cancel
/// across calls in general.
double higham_gamma(int n) {
  constexpr double u = 0.5 * std::numeric_limits<double>::epsilon();  // 2^-53
  const double nu = static_cast<double>(n) * u;
  return nu / (1.0 - nu);
}

}  // namespace

TEST(WindowNewtonMpi, TwoDisjointRanksMatchTheSingleProcessReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  // Reference: every rank computes this identically, no MPI at all --
  // window_runner_mpi_test.cpp's own already-verified "every rank
  // reconstructs the same answer with no communication" convention.
  const CouplingGraph ref_graph =
      build_window_newton_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window_newton(ref_graph, RunnerConfig{});
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_newton_graph(a_ranks, b_ranks);
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  const RunOutcome got = run_window_newton(mpi_graph, cfg);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());

  // Iteration COUNTS: exact match, every rank that takes part --
  // integers decided by a 2-term reduction whose round-off
  // (higham_gamma(2) below) cannot move a convergence test built on
  // rtol = 1e-8.
  for (std::size_t w = 0; w < ref.newton_iterations_per_window.size(); ++w) {
    if (!(hosts_a || hosts_b)) continue;
    EXPECT_EQ(got.newton_iterations_per_window[w], ref.newton_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank()
        << " -- every participant must agree the SAME Newton outer "
           "iteration count (ParticipantSetReducer's own convergence test)";
    EXPECT_EQ(got.gmres_iterations_per_window[w], ref.gmres_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank()
        << " -- and the SAME total GMRES iteration count";
  }

  // Final VALUES: bounded, not bit-identical -- derived, printed.
  const double gamma2 = higham_gamma(2);
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      const double diff = std::fabs(got.x_a_per_window[w] - ref.x_a_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_a_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_a_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-mpi] rank %d window %zu A: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound) << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      const double diff = std::fabs(got.x_b_per_window[w] - ref.x_b_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_b_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_b_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-mpi] rank %d window %zu B: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound) << "window " << w << ", rank " << current_rank();
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(WindowNewtonMpi, ThreeRanksWithAnUncoupledThirdMatchTheReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);

  const CouplingGraph ref_graph =
      build_window_newton_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window_newton(ref_graph, RunnerConfig{});
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  CouplingGraph mpi_graph = build_window_newton_graph(a_ranks, b_ranks);
  // An uncoupled third component on rank 2 -- no channels, no window
  // membership, proving a rank outside the region's own participants
  // does not perturb it (the identical property window_runner_mpi_
  // test.cpp's own three-rank row proves for Picard).
  mpi_graph.add_component(
      ComponentDecl{.name = "C", .ranks = RankSet{2, 1}, .interval = Rational{1}});

  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();

  // run_window_newton() builds its own ComponentSet from A/B only;
  // an uncoupled C with no fields needs no storage, but the Runner
  // still expects every graph-declared component registered. A
  // trivial stand-in mirrors window_runner_mpi_test.cpp's own
  // UncoupledTicker, inlined here since this is its only use.
  class UncoupledTicker final : public Component {
   public:
    [[nodiscard]] std::string name() const override { return "C"; }
    Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
      return Result<void>::success();
    }
    Result<void> advance(const StepContext&) override {
      return Result<void>::success();
    }
  };

  RunOutcome got;
  {
    const auto lowered = fuse::lower(mpi_graph);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
    fuse::SerialMeshBackend backend;
    ASSERT_TRUE(backend.load(mpi_graph.mesh(fuse::MeshId{0})).ok());
    const fuse::ClockSet clocks{mpi_graph};
    const auto sched = fuse::schedule(mpi_graph, lowered.dag, clocks, fuse::make_region_sizes(mpi_graph, lowered.dag, backend));
    ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

    auto a = std::make_unique<LinearOdeComponent>("A", kAA, kAB, kAX0);
    auto b = std::make_unique<LinearOdeComponent>("B", kAC, kAD, kAY0);
    LinearOdeComponent* pa = a.get();
    LinearOdeComponent* pb = b.get();
    ComponentSet components;
    components.add(std::move(a));
    components.add(std::move(b));
    components.add(std::make_unique<UncoupledTicker>());
    FieldRegistry registry;
    for (auto& c : components) {
      const auto r = c.register_fields(registry, mpi_graph);
      ASSERT_TRUE(r.ok()) << r.error().message;
    }
    Runner runner{mpi_graph, lowered.dag, sched.plan, backend};
    const auto init = runner.initialize(components, registry, cfg);
    ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;
    for (int p = 0; p < kPeriods; ++p) {
      const auto ran = runner.run_period(components);
      ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message;
      ASSERT_FALSE(runner.iteration_stats().empty());
      const fuse::IterationStats& st = runner.iteration_stats().front();
      got.x_a_per_window.push_back(pa->x());
      got.x_b_per_window.push_back(pb->x());
      got.newton_iterations_per_window.push_back(st.iterations);
      const auto gmres = runner.debug_newton_gmres_iterations(0);
      ASSERT_TRUE(gmres.ok()) << gmres.error().code << ": " << gmres.error().message;
      std::size_t total_gmres = 0;
      for (const std::size_t it : gmres.value()) total_gmres += it;
      got.gmres_iterations_per_window.push_back(total_gmres);
    }
    got.ok = true;
  }

  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  const double gamma2 = higham_gamma(2);
  for (std::size_t w = 0; w < ref.newton_iterations_per_window.size(); ++w) {
    if (!(hosts_a || hosts_b)) continue;  // rank 2 (C) takes no part
    EXPECT_EQ(got.newton_iterations_per_window[w], ref.newton_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank()
        << " -- an uncoupled third rank must not perturb the agreed "
           "Newton outer iteration count";
    EXPECT_EQ(got.gmres_iterations_per_window[w], ref.gmres_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank();
    if (hosts_a) {
      const double diff = std::fabs(got.x_a_per_window[w] - ref.x_a_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_a_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_a_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-mpi-3] rank %d window %zu A: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound);
    }
    if (hosts_b) {
      const double diff = std::fabs(got.x_b_per_window[w] - ref.x_b_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_b_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_b_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-mpi-3] rank %d window %zu B: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound);
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}
