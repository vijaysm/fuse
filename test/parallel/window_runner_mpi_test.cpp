// Unit 2b (D19), step (e) -- the window path's FIRST actual MPI test.
// Not a fuse_add_mpi_test() gtest binary run through ctest in THIS
// sandbox: it calls MPI_Init, must be launched by mpiexec, and this
// sandbox cannot start MPI at all (KNOWN_ISSUES.md C1) -- build-verified
// only here; registered as a host row (parallel.window_runner_mpi.{2,3}).
//
// THE FIXTURE: window_runner_multirate_test.cpp's own 7:13 LCM-window
// graph (A interval 13, B interval 7, window 91 = lcm(13,7), so neither
// member ever straddles -- the SAME "isolate the cold-start/transfer
// bugs from the separate, still-unimplemented straddle-replay" reason
// that file's own header gives), reproduced here with REAL, DISJOINT
// ranks (that file's own RankSet{0,0} is a tier-0 "don't care" value --
// this_process is never set without a real communicator, so no hosting
// check ever reads it) instead of a meaningful cross-rank test.
//
// THE COMPARISON: every rank independently computes the REFERENCE
// answer too -- the SAME graph, RankSet{0,0}, no RunnerConfig at all,
// exactly window_runner_multirate_test.cpp's own already-verified
// single-process path -- a pure, deterministic, no-MPI computation, so
// every rank arrives at the IDENTICAL reference value without
// communicating about it at all (same principle `rank_agreement.hpp`
// names: nothing here needs MPI to be checkable). The real run (A on
// rank 0, B on rank 1 -- a genuine cross-rank window coupling) is then
// compared against it, EXPECT_EQ, on every rank for the agreed per-window
// iteration count (decide_window_convergence()'s own agreement must be
// identical everywhere), and for each member's own per-window final value
// (x_a(), x_b()) only on the ranks that host that member -- since
// 0eb576a a member advances ONLY on its own processor set, a rank that
// does not host it never touches its real state at all (RankSet::
// contains(), not a literal rank number, decides which comparisons a
// given rank makes). Not merely whether the run completes.
//
// THE 3-RANK CASE adds an UNCOUPLED component C on rank 2 -- no
// channels, no window membership, a plain interval-1 tick -- proving a
// rank outside the window entirely does not perturb it (step (d)'s own
// concern, exercised here for a rank that happens to have ITS OWN
// component rather than none at all; parallel.window_runner_mpi.3's own
// C-less sibling, "an empty third rank", is intentionally covered by
// the SAME graph under `mpiexec -n 3` with world size 3 but no rank-2
// RankSet declared anywhere -- `fuse_add_mpi_test`'s own default RANKS
// list is what exercises that shape, not a separate fixture).

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include <cmath>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/linalg/comm.hpp>
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

// window_runner_multirate_test.cpp's own LinearOdeComponent, unchanged:
// dx/dt = -a*x + b*u, u the mean of the partner's trajectory over the
// substep, closed-form-exact given that mean.
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

// A plain interval-1, no-channel component -- the 3-rank case's own
// uncoupled "C", proving its presence does not perturb the window.
class UncoupledTicker final : public Component {
 public:
  [[nodiscard]] std::string name() const override { return "C"; }
  Result<void> register_fields(FieldRegistry&, const CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext&) override {
    ++advance_count;
    return Result<void>::success();
  }
  int advance_count = 0;
};

constexpr double kAA = 1.3, kAB = 0.7, kAC = 0.9, kAD = 0.5;
constexpr double kAX0 = 1.0, kAY0 = 0.2;

// window_runner_multirate_test.cpp's own make_multirate_graph(), with
// `a_ranks`/`b_ranks` as real parameters instead of that file's own
// tier-0 RankSet{0,0} "don't care". `with_uncoupled_c`: adds a THIRD,
// genuinely uncoupled component on `c_ranks` -- no channels, no window
// membership -- for the 3-rank case.
CouplingGraph build_window_graph(RankSet a_ranks, RankSet b_ranks,
                                 bool with_uncoupled_c = false,
                                 RankSet c_ranks = RankSet{}) {
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
  decl.acceleration.kind = AccelKind::None;
  g.add_iteration_region(decl);

  if (with_uncoupled_c) {
    g.add_component(
        ComponentDecl{.name = "C", .ranks = c_ranks, .interval = Rational{1}});
  }
  return g;
}

struct RunOutcome {
  bool ok = false;
  std::string error;
  std::vector<double> x_a_per_window, x_b_per_window;
  std::vector<int> iterations_per_window;
  // Transport-direct track (KNOWN_ISSUES.md D66/D77): this rank's own
  // RunStats counters after the run, read here so a caller does not need
  // its own Runner handle to check which Transport path a window actually
  // took.
  long long device_transport_moves = 0;
  long long device_boundary_copies = 0;
};

// Runs `periods` periods/window-instances (explicit_interval == period
// here, so they coincide) and records the per-window final state and
// iteration count. `cfg`: default-constructed (unconfigured, tier-0 --
// used for the REFERENCE, which every rank computes identically and
// locally) or a real MPI config (used for the run under test).
RunOutcome run_window(const CouplingGraph& g, int periods,
                      const RunnerConfig& cfg, bool with_c) {
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
      fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) { out.error = sched.diags.to_string(); return out; }

  auto a = std::make_unique<LinearOdeComponent>("A", kAA, kAB, kAX0);
  auto b = std::make_unique<LinearOdeComponent>("B", kAC, kAD, kAY0);
  LinearOdeComponent* pa = a.get();
  LinearOdeComponent* pb = b.get();
  ComponentSet components;
  components.add(std::move(a));
  components.add(std::move(b));
  if (with_c) components.add(std::make_unique<UncoupledTicker>());
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) { out.error = r.error().message; return out; }
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto init = runner.initialize(components, registry, cfg);
  if (!init.ok()) { out.error = init.error().code + ": " + init.error().message; return out; }

  for (int p = 0; p < periods; ++p) {
    const auto ran = runner.run_period(components);
    if (!ran.ok()) { out.error = ran.error().code + ": " + ran.error().message; return out; }
    if (runner.iteration_stats().empty()) { out.error = "no iteration stats"; return out; }
    const fuse::IterationStats& st = runner.iteration_stats().front();
    out.x_a_per_window.push_back(pa->x());
    out.x_b_per_window.push_back(pb->x());
    out.iterations_per_window.push_back(st.iterations);
  }
  out.device_transport_moves =
      static_cast<long long>(runner.stats().device_transport_moves);
  out.device_boundary_copies =
      static_cast<long long>(runner.stats().device_boundary_copies);
  out.ok = true;
  return out;
}

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

constexpr int kPeriods = 5;

}  // namespace

TEST(WindowRunnerMpi, TwoDisjointRanksMatchTheSingleProcessReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  // Reference: every rank computes this identically, no MPI at all --
  // window_runner_multirate_test.cpp's own already-verified shape.
  const CouplingGraph ref_graph =
      build_window_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window(ref_graph, kPeriods, RunnerConfig{}, false);
  ASSERT_TRUE(ref.ok) << ref.error;

  // Under test: A on rank 0, B on rank 1 -- a genuine cross-rank window.
  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_graph(a_ranks, b_ranks);
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  const RunOutcome got = run_window(mpi_graph, kPeriods, cfg, false);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  // Found alongside the hosting defect this host run also exposed
  // (KNOWN_ISSUES.md D19, `WindowMemberConfig::export_consumer_hosted`):
  // since 0eb576a a member advances ONLY on its own processor set, so a
  // rank that does not host a given member never touches that member's
  // OWN `x()` at all -- it stays at its registered initial value there,
  // never the real, partner-updated state. Comparing `x_a_per_window`/
  // `x_b_per_window` on EVERY rank (the pre-this-fix code) therefore
  // compared A's own real trajectory against B's rank's own untouched
  // copy of A's initial condition, which is not a meaningful check and
  // was the direct cause of this test's own `got.x_b_per_window[0] ==
  // 0.2` (B's unmodified `kAY0`) symptom on the host. Each member's own
  // state is compared only on the ranks that host it -- the ranks.
  // contains() membership test the graph itself declares, not a literal
  // rank number -- while the AGREED sweep count stays compared on every
  // rank, since decide_window_convergence() must agree it everywhere.
  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      EXPECT_EQ(got.x_a_per_window[w], ref.x_a_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      EXPECT_EQ(got.x_b_per_window[w], ref.x_b_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    // ...and the agreed sweep count only on the ranks that TAKE PART in
    // the window: the region's participants are the ranks of the
    // components it couples, and this row also runs under `-n 3`, where
    // every further rank hosts neither member. Such a rank enters no
    // agreement, decides locally, and has no agreed count to report.
    // Measured on the host 2026-10-08: of three ranks, rank 2 was the only
    // one that failed, on exactly this comparison.
    if (hosts_a || hosts_b) {
      EXPECT_EQ(got.iterations_per_window[w], ref.iterations_per_window[w])
          << "window " << w << ", rank " << current_rank()
          << " -- every rank that takes part must agree the SAME sweep "
             "count (step (c), decide_window_convergence())";
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(WindowRunnerMpi, ThreeRanksWithAnUncoupledThirdMatchTheReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);

  const CouplingGraph ref_graph =
      build_window_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window(ref_graph, kPeriods, RunnerConfig{}, false);
  ASSERT_TRUE(ref.ok) << ref.error;

  // A on rank 0, B on rank 1, C (uncoupled, no channels) on rank 2.
  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_graph(
      a_ranks, b_ranks, /*with_uncoupled_c=*/true, RankSet{2, 1});
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  const RunOutcome got = run_window(mpi_graph, kPeriods, cfg, /*with_c=*/true);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  // Same correction as TwoDisjointRanksMatchTheSingleProcessReference's own
  // comment: a member's own state is observable only on the ranks that
  // host it (0eb576a). Rank 2 (C, uncoupled) hosts neither A nor B, so it
  // compares only the agreed sweep count, which is exactly the property
  // this test is named for.
  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      EXPECT_EQ(got.x_a_per_window[w], ref.x_a_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      EXPECT_EQ(got.x_b_per_window[w], ref.x_b_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    // Not asserted on C's own rank: C is uncoupled, so that rank carries
    // none of the window's channels and is not one of the region's
    // participants -- it never enters the agreement. What this test is
    // NAMED for is the property that does hold, and is checked here: with
    // an uncoupled component sitting on a third rank, the two ranks that
    // DO take part still reproduce the single-process reference exactly.
    if (hosts_a || hosts_b) {
      EXPECT_EQ(got.iterations_per_window[w], ref.iterations_per_window[w])
          << "window " << w << ", rank " << current_rank()
          << " -- an uncoupled third rank must not perturb the window's own "
             "agreed sweep count";
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}


// Transport-direct track (KNOWN_ISSUES.md D66, closed by D77). This
// fixture's own Transport arcs (a2b's and b2a's coupler-side buffers) are
// coupler storage, not aliased directly to either component's own field --
// under Device execution space they land on the device on BOTH ends, and
// `A`/`B`'s DISJOINT placement (rank 0, rank 1) makes this a genuine MPI
// exchange: exactly the both-arcs-device, matched-plan configuration that
// used to hand MpiTransport::move() a device pointer and segfault in its
// own std::memcpy (parallel.window_runner_mpi.{2,3}.device, this track's
// own RED state). `RunnerConfig::execution_space` is set explicitly, so
// this test runs under Device on the plain ctest row and the `.device`
// twin alike, independent of `fuse_add_mpi_test`'s own DEVICE_TWIN
// mechanism (FUSE_EXECUTION_SPACE only matters when execution_space is
// left at its Default).
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
TEST(WindowRunnerMpi, DeviceDirectTransportMatchesHostAndSetsMovesCounter) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  ::unsetenv("FUSE_TRANSPORT_HOST_STAGING");

  const CouplingGraph ref_graph =
      build_window_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window(ref_graph, kPeriods, RunnerConfig{}, false);
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_graph(a_ranks, b_ranks);
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  cfg.execution_space = RunnerConfig::RunnerExecutionSpace::Device;
  const RunOutcome got = run_window(mpi_graph, kPeriods, cfg, false);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  // Delivery: pure data movement end to end (the ODE integration itself is
  // identical regardless of which Transport path moved the coupled value),
  // so bitwise equality against the no-MPI reference is the right check,
  // exactly like the two tests above.
  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      EXPECT_EQ(got.x_a_per_window[w], ref.x_a_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      EXPECT_EQ(got.x_b_per_window[w], ref.x_b_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_a || hosts_b) {
      EXPECT_EQ(got.iterations_per_window[w], ref.iterations_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
  }

  // Whichever rank sends and which receives is an implementation detail of
  // the matching, not something this test pins; the SUM across the job is
  // what the configuration guarantees, the same discipline runner_
  // transport_test.cpp's own ADeviceResidentRunDeliversTheSourceValues
  // GpuAwareOrNot test uses.
  long long local_moves = got.device_transport_moves;
  long long total_moves = 0;
  ASSERT_EQ(MPI_Allreduce(&local_moves, &total_moves, 1, MPI_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD),
           MPI_SUCCESS);
  if (fuse::linalg::mpi_is_gpu_aware()) {
    EXPECT_GT(total_moves, 0)
        << "GPU-aware MPI did not take the device-direct Transport path "
           "(Transport::move_packed())";
  } else {
    EXPECT_EQ(total_moves, 0);
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// The host-staged fallback, forced via FUSE_TRANSPORT_HOST_STAGING=1, for
// THIS both-arcs-device configuration specifically -- runner_transport_
// test.cpp's own TheHostStagedFallbackDeliversTheSameValuesEvenWhenForced
// never reaches the both-device branch at all (its own header comment),
// so it cannot stand in for this row.
TEST(WindowRunnerMpi, DeviceDirectTransportHostStagedFallbackForced) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  ::setenv("FUSE_TRANSPORT_HOST_STAGING", "1", /*overwrite=*/1);

  const CouplingGraph ref_graph =
      build_window_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window(ref_graph, kPeriods, RunnerConfig{}, false);
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_graph(a_ranks, b_ranks);
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  cfg.execution_space = RunnerConfig::RunnerExecutionSpace::Device;
  const RunOutcome got = run_window(mpi_graph, kPeriods, cfg, false);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      EXPECT_EQ(got.x_a_per_window[w], ref.x_a_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      EXPECT_EQ(got.x_b_per_window[w], ref.x_b_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
    if (hosts_a || hosts_b) {
      EXPECT_EQ(got.iterations_per_window[w], ref.iterations_per_window[w])
          << "window " << w << ", rank " << current_rank();
    }
  }

  long long local_moves = got.device_transport_moves;
  long long local_boundary = got.device_boundary_copies;
  long long total_moves = 0, total_boundary = 0;
  ASSERT_EQ(MPI_Allreduce(&local_moves, &total_moves, 1, MPI_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD),
           MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&local_boundary, &total_boundary, 1, MPI_LONG_LONG,
                          MPI_SUM, MPI_COMM_WORLD),
           MPI_SUCCESS);
  EXPECT_EQ(total_moves, 0)
      << "FUSE_TRANSPORT_HOST_STAGING=1 did not force the fallback";
  EXPECT_GT(total_boundary, 0)
      << "no rank staged the both-device Transport arc through the "
         "host-staged fallback";
  ::unsetenv("FUSE_TRANSPORT_HOST_STAGING");
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}
#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP || KOKKOS_ENABLE_SYCL
