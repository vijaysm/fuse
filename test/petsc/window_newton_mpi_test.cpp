// KNOWN_ISSUES.md D31/D32 -- window Newton over the region's REAL
// participant ranks, PETSc substrate. Its OWN dedicated binary, not
// appended to test/parallel/window_newton_mpi_test.cpp, for the SAME
// reason test/petsc/brusselator_newton_test.cpp's own header comment
// gives: a PETSc call that aborts takes the WHOLE PROCESS down, and
// sharing a process with otherwise-PETSc-independent coverage would
// take that coverage's own ctest row down with it.
//
// Not run in this sandbox (same MPI_Init constraint every other
// parallel.*/petsc.* MPI test here names) -- build-verified only;
// registered as a host row, `petsc.window_newton_mpi.{2,3}`.
//
// THE FIXTURE: test/parallel/window_newton_mpi_test.cpp's own
// LinearOdeComponent/build_window_newton_graph(), reproduced verbatim
// (the identical file-local-helper reasoning that file's own header
// comment gives), with `decl.newton_config.substrate` set to
// `fuse::NewtonSubstrate::Petsc` and `fuse::petsc::install()` called
// once before the first `Runner::initialize()`.
//
// THE COMMUNICATOR. `fuse::petsc::solve()` (src/petsc/solve.cpp) now
// builds its own `Vec`/`Mat`/`SNES` on `RegionRuntime::newton_comm_f`
// -- the SAME split `ParticipantSetReducer` reduces over -- rather than
// `fuse::petsc::comm_self_f()` unconditionally (D32's own pre-D31
// scope). The comparison mirrors the native substrate's own MPI test
// exactly: Newton's own outer-iteration count agreed EXACTLY across
// participants (an integer, decided by SNES's own `VecNorm` over a
// properly-distributed `Vec`, immune to the `n = 2`-term reduction's
// own Higham round-off at this problem's `rtol`), and the final member
// states bounded by the SAME derived tolerance.

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/petsc/solve.hpp>
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
using fuse::NewtonSubstrate;
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

int current_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

// test/parallel/window_newton_mpi_test.cpp's own LinearOdeComponent,
// unchanged.
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

CouplingGraph build_window_newton_petsc_graph(RankSet a_ranks, RankSet b_ranks) {
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
  decl.acceleration.kind = AccelKind::Newton;
  decl.newton_config = NewtonConfig{};
  // KNOWN_ISSUES.md D31/D32: the one difference from the native
  // substrate's own fixture.
  decl.newton_config.substrate = NewtonSubstrate::Petsc;
  g.add_iteration_region(decl);
  return g;
}

struct RunOutcome {
  bool ok = false;
  std::string error;
  std::vector<double> x_a_per_window, x_b_per_window;
  std::vector<int> newton_iterations_per_window;
};

constexpr int kPeriods = 5;

RunOutcome run_window_newton_petsc(const CouplingGraph& g, const RunnerConfig& cfg) {
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

/// test/parallel/window_newton_mpi_test.cpp's own higham_gamma() --
/// Higham's recursive-summation bound for an n-term reduction.
double higham_gamma(int n) {
  const double u = 0.5 * std::numeric_limits<double>::epsilon();
  const double nu = static_cast<double>(n) * u;
  return nu / (1.0 - nu);
}

}  // namespace

TEST(WindowNewtonPetscMpi, TwoDisjointRanksMatchTheSingleProcessReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  fuse::petsc::install();

  const CouplingGraph ref_graph =
      build_window_newton_petsc_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window_newton_petsc(ref_graph, RunnerConfig{});
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  const CouplingGraph mpi_graph = build_window_newton_petsc_graph(a_ranks, b_ranks);
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  const RunOutcome got = run_window_newton_petsc(mpi_graph, cfg);
  ASSERT_TRUE(got.ok) << got.error << " (rank " << current_rank() << ")";

  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());

  for (std::size_t w = 0; w < ref.newton_iterations_per_window.size(); ++w) {
    if (!(hosts_a || hosts_b)) continue;
    EXPECT_EQ(got.newton_iterations_per_window[w], ref.newton_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank()
        << " -- SNES's own VecNorm, over a properly-distributed Vec, "
           "must agree the SAME outer iteration count on every "
           "participant";
  }

  const double gamma2 = higham_gamma(2);
  for (std::size_t w = 0; w < ref.x_a_per_window.size(); ++w) {
    if (hosts_a) {
      const double diff = std::fabs(got.x_a_per_window[w] - ref.x_a_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_a_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_a_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-petsc-mpi] rank %d window %zu A: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound) << "window " << w << ", rank " << current_rank();
    }
    if (hosts_b) {
      const double diff = std::fabs(got.x_b_per_window[w] - ref.x_b_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_b_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_b_per_window[w]);
      std::fprintf(stderr,
                   "[window-newton-petsc-mpi] rank %d window %zu B: |diff|=%.3e bound=%.3e\n",
                   current_rank(), w, diff, bound);
      EXPECT_LE(diff, bound) << "window " << w << ", rank " << current_rank();
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(WindowNewtonPetscMpi, ThreeRanksWithAnUncoupledThirdMatchTheReference) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  fuse::petsc::install();

  const CouplingGraph ref_graph =
      build_window_newton_petsc_graph(RankSet{0, 0}, RankSet{0, 0});
  const RunOutcome ref = run_window_newton_petsc(ref_graph, RunnerConfig{});
  ASSERT_TRUE(ref.ok) << ref.error;

  const RankSet a_ranks{0, 1};
  const RankSet b_ranks{1, 1};
  CouplingGraph mpi_graph = build_window_newton_petsc_graph(a_ranks, b_ranks);
  mpi_graph.add_component(
      ComponentDecl{.name = "C", .ranks = RankSet{2, 1}, .interval = Rational{1}});

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

  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();

  RunOutcome got;
  {
    const auto lowered = fuse::lower(mpi_graph);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
    fuse::SerialMeshBackend backend;
    ASSERT_TRUE(backend.load(mpi_graph.mesh(fuse::MeshId{0})).ok());
    const fuse::ClockSet clocks{mpi_graph};
    const auto sched = fuse::schedule(mpi_graph, lowered.dag, clocks,
                                      fuse::make_region_sizes(mpi_graph, backend));
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
    }
    got.ok = true;
  }

  const bool hosts_a = a_ranks.contains(current_rank());
  const bool hosts_b = b_ranks.contains(current_rank());
  ASSERT_EQ(got.x_a_per_window.size(), ref.x_a_per_window.size());
  const double gamma2 = higham_gamma(2);
  for (std::size_t w = 0; w < ref.newton_iterations_per_window.size(); ++w) {
    if (!(hosts_a || hosts_b)) continue;
    EXPECT_EQ(got.newton_iterations_per_window[w], ref.newton_iterations_per_window[w])
        << "window " << w << ", rank " << current_rank();
    if (hosts_a) {
      const double diff = std::fabs(got.x_a_per_window[w] - ref.x_a_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_a_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_a_per_window[w]);
      EXPECT_LE(diff, bound);
    }
    if (hosts_b) {
      const double diff = std::fabs(got.x_b_per_window[w] - ref.x_b_per_window[w]);
      const double bound = gamma2 * std::fabs(ref.x_b_per_window[w]) +
                           NewtonConfig{}.rtol * std::fabs(ref.x_b_per_window[w]);
      EXPECT_LE(diff, bound);
    }
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}
