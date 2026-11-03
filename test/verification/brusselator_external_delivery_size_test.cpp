// A tier-0 regression for the host failure
// `verification.brusselator.participants.lowfive.{mpmd,accept}.3` (MPICH
// ch4:ofi, the first run where both rows got past the launch-layout check
// and reached `initialize()`). `u_stdout.0.log` (the `lowfive-mpmd-3`
// scratch directory) printed `FAIL(u_main): initialize: status=-4
// ExternalDelivery instruction 1 reads 2816 values from arc %0 into 0 at
// arc %2`; `v_stdout.1.log` printed `ExternalDelivery instruction 1 reads 0
// values from arc %0 into 2816 at arc %2`.
//
// WHY THIS LEVEL. `src/runtime/runner.cpp`'s generic per-instruction
// buffer-size reconciliation (the loop that exempts `Remap` and
// `Transport`) must also exempt `OpKind::ExternalDelivery`; otherwise it
// compares an ExternalDelivery instruction's input arc (the EXPORTING
// component's own decomposition) against its output arc (the IMPORTING
// component's own decomposition -- "a placement is decomposed over its own
// RankSet", a DIFFERENT component's RankSet here, by
// `src/ir/rules_channel.cpp`'s own design: the delivered value lands on the
// consumer's placement, never the producer's). On a rank hosting exactly
// one end of the channel the two decompositions legitimately differ (one
// side's planned size, the other's empty view -- every rank resolves every
// placement, only a hosting rank reads one): a size equality that held
// before per-placement decomposition (or, here, before a LowFive-bound
// channel ever reached a REAL NativeMeshBackend partition) no longer does.
//
// WHY THIS GRAPH SHAPE. `brusselator_graph_tool.cpp`'s accept branch,
// `--ranks-u 1 --ranks-coupler 1 --ranks-v 1 --launch accept --transport
// lowfive` -- the exact row `examples/CMakeLists.txt` registers as
// `verification.brusselator.participants.lowfive.accept.3`: U's own
// RankSet is `{0, 1}`, the remap coupler's `{1, 1}`, V's `{2, 1}` (the
// accept-mode layout). `brusselator_accept_layout_test.cpp`'s own header
// comment explains that the IDENTICAL accept-layout graph with an ORDINARY
// (non-LowFive) transfer channel cannot reach `Runner::initialize()` at
// tier 0 at all -- two different partitions of one mesh need a real
// CommGraph, which needs MPI. Setting `FieldSpec::hdf5_path` on both
// transfer fields (`brusselator_graph_tool.cpp`'s own `--transport lowfive`
// wiring) compiles each transfer channel as `OpKind::ExternalDelivery`
// instead: FUSE moves no data on either channel (op.hpp's own header
// comment on `ExternalDelivery`), so nothing here ever needs a CommGraph --
// which is exactly what lets this graph reach `Runner::initialize()` at
// tier 0, on either end, with no MPI at all.
//
// NO MPI ANYWHERE. `fuse::RunnerConfig::force_this_process`
// (test/runtime/window_hosting_test.cpp's own convention) gives
// `initialize()` a specific rank's own `this_process` with no
// communicator; `fuse::native::NativeMeshBackend`'s own `(rank, num_ranks)`
// constructor arguments are set to agree with it, which is what
// `make_region_sizes()` and the per-rank decomposition this test is pinning
// actually consult.
#include <gtest/gtest.h>

#include <fuse/backend/native_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/field_spec.hpp>
#include <fuse/model/launch.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/schedule/plan.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "brusselator_graph.hpp"

using fuse::AccelKind;
using fuse::ComponentKind;
using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRef;
using fuse::FieldRegistry;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::Layout;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::MemorySpace;
using fuse::Rational;
using fuse::RankSet;
using fuse::Result;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::WaveformIteration;

namespace {

// Mirrors `examples/brusselator/participants/brusselator_grid.h`'s own
// `BRUSS_NX_U`/`BRUSS_NY_U` and `brusselator_graph_tool.cpp`'s own
// `kDtU`/`kDtV`/`kWindow`/`kTolerance`/`kMaxIterations` -- duplicated
// rather than shared through a header for the same reason test/
// verification/brusselator_lowfive_harness_test.cpp's own copy gives
// (the values are few; the two are different translation units). 256
// cells (16x16) times 11 window samples is this file's own oracle for
// the host's measured 2816.
constexpr int kNxU = 16, kNyU = 16;
const Rational kDtU{2, 1000}, kDtV{3, 1000}, kWindow{2, 100};
constexpr double kTolerance = 1e-11;
constexpr int kMaxIterations = 60;

// `test/runtime/window_hosting_test.cpp`'s own `StubBlock`, generalized
// from a one-element scalar to a `kNxU * kNyU`-element buffer: this
// graph's own window-member registration check (`src/runtime/
// iteration.cpp`'s "window member '<name>' registered N values for its
// export, but the plan sizes that field at M on this rank", gated on
// `cfg.hosted` -- it never fires for a component NOT hosted on this
// rank) wants the registered storage's OWN size to equal the field's
// planned size on this rank, which for a ONE-rank component (U's or
// V's own RankSet here) is the field's WHOLE `u_region` extent (256
// cells, both transfer fields' own `.region`, brusselator_graph.hpp's
// file header) -- never 1, the size window_hosting_test.cpp's own
// one-cell `line:1` fixture needed instead. advance() is never called
// by anything this file does (it stops at `Runner::initialize()`,
// never `run()`), so the storage is never read or written, only sized.
class StubBlock final : public fuse::Component {
 public:
  explicit StubBlock(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == Intent::Provides;
      std::vector<double>& storage = is_out ? out_ : in_;
      storage.assign(static_cast<std::size_t>(kNxU) * static_cast<std::size_t>(kNyU), 0.0);
      auto r = reg.bind(f, FieldRef{storage.data(),
                                    Layout{.stride = 1, .count = storage.size()},
                                    MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
  std::vector<double> out_;
  std::vector<double> in_;
};

struct LowFiveAcceptGraph {
  CouplingGraph g;
  fuse::MeshId u_mesh;
  fuse::RegionId u_region;
  brusselator::WindowGraphHandles handles;
};

// The accept-layout shape: `brusselator_graph_tool.cpp`'s own accept branch
// (`--ranks-u 1 --ranks-coupler 1 --ranks-v 1 --launch accept --transport
// lowfive`), built the same way `brusselator_accept_layout_test.cpp`'s
// own `build()` does (same RankSets, same `add_window_coupling()` call),
// plus `write_lowfive_declarations()`'s own two steps: `hdf5_path` on
// `u_out`/`v_out`, and the fieldless `LaunchMode::LowFive` marker
// component spanning every rank in use.
LowFiveAcceptGraph build() {
  LowFiveAcceptGraph out;
  out.u_mesh = out.g.add_mesh(
      {.name = "u", .source = "generated:quad:" + std::to_string(kNxU) + "x" +
                        std::to_string(kNyU)});
  out.u_region =
      out.g.add_region({.name = "u_all", .mesh = out.u_mesh, .dimension = 2});

  const RankSet u_ranks{0, 1};
  const RankSet coupler_ranks{1, 1};
  const RankSet v_ranks{2, 1};

  out.handles = brusselator::add_window_coupling(
      out.g, out.u_region, kDtU, kDtV, kWindow, kTolerance, kMaxIterations,
      AccelKind::None, WaveformIteration::Jacobi, coupler_ranks, u_ranks,
      v_ranks, LaunchSpec{}, LaunchSpec{}, LaunchSpec{});

  const auto u_out = out.g.find_field(out.handles.u_component, "u_out");
  const auto v_out = out.g.find_field(out.handles.v_component, "v_out");
  EXPECT_TRUE(u_out.has_value());
  EXPECT_TRUE(v_out.has_value());
  const_cast<FieldSpec&>(out.g.field(*u_out)).hdf5_path =
      "brusselator_u.h5/grid/u_out";
  const_cast<FieldSpec&>(out.g.field(*v_out)).hdf5_path =
      "brusselator_v.h5/grid/v_out";

  // `brusselator_graph_tool.cpp`'s own fieldless marker: `ComponentKind::
  // Coupler` (not the `ComponentDecl` default, `Physics`) so `Runner::
  // initialize()`'s own "every graph component needs an implementation"
  // rule (`src/runtime/runner.cpp`, the `if (decl.kind == ComponentKind::
  // Coupler) continue;` guard) exempts it exactly as it exempts the remap
  // coupler `add_window_coupling()` already added -- neither needs a
  // `StubBlock` in `components` below.
  (void)out.g.add_component(
      {.name = "lowfive",
       .ranks = RankSet{0, 3},
       .interval = Rational{1},
       .kind = ComponentKind::Coupler,
       .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  return out;
}

// Builds the backend/graph/dag/plan fresh for `this_rank` of 3 (a fresh
// `LowFiveAcceptGraph`/`NativeMeshBackend` per call, since `Runner` keeps
// a reference to both and this function's locals must outlive it) and
// calls `Runner::initialize()` with `RunnerConfig::force_this_process =
// this_rank` -- `window_hosting_test.cpp`'s own `initialize_forced()`
// convention, generalized to return the `Result` instead of asserting,
// so the caller can inspect the exact failure message under the mutation
// proof without `ASSERT_TRUE` aborting the test early.
Result<void> initialize_on_rank(int this_rank) {
  LowFiveAcceptGraph lg = build();
  fuse::native::NativeMeshBackend backend(/*rank=*/this_rank, /*num_ranks=*/3);
  // `backend.load(decl, graph)` (NOT `register_default_placements()`
  // directly, brusselator_accept_layout_test.cpp's own `build()`
  // precedent): the one-argument `load()` registers the mesh itself
  // (`impl_->meshes`/`impl_->names`, which `make_region_sizes()` below
  // consults by id) before the two-argument overload calls `register_
  // default_placements()` on it.
  const auto placed = backend.load(lg.g.mesh(lg.u_mesh), lg.g);
  if (!placed.ok()) return placed.error();

  const auto lowered = fuse::lower(lg.g);
  if (!lowered.ok()) {
    return fuse::Failure{"E-TEST-LOWER", lowered.diags.to_string()};
  }
  const fuse::ClockSet clocks{lg.g};
  const auto sched = fuse::schedule(lg.g, lowered.dag, clocks, fuse::make_region_sizes(lg.g, lowered.dag, backend));
  if (!sched.ok()) {
    return fuse::Failure{"E-TEST-SCHEDULE", sched.diags.to_string()};
  }

  ComponentSet components;
  components.add(std::make_unique<StubBlock>("U"));
  components.add(std::make_unique<StubBlock>("V"));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, lg.g);
    if (!r.ok()) return r;
  }

  Runner runner{lg.g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;
  cfg.force_this_process = this_rank;
  return runner.initialize(components, registry, cfg);
}

}  // namespace

// `src/runtime/runner.cpp`'s per-instruction size check exempts
// `OpKind::ExternalDelivery` the same way it exempts `Remap`/`Transport`;
// without that exemption this fails on BOTH ranks with the host's own two
// messages (asserting the defect's own message would make this test pass
// both with AND without the exemption, which proves nothing about which
// one is running).
//
// WHY THIS ASSERTS `E-RT-REGION`, NOT SUCCESS. `initialize()` on this REAL
// graph (the window-coupled Brusselator, an `IterationRegion` with
// convergence measures on BOTH `u_out` and `v_out`) gets PAST the
// `ExternalDelivery` check on both ranks and fails LATER, on a SEPARATE
// tier-0 limitation:
// `src/runtime/iteration.cpp`'s own `R.participants` (the region's
// cross-rank residual-reduction set) is populated ONLY `if (config.
// base_communicator.has_value())` (iteration.cpp, the `resolve_regions()`
// block starting "Checked against `config.base_communicator` rather than
// the graph's declared `RankSet` cardinality alone"); with no real MPI
// communicator (`RunnerConfig::force_this_process` alone, the only tier-0
// override that exists; MPI cannot start in the sandbox),
// `R.participants` stays empty on EVERY rank, so the SAME function's
// own "an empty coupling value... the region would report convergence
// after one sweep" refusal (iteration.cpp, `if (b.length == 0 &&
// R.participants.size() <= 1)`) fires for whichever convergence measure
// names the component NOT hosted here (`v_out` on U's rank, `u_out` on
// V's) -- exactly the SAME class of tier-0 ceiling `brusselator_accept_
// layout_test.cpp`'s own header comment already documents for the
// ORDINARY (non-LowFive) channel's `Transport`/CommGraph requirement,
// manifesting through a DIFFERENT check. On a REAL multi-process run
// `config.base_communicator` IS set, `R.participants` holds all 3 ranks
// (U, the coupler, V), `R.participants.size() > 1`, and this refusal
// never fires -- so this is a sandbox-only ceiling on DRIVING this real,
// multi-rank iteration region at tier 0, not a defect;
// `InitializeSucceedsOnDisjointRanksWithNoIterationRegion` below isolates
// the ExternalDelivery exemption's own effect with no iteration region in
// the way, and asserts the SUCCESS that only the host's own real
// multi-process run (ctest -R participants.lowfive in build-lowfive) can
// demonstrate for this exact graph.
TEST(ExternalDeliverySizeReconciliation, ReachesPastExternalDeliveryOnBothRanks) {
  const Result<void> u_side = initialize_on_rank(0);
  ASSERT_FALSE(u_side.ok()) << "expected the tier-0-only E-RT-REGION ceiling below, not success";
  EXPECT_EQ(u_side.error().code, "E-RT-REGION");
  EXPECT_NE(u_side.error().message.find("empty coupling value"), std::string::npos)
      << u_side.error().message;
  EXPECT_EQ(u_side.error().message.find("ExternalDelivery"), std::string::npos)
      << "the ExternalDelivery-specific defect must be absent: " << u_side.error().message;

  const Result<void> v_side = initialize_on_rank(2);
  ASSERT_FALSE(v_side.ok()) << "expected the tier-0-only E-RT-REGION ceiling below, not success";
  EXPECT_EQ(v_side.error().code, "E-RT-REGION");
  EXPECT_NE(v_side.error().message.find("empty coupling value"), std::string::npos)
      << v_side.error().message;
  EXPECT_EQ(v_side.error().message.find("ExternalDelivery"), std::string::npos)
      << "the ExternalDelivery-specific defect must be absent: " << v_side.error().message;
}

namespace {

// The narrowest reachable function, for when the real graph above cannot
// reach a clean `initialize()` success at tier 0 (the `E-RT-REGION`
// ceiling just above, unrelated to the exemption). Minimal shape that still compiles a channel as `OpKind::
// ExternalDelivery` on two DISJOINT ranks: two plain (non-window)
// components, A (`RankSet{0, 1}`) and B (`RankSet{1, 1}`), one ordinary
// channel A -> B whose source field carries `hdf5_path`, NO
// `IterationRegion` at all -- so `resolve_regions()`'s own convergence-
// measure/`R.participants` machinery above never runs, and `initialize()`
// exercises exactly, and only, the per-instruction size reconciliation
// (`src/runtime/runner.cpp`'s `resolve_instructions()`, the function `Runner::initialize()` calls this file's
// own real-graph test above also reaches, just without anything AFTER it
// to fail on).
struct DisjointExternalDeliveryGraph {
  CouplingGraph g;
  fuse::MeshId mesh;
  fuse::RegionId region;
  fuse::ComponentId a, b;
  fuse::FieldId a_out, b_in;
};

constexpr int kNxAB = 4, kNyAB = 4;  // 16 cells -- small and distinct from kNxU/kNyU above.

DisjointExternalDeliveryGraph build_disjoint_external_delivery_graph() {
  DisjointExternalDeliveryGraph out;
  out.mesh = out.g.add_mesh(
      {.name = "m", .source = "generated:quad:" + std::to_string(kNxAB) + "x" +
                        std::to_string(kNyAB)});
  out.region = out.g.add_region({.name = "m_all", .mesh = out.mesh, .dimension = 2});

  out.a = out.g.add_component(
      {.name = "A", .ranks = RankSet{0, 1}, .interval = Rational{1}});
  out.b = out.g.add_component(
      {.name = "B", .ranks = RankSet{1, 1}, .interval = Rational{1}});

  out.a_out = out.g.add_field(
      out.a, FieldSpec{.name = "a_out",
                      .region = out.region,
                      .units = fuse::units::kDimensionless,
                      .conservation = fuse::Conservation::Extensive,
                      .intent = Intent::Provides});
  out.b_in = out.g.add_field(
      out.b, FieldSpec{.name = "b_in",
                       .region = out.region,
                       .units = fuse::units::kDimensionless,
                       .conservation = fuse::Conservation::Extensive,
                       .intent = Intent::Requires});
  (void)out.g.add_channel("a2b", out.a_out, out.b_in);

  const_cast<FieldSpec&>(out.g.field(out.a_out)).hdf5_path = "ab.h5/grid/a_out";
  (void)out.g.add_component(
      {.name = "lowfive",
       .ranks = RankSet{0, 2},
       .interval = Rational{1},
       .kind = ComponentKind::Coupler,
       .launch = LaunchSpec{.mode = LaunchMode::LowFive}});
  return out;
}

class DisjointStubBlock final : public fuse::Component {
 public:
  // `this_rank`: unlike `StubBlock` above (every field there is window-
  // scoped, iteration.cpp's own `cfg.hosted`-gated check, skipped
  // entirely for a component not hosted here), an ORDINARY field goes
  // through `runner.cpp`'s own unconditional bind check (`resolve_
  // buffers()`, no hosting exemption): the registered `Layout::count`
  // must equal `b.elements` -- `sizes(region, this component's own
  // rank_set)` -- EXACTLY, on EVERY rank, which is the region's whole
  // extent where this component is hosted and an "empty view of the right
  // dimension" (0) where it is not. `register_fields()` has no
  // other way to learn which this rank is, so it is passed in here.
  DisjointStubBlock(std::string name, int this_rank)
      : name_(std::move(name)), this_rank_(this_rank) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    const bool hosted = g.component(*cid).ranks.contains(this_rank_);
    const std::size_t count =
        hosted ? static_cast<std::size_t>(kNxAB) * static_cast<std::size_t>(kNyAB) : 0;
    for (const fuse::FieldId f : g.component(*cid).fields) {
      storage_.assign(count, 0.0);
      auto r = reg.bind(f, FieldRef{storage_.data(),
                                    Layout{.stride = 1, .count = storage_.size()},
                                    MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> save_state() override { return fuse::Result<void>::success(); }
  fuse::Result<void> restore_state() override { return fuse::Result<void>::success(); }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
  int this_rank_;
  std::vector<double> storage_;
};

Result<void> initialize_disjoint_on_rank(int this_rank) {
  DisjointExternalDeliveryGraph dg = build_disjoint_external_delivery_graph();
  fuse::native::NativeMeshBackend backend(/*rank=*/this_rank, /*num_ranks=*/2);
  const auto placed = backend.load(dg.g.mesh(dg.mesh), dg.g);
  if (!placed.ok()) return placed.error();

  const auto lowered = fuse::lower(dg.g);
  if (!lowered.ok()) return fuse::Failure{"E-TEST-LOWER", lowered.diags.to_string()};
  const fuse::ClockSet clocks{dg.g};
  const auto sched = fuse::schedule(dg.g, lowered.dag, clocks, fuse::make_region_sizes(dg.g, lowered.dag, backend));
  if (!sched.ok()) return fuse::Failure{"E-TEST-SCHEDULE", sched.diags.to_string()};

  ComponentSet components;
  components.add(std::make_unique<DisjointStubBlock>("A", this_rank));
  components.add(std::make_unique<DisjointStubBlock>("B", this_rank));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, dg.g);
    if (!r.ok()) return r;
  }

  Runner runner{dg.g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;
  cfg.force_this_process = this_rank;
  return runner.initialize(components, registry, cfg);
}

}  // namespace

// Asserts that initialize succeeds on both sides, at the narrowest level
// that can actually reach it for an `ExternalDelivery` channel on disjoint ranks with no
// real MPI communicator: A alone (rank 0, the exporter) and B alone
// (rank 1, the importer).
TEST(ExternalDeliverySizeReconciliation, InitializeSucceedsOnDisjointRanksWithNoIterationRegion) {
  const Result<void> a_side = initialize_disjoint_on_rank(0);
  ASSERT_TRUE(a_side.ok()) << a_side.error().code << ": " << a_side.error().message;
  const Result<void> b_side = initialize_disjoint_on_rank(1);
  ASSERT_TRUE(b_side.ok()) << b_side.error().code << ": " << b_side.error().message;
}
