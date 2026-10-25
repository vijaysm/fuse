#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/plan.hpp>

#include <algorithm>
#include <cstddef>
#include <locale>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using fuse::ComponentDecl;
using fuse::CouplingGraph;
using fuse::OpKind;
using fuse::Rational;
namespace u = fuse::units;

namespace {

CouplingGraph pair(Rational src_dt, Rational dst_dt) {
  CouplingGraph g;
  const auto mf = g.add_mesh(fuse::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(fuse::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(fuse::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(fuse::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = src_dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = dst_dt});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).remap(fuse::RemapFlags::Conservative);
  return g;
}

fuse::RegionSizes constant_sizes(std::size_t n) {
  return [n](fuse::RegionId, int /*placement*/) -> std::size_t { return n; };
}

fuse::ScheduleResult build(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  return schedule(g, lowered.dag, cs,
                  [](fuse::RegionId, int /*placement*/) -> std::size_t { return 64; });
}

bool has_code(const fuse::Diagnostics& d, const char* code) {
  for (const auto& item : d.items()) {
    if (item.code == code) return true;
  }
  return false;
}

/// Five components, each providing one field into the same target, with a
/// declared Sum merge -- an entirely ordinary use of spec Section 5.1's Merge that
/// used to compile past lower() as one 5-input Merge node and then fail
/// schedule() with E-SCHED-ARITY, blaming a "lower Merge as a left-fold"
/// step that lower() never performed. All writers are
/// colocated with the target so no Transport/Epoch pair complicates the
/// count of Merge nodes below. `lagged_writer` (or -1 for none) names the
/// one writer, by index, that lags one step behind the rest.
CouplingGraph five_writer_merge(int lagged_writer) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto dst = g.add_component(ComponentDecl{
      .name = "dst", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto in = g.add_field(
      dst, fuse::FieldSpec{.name = "f_in", .region = r,
                           .units = u::kHeatFluxDensity,
                           .conservation = fuse::Conservation::Extensive,
                           .intent = fuse::Intent::Requires});
  for (int i = 0; i < 5; ++i) {
    const auto src = g.add_component(
        ComponentDecl{.name = "src" + std::to_string(i),
                      .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
    const auto out = g.add_field(
        src, fuse::FieldSpec{.name = "f_out", .region = r,
                             .units = u::kHeatFluxDensity,
                             .conservation = fuse::Conservation::Extensive,
                             .intent = fuse::Intent::Provides});
    const auto ch = g.add_channel("ch" + std::to_string(i), out, in);
    if (i == lagged_writer) {
      g.channel(ch).lag(1).allow_time_level_mismatch("explicit lag fixture");
    }
  }
  g.merge(in, fuse::MergeOp::Sum);
  return g;
}

/// Two writers on DIFFERENT ranks than the target and than each other,
/// merged with a declared Sum -- unlike `five_writer_merge()` above, whose
/// writers are colocated with the target so lowering never inserts a
/// Transport/Epoch pair. Colocated writers happened to schedule correctly by
/// accident: with no Transport/Epoch, every node in the graph landed at the
/// same instant and NodeId order alone kept Merge after its inputs. The
/// moment a writer is NOT colocated, its Transport/Epoch pair runs at the
/// destination's tick END, so Merge must not be placed at the tick START:
/// it would sort before the Epoch outputs it reads and the scheduler's own
/// dependence check would refuse the plan with E-SCHED-ORDER.
CouplingGraph merge_across_ranks() {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto dst = g.add_component(ComponentDecl{
      .name = "dst", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto in = g.add_field(
      dst, fuse::FieldSpec{.name = "f_in", .region = r,
                           .units = u::kHeatFluxDensity,
                           .conservation = fuse::Conservation::Extensive,
                           .intent = fuse::Intent::Requires});
  for (int i = 0; i < 2; ++i) {
    const auto src = g.add_component(
        ComponentDecl{.name = "src" + std::to_string(i),
                      .ranks = fuse::RankSet{i + 1, 1},
                      .interval = Rational{1}});
    const auto out = g.add_field(
        src, fuse::FieldSpec{.name = "f_out", .region = r,
                             .units = u::kHeatFluxDensity,
                             .conservation = fuse::Conservation::Extensive,
                             .intent = fuse::Intent::Provides});
    g.add_channel("ch" + std::to_string(i), out, in);
  }
  g.merge(in, fuse::MergeOp::Sum);
  return g;
}

/// Two writers at DIFFERENT intervals -- W1 ticks every 2 time units, W2
/// every 1 -- merged (Sum) into M, which ticks every 1. Colocated (same rank
/// as the target) on purpose, so the only thing this fixture exercises is
/// the interaction between multirate decomposition and the Merge phase, not
/// a Transport/Epoch pair (`MergeAcrossRanksSchedulesInDependenceOrder`
/// already covers that). W1's channel carries a real chain node (`.relax`)
/// so its tail is not the bare Source/Sink arc.
///
/// W1's slab decomposition against M's interval
/// (`decompose(2, 1, period)`) makes its FIRST merge instant target-limited
/// -- the source step has not finished when M's first tick ends, so the
/// causality contract in expand.hpp stamps W1's tail `AfterTicks` there --
/// so the boundary pass must not stamp every Merge `BeforeTicks`
/// unconditionally: at the same instant that is the wrong relative order, and
/// `check_dependence_order()` would refuse this legal graph with
/// `E-SCHED-ORDER`, reading it as "Merge ran before the input it combines."
CouplingGraph multirate_merge() {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto dst = g.add_component(ComponentDecl{
      .name = "dst", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto in = g.add_field(
      dst, fuse::FieldSpec{.name = "f_in", .region = r,
                           .units = u::kHeatFluxDensity,
                           .conservation = fuse::Conservation::Extensive,
                           .intent = fuse::Intent::Requires});
  const auto w1 = g.add_component(ComponentDecl{
      .name = "w1", .ranks = fuse::RankSet{0, 1}, .interval = Rational{2}});
  const auto out1 = g.add_field(
      w1, fuse::FieldSpec{.name = "f_out", .region = r,
                          .units = u::kHeatFluxDensity,
                          .conservation = fuse::Conservation::Extensive,
                          .intent = fuse::Intent::Provides});
  const auto ch1 = g.add_channel("ch1", out1, in);
  g.channel(ch1).relax(0.7);
  const auto w2 = g.add_component(ComponentDecl{
      .name = "w2", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto out2 = g.add_field(
      w2, fuse::FieldSpec{.name = "f_out", .region = r,
                          .units = u::kHeatFluxDensity,
                          .conservation = fuse::Conservation::Extensive,
                          .intent = fuse::Intent::Provides});
  g.add_channel("ch2", out2, in);
  g.merge(in, fuse::MergeOp::Sum);
  return g;
}

/// Independent of the scheduler's own reasoning: walk the emitted stream and
/// check that every operand was written by an earlier instruction. A lagged
/// arc (negative time offset) is exempt -- spec Section 4.3, its value comes from the
/// previous period on purpose.
std::size_t dependence_violations(const fuse::OperatorDAG& dag,
                                  const fuse::ExecutionPlan& plan) {
  std::set<std::uint32_t> produced;
  std::size_t violations = 0;
  for (const auto& ins : plan.instructions()) {
    const fuse::Node& n = dag.node(ins.origin_node);
    for (const fuse::ArcId in : n.inputs) {
      if (!dag.arc(in).producer.valid()) continue;
      if (dag.arc(in).type.time.offset < 0) continue;  // lag: crosses periods
      if (produced.count(in.value()) == 0) ++violations;
    }
    if (ins.out.valid()) produced.insert(ins.out.value());
  }
  return violations;
}

}  // namespace

// ---------------------------------------------------------------------------
// Basic plans
// ---------------------------------------------------------------------------

TEST(ExecutionPlan, BuildsFromAValidGraph) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_FALSE(r.plan.instructions().empty());
}

TEST(ExecutionPlan, InstructionsAreInNonDecreasingTimeOrder) {
  const auto r = build(pair(Rational{1, 3}, Rational{1}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto& is = r.plan.instructions();
  for (std::size_t i = 1; i < is.size(); ++i) {
    EXPECT_LE(is[i - 1].time, is[i].time);
  }
}

TEST(ExecutionPlan, RecordsThePeriodAndLockstepFlag) {
  const auto lock = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(lock.ok());
  EXPECT_TRUE(lock.plan.lockstep());

  const auto multi = build(pair(Rational{1, 3}, Rational{1}));
  ASSERT_TRUE(multi.ok());
  EXPECT_FALSE(multi.plan.lockstep());
  EXPECT_EQ(multi.plan.period(), Rational{1});
}

TEST(ExecutionPlan, EveryInstructionsInputsAreDeclaredBuffers) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok());
  const std::size_t n_buffers = r.plan.buffers().buffers.size();
  for (const auto& ins : r.plan.instructions()) {
    for (std::uint8_t k = 0; k < ins.n_in; ++k) {
      ASSERT_TRUE(ins.in[k].valid());
      EXPECT_LT(ins.in[k].value(), n_buffers);
    }
    if (ins.out.valid()) EXPECT_LT(ins.out.value(), n_buffers);
  }
}

TEST(ExecutionPlan, RejectsCyclicSchedules) {
  // Unlagged two-way coupling.
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = m});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}});
  const auto ao = g.add_field(a, fuse::FieldSpec{
      .name = "ao", .region = ra, .intent = fuse::Intent::Provides});
  const auto bi = g.add_field(b, fuse::FieldSpec{
      .name = "bi", .region = rb, .intent = fuse::Intent::Requires});
  const auto bo = g.add_field(b, fuse::FieldSpec{
      .name = "bo", .region = rb, .intent = fuse::Intent::Provides});
  const auto ai = g.add_field(a, fuse::FieldSpec{
      .name = "ai", .region = ra, .intent = fuse::Intent::Requires});
  g.add_channel("a2b", ao, bi);
  g.add_channel("b2a", bo, ai);

  const auto r = build(g);
  EXPECT_FALSE(r.ok());
  bool saw_cycle_code = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-SCHED-CYCLE") saw_cycle_code = true;
  }
  EXPECT_TRUE(saw_cycle_code) << r.diags.to_string();

  // The gate is hard: a rejected graph yields no plan at all, not a plan the
  // runtime is expected to notice is unusable.
  EXPECT_TRUE(r.plan.instructions().empty());
  EXPECT_TRUE(r.plan.buffers().buffers.empty());
  EXPECT_EQ(r.plan.region_count(), 0u);
}

// A window member's own export is produced by
// OpKind::WindowAdvance -- an SSA root with empty inputs, exactly like
// Source, finish_window_regions() in src/ir/window_lowering.cpp -- not by
// Source itself. The acyclicity prover's black-box component-edge rule
// (src/schedule/acyclicity.cpp) must build that edge from WindowAdvance
// endpoints as well as Source/Sink, or no component edge is formed for a
// window member's step. An unlagged loop through the member via ORDINARY channels declared
// OUTSIDE the window-scoped region -- which neither the window gate's own
// `has_unbroken_cycle()` (src/schedule/window.cpp, scans only
// `region.channels`) nor the prover's endpoint scan could see -- would
// be wrongly certified acyclic. `a` is the window member; `a2c`/`c2a` are
// ordinary, unlagged, and not in the region's channel list.
TEST(ExecutionPlan, WindowMemberGatesAnOrdinaryCycleOutsideTheRegion) {
  CouplingGraph g;
  const auto ma = g.add_mesh(fuse::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(fuse::MeshDecl{.name = "mb"});
  const auto mc = g.add_mesh(fuse::MeshDecl{.name = "mc"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = mc});

  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1, 4}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1, 4}});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1, 4}});

  // The window-scoped pair, `a` <-> `b` -- mirrors
  // window_region_golden_test.cpp's two_member_window_graph(): one unlagged
  // direction (checked by check_window_causality()'s strict rule) and one
  // lagged (its relaxed rule), which is what makes the region itself
  // schedulable.
  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep, not a real-time lag");

  fuse::CouplingWindowScope scope;
  scope.window_component = "a";
  scope.order = 2;
  scope.policies = {{"a", fuse::DtPolicy::Adjustable},
                    {"b", fuse::DtPolicy::Adjustable}};
  scope.iteration = fuse::WaveformIteration::GaussSeidel;
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "window", .channels = {fwd, back}, .max_iterations = 20,
      .window_scope = scope});

  // The ordinary, unlagged cycle runs a -> b -> c -> a, closing back on `a`
  // through `a_out` -- the window member's own export, never a second
  // Provides field declared directly on `a`. `a` must have NO ordinary
  // `Source` node of its own: finish_window_regions() (src/ir/
  // window_lowering.cpp) replaces a window member's export with a
  // `WindowAdvance` node, which is the entire point under test. Giving `a`
  // an unrelated ordinary Provides field here would let prove_acyclic's
  // pre-existing Source/Sink rule close the loop on its own, exercising
  // nothing about the gap this fix closes (measured: with a second Provides
  // field on `a`, this test passed even with the fix reverted). Nor can the
  // cycle's outgoing leg read `a_out` through a SECOND, non-window channel
  // directly: the window member's export is sized for the window's own
  // waveform, so an ordinary reader outside the region fails type-checking
  // before acyclicity is ever reached (measured: E-IR-TYPE on `c.c_in`).
  // Routing the outgoing leg through `b` -- already the window's OTHER,
  // ordinary consumer of `a_out` via `fwd` -- sidesteps both problems: `b`'s
  // own, unrelated Provides field closes the loop into `c`, and `c`'s own
  // Provides field closes it back onto `a_in2`, which is unlagged and
  // genuinely depends on `a_out` only through the missing WindowAdvance
  // component edge this fix adds.
  const auto b_out2 = g.add_field(b, fuse::FieldSpec{
      .name = "b_out2", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto c_in = g.add_field(c, fuse::FieldSpec{
      .name = "c_in", .region = rc, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto c_out = g.add_field(c, fuse::FieldSpec{
      .name = "c_out", .region = rc, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto a_in2 = g.add_field(a, fuse::FieldSpec{
      .name = "a_in2", .region = ra, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  g.add_channel("b2c", b_out2, c_in);
  g.add_channel("c2a", c_out, a_in2);

  const auto r = build(g);
  EXPECT_FALSE(r.ok());  // wrongly accepted before the fix
  bool saw_cycle_code = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-SCHED-CYCLE") saw_cycle_code = true;
  }
  EXPECT_TRUE(saw_cycle_code) << r.diags.to_string();
}

// A component claimed by two DIFFERENT window-scoped
// regions must not be silently resolved first-wins -- ClockSet's ctor
// (include/fuse/schedule/clock.hpp) takes the first region's window length
// and `break`s, and owning_window_region_for_component() (src/schedule/
// expand.cpp) `return`s on the first match, which would give a wrong-but-deterministic
// schedule with no diagnostic. `b` is a member of both `ab` (with `a`) and
// `bc` (with `c`) below; schedule() must refuse it with
// E-SCHED-WINDOW-OVERLAP rather than silently pick one.
TEST(ExecutionPlan, ComponentInTwoWindowRegionsIsRefused) {
  CouplingGraph g;
  const auto ma = g.add_mesh(fuse::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(fuse::MeshDecl{.name = "mb"});
  const auto mc = g.add_mesh(fuse::MeshDecl{.name = "mc"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb});
  const auto rc = g.add_region(fuse::Region{.name = "rc", .mesh = mc});

  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1, 4}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1, 4}});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1, 4}});

  // Region "ab": a <-> b.
  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto ab_fwd = g.add_channel("a2b", a_out, b_in);
  const auto ab_back = g.add_channel("b2a", b_out, a_in);
  g.channel(ab_back).lag(1).allow_time_level_mismatch(
      "resolved by the a-b coupling window's own sweep");

  fuse::CouplingWindowScope ab_scope;
  ab_scope.window_component = "a";
  ab_scope.order = 2;
  ab_scope.policies = {{"a", fuse::DtPolicy::Adjustable},
                       {"b", fuse::DtPolicy::Adjustable}};
  ab_scope.iteration = fuse::WaveformIteration::GaussSeidel;
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "ab", .channels = {ab_fwd, ab_back}, .max_iterations = 20,
      .window_scope = ab_scope});

  // Region "bc": b <-> c. `b` is also a member of "ab" above.
  const auto b_out2 = g.add_field(b, fuse::FieldSpec{
      .name = "b_out2", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto c_in = g.add_field(c, fuse::FieldSpec{
      .name = "c_in", .region = rc, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto c_out = g.add_field(c, fuse::FieldSpec{
      .name = "c_out", .region = rc, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto b_in2 = g.add_field(b, fuse::FieldSpec{
      .name = "b_in2", .region = rb, .units = u::kKelvin,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto bc_fwd = g.add_channel("b2c", b_out2, c_in);
  const auto bc_back = g.add_channel("c2b", c_out, b_in2);
  g.channel(bc_back).lag(1).allow_time_level_mismatch(
      "resolved by the b-c coupling window's own sweep");

  fuse::CouplingWindowScope bc_scope;
  bc_scope.window_component = "c";
  bc_scope.order = 2;
  bc_scope.policies = {{"b", fuse::DtPolicy::Adjustable},
                       {"c", fuse::DtPolicy::Adjustable}};
  bc_scope.iteration = fuse::WaveformIteration::GaussSeidel;
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "bc", .channels = {bc_fwd, bc_back}, .max_iterations = 20,
      .window_scope = bc_scope});

  const auto r = build(g);
  EXPECT_FALSE(r.ok());  // wrongly accepted (first-wins) before the fix
  bool saw_overlap_code = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-SCHED-WINDOW-OVERLAP") saw_overlap_code = true;
  }
  EXPECT_TRUE(saw_overlap_code) << r.diags.to_string();
}

TEST(ExecutionPlan, DumpMentionsEveryOpKindPresent) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok());
  const std::string s = r.plan.to_string();
  EXPECT_NE(s.find("# fuse-plan v1"), std::string::npos);
  EXPECT_NE(s.find("Remap"), std::string::npos);
  EXPECT_NE(s.find("period="), std::string::npos);
}

TEST(ExecutionPlan, DumpIsStable) {
  const auto r = build(pair(Rational{1, 2}, Rational{1}));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.plan.to_string(), r.plan.to_string());
}

// ---------------------------------------------------------------------------
// Ordering and error-reporting regressions
// ---------------------------------------------------------------------------

/// expand() emits in (time, NodeId) order, which is a dependence order only
/// while ascending NodeId is a topological order of the untimed DAG. Here it
/// is not: node 0 consumes what node 1 writes, at the same time. Emitting the
/// stream as-is would make its first instruction read an arc its second
/// instruction writes. The scheduler must *derive* the order from dependence
/// and fix it -- not merely detect it, and certainly not reject the graph.
TEST(ExecutionPlan, OrdersByDependenceNotByNodeId) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto reg = g.add_region(fuse::Region{.name = "r", .mesh = m});
  g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});

  fuse::OperatorDAG dag;
  fuse::ValueType vt;
  vt.region = reg;
  vt.rank_set = 0;
  const fuse::ArcId a0 = dag.add_arc(vt, "a0");
  const fuse::ArcId a1 = dag.add_arc(vt, "a1");
  // Node 0 consumes a0; a0's producer is node 1, appended afterwards. Single
  // assignment still holds, so the DAG accepts it -- but ascending NodeId is
  // no longer a topological order.
  dag.add_node(OpKind::Sink, {a0}, a1, fuse::OpAttrs{}, fuse::ChannelId{},
               "sink");
  dag.add_node(OpKind::Source, {}, a0, fuse::OpAttrs{}, fuse::ChannelId{},
               "source");

  const fuse::ClockSet cs{g};
  const auto timed = expand(g, dag, cs);
  ASSERT_TRUE(timed.ok()) << timed.error().message;
  ASSERT_EQ(timed.value().nodes.size(), 2u);
  // expand() emitted the consumer first, both at t=0.
  EXPECT_EQ(timed.value().nodes[0].node.value(), 0u);
  EXPECT_EQ(timed.value().nodes[1].node.value(), 1u);

  const auto r = schedule(g, dag, cs, constant_sizes(8));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  ASSERT_EQ(r.plan.instructions().size(), 2u);
  EXPECT_EQ(r.plan.instructions()[0].kind, OpKind::Source);
  EXPECT_EQ(r.plan.instructions()[1].kind, OpKind::Sink);
  EXPECT_EQ(dependence_violations(dag, r.plan), 0u);
}

/// A spurious `CycleReport::emission_order_valid == false` must not stop a
/// plan. The flag is derived from prove_acyclic()'s over-approximated
/// dependence graph and fires on clean multirate schedules; this is one of the
/// shapes it fires on. If schedule() ever grows a dependency on the flag, this
/// test is what catches it.
TEST(ExecutionPlan, DoesNotFailOnASpuriousEmissionOrderReport) {
  CouplingGraph g = pair(Rational{2, 3}, Rational{1, 2});
  g.channel(fuse::ChannelId{0}).accumulate(fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  const auto timed = expand(g, lowered.dag, cs);
  ASSERT_TRUE(timed.ok()) << timed.error().message;
  const fuse::CycleReport rep = prove_acyclic(g, lowered.dag, timed.value());
  ASSERT_TRUE(rep.acyclic) << rep.explanation;

  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  EXPECT_TRUE(r.ok()) << "emission_order_valid=" << rep.emission_order_valid
                      << "\n" << rep.order_explanation << "\n"
                      << r.diags.to_string();
  EXPECT_EQ(dependence_violations(lowered.dag, r.plan), 0u);
}

/// A sweep over the multirate space: every plan must build and every plan must
/// be dependence-ordered. This is the property the ~8% spurious
/// emission_order_valid failures would have broken.
TEST(ExecutionPlan, EveryMultirateRatioProducesADependenceOrderedPlan) {
  const std::int64_t nums[] = {1, 2, 3, 5};
  const std::int64_t dens[] = {1, 2, 3, 4, 7};
  std::size_t checked = 0;
  for (const std::int64_t sn : nums) {
    for (const std::int64_t sd : dens) {
      for (const std::int64_t tn : nums) {
        for (const std::int64_t td : dens) {
          for (const bool accum : {false, true}) {
            CouplingGraph g = pair(Rational{sn, sd}, Rational{tn, td});
            if (accum) {
              g.channel(fuse::ChannelId{0}).accumulate(fuse::AccumOp::Mean);
            }
            const auto lowered = lower(g);
            ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
            const fuse::ClockSet cs{g};
            const auto r = schedule(g, lowered.dag, cs, constant_sizes(8));
            // A ratio whose period is unrepresentable is a legitimate
            // E-SCHED-PERIOD; anything else must build.
            if (!r.ok()) {
              ASSERT_TRUE(has_code(r.diags, "E-SCHED-PERIOD"))
                  << sn << "/" << sd << " -> " << tn << "/" << td
                  << " accum=" << accum << ": " << r.diags.to_string();
              continue;
            }
            ASSERT_EQ(dependence_violations(lowered.dag, r.plan), 0u)
                << sn << "/" << sd << " -> " << tn << "/" << td
                << " accum=" << accum;
            ++checked;
          }
        }
      }
    }
  }
  EXPECT_GE(checked, 200u);
}

/// Emitted plans are dependence-ordered for every emission convention
/// expand() uses: chain nodes at the slab end, Sample at the source tick
/// start, reductions at the target step end.
TEST(ExecutionPlan, InstructionsAreInDependenceOrder) {
  const Rational ratios[][2] = {{Rational{1}, Rational{1}},
                                {Rational{1, 3}, Rational{1}},
                                {Rational{1}, Rational{1, 3}},
                                {Rational{2, 3}, Rational{1, 2}}};
  for (const auto& rt : ratios) {
    const CouplingGraph g = pair(rt[0], rt[1]);
    const auto lowered = lower(g);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
    const fuse::ClockSet cs{g};
    const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
    ASSERT_TRUE(r.ok()) << r.diags.to_string();
    EXPECT_EQ(dependence_violations(lowered.dag, r.plan), 0u)
        << rt[0].to_string() << " -> " << rt[1].to_string();
  }
}

TEST(ExecutionPlan, InstructionsAreInDependenceOrderWhenAccumulating) {
  CouplingGraph g = pair(Rational{1, 2}, Rational{1});
  g.channel(fuse::ChannelId{0}).accumulate(fuse::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_EQ(dependence_violations(lowered.dag, r.plan), 0u);
}

/// plan_buffers() reports by throwing. schedule() has a Diagnostics channel,
/// so an init-time setup error must arrive there instead of as an exception
/// unwinding out of the scheduler.
TEST(ExecutionPlan, ReportsAnEmptyRegionSizesCallbackAsADiagnostic) {
  const CouplingGraph g = pair(Rational{1}, Rational{1});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  fuse::ScheduleResult r = schedule(g, lowered.dag, cs, fuse::RegionSizes{});
  EXPECT_FALSE(r.ok());
  EXPECT_TRUE(has_code(r.diags, "E-SCHED-BUFFERS")) << r.diags.to_string();
  EXPECT_TRUE(r.plan.instructions().empty());
}

TEST(ExecutionPlan, ReportsABufferTotalThatDoesNotFitSizeTAsADiagnostic) {
  const CouplingGraph g = pair(Rational{1}, Rational{1});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  // Two such regions already exceed size_t, so the running total wraps unless
  // it is checked. plan_buffers() checks it and throws; schedule() must turn
  // that into a diagnostic rather than let it escape.
  const auto r = schedule(g, lowered.dag, cs,
                          constant_sizes(static_cast<std::size_t>(-1) / 2));
  EXPECT_FALSE(r.ok());
  EXPECT_TRUE(has_code(r.diags, "E-SCHED-BUFFERS")) << r.diags.to_string();
  EXPECT_EQ(r.plan.buffers().total_coupler_elements, 0u);
}

/// More inputs than Instruction::in holds must refuse the plan, not truncate
/// the operand list to three and report an error alongside a usable-looking
/// plan -- and it must report once per node, not once per instance.
TEST(ExecutionPlan, RejectsAnOperatorWithMoreInputsThanAnInstructionHolds) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto reg = g.add_region(fuse::Region{.name = "r", .mesh = m});
  g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1},
      .interval = Rational{1, 4}});  // four instances per period

  fuse::OperatorDAG dag;
  fuse::ValueType vt;
  vt.region = reg;
  vt.rank_set = 0;
  std::vector<fuse::ArcId> ins;
  for (int i = 0; i < 4; ++i) {
    ins.push_back(dag.add_arc(vt, "in" + std::to_string(i)));
  }
  const fuse::ArcId out = dag.add_arc(vt, "out");
  dag.add_node(OpKind::Merge, ins, out, fuse::OpAttrs{}, fuse::ChannelId{},
               "merge4");

  const fuse::ClockSet cs{g};
  const auto r = schedule(g, dag, cs, constant_sizes(8));
  EXPECT_FALSE(r.ok());
  EXPECT_TRUE(has_code(r.diags, "E-SCHED-ARITY")) << r.diags.to_string();
  EXPECT_TRUE(r.plan.instructions().empty());
  // One diagnostic for the node, not one per timed instance.
  EXPECT_EQ(r.diags.size(), 1u) << r.diags.to_string();
}

// A real 5-writer Merge, produced by lower() itself rather than hand-built
// like the test above, must fold into a chain of 2-input Merge nodes and
// then schedule successfully -- the case the hand-built arity test above
// cannot cover, since it never calls lower().
TEST(ExecutionPlan, FiveWriterMergeFoldsToPairwiseMergesAndSchedules) {
  const CouplingGraph g = five_writer_merge(/*lagged_writer=*/-1);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  std::size_t merge_nodes = 0;
  for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
    const fuse::Node& n = lowered.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Merge) continue;
    ++merge_nodes;
    EXPECT_EQ(n.inputs.size(), 2u)
        << "left-fold must chain 2-input Merge nodes, not one N-ary node";
  }
  // Five writers fold as (((w0+w1)+w2)+w3)+w4: four pairwise merges.
  EXPECT_EQ(merge_nodes, 4u);

  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  EXPECT_TRUE(r.ok()) << r.diags.to_string();  // used to fail E-SCHED-ARITY
}

// The merged output's TimeLevel must still reflect the single newest
// contribution when one of several writers is lagged -- and that must hold
// at EVERY node in the fold chain, not just the pairwise Merge that happens
// to consume the lagged writer directly, or the acyclicity prover's
// lag-cutting rule would see the wrong level partway through.
TEST(ExecutionPlan, FiveWriterMergeKeepsNewestTimeLevelThroughTheFoldWhenOneIsLagged) {
  const CouplingGraph g = five_writer_merge(/*lagged_writer=*/2);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  const fuse::TimeLevel newest = fuse::TimeLevel::current();
  std::size_t merge_nodes = 0;
  for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
    const fuse::Node& n = lowered.dag.node(fuse::NodeId{i});
    if (n.kind != OpKind::Merge) continue;
    ++merge_nodes;
    EXPECT_TRUE(lowered.dag.arc(n.output).type.time == newest)
        << "node " << i << " (\"" << n.label << "\") must carry the newest "
           "contribution's time level, not a mid-fold intermediate's";
  }
  EXPECT_EQ(merge_nodes, 4u);
}

// A Merge whose writers are NOT colocated with the target must schedule too,
// and its instruction stream must be a real dependence order -- not just
// "schedule() returned ok", since a wrong-but-lucky order could still pass
// r.ok() if nothing downstream happened to read the misordered arc first.
// Merge must not sort before the Epoch outputs it reads (E-SCHED-ORDER).
TEST(ExecutionPlan, MergeAcrossRanksSchedulesInDependenceOrder) {
  const CouplingGraph g = merge_across_ranks();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  bool saw_transport = false;
  for (std::uint32_t i = 0; i < lowered.dag.node_count(); ++i) {
    if (lowered.dag.node(fuse::NodeId{i}).kind == OpKind::Transport) {
      saw_transport = true;
    }
  }
  ASSERT_TRUE(saw_transport)
      << "fixture must actually exercise a Transport/Epoch pair, or it is "
         "the colocated case FiveWriterMerge* already covers";

  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();  // used to fail E-SCHED-ORDER
  EXPECT_EQ(dependence_violations(lowered.dag, r.plan), 0u);
}

// A Merge combining writers at different intervals
// must schedule, and the stream must be a real dependence order, not an
// accident of E-SCHED-ORDER happening not to fire. Merge's phase tracks the
// latest of its writers' phases at the shared first instant (an
// unconditional BeforeTicks would sort it ahead of W1's target-limited,
// AfterTicks tail), so the plan is a valid order.
TEST(ExecutionPlan, MultirateMergeSchedulesInDependenceOrder) {
  const CouplingGraph g = multirate_merge();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();  // used to fail E-SCHED-ORDER
  EXPECT_EQ(dependence_violations(lowered.dag, r.plan), 0u);
}

// ---------------------------------------------------------------------------
// Iteration regions become explicit loop boundaries (spec Section 2 layer 3)
// ---------------------------------------------------------------------------

// Declaration order must not decide whether a coupling schedules.
//
// Two independent channels on the SAME clock, one of them iterated. Nothing
// couples them, so an order exists that keeps the region's body contiguous.
// The two declaration orders describe the same coupling.
namespace {

/// `region_first`: declare the iterated channel's components and channel
/// before the independent channel's, or after.
CouplingGraph two_channels_one_iterated(bool region_first) {
  CouplingGraph g;
  // Each pair gets its own meshes, as `pair()` does: a channel whose two
  // sides share a region lowers to nothing at all (E-IR-REGION-EMPTY), and a
  // region over no nodes cannot show anything about contiguity.
  auto add_pair = [&](const char* pa, const char* pb, const char* chan) {
    const auto ms = g.add_mesh(fuse::MeshDecl{.name = std::string{pa} + "_m"});
    const auto md = g.add_mesh(fuse::MeshDecl{.name = std::string{pb} + "_m"});
    const auto rs = g.add_region(
        fuse::Region{.name = std::string{pa} + "_r", .mesh = ms});
    const auto rd = g.add_region(
        fuse::Region{.name = std::string{pb} + "_r", .mesh = md});
    const auto ca = g.add_component(ComponentDecl{
        .name = pa, .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
    const auto cb = g.add_component(ComponentDecl{
        .name = pb, .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
    const auto s = g.add_field(ca, fuse::FieldSpec{
        .name = "out", .region = rs, .units = u::kHeatFluxDensity,
        .conservation = fuse::Conservation::Extensive,
        .intent = fuse::Intent::Provides});
    const auto d = g.add_field(cb, fuse::FieldSpec{
        .name = "in", .region = rd, .units = u::kHeatFluxDensity,
        .conservation = fuse::Conservation::Extensive,
        .intent = fuse::Intent::Requires});
    const auto ch = g.add_channel(chan, s, d);
    g.channel(ch).remap(fuse::RemapFlags::Conservative);
    return ch;
  };
  fuse::ChannelId iterated{};
  if (region_first) {
    iterated = add_pair("pa", "pb", "iterated");
    (void)add_pair("qa", "qb", "independent");
  } else {
    (void)add_pair("qa", "qb", "independent");
    iterated = add_pair("pa", "pb", "iterated");
  }
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "outer", .channels = {iterated}, .max_iterations = 4});
  return g;
}

}  // namespace

// A region's members are grouped inside their instant
// (`group_regions_within_instants`), so one assertion is made twice:
// declaration order does not decide whether a coupling schedules.
//
// Keep both spellings. The point is not that one order works; it is that they
// agree, which is the property a user relies on.
TEST(ExecutionPlan, ARegionSchedulesWhicheverOrderItsNeighbourIsDeclaredIn) {
  for (const bool region_first : {false, true}) {
    const CouplingGraph g = two_channels_one_iterated(region_first);
    const auto lowered = lower(g);
    ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
    const fuse::ClockSet cs{g};
    const auto r = schedule(g, lowered.dag, cs, constant_sizes(8));
    ASSERT_TRUE(r.ok()) << "region_first=" << region_first << ": "
                        << r.diags.to_string();
    EXPECT_GT(r.plan.region_count(), 0u) << "region_first=" << region_first;

    // The region's members are one contiguous run with enter/exit stamped on
    // its ends -- what `mark_regions` requires and what the grouping restores.
    std::size_t enters = 0;
    std::size_t exits = 0;
    for (const auto& ins : r.plan.instructions()) {
      if (ins.region_enter >= 0) ++enters;
      if (ins.region_exit >= 0) ++exits;
    }
    EXPECT_EQ(enters, 1u) << "region_first=" << region_first;
    EXPECT_EQ(exits, 1u) << "region_first=" << region_first;
  }
}

TEST(ExecutionPlan, MarksIterationRegionBoundaries) {
  CouplingGraph g = pair(Rational{1, 3}, Rational{1});
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "loop", .channels = {fuse::ChannelId{0}}, .max_iterations = 5});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);

  const fuse::ClockSet cs{g};
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(64));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_EQ(r.plan.region_count(), 1u);

  // Three slabs in the period, so three whole passes over the body.
  std::size_t enters = 0;
  std::size_t exits = 0;
  bool inside = false;
  for (const auto& ins : r.plan.instructions()) {
    if (ins.region_enter >= 0) {
      EXPECT_FALSE(inside) << "nested enter without an intervening exit";
      EXPECT_EQ(ins.region_enter, 0);
      inside = true;
      ++enters;
    }
    if (ins.region_exit >= 0) {
      EXPECT_TRUE(inside) << "exit without a matching enter";
      EXPECT_EQ(ins.region_exit, 0);
      inside = false;
      ++exits;
    }
  }
  EXPECT_FALSE(inside);
  EXPECT_EQ(enters, 3u);
  EXPECT_EQ(exits, 3u);

  // The convergence measure has to be computed inside the loop it controls.
  const auto& is = r.plan.instructions();
  bool residual_inside = false;
  bool in_body = false;
  for (const auto& ins : is) {
    if (ins.region_enter >= 0) in_body = true;
    if (in_body && ins.kind == OpKind::Residual) residual_inside = true;
    if (ins.region_exit >= 0) in_body = false;
  }
  EXPECT_TRUE(residual_inside);
}

TEST(ExecutionPlan, RejectsARegionWhoseBodyIsNotContiguous) {
  // Two one-way channels on different clocks inside one region: their
  // instructions interleave, so no contiguous run is a whole pass over the
  // body and the region cannot become flat enter/exit boundaries.
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r0 = g.add_region(fuse::Region{.name = "r0", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  const auto c = g.add_component(ComponentDecl{
      .name = "c", .ranks = fuse::RankSet{2, 1},
      .interval = Rational{1, 2}});
  const auto d = g.add_component(ComponentDecl{
      .name = "d", .ranks = fuse::RankSet{3, 1},
      .interval = Rational{1, 2}});
  const auto ao = g.add_field(a, fuse::FieldSpec{
      .name = "ao", .region = r0, .intent = fuse::Intent::Provides});
  const auto bi = g.add_field(b, fuse::FieldSpec{
      .name = "bi", .region = r0, .intent = fuse::Intent::Requires});
  const auto co = g.add_field(c, fuse::FieldSpec{
      .name = "co", .region = r0, .intent = fuse::Intent::Provides});
  const auto di = g.add_field(d, fuse::FieldSpec{
      .name = "di", .region = r0, .intent = fuse::Intent::Requires});
  const auto ch0 = g.add_channel("a2b", ao, bi);
  const auto ch1 = g.add_channel("c2d", co, di);
  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "loop", .channels = {ch0, ch1}, .max_iterations = 3});

  const auto r = build(g);
  EXPECT_FALSE(r.ok());
  EXPECT_TRUE(has_code(r.diags, "E-SCHED-REGION")) << r.diags.to_string();
  EXPECT_TRUE(r.plan.instructions().empty());
}

// ---------------------------------------------------------------------------
// The dump is a byte-compared artifact
// ---------------------------------------------------------------------------

namespace {

/// A locale that groups thousands with '.' and uses ',' as the decimal point,
/// i.e. what a host application setting LC_NUMERIC=de_DE does to every
/// default-constructed std::ostringstream in the process.
struct GroupingPunct : std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

/// Restores the global locale even if an assertion throws.
class ScopedGlobalLocale {
 public:
  explicit ScopedGlobalLocale(const std::locale& l) : saved_(std::locale()) {
    std::locale::global(l);
  }
  ScopedGlobalLocale(const ScopedGlobalLocale&) = delete;
  ScopedGlobalLocale& operator=(const ScopedGlobalLocale&) = delete;
  ~ScopedGlobalLocale() { std::locale::global(saved_); }

 private:
  std::locale saved_;
};

}  // namespace

TEST(ExecutionPlan, DumpIsIndependentOfTheGlobalLocale) {
  const CouplingGraph g = pair(Rational{1, 7}, Rational{1});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const fuse::ClockSet cs{g};
  // 4096 elements per region: the coupler total passes 1000, which is where a
  // grouping separator starts to show.
  const auto r = schedule(g, lowered.dag, cs, constant_sizes(4096));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  const std::string classic = r.plan.to_string();
  // 20480, not 16384: the agreement point after the data move carries a
  // buffer of its own until in-place operators land.
  ASSERT_NE(classic.find("coupler_elements=20480"), std::string::npos)
      << classic.substr(0, 200);

  std::string grouped;
  {
    const ScopedGlobalLocale guard(
        std::locale(std::locale::classic(), new GroupingPunct));
    // Sanity: this locale really does change stream output.
    std::ostringstream probe;
    probe << static_cast<std::size_t>(123456) << " " << 0.5;
    ASSERT_EQ(probe.str(), "123.456 0,5");
    grouped = r.plan.to_string();
  }
  EXPECT_EQ(grouped, classic);
}

/// `TickPhase` IS RECOVERABLE FROM THE DUMP, which is the only thing that
/// keeps the dump an honest description of the plan.
///
/// It was briefly not printed, on the argument that `AfterTicks` meant exactly
/// `Sample` and so added nothing to the printed `kind`. That stopped being
/// true when the causality contract (see TickPhase in expand.hpp) began
/// deferring target-limited slab instances too: on 1 against 1/2 the chain at
/// t=1/2 runs after the ticks and the chain at t=1 does not, and every other
/// printed field of the two groups is identical. Without the annotation these
/// two plans -- which deliver to different target steps -- would dump the
/// same, and a byte-compared artifact that cannot distinguish two behaviours
/// is worse than no artifact.
///
/// Byte-exact, like `DumpIsByteExact` below, and for the same reason.
TEST(ExecutionPlan, DumpShowsWhichInstructionsRunAfterTheTicks) {
  const auto r = build(pair(Rational{1}, Rational{1, 2}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const std::string dump = r.plan.to_string();

  // The instruction lines only; the buffer table is DumpIsByteExact's job.
  const auto first = dump.find("  0  t=");
  ASSERT_NE(first, std::string::npos);
  EXPECT_EQ(dump.substr(first),
            // Automatic remap halo: this
            // channel is a cross-rank weighted (non-nodal, non-consistent,
            // {remap=C}) transfer between disjoint placements, so the
            // lowering now emits one `BuildRemapHalo` instruction for it,
            // at the front of the stream (node 0, t=0) -- shifting every
            // other node id by +1 and every instruction index by +1.
            "  0  t=0  BuildRemapHalo  node=0\n"
            "  1  t=0  Source  node=1  out=%0\n"
            "  2  t=1/2  Transport  node=2  in=[%0]  out=%1  {0->1}  "
            "after-ticks\n"
            // The agreement point sits immediately after the move, on both
            // ticks. Every rank that took part in the move reaches it.
            "  3  t=1/2  Epoch  node=3  in=[%1]  out=%2  "
            "{reason=after-transport}  after-ticks\n"
            "  4  t=1/2  Remap  node=4  in=[%2]  out=%3  {remap=C}  "
            "after-ticks\n"
            "  5  t=1/2  Constrain  node=5  in=[%3]  out=%4  after-ticks\n"
            "  6  t=1/2  Diagnose  node=6  in=[%4]  out=%5  after-ticks\n"
            "  7  t=1/2  Sink  node=7  in=[%5]  out=%6  after-ticks\n"
            "  8  t=1  Transport  node=2  in=[%0]  out=%1  {0->1}\n"
            "  9  t=1  Epoch  node=3  in=[%1]  out=%2  "
            "{reason=after-transport}\n"
            "  10  t=1  Remap  node=4  in=[%2]  out=%3  {remap=C}\n"
            "  11  t=1  Constrain  node=5  in=[%3]  out=%4\n"
            "  12  t=1  Diagnose  node=6  in=[%4]  out=%5\n"
            "  13  t=1  Sink  node=7  in=[%5]  out=%6\n");
}

TEST(ExecutionPlan, DumpIsByteExact) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_EQ(r.plan.to_string(),
            // This channel is a cross-rank weighted ({remap=C})
            // transfer between disjoint placements, so `lower()` emits one
            // `BuildRemapHalo` instruction for it (node 0, t=0, no buffer
            // of its own -- OpKind::BuildRemapHalo's own doc comment,
            // op.hpp) ahead of everything else; the buffer table is
            // unchanged, every other node id and instruction index shifts
            // by +1.
            "# fuse-plan v1\n"
            "period=1 lockstep=yes instructions=8 buffers=7 "
            "coupler_elements=320 regions=0\n"
            "buffer 0   %0  elements=64  components=1  region=r0  "
            "owner=component  persistent=no  \"a.f\"\n"
            "buffer 1   %1  elements=64  components=1  region=r0  "
            "owner=coupler  persistent=no  \"c/transport\"\n"
            // The agreement point currently gets a buffer of its own because
            // the plan has no notion yet of an operator whose output aliases
            // its input; it copies rather than allocates nothing. That costs
            // one interface-sized buffer per data move and is fixed when
            // in-place operators land with the runtime, not by removing
            // the agreement point.
            "buffer 2   %2  elements=64  components=1  region=r0  "
            "owner=coupler  persistent=no  \"c/epoch\"\n"
            "buffer 3   %3  elements=64  components=1  region=r1  "
            "owner=coupler  persistent=no  \"c/remap\"\n"
            "buffer 4   %4  elements=64  components=1  region=r1  "
            "owner=coupler  persistent=no  \"c/constrain\"\n"
            "buffer 5   %5  elements=64  components=1  region=r1  "
            "owner=coupler  persistent=no  \"c/diagnose\"\n"
            "buffer 6   %6  elements=64  components=1  region=r1  "
            "owner=component  persistent=no  \"b.f_in\"\n"
            "  0  t=0  BuildRemapHalo  node=0\n"
            "  1  t=0  Source  node=1  out=%0\n"
            "  2  t=1  Transport  node=2  in=[%0]  out=%1  {0->1}\n"
            "  3  t=1  Epoch  node=3  in=[%1]  out=%2  "
            "{reason=after-transport}\n"
            "  4  t=1  Remap  node=4  in=[%2]  out=%3  {remap=C}\n"
            "  5  t=1  Constrain  node=5  in=[%3]  out=%4\n"
            "  6  t=1  Diagnose  node=6  in=[%4]  out=%5\n"
            "  7  t=1  Sink  node=7  in=[%5]  out=%6\n");
}
