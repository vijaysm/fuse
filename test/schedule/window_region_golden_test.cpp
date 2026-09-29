// Byte-exact regression for the "Time windows" subcycling-node design: a
// two-member, TWO-WAY, window-scoped iteration region compiled through
// lower() and schedule() (fuse::CouplingWindowScope, fuse/schedule/
// window.hpp). Both field transfers compile through the ORDINARY
// per-channel rule (Transport/Remap/Epoch/Sink, src/ir/rules_channel.cpp)
// at the window's own clock -- the user's chosen design -- not around it;
// the printed golden IR is what pins that. Mirrors test/ir/golden_ir_test.
// cpp's own byte-for-byte dump-and-compare pattern (that file's own helpers
// are file-local, not exported, so this duplicates the small
// read/write/update utility rather than depend on it). Run with
// FUSE_UPDATE_GOLDEN=1 to regenerate test/golden/window_two_member.{ir,plan}
// after a reviewed, intentional change.

#include <gtest/gtest.h>
#include <fuse/ir/dag.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/schedule/window.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

using fuse::ComponentDecl;
using fuse::Conservation;
using fuse::CouplingGraph;
using fuse::CouplingWindowScope;
using fuse::DtPolicy;
using fuse::ExpandMode;
using fuse::FieldSpec;
using fuse::Intent;
using fuse::IterationRegionDecl;
using fuse::MeshDecl;
using fuse::Rational;
using fuse::Region;
using fuse::RegionId;
using fuse::RankSet;
using fuse::WaveformIteration;
namespace u = fuse::units;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in{path};
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out{path};
  out << text;
}

bool updating() {
  const char* v = std::getenv("FUSE_UPDATE_GOLDEN");
  return v != nullptr && std::string{v} == "1";
}

// Two members, TWO-WAY: `a` (mesh `ma`, ranks {0,1}) and `b` (mesh `mb`,
// ranks {1,1}) exchange a field each way inside one window-scoped
// iteration region. `a2b` is unlagged (checked by check_window_causality()'s
// strict, position-sensitive rule); `b2a` carries `.lag(1)`, read as "the
// previous SWEEP of the SAME window" (checked by that function's relaxed,
// position-independent rule) -- exactly one lagged transfer, mirroring
// examples/fsi_channel/main.cpp's own `wall_force` channel, which is what
// makes a two-way exchange between two empty-input WindowAdvance roots
// (neither side has a real cycle to break; the lag here is about which
// iterate a member sees, not about acyclicity) schedulable at all: see
// src/ir/regions.cpp's own comment on why a window-scoped region's Sink
// nodes must be region MEMBERS (unlike an ordinary region's) for the
// unlagged direction's strict rule to find its own delivery already
// written.
//
// Different meshes and different rank sets on the two members (matching
// test/schedule/multirate_region_test.cpp's own two_clock_region()
// convention) so the printed IR shows a REAL Remap (cross-mesh) and a REAL
// Transport (cross-rank) in both directions' chains, not a degenerate
// same-placement identity.
CouplingGraph two_member_window_graph(
    WaveformIteration mode = WaveformIteration::GaussSeidel,
    bool lag_back = true) {
  CouplingGraph g;
  const auto ma = g.add_mesh(MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(MeshDecl{.name = "mb"});
  const auto ra = g.add_region(Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(Region{.name = "rb", .mesh = mb});

  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = RankSet{0, 1}, .interval = Rational{1, 4}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = RankSet{1, 1}, .interval = Rational{1, 4}});

  const auto a_out = g.add_field(a, FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Provides});
  const auto b_in = g.add_field(b, FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Requires});
  const auto b_out = g.add_field(b, FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Provides});
  const auto a_in = g.add_field(a, FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  if (lag_back) {
    g.channel(back).lag(1).allow_time_level_mismatch(
        "resolved by the a-b coupling window's own sweep, not a real-time "
        "lag");
  }

  CouplingWindowScope scope;
  scope.window_component = "a";
  scope.order = 2;
  scope.policies = {{"a", DtPolicy::Adjustable}, {"b", DtPolicy::Adjustable}};
  // Default GaussSeidel with `back` lagged, not Jacobi: this fixture's
  // whole point (comment above) is exactly one lagged transfer and one
  // unlagged one, checked by check_window_causality()'s two different
  // rules. Jacobi would force BOTH directions loop-carried regardless of
  // `back`'s own `.lag(1)`, which never exercises the strict (unlagged)
  // rule at all. Callers that want the Jacobi/no-lag reference shape pass
  // `mode`/`lag_back` explicitly (below).
  scope.iteration = mode;

  IterationRegionDecl decl;
  decl.name = "window";
  decl.channels = {fwd, back};
  decl.max_iterations = 20;
  decl.window_scope = scope;
  g.add_iteration_region(decl);
  return g;
}

}  // namespace

TEST(WindowRegionGolden, TwoMemberWindowCompilesThroughScheduleAndMatchesTheGolden) {
  const CouplingGraph g = two_member_window_graph();

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  const fuse::ClockSet cs{g};
  const auto r = fuse::schedule(
      g, lowered.dag, cs,
      [](RegionId, int) -> std::size_t { return 64; });
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  const std::string ir_path =
      std::string{FUSE_TEST_GOLDEN_DIR} + "/window_two_member.ir";
  const std::string plan_path =
      std::string{FUSE_TEST_GOLDEN_DIR} + "/window_two_member.plan";

  const std::string actual_ir = dump_text(lowered.dag);
  const std::string actual_plan = r.plan.to_string();

  if (updating()) {
    write_file(ir_path, actual_ir);
    write_file(plan_path, actual_plan);
    GTEST_SKIP() << "golden files updated: " << ir_path << ", " << plan_path;
  }

  const std::string expected_ir = read_file(ir_path);
  ASSERT_FALSE(expected_ir.empty())
      << "missing golden file " << ir_path
      << "\nrun with FUSE_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual_ir, expected_ir)
      << "IR changed. If intentional, run:\n"
      << "  FUSE_UPDATE_GOLDEN=1 ctest --test-dir <build> -R window_region_golden\n"
      << "and review the diff.";

  const std::string expected_plan = read_file(plan_path);
  ASSERT_FALSE(expected_plan.empty())
      << "missing golden file " << plan_path
      << "\nrun with FUSE_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual_plan, expected_plan)
      << "Schedule changed. If intentional, run:\n"
      << "  FUSE_UPDATE_GOLDEN=1 ctest --test-dir <build> -R window_region_golden\n"
      << "and review the diff.";
}

// The WaveformIteration::Jacobi reference shape: the SAME two-way fixture,
// but with NEITHER channel lagged. GaussSeidel mode on this exact graph is
// `WindowScope.GaussSeidelWithNoLagAtAllIsRefused`'s own case (schedule/
// window_scope_test.cpp) -- refused at validate_coupling_window() before
// expand() even runs. Jacobi needs no `.lag()` at all: every transfer in
// the window's own group is forced loop-carried regardless of what the
// user wrote, so this compiles and schedules cleanly through the full
// lower()/schedule() pipeline, proving the mode -- not the channel's own
// `.lag()` -- is what governs a Jacobi window's causality.
// A separate golden pair (window_two_member_jacobi.{ir,plan}), byte-compared
// the same way as the GaussSeidel fixture above, so the two modes'
// compiled IR genuinely differ where they must: BOTH `a2b` and `b2a`
// showing the loop-carried [WindowEnd-1, k] time level here (grep the dump
// below), against exactly ONE of them (`b2a`) in the GaussSeidel golden.
// If `WaveformIteration` were read only by check_window_causality() and
// never by lower() itself, this dump would be byte-identical to a
// same-shaped unlagged GaussSeidel one instead of refusing to compile at
// all (E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG) -- so this file is the reference
// output the design asked for, not the accept/refuse outcome alone.
TEST(WindowRegionGolden, TwoMemberWindowWithNeitherChannelLaggedSchedulesUnderJacobi) {
  const CouplingGraph g = two_member_window_graph(
      /*mode=*/WaveformIteration::Jacobi, /*lag_back=*/false);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  const fuse::ClockSet cs{g};
  const auto r = fuse::schedule(
      g, lowered.dag, cs,
      [](RegionId, int) -> std::size_t { return 64; });
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  const std::string ir_path =
      std::string{FUSE_TEST_GOLDEN_DIR} + "/window_two_member_jacobi.ir";
  const std::string plan_path =
      std::string{FUSE_TEST_GOLDEN_DIR} + "/window_two_member_jacobi.plan";

  const std::string actual_ir = dump_text(lowered.dag);
  const std::string actual_plan = r.plan.to_string();

  // Both directions loop-carried, not just the one channel that happened
  // to carry an explicit `.lag(1)` in the GaussSeidel fixture above.
  EXPECT_NE(actual_ir.find("[WindowEnd-1, k]"), std::string::npos);
  const std::size_t first = actual_ir.find("[WindowEnd-1, k]");
  EXPECT_NE(actual_ir.find("[WindowEnd-1, k]", first + 1), std::string::npos)
      << "expected the shift on BOTH transfers under Jacobi:\n" << actual_ir;

  if (updating()) {
    write_file(ir_path, actual_ir);
    write_file(plan_path, actual_plan);
    GTEST_SKIP() << "golden files updated: " << ir_path << ", " << plan_path;
  }

  const std::string expected_ir = read_file(ir_path);
  ASSERT_FALSE(expected_ir.empty())
      << "missing golden file " << ir_path
      << "\nrun with FUSE_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual_ir, expected_ir)
      << "IR changed. If intentional, run:\n"
      << "  FUSE_UPDATE_GOLDEN=1 ctest --test-dir <build> -R window_region_golden\n"
      << "and review the diff.";

  const std::string expected_plan = read_file(plan_path);
  ASSERT_FALSE(expected_plan.empty())
      << "missing golden file " << plan_path
      << "\nrun with FUSE_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual_plan, expected_plan)
      << "Schedule changed. If intentional, run:\n"
      << "  FUSE_UPDATE_GOLDEN=1 ctest --test-dir <build> -R window_region_golden\n"
      << "and review the diff.";
}

// The same unlagged shape under GaussSeidel, through the PUBLIC
// schedule() entry point rather than window_scope_test.cpp's direct call
// to validate_coupling_window(): confirms schedule() actually reaches that
// gate (src/schedule/scheduler.cpp wires it in before expand()) for a
// window-scoped region compiled from a real graph, not just that the gate
// itself, called in isolation, would refuse this shape.
TEST(WindowRegionGolden, TwoMemberWindowWithNeitherChannelLaggedIsRefusedUnderGaussSeidel) {
  const CouplingGraph g = two_member_window_graph(
      /*mode=*/WaveformIteration::GaussSeidel, /*lag_back=*/false);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  const fuse::ClockSet cs{g};
  const auto r = fuse::schedule(
      g, lowered.dag, cs,
      [](RegionId, int) -> std::size_t { return 64; });
  ASSERT_FALSE(r.ok());
  bool saw_gaussseidel_nolag = false;
  for (const auto& item : r.diags.items()) {
    if (item.code == "E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG") saw_gaussseidel_nolag = true;
  }
  EXPECT_TRUE(saw_gaussseidel_nolag) << r.diags.to_string();
}

// NOTE on what this file does NOT re-verify: two_member_window_graph()
// above has only ONE channel, so it was never a case mark_regions()
// refused in the first place -- a one-channel region's single member
// chain cannot interleave with anything else in a graph that contains
// nothing else. Confirming "the multirate probe still refuses the same
// graph without the window_scope opt-in" (this pass's own requirement)
// therefore means test/schedule/multirate_region_test.cpp's OWN two-
// channel (accumulate + relax) shape, which DOES trigger E-SCHED-REGION at
// ratio != 1 -- unmodified by this pass and re-run directly
// (ctest -R multirate_region) as part of this pass's own verification.
//
// This test instead pins the same invariant at THIS file's own ratio
// (13:7, matching the golden fixture above) with a genuine two-way,
// two-channel region -- accumulate forward, relax backward, exactly
// multirate_region_test.cpp's own two_clock_region() shape -- confirming
// it is STILL refused today without a window_scope opt-in, the same
// belt-and-braces re-check that file's own RatioTwoIsRefusedByThe
// SchedulerToday/RatioFourIsRefusedByTheSchedulerToday give at their own
// ratios.
TEST(WindowRegionGolden, TheTwoChannelMultirateShapeAtThisRatioIsStillRefused) {
  CouplingGraph g2;
  const auto mc = g2.add_mesh(MeshDecl{.name = "mc"});
  const auto mf = g2.add_mesh(MeshDecl{.name = "mf"});
  const auto rc = g2.add_region(Region{.name = "rc", .mesh = mc});
  const auto rf = g2.add_region(Region{.name = "rf", .mesh = mf});
  const auto coarse = g2.add_component(ComponentDecl{
      .name = "coarse", .ranks = RankSet{0, 1}, .interval = Rational{13}});
  const auto fine = g2.add_component(ComponentDecl{
      .name = "fine", .ranks = RankSet{1, 1}, .interval = Rational{7}});
  const auto fine_out = g2.add_field(fine, FieldSpec{
      .name = "fine_out", .region = rf, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Provides});
  const auto coarse_in = g2.add_field(coarse, FieldSpec{
      .name = "coarse_in", .region = rc, .units = u::kKelvin,
      .conservation = Conservation::Extensive,
      .intent = Intent::Requires});
  const auto coarse_out = g2.add_field(coarse, FieldSpec{
      .name = "coarse_out", .region = rc, .units = u::kKelvin,
      .intent = Intent::Provides});
  const auto fine_in = g2.add_field(fine, FieldSpec{
      .name = "fine_in", .region = rf, .units = u::kKelvin,
      .intent = Intent::Requires});
  const auto fwd = g2.add_channel("fine2coarse", fine_out, coarse_in);
  g2.channel(fwd).accumulate(fuse::AccumOp::Mean);
  const auto back = g2.add_channel("coarse2fine", coarse_out, fine_in);
  g2.channel(back).relax(0.7);

  IterationRegionDecl decl;
  decl.name = "multirate";
  decl.channels = {fwd, back};
  decl.max_iterations = 20;
  // window_scope left unset.
  g2.add_iteration_region(decl);

  const auto lowered = lower(g2);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet cs{g2};
  const auto r = fuse::schedule(
      g2, lowered.dag, cs,
      [](RegionId, int) -> std::size_t { return 64; });
  EXPECT_FALSE(r.ok());
  bool saw_region_error = false;
  for (const auto& item : r.diags.items()) {
    if (item.code == "E-SCHED-REGION") saw_region_error = true;
  }
  EXPECT_TRUE(saw_region_error) << r.diags.to_string();
}

// Pins schedule()'s own CALL to detail::check_window_causality() (Unit P),
// which test/schedule/window_causality_test.cpp's own tests -- by that
// file's own header comment -- deliberately do NOT cover, since they feed
// a hand-built instruction stream straight to the function and never go
// through lower()/expand()/schedule() at all. Now that this pass compiles
// a real window-scoped region (build_window_regions(), src/ir/
// window_lowering.cpp), that gap can be closed: this DAG is ALSO hand-
// built (bypassing lower(), matching window_causality_test.cpp's own
// convention, since forcing an invalid window_import out of the real
// lowering rule would fight its own correct-by-construction ordering
// instead of directly pinning the checked call), but is run through the
// real, public fuse::schedule() -- exercising expand()'s window-clock
// ticking, check_dependence_order(), and check_window_causality() as
// schedule() actually wires them together, not the function in isolation.
//
// The window_import here is backwards on purpose: node 0 (which
// expand()/group_regions_within_instants() run FIRST at every window
// instance, absent any real edge to say otherwise -- the same NodeId-
// order argument build_window_regions()'s own doc comment relies on)
// imports node 1's output, which is written only AFTER node 0 runs. A
// mutation removing schedule()'s call to check_window_causality() would
// let this exact graph -- which check_dependence_order() cannot catch,
// since neither node has a real input edge at all -- schedule cleanly
// where it must be refused.
TEST(WindowRegionGolden, SchedulerCallsWindowCausalityOnARealCompiledWindow) {
  CouplingGraph g = two_member_window_graph();

  fuse::OperatorDAG dag;
  const fuse::ChannelId origin{0};  // the only channel in the fixture above

  // A default-constructed ValueType has no region and an unplaced rank_set,
  // which plan_buffers() refuses (E-SCHED-BUFFERS) before schedule() ever
  // reaches check_window_causality() -- so these arcs need a real type,
  // matching what build_window_regions() itself would produce for the
  // channel's own source field.
  const fuse::Channel& ch0 = g.channel_at(origin);
  const fuse::ValueType vt{
      .kind = fuse::ValueKind::Instant,
      .region = g.field(ch0.source).region,
      .components = g.field(ch0.source).components,
      .units = g.field(ch0.source).units,
      .conservation = g.field(ch0.source).conservation,
      .time = fuse::TimeLevel::current(),
      .rank_set = static_cast<int>(g.owner_of(ch0.source).value())};

  const fuse::ArcId out0 = dag.add_arc(vt, "member0.window");
  const fuse::ArcId out1 = dag.add_arc(vt, "member1.window");

  fuse::OpAttrs attrs0;
  attrs0.window_import = out1;  // WRONG: out1 is not written until node 1 runs
  dag.add_node(fuse::OpKind::WindowAdvance, {}, out0, attrs0, origin,
              "member0.window");
  dag.add_node(fuse::OpKind::WindowAdvance, {}, out1, fuse::OpAttrs{}, origin,
              "member1.window");

  const fuse::ClockSet cs{g};
  const auto r = fuse::schedule(
      g, dag, cs, [](RegionId, int) -> std::size_t { return 64; });
  ASSERT_FALSE(r.ok());
  bool saw_causality_error = false;
  for (const auto& item : r.diags.items()) {
    if (item.code == "E-SCHED-WINDOW-CAUSALITY") saw_causality_error = true;
  }
  EXPECT_TRUE(saw_causality_error) << r.diags.to_string();
}
