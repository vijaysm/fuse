#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/num/convergence.hpp>
#include <soma/runtime/component.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

using soma::ComponentSet;
using soma::ConvergenceCriterion;
using soma::CouplingGraph;
using soma::derive_checkpoint_set;
using soma::Rational;
using soma::verify_iteration_region;
using soma::verify_rollback_support;
namespace u = soma::units;

namespace {

/// Two-way coupling inside an iteration region, with relaxation on the return
/// channel so there is a genuine loop-carried value.
CouplingGraph iterated_two_way(bool relax_return) {
  CouplingGraph g;
  const auto ma = g.add_mesh(soma::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(soma::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(soma::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(soma::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, soma::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto b_in = g.add_field(b, soma::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto b_out = g.add_field(b, soma::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kHeatFluxDensity,
      .intent = soma::Intent::Provides});
  const auto a_in = g.add_field(a, soma::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kHeatFluxDensity,
      .intent = soma::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");
  if (relax_return) g.channel(back).relax(0.7);

  g.add_iteration_region(soma::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = 30});
  return g;
}

}  // namespace

TEST(Checkpoint, RegionCarriesTheRelaxedValue) {
  const CouplingGraph g = iterated_two_way(/*relax_return=*/true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  EXPECT_FALSE(region.carried.empty())
      << "a relaxed channel has a previous-iterate value to carry";
}

TEST(Checkpoint, ComponentsInsideTheRegionNeedCheckpoints) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_EQ(set.components.size(), 2u)
      << "both components advance inside the region";
}

TEST(Checkpoint, ExplanationIsNonEmpty) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_FALSE(set.explanation.empty())
      << "the derived set must be explainable to the user";
}

// The point of deriving rather than asking: pure recomputable values are not
// saved.
TEST(Checkpoint, PureValuesAreNotCheckpointed) {
  const CouplingGraph g = iterated_two_way(/*relax_return=*/false);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  // With no relaxation there is no previous-iterate coupler buffer to save;
  // only component state needs restoring.
  EXPECT_TRUE(set.coupler_arcs.empty())
      << "without relaxation nothing coupler-side is loop-carried";
  EXPECT_FALSE(set.components.empty());
}

TEST(Checkpoint, DerivedSetIsStable) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto a = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  const auto b = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_EQ(a.coupler_arcs, b.coupler_arcs);
  EXPECT_EQ(a.components, b.components);
}

// ---------------------------------------------------------------------------
// Beyond the brief. Everything below tests behaviour the brief specified in
// prose but did not pin, plus the two hazards Tasks 3.4 and 7.1 handed over.
// ---------------------------------------------------------------------------

namespace {

/// Minimal Component: it exists only to answer supports_rollback(). Nothing
/// under test touches the registry or advances it.
class Stub final : public soma::Component {
 public:
  Stub(std::string name, bool rollback)
      : name_(std::move(name)), rollback_(rollback) {}
  [[nodiscard]] std::string name() const override { return name_; }
  soma::Result<void> register_fields(soma::FieldRegistry&,
                                     const CouplingGraph&) override {
    return soma::Result<void>::success();
  }
  soma::Result<void> advance(const soma::StepContext&) override {
    return soma::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return rollback_; }

 private:
  std::string name_;
  bool        rollback_;
};

ComponentSet make_set(bool a_rolls_back, bool b_rolls_back) {
  ComponentSet s;
  s.add(std::make_unique<Stub>("a", a_rolls_back));
  s.add(std::make_unique<Stub>("b", b_rolls_back));
  return s;
}

[[nodiscard]] bool has_code(const soma::Diagnostics& d, const char* code) {
  return std::any_of(
      d.items().begin(), d.items().end(),
      [code](const soma::Diagnostic& x) { return x.code == code; });
}

ConvergenceCriterion one_measure() {
  ConvergenceCriterion c;
  c.add(soma::Converged::Relative(1e-6, "a_in"));
  return c;
}

}  // namespace

// The wiring: what lowering records must be what the analysis derives, or the
// runner and the plan dump disagree about what is saved.
TEST(Checkpoint, LoweringRecordsExactlyTheDerivedCouplerArcs) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  const auto set = derive_checkpoint_set(g, lowered.dag, region);
  EXPECT_EQ(region.carried, set.coupler_arcs);
}

// Not merely "non-empty": the carried value must be the relaxed one, produced
// by the Relax node. A set that happened to be non-empty for another reason
// would pass the brief's first test and still be wrong.
TEST(Checkpoint, TheCarriedValueIsTheRelaxOutput) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  ASSERT_EQ(set.coupler_arcs.size(), 1u);
  const soma::NodeId producer =
      lowered.dag.arc(set.coupler_arcs.front()).producer;
  ASSERT_TRUE(producer.valid());
  EXPECT_EQ(lowered.dag.node(producer).kind, soma::OpKind::Relax);
}

// A degenerate region handed straight to the analysis must not reach past the
// end of anything. `assert` would compile out under -DNDEBUG; the bounds
// checks in derive_checkpoint_set() do not.
TEST(Checkpoint, MalformedRegionIsSkippedNotIndexed) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());

  soma::OperatorDAG::IterationRegion bogus;
  bogus.name = "bogus";
  bogus.body = {soma::NodeId{999999}, soma::NodeId{}, soma::NodeId{0},
                soma::NodeId{0}};
  const auto set = derive_checkpoint_set(g, lowered.dag, bogus);
  EXPECT_TRUE(set.coupler_arcs.empty());
  EXPECT_TRUE(set.components.empty());
  EXPECT_FALSE(set.explanation.empty());
}

// The explanation is the only way a user can disagree with a set they never
// wrote, so it has to name the things it decided about.
TEST(Checkpoint, ExplanationNamesTheBufferAndTheComponents) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_NE(set.explanation.find("fsi"), std::string::npos) << set.explanation;
  EXPECT_NE(set.explanation.find("b2a/relax"), std::string::npos)
      << set.explanation;
  EXPECT_NE(set.explanation.find("Relax"), std::string::npos)
      << set.explanation;
}

// ---------------------------------------------------------------------------
// Verification. Spec §5.3's distinguishing claim is not that SOMA computes the
// set but that it then checks the component honoured it.
// ---------------------------------------------------------------------------

TEST(Checkpoint, RollbackSupportIsAcceptedWhenBothComponentsHaveIt) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  const ComponentSet cs = make_set(true, true);
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

TEST(Checkpoint, ComponentThatCannotRollBackIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  const ComponentSet cs = make_set(true, /*b_rolls_back=*/false);
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-ITER-NOROLLBACK")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos)
      << diags.to_string();
}

// An unverifiable claim is refused, not assumed. A component the analysis
// named but that is not registered cannot be asked whether it rolls back, and
// silently passing it would defeat the point of verifying at all.
TEST(Checkpoint, UnregisteredComponentIsRefusedRatherThanAssumedFine) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  const ComponentSet empty;
  const auto diags = verify_rollback_support(g, set, empty);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_EQ(diags.size(), 2u) << diags.to_string();
  EXPECT_TRUE(has_code(diags, "E-ITER-NOROLLBACK")) << diags.to_string();
}

// Task 7.1's hazard. all_satisfied() on an empty criterion is true, because
// true is the identity of AND. A region carrying one would stop after a single
// sweep and report success: an explicit scheme wearing an implicit scheme's
// report. convergence.hpp exposes empty() so the region can refuse it.
TEST(Checkpoint, EmptyConvergenceCriterionIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  const ComponentSet cs = make_set(true, true);

  // The hazard itself, so the test is not merely asserting a message exists.
  const ConvergenceCriterion empty;
  ASSERT_TRUE(empty.empty());
  ASSERT_TRUE(empty.all_satisfied({}, {}))
      << "an empty criterion really does report converged";

  const auto diags = verify_iteration_region(g, lowered.dag, region, empty, cs);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-ITER-NOCONVERGE")) << diags.to_string();
}

TEST(Checkpoint, ARegionWithAMeasureAndRollbackIsAccepted) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  const ComponentSet cs = make_set(true, true);
  const auto diags =
      verify_iteration_region(g, lowered.dag, region, one_measure(), cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

// The single entry point must not let rollback be checked without convergence,
// nor the other way round: both faults at once must both be reported.
TEST(Checkpoint, BothFaultsAreReportedTogether) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  const ComponentSet cs = make_set(true, /*b_rolls_back=*/false);
  const auto diags = verify_iteration_region(g, lowered.dag, region,
                                             ConvergenceCriterion{}, cs);
  EXPECT_TRUE(has_code(diags, "E-ITER-NOCONVERGE")) << diags.to_string();
  EXPECT_TRUE(has_code(diags, "E-ITER-NOROLLBACK")) << diags.to_string();
}

TEST(Checkpoint, ARegionThatWouldNeverRunIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  soma::OperatorDAG::IterationRegion region =
      lowered.dag.region_at(soma::RegionIdx{0});
  region.max_iterations = 0;
  const ComponentSet cs = make_set(true, true);
  const auto diags =
      verify_iteration_region(g, lowered.dag, region, one_measure(), cs);
  EXPECT_TRUE(has_code(diags, "E-ITER-MAXITER")) << diags.to_string();
}

TEST(Checkpoint, NoRegionMeansNoCheckpoints) {
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(soma::ComponentDecl{.name = "a"});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r, .intent = soma::Intent::Requires});
  g.add_channel("c", s, t);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  EXPECT_EQ(lowered.dag.region_count(), 0u);
}
