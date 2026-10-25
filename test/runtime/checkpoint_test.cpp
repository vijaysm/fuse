#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/iteration.hpp>
#include <fuse/runtime/component.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

using fuse::ComponentSet;
using fuse::ConvergenceCriterion;
using fuse::CouplingGraph;
using fuse::derive_checkpoint_set;
using fuse::Rational;
using fuse::verify_iteration_region;
using fuse::verify_rollback_support;
namespace u = fuse::units;

namespace {

/// Two-way coupling inside an iteration region, with relaxation on the return
/// channel so there is a genuine loop-carried value.
CouplingGraph iterated_two_way(bool relax_return) {
  CouplingGraph g;
  const auto ma = g.add_mesh(fuse::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(fuse::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(fuse::ComponentDecl{
      .name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, fuse::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(b, fuse::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(b, fuse::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kHeatFluxDensity,
      .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(a, fuse::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kHeatFluxDensity,
      .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");
  if (relax_return) g.channel(back).relax(0.7);

  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = 30});
  return g;
}

}  // namespace

TEST(Checkpoint, RegionCarriesTheRelaxedValue) {
  const CouplingGraph g = iterated_two_way(/*relax_return=*/true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  EXPECT_FALSE(region.carried.empty())
      << "a relaxed channel has a previous-iterate value to carry";
}

TEST(Checkpoint, ComponentsInsideTheRegionNeedCheckpoints) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  EXPECT_EQ(set.components.size(), 2u)
      << "both components advance inside the region";
}

TEST(Checkpoint, ExplanationIsNonEmpty) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
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
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
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
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const auto b = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  EXPECT_EQ(a.coupler_arcs, b.coupler_arcs);
  EXPECT_EQ(a.components, b.components);
}

// ---------------------------------------------------------------------------
// Additional checks. Everything below tests behaviour specified in prose
// elsewhere but not otherwise pinned.
// ---------------------------------------------------------------------------

namespace {

/// Minimal Component: it exists only to answer capabilities(). Nothing
/// under test touches the registry or advances it.
///
/// `determinism` defaults to Bitwise so every existing two-argument call
/// (`Stub("a", true)`) keeps describing exactly the component it described
/// before capabilities() became a bitmask instead of a bool.
class Stub final : public fuse::Component {
 public:
  Stub(std::string name, bool rollback,
       fuse::DeterminismLevel determinism = fuse::DeterminismLevel::Bitwise)
      : name_(std::move(name)), rollback_(rollback), determinism_(determinism) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(rollback_, determinism_, false);
  }

 private:
  std::string             name_;
  bool                    rollback_;
  fuse::DeterminismLevel determinism_;
};

ComponentSet make_set(bool a_rolls_back, bool b_rolls_back) {
  ComponentSet s;
  s.add(std::make_unique<Stub>("a", a_rolls_back));
  s.add(std::make_unique<Stub>("b", b_rolls_back));
  return s;
}

[[nodiscard]] bool has_code(const fuse::Diagnostics& d, const char* code) {
  return std::any_of(
      d.items().begin(), d.items().end(),
      [code](const fuse::Diagnostic& x) { return x.code == code; });
}

ConvergenceCriterion one_measure() {
  ConvergenceCriterion c;
  c.add(fuse::Converged::Relative(1e-6, "a_in"));
  return c;
}

}  // namespace

// The wiring: what lowering records must be what the analysis derives, or the
// runner and the plan dump disagree about what is saved.
TEST(Checkpoint, LoweringRecordsExactlyTheDerivedCouplerArcs) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const auto set = derive_checkpoint_set(g, lowered.dag, region);
  EXPECT_EQ(region.carried, set.coupler_arcs);
}

// Not merely "non-empty": the carried value must be the relaxed one, produced
// by the Relax node. A set that happened to be non-empty for another reason
// would pass the basic non-empty test and still be wrong.
TEST(Checkpoint, TheCarriedValueIsTheRelaxOutput) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  ASSERT_EQ(set.coupler_arcs.size(), 1u);
  const fuse::NodeId producer =
      lowered.dag.arc(set.coupler_arcs.front()).producer;
  ASSERT_TRUE(producer.valid());
  EXPECT_EQ(lowered.dag.node(producer).kind, fuse::OpKind::Relax);
}

// A degenerate region handed straight to the analysis must not reach past the
// end of anything. `assert` would compile out under -DNDEBUG; the bounds
// checks in derive_checkpoint_set() do not.
TEST(Checkpoint, MalformedRegionIsSkippedNotIndexed) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());

  fuse::OperatorDAG::IterationRegion bogus;
  bogus.name = "bogus";
  bogus.body = {fuse::NodeId{999999}, fuse::NodeId{}, fuse::NodeId{0},
                fuse::NodeId{0}};
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
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  EXPECT_NE(set.explanation.find("fsi"), std::string::npos) << set.explanation;
  EXPECT_NE(set.explanation.find("b2a/relax"), std::string::npos)
      << set.explanation;
  EXPECT_NE(set.explanation.find("Relax"), std::string::npos)
      << set.explanation;
}

// ---------------------------------------------------------------------------
// Verification. Spec Section 5.3's distinguishing claim is not that FUSE computes the
// set but that it then checks the component honoured it.
// ---------------------------------------------------------------------------

TEST(Checkpoint, RollbackSupportIsAcceptedWhenBothComponentsHaveIt) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs = make_set(true, true);
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

TEST(Checkpoint, ComponentThatCannotRollBackIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
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
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet empty;
  const auto diags = verify_rollback_support(g, set, empty);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_EQ(diags.size(), 2u) << diags.to_string();
  EXPECT_TRUE(has_code(diags, "E-ITER-NOROLLBACK")) << diags.to_string();
}

// ---------------------------------------------------------------------------
// verify_rollback_support() also checks the determinism
// bit capabilities() now carries -- the refusal src/model/multiscale.cpp's
// review_surrogate_contract() predicts ("it is an error to use [a
// nondeterministic component] inside an iteration region") but cannot itself
// check, since it has no graph and no ComponentSet.
// ---------------------------------------------------------------------------

TEST(Checkpoint, NondeterministicComponentInsideIterationRegionIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  ComponentSet cs;
  cs.add(std::make_unique<Stub>("a", /*rollback=*/true,
                                fuse::DeterminismLevel::Bitwise));
  cs.add(std::make_unique<Stub>("b", /*rollback=*/true,
                                fuse::DeterminismLevel::Nondeterministic));
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-ITER-NONDETERMINISTIC")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos)
      << diags.to_string();
  // 'a' declared Bitwise and rollback: it must not also be flagged.
  EXPECT_EQ(diags.size(), 1u) << diags.to_string();
}

namespace {
/// Rollback, and nothing said about determinism: every component written
/// before the bitmask existed looks like this.
class RollbackOnly final : public fuse::Component {
 public:
  explicit RollbackOnly(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRollback;
  }

 private:
  std::string name_;
};
}  // namespace

// Silence about determinism is not a Bitwise claim (capabilities.hpp). It
// is also not a refusal -- that would reject every pre-bitmask component --
// so the region WARNS, once per undeclared component, and stays runnable.
TEST(Checkpoint, UndeclaredDeterminismInsideIterationRegionWarnsAndRuns) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  ComponentSet cs;
  cs.add(std::make_unique<RollbackOnly>("a"));
  cs.add(std::make_unique<Stub>("b", /*rollback=*/true,
                                fuse::DeterminismLevel::Bitwise));
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(has_code(diags, "W-ITER-DETERMINISM-UNDECLARED")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'a'"), std::string::npos) << diags.to_string();
  // 'b' declared a level: exactly one diagnostic, and it names 'a'.
  EXPECT_EQ(diags.size(), 1u) << diags.to_string();
}

// Agreement: a component that declares determinism explicitly, and agrees
// with what an iteration region needs, passes clean -- the check is a real
// gate, not one that always fires.
TEST(Checkpoint, ExplicitlyBitwiseComponentInsideIterationRegionIsAccepted) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  ComponentSet cs;
  cs.add(std::make_unique<Stub>("a", /*rollback=*/true,
                                fuse::DeterminismLevel::Bitwise));
  cs.add(std::make_unique<Stub>("b", /*rollback=*/true,
                                fuse::DeterminismLevel::RunToRun));
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

// The default (no override at all, hence capabilities() == 0) must decode as
// Bitwise and therefore pass -- proven directly against make_set(), whose
// Stubs never touch determinism.
TEST(Checkpoint, DefaultDeterminismInsideIterationRegionIsAccepted) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs = make_set(true, true);
  const auto diags = verify_rollback_support(g, set, cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
}

// Task 7.1's hazard. all_satisfied() on an empty criterion is true, because
// true is the identity of AND. A region carrying one would stop after a single
// sweep and report success: an explicit scheme wearing an implicit scheme's
// report. convergence.hpp exposes empty() so the region can refuse it.
TEST(Checkpoint, EmptyConvergenceCriterionIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
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
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
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
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const ComponentSet cs = make_set(true, /*b_rolls_back=*/false);
  const auto diags = verify_iteration_region(g, lowered.dag, region,
                                             ConvergenceCriterion{}, cs);
  EXPECT_TRUE(has_code(diags, "E-ITER-NOCONVERGE")) << diags.to_string();
  EXPECT_TRUE(has_code(diags, "E-ITER-NOROLLBACK")) << diags.to_string();
}

TEST(Checkpoint, ARegionThatWouldNeverRunIsRejected) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  fuse::OperatorDAG::IterationRegion region =
      lowered.dag.region_at(fuse::RegionIdx{0});
  region.max_iterations = 0;
  const ComponentSet cs = make_set(true, true);
  const auto diags =
      verify_iteration_region(g, lowered.dag, region, one_measure(), cs);
  EXPECT_TRUE(has_code(diags, "E-ITER-MAXITER")) << diags.to_string();
}

TEST(Checkpoint, NoRegionMeansNoCheckpoints) {
  CouplingGraph g;
  const auto m = g.add_mesh(fuse::MeshDecl{.name = "m"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(fuse::ComponentDecl{.name = "a"});
  const auto b = g.add_component(fuse::ComponentDecl{
      .name = "b", .ranks = fuse::RankSet{1, 1}});
  const auto s = g.add_field(a, fuse::FieldSpec{
      .name = "f", .region = r, .intent = fuse::Intent::Provides});
  const auto t = g.add_field(b, fuse::FieldSpec{
      .name = "f_in", .region = r, .intent = fuse::Intent::Requires});
  g.add_channel("c", s, t);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  EXPECT_EQ(lowered.dag.region_count(), 0u);
}
