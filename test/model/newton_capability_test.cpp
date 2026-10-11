/// S7.1's Newton-Krylov seam: the capability-rejection tests, written before
/// any Newton solver exists (this track's own brief). Mirrors test/runtime/
/// checkpoint_test.cpp's own fixtures and idioms -- same two-way graph
/// shape, same Stub/has_code helpers, duplicated locally rather than shared
/// because neither file exports them.
#include <gtest/gtest.h>
#include <fuse/base/units.hpp>
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
using fuse::verify_newton_capability_support;
namespace u = fuse::units;

namespace {

/// Same shape as checkpoint_test.cpp's iterated_two_way(), no relaxation:
/// the Newton capability check does not care about loop-carried buffers, so
/// the simpler (no-Relax) graph is enough here.
CouplingGraph iterated_two_way() {
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

  g.add_iteration_region(fuse::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = 30});
  return g;
}

/// Capability bits are fully configurable, independent of rollback and
/// determinism -- the Newton check and the existing rollback/determinism
/// check are independent gates over the same ComponentSet, and a test that
/// only ever varied them together could not tell the two apart.
class Stub final : public fuse::Component {
 public:
  Stub(std::string name, bool rollback, bool residual, bool linearization,
       fuse::DeterminismLevel determinism = fuse::DeterminismLevel::Bitwise)
      : name_(std::move(name)),
        rollback_(rollback),
        residual_(residual),
        linearization_(linearization),
        determinism_(determinism) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    std::uint64_t caps = fuse::make_capabilities(rollback_, determinism_, false);
    if (residual_) caps |= fuse::kCapabilityResidual;
    if (linearization_) caps |= fuse::kCapabilityLinearization;
    return caps;
  }

 private:
  std::string             name_;
  bool                    rollback_;
  bool                    residual_;
  bool                    linearization_;
  fuse::DeterminismLevel determinism_;
};

ComponentSet make_set(bool a_residual, bool b_residual,
                      bool a_linearization = false,
                      bool b_linearization = false) {
  ComponentSet s;
  s.add(std::make_unique<Stub>("a", /*rollback=*/true, a_residual,
                               a_linearization));
  s.add(std::make_unique<Stub>("b", /*rollback=*/true, b_residual,
                               b_linearization));
  return s;
}

[[nodiscard]] bool has_code(const fuse::Diagnostics& d, const char* code) {
  return std::any_of(
      d.items().begin(), d.items().end(),
      [code](const fuse::Diagnostic& x) { return x.code == code; });
}

ConvergenceCriterion two_measures() {
  ConvergenceCriterion c;
  c.add(fuse::Converged::Relative(1e-6, "a_in"));
  c.add(fuse::Converged::Relative(1e-6, "a_out"));
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// verify_newton_capability_support() directly.
// ---------------------------------------------------------------------------

// Neither flag set (IterationRegionDecl's own default: no Newton solver
// selected for this region) must be a no-op regardless of what the
// components declare -- same rule an un-accelerated region's rollback check
// already follows.
TEST(NewtonCapability, NeitherFlagRequestedIsANoOp) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs = make_set(/*a_residual=*/false, /*b_residual=*/false);
  const auto diags = verify_newton_capability_support(
      g, set, cs, /*requires_residual=*/false, /*requires_linearization=*/false);
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

TEST(NewtonCapability, ComponentWithoutResidualCapabilityIsRejected) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/false);
  const auto diags = verify_newton_capability_support(
      g, set, cs, /*requires_residual=*/true, /*requires_linearization=*/false);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos)
      << diags.to_string();
  // 'a' has the capability: it must not also be flagged.
  EXPECT_EQ(diags.size(), 1u) << diags.to_string();
}

TEST(NewtonCapability, BothComponentsWithResidualCapabilityAreAccepted) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/true);
  const auto diags = verify_newton_capability_support(
      g, set, cs, /*requires_residual=*/true, /*requires_linearization=*/false);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

// Linearization is checked independently of residual: a component can have
// the residual capability and still be rejected for lacking the
// linearization one.
TEST(NewtonCapability, ComponentWithResidualButNotLinearizationIsRejected) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs =
      make_set(/*a_residual=*/true, /*b_residual=*/true,
               /*a_linearization=*/true, /*b_linearization=*/false);
  const auto diags = verify_newton_capability_support(
      g, set, cs, /*requires_residual=*/true, /*requires_linearization=*/true);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos)
      << diags.to_string();
  EXPECT_EQ(diags.size(), 1u) << diags.to_string();
}

TEST(NewtonCapability, BothCapabilitiesPresentIsAccepted) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet cs =
      make_set(/*a_residual=*/true, /*b_residual=*/true,
               /*a_linearization=*/true, /*b_linearization=*/true);
  const auto diags = verify_newton_capability_support(
      g, set, cs, /*requires_residual=*/true, /*requires_linearization=*/true);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

// An unregistered component is already reported by verify_rollback_support()
// over the same checkpoint set; this function must not pile a second,
// redundant diagnostic onto the same unverifiable claim.
TEST(NewtonCapability, UnregisteredComponentAddsNoSecondDiagnostic) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(fuse::RegionIdx{0}));
  const ComponentSet empty;
  const auto diags = verify_newton_capability_support(
      g, set, empty, /*requires_residual=*/true, /*requires_linearization=*/false);
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

// ---------------------------------------------------------------------------
// verify_iteration_region(): the single entry point initialize() calls,
// Newton capability alongside rollback and convergence.
// ---------------------------------------------------------------------------

TEST(NewtonCapability, RegionRequiringResidualRejectsAComponentLackingIt) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/false);
  const auto diags = verify_iteration_region(
      g, lowered.dag, region, two_measures(), cs,
      /*requires_residual_capability=*/true,
      /*requires_linearization_capability=*/false);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
}

TEST(NewtonCapability, RegionRequiringResidualAcceptsWhenBothHaveIt) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/true);
  const auto diags = verify_iteration_region(
      g, lowered.dag, region, two_measures(), cs,
      /*requires_residual_capability=*/true,
      /*requires_linearization_capability=*/false);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
}

// Pre-S7.1 call sites that never pass the two new arguments at all must
// still behave exactly as before: the defaults are false, so nothing new is
// ever checked.
TEST(NewtonCapability, OmittingTheNewtonArgumentsChecksNothingNew) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  // Neither component has the residual capability; with no Newton solver
  // requested this must still pass clean.
  const ComponentSet cs = make_set(/*a_residual=*/false, /*b_residual=*/false);
  const auto diags =
      verify_iteration_region(g, lowered.dag, region, two_measures(), cs);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
}

// The hazard this track's brief names explicitly: a Nondeterministic
// component inside a Newton-solved region must still be refused through the
// EXISTING E-ITER-NONDETERMINISTIC path (verify_rollback_support(), inside
// verify_iteration_region()) -- Newton gains no separate, weaker
// determinism rule of its own.
TEST(NewtonCapability, NondeterministicComponentInNewtonRegionUsesTheExistingRefusal) {
  const CouplingGraph g = iterated_two_way();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  ComponentSet cs;
  cs.add(std::make_unique<Stub>("a", /*rollback=*/true, /*residual=*/true,
                                /*linearization=*/false,
                                fuse::DeterminismLevel::Bitwise));
  cs.add(std::make_unique<Stub>("b", /*rollback=*/true, /*residual=*/true,
                                /*linearization=*/false,
                                fuse::DeterminismLevel::Nondeterministic));
  const auto diags = verify_iteration_region(
      g, lowered.dag, region, two_measures(), cs,
      /*requires_residual_capability=*/true,
      /*requires_linearization_capability=*/false);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-ITER-NONDETERMINISTIC")) << diags.to_string();
  // Both declared the residual capability: Newton's own check must stay
  // silent, so the ONLY error present is the existing determinism one.
  EXPECT_FALSE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
  EXPECT_EQ(diags.size(), 1u) << diags.to_string();
}

// ---------------------------------------------------------------------------
// E-NEWTON-SHAPE: fuse::verify_linearization_shape() (fuse/runtime/
// component.hpp), the token-shape half of the capability-rejection brief.
// A pure function over sizes and a token, so no graph or ComponentSet is
// needed to make it fire.
// ---------------------------------------------------------------------------

namespace {

/// Rows/cols fixed at construction; the three apply_* methods are never
/// exercised by this test and need only type-check.
class FakeToken final : public fuse::LinearizationToken {
 public:
  FakeToken(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols) {}
  fuse::Result<void> apply_jacobian(std::span<const double>,
                                    std::span<double>) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> apply_jacobian_transpose(std::span<const double>,
                                              std::span<double>) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> apply_preconditioner(std::span<const double>,
                                          std::span<double>) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::size_t rows() const override { return rows_; }
  [[nodiscard]] std::size_t cols() const override { return cols_; }
  [[nodiscard]] std::uint64_t signature() const override { return 0; }

 private:
  std::size_t rows_;
  std::size_t cols_;
};

}  // namespace

TEST(NewtonCapability, TokenMatchingTheExpectedShapeIsAccepted) {
  const FakeToken token(/*rows=*/4, /*cols=*/3);
  const auto diags =
      fuse::verify_linearization_shape("c", token, /*expected_rows=*/4,
                                       /*expected_cols=*/3);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
  EXPECT_TRUE(diags.empty()) << diags.to_string();
}

TEST(NewtonCapability, TokenWithWrongRowsIsRejected) {
  const FakeToken token(/*rows=*/5, /*cols=*/3);
  const auto diags =
      fuse::verify_linearization_shape("c", token, /*expected_rows=*/4,
                                       /*expected_cols=*/3);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-SHAPE")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'c'"), std::string::npos)
      << diags.to_string();
}

TEST(NewtonCapability, TokenWithWrongColsIsRejected) {
  const FakeToken token(/*rows=*/4, /*cols=*/9);
  const auto diags =
      fuse::verify_linearization_shape("c", token, /*expected_rows=*/4,
                                       /*expected_cols=*/3);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-SHAPE")) << diags.to_string();
}
