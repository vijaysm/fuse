/// S7.2's own graph-level wiring test: "the declaration sets
/// requires_residual_capability (and linearization when a block
/// preconditioner is requested)" (this track's own plan step 2), proved
/// through the SAME `verify_iteration_region()` entry point
/// `Runner::Impl::resolve_regions()` calls at `initialize()`
/// (src/runtime/iteration.cpp), fed the ACTUAL
/// `IterationRegionDecl::requires_residual_capability`/
/// `requires_linearization_capability` a graph built with
/// `IterationBuilder::newton()`/`::accelerate()` produced -- never a
/// hand-set bool standing in for what the builder is supposed to do. Same
/// two-way fixture and `Stub`/`has_code` idioms as
/// test/model/newton_capability_test.cpp (S7.1's own test, which fixed the
/// two booleans directly); duplicated locally for the same reason that
/// file states its own duplication: neither file exports them.
#include <gtest/gtest.h>
#include <fuse/base/units.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/iteration.hpp>
#include <fuse/runtime/component.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

using fuse::AccelConfig;
using fuse::AccelKind;
using fuse::ComponentSet;
using fuse::ConvergenceCriterion;
using fuse::CouplingGraph;
using fuse::derive_checkpoint_set;
using fuse::NewtonConfig;
using fuse::Rational;
using fuse::verify_iteration_region;

namespace {

class Stub final : public fuse::Component {
 public:
  Stub(std::string name, bool residual, bool linearization)
      : name_(std::move(name)), residual_(residual), linearization_(linearization) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&, const CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    std::uint64_t caps = fuse::make_capabilities(/*rollback=*/true,
                                                 fuse::DeterminismLevel::Bitwise, false);
    if (residual_) caps |= fuse::kCapabilityResidual;
    if (linearization_) caps |= fuse::kCapabilityLinearization;
    return caps;
  }

 private:
  std::string name_;
  bool residual_;
  bool linearization_;
};

ComponentSet make_set(bool a_residual, bool b_residual, bool a_linearization = false,
                      bool b_linearization = false) {
  ComponentSet s;
  s.add(std::make_unique<Stub>("a", a_residual, a_linearization));
  s.add(std::make_unique<Stub>("b", b_residual, b_linearization));
  return s;
}

[[nodiscard]] bool has_code(const fuse::Diagnostics& d, const char* code) {
  return std::any_of(d.items().begin(), d.items().end(),
                     [code](const fuse::Diagnostic& x) { return x.code == code; });
}

ConvergenceCriterion two_measures() {
  ConvergenceCriterion c;
  c.add(fuse::Converged::Relative(1e-6, "a_in"));
  c.add(fuse::Converged::Relative(1e-6, "a_out"));
  return c;
}

/// Same two-way coupled pair test/model/newton_capability_test.cpp builds.
/// Returns the graph with its iteration region declared but NOT yet
/// accelerated -- each test below builds its own `IterationBuilder` over
/// the returned graph's region 0 and calls `.accelerate()`/`.newton()`/
/// `.requires_residual()` itself, since that call is the thing under
/// test, not a hand-built struct standing in for it.
CouplingGraph iterated_two_way_graph() {
  CouplingGraph g;
  const auto ma = g.add_mesh(fuse::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(fuse::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(fuse::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(fuse::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(
      fuse::ComponentDecl{.name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto b = g.add_component(
      fuse::ComponentDecl{.name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(
      a, fuse::FieldSpec{.name = "a_out", .region = ra, .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Provides});
  const auto b_in = g.add_field(
      b, fuse::FieldSpec{.name = "b_in", .region = rb, .units = fuse::units::kKelvin,
                        .intent = fuse::Intent::Requires});
  const auto b_out = g.add_field(
      b, fuse::FieldSpec{.name = "b_out", .region = rb, .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Provides});
  const auto a_in = g.add_field(
      a, fuse::FieldSpec{.name = "a_in", .region = ra, .units = fuse::units::kHeatFluxDensity,
                        .intent = fuse::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");

  g.add_iteration_region(fuse::IterationRegionDecl{.name = "fsi", .channels = {fwd, back},
                                                   .max_iterations = 30});
  return g;
}

}  // namespace

// ---------------------------------------------------------------------------
// Selecting Newton through the builder sets requires_residual_capability.
// ---------------------------------------------------------------------------

TEST(NewtonRegion, AccelerateWithNewtonKindSetsRequiresResidualCapability) {
  CouplingGraph g = iterated_two_way_graph();
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0})
      .accelerate(AccelConfig{.kind = AccelKind::Newton});

  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_TRUE(decl.requires_residual_capability);
  EXPECT_FALSE(decl.requires_linearization_capability)
      << "no block-Jacobi preconditioner was requested, so linearization must stay false";
}

TEST(NewtonRegion, NewtonWithBlockJacobiPreconditionerSetsBothCapabilityFlags) {
  CouplingGraph g = iterated_two_way_graph();
  NewtonConfig cfg;
  cfg.use_block_jacobi_preconditioner = true;
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0}).newton(cfg);

  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_EQ(decl.acceleration.kind, AccelKind::Newton);
  EXPECT_TRUE(decl.requires_residual_capability);
  EXPECT_TRUE(decl.requires_linearization_capability);
}

TEST(NewtonRegion, SelectingAnyOtherKindLeavesBothCapabilityFlagsUntouched) {
  CouplingGraph g = iterated_two_way_graph();
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0})
      .accelerate(AccelConfig{.kind = AccelKind::Aitken});

  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_FALSE(decl.requires_residual_capability);
  EXPECT_FALSE(decl.requires_linearization_capability);
}

TEST(NewtonRegion, ExplicitRequiresResidualSetterWorksIndependentlyOfNewton) {
  // S7.1's own fluent setters, added by this track: a region can ask for
  // the residual capability WITHOUT selecting Newton at all.
  CouplingGraph g = iterated_two_way_graph();
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0}).requires_residual(true);

  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  EXPECT_EQ(decl.acceleration.kind, AccelKind::None);
  EXPECT_TRUE(decl.requires_residual_capability);
  EXPECT_FALSE(decl.requires_linearization_capability);
}

// ---------------------------------------------------------------------------
// The same selection, carried through lower()/verify_iteration_region() --
// the ACTUAL entry point Runner::Impl::resolve_regions() calls at
// initialize() -- fires E-NEWTON-CAPABILITY for a component lacking the
// bit, end to end from the builder call to the refusal.
// ---------------------------------------------------------------------------

TEST(NewtonRegion, GraphLevelNewtonSelectionRefusesAComponentLackingResidualCapability) {
  CouplingGraph g = iterated_two_way_graph();
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0})
      .accelerate(AccelConfig{.kind = AccelKind::Newton});

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/false);

  const auto diags =
      verify_iteration_region(g, lowered.dag, region, two_measures(), cs,
                              decl.requires_residual_capability,
                              decl.requires_linearization_capability);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos) << diags.to_string();
}

TEST(NewtonRegion, GraphLevelNewtonSelectionAcceptsWhenEveryMemberHasResidualCapability) {
  CouplingGraph g = iterated_two_way_graph();
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0})
      .accelerate(AccelConfig{.kind = AccelKind::Newton});

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  const ComponentSet cs = make_set(/*a_residual=*/true, /*b_residual=*/true);

  const auto diags =
      verify_iteration_region(g, lowered.dag, region, two_measures(), cs,
                              decl.requires_residual_capability,
                              decl.requires_linearization_capability);
  EXPECT_FALSE(diags.has_errors()) << diags.to_string();
}

TEST(NewtonRegion, GraphLevelNewtonWithBlockJacobiRefusesAComponentLackingLinearization) {
  CouplingGraph g = iterated_two_way_graph();
  NewtonConfig cfg;
  cfg.use_block_jacobi_preconditioner = true;
  fuse::IterationBuilder(g, fuse::IterationRegionIdx{0}).newton(cfg);

  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const auto& region = lowered.dag.region_at(fuse::RegionIdx{0});
  const auto& decl = g.iteration_region(fuse::IterationRegionIdx{0});
  // Both have the residual capability; only 'a' also has linearization.
  const ComponentSet cs =
      make_set(/*a_residual=*/true, /*b_residual=*/true,
               /*a_linearization=*/true, /*b_linearization=*/false);

  const auto diags =
      verify_iteration_region(g, lowered.dag, region, two_measures(), cs,
                              decl.requires_residual_capability,
                              decl.requires_linearization_capability);
  EXPECT_TRUE(diags.has_errors());
  EXPECT_TRUE(has_code(diags, "E-NEWTON-CAPABILITY")) << diags.to_string();
  EXPECT_NE(diags.to_string().find("'b'"), std::string::npos) << diags.to_string();
}
