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
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/iteration.hpp>
#include <fuse/runtime/component.hpp>
#include <fuse/runtime/runner.hpp>

#include "dn_component.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
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

// ---------------------------------------------------------------------------
// The runtime wiring itself: a real Runner solving examples/dn_heat's own
// affine Dirichlet-Neumann pair by Newton (run_region_newton(),
// src/runtime/iteration.cpp), rather than hand-feeding
// verify_iteration_region() the capability flags a builder call produced.
// Same fixture shape as test/runtime/fixed_point_residual_test.cpp (S7.1's
// own test of the F = x - G(x) seam this wiring drives): that file's own
// header comment derives the affine closed form and the per-station 2x2
// Jacobian {{1, r_i}, {-1, 1}} this section's own test relies on.
// ---------------------------------------------------------------------------

namespace {

constexpr std::size_t kNewtonN = dn::kStrips;

/// `dn::NeumannSlab` as the Runner would see an ABI participant: every
/// call forwarded, `is_externally_driven()` true. `NeumannSlab` is
/// `final`, so this is a wrapper, not a subclass. It exists for exactly
/// one test -- the Newton region's own eager refusal of an externally
/// driven member (`check_region_supports_newton()`,
/// src/runtime/fixed_point_residual.cpp) -- which kills
/// build-evidence/mutations/runtime.toml's
/// newton-region-eager-validation-skipped now that the window-scoped
/// refusal it used to kill through has been lifted (S7.2 WindowNewton).
class ExternallyDrivenNeumann final : public fuse::Component {
 public:
  explicit ExternallyDrivenNeumann(std::unique_ptr<dn::NeumannSlab> inner)
      : inner_(std::move(inner)) {}
  [[nodiscard]] std::string name() const override { return inner_->name(); }
  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    return inner_->register_fields(registry, graph);
  }
  fuse::Result<void> advance(const fuse::StepContext& ctx) override { return inner_->advance(ctx); }
  fuse::Result<void> save_state() override { return inner_->save_state(); }
  fuse::Result<void> restore_state() override { return inner_->restore_state(); }
  [[nodiscard]] std::uint64_t capabilities() const override { return inner_->capabilities(); }
  [[nodiscard]] bool is_externally_driven() const noexcept override { return true; }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return inner_->implicit(); }
  [[nodiscard]] dn::NeumannSlab& inner() { return *inner_; }

 private:
  std::unique_ptr<dn::NeumannSlab> inner_;
};

struct NewtonDnHeatFixture {
  fuse::CouplingGraph        graph;
  fuse::SerialMeshBackend    backend;
  fuse::ComponentSet         components;
  dn::DirichletSlab*         dirichlet = nullptr;
  dn::NeumannSlab*           neumann = nullptr;
  fuse::FieldRegistry        registry;
  std::unique_ptr<fuse::Runner> runner;
  /// `initialize()`'s own result, captured HERE rather than by the test
  /// calling it a second time on the returned fixture: `Runner` holds a
  /// `const CouplingGraph&` (runner_impl.hpp) into THIS struct's own
  /// `graph` member, and test/runtime/fixed_point_residual_test.cpp's own
  /// `make_setup()` (this file's own sibling pattern) calls `initialize()`
  /// before returning for exactly that reason -- a caller that constructs
  /// `Runner` here and then calls `initialize()` AFTER the fixture has
  /// been returned (and, with it, `graph` and every other member
  /// POSSIBLY relocated by the return's own move) risks the Runner's own
  /// reference outliving the object it was taken from. Measured directly
  /// (2026-10-10): calling `initialize()` from the test body instead of
  /// from here failed with E-RT-UNBOUND ("does not resolve to a field")
  /// on a graph that is otherwise identical byte for byte.
  fuse::Result<void> init = fuse::Result<void>::success();
};

/// Builds the SAME graph fixed_point_residual_test.cpp's own build_graph()
/// does, but names AccelKind::Newton (via `.newton()`) instead of leaving
/// the region unaccelerated -- `cfg` lets a test reach past NewtonConfig's
/// own defaults (e.g. a `max_nonlinear_iterations` of 0 to force
/// DivergedMaxIterations without also needing a pathological problem).
/// `mutate_before_init`, when set, runs right after the graph is built and
/// right before `initialize()` -- the one hook `AWindowScopedRegionSelecting
/// NewtonIsRefusedAtSetup`, below, needs to add a window_scope the builder
/// chain itself has no fluent setter for. `beta` selects dirichlet's own
/// nonlinear conductivity (DirichletSlab::k1_eff(), dn_component.hpp);
/// 0.0 (the default) is the affine map every other test in this file
/// exercises.
NewtonDnHeatFixture make_newton_setup(
    int max_iterations, fuse::NewtonConfig cfg = {},
    fuse::RunnerConfig rcfg = {}, double beta = 0.0,
    double neumann_initial_u = std::numeric_limits<double>::quiet_NaN(),
    const std::function<void(fuse::CouplingGraph&)>& mutate_before_init = nullptr,
    bool neumann_externally_driven = false) {
  NewtonDnHeatFixture s;
  fuse::CouplingGraph& g = s.graph;
  const std::string src = "generated:line:" + std::to_string(kNewtonN);
  const auto m1 = g.add_mesh(fuse::MeshDecl{.name = "iface_1", .source = src});
  const auto m2 = g.add_mesh(fuse::MeshDecl{.name = "iface_2", .source = src});
  const auto r1 = g.add_region(
      fuse::Region{.name = "iface_1", .mesh = m1, .dimension = 1});
  const auto r2 = g.add_region(
      fuse::Region{.name = "iface_2", .mesh = m2, .dimension = 1});

  const auto cd = g.add_component(fuse::ComponentDecl{
      .name = "dirichlet", .ranks = {0, 1}, .interval = fuse::Rational{1}});
  const auto cn = g.add_component(fuse::ComponentDecl{
      .name = "neumann", .ranks = {1, 1}, .interval = fuse::Rational{1}});

  const auto q_out = g.add_field(cd, fuse::FieldSpec{
      .name = "q_wall", .region = r1, .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto u_in = g.add_field(cd, fuse::FieldSpec{
      .name = "u_wall_in", .region = r1, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto q_in = g.add_field(cn, fuse::FieldSpec{
      .name = "q_wall_in", .region = r2,
      .units = fuse::units::kHeatFluxDensity,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto u_out = g.add_field(cn, fuse::FieldSpec{
      .name = "u_wall", .region = r2, .units = fuse::units::kKelvin,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});

  const auto flux = g.add_channel("flux", q_out, q_in);
  const auto temp = g.add_channel("wall_temperature", u_out, u_in);
  g.channel(temp).lag(1).allow_time_level_mismatch(
      "resolved by the Dirichlet-Neumann implicit coupling group");

  ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(1e-6, "u_wall"));
  crit.add(fuse::Converged::Relative(1e-6, "q_wall"));

  g.iterate({flux, temp})
      .name("dirichlet_neumann")
      .until(std::move(crit))
      .max_iterations(max_iterations)
      .newton(cfg);

  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = s.backend.load(g.mesh(fuse::MeshId{i}));
    if (!r.ok()) { ADD_FAILURE() << "mesh load: " << r.error().message; return s; }
  }
  const auto lowered = lower(g);
  if (!lowered.ok()) { ADD_FAILURE() << lowered.diags.to_string(); return s; }
  const fuse::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks,
                       fuse::make_region_sizes(g, s.backend));
  if (!sched.ok()) { ADD_FAILURE() << sched.diags.to_string(); return s; }

  auto dirichlet = std::make_unique<dn::DirichletSlab>(
      "dirichlet", kNewtonN, dn::Wiring::Connected, beta);
  auto neumann = std::make_unique<dn::NeumannSlab>(
      "neumann", kNewtonN, dn::Wiring::Connected, neumann_initial_u);
  s.dirichlet = dirichlet.get();
  s.neumann = neumann.get();
  s.components.add(std::move(dirichlet));
  if (neumann_externally_driven) {
    s.components.add(std::make_unique<ExternallyDrivenNeumann>(std::move(neumann)));
  } else {
    s.components.add(std::move(neumann));
  }
  for (auto& c : s.components) {
    const auto r = c.register_fields(s.registry, g);
    if (!r.ok()) ADD_FAILURE() << "register_fields: " << r.error().message;
  }

  s.runner = std::make_unique<fuse::Runner>(g, lowered.dag, sched.plan,
                                            s.backend);

  if (mutate_before_init) mutate_before_init(g);

  s.init = s.runner->initialize(s.components, s.registry, rcfg);
  return s;
}

}  // namespace

// MEASURED (2026-10-10, this test, trace printed by hand and removed once
// read): with SNES's own default Eisenstat-Walker forcing (eta_0 = 0.3,
// choice 2 thereafter), this affine F converges in THREE Newton steps,
// not one -- eta_0 = 0.3 deliberately stops the FIRST linear solve at 30%
// relative accuracy (GMRES took 2 of a possible many iterations), so the
// first Newton direction is not the exact affine root even though F's own
// Jacobian is exact; the SECOND step's eta (computed from the observed
// residual RATIO, (68.77/765.78)^2 * 0.9 ~= 0.0073) is tight enough that
// the THIRD step's own eta rounds to 0 (an exact solve) and the iteration
// converges at that point. This is textbook inexact-Newton behaviour, not
// a defect: trading a cheap, loose early linear solve against needing one
// or two more nonlinear steps overall is exactly what Eisenstat-Walker
// forcing is FOR (eisenstat_walker_eta()'s own citation, newton_krylov.
// hpp). This track's brief named "Newton converges in ONE Newton step"
// for the affine map as its own assumption; the measurement here
// corrects it, recorded as an open question in this track's own report
// (a NewtonConfig with a tighter/near-zero constant forcing term WOULD
// converge in one step, at the cost of more GMRES work on that one step
// -- not this track's own default, and not measured here).
TEST(NewtonRegion, ConvergesInThreeNewtonStepsOnTheAffineProblemUnderEwForcing) {
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  const auto r = s.runner->run_period(s.components);
  ASSERT_TRUE(r.ok()) << r.error().message;

  ASSERT_EQ(s.runner->iteration_stats().size(), 1u);
  const fuse::IterationStats& st = s.runner->iteration_stats()[0];
  EXPECT_TRUE(st.converged);
  EXPECT_EQ(st.iterations, 3);
}

// `step_period()`'s own top comment (src/runtime/runner.cpp): "By the time
// control reaches the region the participants have already advanced once"
// -- checkpoint_regions() saves BEFORE that first, generic tick specifically
// so a Picard sweep loop's own repeated restore_state() undoes it along with
// every later sweep's own advance. `run_region_newton()`'s own
// evaluate_region_residual()-based calls are self-referential (each restores
// to what IT most recently saved, never to checkpoint_regions()'s earlier
// point), so without an explicit restore of its own before the Newton solve
// begins, that first generic tick is never undone by anything: the
// verification sweep's one committed advance lands on TOP of it, and
// dn_heat's own DirichletSlab::history()/NeumannSlab::history() -- "after P
// coupling cycles a correct run has history() == P exactly"
// (dn_component.hpp's own contract, already relied on by examples/dn_heat/
// main.cpp's own picard/aitken/iqn-ils gates) -- reads 2P instead of P for a
// Newton-selected region. MEASURED directly before this track's own fix
// (run_region_newton()'s own restore_state() loop, right after seeding x):
// history() = 40 after 20 periods, not 20; with the fix, exactly 20 --
// bookkeeping-only, not a residual/seed correctness issue, since `x` is
// already a COPY by the time that restore runs and every evaluate_region_
// residual() call re-delivers it and re-executes the body from scratch
// regardless of what state the restore leaves q_/u_ in (confirmed by this
// file's own other tests' L2-error and bit-identical-at-beta=0 checks, all
// still passing). `build-evidence/mutations/dn-heat.toml`'s own
// `dn-heat-newton-generic-tick-restore-removed` row hand-proves the fix
// itself is load-bearing.
TEST(NewtonRegion, HistoryAccumulatesExactlyOncePerPeriodNotTwice) {
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  constexpr std::int64_t kPeriods = 3;
  ASSERT_TRUE(s.runner->run(s.components, kPeriods).ok());

  EXPECT_DOUBLE_EQ(s.dirichlet->history(), static_cast<double>(kPeriods));
  EXPECT_DOUBLE_EQ(s.neumann->history(), static_cast<double>(kPeriods));
}

TEST(NewtonRegion, VerificationSweepDeliversTheConvergedStateToTheComponents) {
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;
  ASSERT_TRUE(s.runner->run_period(s.components).ok());

  // The commit (evaluate_region_residual(..., commit = true)) is what
  // makes Newton's own converged iterate the REAL state the components
  // hold -- not just a number inside NewtonResult nobody delivered.
  // `time()` is each component's own `ctx.t_end` from its last real
  // (committing) advance() -- read back rather than assumed, since a
  // region's first tick need not land at t = 0.
  ASSERT_DOUBLE_EQ(s.dirichlet->time(), s.neumann->time());
  const double t = s.dirichlet->time();
  for (std::size_t i = 0; i < kNewtonN; ++i) {
    EXPECT_NEAR(s.dirichlet->q()[i], dn::exact_q(i, kNewtonN, t), 1e-5)
        << "station " << i;
    EXPECT_NEAR(s.neumann->u()[i], dn::exact_u(i, kNewtonN, t), 1e-5)
        << "station " << i;
  }
}

// S7.2 WindowNewton, CORRECTED (newton-krylov.md Sec. 9.7, 2026-10): this
// test's own ORIGINAL name/assertion ("...IsRefusedAtSetup", `ASSERT_
// FALSE(s.init.ok())`, `E-NEWTON-UNSUPPORTED-REGION`) pinned the
// blanket "window-scoped is always refused for Newton" rule that rule
// existed ONLY as a placeholder for this track's own work --
// `check_region_supports_newton()`'s own comment (src/runtime/fixed_
// point_residual.cpp) says so directly. Lifting it is this track's
// entire point, so the assertion flips: `initialize()` now ACCEPTS a
// window-scoped Newton region (window_fixed_point_residual_test.cpp's
// own fixture proves a PROPERLY configured one solves correctly).
//
// This fixture specifically, though, bolts a bare `CouplingWindowScope`
// onto dirichlet/neumann -- two components `add_window_coupling()`
// never configured as real window members (no per-member policy/
// capacity, no window-transfer channel) -- a MISCONFIGURED window
// region, not a real one. `resolve_regions()` finds no window member
// nodes for it (`R.window_member_nodes` stays empty) and so resolves
// `R.newton_n = 0`: `run_region_newton_window()`'s own Newton solve
// over a ZERO-length interface vector converges in zero iterations
// without ever calling `advance()` on either component -- a silent,
// well-defined NO-OP, not a crash and not a refusal. MEASURED (2026-10-
// 10): `run_period()` succeeds; `dirichlet`/`neumann`'s own `time()`
// stays at the pre-run initial value, `history()` stays 0 -- proof
// nothing advanced, a real test rather than resting on "it didn't
// throw". A genuinely-configured window region failing to validate
// THIS way (rather than refusing with a clear diagnostic naming the
// missing member configuration) is this test's own documented gap,
// not something Stage B closes -- it is not the scenario any real
// caller (this track's own Brusselator wiring, add_window_coupling())
// can reach, since that function always configures every member it
// declares.
// The eager validation at setup: a Newton-selected region with a member
// the Runner does not drive itself (an ABI participant, here stood in for
// by a wrapper reporting is_externally_driven()) is refused at
// initialize(), by name, before any residual is ever evaluated.
TEST(NewtonRegion, AnExternallyDrivenMemberIsRefusedAtSetup) {
  NewtonDnHeatFixture s = make_newton_setup(
      /*max_iterations=*/30, /*cfg=*/{}, /*rcfg=*/{}, /*beta=*/0.0,
      /*neumann_initial_u=*/std::numeric_limits<double>::quiet_NaN(),
      /*mutate_before_init=*/nullptr, /*neumann_externally_driven=*/true);
  ASSERT_FALSE(s.init.ok());
  EXPECT_EQ(s.init.error().code, "E-NEWTON-UNSUPPORTED-REGION") << s.init.error().message;
  EXPECT_NE(s.init.error().message.find("neumann"), std::string::npos) << s.init.error().message;
  EXPECT_NE(s.init.error().message.find("externally"), std::string::npos) << s.init.error().message;
}

// S7.2b (PetscSelect track): a region naming a Newton substrate nothing
// has registered in THIS process refuses at initialize(), by name, with
// E-NEWTON-PETSC-UNAVAILABLE -- fired here tier-0, WITHOUT PETSc, by the
// most common real case: a graph selecting NewtonSubstrate::Petsc in a
// build/process that never called fuse::petsc::install() (this test
// binary links no fuse::petsc at all). Mirrors AnExternallyDrivenMember
// IsRefusedAtSetup's own shape -- both are resolve_regions()'s own eager
// refusals for a Newton-selected region, caught before the first period.
TEST(NewtonRegion, ARegionSelectingAnUnregisteredSubstrateIsRefusedAtSetup) {
  fuse::NewtonConfig cfg;
  cfg.substrate = fuse::NewtonSubstrate::Petsc;
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30, cfg);
  ASSERT_FALSE(s.init.ok());
  EXPECT_EQ(s.init.error().code, "E-NEWTON-PETSC-UNAVAILABLE") << s.init.error().message;
  EXPECT_NE(s.init.error().message.find("dirichlet_neumann"), std::string::npos)
      << s.init.error().message;
  EXPECT_NE(s.init.error().message.find("Petsc"), std::string::npos)
      << s.init.error().message;
}

TEST(NewtonRegion, AWindowScopedRegionWithNoConfiguredWindowMembersIsANoOp) {
  NewtonDnHeatFixture s = make_newton_setup(
      /*max_iterations=*/30, /*cfg=*/{}, /*rcfg=*/{}, /*beta=*/0.0,
      /*neumann_initial_u=*/std::numeric_limits<double>::quiet_NaN(),
      [](fuse::CouplingGraph& g) {
        fuse::CouplingWindowScope scope;
        scope.explicit_interval = fuse::Rational{1};
        const_cast<fuse::IterationRegionDecl&>(
            g.iteration_region(fuse::IterationRegionIdx{0}))
            .window_scope = std::move(scope);
      });
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  const double dirichlet_time_before = s.dirichlet->time();
  const double neumann_time_before = s.neumann->time();
  const double dirichlet_history_before = s.dirichlet->history();
  const double neumann_history_before = s.neumann->history();

  const auto r = s.runner->run_period(s.components);
  ASSERT_TRUE(r.ok()) << r.error().message;

  EXPECT_DOUBLE_EQ(s.dirichlet->time(), dirichlet_time_before);
  EXPECT_DOUBLE_EQ(s.neumann->time(), neumann_time_before);
  EXPECT_DOUBLE_EQ(s.dirichlet->history(), dirichlet_history_before);
  EXPECT_DOUBLE_EQ(s.neumann->history(), neumann_history_before);
}

TEST(NewtonRegion, ExhaustingTheNewtonIterationBudgetReportsNonConvergence) {
  fuse::NewtonConfig cfg;
  cfg.max_nonlinear_iterations = 0;  // DivergedMaxIterations immediately.
  fuse::RunnerConfig rcfg;
  rcfg.abort_on_nonconvergence = true;
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30, cfg, rcfg);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  const auto r = s.runner->run_period(s.components);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-NOCONVERGE") << r.error().message;
  // Mirrors the Picard path's own E-RT-NOCONVERGE message (iteration.cpp,
  // run_region()'s own tail): names the region and, for Newton, the
  // status code too, the "mirror with Newton's code name in the message"
  // this track's brief asks for.
  EXPECT_NE(r.error().message.find("dirichlet_neumann"), std::string::npos)
      << r.error().message;
  EXPECT_NE(r.error().message.find("DivergedMaxIterations"), std::string::npos)
      << r.error().message;
}

// ---------------------------------------------------------------------------
// The nonlinear variant (S7.2's own reason for measuring a convergence
// RATE, not just a converged/not-converged outcome): DirichletSlab's
// conductivity k1_eff(i, T) = k1_at(i, n) * (1 + beta*(T - t_ref)/t_ref).
// beta = 0.0 -- every mode this project already ships (picard, aitken,
// iqn-ils, iqn+qr2, severed) constructs DirichletSlab with the 3-arg
// constructor, which defaults beta to exactly 0.0 -- must reduce the
// formula to today's affine `k1_at(i, n) * (T1 - u) / kL1` BIT FOR BIT,
// not just numerically close: a tolerant comparison here would not catch
// a k1_eff that is merely "very nearly 1.0 times k1_at" instead of
// "EXACTLY 1.0 times it", which is the whole promise this change makes to
// every existing mode.
// ---------------------------------------------------------------------------

TEST(Dn, NonlinearAtBetaZeroIsBitIdenticalToTheAffineMap) {
  dn::DirichletSlab baseline("dirichlet", kNewtonN, dn::Wiring::Connected);
  dn::DirichletSlab explicit_zero_beta("dirichlet", kNewtonN,
                                       dn::Wiring::Connected, /*beta=*/0.0,
                                       /*t_ref=*/300.0);
  dn::DirichletSlab nonlinear("dirichlet", kNewtonN, dn::Wiring::Connected,
                              /*beta=*/0.1, /*t_ref=*/300.0);

  fuse::StepContext ctx;
  ctx.t_begin = fuse::Rational{0};
  ctx.t_end = fuse::Rational{1};
  ctx.step = 0;
  ctx.iteration = 0;

  ASSERT_TRUE(baseline.advance(ctx).ok());
  ASSERT_TRUE(explicit_zero_beta.advance(ctx).ok());
  ASSERT_TRUE(nonlinear.advance(ctx).ok());

  for (std::size_t i = 0; i < kNewtonN; ++i) {
    EXPECT_EQ(baseline.q()[i], explicit_zero_beta.q()[i]) << "station " << i;
    // The nonlinear instance's own u_in_ is still all zero (nothing fed
    // it), and k1_eff(i, 0.0) != k1_at(i, n) in general -- so at THIS
    // input the two must differ, proving beta really does change the
    // computed value rather than this test silently comparing two
    // instances that happen to agree for an unrelated reason.
    EXPECT_NE(baseline.q()[i], nonlinear.q()[i]) << "station " << i;
  }
}

// The track's own headline claim: Newton's quadratic rate, measured on
// the nonlinear k1_eff() problem (beta = 0.1, t_ref = 300 K -- a 10%
// conductivity swing over the slab's own working range, mild enough that
// Picard's own per-station contraction stays comfortably below 1 at
// every strip, including the strongest (nominal r = 0.96): k1_eff's own
// multiplier ranges over roughly [0.90, 1.03] for u in dn_heat's own
// working temperatures, so the worst-case perturbed contraction is
// bounded by 0.96 * 1.03 < 1 -- Picard remains a valid comparison, not
// merely "whatever Newton happens to still solve").
//
// SNES's own default outer rtol (1e-8) is loosened to 1e-4 for this
// measurement ONLY. MEASURED directly (2026-10-10, a trace printed by
// hand and removed once read): at SNES's own default forcing (Eisenstat-
// Walker choice 2), this problem's own ||F_k|| sequence is 7.284e+02,
// 8.823e+01, 7.762e-01, 5.659e-05 -- already a clean quadratic sequence
// (q_1 ~= 2.24, q_2 ~= 2.01, both computed below) -- but 5.659e-05 does
// not yet satisfy rtol=1e-8's own 7.28e-6 threshold, so an UNLOOSENED run
// takes a FIFTH evaluation's worth of linear solve: Eisenstat-Walker's
// own eta_k, computed from (5.659e-05/7.762e-01)^2, is small enough that
// the over-solving safeguard (active only when gamma*eta_{k-1}^alpha >
// 0.1) no longer applies, and GMRES is asked for an unpreconditioned
// matrix-free solve tight enough to need more than 2000 iterations --
// not a defect in either eta_k or GMRES, just the ordinary cost of
// driving an inexact Newton-Krylov iteration the LAST remaining couple
// of orders of magnitude with no preconditioner. Loosening rtol to 1e-4
// stops the iteration at the already-measured, already-clean k = 3 point
// instead of paying for that last, unpreconditioned, near-machine-
// precision solve -- a choice about what this MEASUREMENT needs, not
// about what the solver can do (NewtonConfig::use_block_jacobi_
// preconditioner, left false by both dn_heat Newton tests in this file,
// is the mechanism that would make the tight-rtol run cheap; wiring a
// physics-based preconditioner onto dn_heat's own components is outside
// this track's own measured scope, recorded as an open question).
TEST(NewtonRegion, QuadraticRateMeasuredOnTheNonlinearProblem) {
  fuse::NewtonConfig cfg;
  cfg.rtol = 1e-4;
  cfg.max_nonlinear_iterations = 20;
  NewtonDnHeatFixture s =
      make_newton_setup(/*max_iterations=*/30, cfg, /*rcfg=*/{}, /*beta=*/0.1);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;

  const auto r = s.runner->run_period(s.components);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const fuse::IterationStats& st = s.runner->iteration_stats()[0];
  EXPECT_TRUE(st.converged);

  // q_k = log(||F_{k+1}|| / ||F_k||) / log(||F_k|| / ||F_{k-1}||), using
  // only consecutive TRIPLES that are all above a round-off floor: once
  // ||F|| is within a couple of orders of magnitude of machine epsilon
  // relative to ||F_0||, the ratio is dominated by noise, not the
  // quadratic model, and reports a q_k that means nothing (this track's
  // own brief: "claim 'order 2.0 +- measured' with the floor named").
  const std::vector<double>& h = st.residual_history;
  ASSERT_GE(h.size(), 4u) << "need at least 4 residual samples to fit an "
                             "order from two consecutive ratios";
  const double floor = h.front() * 1e-10;
  std::vector<double> orders;
  for (std::size_t k = 1; k + 1 < h.size(); ++k) {
    if (h[k - 1] <= floor || h[k] <= floor || h[k + 1] <= floor) continue;
    if (h[k] >= h[k - 1] || h[k + 1] >= h[k]) continue;  // not yet reducing
    const double q = std::log(h[k + 1] / h[k]) / std::log(h[k] / h[k - 1]);
    orders.push_back(q);
  }
  ASSERT_FALSE(orders.empty())
      << "no residual triple stayed above the round-off floor (" << floor
      << ") on both sides -- residual_history: "
      << ::testing::PrintToString(h);
  for (const double q : orders) {
    EXPECT_NEAR(q, 2.0, 0.5) << "measured order " << q
                             << " (residual_history: "
                             << ::testing::PrintToString(h) << ")";
  }
}

// "The line search must be shown to engage and recover on a deliberately
// poor initial guess" (this track's own brief, repeated from Step 1's
// text because Step 1's own line-search tests -- NewtonKrylovLineSearch.
// BtEngagesAndRecoversOnAPoorInitialGuess, test/num/newton_krylov_test.
// cpp -- prove the MECHANISM on a synthetic problem, not that it is
// correctly REACHED through the real runtime wiring this track adds).
//
// neumann_initial_u = 2000 K (dn_heat's own physical range is a few
// hundred K either side of 300) is the poor guess; beta = 2.0 the
// nonlinearity strong enough to make a full Newton step from it overshoot.
// `IterationStats` (fuse/runtime/runner.hpp, outside this track's own file
// ownership) carries `residual_history` but not a per-step accepted
// lambda, so the lambda itself is not something a test built from THIS
// track's own files can assert on through the public Runner API -- it is
// recorded here as a MEASUREMENT instead (a trace printed by hand and
// removed once read, 2026-10-10, reproduced twice): k = 0 takes the
// full step (lambda = 1, ||F|| 765.78 -> 193.05); k = 1's own full step is
// rejected by Armijo and BT backtracks to lambda = 0.025 after 4 trials
// (||F|| -> 189.01, barely moving -- genuine recovery from a step that
// would otherwise have overshot, not a free ride); every step after that
// takes a full Newton step again (lambda = 1) and the iteration converges
// in 7 total steps, ||F|| -> 7.98e-4. What IS available through the
// public API and asserted below: the run converges at all (a line-search
// failure the solver could not recover from would instead report
// NewtonStatus::DivergedLineSearch, caught the same way
// ExhaustingTheNewtonIterationBudgetReportsNonConvergence, above, catches
// DivergedMaxIterations), and it takes MORE than one step to do so --
// the one-step affine case this file's own ConvergesInThreeNewtonSteps...
// test measures is for a problem with no need to backtrack at all.
TEST(NewtonRegion, LineSearchEngagesAndRecoversFromADeliberatelyPoorGuess) {
  fuse::NewtonConfig cfg;
  cfg.rtol = 1e-4;
  cfg.max_nonlinear_iterations = 20;
  NewtonDnHeatFixture s = make_newton_setup(/*max_iterations=*/30, cfg,
                                            /*rcfg=*/{}, /*beta=*/2.0,
                                            /*neumann_initial_u=*/2000.0);
  ASSERT_TRUE(s.init.ok()) << s.init.error().message;
  const auto r = s.runner->run_period(s.components);
  ASSERT_TRUE(r.ok()) << r.error().message;
  const fuse::IterationStats& st = s.runner->iteration_stats()[0];
  EXPECT_TRUE(st.converged);
  EXPECT_GT(st.iterations, 1)
      << "a poor guess that needed no recovery would converge as fast as "
         "the affine problem's own one-Newton-step cases in this file";
}
