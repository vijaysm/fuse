/// S7.1's Newton-Krylov seam: `Runner::region_fixed_point_residual()` and
/// the `fuse::FixedPointResidual` it returns, exercised on examples/
/// dn_heat's own Dirichlet-Neumann pair (dn_component.hpp, on this file's
/// own include path -- see test/CMakeLists.txt).
///
/// THE CLOSED FORM THIS TEST CHECKS AGAINST. dn_heat fixes kL1 = kL2 = kK2
/// = 1 (dn_component.hpp), so station i's contraction factor r_i =
/// k1_at(i,n) * kL2 / (kK2 * kL1) reduces to r_i = k1_at(i,n) exactly.
/// Block 0 of the region's stacked interface vector is the flux channel
/// ("flux": dirichlet -> neumann); block 1 is the temperature channel
/// ("wall_temperature": neumann -> dirichlet, lagged). Writing x = (q, u)
/// for the two blocks (q = what is about to be delivered to neumann as
/// q_wall_in, u = what is about to be delivered to dirichlet as
/// u_wall_in_i):
///
///   G(x)_q[i] = k1_at(i,n) * (T1 - x_u[i]) / kL1 = r_i * (T1 - x_u[i])
///   G(x)_u[i] = T2 + x_q[i] * kL2 / kK2            = T2 + x_q[i]
///
/// (dirichlet's own `advance()`, and neumann's, each depend on only the
/// OTHER block -- see dn_component.hpp's own `advance()` bodies). Both are
/// affine in x, so F(x) = x - G(x) is affine, with a PER-STATION 2x2
/// Jacobian
///
///   dF_q/dq = 1   dF_q/du =  r_i
///   dF_u/dq = -1  dF_u/du = 1
///
/// -- the off-diagonal PRODUCT magnitude, (r_i)*(1), is exactly r_i: the
/// documented contraction factor is the loop gain around dirichlet ->
/// neumann -> dirichlet, which is exactly what the two off-diagonal
/// entries of this Jacobian multiply out to. (The scalar "u <- T2 + r(T1 -
/// u)" map dn_component.hpp's own header comment documents is the
/// GAUSS-SEIDEL map obtained by ELIMINATING q from this same system --
/// its own slope is 1 + r_i, not 1 - r_i, by the same elimination; this
/// file tests the JACOBI residual `F` the region's own interface-vector
/// assembly produces, not that eliminated scalar map, so it checks the 2x2
/// Jacobian above rather than a single "slope".)
#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include "dn_component.hpp"

#include <memory>
#include <vector>

namespace {

constexpr std::size_t kN = dn::kStrips;  // one station per material strip

fuse::CouplingGraph build_graph(int max_iterations) {
  fuse::CouplingGraph g;
  const std::string src = "generated:line:" + std::to_string(kN);
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

  fuse::ConvergenceCriterion crit;
  crit.add(fuse::Converged::Relative(1e-6, "u_wall"));
  crit.add(fuse::Converged::Relative(1e-6, "q_wall"));

  g.iterate({flux, temp})
      .name("dirichlet_neumann")
      .until(std::move(crit))
      .max_iterations(max_iterations);
  return g;
}

struct DnHeatFixture {
  fuse::CouplingGraph        graph;
  fuse::SerialMeshBackend    backend;
  fuse::ComponentSet         components;
  dn::DirichletSlab*         dirichlet = nullptr;
  dn::NeumannSlab*           neumann = nullptr;
  fuse::FieldRegistry        registry;
  std::unique_ptr<fuse::Runner> runner;
};

/// Builds the graph, registers dn_heat's real components and initializes a
/// Runner -- everything test/runtime/checkpoint_test.cpp-style tests do by
/// hand, pulled into one helper because every test below needs it.
DnHeatFixture make_setup(int max_iterations = 30) {
  DnHeatFixture s;
  s.graph = build_graph(max_iterations);
  for (std::uint32_t i = 0; i < s.graph.mesh_count(); ++i) {
    const auto r = s.backend.load(s.graph.mesh(fuse::MeshId{i}));
    if (!r.ok()) {
      ADD_FAILURE() << "mesh load: " << r.error().message;
      return s;
    }
  }
  const auto lowered = lower(s.graph);
  if (!lowered.ok()) {
    ADD_FAILURE() << lowered.diags.to_string();
    return s;
  }
  const fuse::ClockSet clocks{s.graph};
  auto sched = schedule(s.graph, lowered.dag, clocks, fuse::make_region_sizes(s.graph, lowered.dag, s.backend));
  if (!sched.ok()) {
    ADD_FAILURE() << sched.diags.to_string();
    return s;
  }

  auto dirichlet = std::make_unique<dn::DirichletSlab>(
      "dirichlet", kN, dn::Wiring::Connected);
  auto neumann = std::make_unique<dn::NeumannSlab>(
      "neumann", kN, dn::Wiring::Connected);
  s.dirichlet = dirichlet.get();
  s.neumann = neumann.get();
  s.components.add(std::move(dirichlet));
  s.components.add(std::move(neumann));
  for (auto& c : s.components) {
    const auto r = c.register_fields(s.registry, s.graph);
    if (!r.ok()) ADD_FAILURE() << "register_fields: " << r.error().message;
  }

  s.runner = std::make_unique<fuse::Runner>(s.graph, lowered.dag, sched.plan,
                                            s.backend);
  fuse::RunnerConfig cfg;
  const auto r = s.runner->initialize(s.components, s.registry, cfg);
  if (!r.ok()) ADD_FAILURE() << "initialize: " << r.error().message;
  return s;
}

}  // namespace

TEST(FixedPointResidual, BlockLayoutMatchesTheTwoChannels) {
  DnHeatFixture s = make_setup();
  const auto fpr = s.runner->region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(fpr.ok()) << fpr.error().message;
  EXPECT_EQ(fpr.value().n, 2 * kN);
  ASSERT_EQ(fpr.value().offsets.size(), 2u);
  ASSERT_EQ(fpr.value().lengths.size(), 2u);
  EXPECT_EQ(fpr.value().lengths[0], kN);
  EXPECT_EQ(fpr.value().lengths[1], kN);
  EXPECT_EQ(fpr.value().channel_names[0], "flux");
  EXPECT_EQ(fpr.value().channel_names[1], "wall_temperature");
}

TEST(FixedPointResidual, OutOfRangeRegionIndexIsRefused) {
  DnHeatFixture s = make_setup();
  const auto fpr = s.runner->region_fixed_point_residual(1, s.components);
  EXPECT_FALSE(fpr.ok());
  EXPECT_EQ(fpr.error().code, "E-RT-REGION");
}

// The closed-form check this file's own header comment derives: F is
// affine, and its 2x2-per-station Jacobian is exactly {{1, r_i}, {-1, 1}}.
// Measured by finite difference at an arbitrary nonzero step -- since F is
// genuinely affine, a one-sided difference at ANY step size is exact to
// floating-point rounding, not merely approximate, so this is not a
// truncation-error measurement.
TEST(FixedPointResidual, JacobianMatchesTheClosedFormToRoundoff) {
  DnHeatFixture s = make_setup();
  const auto fpr = s.runner->region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(fpr.ok()) << fpr.error().message;
  const std::size_t n = fpr.value().n;

  std::vector<double> x0(n, 0.0), f0(n, 0.0);
  for (std::size_t i = 0; i < kN; ++i) {
    x0[i] = 40.0 + 0.1 * static_cast<double>(i);        // q block, W/m^2
    x0[kN + i] = 300.0 + 0.5 * static_cast<double>(i);  // u block, K
  }
  ASSERT_TRUE(fpr.value().evaluate(x0, f0).ok());

  constexpr double kStep = 1.0;
  for (std::size_t i = 0; i < kN; ++i) {
    // Perturb q[i] alone: expect dF_q[i]/dq[i] = 1, dF_u[i]/dq[i] = -1,
    // every other entry untouched (the map is diagonal in station index).
    std::vector<double> xq = x0, fq(n, 0.0);
    xq[i] += kStep;
    ASSERT_TRUE(fpr.value().evaluate(xq, fq).ok());
    EXPECT_NEAR((fq[i] - f0[i]) / kStep, 1.0, 1e-9) << "station " << i;
    EXPECT_NEAR((fq[kN + i] - f0[kN + i]) / kStep, -1.0, 1e-9)
        << "station " << i;

    // Perturb u[i] alone: expect dF_q[i]/du[i] = r_i, dF_u[i]/du[i] = 1.
    std::vector<double> xu = x0, fu(n, 0.0);
    xu[kN + i] += kStep;
    ASSERT_TRUE(fpr.value().evaluate(xu, fu).ok());
    const double r_i = dn::strip_ratio(i, kN);
    EXPECT_NEAR((fu[i] - f0[i]) / kStep, r_i, 1e-9) << "station " << i;
    EXPECT_NEAR((fu[kN + i] - f0[kN + i]) / kStep, 1.0, 1e-9)
        << "station " << i;

    // The documented contraction factor is exactly the magnitude of the
    // loop gain around the two off-diagonal entries just measured.
    const double loop_gain =
        ((fu[i] - f0[i]) / kStep) * (-(fq[kN + i] - f0[kN + i]) / kStep);
    EXPECT_NEAR(loop_gain, r_i, 1e-9) << "station " << i;
  }
}

// Affine, not merely locally linear: superposition must hold EXACTLY
// (to round-off) between two arbitrary points, not just in a small
// neighbourhood of one.
TEST(FixedPointResidual, SuperpositionHoldsBetweenTwoArbitraryPoints) {
  DnHeatFixture s = make_setup();
  const auto fpr = s.runner->region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(fpr.ok());
  const std::size_t n = fpr.value().n;

  std::vector<double> x1(n), x2(n), xm(n);
  for (std::size_t i = 0; i < n; ++i) {
    x1[i] = 310.0 + static_cast<double>(i);
    x2[i] = 20.0 - 2.0 * static_cast<double>(i);
    xm[i] = 0.3 * x1[i] + 0.7 * x2[i];
  }
  std::vector<double> f1(n), f2(n), fm(n);
  ASSERT_TRUE(fpr.value().evaluate(x1, f1).ok());
  ASSERT_TRUE(fpr.value().evaluate(x2, f2).ok());
  ASSERT_TRUE(fpr.value().evaluate(xm, fm).ok());
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(fm[i], 0.3 * f1[i] + 0.7 * f2[i], 1e-9) << "entry " << i;
  }
}

// Two evaluations at the SAME x, with an intervening call at a different
// point in between, must be bit-identical -- a Krylov iteration or a
// finite-difference Jacobian action can revisit a point, and the result
// must not depend on evaluation history.
TEST(FixedPointResidual, RepeatedEvaluationAtTheSamePointIsBitIdentical) {
  DnHeatFixture s = make_setup();
  const auto fpr = s.runner->region_fixed_point_residual(0, s.components);
  ASSERT_TRUE(fpr.ok());
  const std::size_t n = fpr.value().n;
  std::vector<double> x(n), other(n), f1(n), f2(n), scratch(n);
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = 305.0 - static_cast<double>(i);
    other[i] = -40.0 + static_cast<double>(i);
  }
  ASSERT_TRUE(fpr.value().evaluate(x, f1).ok());
  ASSERT_TRUE(fpr.value().evaluate(other, scratch).ok());
  ASSERT_TRUE(fpr.value().evaluate(x, f2).ok());
  EXPECT_EQ(f1, f2);
}

// The no-side-effect property, over the WHOLE Runner, not just one
// component: evaluating F some number of times and then letting the
// Runner's own Picard iteration run from the same post-initialize() state
// must reach the same fixed point, bit for bit, as a fresh run that never
// evaluated F at all.
TEST(FixedPointResidual, EvaluatingFDoesNotPerturbTheRunnersOwnPicardIteration) {
  DnHeatFixture baseline = make_setup();
  ASSERT_TRUE(baseline.runner->run(baseline.components, 1).ok());
  const std::vector<double> baseline_u = baseline.neumann->u();
  const std::vector<double> baseline_q = baseline.dirichlet->q();

  DnHeatFixture probed = make_setup();
  const auto fpr = probed.runner->region_fixed_point_residual(0,
                                                              probed.components);
  ASSERT_TRUE(fpr.ok());
  const std::size_t n = fpr.value().n;
  std::vector<double> x(n), f(n);
  for (std::size_t i = 0; i < kN; ++i) {
    x[i] = 1000.0;              // deliberately far from the fixed point
    x[kN + i] = -1000.0;
  }
  // Evaluate several times, including at the SAME point twice, before
  // letting the Runner's own Picard iteration proceed.
  ASSERT_TRUE(fpr.value().evaluate(x, f).ok());
  ASSERT_TRUE(fpr.value().evaluate(x, f).ok());
  for (double& v : x) v *= 0.5;
  ASSERT_TRUE(fpr.value().evaluate(x, f).ok());

  ASSERT_TRUE(probed.runner->run(probed.components, 1).ok());
  EXPECT_EQ(probed.neumann->u(), baseline_u);
  EXPECT_EQ(probed.dirichlet->q(), baseline_q);
}
