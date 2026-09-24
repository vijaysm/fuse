// Step 3 of the capabilities plan: confirm that FUSE can move data across a
// NON-MATCHING 1D interface -- the shape the FSI channel (Nx_f fluid wall
// edges, Nx_b beam nodes, Nx_f != Nx_b) needs -- through the Runner, on a
// real two-component graph, not by calling build_supermesh_weights() directly
// the way test/backend/supermesh_test.cpp does.
//
// WHAT ALREADY EXISTED (read before writing this file):
//   * test/backend/supermesh_test.cpp: build_supermesh_weights() on
//     non-matching 1D line pairs is thoroughly verified -- exact conservation
//     to round-off, observed order >= 1, closed-form weight checks.
//   * Runner::Impl::resolve_remap() (src/runtime/runner.cpp) already builds
//     real conservative weights via build_weights() whenever the two regions
//     do not coincide, and falls back to an identity copy only when they do.
//   * Every existing example/verification case (mms_pair, mms_transfer) uses
//     COINCIDENT meshes, so none of them ever exercises the non-identity path
//     end to end.
// So the gap this file closes is purely the end-to-end demonstration: does a
// channel between two DIFFERENT-resolution line meshes deliver a correct,
// checkable answer when driven by lower -> schedule -> Runner, in both the
// pointwise and the conservative direction, on one graph?
//
// TWO TRANSFER KINDS, ONE GRAPH:
//   * "beam_out" -> "fluid_in": a POINTWISE quantity (wall velocity), plain
//     remap (no `.conservative()`). FUSE's only remap kernel for the native
//     backend is the conservative P0 finite-volume cell-average map -- there
//     is no separate nodal/"consistent" kernel that reconstructs a linear
//     variation within a cell. CORRECTED FROM A FIRST DRAFT OF THIS FILE:
//     a P0 map is exact for a spatially CONSTANT field on any partition (the
//     mean of a constant is itself), but it is NOT pointwise-exact for a
//     LINEAR field at a fixed, genuinely non-nested resolution -- treating
//     each source cell's average as flat across the whole cell introduces an
//     O(h) error near every cell boundary that does not line up with the
//     target grid, same order as for any other smooth field. Measured
//     directly (worst error 0.075 on the case below) before this comment was
//     corrected to match. `PointwiseTransferIsExactForAConstantField` is
//     the exactness property the P0 kernel genuinely has;
//     `TransferOrderOnASmoothField` is the refinement-order property (>= 1)
//     that covers a linear field at VARYING resolution instead. A true
//     pointwise-consistent (order >= 2, exact-for-linear-at-fixed-resolution)
//     kernel would need a genuinely different reconstruction (P1/nodal) that
//     does not exist yet for the native backend -- see the task's
//     open_questions.
//   * "fluid_force" -> "beam_force": an INTEGRATED quantity (traction
//     integrated to a per-cell force density), Conservation::Extensive,
//     channel marked `.conservative()`. `ConservativeTransferPreservesTotalForce`
//     checks the measure-weighted SUM delivered equals the measure-weighted
//     sum sent, to round-off (Extensive fields are densities: the conserved
//     amount is value*measure, confirmed empirically -- the un-weighted sums
//     disagreed by exactly the n_fluid/n_beam ratio).
//
// `TransferOrderOnASmoothField` repeats the pointwise study over four
// resolutions and measures the order, mirroring
// Supermesh.ObservedOrderOnANonMatchingPairIsAtLeastOne but through the
// Runner rather than by calling the weight builder directly.

#include <gtest/gtest.h>
#include <fuse/ir/lowering.hpp>
#include <fuse/backend/consistent_interp.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>

#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::SerialMeshBackend;
using fuse::StepContext;
namespace u = fuse::units;

namespace {

/// Cell centre of cell `i` of an `n`-cell uniform mesh on [0, 1] -- the same
/// convention test/verification/mms_transfer_test.cpp documents for
/// "generated:line:n".
[[nodiscard]] double cell_x(std::size_t i, std::size_t n) {
  return (static_cast<double>(i) + 0.5) / static_cast<double>(n);
}

/// Velocity units: metres per second. include/fuse/base/units.hpp does not
/// define this composite, so it is built directly from the base exponents.
inline constexpr fuse::Units kMetersPerSecond{.length = 1, .time = -1};

/// One period through the Runner of a graph with two independent channels:
///   beam_out  --(plain remap)-->      fluid_in     (pointwise, "velocity")
///   fluid_out --(conservative remap)--> beam_in    (integrated, "force")
/// on two non-matching 1D line meshes. `n_beam != n_fluid` so every weight
/// is a genuine partial overlap, not a coincidence.
struct TransferCase {
  std::size_t n_beam;
  std::size_t n_fluid;
  std::function<double(double)> velocity;  // beam -> fluid, pointwise
  std::function<double(double)> traction;  // fluid -> beam, per-cell force
};

struct RunResult {
  std::vector<double> fluid_in;    // received velocity, one per fluid cell
  std::vector<double> beam_force;  // received force, one per beam cell
  std::vector<double> beam_sent;   // traction sent per fluid cell (source)
};

RunResult run_once(const TransferCase& s) {
  CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{
      .name = "beam_mesh", .source = "generated:line:" + std::to_string(s.n_beam)});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{
      .name = "fluid_mesh", .source = "generated:line:" + std::to_string(s.n_fluid)});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});

  const auto beam = g.add_component(fuse::ComponentDecl{
      .name = "beam", .ranks = {0, 1}, .interval = Rational{1}});
  const auto fluid = g.add_component(fuse::ComponentDecl{
      .name = "fluid", .ranks = {0, 1}, .interval = Rational{1}});

  // Pointwise channel: beam velocity -> fluid wall. Intensive: the value at a
  // cell IS the pointwise quantity, not an amount to be conserved.
  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  g.add_channel("wall_velocity", vel_out, vel_in);  // plain remap: no .conservative()

  // Conservative channel: fluid traction (integrated to a per-cell force) ->
  // beam. Extensive + .conservative() is what makes lowering emit Constrain
  // and asks build_weights() for a mass-preserving map.
  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = u::kNewton,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Provides});
  const auto force_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_force_in", .region = r_beam, .units = u::kNewton,
      .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  // Both fields are pure functions of x, not of time, so a one-period lag on
  // the return channel changes nothing about what is measured -- it exists
  // only to turn this two-way pair into a legal explicit scheme (spec
  // §8.2's E-SCHED-CYCLE hard gate refuses an unlagged algebraic loop; this
  // test is about transfer accuracy, not about the FSI model's own strong
  // coupling, which is a later phase's concern).
  const auto fch = g.add_channel("wall_force", force_out, force_in);
  g.channel(fch).conservative().lag(1).allow_time_level_mismatch(
      "capability test: only transfer accuracy is under test here");

  SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error(r.error().message);
  }

  const auto lowered = lower(g);
  if (!lowered.ok()) throw std::runtime_error(lowered.diags.to_string());

  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks,
                              fuse::make_region_sizes(g, backend));
  if (!sched.ok()) throw std::runtime_error(sched.diags.to_string());

  ComponentSet components;
  // Two components in the graph, "beam" and "fluid", each both a source AND
  // a sink of one field -- so the ComponentSet needs exactly one Component
  // per graph component that owns BOTH of its fields. Fold the pairs.
  class BeamComponent final : public fuse::Component {
   public:
    BeamComponent(std::size_t n, std::function<double(double)> vel)
        : n_(n), vel_(std::move(vel)), velocity_(n, 0.0), force_in_(n, 0.0) {}
    [[nodiscard]] std::string name() const override { return "beam"; }
    fuse::Result<void> register_fields(FieldRegistry& reg,
                                       const CouplingGraph& g) override {
      const auto cid = g.find_component("beam");
      for (const fuse::FieldId f : g.component(*cid).fields) {
        const std::string& fname = g.field(f).name;
        double* buf = fname == "beam_velocity" ? velocity_.data() : force_in_.data();
        auto r = reg.bind(f, fuse::FieldRef{buf, fuse::Layout{.stride = 1, .count = n_},
                                            fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override {
      for (std::size_t i = 0; i < n_; ++i) velocity_[i] = vel_(cell_x(i, n_));
      return fuse::Result<void>::success();
    }
    [[nodiscard]] const std::vector<double>& force_in() const { return force_in_; }

   private:
    std::size_t                   n_;
    std::function<double(double)> vel_;
    std::vector<double>           velocity_;
    std::vector<double>           force_in_;
  };
  class FluidComponent final : public fuse::Component {
   public:
    FluidComponent(std::size_t n, std::function<double(double)> trac)
        : n_(n), trac_(std::move(trac)), force_(n, 0.0), velocity_in_(n, 0.0) {}
    [[nodiscard]] std::string name() const override { return "fluid"; }
    fuse::Result<void> register_fields(FieldRegistry& reg,
                                       const CouplingGraph& g) override {
      const auto cid = g.find_component("fluid");
      for (const fuse::FieldId f : g.component(*cid).fields) {
        const std::string& fname = g.field(f).name;
        double* buf = fname == "fluid_force" ? force_.data() : velocity_in_.data();
        auto r = reg.bind(f, fuse::FieldRef{buf, fuse::Layout{.stride = 1, .count = n_},
                                            fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override {
      for (std::size_t i = 0; i < n_; ++i) force_[i] = trac_(cell_x(i, n_));
      return fuse::Result<void>::success();
    }
    [[nodiscard]] const std::vector<double>& velocity_in() const { return velocity_in_; }
    [[nodiscard]] const std::vector<double>& force() const { return force_; }

   private:
    std::size_t                   n_;
    std::function<double(double)> trac_;
    std::vector<double>           force_;
    std::vector<double>           velocity_in_;
  };

  auto beam_comp = std::make_unique<BeamComponent>(s.n_beam, s.velocity);
  auto fluid_comp = std::make_unique<FluidComponent>(s.n_fluid, s.traction);
  BeamComponent* beam_ptr = beam_comp.get();
  FluidComponent* fluid_ptr = fluid_comp.get();
  components.add(std::move(beam_comp));
  components.add(std::move(fluid_comp));

  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) throw std::runtime_error(r.error().message);
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry); !r.ok()) {
    throw std::runtime_error(r.error().message);
  }
  // Two periods: the lagged force channel delivers period 0's sent value at
  // period 1. Both source functions are time-independent, so "delivered
  // after 2 periods" and "sent this period" are the same number.
  if (const auto r = runner.run(components, /*periods=*/2); !r.ok()) {
    throw std::runtime_error(r.error().message);
  }

  RunResult out;
  out.fluid_in = fluid_ptr->velocity_in();
  out.beam_force = beam_ptr->force_in();
  out.beam_sent = fluid_ptr->force();
  return out;
}

double measure_of(std::size_t n) { return 1.0 / static_cast<double>(n); }

}  // namespace

TEST(InterfaceTransfer, PointwiseTransferIsExactForAConstantField) {
  // The P0 remap's genuine exactness property: a spatially constant field's
  // cell average is itself, on any partition, so the delivered value at
  // every one of a non-matching partner's cells must equal the constant --
  // to round-off. (A linear field is NOT reproduced exactly at fixed
  // resolution on a non-nested pair; see the file header and
  // TransferOrderOnASmoothField for that property instead.)
  const TransferCase s{.n_beam = 5, .n_fluid = 8,
               .velocity = [](double) { return 4.5; },
               .traction = [](double) { return -2.0; }};
  const RunResult r = run_once(s);
  ASSERT_EQ(r.fluid_in.size(), s.n_fluid);
  double worst = 0.0;
  for (const double v : r.fluid_in) worst = std::max(worst, std::abs(v - 4.5));
  EXPECT_LT(worst, 1e-12) << "worst pointwise error on a constant field";
}

TEST(InterfaceTransfer, ConservativeTransferPreservesTotalForce) {
  const TransferCase s{.n_beam = 5, .n_fluid = 8,
               .velocity = [](double x) { return 1.0 + 2.0 * x; },
               .traction = [](double x) { return 3.0 + 5.0 * std::sin(3.0 * x); }};
  const RunResult r = run_once(s);
  ASSERT_EQ(r.beam_force.size(), s.n_beam);
  ASSERT_EQ(r.beam_sent.size(), s.n_fluid);

  // "force" is a per-cell amount (already integrated), so the TOTAL is the
  // plain sum across cells -- no measure weighting, unlike a density.
  // Extensive fields are densities (spec's FV convention: a cell's stored
  // value integrates to an amount only once multiplied by the cell's
  // measure), so the conserved total is the measure-weighted sum, not the
  // plain sum of per-cell values -- confirmed empirically: the plain sums
  // disagreed by exactly the n_fluid/n_beam ratio before this weighting was
  // added.
  double total_sent = 0.0;
  for (const double v : r.beam_sent) total_sent += v * measure_of(s.n_fluid);
  double total_received = 0.0;
  for (const double v : r.beam_force) total_received += v * measure_of(s.n_beam);

  EXPECT_NEAR(total_received, total_sent, 1e-11 * std::abs(total_sent))
      << "sent " << total_sent << " received " << total_received;
}

TEST(InterfaceTransfer, TransferOrderOnASmoothField) {
  // Same study as Supermesh.ObservedOrderOnANonMatchingPairIsAtLeastOne, but
  // through lower -> schedule -> Runner rather than build_supermesh_weights()
  // directly. 11/7 is deliberately non-integer so no target cell is a union
  // of whole source cells (which would be exact and measure nothing).
  const auto f = [](double x) {
    constexpr double two_pi = 6.283185307179586476925287;
    return std::sin(two_pi * x);
  };
  std::vector<double> hs, errs;
  for (const std::size_t nb : {14u, 28u, 56u, 112u}) {
    const std::size_t nf = nb * 11 / 7;
    const TransferCase s{.n_beam = nb, .n_fluid = nf, .velocity = f,
                 .traction = [](double) { return 0.0; }};
    const RunResult r = run_once(s);
    double l2 = 0.0;
    for (std::size_t j = 0; j < nf; ++j) {
      const double want = f(cell_x(j, nf));
      const double e = r.fluid_in[j] - want;
      l2 += e * e * measure_of(nf);
    }
    hs.push_back(1.0 / static_cast<double>(nb));
    errs.push_back(std::sqrt(l2));
  }
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(hs.size());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    const double x = std::log(hs[k]), y = std::log(errs[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const double order = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  EXPECT_GT(order, 0.9) << "observed order " << order
                        << ", finest error " << errs.back();
  EXPECT_LT(order, 3.0) << "suspiciously high: pair accidentally nested?";
}

// ===========================================================================
// Consistent H / conservative H^T, exercised through build_weights() --
// intx_weights.cpp's public RemapPlan-level entry point, the SAME function
// resolve_remap() (src/runtime/runner.cpp) calls -- against a REAL
// CouplingGraph + SerialMeshBackend, at the level the runtime actually
// dispatches at. Not run through lower()/schedule()/Runner: a channel
// between a genuinely nodal field (Discretization::ContinuousGLL order <= 1,
// SemanticLocation::Vertex) and an FV field (SemanticLocation::Element) is
// refused by include/fuse/model/connect.hpp's check_channel(), called from
// src/ir/rules_channel.cpp during lower() -- neither file is owned by this
// track, and check_channel() hardcodes `request.transfer.kind =
// TransferKind::Copy` unconditionally, so a location-changing channel has no
// way to declare the transfer that would let it through. This is reported
// exactly as found in the task's open_questions; it is not worked around
// here. What IS tested here is everything this track owns: build_weights()'s
// dispatch on `RemapPlan::method`, region resolution through the graph, the
// discretization gate, and that the weights returned match
// build_consistent_interp_weights() called directly on the same MeshViews.
// ===========================================================================

namespace {

/// A nodal region (dimension 0, one entity per vertex) and an FV region
/// (dimension 1, one entity per cell) on two independently-resolved
/// "generated:line" meshes, with no fields or channels -- build_weights()
/// only needs the graph's meshes and regions, not a validated channel.
struct PlanFixture {
  CouplingGraph        g;
  SerialMeshBackend    backend;
  fuse::RegionId       r_beam_nodal;   // dimension 0, n_beam + 1 entities
  fuse::RegionId       r_fluid_cells;  // dimension 1, n_fluid entities
};

[[nodiscard]] std::unique_ptr<PlanFixture> plan_fixture(std::size_t n_beam,
                                                        std::size_t n_fluid) {
  auto pf = std::make_unique<PlanFixture>();
  const auto m_beam = pf->g.add_mesh(
      fuse::MeshDecl{.name = "beam_mesh", .source = "generated:line:" + std::to_string(n_beam)});
  const auto m_fluid = pf->g.add_mesh(
      fuse::MeshDecl{.name = "fluid_mesh", .source = "generated:line:" + std::to_string(n_fluid)});
  pf->r_beam_nodal = pf->g.add_region(
      fuse::Region{.name = "r_beam_nodal", .mesh = m_beam, .dimension = 0});
  pf->r_fluid_cells = pf->g.add_region(
      fuse::Region{.name = "r_fluid_cells", .mesh = m_fluid, .dimension = 1});
  for (std::uint32_t i = 0; i < pf->g.mesh_count(); ++i) {
    if (!pf->backend.load(pf->g.mesh(fuse::MeshId{i})).ok()) return nullptr;
  }
  return pf;
}

}  // namespace

TEST(ConsistentInterpPlan, ConsistentMethodMatchesTheDirectBuilder) {
  auto pf = plan_fixture(5, 8);
  ASSERT_TRUE(pf);

  fuse::RemapPlan plan;
  plan.src = pf->r_beam_nodal;
  plan.tgt = pf->r_fluid_cells;
  plan.src_disc = fuse::Discretization::ContinuousGLL;
  plan.src_order = 1;
  plan.tgt_disc = fuse::Discretization::FV;
  plan.tgt_order = 0;
  plan.method = "consistent";

  const auto via_plan = fuse::build_weights(plan, pf->g, pf->backend);
  ASSERT_TRUE(via_plan.ok()) << via_plan.error().message;

  const auto bid = pf->backend.mesh_id("beam_mesh");
  const auto fid = pf->backend.mesh_id("fluid_mesh");
  ASSERT_TRUE(bid.ok() && fid.ok());
  const auto bv = pf->backend.view(bid.value());
  const auto fv = pf->backend.view(fid.value());
  ASSERT_TRUE(bv.ok() && fv.ok());
  const auto direct = fuse::build_consistent_interp_weights(*bv.value(), *fv.value(), 1);
  ASSERT_TRUE(direct.ok()) << direct.error().message;

  const fuse::RemapWeights& a = via_plan.value();
  const fuse::RemapWeights& b = direct.value();
  ASSERT_EQ(a.n_rows, b.n_rows);
  ASSERT_EQ(a.n_cols, b.n_cols);
  ASSERT_EQ(a.nnz(), b.nnz());
  for (std::size_t k = 0; k < a.val.size(); ++k) {
    EXPECT_EQ(a.row[k], b.row[k]) << "entry " << k;
    EXPECT_EQ(a.col[k], b.col[k]) << "entry " << k;
    EXPECT_DOUBLE_EQ(a.val[k], b.val[k]) << "entry " << k;
  }
}

TEST(ConsistentInterpPlan, ConsistentTransposeMethodDeliversTheConservativeLoad) {
  auto pf = plan_fixture(5, 8);
  ASSERT_TRUE(pf);

  // The paired channel: fluid (FV, density) -> beam (nodal, load).
  fuse::RemapPlan plan;
  plan.src = pf->r_fluid_cells;
  plan.tgt = pf->r_beam_nodal;
  plan.src_disc = fuse::Discretization::FV;
  plan.src_order = 0;
  plan.tgt_disc = fuse::Discretization::ContinuousGLL;
  plan.tgt_order = 1;
  plan.method = "consistent-transpose";

  const auto via_plan = fuse::build_weights(plan, pf->g, pf->backend);
  ASSERT_TRUE(via_plan.ok()) << via_plan.error().message;
  const fuse::RemapWeights& HtD = via_plan.value();

  const auto bid = pf->backend.mesh_id("beam_mesh");
  const auto fid = pf->backend.mesh_id("fluid_mesh");
  const auto bv = pf->backend.view(bid.value());
  const auto fv = pf->backend.view(fid.value());
  ASSERT_TRUE(bv.ok() && fv.ok());

  ASSERT_EQ(HtD.n_rows, 6u);  // beam vertices
  ASSERT_EQ(HtD.n_cols, 8u);  // fluid cells

  // Total force conserved: sum(H^T D rho) == sum(D rho), for an arbitrary
  // density -- the property build_consistent_weights_for_plan's header
  // claims follows purely from H's row-sum-one identity.
  std::vector<double> rho(8);
  for (std::size_t i = 0; i < 8; ++i) rho[i] = 1.7 - 0.3 * static_cast<double>(i);
  std::vector<double> F(6, 0.0);
  ASSERT_TRUE(fuse::apply_weights(HtD, rho.data(), F.data(), 1).ok());
  double sent = 0.0;
  for (std::size_t j = 0; j < 8; ++j) sent += rho[j] * fv.value()->measure(1, j);
  double received = 0.0;
  for (double v : F) received += v;
  EXPECT_NEAR(sent, received, 1e-12 * std::abs(sent));
}

TEST(ConsistentInterpPlan, WrongDiscretizationOnTheNodalSideRefuses) {
  auto pf = plan_fixture(5, 8);
  ASSERT_TRUE(pf);
  fuse::RemapPlan plan;
  plan.src = pf->r_beam_nodal;
  plan.tgt = pf->r_fluid_cells;
  // Left at the RemapPlan default: Discretization::FV, not the nodal
  // ContinuousGLL order<=1 the "consistent" method requires of its source.
  plan.method = "consistent";
  const auto r = fuse::build_weights(plan, pf->g, pf->backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTERP-DISC");
}

TEST(ConsistentInterpPlan, UnknownMethodStringRefuses) {
  auto pf = plan_fixture(5, 8);
  ASSERT_TRUE(pf);
  fuse::RemapPlan plan;
  plan.src = pf->r_beam_nodal;
  plan.tgt = pf->r_fluid_cells;
  plan.src_disc = fuse::Discretization::ContinuousGLL;
  plan.src_order = 1;
  plan.method = "consistent-but-misspelled";
  const auto r = fuse::build_consistent_weights_for_plan(plan, pf->g, pf->backend);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTERP-METHOD");
}
