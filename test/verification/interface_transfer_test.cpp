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
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <map>
#include <string>
#include <vector>

// Nodal-remap track, round 3: apply_nodal_dist_remap's own per-period
// allocation check (below, ApplyNodalDistRemapHasAConstantMeasuredPerPeriodAllocationCount)
// needs this. Safe to include here: this file's own test target compiles
// exactly one .cpp against the prebuilt fuse_test_main -- see the header's
// own comment on why it may appear in only one TU per binary.
#include "../support/allocation_counter.hpp"

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

// ---- consistent H and its transpose, end to end through the Runner --------
//
// The FSI pairing: beam velocity (nodal, ContinuousGLL order 1) to the fluid
// wall (FV) through H, and the fluid's per-cell force back to beam nodal
// loads through H^T. Through lower -> schedule -> Runner, not RemapPlan, so a
// refusal anywhere on the public path shows up here.
namespace {

struct NodalResult {
  std::vector<double> v_nodes, fluid_in, fluid_force, beam_load;
};

NodalResult run_consistent_pair(std::size_t n_beam, std::size_t n_fluid,
                                const std::function<double(double)>& vel,
                                const std::function<double(double)>& trac,
                                fuse::RankSet beam_ranks = {0, 1},
                                // Round 2 (nodal-remap track): lets a caller
                                // force the "wall_force" channel's own
                                // method away from "consistent-transpose",
                                // to exercise the refusal a non-consistent
                                // method into a multi-rank nodal target
                                // still gets (E-RT-REMAP-NODAL-PARALLEL,
                                // KNOWN_ISSUES D5 item 3) through this same,
                                // already-built graph shape instead of a
                                // second hand-rolled one.
                                const std::string& force_method = "consistent-transpose",
                                const std::string& vel_method = "consistent") {
  CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{
      .name = "beam_mesh", .source = "generated:line:" + std::to_string(n_beam)});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{
      .name = "fluid_mesh", .source = "generated:line:" + std::to_string(n_fluid)});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
  const auto beam = g.add_component(fuse::ComponentDecl{
      .name = "beam", .ranks = beam_ranks, .interval = Rational{1}});
  const auto fluid = g.add_component(fuse::ComponentDecl{
      .name = "fluid", .ranks = {0, 1}, .interval = Rational{1}});
  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method(vel_method);
  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = u::kNewton,
      .conservation = fuse::Conservation::Extensive, .intent = fuse::Intent::Provides});
  const auto force_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_load", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = u::kNewton, .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fch = g.add_channel("wall_force", force_out, force_in);
  g.channel(fch).remap_method(force_method).conservative().lag(1)
      .allow_time_level_mismatch("capability test: only transfer accuracy is under test");

  SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    if (!r.ok()) throw std::runtime_error(r.error().message);
  }
  const auto lowered = lower(g);
  if (!lowered.ok()) throw std::runtime_error("lower: " + lowered.diags.to_string());
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  if (!sched.ok()) throw std::runtime_error("schedule: " + sched.diags.to_string());

  // One buffer per field, sized by what the backend says the region holds;
  // a nodal field needs n+1 values, which is what this test is checking.
  const auto sizes = fuse::make_region_sizes(g, backend);
  NodalResult out;
  class Side final : public fuse::Component {
   public:
    Side(std::string name, std::function<void(Side&)> fill) : name_(std::move(name)), fill_(std::move(fill)) {}
    [[nodiscard]] std::string name() const override { return name_; }
    fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
      for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
        auto& buf = bufs[g.field(f).name];
        buf.assign(count(g, f), 0.0);
        auto r = reg.bind(f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = buf.size()},
                                            fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override { fill_(*this); return fuse::Result<void>::success(); }
    std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> count;
    std::map<std::string, std::vector<double>> bufs;
   private:
    std::string name_;
    std::function<void(Side&)> fill_;
  };
  auto b = std::make_unique<Side>("beam", [&](Side& me) {
    auto& v = me.bufs["beam_velocity"];
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = vel(static_cast<double>(i) / n_beam);
  });
  auto f = std::make_unique<Side>("fluid", [&](Side& me) {
    auto& q = me.bufs["fluid_force"];
    for (std::size_t j = 0; j < q.size(); ++j) q[j] = trac(cell_x(j, n_fluid));
  });
  const auto nodal = [&](const CouplingGraph& gg, fuse::FieldId fid) -> std::size_t {
    const auto& fs = gg.field(fid);
    const std::size_t cells = fs.region == r_beam ? n_beam : n_fluid;
    return fs.disc == fuse::Discretization::ContinuousGLL ? cells + 1 : cells;
  };
  b->count = nodal; f->count = nodal;
  Side* bp = b.get(); Side* fp = f.get();
  ComponentSet components;
  components.add(std::move(b)); components.add(std::move(f));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    if (!r.ok()) throw std::runtime_error("bind: " + r.error().message);
  }
  (void)sizes;
  Runner runner{g, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry); !r.ok())
    throw std::runtime_error("initialize: " + r.error().code + ": " + r.error().message);
  if (const auto r = runner.run(components, 2); !r.ok())
    throw std::runtime_error("run: " + r.error().code + ": " + r.error().message);
  out.v_nodes = bp->bufs["beam_velocity"];
  out.beam_load = bp->bufs["beam_load"];
  out.fluid_in = fp->bufs["fluid_velocity_in"];
  out.fluid_force = fp->bufs["fluid_force"];
  return out;
}

}  // namespace

TEST(InterfaceTransfer, ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower) {
  NodalResult r;
  try {
    r = run_consistent_pair(7, 11, [](double x) { return 0.3 + 2.0 * x; },
                            [](double x) { return 1.0 + std::sin(3.0 * x); });
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
  ASSERT_EQ(r.v_nodes.size(), 8u);
  ASSERT_EQ(r.fluid_in.size(), 11u);
  ASSERT_EQ(r.beam_load.size(), 8u);
  for (std::size_t j = 0; j < r.fluid_in.size(); ++j)
    EXPECT_NEAR(r.fluid_in[j], 0.3 + 2.0 * cell_x(j, 11), 1e-13) << "H is not exact for a linear field at cell " << j;
  double sent = 0.0, got = 0.0, p_fluid = 0.0, p_beam = 0.0;
  // An FV source is a density: H^T D scales it by the cell measure, so the
  // conserved quantities are the integrals, not the plain sums.
  const double h_f = measure_of(11);
  for (std::size_t j = 0; j < 11; ++j) { sent += r.fluid_force[j] * h_f; p_fluid += r.fluid_in[j] * r.fluid_force[j] * h_f; }
  for (std::size_t i = 0; i < 8; ++i) { got += r.beam_load[i]; p_beam += r.v_nodes[i] * r.beam_load[i]; }
  EXPECT_NEAR(got, sent, 1e-13 * std::abs(sent)) << "H^T lost force";
  EXPECT_NEAR(p_beam, p_fluid, 1e-13 * std::abs(p_fluid)) << "interface power does not balance";
}

// ---- a nodal field through a PLAIN channel, no remap at all ---------------
//
// Both ends declare the SAME region on the SAME mesh, so lower/schedule/
// Runner take the identity-copy path (Runner::Impl's regions_coincide()),
// never touching consistent_interp.hpp. This isolates the buffer-sizing fix
// (mesh_view.hpp's field_extent/region_extent_kind, wired into
// make_region_sizes) from the H/H^T operator: if a nodal field's buffers were
// still sized by cell count, this would fail at Runner::initialize with
// E-RT-SIZE exactly like the acceptance test did before the fix, even though
// no interpolation is involved at all.

TEST(InterfaceTransfer, NodalFieldPassesThroughAPlainChannelOnTheSameMesh) {
  CouplingGraph g;
  const auto m = g.add_mesh(
      fuse::MeshDecl{.name = "beam_mesh", .source = "generated:line:6"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 1});
  const auto src = g.add_component(fuse::ComponentDecl{
      .name = "src", .ranks = {0, 1}, .interval = Rational{1}});
  const auto dst = g.add_component(fuse::ComponentDecl{
      .name = "dst", .ranks = {0, 1}, .interval = Rational{1}});
  const auto f_out = g.add_field(src, fuse::FieldSpec{
      .name = "v_out", .region = r, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .intent = fuse::Intent::Provides});
  const auto f_in = g.add_field(dst, fuse::FieldSpec{
      .name = "v_in", .region = r, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .intent = fuse::Intent::Requires});
  g.add_channel("v", f_out, f_in);  // no .remap_method(): plain copy

  SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m)).ok());
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sizes = fuse::make_region_sizes(g, backend);
  auto sched = fuse::schedule(g, lowered.dag, clocks, sizes);
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  class Side final : public fuse::Component {
   public:
    Side(std::string name, std::vector<double> initial)
        : name_(std::move(name)), buf(std::move(initial)) {}
    [[nodiscard]] std::string name() const override { return name_; }
    fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& gg) override {
      for (const fuse::FieldId f : gg.component(*gg.find_component(name_)).fields) {
        auto r2 = reg.bind(f, fuse::FieldRef{buf.data(),
                                             fuse::Layout{.stride = 1, .count = buf.size()},
                                             fuse::MemorySpace::Host});
        if (!r2.ok()) return r2;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
    std::vector<double> buf;
   private:
    std::string name_;
  };

  std::vector<double> v_out(7);  // 6 cells -> 7 vertices
  for (std::size_t i = 0; i < v_out.size(); ++i) v_out[i] = 1.0 + 0.5 * static_cast<double>(i);
  auto s = std::make_unique<Side>("src", v_out);
  auto d = std::make_unique<Side>("dst", std::vector<double>(7, 0.0));
  Side* dp = d.get();
  ComponentSet components;
  components.add(std::move(s));
  components.add(std::move(d));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto rb = c.register_fields(registry, g);
    ASSERT_TRUE(rb.ok()) << rb.error().message;
  }
  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto ri = runner.initialize(components, registry);
  ASSERT_TRUE(ri.ok()) << ri.error().code << ": " << ri.error().message;
  const auto rr = runner.run(components, 1);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;

  ASSERT_EQ(dp->buf.size(), 7u);
  for (std::size_t i = 0; i < 7; ++i) {
    EXPECT_DOUBLE_EQ(dp->buf[i], v_out[i]) << "vertex " << i;
  }
}

// ---- the refusal fires end to end, not just inside field_extent() ---------
//
// A ContinuousGLL field at order 2 (a spectral element this codebase does not
// implement an extent for) reaches the Runner's own buffer-planning path
// (make_region_sizes -> plan_buffers -> schedule()) and is refused there,
// named E-EXTENT-DISC, rather than silently sized by cell count.

TEST(InterfaceTransfer, AnUnimplementedNodalOrderRefusesThroughTheRealSchedulingPath) {
  CouplingGraph g;
  const auto m = g.add_mesh(
      fuse::MeshDecl{.name = "beam_mesh", .source = "generated:line:6"});
  const auto r = g.add_region(fuse::Region{.name = "r", .mesh = m, .dimension = 1});
  const auto c = g.add_component(fuse::ComponentDecl{
      .name = "c", .ranks = {0, 1}, .interval = Rational{1}});
  g.add_field(c, fuse::FieldSpec{
      .name = "spectral", .region = r, .disc = fuse::Discretization::ContinuousGLL,
      .order = 2, .units = kMetersPerSecond, .intent = fuse::Intent::Provides});

  SerialMeshBackend backend;
  ASSERT_TRUE(backend.load(g.mesh(m)).ok());
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  const auto sizes = fuse::make_region_sizes(g, backend);

  bool threw = false;
  std::string what;
  try {
    (void)fuse::schedule(g, lowered.dag, clocks, sizes);
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  ASSERT_TRUE(threw) << "expected the unimplemented discretization to refuse, "
                        "not to be sized by cell count";
  EXPECT_NE(what.find("E-EXTENT-DISC"), std::string::npos) << what;
}

// KNOWN_ISSUES D8, round 2 (2026-09-25): a Remap into a nodal field on more
// than one rank via "consistent-transpose" is no longer refused --
// `Runner::Impl::build_nodal_dist_remap` (runner.cpp) builds distributed
// rows through nodal_remap_distributed.hpp instead. The placement is
// DECLARED (`fuse::RankSet{0, 2}`, two ranks in the graph) but this test
// still RUNS as one process with no `RunnerConfig::base_communicator` set,
// so `comm_handle == fuse::search::kNullComm`: exactly the single-
// participant path test/backend/nodal_remap_distributed_nullcomm_test.cpp
// exercises on its own hand-built meshes, exercised HERE through the real
// Runner/CouplingGraph/SerialMeshBackend stack on the SAME channel
// `ConsistentPairRunsThroughTheRunnerExactlyAndBalancesPower` above uses at
// one rank. With no real communicator, this single process already holds
// (and owns) every entity of both meshes, so the correct answer is
// IDENTICAL to the one-rank baseline -- proving the declared rank count by
// itself changes nothing about the physics, only which code path computes
// it (build_nodal_dist_remap vs. the plain local weights-apply).
TEST(InterfaceTransfer, ARemapIntoANodalFieldOnTwoDeclaredRanksIsAcceptedAndMatchesOneRank) {
  NodalResult one_rank, two_ranks;
  ASSERT_NO_THROW(one_rank = run_consistent_pair(
                      7, 11, [](double x) { return 0.3 + 2.0 * x; },
                      [](double x) { return 1.0 + std::sin(3.0 * x); }, fuse::RankSet{0, 1}));
  ASSERT_NO_THROW(two_ranks = run_consistent_pair(
                      7, 11, [](double x) { return 0.3 + 2.0 * x; },
                      [](double x) { return 1.0 + std::sin(3.0 * x); }, fuse::RankSet{0, 2}));

  ASSERT_EQ(two_ranks.v_nodes.size(), one_rank.v_nodes.size());
  ASSERT_EQ(two_ranks.beam_load.size(), one_rank.beam_load.size());
  ASSERT_EQ(two_ranks.fluid_in.size(), one_rank.fluid_in.size());
  for (std::size_t i = 0; i < one_rank.beam_load.size(); ++i) {
    EXPECT_NEAR(two_ranks.beam_load[i], one_rank.beam_load[i], 1e-12 * std::abs(one_rank.beam_load[i]) + 1e-13)
        << "beam_load[" << i << "]: the declared rank count must not change the physics";
  }
  for (std::size_t j = 0; j < one_rank.fluid_in.size(); ++j) {
    EXPECT_NEAR(two_ranks.fluid_in[j], one_rank.fluid_in[j], 1e-13) << "fluid_in[" << j << "]";
    EXPECT_NEAR(two_ranks.fluid_in[j], 0.3 + 2.0 * cell_x(j, 11), 1e-13)
        << "H is still exact for a linear field at cell " << j << " on the two-rank placement";
  }

  double sent = 0.0, got = 0.0, p_fluid = 0.0, p_beam = 0.0;
  const double h_f = measure_of(11);
  for (std::size_t j = 0; j < 11; ++j) {
    sent += two_ranks.fluid_force[j] * h_f;
    p_fluid += two_ranks.fluid_in[j] * two_ranks.fluid_force[j] * h_f;
  }
  for (std::size_t i = 0; i < 8; ++i) {
    got += two_ranks.beam_load[i];
    p_beam += two_ranks.v_nodes[i] * two_ranks.beam_load[i];
  }
  EXPECT_NEAR(got, sent, 1e-13 * std::abs(sent)) << "H^T lost force on the two-rank placement";
  EXPECT_NEAR(p_beam, p_fluid, 1e-13 * std::abs(p_fluid))
      << "interface power does not balance on the two-rank placement";
}

// A method OTHER than "consistent"/"consistent-transpose" into a nodal
// target on more than one rank is unaffected by round 2: per-placement
// supermesh weights remain unimplemented (KNOWN_ISSUES D5 item 3), so this
// still refuses -- now for a method-specific reason, since the SAME
// declared placement with method "consistent-transpose" (the test above)
// is accepted. `run_consistent_pair`'s own default empty method ("") is
// neither "consistent" nor "consistent-transpose", so build_weights() would
// fall through to the supermesh path, which builds FV<->FV weights only --
// exactly the case this refusal exists for.
TEST(InterfaceTransfer, ARemapIntoANodalFieldOnSeveralRanksViaASupermeshMethodStillRefuses) {
  try {
    (void)run_consistent_pair(7, 11, [](double x) { return x; }, [](double) { return 1.0; },
                              fuse::RankSet{0, 2}, /*force_method=*/"");
    FAIL() << "a two-rank nodal Remap target via a non-consistent method initialised";
  } catch (const std::exception& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("E-RT-REMAP-NODAL-PARALLEL"), std::string::npos) << what;
  }
}

// Nodal-remap track, round 3: apply_nodal_dist_remap's own per-period
// allocation, checked with the SAME `AllocationCounter` instrument
// Runner.AllocatesNothingWhileRunning (test/runtime/runner_test.cpp) uses
// for the ordinary weights-apply path. `Runner.AllocatesNothingWhileRunning`'s
// own harness (`build(HarnessOpts{...})`, runner_test.cpp) has no
// nodal-target-on-several-ranks case at all -- extending IT to cover this
// path would mean teaching that shared, heavily-reused harness a new
// mesh/field/channel shape it has never needed before, a materially bigger
// and riskier change than a second, focused test next to it. This test is
// that second test -- and it does NOT assert zero, unlike its namesake,
// because zero is not the true answer here: `apply_distributed_H`/
// `apply_distributed_HT` (nodal_remap_distributed.hpp) build fresh
// Allgatherv buffers, an unordered_map and (in the H branch) a local-slot
// copy on EVERY call by design, a cost KNOWN_ISSUES.md D8 already named as
// "a distinct, documented cost" in round 2 -- this test MEASURES it (a
// small, constant number per period, not zero, not growing) rather than
// asserting a bound that was never true, and records the measured number
// in KNOWN_ISSUES.md D8 as the concrete baseline a future point-to-point
// implementation should reduce.
TEST(InterfaceTransfer, ApplyNodalDistRemapHasAConstantMeasuredPerPeriodAllocationCount) {
  CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{.name = "beam_mesh", .source = "generated:line:7"});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{.name = "fluid_mesh", .source = "generated:line:11"});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
  const auto beam = g.add_component(
      fuse::ComponentDecl{.name = "beam", .ranks = {0, 2}, .interval = Rational{1}});
  const auto fluid = g.add_component(
      fuse::ComponentDecl{.name = "fluid", .ranks = {0, 1}, .interval = Rational{1}});
  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method("consistent");
  const auto force_out = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_force", .region = r_fluid, .units = u::kNewton,
      .conservation = fuse::Conservation::Extensive, .intent = fuse::Intent::Provides});
  const auto force_in = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_load", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = u::kNewton, .conservation = fuse::Conservation::Extensive,
      .intent = fuse::Intent::Requires});
  const auto fch = g.add_channel("wall_force", force_out, force_in);
  g.channel(fch).remap_method("consistent-transpose").conservative().lag(1)
      .allow_time_level_mismatch("capability test: only transfer accuracy is under test");

  SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    ASSERT_TRUE(backend.load(g.mesh(fuse::MeshId{i})).ok());
  }
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  class Side final : public fuse::Component {
   public:
    Side(std::string name, std::function<void(Side&)> fill)
        : name_(std::move(name)), fill_(std::move(fill)) {}
    [[nodiscard]] std::string name() const override { return name_; }
    fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
      for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
        auto& buf = bufs[g.field(f).name];
        buf.assign(count(g, f), 0.0);
        const auto r = reg.bind(
            f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = buf.size()},
                              fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override {
      fill_(*this);
      return fuse::Result<void>::success();
    }
    std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> count;
    std::map<std::string, std::vector<double>> bufs;

   private:
    std::string name_;
    std::function<void(Side&)> fill_;
  };
  auto b = std::make_unique<Side>("beam", [](Side& me) {
    auto& v = me.bufs["beam_velocity"];
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = static_cast<double>(i) / 7.0;
  });
  auto f = std::make_unique<Side>("fluid", [](Side& me) {
    auto& q = me.bufs["fluid_force"];
    for (double& x : q) x = 1.0;
  });
  const auto nodal = [&](const CouplingGraph& gg, fuse::FieldId fid) -> std::size_t {
    const auto& fs = gg.field(fid);
    const std::size_t cells = fs.region == r_beam ? 7u : 11u;
    return fs.disc == fuse::Discretization::ContinuousGLL ? cells + 1 : cells;
  };
  b->count = nodal;
  f->count = nodal;
  ComponentSet components;
  components.add(std::move(b));
  components.add(std::move(f));
  FieldRegistry registry;
  for (auto& c : components) {
    ASSERT_TRUE(c.register_fields(registry, g).ok());
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  ASSERT_TRUE(runner.initialize(components, registry).ok());

  // Warm-up, outside the counter -- mirrors Runner.AllocatesNothingWhileRunning
  // (runner_test.cpp) exactly: anything the FIRST call allocates lazily (a
  // std::function's heap state, a container grown from empty) is still
  // caught by the counted periods below, since it would already exist by
  // then were this the only allocation site.
  ASSERT_TRUE(runner.run_period(components).ok());

  // MEASURED, NOT ASSERTED ZERO -- and that is the finding this test exists
  // to pin down, not a bug in the TEST. apply_nodal_dist_remap calls
  // apply_distributed_H/apply_distributed_HT (nodal_remap_distributed.hpp)
  // every period, and those functions build fresh Allgatherv buffers, an
  // unordered_map, and (in the H branch) a copy of the local input slot on
  // EVERY call by design -- exactly the "Allgatherv buffers apply_distributed_H/HT
  // allocate per call are a distinct, documented cost" this track's own round-2
  // KNOWN_ISSUES.md D8 entry already named, now measured for the first time
  // rather than only asserted. `DistributedInterpRows` itself (the expensive
  // part: locating every point's containing cell) is genuinely built ONCE, at
  // setup -- what this test actually proves is that the PER-PERIOD count is
  // small, deterministic and does not grow across periods (no leak, nothing
  // accumulating), not that it is zero, because it provably is not.
  std::size_t first_period_allocations = 0;
  {
    const AllocationCounter counter;
    ASSERT_TRUE(runner.run_period(components).ok());
    first_period_allocations = counter.count();
  }
  std::printf("ApplyNodalDistRemapHasAConstantMeasuredPerPeriodAllocationCount: "
             "%zu allocations/period\n", first_period_allocations);
  EXPECT_GT(first_period_allocations, 0u)
      << "expected a NONZERO, measured allocation count from "
         "apply_distributed_H/HT's own per-call buffers -- see this test's own "
         "comment; a zero here would mean this measurement itself is broken, "
         "not that the code got faster";

  std::size_t total_allocations = 0;
  bool ok = false;
  {
    const AllocationCounter counter;
    ok = runner.run(components, 8).ok();
    total_allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  EXPECT_EQ(total_allocations, first_period_allocations * 8)
      << "total=" << total_allocations << ", first_period=" << first_period_allocations
      << " -- the per-period allocation count must be CONSTANT across periods (no "
         "growth, no leak); it is measured here as " << first_period_allocations
      << " allocations/period, not zero (see this test's own comment on why), and "
         "KNOWN_ISSUES.md D8 records this number as the Allgatherv-per-apply cost "
         "a future point-to-point implementation should reduce, not silently accept "
         "as unmeasurable";
}

// Moving least squares through the Runner (docs/user/point-cloud-remap.md):
// beam nodal velocity to fluid cells with a degree-1 fit, which reproduces a
// linear field exactly, on the same serial path every other method takes.
TEST(InterfaceTransfer, MlsVelocityChannelRunsThroughTheRunnerAndIsExactForLinear) {
  NodalResult r;
  try {
    r = run_consistent_pair(7, 11, [](double x) { return 0.3 + 2.0 * x; },
                            [](double) { return 1.0; }, fuse::RankSet{0, 1},
                            "consistent-transpose", "mls:degree=1,k=4,radial=wendland2");
  } catch (const std::exception& e) {
    FAIL() << e.what();
  }
  ASSERT_EQ(r.fluid_in.size(), 11u);
  for (std::size_t j = 0; j < 11; ++j)
    EXPECT_NEAR(r.fluid_in[j], 0.3 + 2.0 * cell_x(j, 11), 1e-12) << "cell " << j;
}

// MLS is serial: a partitioned cloud gives each rank different neighbours.
// Refused from the declared placement, before any weights are built.
TEST(InterfaceTransfer, AnMlsChannelOnSeveralRanksIsRefused) {
  try {
    (void)run_consistent_pair(7, 11, [](double x) { return x; }, [](double) { return 1.0; },
                              fuse::RankSet{0, 2}, "consistent-transpose",
                              "mls:degree=1,k=4,radial=wendland2");
    FAIL() << "an MLS channel from a two-rank component initialised";
  } catch (const std::exception& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("E-RT-REMAP-MLS-PARALLEL"), std::string::npos) << what;
  }
}
