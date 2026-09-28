// Runner + CommGraph + MpiTransport under a real mpiexec. Not a
// fuse_add_mpi_test(): it calls MPI_Init, must be launched by mpiexec, and must
// fail the job rather than return on a per-rank failure.

#include <gtest/gtest.h>
#include <mpi.h>

// R-50 (REVIEW.md): the documented rank-requirement convention
// (docs/dev/testing.md).
#include "fuse_test_main.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fuse/model/component.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>

#include "../support/allocation_counter.hpp"

using fuse::Rational;
using fuse::Runner;

int world_size() {
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

int current_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

// PartitionedTestMesh / PartitionedTestBackend moved verbatim to
// test/support/partitioned_test_backend.hpp (2026-09-23) so
// examples/layout_driver can share the one per-placement decomposition
// instead of carrying a second, drifting copy. See that header for the
// class bodies; nothing about their behaviour changed.
#include "../support/partitioned_test_backend.hpp"


/// Registers one field of `count` doubles, nothing more. `build_ticks()`
/// (runner.cpp:1146-1164) requires a `Component` object for EVERY graph
/// component on EVERY rank regardless of that component's `RankSet` -- so
/// both "fluid" and "solid" must exist here even on a rank that physically
/// belongs to only one of them.
class StubComponent final : public fuse::Component {
 public:
  /// `rollback`: R-10 (REVIEW.md). An iteration region requires every
  /// component it checkpoints to support rollback (E-ITER-NOROLLBACK);
  /// default false matches every existing caller, none of which puts this
  /// component in a region.
  /// N-transport-runtime-2 (REVIEW.md pass2): fail_after -- -1 (the
  /// default, every existing caller) never refuses; 0 refuses starting with
  /// this component's very FIRST advance() call, the shape
  /// RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi needs
  /// to force an iteration region's re-advance to fail on sweep 0 under a
  /// real multi-rank run.
  StubComponent(std::string name, fuse::FieldId field, std::size_t count,
               bool rollback = false, int fail_after = -1)
      : name_(std::move(name)), field_(field), storage_(count, 0.0),
        rollback_(rollback), fail_after_(fail_after) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph&) override {
    fuse::Layout layout;
    layout.count = storage_.size();
    return registry.bind(
        field_,
        fuse::FieldRef{storage_.data(), layout, fuse::MemorySpace::Host});
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    // `calls_` counts EVERY advance, so `calls()` means what its name says.
    // It used to be incremented inside the condition below, to the right of
    // `fail_after_ >= 0 &&` -- so for a stub with no failure configured
    // (`fail_after_ == -1`) short-circuit evaluation skipped the increment
    // entirely and `calls()` was stuck at 0 however often the component ran.
    // StillRunsASecondChannelAfterARegionFailsUnderMpi asserts
    // `cons_b->calls() >= 1` on exactly such a stub, so it could not pass
    // whatever the runtime did; four separate real defects were found and
    // fixed against it before the counter itself was read.
    //
    // The comparison keeps the pre-increment value, so `fail_after_`'s
    // meaning is unchanged: with 0 the first call fails, with 1 the second.
    const int n = calls_++;
    if (fail_after_ >= 0 && n >= fail_after_) {
      return fuse::Failure{"E-TEST-ADVANCE-REFUSED",
                          "'" + name_ + "' refuses advance() past call " +
                              std::to_string(fail_after_)};
    }
    return fuse::Result<void>::success();  // the test writes storage directly
  }

  fuse::Result<void> save_state() override {
    if (rollback_) saved_ = storage_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    if (rollback_) storage_ = saved_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return rollback_ ? fuse::kCapabilityRollback : 0; }

  [[nodiscard]] double*     data() noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }
  [[nodiscard]] int         calls() const noexcept { return calls_; }

 private:
  std::string          name_;
  fuse::FieldId        field_;
  std::vector<double>  storage_;
  bool                 rollback_;
  std::vector<double>  saved_;
  int                  fail_after_ = -1;
  int                  calls_ = 0;
};

/// Everything a test needs: the graph, the plan, the backend, and the two
/// stub components, plus helpers that read/write through the SAME region both
/// components' fields reference (Transport preserves region -- see the plan's
/// Scope Decision -- so there is exactly one to read).
/// How the source and destination placements relate (matrix rows 1-6 at the
/// CommGraph level; this is the same three shapes through the Runner).
enum class Overlap {
  Disjoint,  ///< fluid on the first half, solid on the rest -- the default
  Partial,   ///< fluid on 0..W-2, solid on 1..W-1: shared ranks 1..W-2 plus
             ///< one rank unique to each side; genuinely partial from W = 3
  Full,      ///< both on every rank: identical placements, no Transport
};

/// Where the channel's Remap runs (RemapSpec::remap_on): on the target, which
/// is today's default and emits ONE Transport, or on a separately placed
/// coupler, which emits fluid -> coupler and coupler -> solid with either hop
/// elided when the placements coincide.
enum class CouplerShape {
  None,            ///< no coupler declared: remap on the target
  Disjoint,        ///< the coupler is the LAST rank alone; fluid and solid
                   ///< share the rest as Overlap::Disjoint over W-1 ranks
  EqualsTarget,    ///< the coupler's ranks are exactly solid's: the second
                   ///< hop is elided and the result must equal no-coupler
  OverlapsSource,  ///< the coupler is fluid's last rank plus solid's first:
                   ///< identical to neither side, sharing a rank with each
};

[[nodiscard]] fuse::RankSet placement_for(Overlap shape, bool is_fluid,
                                          CouplerShape coupler = CouplerShape::None) {
  // With a DISJOINT coupler the last rank is the coupler's alone, and the
  // two physics components split the remaining W-1 ranks.
  const int w = world_size() - (coupler == CouplerShape::Disjoint ? 1 : 0);
  switch (shape) {
    case Overlap::Disjoint:
      return is_fluid ? fuse::RankSet{0, w / 2} : fuse::RankSet{w / 2, w - w / 2};
    case Overlap::Partial:
      return is_fluid ? fuse::RankSet{0, w - 1} : fuse::RankSet{1, w - 1};
    case Overlap::Full:
      return fuse::RankSet{0, w};
  }
  return fuse::RankSet{0, 0};
}

[[nodiscard]] fuse::RankSet coupler_placement(CouplerShape coupler,
                                              const fuse::RankSet& fluid,
                                              const fuse::RankSet& solid) {
  switch (coupler) {
    case CouplerShape::None:
      return fuse::RankSet{0, 0};
    case CouplerShape::Disjoint:
      return fuse::RankSet{world_size() - 1, 1};
    case CouplerShape::EqualsTarget:
      return solid;
    case CouplerShape::OverlapsSource: {
      // fluid's last rank and solid's first: for Overlap::Disjoint these are
      // adjacent, so the coupler is the two ranks straddling the boundary.
      const int lo = std::min(fluid.max_rank(), solid.min_rank());
      const int hi = std::max(fluid.max_rank(), solid.min_rank());
      return fuse::RankSet{lo, hi - lo + 1};
    }
  }
  return fuse::RankSet{0, 0};
}

class TwoComponentJob {
 public:
  /// `entity_set`, when non-empty, is set on the shared region. Task 3's
  /// E-RT-TRANSPORT-REGION test passes "wall"; every other test passes "".
  /// `with_iteration_region`: R-10 (REVIEW.md) -- wraps the "link" channel
  /// in an iteration region with a real convergence measure, and makes both
  /// components rollback-capable so E-ITER-NOROLLBACK does not mask the
  /// check this exists to reach. Default false leaves every existing caller
  /// of this class exactly as it was. `accel_kind`: which accelerator the
  /// region declares; only read when `with_iteration_region` is true.
  /// AccelKind::None (the default, and IterationRegionDecl's own) is one of
  /// the two kinds R-10's fix genuinely supports for a region spanning more
  /// than one rank; a dynamic kind (Aitken and stronger) is the case it
  /// still refuses -- see RefusesADynamicAcceleratorOnACrossRankRegion.
  /// `with_independent_channel`: N-transport-runtime-2 (REVIEW.md pass2) --
  /// adds a SECOND channel, "linkB", between two more components on the SAME
  /// fluid_/solid_ ranks and the SAME shared region, but NOT one of the
  /// region's channels. Default false leaves every existing caller
  /// unchanged. `fail_fluid_after`: forwarded to "fluid"'s StubComponent
  /// (see its own doc); -1 (default) never refuses.
  /// `independent_interval`: D3 (KNOWN_ISSUES) -- producer_b/consumer_b's own
  /// clock. Default `Rational{1}` (one tick per period, unchanged from every
  /// prior caller). `StillRunsASecondChannelAfterARegionFailsUnderMpi` passes
  /// `Rational{1,2}` (two ticks per period) so its SECOND tick sits at
  /// `t=1/2`, strictly later than anything at `t=0` by the schedule's own
  /// (time, phase, rank) sort key (`plan.hpp`'s own contract) -- not by any
  /// same-instant tie-break `group_regions_within_instants()` might resolve
  /// either way. A survivor abandoning "the rest of the period" the moment
  /// the region fails abandons that `t=1/2` tick regardless of where the
  /// greedy grouping would have placed a same-time observable.
  explicit TwoComponentJob(const std::string& entity_set,
                           bool with_iteration_region = false,
                           fuse::AccelKind accel_kind = fuse::AccelKind::None,
                           bool with_independent_channel = false,
                           int fail_fluid_after = -1,
                           Overlap shape = Overlap::Disjoint,
                           fuse::Rational independent_interval = fuse::Rational{1},
                           CouplerShape coupler = CouplerShape::None)
      // MEMBER DECLARATION ORDER, NOT INITIALIZER-LIST ORDER, decides when
      // each of these actually runs. `backend` is declared below
      // fluid_/solid_/n_entities_/dim_ for exactly this reason: it is built
      // FROM them, so they must already hold their values.
      : fluid_{placement_for(shape, true, coupler)},
        solid_{placement_for(shape, false, coupler)},
        n_entities_(12), dim_(2),
        backend(fluid_, solid_, n_entities_, dim_) {
    const auto mesh =
        graph.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
    region_ = graph.add_region(fuse::Region{.name = "r", .mesh = mesh,
                                            .entity_set = entity_set,
                                            .dimension = dim_});

    fuse::FieldId src_field_b, dst_field_b;

    const auto pf = graph.add_component(fuse::ComponentDecl{
        .name = "fluid", .ranks = fluid_, .interval = Rational{1}});
    const auto sc = graph.add_component(fuse::ComponentDecl{
        .name = "solid", .ranks = solid_, .interval = Rational{1}});
    src_field_ = graph.add_field(
        pf, fuse::FieldSpec{.name = "out", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Provides});
    dst_field_ = graph.add_field(
        sc, fuse::FieldSpec{.name = "in", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Requires});
    const auto channel = graph.add_channel("link", src_field_, dst_field_);
    if (coupler != CouplerShape::None) {
      // Both fields share one region, so nothing would emit a Remap for this
      // channel and remap_on would be W-CHANNEL-REMAP-ON-UNUSED; Conservative
      // asks for one unconditionally, exactly as the layout driver does. The
      // coupler is declared AFTER fluid and solid, so ComponentId 0 and 1
      // stay fluid and solid for every helper in this class. (No caller
      // combines a coupler with with_independent_channel; producer_b and
      // consumer_b would then be ids 3 and 4, and size_of() below says 2, 3.)
      coupler_ = coupler_placement(coupler, fluid_, solid_);
      const auto cid = graph.add_coupler(
          fuse::CouplerDecl{.name = "coupler", .ranks = coupler_});
      graph.channel(channel).remap(fuse::RemapFlags::Conservative).remap_on(cid);
    }


    if (with_iteration_region) {
      fuse::ConvergenceCriterion conv;
      conv.add(fuse::Converged::Relative(1e-6, "link"));
      graph.add_iteration_region(fuse::IterationRegionDecl{
          .name = "outer", .channels = {channel}, .max_iterations = 4,
          .convergence = conv,
          .acceleration = fuse::AccelConfig{.kind = accel_kind}});
    }

    // Declared AFTER the region, which is the natural order and which the
    // scheduler now honours: `group_regions_within_instants()` keeps the
    // region's body contiguous whatever order its neighbour was declared in
    // (KNOWN_ISSUES D1, fixed 2026-09-21). This fixture used to declare it
    // FIRST to work around that, and the workaround is why three mutations
    // survived: with the independent channel ahead of the region, everything
    // this test observes happened before the failure, so "the rest of the
    // period" was empty and R-09's mutant skipped nothing (D3).
    if (with_independent_channel) {
      const auto pb = graph.add_component(fuse::ComponentDecl{
          .name = "producer_b", .ranks = fluid_,
          .interval = independent_interval});
      const auto cb = graph.add_component(fuse::ComponentDecl{
          .name = "consumer_b", .ranks = solid_,
          .interval = independent_interval});
      src_field_b = graph.add_field(
          pb, fuse::FieldSpec{.name = "outB", .region = region_,
                             .units = fuse::units::kKelvin,
                             .intent = fuse::Intent::Provides});
      dst_field_b = graph.add_field(
          cb, fuse::FieldSpec{.name = "inB", .region = region_,
                             .units = fuse::units::kKelvin,
                             .intent = fuse::Intent::Requires});
      graph.add_channel("linkB", src_field_b, dst_field_b);
    }

    // Every component's own decomposition (KNOWN_ISSUES D5), built before
    // the load that lays them out.
    std::vector<fuse::RankSet> placements;
    for (std::uint32_t c = 0; c < graph.component_count(); ++c) {
      placements.push_back(graph.component(fuse::ComponentId{c}).ranks);
    }
    backend.set_placements(std::move(placements));
    EXPECT_TRUE(backend.load(graph.mesh(mesh)).ok());

    const auto lowered = lower(graph);
    EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
    dag = lowered.dag;

    const fuse::ClockSet clocks{graph};
    const auto sched =
        schedule(graph, dag, clocks, fuse::make_region_sizes(graph, backend));
    EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
    plan = sched.plan;

    // Each component's storage is ITS OWN decomposition's size: on a rank in
    // both partially overlapping placements the two differ. (Component ids
    // follow declaration order: fluid 0, solid 1, producer_b 2, consumer_b 3.)
    // NOTE: ASSERT_TRUE cannot be used here -- it expands to `return ...;`,
    // which only compiles in a void-returning function, and a constructor is
    // not one. EXPECT_TRUE matches every other check in this constructor.
    auto size_of = [&](std::uint32_t cid) -> std::size_t {
      const auto n = backend.placed_region_size(graph.region(region_),
                                                fuse::ComponentId{cid});
      EXPECT_TRUE(n.ok());
      return n.ok() ? n.value() : 0;
    };
    components.add(std::make_unique<StubComponent>(
        "fluid", src_field_, size_of(0), with_iteration_region,
        fail_fluid_after));
    components.add(std::make_unique<StubComponent>(
        "solid", dst_field_, size_of(1), with_iteration_region));
    if (with_independent_channel) {
      components.add(std::make_unique<StubComponent>(
          "producer_b", src_field_b, size_of(2)));
      components.add(std::make_unique<StubComponent>(
          "consumer_b", dst_field_b, size_of(3)));
    }
  }

  void write_source_by_global_id() {
    auto* src = dynamic_cast<StubComponent*>(components.find("fluid"));
    const auto v =
        backend.placed_view(graph.region(region_).mesh, fuse::ComponentId{0})
            .value();
    for (std::size_t i = 0; i < src->size(); ++i) {
      src->data()[i] = static_cast<double>(v->global_id(dim_, i));
    }
  }

  [[nodiscard]] std::size_t count_destination_matches() const {
    // `fluid` and `solid` share one region and one backend view per rank
    // (the class comment above explains why), so on a rank that belongs to
    // `fluid_` this same view is FLUID's local block, not solid's -- the
    // "solid" arc's storage there is never written by a correct Transport
    // (its scatter map is empty; see attach_comm_graphs()) and comparing it
    // against this rank's own (fluid-side) global ids would fail for reasons
    // that have nothing to do with delivery correctness. Only a rank actually
    // in `solid_`'s RankSet is a real destination worth checking.
    if (!solid_.contains(current_rank())) return 0;
    auto* dst = dynamic_cast<StubComponent*>(
        const_cast<fuse::ComponentSet&>(components).find("solid"));
    const auto v =
        backend.placed_view(graph.region(region_).mesh, fuse::ComponentId{1})
            .value();
    std::size_t ok = 0;
    for (std::size_t i = 0; i < dst->size(); ++i) {
      if (dst->data()[i] == static_cast<double>(v->global_id(dim_, i))) ++ok;
    }
    return ok;
  }

  [[nodiscard]] std::size_t local_destination_entity_count() const {
    // See count_destination_matches(): a rank outside solid_ owns zero
    // destination entities, even though backend.region_size() would report
    // its own (fluid-side) count instead, absent this guard.
    if (!solid_.contains(current_rank())) return 0;
    return backend
        .placed_region_size(graph.region(region_), fuse::ComponentId{1})
        .value();
  }
  [[nodiscard]] std::size_t global_destination_entity_count() const {
    return n_entities_;  // every id is wanted exactly once
  }

  /// The same three helpers as above, for "linkB" (producer_b -> consumer_b,
  /// ComponentId{2}/{3}) instead of "link" (fluid -> solid, ComponentId{0}/
  /// {1}). Only meaningful when `with_independent_channel` was true; every
  /// caller of these three checks `find("producer_b")`/`find("consumer_b")`
  /// is non-null first via ASSERT_NE, so a null dereference here would mean
  /// the fixture itself was misconfigured, not a Transport defect.
  void write_source_b_by_global_id() {
    auto* src = dynamic_cast<StubComponent*>(components.find("producer_b"));
    const auto v =
        backend.placed_view(graph.region(region_).mesh, fuse::ComponentId{2})
            .value();
    for (std::size_t i = 0; i < src->size(); ++i) {
      src->data()[i] = static_cast<double>(v->global_id(dim_, i));
    }
  }

  [[nodiscard]] std::size_t count_destination_b_matches() const {
    // See count_destination_matches(): only a rank actually in `solid_` (the
    // RankSet consumer_b was placed on) is a real destination worth
    // checking.
    if (!solid_.contains(current_rank())) return 0;
    auto* dst = dynamic_cast<StubComponent*>(
        const_cast<fuse::ComponentSet&>(components).find("consumer_b"));
    const auto v =
        backend.placed_view(graph.region(region_).mesh, fuse::ComponentId{3})
            .value();
    std::size_t ok = 0;
    for (std::size_t i = 0; i < dst->size(); ++i) {
      if (dst->data()[i] == static_cast<double>(v->global_id(dim_, i))) ++ok;
    }
    return ok;
  }

  [[nodiscard]] std::size_t local_destination_b_entity_count() const {
    if (!solid_.contains(current_rank())) return 0;
    return backend
        .placed_region_size(graph.region(region_), fuse::ComponentId{3})
        .value();
  }

  // Declared in the order the constructor's initializer list must run them
  // (see the constructor comment): fluid_/solid_/n_entities_/dim_, then
  // `backend`, which is built from them. `graph`/`dag`/`plan`/`components`/
  // `registry` are default-constructed and filled in by the constructor
  // body, so their position relative to the others does not matter.
  fuse::RankSet          fluid_, solid_;
  fuse::RankSet          coupler_{0, 0};  ///< empty unless a CouplerShape was asked for
  std::size_t            n_entities_ = 0;
  int                    dim_ = 0;
  PartitionedTestBackend backend;

  fuse::CouplingGraph  graph;
  fuse::OperatorDAG    dag;
  fuse::ExecutionPlan  plan;
  fuse::ComponentSet   components;
  fuse::FieldRegistry  registry;

 private:
  fuse::RegionId region_;
  fuse::FieldId  src_field_, dst_field_;
};

TwoComponentJob make_two_component_job(const std::string& entity_set) {
  return TwoComponentJob{entity_set};
}

/// Bind every component's storage; the Runner refuses a plan otherwise. Not
/// part of the brief's TwoComponentJob text, but required by the same
/// contract test/unit/runner_test.cpp:399-404 documents (`bind_all`): nothing
/// in Runner::initialize() calls Component::register_fields() for you.
void bind_all(TwoComponentJob& h) {
  for (auto& c : h.components) {
    EXPECT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

// ---------------------------------------------------------------------------
// R-10 (REVIEW.md): a genuine two-way iterated coupling, for the positive
// convergence test below. TwoComponentJob's one-way "link" channel has
// nothing to iterate on -- the whole point here is a fixed point with real
// feedback, split across ranks with an UNEQUAL local partition.
// ---------------------------------------------------------------------------

/// out[i] = a + k * in[i]. Mirrors test/verification/iteration_test.cpp's
/// LinearBlock exactly (same role, same save/restore contract) so the same
/// a/k coefficients and entity count there give this fixture the identical
/// fixed point and contraction rate: the map is applied entity-wise with no
/// coupling BETWEEN entities, so splitting the same entities across ranks
/// changes nothing about either, only how the Euclidean residual is split
/// into local contributions before reduce_region_residuals() reduces them
/// back together.
class LinearRelaxComponent final : public fuse::Component {
 public:
  LinearRelaxComponent(std::string name, double a, double k, std::size_t n)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == fuse::Intent::Provides;
      std::vector<double>& buf = is_out ? out_ : in_;
      fuse::Layout layout;
      layout.count = buf.size();
      auto r = reg.bind(
          f, fuse::FieldRef{buf.data(), layout, fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = a_ + k_ * in_[i];
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    for (std::size_t i = 0; i < out_.size(); ++i) out_[i] = saved_out_[i];
    for (std::size_t i = 0; i < in_.size(); ++i) in_[i] = saved_in_[i];
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override { return fuse::kCapabilityRollback; }

  [[nodiscard]] const std::vector<double>& out() const { return out_; }

 private:
  std::string          name_;
  double               a_, k_;
  std::vector<double>  out_, in_, saved_out_, saved_in_;
};

/// `A`/`B` both span the WHOLE communicator (unlike TwoComponentJob's
/// disjoint halves), so every rank owns a slice of both components and
/// `resolve_regions()`'s participant set is every rank in the job.
/// `kEntities == 4` does not divide evenly at 3 ranks (2, 1, 1) and leaves a
/// spare at 4+ ranks too, so every rank count this fixture is used at
/// exercises an unequal local split.
class LinearFixedPointJob {
 public:
  static constexpr std::size_t kEntities = 4;

  /// `kA`/`tol`: D3 (KNOWN_ISSUES) -- `RegionResidualLocalConvergenceIsNot-
  /// EnoughUnderMpi` passes a `kA` CHOSEN PER RANK (`current_rank() == 0 ?
  /// fast : slow`) so this rank's own LOCAL residual and the region's
  /// GLOBALLY-AGREED residual reach different convergence verdicts at the
  /// same sweep -- see that test's own comment for the numbers. Every
  /// existing caller passes neither, keeping `kA = 0.5`/`tol = 1e-10`
  /// (this class's own long-standing values) exactly as before.
  explicit LinearFixedPointJob(int max_iterations, double kA = 0.5,
                               double tol = 1e-10)
      : whole_{0, world_size()}, backend(whole_, whole_, kEntities, 1) {
    const auto mesh =
        graph.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
    region_ = graph.add_region(
        fuse::Region{.name = "r", .mesh = mesh, .dimension = 1});

    const auto ca = graph.add_component(fuse::ComponentDecl{
        .name = "A", .ranks = whole_, .interval = Rational{1}});
    const auto cb = graph.add_component(fuse::ComponentDecl{
        .name = "B", .ranks = whole_, .interval = Rational{1}});
    const auto a_out = graph.add_field(
        ca, fuse::FieldSpec{.name = "a_out", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Provides});
    const auto a_in = graph.add_field(
        ca, fuse::FieldSpec{.name = "a_in", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Requires});
    const auto b_in = graph.add_field(
        cb, fuse::FieldSpec{.name = "b_in", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Requires});
    const auto b_out = graph.add_field(
        cb, fuse::FieldSpec{.name = "b_out", .region = region_,
                           .units = fuse::units::kKelvin,
                           .intent = fuse::Intent::Provides});

    const auto fwd = graph.add_channel("a2b", a_out, b_in);
    const auto back = graph.add_channel("b2a", b_out, a_in);
    graph.channel(back).lag(1).allow_time_level_mismatch("iterated");

    fuse::ConvergenceCriterion conv;
    conv.add(fuse::Converged::Relative(tol, "a2b"));
    conv.add(fuse::Converged::Relative(tol, "b2a"));
    // .acceleration defaults to AccelKind::None (plain Picard) -- the only
    // kind R-10's fix supports for a region spanning more than one rank
    // (REVIEW.md; see resolve_regions()'s accelerator-kind check), and the
    // kind test/verification/iteration_test.cpp's
    // AitkenNeedsNoMoreIterationsThanPicard measured at 30 iterations for
    // this exact pair of coefficients, entity count and tolerance.
    graph.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {fwd, back},
        .max_iterations = max_iterations, .convergence = conv});

    EXPECT_TRUE(backend.load(graph.mesh(mesh)).ok());
    const auto lowered = lower(graph);
    EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
    dag = lowered.dag;

    const fuse::ClockSet clocks{graph};
    const auto sched =
        schedule(graph, dag, clocks, fuse::make_region_sizes(graph, backend));
    EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
    plan = sched.plan;

    const auto count = backend.region_size(graph.region(region_));
    EXPECT_TRUE(count.ok());
    components.add(
        std::make_unique<LinearRelaxComponent>("A", 1.0, kA, count.value()));
    components.add(
        std::make_unique<LinearRelaxComponent>("B", 2.0, 0.4, count.value()));
  }

  fuse::RankSet          whole_;
  PartitionedTestBackend backend;

  fuse::CouplingGraph  graph;
  fuse::OperatorDAG    dag;
  fuse::ExecutionPlan  plan;
  fuse::ComponentSet   components;
  fuse::FieldRegistry  registry;

 private:
  fuse::RegionId region_;
};

void bind_all(LinearFixedPointJob& h) {
  for (auto& c : h.components) {
    EXPECT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

/// bind_all() with one component deliberately left out, so this rank's
/// resolve_buffers() fails with E-RT-UNBOUND while every other rank's
/// succeeds. Used only by ARankDivergentSetupFailureRefusesOnEveryRank.
void bind_all_except(TwoComponentJob& h, std::string_view skip) {
  for (auto& c : h.components) {
    if (c.name() == skip) continue;
    EXPECT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

TEST(RunnerTransport, BaseCommunicatorIsOptionalAndDefaultsToUnset) {
  const fuse::RunnerConfig cfg;
  EXPECT_FALSE(cfg.base_communicator.has_value())
      << "an unset base_communicator is what every existing caller relies on";

  fuse::RunnerConfig with;
  with.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  ASSERT_TRUE(with.base_communicator.has_value());
  EXPECT_EQ(*with.base_communicator, MPI_Comm_c2f(MPI_COMM_WORLD));
}

// The split is internal to attach_comm_graphs(), so this checks the property
// it must have rather than reaching into it: colouring by RankSet::contains
// puts exactly the component's ranks in its communicator, and a rank in no
// component gets MPI_COMM_NULL.
TEST(RunnerTransport, SplittingByRankSetReproducesComponentMembership) {
  // Was a hard ASSERT_GE(world, 2), which held only because this file was
  // registered from 2 ranks up. Adding rank 1 (S5.5) falsified that premise
  // and the assertion fired as designed; a skip states the requirement where
  // the reader is, and does not depend on the registration staying put.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int world = 0, me = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // Two disjoint components spanning the job, as spec §6 mode 1 describes.
  const fuse::RankSet src{0, world / 2};
  const fuse::RankSet dst{world / 2, world - world / 2};
  EXPECT_FALSE(src.overlaps(dst));

  MPI_Comm c = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, src.contains(me) ? 0 : MPI_UNDEFINED, me, &c);
  if (src.contains(me)) {
    ASSERT_NE(c, MPI_COMM_NULL);
    int sz = 0;
    MPI_Comm_size(c, &sz);
    EXPECT_EQ(sz, static_cast<int>(src.size()));
    MPI_Comm_free(&c);
  } else {
    EXPECT_EQ(c, MPI_COMM_NULL);
  }
}

TEST(RunnerTransport, ASubSetRegionIsRefusedRatherThanGuessed) {
  // MeshBackend can count a region's entities but not enumerate them, so a
  // buffer for `entity_set: wall` cannot be said to be indexed by set position
  // or by mesh-local index. remap_plan.hpp:356-363 already refuses this with
  // E-REMAP-REGION for the same reason; a Transport that guessed would deliver
  // a permutation of the right values, which is worse than a refusal because
  // it looks like an answer.
  auto h = make_two_component_job(/*entity_set=*/"wall");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-TRANSPORT-REGION");
}

// Defined below, with its rationale.
std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup();

TEST(RunnerTransport, AWholeMeshRegionIsNotRefused) {
  // The control for the test above: without it, a guard that refused
  // everything would pass. Same configuration as that test -- the mpi
  // transport over the base communicator -- so the entity set is the ONLY
  // difference between the two.
  //
  // It used the "copy" transport with no communicator until 2026-09-23. That
  // passed only while both ends of the Transport shared one mesh view; with
  // each end in its own component's decomposition it copies fluid's local
  // block whole into solid's, which on a rank in one placement and not the
  // other are different sizes. That is refused (E-RT-SIZE) and pinned by the
  // test below, so this control no longer rides on it.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);  // two disjoint placements
  auto h = make_two_component_job(/*entity_set=*/"");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto r = runner.initialize(h.components, h.registry, cfg);
  EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

TEST(RunnerTransport, AWholeArcCopyBetweenDifferentDecompositionsIsRefused) {
  // With no communicator there is no CommGraph, so the Transport moves its
  // whole input arc into its whole output arc. Across two DISJOINT
  // placements those are two components' decompositions of the region --
  // 12 entities and 0 on a rank in one of them -- and an element-wise copy
  // would put one component's local values into another's slots. The
  // refusal is check_transport_extents()'s whole-arc-copy branch.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job(/*entity_set=*/"");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport = "copy";
  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_FALSE(r.ok())
      << "copying one component's local block into another's is not a "
         "transfer between their decompositions";
  EXPECT_EQ(r.error().code, "E-RT-SIZE") << r.error().message;
  EXPECT_NE(r.error().message.find("without a CommGraph"), std::string::npos)
      << r.error().message;
}

// RunnerConfig::transport_setup is the existing hook for a backend's
// one-time, concrete-typed setup (runner.hpp; invoked at runner.cpp:1571-1576).
// build_plan cannot go here -- it needs the spec, which does not exist until
// attach_comm_graphs has run -- so this supplies only the communicator.
std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) {
      return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    }
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

// R-10 (REVIEW.md). Originally: an iteration region's convergence decision
// was computed from a purely rank-local residual with no cross-rank
// agreement, and ANY region spanning more than one rank was refused
// (E-RT-REGION-MPI) as unverified. Superseded: resolve_regions() and
// run_region() now reduce the residual across the region's participants on
// every sweep (reduce_region_residuals(), src/runtime/runner.cpp), so a
// region using AccelKind::None or AccelKind::Constant is genuinely
// supported -- see RegionConvergesInTheSerialIterationCount below, which is
// exactly the TwoComponentJob("", true) scenario this test used to refuse.
//
// What is NOT supported: a DYNAMIC accelerator (Aitken and stronger)
// derives its relaxation coefficient from an inner product over this
// rank's own local iterate history, computed inside fuse::num
// (src/num/accel_quasi_newton.cpp, accel_relaxation.cpp's Aitken branch) --
// outside this fixer's owned paths, and Accelerator::update()'s signature
// has no hook for a caller-supplied global reduction. Two ranks with
// different local slices would compute different coefficients from that
// inner product and take different accelerator paths even with the
// residual itself correctly agreed, reopening the desynchronisation risk
// R-10 was about. `fluid`/`solid` are on disjoint rank sets (as in every
// other TwoComponentJob test), so this still needs `FUSE_REQUIRE_AT_LEAST_
// RANKS(2)` to reach a real cross-rank configuration; the sandbox this was
// written in cannot launch MPI at all (every test here dies in
// MPI_Init_thread), so it is recorded verified only by the tier 1 BUILD
// succeeding, and needs a host run to execute.
TEST(RunnerTransport, RefusesADynamicAcceleratorOnACrossRankRegion) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = TwoComponentJob("", /*with_iteration_region=*/true,
                           fuse::AccelKind::Aitken);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_FALSE(r.ok())
      << "fluid and solid are on disjoint ranks, a real communicator is "
         "configured, and the region names AccelKind::Aitken -- the one "
         "case R-10's reduction fix still refuses";
  EXPECT_EQ(r.error().code, "E-RT-REGION-MPI") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// N-transport-runtime-2 / R-09 (REVIEW.md pass2, re-transport-runtime.md):
// what R-09's fix (step_period(), src/runtime/runner.cpp) is actually FOR --
// a region failure must not abandon a LATER, unrelated channel's collective
// instructions in the same period. Under Tier 0 (no MPI) that abandonment is
// unobservable: nothing is waiting on the other end. Under a real multi-rank
// run it is exactly the hang rank-agreement.md exists to prevent -- "fluid"'s
// rank(s) return out of step_period() the moment the region fails (pre-R-09
// behaviour) while "consumer_b"'s rank(s), on the disjoint `solid_` set,
// still walk forward to "linkB"'s Epoch and block waiting for a partner that
// already left.
//
// D1 (KNOWN_ISSUES, fixed 2026-09-21) means "linkB" no longer has to be
// declared before the region to survive scheduling -- `with_independent_
// channel`'s block is declared AFTER `add_iteration_region` now (see the
// class comment above) -- but D3's own history is that this declaration
// reorder ALONE did not kill the three survivors on the host: producer_b/
// consumer_b's own component TICK still lands at the same `t=0` the region
// occupies, and `group_regions_within_instants()` only reorders the
// INSTRUCTION stream inside one (time, phase) window -- it says nothing
// about which same-time observable an unrelated reader happens to see, and
// a test that depends on winning that tie-break is a test that depends on
// an implementation detail the mutation is not really about.
//
// So this fixture puts `producer_b`/`consumer_b` on their OWN clock,
// `Rational{1,2}` -- two ticks per period, at `t=0` and `t=1/2` -- instead of
// sharing the region's `t=0`. `schedule()`'s own contract (`plan.hpp`:
// "schedule() orders by (time, TickPhase, topological rank..., NodeId)")
// means the SECOND tick, at `t=1/2`, is placed strictly after every `t=0`
// instruction by the sort key itself, not by any greedy preference that
// could go either way. Under the fix, `step_period()` carries the region's
// failure forward and keeps walking the stream, so both ticks of producer_b/
// consumer_b still run (their own field has nothing to do with the region,
// so `block_advances_from()` never reaches their slot) and "linkB"'s
// Transport+Epoch at `t=1/2` complete on every participant. Reverting R-09's
// fix makes `step_period()` return the instant the region's `run_region()`
// reports failure, at `t=0` -- strictly BEFORE `t=1/2` -- so consumer_b's
// second tick, and "linkB"'s second exchange, are abandoned outright: no
// hang is even needed for the mutant to be visibly wrong here, though a
// version of this shape that put producer_b/consumer_b on the SAME `t=0`
// clock would still hang the way the ctest TIMEOUT property (matching the
// existing mutation entries in build-evidence/mutations/transport-runtime.
// toml) is there to catch.
//
// With the fix, every rank -- both the ones that saw "fluid" fail and the
// ones that never did -- must return from run_period() with the SAME
// reported failure, and consumer_b's rank(s) must show BOTH of their own
// ticks having run.
TEST(RunnerTransport, StillRunsASecondChannelAfterARegionFailsUnderMpi) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = TwoComponentJob("", /*with_iteration_region=*/true,
                           fuse::AccelKind::None,
                           /*with_independent_channel=*/true,
                           /*fail_fluid_after=*/0, /*shape=*/Overlap::Disjoint,
                           /*independent_interval=*/fuse::Rational{1, 2});
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  auto* cons_b = dynamic_cast<StubComponent*>(h.components.find("consumer_b"));
  ASSERT_NE(cons_b, nullptr);

  // PRECONDITION of the kill (D3, host mutation run 2026-09-28: SURVIVED).
  // 'region-failure-abandons-the-rest-of-the-period' returns out of
  // step_period() straight after run_region(), so it can only lose what the
  // stream orders AFTER the region. step_period() advances a tick as soon as
  // an instruction stamped strictly later is reached, so consumer_b's t=1/2
  // tick is after the region only if the region is entered BEFORE t=1/2. If
  // the scheduler stamps the region at t >= 1/2, both ticks run first,
  // calls() is 2 either way, and the EXPECT_GE below cannot see the mutant.
  // Asserted, not assumed, and the stream printed so the host run shows
  // where the region actually sits.
  std::optional<fuse::Rational> region_time;
  for (const auto& ins : h.plan.instructions()) {
    if (ins.region_enter >= 0) { region_time = ins.time; break; }
  }
  if (current_rank() == 0) {
    std::printf("[D3] region enters at t=%s; stream:\n%s\n",
                region_time ? region_time->to_string().c_str() : "(none)",
                h.plan.to_string().c_str());
  }
  ASSERT_TRUE(region_time.has_value());
  EXPECT_TRUE(*region_time < fuse::Rational(1, 2))
      << "the region is entered at t=" << region_time->to_string()
      << ", not before consumer_b's t=1/2 tick, so this test cannot detect "
         "'region-failure-abandons-the-rest-of-the-period': nothing it "
         "checks is ordered after the region";

  const auto r = runner.run_period(h.components);
  std::printf("[D3] rank %d: ok=%d code=%s consumer_b.calls=%d\n", current_rank(),
              r.ok() ? 1 : 0, r.ok() ? "-" : r.error().code.c_str(),
              static_cast<int>(cons_b->calls()));
  std::fflush(stdout);
  ASSERT_FALSE(r.ok())
      << "fluid refuses its first advance() inside the region; the period "
         "must report that failure on every rank, including ranks that "
         "belong only to the independent channel";
  EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << r.error().message;
  EXPECT_GE(cons_b->calls(), 2)
      << "consumer_b ticks TWICE this period (t=0 and t=1/2), on a clock "
         "that has nothing to do with the region; both ticks must still run "
         "rather than being abandoned when step_period() returns out of "
         "the region's t=0 failure -- 'region-failure-abandons-the-rest-"
         "of-the-period' (D3) abandons the t=1/2 tick specifically, which a "
         "same-time-as-the-region observable could not distinguish from "
         "an ordinary greedy-grouping choice";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Mutation "coupler-region-participants-missing-its-ranks"
// (transport-runtime.toml): resolve_regions()'s coupler-rank union scan --
// the `for (i = R.enter; i <= R.exit; ++i)` loop over the region's own
// Transport/Epoch instructions, added specifically so a coupler-routed
// channel's coupler is counted among `R.participants` even though it owns
// no checkpointed field and so is invisible to `derive_checkpoint_set()`'s
// `cp.components` -- is mutated to `i < R.enter`, an empty loop, so
// `region_ranks` never gains the coupler's ranks at all.
//
// SHAPE: `Overlap::Full` + `CouplerShape::Disjoint` puts "fluid" AND
// "solid" on the SAME `RankSet{0, world_size()-1}` (coincident with EACH
// OTHER) and the coupler alone on the one remaining rank, disjoint from
// both -- so `cp.components`' own ranks (fluid/solid, coincident) give
// `region_ranks.size() == 1` BEFORE the coupler scan runs at all. Under
// correct code the scan pushes the coupler's rank too, `region_ranks.
// size() == 2`, `R.participants` is set to it. Under the mutation
// `region_ranks` stays size 1, so `R.participants` is left EMPTY (`if
// (region_ranks.size() > 1)` never fires) -- exactly the "no communicator,
// or every checkpointed component on one rank" case resolve_regions()'s own
// comment says is legitimately treated as "purely local".
//
// TRACED KILL, no hang needed (like `StillRunsASecondChannelAfterARegionFailsUnderMpi`
// above, this fixture's failure is a clean, local verdict mismatch, not a
// collective one): `resolve_regions()`'s OWN per-block guard,
// `if (b.length == 0 && R.participants.size() <= 1) return Failure{
// "E-RT-REGION", "... would report convergence after one sweep"};`
// (src/runtime/iteration.cpp, immediately below the participants scan --
// see that function's own comment: "a rank that IS a genuine participant
// of a multi-rank region... is ordinary... Outside that case... an empty
// coupling value is still what it always was... refused below"), fires on
// the COUPLER's rank specifically: the coupler's own local `b.length`
// (this region's coupling-value block, sized from ITS OWN buffer) is 0
// (it owns no checkpointed field), and under the mutation
// `R.participants.size() <= 1` on every rank (participants is graph-
// derived, hence identical everywhere) -- so the refusal that is correctly
// SKIPPED (treated as the ordinary "n == 0" no-op) when the coupler is
// correctly counted instead FIRES, on the coupler's rank only, the moment
// `runner.initialize()` reaches this region. `fluid`/`solid`'s own rank
// never hits it (their block length is > 0 there), so this is an
// asymmetric, per-rank INITIALIZE()-time verdict split -- observable with
// nothing collective having run yet, unlike the region-body mutation above.
TEST(RunnerTransport, IterationRegionParticipantsIncludeADisjointCoupler) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);  // one rank for fluid/solid (coincident), one for the coupler
  auto h = TwoComponentJob("", /*with_iteration_region=*/true,
                           fuse::AccelKind::None,
                           /*with_independent_channel=*/false,
                           /*fail_fluid_after=*/-1, /*shape=*/Overlap::Full,
                           /*independent_interval=*/fuse::Rational{1},
                           /*coupler=*/CouplerShape::Disjoint);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok())
      << "rank " << current_rank() << ": a coupler disjoint from both "
         "checkpointed components, on a region whose components alone "
         "coincide on one rank, must be counted among the region's "
         "residual-agreement participants -- otherwise resolve_regions()'s "
         "own \"would report convergence after one sweep\" refusal fires on "
         "the coupler's rank for a coupling value that is legitimately "
         "zero-length there: " << init.error().code << ": " << init.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// R-10 (REVIEW.md), the positive case: `A` and `B` each span the WHOLE
// communicator (LinearFixedPointJob, above), so every rank owns a slice of
// both -- and at 3 ranks, kEntities == 4 splits 2/1/1, giving genuinely
// unequal per-rank local residuals. `initialize()` must now SUCCEED (not
// refuse: both components use AccelKind::None, the supported case above),
// and every rank must exit the region on the SAME sweep with the SAME
// iteration count test/verification/iteration_test.cpp's own
// Iteration.AitkenNeedsNoMoreIterationsThanPicard printed for the identical
// coefficients (a=1.0/k=0.5, a=2.0/k=0.4), entity count (4) and tolerance
// (1e-10) run on one process: 30. The map is applied entity-wise with no
// coupling between entities, so the fixed point and its contraction rate
// do not depend on how the 4 entities are split across ranks -- only
// reduce_region_residuals() combining the per-rank Euclidean sums back into
// the same global norm makes that hold under a real partition instead of
// desynchronising. Needs a host run: this sandbox cannot launch MPI at all.
TEST(RunnerTransport, RegionConvergesInTheSerialIterationCount) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  constexpr int kSerialIterations = 30;
  auto h = LinearFixedPointJob(kSerialIterations + 10);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok())
      << "AccelKind::None on a region spanning every rank is the supported "
         "case; " << (init.ok() ? "" : init.error().message);

  const auto run = runner.run_period(h.components);
  ASSERT_TRUE(run.ok()) << run.error().code << ": " << run.error().message;
  ASSERT_FALSE(runner.iteration_stats().empty());
  const fuse::IterationStats& st = runner.iteration_stats()[0];
  EXPECT_TRUE(st.converged);
  EXPECT_EQ(st.iterations, kSerialIterations)
      << "every rank must exit on the same sweep the serial run does, "
         "whatever this rank's own unequal local slice of the 4 entities "
         "looks like";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// D3 (KNOWN_ISSUES), `one-transport-shared-across-every-plan`:
// `Runner::Impl::attach_comm_graphs()` (src/runtime/runner.cpp) builds ONE
// `Transport` object per CommGraph-backed plan and stores it at
// `graph_transports[pi]`; the mutant changes `graph_transports[pi] =
// std::move(own)` to `if (pi == 0) graph_transports[pi] = std::move(own)`,
// so every plan index PAST THE FIRST is left holding a null `unique_ptr`
// even though `build_plan()` (called unconditionally, for every `pi`, just
// above) already succeeded for it. `TwoComponentJob("", ..., with_
// independent_channel=true)` lowers to exactly two CommGraph-backed
// Transport instructions -- "link" (fluid -> solid) gets `pi=0`, "linkB"
// (producer_b -> consumer_b) gets `pi=1`, in declaration order
// (src/runtime/runner.cpp's `case OpKind::Transport:` pushes one plan per
// instruction, unconditionally, with no dedup by shape even though both
// channels happen to share the same fluid_/solid_ rank split) -- so under
// the mutant "linkB"'s own execute() (OpKind::Transport, `p.has_graph`)
// reads `graph_transports[1].get() == nullptr` and returns a clean
// `E-RT-TRANSPORT` failure ("the CommGraph-backed plan for this Transport
// instruction has no Transport instance").
//
// WHY THE EXISTING `StillRunsASecondChannelAfterARegionFailsUnderMpi` DOES
// NOT CATCH THIS. That test ALSO forces "fluid" to fail inside the region
// (`fail_fluid_after=0`), which sets `step_failure` before "linkB"'s
// Transport instruction is even reached; `step_period()`'s failure branch
// only OVERWRITES `step_failure` `if (!step_failure.has_value())` (see the
// R-09 mutation's own `old` line, just above this in the same function),
// so the region's "E-TEST-ADVANCE-REFUSED" wins and "linkB"'s distinct
// "E-RT-TRANSPORT" verdict is silently discarded -- masked, not fixed. And
// `cons_b->calls()` does not depend on whether "linkB"'s DATA arrived
// correctly at all: a component's own tick runs from whatever its input
// buffer already holds, null Transport or not.
//
// So THIS test forces nothing to fail (`fail_fluid_after` stays at its
// default, no iteration region at all -- `with_iteration_region=false`, the
// mutant needs no region to manifest, and one fewer moving part is one
// fewer thing masking a different failure), and checks the DATA delivered
// through BOTH channels, using the same by-global-id technique
// `ARunDeliversTheSourceValuesToTheDestinationRanks` uses for "link".
// Under the mutant, "linkB"'s own `run_period()` (called directly, since
// there is no region to iterate) returns `E-RT-TRANSPORT`, which the
// `ASSERT_TRUE` below catches outright; if some future refactor changed
// which plan gets the null instance, checking BOTH channels' delivered
// data (not just "did it fail") is what still catches a shared instance
// that happens to work for one channel and silently corrupt the other.
TEST(RunnerTransport, TwoIndependentCommGraphChannelsBothDeliverInOnePeriod) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = TwoComponentJob("", /*with_iteration_region=*/false,
                           fuse::AccelKind::None,
                           /*with_independent_channel=*/true);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok())
      << "build_plan() must succeed for both 'link' and 'linkB' regardless "
         "of the mutation: only the FINAL move-into-storage is skipped, not "
         "the plan build itself";

  h.write_source_by_global_id();
  h.write_source_b_by_global_id();
  const auto r = runner.run_period(h.components);
  ASSERT_TRUE(r.ok())
      << r.error().code << ": " << r.error().message
      << " -- nothing in this fixture is configured to fail, so any error "
         "here means a CommGraph-backed plan lost its Transport instance";
  EXPECT_EQ(h.count_destination_matches(), h.local_destination_entity_count())
      << "'link' (pi=0, the mutant's surviving index) must still deliver "
         "correctly";
  EXPECT_EQ(h.count_destination_b_matches(),
           h.local_destination_b_entity_count())
      << "'linkB' (pi=1) is the mutant's actual target: a shared Transport "
         "instance moved to pi=0 leaves this plan's own instance nulled, "
         "and its data never arrives";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// D3 (KNOWN_ISSUES), `region-residual-skips-the-cross-rank-reduction` (R-10):
// `run_region()`'s `if (R.participants.size() > 1) { ... reduce_region_
// residuals(...) ... }` (src/runtime/iteration.cpp) agrees `R.block_residual`/
// `R.block_value` across every rank in the region BEFORE the convergence
// criterion reads them; the mutant (`if (false)`) skips this unconditionally,
// so `R.block_residual`/`R.block_value` keep whatever this rank's OWN LOCAL
// sum (computed just above the mutated line: `rs += d*d; ... R.block_
// residual[bi] = std::sqrt(rs);`, over ONLY this rank's owned entities)
// already held -- every rank decides convergence from its own slice.
//
// `RegionConvergesInTheSerialIterationCount` (above) does not catch this: it
// gives `A` (`LinearRelaxComponent`) the SAME `k` on every rank, and because
// the map is applied entity-wise with no coupling between entities, a
// RELATIVE criterion (residual / value) decays at the SAME RATE whether it
// is summed over 1 entity, 2 entities, or the global 4 -- the local and
// global verdicts happen to AGREE at every sweep for a uniform `k`, so the
// mutation is invisible there no matter how unequal the entity SPLIT is.
//
// This fixture instead varies `k` (not the entity count) BY RANK: rank 0
// builds its own `A` with `kA = kFast = 0.001`, every other rank with
// `kA = kSlow = 0.5` (both pass the SAME graph structure to every rank --
// `CouplingGraph` carries no coefficients, only field/channel/region shapes
// -- so this is one legitimate way for a coupled code to vary a local
// feedback gain by subdomain, not a divergent graph). Simulating the
// Gauss-Seidel update (`a_out' = 1 + kA*a_in`, `b_out' = 2 + 0.4*a_out'`,
// `a_in` lagged one sweep) from zero:
//   rank 0 (kA=0.001): relative "a2b" residual is 9.6e-7 at sweep 3, 3.8e-10
//     at sweep 4 -- under `tol = 1e-6` (chosen well above both), CONVERGED
//     by its own local residual at sweep 4.
//   every other rank (kA=0.5): relative "a2b" residual is 3.8e-3 at sweep 5
//     -- three orders of magnitude above `tol`, nowhere near converged.
// At `max_iterations = 5`, the GLOBALLY AGREED residual (dominated by the
// slow ranks, since `num::combine`-style reduction sums OWNED entities
// across every participant) never drops under `tol`: `st.converged` must be
// `false` on EVERY rank under the fix, having spent the whole budget.
// Under the mutant, rank 0 decides FROM ITS OWN LOCAL RESIDUAL ALONE,
// already under `tol` by sweep 4, and reports `st.converged == true` at a
// SMALLER `st.iterations` -- exactly the "reaches a different convergence
// decision" this task names, caught here by an outright `EXPECT_FALSE`/
// `EXPECT_EQ` mismatch on rank 0 without needing the desync to hang first
// (though a hang on a LATER sweep, once rank 0 has stopped participating in
// the region's own cross-rank exchange while the slow ranks keep going, is
// an equally correct kill -- see the mutation's own `describes` text).
TEST(RunnerTransport, RegionResidualLocalConvergenceIsNotEnoughUnderMpi) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  constexpr double kFast = 0.001;
  constexpr double kSlow = 0.5;
  constexpr double kTol  = 1e-6;
  constexpr int    kMaxIterations = 5;
  const double kA = (current_rank() == 0) ? kFast : kSlow;
  auto h = LinearFixedPointJob(kMaxIterations, kA, kTol);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok())
      << "AccelKind::None on a region spanning every rank is the supported "
         "case; " << (init.ok() ? "" : init.error().message);

  const auto run = runner.run_period(h.components);
  ASSERT_TRUE(run.ok()) << run.error().code << ": " << run.error().message;
  ASSERT_FALSE(runner.iteration_stats().empty());
  const fuse::IterationStats& st = runner.iteration_stats()[0];
  EXPECT_FALSE(st.converged)
      << "rank " << current_rank() << ": the GLOBALLY agreed residual is "
         "still dominated by the slow ranks (kA=" << kSlow << ") at sweep "
      << kMaxIterations << "; only a rank deciding from its OWN local "
         "residual (rank 0's kA=" << kFast << " converges by itself) could "
         "report converged here";
  EXPECT_EQ(st.iterations, kMaxIterations)
      << "the fix spends the whole iteration budget because the AGREED "
         "residual never clears tol; a smaller count means this rank exited "
         "on its own local (unreduced) residual instead";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(RunnerTransport, InitializeBuildsAPlanForEveryTransport) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
}

// base_communicator set but transport left (or set) to "copy" -- an easy,
// realistic misconfiguration. attach_comm_graphs() still builds a real,
// disjoint CommGraph and sets has_graph = true, but "copy" cannot execute a
// subset gather/scatter plan: it would silently move the wrong data once
// has_graph starts driving execution. This must be a clean E-RT-TRANSPORT
// refusal, not a silent success.
TEST(RunnerTransport, ACommGraphBackedPlanRefusesANonMpiTransport) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "copy";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-TRANSPORT");
}

TEST(RunnerTransport, ARunDeliversTheSourceValuesToTheDestinationRanks) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok());

  // Values derived from each entity's global id, so a permutation is
  // distinguishable from a correct delivery. A constant would not be.
  h.write_source_by_global_id();
  ASSERT_TRUE(runner.run_period(h.components).ok());
  EXPECT_EQ(h.count_destination_matches(), h.local_destination_entity_count());
}

TEST(RunnerTransport, TheTransportPathAllocatesNothingWhileRunning) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  // The same guarantee Runner.AllocatesNothingWhileRunning pins for the serial
  // path (spec §2 layer 3). This is what proves that holding the TransportSpec
  // behind an index, rather than by value in Resolved, actually removed the
  // per-step allocation transport.hpp:70-77 warns about.
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);
  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok());
  ASSERT_TRUE(runner.run_period(h.components).ok());  // warm-up, uncounted

  std::size_t allocations = 0;
  bool        ok = false;
  {
    const AllocationCounter counter;
    ok = runner.run(h.components, 4).ok();
    allocations = counter.count();
  }
  ASSERT_TRUE(ok);
  EXPECT_EQ(allocations, 0u)
      << "the Transport path allocated " << allocations << " times";
}

// The rank-count-invariance check. Every per-rank assertion above (e.g.
// ARunDeliversTheSourceValuesToTheDestinationRanks) compares a rank's
// received count against ITS OWN expected count, so a gather or scatter that
// silently dropped one rank's entities entirely -- e.g. by skipping that
// rank's contribution to p.send/p.recv without erroring -- could still leave
// every OTHER rank's local check green. This test instead sums the delivered
// match count across the whole job with MPI_Allreduce and compares it against
// the job-wide expected count, so a dropped rank's absence shows up here even
// though it shows up nowhere else.
//
// MUTATION RESULTS (Steps 4-5 of the plan): temporarily replacing
// `in[p.gather[k] * nc + c]` with `in[k * nc + c]` in the gather loop, and
// separately `out[p.scatter[k] * nc + c]` with `out[k * nc + c]` in the
// scatter loop (src/runtime/runner.cpp), was run at 2, 3, 4 and 5 ranks.
//
// BOTH MUTATIONS SURVIVED -- this test PASSED when it should have failed --
// AT EVERY RANK COUNT (2, 3, 4 and 5), against the original
// `PartitionedTestBackend::block()`, which handed out each rank's ids in
// ascending order. CommGraph's send/recv lists are sorted by (global_id, ...)
// (comm_graph.cpp:1125-1133), and with an ascending block a rank's local
// storage index already equals its position in that sorted order, so
// `p.gather[k] == k` and `p.scatter[k] == k` on every rank: the mutation
// changed nothing. This was the identity-permutation gap the plan's Task 5
// notes flagged as a possible, non-blocking outcome -- and it was the index
// maps that were degenerate, not the per-entity source values (those were
// already distinguishable: `write_source_by_global_id()` writes each id's own
// value). Fixed by reversing each rank's LOCAL STORAGE order in `block()`
// (this file) so local index 0 holds the block's largest global id rather
// than its smallest, decoupling "sorted-by-gid position" from "local index"
// without changing which ids a rank owns. After that fix, both mutations were
// caught -- this test FAILED as expected -- at every rank count tried (2, 3,
// 4 and 5). Neither mutation survives anywhere post-fix.
TEST(RunnerTransport, TheDeliveredFieldDoesNotDependOnTheRankCount) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  // The physics is rank-count invariant, so the delivered values must be too.
  // A gather or scatter that silently dropped one rank's entities passes every
  // per-rank check and fails only this one, because only this one compares a
  // JOB-WIDE count against what the graph says it should be.
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);
  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok());

  h.write_source_by_global_id();
  ASSERT_TRUE(runner.run_period(h.components).ok());

  unsigned long local_ok = h.count_destination_matches();
  unsigned long total_ok = 0;
  MPI_Allreduce(&local_ok, &total_ok, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  EXPECT_EQ(total_ok, h.global_destination_entity_count());
}

// ---- the three overlap shapes through the Runner (KNOWN_ISSUES D5) -------
//
// CommGraph is tested on all three shapes (rows 1-6). These put the same
// shapes through Runner::initialize + run_period, which is where CommGraph's
// output is consumed. PARTIAL is the case that could not be expressed before
// per-placement decomposition: the Runner handed both CommGraph sides one
// view, so each rank held one piece of the mesh shared by both components,
// and two partially overlapping placements could not both cover the region.
// It is also the only shape where a peer-to-peer plan and a same-rank local
// copy run in the same exchange.

static void expect_every_destination_entity_delivered(
    Overlap shape, CouplerShape coupler = CouplerShape::None) {
  TwoComponentJob h{"", false, fuse::AccelKind::None, false, -1, shape,
                    fuse::Rational{1}, coupler};
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);
  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  h.write_source_by_global_id();
  const auto run = runner.run_period(h.components);
  ASSERT_TRUE(run.ok()) << run.error().code << ": " << run.error().message;

  unsigned long local_ok = h.count_destination_matches();
  unsigned long local_n  = h.local_destination_entity_count();
  unsigned long total_ok = 0, total_n = 0;
  MPI_Allreduce(&local_ok, &total_ok, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&local_n, &total_n, 1, MPI_UNSIGNED_LONG, MPI_SUM,
                MPI_COMM_WORLD);
  // Both counts, because each catches what the other cannot: a destination
  // decomposition that lost or duplicated entities changes total_n, and a
  // delivery that put the right values in the wrong slots changes total_ok.
  EXPECT_EQ(total_n, h.global_destination_entity_count())
      << "the destination's own decomposition does not cover the region "
         "exactly once";
  EXPECT_EQ(total_ok, h.global_destination_entity_count())
      << "every destination entity must hold its own global id";
}

TEST(RunnerTransport, APartiallyOverlappingPlacementDeliversEveryEntity) {
  // Fluid on 0..W-2, solid on 1..W-1. At two ranks that is {0} and {1}, which
  // is disjoint -- so three is the first rank count where the shape is
  // genuinely partial: rank 0 only sends, rank W-1 only receives, and every
  // rank between holds a block of EACH component's decomposition.
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  expect_every_destination_entity_delivered(Overlap::Partial);
}

TEST(RunnerTransport, AFullyOverlappingPlacementDeliversEveryEntity) {
  // Identical placements: the IR emits no Transport (rules_channel.cpp,
  // same_as), so this pins that identical placements get the identical
  // decomposition the MeshBackend contract requires -- a different one would
  // leave values at the wrong local indices with nothing to move them.
  expect_every_destination_entity_delivered(Overlap::Full);
}

TEST(RunnerTransport, ADisjointPlacementDeliversEveryEntityThroughTheSameHarness) {
  // The third shape through the same helper, so the three differ only in the
  // placement and not in what is checked.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  expect_every_destination_entity_delivered(Overlap::Disjoint);
}

// ---- the same delivery check with the Remap routed through a coupler ------
//
// Coupler placement (docs/dev/coupler-placement.md): the channel's value moves
// fluid -> coupler, is remapped on the coupler's decomposition, and moves
// coupler -> solid; a hop between identical placements is elided. The three
// shapes below are the ones the layout driver registers (layout.coupler.*),
// asserted here through the same helper as the no-coupler shapes so the only
// difference is the route. The first host run of those driver rows failed on
// every shape at initialize (E-MESH-PLACEMENT, 2026-09-27): the driver had
// told the backend about two placements and the coupler was the third. This
// fixture reads its placements from the graph, which is why it could not have
// caught that -- and why the driver now does the same.

TEST(RunnerTransport, ADisjointCouplerDeliversEveryEntity) {
  // The coupler is the last rank alone; fluid and solid split the others.
  // Three ranks is the first count with a rank for each of the three roles;
  // both hops are real Transports (peer plans, no rank in common).
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  expect_every_destination_entity_delivered(Overlap::Disjoint, CouplerShape::Disjoint);
}

TEST(RunnerTransport, ACouplerOnTheTargetsRanksDeliversEveryEntity) {
  // The coupler's placement IS solid's, so the coupler -> solid hop is
  // elided (same_as) and the route must reproduce the no-coupler result:
  // the same check, the same count, exactly.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  expect_every_destination_entity_delivered(Overlap::Disjoint, CouplerShape::EqualsTarget);
}

TEST(RunnerTransport, ACouplerOverlappingTheSourceDeliversEveryEntity) {
  // The coupler straddles the fluid/solid boundary: one rank in common with
  // each side and identical to neither, so both hops are partial-overlap
  // plans -- a local copy on the shared rank, a peer exchange for the rest.
  // Four ranks gives fluid {0,1}, solid {2,3}, coupler {1,2}.
  FUSE_REQUIRE_AT_LEAST_RANKS(4);
  expect_every_destination_entity_delivered(Overlap::Disjoint, CouplerShape::OverlapsSource);
}

// Runner::initialize() is re-callable by contract: it clears `initialized` at
// entry, and resolve_instructions() resets every setup-owned container so the
// second call starts from exactly the state the first one did.
// `transport_plans`/`transport_instruction` were the two containers that got
// appended to and never cleared, so a SECOND initialize() of a plan holding
// ONE Transport saw two entries and refused it with a bogus E-RT-TRANSPORT
// ("the plan has 2"). The MPI config is not decoration here: the guard that
// tripped lives in attach_comm_graphs(), which is a no-op without
// base_communicator, so a copy-transport version of this test cannot see the
// bug at all.
TEST(RunnerTransport, InitializeIsRepeatableOnTheSameRunner) {
  // Two DISJOINT placements need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` refuses the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();

  const auto first = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(first.ok()) << "first initialize(): " << first.error().code
                          << ": " << first.error().message;
  const auto second = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(second.ok()) << "second initialize(): " << second.error().code
                           << ": " << second.error().message;
}

// attach_comm_graphs() is collective over base_communicator, but the two
// resolutions that run before it are not: they read each rank's own
// FieldRegistry and fail rank-divergently on an ordinary misconfiguration.
// Before the status-agreement checkpoint in setup(), the rank that failed
// returned immediately while every healthy rank walked into MPI_Comm_split and
// blocked there forever. This test is that exact shape.
//
// NO IN-TEST TIMEOUT, ON PURPOSE. What this test asserts is "initialize()
// returned at all", and the hang-safety net for that is the CTest
// `TIMEOUT 180` property carried by every runner_transport_*rank registration
// (test/CMakeLists.txt). A busy-wait or alarm inside the test would duplicate
// a budget CTest already owns, and would have to guess at it.
TEST(RunnerTransport, ARankDivergentSetupFailureRefusesOnEveryRank) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  // Rank 0 alone leaves "solid" unbound, so resolve_buffers() fails there with
  // E-RT-UNBOUND and nowhere else.
  if (current_rank() == 0) {
    bind_all_except(h, "solid");
  } else {
    bind_all(h);
  }

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();

  const auto r = runner.initialize(h.components, h.registry, cfg);
  ASSERT_FALSE(r.ok()) << "rank " << current_rank()
                       << " accepted a job that rank 0 could not set up";
  // Each rank returns the verdict it is entitled to: the broken rank its own
  // local diagnostic, the healthy ranks the checkpoint's refusal.
  if (current_rank() == 0) {
    EXPECT_EQ(r.error().code, "E-RT-UNBOUND") << r.error().message;
  } else {
    EXPECT_EQ(r.error().code, "E-RT-SETUP-PEER") << r.error().message;
  }
}

// No main(): test/support/fuse_test_main reduces the failure status across
// MPI_COMM_WORLD, so a failure on one rank fails every rank instead of leaving
// the others to block in the next collective. A test that keeps its own
// entry point silently loses that.
