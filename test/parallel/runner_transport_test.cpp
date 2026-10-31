// Runner + CommGraph + MpiTransport under a real mpiexec. Not a
// fuse_add_mpi_test(): it calls MPI_Init, must be launched by mpiexec, and must
// fail the job rather than return on a per-rank failure.

#include <gtest/gtest.h>
#include <mpi.h>

// The documented rank-requirement convention (docs/dev/testing.md).
#include "fuse_test_main.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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
// test/support/partitioned_test_backend.hpp so
// examples/layout_driver can share the one per-placement decomposition
// instead of carrying a second, drifting copy. See that header for the
// class bodies.
#include "../support/partitioned_test_backend.hpp"


/// Registers one field of `count` doubles, nothing more. `build_ticks()`
/// (runner.cpp:1146-1164) requires a `Component` object for EVERY graph
/// component on EVERY rank regardless of that component's `RankSet` -- so
/// both "fluid" and "solid" must exist here even on a rank that physically
/// belongs to only one of them.
class StubComponent final : public fuse::Component {
 public:
  /// `rollback`: an implicit coupling group requires every
  /// component it checkpoints to support rollback (E-ITER-NOROLLBACK);
  /// default false matches every existing caller, none of which puts this
  /// component in an implicit coupling group.
  /// `fail_after`: -1 (the
  /// default, every existing caller) never rejects; 0 rejects starting with
  /// this component's very FIRST advance() call, the shape
  /// RunnerTransport.StillRunsASecondChannelAfterARegionFailsUnderMpi needs
  /// to force an implicit coupling group's re-advance to fail on sweep 0 under a
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
    // It must not be incremented inside the condition below, to the right of
    // `fail_after_ >= 0 &&` -- for a stub with no failure configured
    // (`fail_after_ == -1`) short-circuit evaluation would skip the increment
    // entirely and `calls()` would stay at 0 however often the component ran.
    // StillRunsASecondChannelAfterARegionFailsUnderMpi asserts
    // `cons_b->calls() >= 1` on exactly such a stub.
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
/// stub components, plus helpers that read/write through the SAME implicit coupling group both
/// components' fields reference (Transport preserves implicit coupling group -- see the plan's
/// Scope Decision -- so there is exactly one to read).
/// How the source and destination processor sets relate (matrix rows 1-6 at the
/// CommGraph level; this is the same three shapes through the Runner).
enum class Overlap {
  Disjoint,  ///< fluid on the first half, solid on the rest -- the default
  Partial,   ///< fluid on 0..W-2, solid on 1..W-1: shared ranks 1..W-2 plus
             ///< one rank unique to each side; genuinely partial from W = 3
  Full,      ///< both on every rank: identical placements, no Transport
};

/// Where the field transfer's Remap runs (RemapSpec::remap_on): on the target, which
/// is today's default and emits ONE Transport, or on a separately placed
/// coupler, which emits fluid -> coupler and coupler -> solid with either hop
/// elided when the processor sets coincide.
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
                                          CouplerShape coupler = CouplerShape::None,
                                          bool reserve_last_rank = false) {
  // With a DISJOINT coupler, or `reserve_last_rank`, the last rank belongs to
  // neither physics component, and the two split the remaining W-1 ranks.
  const int w = world_size() -
                ((coupler == CouplerShape::Disjoint || reserve_last_rank) ? 1 : 0);
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
  /// `entity_set`, when non-empty, is set on the shared implicit coupling group. The
  /// E-RT-TRANSPORT-REGION test passes "wall"; every other test passes "".
  /// `with_iteration_region`: wraps the "link" field transfer
  /// in an implicit coupling group with a real convergence measure, and makes both
  /// components rollback-capable so E-ITER-NOROLLBACK does not mask the
  /// check this exists to reach. Default false adds no implicit coupling
  /// group. `accel_kind`: which accelerator the
  /// implicit coupling group declares; only read when `with_iteration_region` is true.
  /// AccelKind::None (the default, and IterationRegionDecl's own) is one of
  /// the two kinds genuinely supported for an implicit coupling group spanning more
  /// than one rank; a dynamic kind (Aitken and stronger) is the case it
  /// still rejects -- see RefusesADynamicAcceleratorOnACrossRankRegion.
  /// `with_independent_channel`: adds a SECOND field transfer, "linkB", between two
  /// more components on the SAME
  /// fluid_/solid_ ranks and the SAME shared implicit coupling group, but NOT one of the
  /// implicit coupling group's field transfers. Default false adds no such
  /// transfer. `fail_fluid_after`: forwarded to "fluid"'s StubComponent
  /// (see its own doc); -1 (default) never rejects.
  /// `independent_interval`: producer_b/consumer_b's own
  /// time grid. Default `Rational{1}` (one tick per coupling cycle).
  /// `StillRunsASecondChannelAfterARegionFailsUnderMpi` passes
  /// `Rational{1,2}` (two ticks per coupling cycle) so its SECOND tick sits at
  /// `t=1/2`, strictly later than anything at `t=0` by the schedule's own
  /// (time, phase, rank) sort key (`plan.hpp`'s own contract) -- not by any
  /// same-instant tie-break `group_regions_within_instants()` might resolve
  /// either way. A survivor abandoning "the rest of the coupling cycle" the moment
  /// the implicit coupling group fails abandons that `t=1/2` tick regardless of where the
  /// greedy grouping would have placed a same-time observable.
  explicit TwoComponentJob(const std::string& entity_set,
                           bool with_iteration_region = false,
                           fuse::AccelKind accel_kind = fuse::AccelKind::None,
                           bool with_independent_channel = false,
                           int fail_fluid_after = -1,
                           Overlap shape = Overlap::Disjoint,
                           fuse::Rational independent_interval = fuse::Rational{1},
                           CouplerShape coupler = CouplerShape::None,
                           bool independent_consumer_outside_region = false)
      // MEMBER DECLARATION ORDER, NOT INITIALIZER-LIST ORDER, decides when
      // each of these actually runs. `backend` is declared below
      // fluid_/solid_/n_entities_/dim_ for exactly this reason: it is built
      // FROM them, so they must already hold their values.
      : fluid_{placement_for(shape, true, coupler, independent_consumer_outside_region)},
        solid_{placement_for(shape, false, coupler, independent_consumer_outside_region)},
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
      // Both fields share one implicit coupling group, so nothing would emit a Remap for this
      // field transfer and remap_on would be W-CHANNEL-REMAP-ON-UNUSED; Conservative
      // asks for one unconditionally, exactly as the processor-layout driver does. The
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

    // Declared AFTER the implicit coupling group, which is the natural order and which the
    // scheduler honours: `group_regions_within_instants()` keeps the
    // implicit coupling group's body contiguous whatever order its neighbour was declared in.
    // Declaring it FIRST would hide failures: with the independent field transfer ahead of
    // the implicit coupling group, everything this test observes happens before the
    // failure, so "the rest of the coupling cycle" would be empty and a mutant that skips
    // it would skip nothing.
    if (with_independent_channel) {
      const auto pb = graph.add_component(fuse::ComponentDecl{
          .name = "producer_b", .ranks = fluid_,
          .interval = independent_interval});
      // `independent_consumer_outside_region`: consumer_b alone on the
      // last rank, which neither implicit coupling group component uses, so one rank of
      // linkB is NOT a participant of the implicit coupling group -- the only shape in which
      // a rank can be left waiting in linkB for a partner that returned out
      // of the implicit coupling group.
      const auto cb = graph.add_component(fuse::ComponentDecl{
          .name = "consumer_b",
          .ranks = independent_consumer_outside_region
                       ? fuse::RankSet{world_size() - 1, 1}
                       : solid_,
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

    // Every component's own decomposition, built before
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
        schedule(graph, dag, clocks, fuse::make_region_sizes(graph, dag, backend));
    EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
    plan = sched.plan;

    // Each component's storage is ITS OWN decomposition's size: on a rank in
    // both partially overlapping processor sets the two differ. (Component ids
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
    // `fluid` and `solid` share one implicit coupling group and one backend view per rank
    // (the class comment above explains why), so on a rank that belongs to
    // `fluid_` this same view is FLUID's local block, not solid's -- the
    // "solid" edge's storage there is never written by a correct Transport
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

/// Bind every component's storage; the Runner rejects a plan otherwise.
/// Required by the same contract test/unit/runner_test.cpp:399-404 documents (`bind_all`): nothing
/// in Runner::initialize() calls Component::register_fields() for you.
void bind_all(TwoComponentJob& h) {
  for (auto& c : h.components) {
    EXPECT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

// ---------------------------------------------------------------------------
// A genuine two-way iterated coupling, for the positive
// convergence test below. TwoComponentJob's one-way "link" field transfer has
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
  /// `fail_restore_after`: -1 (the default,
  /// every existing caller) never rejects; N makes the (N+1)-th
  /// restore_state() call (0-based, matching StubComponent's own
  /// fail_after convention) return a Failure instead of restoring,
  /// forcing a mid-sweep restore_state() failure on exactly this
  /// component's own rank without perturbing a peer on a different rank.
  LinearRelaxComponent(std::string name, double a, double k, std::size_t n,
                       int fail_restore_after = -1)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0),
        fail_restore_after_(fail_restore_after) {}

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
    const int n = restore_calls_++;
    if (fail_restore_after_ >= 0 && n >= fail_restore_after_) {
      return fuse::Failure{"E-TEST-RESTORE-REFUSED",
                           "'" + name_ + "' refuses restore_state() past "
                               "call " + std::to_string(fail_restore_after_)};
    }
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
  int                  fail_restore_after_ = -1;
  int                  restore_calls_ = 0;
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

  /// `kA`/`tol`: `RegionResidualLocalConvergenceIsNot-
  /// EnoughUnderMpi` passes a `kA` CHOSEN PER RANK (`current_rank() == 0 ?
  /// fast : slow`) so this rank's own LOCAL residual and the implicit coupling group's
  /// GLOBALLY-AGREED residual reach different convergence outcomes at the
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
    // kind supported for an implicit coupling group spanning more than one rank
    // (see resolve_regions()'s accelerator-kind check), and the
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
        schedule(graph, dag, clocks, fuse::make_region_sizes(graph, dag, backend));
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
  // A skip states the requirement where the reader is, and does not depend
  // on the registration staying put.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int world = 0, me = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);

  // Two disjoint components spanning the job, as spec Section 6 mode 1 describes.
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
  // MeshBackend can count an implicit coupling group's entities but not enumerate them, so a
  // buffer for `entity_set: wall` cannot be said to be indexed by set position
  // or by mesh-local index. remap_plan.hpp:356-363 already rejects this with
  // E-REMAP-REGION for the same reason; a Transport that guessed would deliver
  // a permutation of the right values, which is worse than a rejection because
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
  // The control for the test above: without it, a guard that rejected
  // everything would pass. Same configuration as that test -- the mpi
  // transport over the base communicator -- so the entity set is the ONLY
  // difference between the two.
  //
  // The "copy" transport with no communicator would pass only while both ends
  // of the Transport shared one mesh view; with each end in its own
  // component's decomposition it copies fluid's local block whole into
  // solid's, which on a rank in one processor set and not the other are
  // different sizes. That is rejected (E-RT-SIZE) and pinned by the test
  // below, so this control does not ride on it.
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
  // whole input edge into its whole output edge. Across two DISJOINT
  // processor sets those are two components' decompositions of the implicit coupling group --
  // 12 entities and 0 on a rank in one of them -- and an element-wise copy
  // would put one component's local values into another's slots. The
  // rejection is check_transport_extents()'s whole-edge-copy branch.
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

// An implicit coupling group's convergence decision must not be computed
// from a purely rank-local residual. resolve_regions() and run_region()
// reduce the residual across the implicit coupling group's participants on
// every sweep (reduce_region_residuals(), src/runtime/runner.cpp), so an
// implicit coupling group using AccelKind::None or AccelKind::Constant is
// supported -- see RegionConvergesInTheSerialIterationCount below, which is
// the TwoComponentJob("", true) scenario.
//
// What is NOT supported: a DYNAMIC accelerator (Aitken and stronger)
// derives its relaxation coefficient from an inner product over this
// rank's own local iterate history, computed inside fuse::num
// (src/num/accel_quasi_newton.cpp, accel_relaxation.cpp's Aitken branch),
// and Accelerator::update()'s signature has no hook for a caller-supplied
// global reduction. Two ranks with
// different local slices would compute different coefficients from that
// inner product and take different accelerator paths even with the
// residual itself correctly agreed, reopening the desynchronisation risk.
// `fluid`/`solid` are on disjoint rank sets (as in every
// other TwoComponentJob test), so this needs `FUSE_REQUIRE_AT_LEAST_
// RANKS(2)` to reach a real cross-rank configuration.
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
         "case the residual reduction still refuses";
  EXPECT_EQ(r.error().code, "E-RT-REGION-MPI") << r.error().message;
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// What step_period() (src/runtime/runner.cpp) carrying a failure forward is
// FOR: an implicit coupling group failure must not abandon a LATER, unrelated field
// transfer's collective
// instructions in the same coupling cycle. Under the base build configuration (no MPI) that abandonment is
// unobservable: nothing is waiting on the other end. Under a real multi-rank
// run it is exactly the hang rank-agreement.md exists to prevent -- "fluid"'s
// rank(s) would return out of step_period() the moment the implicit coupling group fails
// (without the carry-forward) while "consumer_b"'s rank(s), on the disjoint `solid_` set,
// still walk forward to "linkB"'s Epoch and block waiting for a partner that
// already left.
//
// "linkB" does not have to be declared before the implicit coupling group
// to survive scheduling -- `with_independent_channel`'s block is declared
// AFTER `add_iteration_region` (see the class comment above) -- but the
// declaration order ALONE does not expose the failure: producer_b/
// consumer_b's own component TICK would still land at the same `t=0` the
// implicit coupling group occupies, and `group_regions_within_instants()` only reorders the
// INSTRUCTION stream inside one (time, phase) window -- it says nothing
// about which same-time observable an unrelated reader happens to see, and
// a test that depends on winning that tie-break is a test that depends on
// an implementation detail the mutation is not really about.
//
// So this fixture puts `producer_b`/`consumer_b` on their OWN time grid,
// `Rational{1,2}` -- two ticks per coupling cycle, at `t=0` and `t=1/2` -- instead of
// sharing the implicit coupling group's `t=0`. `schedule()`'s own contract (`plan.hpp`:
// "schedule() orders by (time, TickPhase, topological rank..., NodeId)")
// means the SECOND tick, at `t=1/2`, is placed strictly after every `t=0`
// instruction by the sort key itself, not by any greedy preference that
// could go either way. With the carry-forward, `step_period()` carries the implicit
// coupling group's
// failure forward and keeps walking the stream, so both ticks of producer_b/
// consumer_b still run (their own field has nothing to do with the implicit coupling group,
// so `block_advances_from()` never reaches their slot) and "linkB"'s
// Transport+Epoch at `t=1/2` complete on every process. Without it,
// `step_period()` would return the instant the implicit coupling group's `run_region()`
// reports failure, at `t=0` -- strictly BEFORE `t=1/2` -- so consumer_b's
// second tick, and "linkB"'s second exchange, would be abandoned outright:
// no hang is even needed for a mutant to be visibly wrong here, though a
// version of this shape that put producer_b/consumer_b on the SAME `t=0`
// time grid would still hang the way the ctest TIMEOUT property (matching the
// existing mutation entries in build-evidence/mutations/transport-runtime.
// toml) is there to catch.
//
// With the carry-forward, every rank -- both the ones that saw "fluid" fail and the
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

  // What this test does NOT detect: the implicit coupling group is
  // entered at t=1, after both of consumer_b's ticks, and every rank of linkB
  // is also an implicit coupling group participant, so a coupling cycle abandoned straight after the
  // implicit coupling group loses nothing observable here. It pins the tick
  // carry-forward (ticks unrelated to the
  // failure still run). 'region-failure-abandons-the-rest-of-the-period' is
  // killed by AChannelOutsideTheRegionStillCompletesAfterTheRegionFails below.
  const auto r = runner.run_period(h.components);
  ASSERT_FALSE(r.ok())
      << "fluid refuses its first advance() inside the region; the period "
         "must report that failure on every rank, including ranks that "
         "belong only to the independent channel";
  // The fluid is stepped only on its own processes, so only
  // they see its refusal; every other process learns of it at an agreement
  // point -- linkB's Epoch or the implicit coupling group's residual
  // agreement, whichever its stream reaches first -- and reports that a
  // partner failed. Exactly those codes, nothing else.
  if (h.fluid_.contains(current_rank())) {
    EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << r.error().message;
  } else {
    EXPECT_TRUE(r.error().code == "E-RT-EPOCH-PEER" ||
                r.error().code == "E-RT-REGION-PEER")
        << "process " << current_rank() << " does not step the fluid, so it "
        << "must report a partner's failure, not " << r.error().code << ": "
        << r.error().message;
  }
  // Only consumer_b's own processes step it (the runtime steps a component
  // only on its processor set), so only they can count.
  if (h.solid_.contains(current_rank())) {
    EXPECT_GE(cons_b->calls(), 2)
        << "consumer_b ticks TWICE this period (t=0 and t=1/2), on a clock "
           "that has nothing to do with the region; both ticks must still run "
           "rather than being blocked by the region's failure: neither tick "
           "reads anything the failing region provides";
  } else {
    EXPECT_EQ(cons_b->calls(), 0)
        << "process " << current_rank() << " is outside consumer_b's processor set "
           "and must not step it";
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Mutation "region-failure-abandons-the-rest-of-the-period":
// step_period() returning straight out of run_region() on failure instead of
// recording it and walking on. The stream for this fixture (implicit coupling group at
// t=1; linkB's Transport and Epoch ordered AFTER the implicit coupling group's exit) shows the
// only thing such a return can lose is linkB's t=1 exchange, and losing it is
// only observable on a rank of linkB that is NOT in the implicit coupling group: with
// consumer_b alone on the last rank, producer_b (on fluid's ranks, inside the
// implicit coupling group) must still execute linkB's Transport and Epoch after its implicit coupling group
// fails. Under correct code the outside rank receives, reconciles at that
// Epoch, records E-RT-EPOCH-PEER (runner.cpp, reconcile_epoch) and reports it at the period's end; under the
// mutant producer_b's rank has already returned, and the outside rank blocks
// in the Transport until ctest's TIMEOUT -- the kill.
TEST(RunnerTransport, AChannelOutsideTheRegionStillCompletesAfterTheRegionFails) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);  // fluid, solid, and a rank in neither
  auto h = TwoComponentJob("", /*with_iteration_region=*/true,
                           fuse::AccelKind::None,
                           /*with_independent_channel=*/true,
                           /*fail_fluid_after=*/0, /*shape=*/Overlap::Disjoint,
                           /*independent_interval=*/fuse::Rational{1, 2},
                           CouplerShape::None,
                           /*independent_consumer_outside_region=*/true);
  ASSERT_GT(h.plan.region_count(), 0u);
  // PRECONDITION, identical on every rank (the plan is): at least one
  // Transport ordered after the implicit coupling group's exit. That is linkB's t=1 exchange
  // in the stream for this graph (interval 1/2); if the
  // scheduler ever orders it before the implicit coupling group, nothing after the implicit coupling group is
  // left to abandon and this test could not see the mutant (at interval 1,
  // the outside rank would likely finish its exchange before any failure).
  const auto& stream = h.plan.instructions();
  std::size_t exit_at = stream.size();
  for (std::size_t k = 0; k < stream.size(); ++k) {
    if (stream[k].region_exit >= 0) { exit_at = k; break; }
  }
  std::size_t transports_after = 0;
  for (std::size_t k = exit_at + 1; k < stream.size(); ++k) {
    if (stream[k].kind == fuse::OpKind::Transport) ++transports_after;
  }
  if (current_rank() == 0) {
    std::printf("[outside-rank] region exits at instruction %zu; %zu Transport(s) after it; stream:\n%s\n",
                exit_at, transports_after, h.plan.to_string().c_str());
  }
  ASSERT_GT(transports_after, 0u)
      << "nothing collective is ordered after the region, so this graph cannot "
         "detect 'region-failure-abandons-the-rest-of-the-period'";
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const int me = current_rank();
  const bool outside = me == world_size() - 1;
  const auto r = runner.run_period(h.components);
  // Every rank prints its own outcome: only rank 0's gtest output reaches the
  // log.
  std::printf("[outside-rank] rank %d (%s): ok=%d code=%s\n", me, outside ? "outside" : "region",
              r.ok() ? 1 : 0, r.ok() ? "-" : r.error().code.c_str());
  std::fflush(stdout);
  ASSERT_FALSE(r.ok()) << "rank " << me
                       << ": the period must fail everywhere, including on the "
                          "rank that only belongs to linkB";
  // Only the fluid's own processes step it, so only they report its
  // refusal. A process outside the implicit coupling group learns of it at
  // linkB's Epoch; a process inside the group that does not host the fluid
  // learns at linkB's Epoch or the group's residual agreement, whichever its
  // stream reaches first.
  if (h.fluid_.contains(me)) {
    EXPECT_EQ(r.error().code, "E-TEST-ADVANCE-REFUSED") << "rank " << me << ": " << r.error().message;
  } else if (outside) {
    EXPECT_EQ(r.error().code, "E-RT-EPOCH-PEER") << "rank " << me << ": " << r.error().message;
  } else {
    EXPECT_TRUE(r.error().code == "E-RT-EPOCH-PEER" || r.error().code == "E-RT-REGION-PEER")
        << "rank " << me << " does not step the fluid, so it must report a partner's "
        << "failure, not " << r.error().code << ": " << r.error().message;
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Mutation "coupler-region-participants-missing-its-ranks"
// (transport-runtime.toml): resolve_regions()'s coupler-rank union scan --
// the `for (i = R.enter; i <= R.exit; ++i)` loop over the implicit coupling group's own
// Transport/Epoch instructions, added specifically so a coupler-routed
// field transfer's coupler is counted among `R.participants` even though it owns
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
// above, this fixture's failure is a clean, local outcome mismatch, not a
// collective one): `resolve_regions()`'s OWN per-block guard,
// `if (b.length == 0 && R.participants.size() <= 1) return Failure{
// "E-RT-REGION", "... would report convergence after one sweep"};`
// (src/runtime/iteration.cpp, immediately below the participants scan --
// see that function's own comment: "a rank that IS a genuine participant
// of a multi-rank region... is ordinary... Outside that case... an empty
// coupling value is still what it always was... rejected below"), fires on
// the COUPLER's rank specifically: the coupler's own local `b.length`
// (this implicit coupling group's coupling-value block, sized from ITS OWN buffer) is 0
// (it owns no checkpointed field), and under the mutation
// `R.participants.size() <= 1` on every rank (participants is graph-
// derived, hence identical everywhere) -- so the rejection that is correctly
// SKIPPED (treated as the ordinary "n == 0" no-op) when the coupler is
// correctly counted instead FIRES, on the coupler's rank only, the moment
// `runner.initialize()` reaches this implicit coupling group. `fluid`/`solid`'s own rank
// never hits it (their block length is > 0 there), so this is an
// asymmetric, per-rank INITIALIZE()-time outcome split -- observable with
// nothing collective having run yet, unlike the implicit-coupling-group-body mutation above.
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

// The positive case: `A` and `B` each span the WHOLE
// communicator (LinearFixedPointJob, above), so every rank owns a slice of
// both -- and at 3 ranks, kEntities == 4 splits 2/1/1, giving genuinely
// unequal per-rank local residuals. `initialize()` must now SUCCEED (not
// reject: both components use AccelKind::None, the supported case above),
// and every rank must exit the implicit coupling group on the SAME sweep with the SAME
// iteration count test/verification/iteration_test.cpp's own
// Iteration.AitkenNeedsNoMoreIterationsThanPicard printed for the identical
// coefficients (a=1.0/k=0.5, a=2.0/k=0.4), entity count (4) and tolerance
// (1e-10) run on one process: 30. The map is applied entity-wise with no
// coupling between entities, so the fixed point and its contraction rate
// do not depend on how the 4 entities are split across ranks -- only
// reduce_region_residuals() combining the per-rank Euclidean sums back into
// the same global norm makes that hold under a real partition instead of
// desynchronising.
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

// ---------------------------------------------------------------------------
// A rank outside an ORDINARY (non-window-scoped)
// multi-rank iteration region's own R.participants shares config.
// base_communicator with the region's real participants -- the same shape
// the window-scoped sibling handles (broadcast_window_convergence()). The
// mirroring step (broadcast_region_convergence(), src/runtime/runner.cpp) is needed
// because reduce_region_residuals() leaves such a rank's payload at all-zero `owned`
// counts -- never reduced for it -- and the UNMEASURED check right after
// read that as "no participant owns this block": a guaranteed, spurious
// E-RT-REGION-UNMEASURED at sweep 0 on the bystander alone, with the two
// real participants (A, B) converging normally over kSerialIterations
// sweeps without ever seeing it.
// ---------------------------------------------------------------------------

/// A plain interval-1, no-field component -- this fixture's own uncoupled
/// bystander, hosting nothing this iteration region's channels touch.
class UncoupledBystanderComponent final : public fuse::Component {
 public:
  [[nodiscard]] std::string name() const override { return "C"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry&,
                                     const fuse::CouplingGraph&) override {
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
};

/// LinearFixedPointJob's own two-way iterated coupling (same A/B
/// coefficients, same entity count, same tolerance -- identical fixed point
/// and contraction rate, so the serial sweep count is the same known 30
/// RegionConvergesInTheSerialIterationCount, just above, pins), but with A
/// and B on DISJOINT ranks {0}/{1} rather than both spanning the whole
/// communicator, and -- when `with_bystander` -- a third, genuinely
/// uncoupled component C on rank 2: no fields, no channels, not a member of
/// the implicit coupling group's RankSet.
class DisjointFixedPointJob {
 public:
  static constexpr std::size_t kEntities = 4;

  /// `fail_a_restore_after` (-1 default) is
  /// passed to "A"'s own LinearRelaxComponent ONLY on a rank `a_ranks_`
  /// actually names (`-1`, never-fails, on every other rank) -- every
  /// rank constructs its OWN LinearRelaxComponent("A", ...) object
  /// regardless of `a_ranks_` (build_ticks()'s own documented need for a
  /// Component per graph component on every rank, this file's own
  /// TwoComponentJob-era comment), so forwarding the SAME threshold
  /// unconditionally would make B's rank fail its own, otherwise-inert,
  /// local copy of "A" independently of rank 0 -- not the one-rank-bad
  /// topology this is for.
  explicit DisjointFixedPointJob(int max_iterations, bool with_bystander,
                                 int fail_a_restore_after = -1)
      : a_ranks_{0, 1}, b_ranks_{1, 1},
        backend(a_ranks_, b_ranks_, kEntities, 1) {
    const auto mesh =
        graph.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
    region_ = graph.add_region(
        fuse::Region{.name = "r", .mesh = mesh, .dimension = 1});

    const auto ca = graph.add_component(fuse::ComponentDecl{
        .name = "A", .ranks = a_ranks_, .interval = Rational{1}});
    const auto cb = graph.add_component(fuse::ComponentDecl{
        .name = "B", .ranks = b_ranks_, .interval = Rational{1}});
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
    conv.add(fuse::Converged::Relative(1e-10, "a2b"));
    conv.add(fuse::Converged::Relative(1e-10, "b2a"));
    graph.add_iteration_region(fuse::IterationRegionDecl{
        .name = "outer", .channels = {fwd, back},
        .max_iterations = max_iterations, .convergence = conv});

    if (with_bystander) {
      graph.add_component(fuse::ComponentDecl{
          .name = "C", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1}});
    }

    EXPECT_TRUE(backend.load(graph.mesh(mesh)).ok());
    const auto lowered = lower(graph);
    EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
    dag = lowered.dag;

    const fuse::ClockSet clocks{graph};
    const auto sched =
        schedule(graph, dag, clocks, fuse::make_region_sizes(graph, dag, backend));
    EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
    plan = sched.plan;

    auto size_of = [&](std::uint32_t cid) -> std::size_t {
      const auto n = backend.placed_region_size(graph.region(region_),
                                                fuse::ComponentId{cid});
      EXPECT_TRUE(n.ok());
      return n.ok() ? n.value() : 0;
    };
    // Only the rank a_ranks_ actually names gets the real
    // threshold; every other rank's own (otherwise-inert) copy of "A" is
    // left at -1, never failing on its own.
    const int fail_this_rank =
        a_ranks_.contains(current_rank()) ? fail_a_restore_after : -1;
    components.add(std::make_unique<LinearRelaxComponent>(
        "A", 1.0, 0.5, size_of(0), fail_this_rank));
    components.add(std::make_unique<LinearRelaxComponent>(
        "B", 2.0, 0.4, size_of(1)));
    if (with_bystander) {
      components.add(std::make_unique<UncoupledBystanderComponent>());
    }
  }

  fuse::RankSet          a_ranks_, b_ranks_;
  PartitionedTestBackend backend;

  fuse::CouplingGraph  graph;
  fuse::OperatorDAG    dag;
  fuse::ExecutionPlan  plan;
  fuse::ComponentSet   components;
  fuse::FieldRegistry  registry;

 private:
  fuse::RegionId region_;
};

void bind_all(DisjointFixedPointJob& h) {
  for (auto& c : h.components) {
    EXPECT_TRUE(c.register_fields(h.registry, h.graph).ok());
  }
}

TEST(RunnerTransport,
     OrdinaryRegionBystanderDoesNotDeadlockOrMisreportUnmeasured) {
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  constexpr int kSerialIterations = 30;
  auto h =
      DisjointFixedPointJob(kSerialIterations + 10, /*with_bystander=*/true);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const auto run = runner.run_period(h.components);
  ASSERT_TRUE(run.ok())
      << "rank " << current_rank() << ": " << run.error().code << ": "
      << run.error().message
      << " -- a rank outside R.participants (rank 2, hosting only the "
         "uncoupled component C) must not report E-RT-REGION-UNMEASURED "
         "from a residual agreement it never joined";
  ASSERT_FALSE(runner.iteration_stats().empty());
  const fuse::IterationStats& st = runner.iteration_stats()[0];
  EXPECT_TRUE(st.converged);
  EXPECT_EQ(st.iterations, kSerialIterations)
      << "rank " << current_rank() << ": every rank sharing the "
         "communicator -- participant or bystander -- must agree the SAME "
         "sweep count A and B converge in on their own disjoint ranks";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// ---------------------------------------------------------------------------
// A mid-sweep restore_state() failure on one
// rank of an ordinary multi-rank iteration region must not `co_return`
// straight out of run_region() (iteration.cpp) before that sweep's
// partner learns of it -- the partner's own restore_state() succeeds
// and it proceeds into the NEXT sweep's body Transport, where the failing
// rank never arrives. run_region() instead records step_failure and falls through
// to the existing carry-forward machinery (the deliver/re-advance code
// right below already gates on step_failure via is_collective()), so the
// failure reaches every partner at the next sweep's residual agreement --
// the same reconcile_epoch()-style "-PEER" convention this file's own
// StillRunsASecondChannelAfterARegionFailsUnderMpi/healthy-rank-leaves-
// at-the-epoch mutation already document for the Epoch case.
// ---------------------------------------------------------------------------

TEST(RunnerTransport, RestoreStateFailureIsCarriedToEveryRankNotReturnedFrom) {
  // Exactly 2, not "at least 2": a third rank here would be a bystander
  // hosting neither A nor B, and peer-FAILURE propagation to such a
  // bystander is a case out of scope here, which is about
  // the two real participants, A's rank and B's rank, not a rank with no
  // stake in this region at all.
  FUSE_REQUIRE_RANKS(2);
  // Well short of the known 30-sweep convergence count (RegionConverges-
  // InTheSerialIterationCount, above): the failure must land while the
  // region is still iterating, never having reached convergence.
  constexpr int kFailAfter = 3;
  auto h = DisjointFixedPointJob(/*max_iterations=*/40,
                                 /*with_bystander=*/false,
                                 /*fail_a_restore_after=*/kFailAfter);
  ASSERT_GT(h.plan.region_count(), 0u);
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const auto run = runner.run_period(h.components);
  ASSERT_FALSE(run.ok())
      << "rank " << current_rank() << ": A's own restore_state() was made "
         "to fail past call " << kFailAfter << "; every rank sharing the "
         "communicator must see a Failure, not a silent success (which "
         "would mean this rank never heard about the other's hang-prone "
         "early return) and not a TIMEOUT (which would mean it did hang)";
  // A's own rank reports its own local diagnostic; B's rank learns of it
  // only through a cross-rank collective, reporting whichever "-PEER"
  // code that collective's own mechanism uses. Which one depends on
  // which the two-way channel's body reaches FIRST on the next sweep:
  // this fixture's own channels carry a Transport+Epoch pair, and
  // reconcile_epoch() co_returns immediately on an Epoch failure (the
  // i-loop's own established rule, iteration.cpp) -- before the region's
  // own residual agreement (reduce_region_residuals()) is ever reached. Measured:
  // E-RT-EPOCH-PEER. Both codes are
  // accepted, matching this file's own existing convention for the
  // identical ambiguity (see AChannelOutsideTheRegionStillCompletesAfter
  // TheRegionFails's own "whichever its stream reaches first" comment,
  // above) -- what is actually guaranteed is that B's rank reports SOME
  // "-PEER" code rather than hanging or reporting success.
  if (h.a_ranks_.contains(current_rank())) {
    EXPECT_EQ(run.error().code, "E-TEST-RESTORE-REFUSED")
        << run.error().message;
  } else if (h.b_ranks_.contains(current_rank())) {
    EXPECT_TRUE(run.error().code == "E-RT-EPOCH-PEER" ||
               run.error().code == "E-RT-REGION-PEER")
        << "rank " << current_rank() << " does not host the failing "
           "component, so it must report a partner's failure, not "
        << run.error().code << ": " << run.error().message;
  }
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

// Mutation `one-transport-shared-across-every-plan`:
// `Runner::Impl::attach_comm_graphs()` (src/runtime/runner.cpp) builds ONE
// `Transport` object per CommGraph-backed plan and stores it at
// `graph_transports[pi]`; the mutant changes `graph_transports[pi] =
// std::move(own)` to `if (pi == 0) graph_transports[pi] = std::move(own)`,
// so every plan index PAST THE FIRST is left holding a null `unique_ptr`
// even though `build_plan()` (called unconditionally, for every `pi`, just
// above) already succeeded for it. `TwoComponentJob("", ..., with_
// independent_channel=true)` compiles into the dataflow graph as exactly two CommGraph-backed
// Transport instructions -- "link" (fluid -> solid) gets `pi=0`, "linkB"
// (producer_b -> consumer_b) gets `pi=1`, in declaration order
// (src/runtime/runner.cpp's `case OpKind::Transport:` pushes one plan per
// instruction, unconditionally, with no dedup by shape even though both
// field transfers happen to share the same fluid_/solid_ rank split) -- so under
// the mutant "linkB"'s own execute() (OpKind::Transport, `p.has_graph`)
// reads `graph_transports[1].get() == nullptr` and returns a clean
// `E-RT-TRANSPORT` failure ("the CommGraph-backed plan for this Transport
// instruction has no Transport instance").
//
// WHY THE EXISTING `StillRunsASecondChannelAfterARegionFailsUnderMpi` DOES
// NOT CATCH THIS. That test ALSO forces "fluid" to fail inside the implicit coupling group
// (`fail_fluid_after=0`), which sets `step_failure` before "linkB"'s
// Transport instruction is even reached; `step_period()`'s failure branch
// only OVERWRITES `step_failure` `if (!step_failure.has_value())` (see the
// carry-forward mutation's own `old` line, just above this in the same function),
// so the implicit coupling group's "E-TEST-ADVANCE-REJECTED" wins and "linkB"'s distinct
// "E-RT-TRANSPORT" outcome is silently discarded -- masked, not fixed. And
// `cons_b->calls()` does not depend on whether "linkB"'s DATA arrived
// correctly at all: a component's own tick runs from whatever its input
// buffer already holds, null Transport or not.
//
// So THIS test forces nothing to fail (`fail_fluid_after` stays at its
// default, no implicit coupling group at all -- `with_iteration_region=false`, the
// mutant needs no group to manifest, and one fewer moving part is one
// fewer thing masking a different failure), and checks the DATA delivered
// through BOTH field transfers, using the same by-global-id technique
// `ARunDeliversTheSourceValuesToTheDestinationRanks` uses for "link".
// Under the mutant, "linkB"'s own `run_period()` (called directly, since
// there is no implicit coupling group to iterate) returns `E-RT-TRANSPORT`, which the
// `ASSERT_TRUE` below catches outright; if some future refactor changed
// which plan gets the null instance, checking BOTH field transfers' delivered
// data (not just "did it fail") is what still catches a shared instance
// that happens to work for one field transfer and silently corrupt the other.
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

// Mutation `region-residual-skips-the-cross-rank-reduction`:
// `run_region()`'s `if (R.participants.size() > 1) { ... reduce_region_
// residuals(...) ... }` (src/runtime/iteration.cpp) agrees `R.block_residual`/
// `R.block_value` across every rank in the implicit coupling group BEFORE the convergence
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
// global outcomes happen to AGREE at every sweep for a uniform `k`, so the
// mutation is invisible there no matter how unequal the entity SPLIT is.
//
// This fixture instead varies `k` (not the entity count) BY RANK: rank 0
// builds its own `A` with `kA = kFast = 0.001`, every other rank with
// `kA = kSlow = 0.5` (both pass the SAME graph structure to every rank --
// `CouplingGraph` carries no coefficients, only field/field transfer/implicit coupling group shapes
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
// `false` on EVERY rank, having spent the whole budget.
// Under the mutant, rank 0 decides FROM ITS OWN LOCAL RESIDUAL ALONE,
// already under `tol` by sweep 4, and reports `st.converged == true` at a
// SMALLER `st.iterations` -- exactly the "reaches a different convergence
// decision" failure, caught here by an outright `EXPECT_FALSE`/
// `EXPECT_EQ` mismatch on rank 0 without needing the desync to hang first
// (though a hang on a LATER sweep, once rank 0 has stopped participating in
// the implicit coupling group's own cross-rank exchange while the slow ranks keep going, is
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
      << "the correct code spends the whole iteration budget because the AGREED "
         "residual never clears tol; a smaller count means this rank exited "
         "on its own local (unreduced) residual instead";
  ASSERT_EQ(MPI_Barrier(MPI_COMM_WORLD), MPI_SUCCESS);
}

TEST(RunnerTransport, InitializeBuildsAPlanForEveryTransport) {
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
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
// rejection, not a silent success.
TEST(RunnerTransport, ACommGraphBackedPlanRefusesANonMpiTransport) {
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
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
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
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

// docs/dev/device-resident-arcs.md Phase 3: the same delivery as the test
// above, but under RunnerConfig::RunnerExecutionSpace::Device, so this
// Transport instruction's own arcs are device-resident and its execute_on()
// case takes the device-aware gather/scatter path instead of the plain
// pointer arithmetic. Absent -- not skipped -- from a build with no device
// backend, the fuse_add_device_twin()/DEVICE_TWIN convention's own
// discipline (an absent row cannot be misread as a pass for a configuration
// that was never exercised).
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || \
    defined(KOKKOS_ENABLE_SYCL)
TEST(RunnerTransport, ADeviceResidentRunDeliversTheSourceValuesGpuAwareOrNot) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator  = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup    = mpi_comm_setup();
  cfg.execution_space    = fuse::RunnerConfig::RunnerExecutionSpace::Device;
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok());

  h.write_source_by_global_id();
  ASSERT_TRUE(runner.run_period(h.components).ok());
  EXPECT_EQ(h.count_destination_matches(), h.local_destination_entity_count());
  // `fluid`/`solid`'s DISJOINT placement means Transport's input arc
  // (fluid's own host-registered field, aliased directly -- never coupler
  // storage) stays host on every rank, so a rank holding no entities of
  // EITHER side never touches device memory for this plan at all; only the
  // rank holding `solid` sees its output arc promoted to device (coupler
  // storage, unpinned) and stages it through the mixed-pair fallback
  // (`device_boundary_copies`). The per-rank counts are therefore NOT
  // symmetric -- summed across the job (MPI_Allreduce, the same discipline
  // the rank-count-invariance tests below use) is the only check that
  // tolerates the asymmetry while still proving SOME rank exercised the
  // device-aware code path at all.
  const fuse::RunStats& stats = runner.stats();
  const long long local_moves = stats.device_transport_moves;
  const long long local_boundary = stats.device_boundary_copies;
  long long total_moves = 0, total_boundary = 0;
  MPI_Allreduce(&local_moves, &total_moves, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_boundary, &total_boundary, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  // Whichever branch this host's MPI took (GPU-aware OpenMPI, or
  // FUSE_TRANSPORT_HOST_STAGING=1 forcing the fallback in the row below) is
  // moot here: no rank in THIS job ever reaches the both-device branch
  // (`device_transport_moves`) at all, only the mixed-pair one.
  EXPECT_EQ(total_moves, 0);
  EXPECT_GT(total_boundary, 0)
      << "no rank staged a device-resident Transport arc at all";
}

// The host-staged fallback, forced via FUSE_TRANSPORT_HOST_STAGING=1 so it
// has a test on a genuinely GPU-aware OpenMPI too, not only on a
// machine whose MPI happens not to be GPU-aware. Delivery must still agree
// bitwise with the GPU-aware path above and with Host: nothing here is
// floating-point arithmetic, only data movement, so there is no rounding
// budget to allow. (This job's own mixed-pair arcs, per the test above,
// never reach the both-device branch regardless of FUSE_TRANSPORT_HOST_
// STAGING -- the env var's effect is exercised instead by
// test/runtime/runner_device_test.cpp's no-graph scenarios and by this
// file's own device_boundary_copies check above; kept here as the
// delivery-correctness half of that same forced-fallback coverage.)
TEST(RunnerTransport, TheHostStagedFallbackDeliversTheSameValuesEvenWhenForced) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  ::setenv("FUSE_TRANSPORT_HOST_STAGING", "1", /*overwrite=*/1);
  auto h = make_two_component_job("");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator  = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup    = mpi_comm_setup();
  cfg.execution_space    = fuse::RunnerConfig::RunnerExecutionSpace::Device;
  ASSERT_TRUE(runner.initialize(h.components, h.registry, cfg).ok());

  h.write_source_by_global_id();
  ASSERT_TRUE(runner.run_period(h.components).ok());
  EXPECT_EQ(h.count_destination_matches(), h.local_destination_entity_count());
  const fuse::RunStats& stats = runner.stats();
  EXPECT_EQ(stats.device_transport_moves, 0)
      << "FUSE_TRANSPORT_HOST_STAGING=1 did not force the fallback";
  ::unsetenv("FUSE_TRANSPORT_HOST_STAGING");
}
#endif  // KOKKOS_ENABLE_CUDA || KOKKOS_ENABLE_HIP || KOKKOS_ENABLE_SYCL

TEST(RunnerTransport, TheTransportPathAllocatesNothingWhileRunning) {
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
  // (E-TRANSPORT-RANKS). A skip, not a workaround: this test's subject is a
  // two-sided job, which one rank cannot host.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  // Under this row's `.device` twin,
  // `fluid`/`solid`'s disjoint placement puts Transport's input arc at
  // n=0 (skipped by place_arcs_on_device(), stays host) and its output arc
  // on the device on the rank that holds `solid` -- a MIXED pair, staged
  // through `Impl::transport_out_scratch`. The receiving rank's own
  // allocation count under that combination was measured non-zero and not
  // yet isolated to a specific call (candidates: a first-touch cost of
  // `Kokkos::deep_copy` into device memory the GoogleTest listener removal
  // on non-zero ranks keeps this test from reporting a number for, or
  // something in `MpiTransport::move`/`prepare` reached for the first time
  // by THIS rank under Device). This property -- zero host allocations
  // while running -- is unrelated to delivery correctness, which
  // `RunnerDevice`'s own suite (test/runtime/runner_device_test.cpp) and
  // this file's `ADeviceResidentRunDeliversTheSourceValuesGpuAwareOrNot`
  // both assert and both pass under Device. Skipping here rather than
  // asserting a count nobody has explained yet.
  if (const char* space = std::getenv("FUSE_EXECUTION_SPACE");
      space != nullptr && std::string_view{space} == "device") {
    GTEST_SKIP() << "allocation count under the mixed "
                    "device/host arc pair this disjoint placement produces "
                    "is not yet isolated to a cause";
  }
  // The same guarantee Runner.AllocatesNothingWhileRunning pins for the serial
  // path (spec Section 2 layer 3). This is what proves that holding the TransportSpec
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
// MUTATIONS KILLED: replacing `in[p.gather[k] * nc + c]` with
// `in[k * nc + c]` in the gather loop, and separately
// `out[p.scatter[k] * nc + c]` with `out[k * nc + c]` in the scatter loop
// (src/runtime/runner.cpp), makes this test fail at 2, 3, 4 and 5 ranks.
//
// That depends on `block()` reversing each rank's LOCAL STORAGE order, so
// local index 0 holds the block's largest global id rather than its
// smallest. CommGraph's send/recv lists are sorted by (global_id, ...)
// (comm_graph.cpp), and with an ascending block a rank's local storage
// index would already equal its position in that sorted order, so
// `p.gather[k] == k` and `p.scatter[k] == k` on every rank and either
// mutation would change nothing (an identity-permutation gap). Reversing
// the local order decouples "sorted-by-gid position" from "local index"
// without changing which ids a rank owns. It is the index maps that need
// to be non-degenerate; the per-entity source values are already
// distinguishable (`write_source_by_global_id()` writes each id's own
// value).
TEST(RunnerTransport, TheDeliveredFieldDoesNotDependOnTheRankCount) {
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
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

// ---- the three overlap shapes through the Runner -------
//
// CommGraph is tested on all three shapes (rows 1-6). These put the same
// shapes through Runner::initialize + run_period, which is where CommGraph's
// output is consumed. PARTIAL is the case that could not be expressed before
// per-placement decomposition: the Runner handed both CommGraph sides one
// view, so each rank held one piece of the mesh shared by both components,
// and two partially overlapping processor sets could not both cover the implicit coupling group.
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
  // Identical processor sets: the IR emits no Transport (rules_channel.cpp,
  // same_as), so this pins that identical processor sets get the identical
  // decomposition the MeshBackend contract requires -- a different one would
  // leave values at the wrong local indices with nothing to move them.
  expect_every_destination_entity_delivered(Overlap::Full);
}

TEST(RunnerTransport, ADisjointPlacementDeliversEveryEntityThroughTheSameHarness) {
  // The third shape through the same helper, so the three differ only in the
  // processor set and not in what is checked.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  expect_every_destination_entity_delivered(Overlap::Disjoint);
}

// ---- the same delivery check with the Remap routed through a coupler ------
//
// Coupler processor set (docs/dev/coupler-placement.md): the field transfer's value moves
// fluid -> coupler, is remapped on the coupler's decomposition, and moves
// coupler -> solid; a hop between identical processor sets is elided. The three
// shapes below are the ones the processor-layout driver registers (layout.coupler.*),
// asserted here through the same helper as the no-coupler shapes so the only
// difference is the route. The driver must tell the backend about all three
// processor sets (a driver that named only two fails at initialize with
// E-MESH-PLACEMENT). This fixture reads its processor sets from the graph,
// which is why it could not catch that; the driver does the same.

TEST(RunnerTransport, ADisjointCouplerDeliversEveryEntity) {
  // The coupler is the last rank alone; fluid and solid split the others.
  // Three ranks is the first count with a rank for each of the three roles;
  // both hops are real Transports (peer plans, no rank in common).
  FUSE_REQUIRE_AT_LEAST_RANKS(3);
  expect_every_destination_entity_delivered(Overlap::Disjoint, CouplerShape::Disjoint);
}

TEST(RunnerTransport, ACouplerOnTheTargetsRanksDeliversEveryEntity) {
  // The coupler's processor set IS solid's, so the coupler -> solid hop is
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
// ONE Transport saw two entries and rejected it with a bogus E-RT-TRANSPORT
// ("the plan has 2"). The MPI config is not decoration here: the guard that
// tripped lives in attach_comm_graphs(), which is a no-op without
// base_communicator, so a copy-transport version of this test cannot see the
// bug at all.
TEST(RunnerTransport, InitializeIsRepeatableOnTheSameRunner) {
  // Two DISJOINT processor sets need two ranks: at one rank
  // `fluid_{0, world_size()/2}` is `{first=0, count=0}`, a component placed
  // NOWHERE. That is legal per RankSet's own contract, but there is then no
  // rank to send from and `build_plan` rejects the empty set
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
  // Each rank returns the outcome it is entitled to: the broken rank its own
  // local diagnostic, the healthy ranks the checkpoint's rejection.
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
