// Runner + CommGraph + MpiTransport under a real mpiexec. Not a
// fuse_add_mpi_test(): it calls MPI_Init, must be launched by mpiexec, and must
// fail the job rather than return on a per-rank failure.

#include <gtest/gtest.h>
#include <mpi.h>

#include <algorithm>
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

// ---------------------------------------------------------------------------
// A MeshView whose local entities are a contiguous, population-balanced slice
// of a shared global id space -- unlike SerialMeshBackend, genuinely different
// per rank. Point/centroid are unused: GlobalId matching never reads them.
// ---------------------------------------------------------------------------
class PartitionedTestMesh final : public fuse::MeshView {
 public:
  PartitionedTestMesh(int dim, std::vector<std::size_t> gids)
      : dim_(dim), gid_(std::move(gids)) {}
  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == dim_ ? gid_.size() : 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override {
    return fuse::Point{};
  }
  [[nodiscard]] double measure(int, std::size_t) const override { return 1.0; }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim != dim_ || i >= gid_.size()) {
      throw std::out_of_range("PartitionedTestMesh: index out of range");
    }
    return gid_[i];
  }

 private:
  int                      dim_;
  std::vector<std::size_t> gid_;
};

/// One mesh, split into a contiguous block per rank within whichever of
/// `fluid_`/`solid_` contains the calling rank; empty on a rank in neither.
/// `entity_set` on the `Region` is what Task 3's guard checks -- this backend
/// does not itself narrow by it, unlike SerialMeshBackend's real one
/// (src/mesh/serial_backend.cpp:438-457).
class PartitionedTestBackend final : public fuse::MeshBackend {
 public:
  PartitionedTestBackend(fuse::RankSet fluid, fuse::RankSet solid,
                         std::size_t n_entities, int dim)
      : fluid_(fluid), solid_(solid), n_(n_entities), dim_(dim) {}

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    int me = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    std::vector<std::size_t> gids;
    if (fluid_.contains(me)) {
      gids = block(fluid_, me);
    } else if (solid_.contains(me)) {
      gids = block(solid_, me);
    }
    meshes_.push_back(
        std::make_unique<PartitionedTestMesh>(dim_, std::move(gids)));
    names_.push_back(decl.name);
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(
      std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) {
        return fuse::MeshId{static_cast<std::uint32_t>(i)};
      }
    }
    return fuse::Failure{"E-MESH-NAME",
                       "no mesh named '" + std::string(name) + "'"};
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(
      fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    return meshes_[id.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> region_size(
      const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] std::string name() const override { return "partitioned-test"; }

 private:
  [[nodiscard]] std::vector<std::size_t> block(fuse::RankSet rs, int me) const {
    const int         pos  = me - rs.first;
    const std::size_t base = n_ / static_cast<std::size_t>(rs.count);
    const std::size_t rem  = n_ % static_cast<std::size_t>(rs.count);
    const std::size_t begin =
        static_cast<std::size_t>(pos) * base + std::min<std::size_t>(pos, rem);
    const std::size_t count =
        base + (static_cast<std::size_t>(pos) < rem ? 1 : 0);
    std::vector<std::size_t> gids(count);
    // REVERSED, not ascending: CommGraph's send/recv lists are sorted by
    // (global_id, ...) -- comm_graph.cpp:1125-1133 -- so an ascending block
    // here would put each entity's SORTED-MATCH position at exactly its own
    // LOCAL STORAGE index, making source_indices()/dest_indices() (and so
    // Runner's p.gather/p.scatter) the identity permutation on every rank at
    // every rank count this suite tries. That made TheDeliveredFieldDoesNot-
    // DependOnTheRankCount's mutation checks (Task 6 Steps 4-5, replacing
    // p.gather[k]/p.scatter[k] with plain k) pass when they should fail: a
    // dropped index map went undetected because k already equalled the right
    // answer. Reversing this rank's local order (local index 0 holds the
    // block's LARGEST global id, not its smallest) decouples "sorted-by-gid
    // position" from "local storage index" without changing which ids this
    // rank owns, which is all CommGraph's matching depends on.
    for (std::size_t k = 0; k < count; ++k) gids[k] = begin + (count - 1 - k);
    return gids;
  }

  fuse::RankSet fluid_, solid_;
  std::size_t   n_;
  int           dim_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::string>                     names_;
};

/// Registers one field of `count` doubles, nothing more. `build_ticks()`
/// (runner.cpp:1146-1164) requires a `Component` object for EVERY graph
/// component on EVERY rank regardless of that component's `RankSet` -- so
/// both "fluid" and "solid" must exist here even on a rank that physically
/// belongs to only one of them.
class StubComponent final : public fuse::Component {
 public:
  StubComponent(std::string name, fuse::FieldId field, std::size_t count)
      : name_(std::move(name)), field_(field), storage_(count, 0.0) {}

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
    return fuse::Result<void>::success();  // the test writes storage directly
  }

  [[nodiscard]] double*     data() noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

 private:
  std::string          name_;
  fuse::FieldId        field_;
  std::vector<double>  storage_;
};

/// Everything a test needs: the graph, the plan, the backend, and the two
/// stub components, plus helpers that read/write through the SAME region both
/// components' fields reference (Transport preserves region -- see the plan's
/// Scope Decision -- so there is exactly one to read).
class TwoComponentJob {
 public:
  /// `entity_set`, when non-empty, is set on the shared region. Task 3's
  /// E-RT-TRANSPORT-REGION test passes "wall"; every other test passes "".
  explicit TwoComponentJob(const std::string& entity_set)
      // MEMBER DECLARATION ORDER, NOT INITIALIZER-LIST ORDER, decides when
      // each of these actually runs. `backend` is declared below
      // fluid_/solid_/n_entities_/dim_ for exactly this reason: it is built
      // FROM them, so they must already hold their values.
      : fluid_{.first = 0, .count = world_size() / 2},
        solid_{.first = world_size() / 2,
              .count = world_size() - world_size() / 2},
        n_entities_(12), dim_(2),
        backend(fluid_, solid_, n_entities_, dim_) {
    const auto mesh =
        graph.add_mesh(fuse::MeshDecl{.name = "shared", .source = "test"});
    region_ = graph.add_region(fuse::Region{.name = "r", .mesh = mesh,
                                            .entity_set = entity_set,
                                            .dimension = dim_});

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
    graph.add_channel("link", src_field_, dst_field_);

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
    // NOTE: ASSERT_TRUE cannot be used here -- it expands to `return ...;`,
    // which only compiles in a void-returning function, and a constructor is
    // not one. EXPECT_TRUE matches every other check in this constructor.
    EXPECT_TRUE(count.ok());
    components.add(std::make_unique<StubComponent>("fluid", src_field_,
                                                    count.value()));
    components.add(std::make_unique<StubComponent>("solid", dst_field_,
                                                    count.value()));
  }

  void write_source_by_global_id() {
    auto* src = dynamic_cast<StubComponent*>(components.find("fluid"));
    const auto v = backend.view(graph.region(region_).mesh).value();
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
    const auto v = backend.view(graph.region(region_).mesh).value();
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
    return backend.region_size(graph.region(region_)).value();
  }
  [[nodiscard]] std::size_t global_destination_entity_count() const {
    return n_entities_;  // every id is wanted exactly once
  }

  // Declared in the order the constructor's initializer list must run them
  // (see the constructor comment): fluid_/solid_/n_entities_/dim_, then
  // `backend`, which is built from them. `graph`/`dag`/`plan`/`components`/
  // `registry` are default-constructed and filled in by the constructor
  // body, so their position relative to the others does not matter.
  fuse::RankSet          fluid_, solid_;
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
  int world = 0, me = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  ASSERT_GE(world, 2);

  // Two disjoint components spanning the job, as spec §6 mode 1 describes.
  const fuse::RankSet src{.first = 0, .count = world / 2};
  const fuse::RankSet dst{.first = world / 2, .count = world - world / 2};
  EXPECT_FALSE(src.overlaps(dst));

  MPI_Comm c = MPI_COMM_NULL;
  MPI_Comm_split(MPI_COMM_WORLD, src.contains(me) ? 0 : MPI_UNDEFINED, me, &c);
  if (src.contains(me)) {
    ASSERT_NE(c, MPI_COMM_NULL);
    int sz = 0;
    MPI_Comm_size(c, &sz);
    EXPECT_EQ(sz, src.count);
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

TEST(RunnerTransport, AWholeMeshRegionIsNotRefused) {
  // The control for the test above: without it, a guard that refused
  // everything would pass.
  auto h = make_two_component_job(/*entity_set=*/"");
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "copy";  // no communicator: no CommGraph, no refusal
  const auto r = runner.initialize(h.components, h.registry, cfg);
  EXPECT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;
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

TEST(RunnerTransport, InitializeBuildsAPlanForEveryTransport) {
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
  if (world_size() < 2) {
    GTEST_SKIP() << "needs a second rank to diverge from; world_size() is "
                 << world_size();
  }

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
