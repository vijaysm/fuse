// What happens to the other ranks when one of them fails mid-step.
//
// THE FAILURE THIS GUARDS AGAINST. Two components on disjoint rank sets
// exchange a field. One rank's component refuses its step. Without a
// rank-agreement point, that rank returns from run_period() immediately while
// its partners are already inside the transport's collective, waiting for a
// message it will never send. They block until the scheduler kills the job,
// and the log shows no failure on any of the ranks that were still running.
//
// WHAT THE LIBRARY DOES INSTEAD. The compiler put an `Epoch` after the data
// move (docs/dev/rank-agreement.md). The failing rank does NOT return at the
// point of failure: it carries the failure forward, skipping every instruction
// except the agreement points, so it still arrives at the reduction its
// partners are waiting at. Every participant then learns that someone failed
// and stops, with the failing rank reporting the cause and the others
// reporting E-RT-EPOCH-PEER.
//
// WHY THE TIMEOUT IN test/CMakeLists.txt MATTERS HERE. The regression this
// test exists to catch is a hang, and a hung test does not fail -- it stalls
// the leg until someone notices. The registration gives every rank count a
// TIMEOUT, so a regression is reported as a failure within two minutes.

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

 public:
  [[nodiscard]] fuse::FieldId  source_field() const { return src_field_; }
  [[nodiscard]] fuse::FieldId  target_field() const { return dst_field_; }
  [[nodiscard]] fuse::RegionId region() const { return region_; }
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


namespace {

/// Hands MPI_COMM_WORLD to the MPI transport. Same helper as
/// runner_transport_test.cpp: with a base communicator set, the transport is
/// the "mpi" one and it needs a communicator of its own.
std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& tr) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&tr);
    if (mpi == nullptr) {
      return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    }
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

/// A component that refuses its step on exactly one rank.
///
/// Refusing in `advance()` rather than at setup is deliberate: a setup failure
/// is already reconciled by the status-agreement checkpoint in initialize(),
/// and reusing that path would test the wrong mechanism. This one fails after
/// the plan is running, which is where only the agreement points can help.
class FailsOnOneRank final : public fuse::Component {
 public:
  FailsOnOneRank(std::string name, fuse::FieldId field, std::size_t n,
                 int failing_rank)
      : name_(std::move(name)), field_(field), storage_(n, 0.0),
        failing_rank_(failing_rank) {}

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph&) override {
    fuse::Layout layout;
    layout.count = storage_.size();
    return reg.bind(
        field_, fuse::FieldRef{storage_.data(), layout, fuse::MemorySpace::Host});
  }

  fuse::Result<void> advance(const fuse::StepContext&) override {
    ++steps;
    if (current_rank() == failing_rank_) {
      return fuse::Failure{"E-TEST-REFUSED",
                           "this rank's component refuses to advance"};
    }
    return fuse::Result<void>::success();
  }

  std::size_t steps = 0;

 private:
  std::string         name_;
  fuse::FieldId       field_;
  std::vector<double> storage_;
  int                 failing_rank_ = -1;
};

}  // namespace

TEST(EpochReconcile, TheJobNeedsAtLeastTwoRanks) {
  // Not a skip: with one rank there is nothing to reconcile, and the rank
  // counts this test is registered at all provide at least two.
  ASSERT_GE(world_size(), 2) << "launch with mpiexec -n 2 or more";
}

TEST(EpochReconcile, EveryRankStopsWhenOneOfThemFails) {
  auto h = make_two_component_job("");
  // The component on the LAST rank of the source side refuses. On two ranks
  // that is rank 0; on five it is rank 1. Either way it is a rank that takes
  // part in the exchange, which is the case that can hang.
  const int failing = h.fluid_.count > 0 ? h.fluid_.first + h.fluid_.count - 1 : 0;

  fuse::ComponentSet cs;
  const auto n = h.backend.region_size(h.graph.region(h.region())).value();
  cs.add(std::make_unique<FailsOnOneRank>("fluid", h.source_field(), n, failing));
  cs.add(std::make_unique<FailsOnOneRank>("solid", h.target_field(), n, -1));

  fuse::FieldRegistry reg;
  for (auto& c : cs) {
    const auto b = c.register_fields(reg, h.graph);
    ASSERT_TRUE(b.ok()) << b.error().code << ": " << b.error().message;
  }

  // All three, not just the communicator: with a base communicator the plan
  // gets rank-matching graphs attached, and a graph-backed plan requires the
  // "mpi" transport -- RunnerTransport.ACommGraphBackedPlanRefusesANonMpiTransport
  // asserts that refusal. Setting only base_communicator is how the first
  // version of this test failed, at initialize() rather than at the behaviour
  // it meant to check.
  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  const auto init = runner.initialize(cs, reg, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;

  const auto r = runner.run_period(cs);

  // EVERY rank fails. That is the whole point: the run does not continue on
  // the ranks that were fine, because continuing means blocking alone at the
  // next exchange.
  ASSERT_FALSE(r.ok()) << "rank " << current_rank()
                       << " reported success while rank " << failing
                       << " refused its step";

  if (current_rank() == failing) {
    EXPECT_EQ(r.error().code, "E-TEST-REFUSED")
        << "the failing rank reports its own cause, not a peer's";
  } else {
    EXPECT_EQ(r.error().code, "E-RT-EPOCH-PEER")
        << "a rank that did not fail reports that a partner did";
  }
}

TEST(EpochReconcile, AClearRunReconcilesAtEveryAgreementPoint) {
  // The same plan with nobody failing: the agreement points still run, and the
  // count is a fact worth pinning -- an implementation that skipped them when
  // all was well would pass every correctness test and hang on the first
  // failure.
  auto h = make_two_component_job("");
  bind_all(h);

  fuse::RunnerConfig cfg;
  cfg.transport         = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup   = mpi_comm_setup();
  Runner runner{h.graph, h.dag, h.plan, h.backend};
  const auto init = runner.initialize(h.components, h.registry, cfg);
  ASSERT_TRUE(init.ok()) << init.error().code << ": " << init.error().message;
  const auto ran = runner.run_period(h.components);
  ASSERT_TRUE(ran.ok()) << ran.error().code << ": " << ran.error().message;

  EXPECT_GT(runner.stats().epochs_reconciled, 0u)
      << "a rank taking part in the exchange reconciled no agreement point";
}
