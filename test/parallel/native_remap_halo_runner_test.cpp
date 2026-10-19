// Scaled-bench track, D73 fix follow-up; RemapHalo track made this GREEN
// (round 1, owner decision 2026-10-18) and then moved its own execution
// point (round 2, same date). KNOWN_ISSUES.md D73: a general (non-nodal)
// weighted Remap between two components on DISJOINT MPI rank sets, backed
// by `fuse::native::NativeMeshBackend`, used to build a SILENTLY EMPTY
// conservative matrix (`RunStats::remap_weight_nonzeros == 0`) at every
// rank count above 1 -- no automatic remap-halo exchange, and no refusal
// either. The scaled-bench track's own interim fix made `Runner::
// initialize()` REFUSE by name (`E-RT-REMAP-NO-HALO`) instead of silently
// proceeding; the RemapHalo track replaced that refusal with an automatic
// halo, driven by the lowering's own `OpKind::BuildRemapHalo` instruction
// (fuse/ir/op.hpp). Round 1 had application code call `fuse::execute_
// remap_halo_instructions()` directly; round 2 (KNOWN_ISSUES.md D73 ROUND
// 3, D77 closed) moved that call INSIDE `fuse::make_region_sizes()`
// (fuse/backend/mesh_view.hpp) itself, so `build()` below no longer makes
// that call by hand for the GREEN row -- `make_region_sizes(g, dag,
// backend)` does it automatically, as the first thing it does.
//
// TWO rows: GREEN (`make_region_sizes()` builds the halo before
// `schedule()` reads any entity count; `initialize()` and a real period
// succeed, and the transferred field conserves) and RED (a caller that
// BYPASSES `make_region_sizes()` entirely -- hand-rolling its own
// `RegionSizes`, which the type system cannot prevent -- never builds a
// halo; `initialize()` must still refuse, with `E-RT-REMAP-HALO-NOT-
// BUILT`, the check `Runner::Impl::resolve_remap_halo()` performs for
// every `BuildRemapHalo` instruction in the plan). Round 1's RED row
// modeled "the two steps called in the wrong order"; round 2 retired the
// two-argument `make_region_sizes()` overload that would have let that
// happen, so round 2's own RED row models the one bypass still possible:
// not using `make_region_sizes()` at all.
//
// test/parallel/nodal_remap_runner.* and its serial twin exercise the
// NODAL path this same disjoint-placement shape also supports (an
// orthogonal, unaffected code path -- KNOWN_ISSUES.md D73's own header
// explains why the two differ); this file is the general-weighted path's
// own disjoint-placement row.

#include <fuse/ir/lowering.hpp>
#include <fuse/backend/native_backend.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;

namespace {

int world_size() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

/// A single real (`double`-backed) field, host-resident. The provider's
/// own buffer is filled with a known value (`fill`) by `register_fields`
/// so a successful `run_period()` (the GREEN row) has something non-trivial
/// to check conservation against; the consumer's starts at zero, as
/// before, and `initialize()` is expected to refuse before any period runs
/// on the RED row, so a 0-sized/zero-filled bind for a rank hosting
/// neither component is legal either way.
class Side final : public fuse::Component {
 public:
  explicit Side(std::string name) : name_(std::move(name)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
      auto& buf = bufs[g.field(f).name];
      buf.assign(local_count(g, f), fill);
      const auto r = reg.bind(
          f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = buf.size()},
                            fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
  std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> local_count;
  std::map<std::string, std::vector<double>> bufs;
  /// GREEN row only: the provider's own uniform fill value (every cell
  /// carries the SAME Extensive amount, so "sum over source" is simply
  /// `fill * (this rank's own owned source-cell count)`, summed again over
  /// every provider rank by the caller's own `MPI_Allreduce`). The
  /// consumer leaves this at its default (0.0): its own buffer starts at
  /// zero and `run_period()` is what is expected to change it.
  double fill = 0.0;

 private:
  std::string name_;
};

std::function<fuse::Result<void>(fuse::Transport&)> mpi_comm_setup() {
  return [](fuse::Transport& t) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
    if (mpi == nullptr) return fuse::Failure{"E-TEST-TRANSPORT", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
}

/// Disjoint, matching `scaled_device_benchmark.cpp`'s own `placement_for`:
/// provider owns the first half of the ranks, consumer the rest. Both
/// `fuse_add_mpi_test`'s `RANKS 2 3` split this evenly (one rank each at
/// world 2) and unevenly (one rank, two ranks, at world 3) -- either way,
/// genuinely disjoint, which is D73's own repro condition.
struct Placement {
  fuse::RankSet provider;
  fuse::RankSet consumer;
};
[[nodiscard]] Placement placement_for(int world) {
  const int half = world / 2;
  return Placement{fuse::RankSet{0, half}, fuse::RankSet{half, world - half}};
}

struct Harness {
  CouplingGraph                   graph;
  fuse::native::NativeMeshBackend backend;
  fuse::OperatorDAG               dag;
  fuse::ExecutionPlan             plan;
  ComponentSet                    components;
  FieldRegistry                   registry;

  explicit Harness(int rank, int world) : backend(rank, world) {}
};

/// Two differently-resolved native quad meshes (2x2 and 3x3 cells, both
/// plain FV at dimension 2) on disjoint rank sets, through a GENERAL
/// (non-nodal) weighted channel -- no `.remap_method()` override, so
/// `Runner::Impl::resolve_remap()` takes the branch this file's own D73
/// fix touches, never the nodal-distributed one. `Extensive` conservation
/// plus `RemapFlags::Conservative` (new for the GREEN row -- the RED-only
/// predecessor needed neither, since it never got as far as building real
/// weights) is what gives this channel the Runner's own `Constrain` check
/// (spec Section 5.1) in addition to this file's OWN before/after sum
/// comparison below.
///
/// `build_halo`: when true (the GREEN row), sizes buffers through `fuse::
/// make_region_sizes(g, dag, backend)` -- round 2's own call, which builds
/// every `BuildRemapHalo` instruction `dag` carries BEFORE reading any
/// entity count (fuse/backend/mesh_view.hpp's own header comment) -- so
/// the mesh is already widened by the time `schedule()` reads its sizes.
/// When false (the RED row), `schedule()` is instead given a hand-rolled
/// `RegionSizes` that reads the backend's entity counts DIRECTLY
/// (`backend.placed_region_size()`, the same primitive `make_region_
/// sizes()` itself calls, minus the halo build) -- modeling a caller that
/// bypasses `make_region_sizes()` entirely, the one bypass round 2's
/// retired two-argument overload no longer lets a caller reach by
/// calling the right functions in the wrong order. Either way, `Runner::
/// Impl::resolve_remap_halo()`'s own verification is the gate expected to
/// refuse when the halo genuinely was never built.
[[nodiscard]] std::unique_ptr<Harness> build(bool build_halo,
                                             fuse::Conservation conservation = fuse::Conservation::Extensive) {
  const int rank = world_rank();
  const int world = world_size();
  const Placement pl = placement_for(world);
  auto h = std::make_unique<Harness>(rank, world);
  CouplingGraph& g = h->graph;

  const auto mesh_provider =
      g.add_mesh(fuse::MeshDecl{.name = "provider_mesh", .source = "generated:quad:2x2"});
  const auto mesh_consumer =
      g.add_mesh(fuse::MeshDecl{.name = "consumer_mesh", .source = "generated:quad:3x3"});
  const auto r_provider =
      g.add_region(fuse::Region{.name = "r_provider", .mesh = mesh_provider, .dimension = 2});
  const auto r_consumer =
      g.add_region(fuse::Region{.name = "r_consumer", .mesh = mesh_consumer, .dimension = 2});

  const auto provider_id = g.add_component(
      fuse::ComponentDecl{.name = "provider", .ranks = pl.provider, .interval = Rational{1}});
  const auto consumer_id = g.add_component(
      fuse::ComponentDecl{.name = "consumer", .ranks = pl.consumer, .interval = Rational{1}});

  const auto out = g.add_field(
      provider_id, fuse::FieldSpec{.name = "out", .region = r_provider,
                                  .units = fuse::units::kKelvin,
                                  .conservation = conservation,
                                  .intent = fuse::Intent::Provides});
  const auto in = g.add_field(
      consumer_id, fuse::FieldSpec{.name = "in", .region = r_consumer,
                                  .units = fuse::units::kKelvin,
                                  .conservation = conservation,
                                  .intent = fuse::Intent::Requires});
  const auto link = g.add_channel("link", out, in);  // no remap_method(): the general weighted path
  g.channel(link).remap(fuse::RemapFlags::Conservative);

  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    EXPECT_TRUE(h->backend.load(g.mesh(fuse::MeshId{i}), g).ok());
  }

  // ROUND 4 (KNOWN_ISSUES.md D73, host measurement 2026-10-18): this call
  // was MISSING entirely. `cfg.transport_setup` (below, at Runner
  // construction) adopts a real MPI communicator for the TRANSPORT Runner
  // uses for Transport/Epoch -- a DIFFERENT object from `h->backend`, the
  // NativeMeshBackend `fuse::make_region_sizes()` calls immediately below
  // to build every `BuildRemapHalo` instruction. Without this,
  // `h->backend.communicator()` stayed `kNullComm` for the entire `build()`
  // call (native_backend.hpp's own doc on `set_communicator`: "`kNullComm`
  // ... means no covering-set exchange is possible ... treats every mesh
  // here as serial"), so `fuse::native::build_native_remap_halo`'s own
  // `ranks <= 1 || backend.communicator() == kNullComm` short-circuit
  // (native_halo.cpp) fired UNCONDITIONALLY regardless of how many real
  // MPI ranks this row actually ran under -- the real covering-set
  // exchange this test exists to exercise, and every
  // `FUSE_REMAP_HALO_TRACE` line inside it, never ran at all. Measured on
  // the owner's host 2026-10-18: `parallel.native_remap_halo_runner.{2,3}`
  // produced NEITHER of native_halo.cpp's own trace lines on either rank,
  // only runner.cpp's per-channel nonzero count -- exactly what a
  // short-circuited halo build looks like. `test/parallel/native_halo_mpi_
  // test.cpp:63` is the established pattern this call now matches: set the
  // BACKEND's own communicator right after construction/`load()`, before
  // anything that can build or need a halo.
  h->backend.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const fuse::ClockSet clocks{g};
  fuse::ScheduleResult sched;
  if (build_halo) {
    // D73 ROUND 2: `make_region_sizes()` builds every halo `h->dag`
    // carries BEFORE this call reads a single entity count -- the ONE
    // call site that can run the real geometry exchange at all (fuse/
    // backend/mesh_view.hpp's own header comment). No separate call to
    // `fuse::execute_remap_halo_instructions()` is made by this harness
    // any more; round 1 made that call directly, round 2 moved it here.
    sched = schedule(g, h->dag, clocks, fuse::make_region_sizes(g, h->dag, h->backend));
  } else {
    // RED row: bypass `make_region_sizes()` entirely with a hand-rolled
    // `RegionSizes` that reads `backend.placed_region_size()` directly --
    // the same primitive `make_region_sizes()` itself calls, minus the
    // halo build it now performs first. Both this fixture's regions are
    // plain FV (cell-shaped), so this naive lambda does not need `make_
    // region_sizes()`'s own nodal-vertex branch to answer correctly for
    // THIS test -- it is deliberately not a general replacement, only a
    // stand-in for "a caller that never goes through make_region_sizes()
    // at all", the one bypass left once the two-argument overload that
    // let round 1's RED row call the two steps in the wrong order was
    // retired.
    fuse::native::NativeMeshBackend* const backend_ptr = &h->backend;
    const fuse::RegionSizes naive_sizes =
        [&g, backend_ptr](fuse::RegionId id, int placement) -> std::size_t {
      const fuse::ComponentId who = placement >= 0
          ? fuse::ComponentId{static_cast<std::uint32_t>(placement)}
          : fuse::ComponentId{};
      const auto n = backend_ptr->placed_region_size(g.region(id), who);
      return n.ok() ? n.value() : std::size_t{0};
    };
    sched = schedule(g, h->dag, clocks, naive_sizes);
  }
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  // Real local counts, read off the backend's own placed region size --
  // `Side::register_fields` only needs a non-throwing buffer size (the RED
  // row never reaches a period: `initialize()` is expected to refuse
  // first); a 0-sized bind for a rank hosting neither component is legal.
  // Captures the backend's own STABLE heap address (h.get(), via the
  // Harness's unique_ptr) -- NOT `&h`, which names the local variable this
  // function is about to move out of on return, and would dangle the
  // instant it does.
  fuse::native::NativeMeshBackend* const backend_ptr = &h->backend;
  auto provider_side = std::make_unique<Side>("provider");
  provider_side->fill = 7.0;  // GREEN row: an arbitrary, non-zero, exactly-representable value
  provider_side->local_count = [backend_ptr, provider_id](const CouplingGraph& gg,
                                                           fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), provider_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  auto consumer_side = std::make_unique<Side>("consumer");
  consumer_side->local_count = [backend_ptr, consumer_id](const CouplingGraph& gg,
                                                           fuse::FieldId f) {
    const auto sz = backend_ptr->placed_region_size(gg.region(gg.field(f).region), consumer_id);
    return sz.ok() ? sz.value() : std::size_t{0};
  };
  h->components.add(std::move(provider_side));
  h->components.add(std::move(consumer_side));
  return h;
}

/// Higham, *Accuracy and Stability of Numerical Algorithms* (2nd ed.),
/// Lemma 8.4 / Theorem 8.1: the recursive (left-to-right) sum of `n`
/// nonnegative floating-point terms satisfies
/// `|fl(sum) - sum| <= gamma_{n-1} * sum(|x_i|)`, with the standard
/// `gamma_k = k*u / (1 - k*u)` for unit roundoff `u`. `n` here is the
/// largest term count ANY sum this test compares is built from: at most
/// `provider_cells` terms for the provider-side global reduction (one per
/// owned source cell, summed by `MPI_Allreduce` -- an Allreduce's own
/// pairwise/tree reduction has strictly fewer rounding steps than a naive
/// recursive sum of the same terms, so bounding by the recursive-sum gamma
/// is conservative, not tight), and at most `consumer_cells` for the
/// consumer side (same argument) plus, inside the matvec itself, at most
/// `provider_cells` nonzero weight contributions per target cell row (a
/// row cannot have more nonzeros than there are source cells to
/// intersect). `n = provider_cells + consumer_cells` is therefore an
/// upper bound on every summation this test's two checks perform,
/// generous rather than fitted: a real measurement (remap_plan.hpp's own
/// `build_remap_halo`-doc figures, 1.2e-15 at two ranks and 7.4e-16 at
/// four, from a different, much larger fixture) comes out far tighter.
[[nodiscard]] double higham_gamma(std::size_t n) {
  const double u = std::numeric_limits<double>::epsilon() / 2.0;
  const double k = static_cast<double>(n);
  return (k * u) / (1.0 - k * u);
}

}  // namespace

TEST(NativeRemapHalo, DisjointGeneralWeightedRemapConservesOnceTheHaloIsBuilt) {
  // D73's own repro shape, at whatever rank count this ctest row runs
  // (`fuse_add_mpi_test`'s own RANKS 2 3): provider and consumer never
  // share a rank, so every rank's local-only intersection search would
  // otherwise see only one side's geometry. With the halo built BEFORE
  // `schedule()` (this test's own `build(true)`), `initialize()` must now
  // succeed, and a real period must deliver a conserved (to the derived
  // tolerance above), genuinely non-empty transfer.
  ASSERT_GE(world_size(), 2) << "this row is registered RANKS 2 3; at 1 rank "
                                "provider and consumer coincide and nothing is disjoint";
  auto h = build(/*build_halo=*/true);

  // The CALLER's responsibility, not Runner::initialize()'s own (every
  // other Harness/run() helper in this tree does the same, immediately
  // before constructing Runner -- scaled_device_benchmark.cpp's own
  // run_one(), runner_device_test.cpp's run()).
  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  cfg.check_conservation = true;

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_TRUE(init.ok()) << (init.ok() ? std::string{} : init.error().message)
      << " -- expected success now that make_region_sizes() built the halo before "
         "reading any entity count; if this still refuses, either the automatic-halo "
         "fix regressed or this test's own build(true) stopped routing through "
         "fuse::make_region_sizes()";

  // "the weights' row sums": `RunStats::remap_weight_nonzeros` is the one
  // weight-matrix diagnostic the public `Runner` API exposes (`Runner::
  // stats()`) -- D73's own defect was `remap_weight_builds == 1` with
  // `remap_weight_nonzeros == 0` (a genuinely empty matrix); a literal
  // per-row sum would need a new debug accessor this track did not add
  // (open_questions names this). A non-empty matrix is necessary, not
  // sufficient, for "the row sums are right" -- the conservation check
  // right below is what actually measures that, through the one channel
  // the public API exposes it by (the delivered field's own sum).
  EXPECT_EQ(runner.stats().remap_weight_builds, 1);
  // ROUND 4 (owner request, 2026-10-18): PER RANK, not globally. The weight
  // matrix's own ROWS are the TARGET region's entities (build_supermesh_
  // weights: "row(target)"); a rank outside the target component's own
  // placement/RankSet (`placement_for(world).consumer` here) owns zero
  // target cells by construction (register_default_placements's own
  // "outside the processor set: an EMPTY view" rule, native_backend.cpp),
  // so its own local weight build correctly has zero rows and therefore
  // zero nonzeros -- not KNOWN_ISSUES.md D73's own defect (a target-hosting
  // rank's matrix coming back empty), but the ordinary, expected shape of a
  // rank that was never asked to build anything. Asserting `> 0`
  // unconditionally on every rank, as this row did through round 3, would
  // have FAILED on the provider-only rank even with the halo working
  // perfectly -- it happened not to, only because `world_size() == 2` in
  // this fixture's own even 50/50 split (`placement_for`) puts the
  // provider-only rank and the one assertion-running process at DIFFERENT
  // ranks, and gtest's `EXPECT_*` macros run (and would fail) independently
  // on whichever rank reaches them -- so the gap was real, just not yet
  // triggered by this specific RANKS 2 case the way RANKS 3 (one provider
  // rank, two consumer ranks, or vice versa) could.
  const int this_rank = world_rank();
  const Placement pl = placement_for(world_size());
  const bool this_rank_hosts_target = pl.consumer.contains(this_rank);
  if (this_rank_hosts_target) {
    EXPECT_GT(runner.stats().remap_weight_nonzeros, 0)
        << "KNOWN_ISSUES.md D73's own defect: a genuinely empty matrix, built but with "
           "nothing in it, on a rank that DOES host the target component's own cells";
  } else {
    EXPECT_EQ(runner.stats().remap_weight_nonzeros, 0)
        << "this rank owns no target cells at all (outside the target component's own "
           "placement), so its local weight build correctly has zero rows; a non-zero "
           "count here would mean rows were built for entities this rank does not own";
  }

  const auto run = runner.run_period(h->components);
  ASSERT_TRUE(run.ok()) << (run.ok() ? std::string{} : run.error().message);

  // Conservation: sum over target equals sum over source, to the derived
  // tolerance above. Every provider rank's own owned source cells carry
  // the SAME Extensive value (`Side::fill`); every consumer rank reads
  // back whatever `run_period()` actually delivered into its own
  // registered buffer. Both sums are reduced over the WHOLE base
  // communicator (not just the ranks that host either component), so a
  // rank hosting neither contributes exactly 0.0 to both and changes
  // neither sum.
  // `Side::bufs` is this rank's OWN registered storage, read back directly
  // rather than through the ABI.
  Side* const provider_ptr = dynamic_cast<Side*>(h->components.find("provider"));
  Side* const consumer_ptr = dynamic_cast<Side*>(h->components.find("consumer"));
  ASSERT_NE(provider_ptr, nullptr);
  ASSERT_NE(consumer_ptr, nullptr);
  Side& provider_side = *provider_ptr;
  Side& consumer_side = *consumer_ptr;

  const double local_source_sum = std::accumulate(provider_side.bufs["out"].begin(),
                                                   provider_side.bufs["out"].end(), 0.0);
  const double local_target_sum = std::accumulate(consumer_side.bufs["in"].begin(),
                                                   consumer_side.bufs["in"].end(), 0.0);

  double global_source_sum = 0.0, global_target_sum = 0.0;
  ASSERT_EQ(MPI_Allreduce(&local_source_sum, &global_source_sum, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);
  ASSERT_EQ(MPI_Allreduce(&local_target_sum, &global_target_sum, 1, MPI_DOUBLE, MPI_SUM,
                          MPI_COMM_WORLD),
            MPI_SUCCESS);

  constexpr std::size_t kProviderCells = 4;  // generated:quad:2x2
  constexpr std::size_t kConsumerCells = 9;  // generated:quad:3x3
  const double tol = higham_gamma(kProviderCells + kConsumerCells) *
                     std::max(global_source_sum, global_target_sum);
  EXPECT_NEAR(global_source_sum, global_target_sum, tol)
      << "conservative remap must deliver the source's total Extensive amount to the "
         "target, to the Higham recursive-summation bound derived above (tol=" << tol
      << "); global_source_sum=" << global_source_sum
      << " global_target_sum=" << global_target_sum;
  // Genuinely non-trivial: D73's own defect delivered a plausible-looking
  // ZERO (the empty matrix's own silent answer), which the bound above
  // alone would not catch if `fill` summed to something `tol`-close to
  // zero -- it does not (7.0 * 4 = 28.0), but this makes the intent
  // explicit rather than relying on that arithmetic coincidence.
  EXPECT_GT(global_target_sum, tol)
      << "the delivered total must be genuinely non-zero, not merely within tolerance "
         "of zero (KNOWN_ISSUES.md D73's own silent-empty-matrix failure mode)";
}

// ---------------------------------------------------------------------------
// KNOWN_ISSUES.md D82 (owner decision 2026-10-19): the SAME disjoint
// provider/consumer shape, under `Conservation::Intensive` instead --
// `build_supermesh_weights`'s own ORIGINAL (unchanged) target-measure
// normalisation, which makes a CONSTANT field reproduce exactly (not just
// conserve its total): every target cell, fully covered by a provider whose
// own field is a single uniform value, must read back that SAME value,
// because the weights in its row sum to 1 regardless of which individual
// source cells they came from.
// ---------------------------------------------------------------------------
TEST(NativeRemapHalo, DisjointGeneralWeightedRemapPreservesAConstantFieldExactlyUnderIntensiveConservation) {
  ASSERT_GE(world_size(), 2) << "this row is registered RANKS 2 3; at 1 rank "
                                "provider and consumer coincide and nothing is disjoint";
  auto h = build(/*build_halo=*/true, fuse::Conservation::Intensive);

  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();
  cfg.check_conservation = true;

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_TRUE(init.ok()) << (init.ok() ? std::string{} : init.error().message);

  const auto run = runner.run_period(h->components);
  ASSERT_TRUE(run.ok()) << (run.ok() ? std::string{} : run.error().message);

  Side* const consumer_ptr = dynamic_cast<Side*>(h->components.find("consumer"));
  ASSERT_NE(consumer_ptr, nullptr);
  Side& consumer_side = *consumer_ptr;

  // Every OWNED consumer cell this rank holds must read back the SAME
  // constant the provider's own field was filled with (`Side::fill`,
  // 7.0) -- to rounding, not to the Higham bound the Extensive test above
  // needs for a SUM: a single row-normalised matvec on a uniform input is
  // one weighted average per output, not an accumulation across many
  // terms, so a much tighter, near-epsilon tolerance is the honest one.
  constexpr double kFillValue = 7.0;
  for (double v : consumer_side.bufs["in"]) {
    EXPECT_NEAR(v, kFillValue, 1e-12)
        << "Intensive conservation (row-sum-to-1 normalisation) must reproduce a constant "
           "field exactly, regardless of how many source cells intersected this target cell";
  }
}

TEST(NativeRemapHalo, DisjointGeneralWeightedRemapRefusesWhenTheHaloWasNeverBuilt) {
  // The RED row this track's own fix must still produce: a caller that
  // bypasses `fuse::make_region_sizes()` entirely (`build(false)`'s own
  // hand-rolled `RegionSizes`, above) never builds a halo, and gets a
  // named refusal, not a silently empty matrix and not a crash.
  ASSERT_GE(world_size(), 2) << "this row is registered RANKS 2 3; at 1 rank "
                                "provider and consumer coincide and nothing is disjoint";
  auto h = build(/*build_halo=*/false);

  for (auto& comp : h->components) {
    const auto r = comp.register_fields(h->registry, h->graph);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  RunnerConfig cfg;
  cfg.transport = "mpi";
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport_setup = mpi_comm_setup();

  const auto init = runner.initialize(h->components, h->registry, cfg);
  ASSERT_FALSE(init.ok()) << "expected E-RT-REMAP-HALO-NOT-BUILT; initialize() succeeded "
                             "instead (the D73 automatic-halo fix regressed, or this test's "
                             "own build(false) built a halo some other way -- in either "
                             "case, update this test)";
  EXPECT_EQ(init.error().code, "E-RT-REMAP-HALO-NOT-BUILT");
}
