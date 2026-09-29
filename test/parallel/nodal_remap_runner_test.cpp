// Nodal-remap track, round 3 (KNOWN_ISSUES.md D8): drives ONE real
// Runner/CouplingGraph pair through lower -> schedule -> initialize ->
// run_period, with a beam (nodal, ContinuousGLL order 1) / fluid (FV)
// field transfer pair exactly like the FSI channel uses -- "consistent" (H)
// forward, "consistent-transpose" (H^T D) backward -- at 1 to 5 real MPI
// ranks, with beam and fluid INDEPENDENTLY placed on disjoint,
// partially-overlapping or identical RankSets via the new
// MultiMeshGeometricBackend (test/support/partitioned_test_backend.hpp).
//
// ONE-RANK REFERENCE, COMPUTED IN THIS SAME TEST (not a separate job
// launch): `build_consistent_interp_weights` is called directly against a
// SERIAL (single, unpartitioned) `GeometricLineMesh` built with
// `num_ranks_in_set = 1` for both beam and fluid, at the SAME n_beam/
// n_fluid this test's real (possibly multi-rank) run uses -- exactly the
// same primitive `Runner::Impl::build_nodal_dist_remap` calls at
// `comm_handle == fuse::search::kNullComm`
// (`test/backend/nodal_remap_distributed_nullcomm_test.cpp` already proved
// those two null-comm and real-comm code paths agree). `fluid_velocity_in`
// is additionally checked against the closed form `vel(x) = a + b*x`
// directly (H is P1-exact for an affine field, general_interp.hpp's own
// exactness property), needing no reference computation at all.
//
// "BIT-IDENTICAL ACROSS RANK COUNTS", the property this file's own name
// promises: matching the SAME rank-count-independent reference at
// WHATEVER rank count this job happens to run (1 through 5, per this
// track's own parallel-matrix.toml entry) is the standard way this
// codebase already proves that property without two separate job launches
// compared against each other -- see
// nodal_remap_distributed_test.cpp's own header comment for the identical
// argument, made there for the underlying primitive this file exercises
// through the Runner instead.
//
// MPI CANNOT INITIALISE IN THIS SANDBOX (docs/dev/testing.md): every case
// here is compiled, never run, in the environment this file was written
// in. outcomes are a host action, same posture as every other MPI-gated
// case in this codebase.

#include <fuse/ir/lowering.hpp>
#include <fuse/backend/consistent_interp.hpp>
#include <fuse/backend/remap_plan.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/transport/transport.hpp>

#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

#include "../support/partitioned_test_backend.hpp"

using fuse::ComponentSet;
using fuse::CouplingGraph;
using fuse::FieldRegistry;
using fuse::Rational;
using fuse::Runner;
using fuse::RunnerConfig;
using fuse::StepContext;

namespace u = fuse::units;
inline constexpr fuse::Units kMetersPerSecond{.length = 1, .time = -1};

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

/// Registers ONE named field of `count` doubles, bound to the buffer this
/// class owns; `fill_` writes it once per `advance()`, exactly like
/// `run_consistent_pair`'s own local `Side` class
/// (test/verification/interface_transfer_test.cpp), generalised here to a
/// LOCAL (per-rank) size read from the actual placed `MeshView`, not a
/// hardcoded whole-mesh formula -- the one thing that must change to be
/// correct under a real multi-rank partition.
class Side final : public fuse::Component {
 public:
  Side(std::string name, std::function<void(Side&)> fill)
      : name_(std::move(name)), fill_(std::move(fill)) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
    for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
      auto& buf = bufs[g.field(f).name];
      buf.assign(local_count(g, f), 0.0);
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
  std::function<std::size_t(const CouplingGraph&, fuse::FieldId)> local_count;
  std::map<std::string, std::vector<double>> bufs;

 private:
  std::string name_;
  std::function<void(Side&)> fill_;
};

/// The serial (single, unpartitioned) reference this file's own header
/// comment describes: H via `build_consistent_interp_weights` on a
/// `num_ranks_in_set = 1` `GeometricLineMesh` for both sides, then H^T D
/// built by hand exactly as `build_consistent_weights_for_plan`'s own
/// transpose branch does (consistent_interp.hpp) -- nodal side = beam,
/// eval side = fluid, scaled by the fluid side's OWN cell measure since
/// an FV source is a density, not yet a load.
struct ReferenceResult {
  std::vector<double> beam_load;  // size n_beam + 1, indexed by GLOBAL beam vertex id
};

[[nodiscard]] ReferenceResult compute_reference(std::size_t n_beam, std::size_t n_fluid) {
  GeometricLineMesh ref_beam(n_beam, /*num_ranks_in_set=*/1, /*my_rank_in_set=*/0, 0.0, 1.0);
  GeometricLineMesh ref_fluid(n_fluid, 1, 0, 0.0, 1.0);

  // H: nodal = beam (src), eval = fluid cells (tgt, dim 1) -- the SAME
  // orientation the "consistent-transpose" field transfer's own H build uses
  // (build_consistent_weights_for_plan's comment: "nodal side = plan.tgt
  // (here, the beam), point side = plan.src (here, the fluid)").
  const auto h = fuse::build_consistent_interp_weights(ref_beam, ref_fluid, /*tgt_dim=*/1);
  if (!h.ok()) throw std::runtime_error("compute_reference: H failed: " + h.error().message);
  const fuse::RemapWeights& hw = h.value();

  fuse::RemapWeights ht;
  ht.n_rows = hw.n_cols;  // beam vertices
  ht.n_cols = hw.n_rows;  // fluid cells
  ht.row.reserve(hw.val.size());
  ht.col.reserve(hw.val.size());
  ht.val.reserve(hw.val.size());
  for (std::size_t k = 0; k < hw.val.size(); ++k) {
    const auto fluid_cell = static_cast<std::size_t>(hw.row[k]);
    const auto beam_vertex = static_cast<std::size_t>(hw.col[k]);
    const double scale = ref_fluid.measure(1, fluid_cell);  // density -> load
    ht.row.push_back(static_cast<std::int32_t>(beam_vertex));
    ht.col.push_back(static_cast<std::int32_t>(fluid_cell));
    ht.val.push_back(hw.val[k] * scale);
  }

  const std::vector<double> density(n_fluid, 1.0);  // uniform unit density
  ReferenceResult out;
  out.beam_load.assign(n_beam + 1, 0.0);
  const auto applied = fuse::apply_weights(ht, density.data(), out.beam_load.data(), 1);
  if (!applied.ok()) throw std::runtime_error("compute_reference: H^T D failed: " + applied.error().message);
  return out;
}

/// Runs the beam/fluid consistent + consistent-transpose field transfer pair
/// through a real, MPI-configured Runner, with beam on `beam_ranks` and
/// fluid on `fluid_ranks` (independently -- possibly disjoint, partially
/// overlapping, or identical). `vel(x) = a + b*x` (affine, exact for H);
/// the fluid traction density is a uniform 1.0, matching `compute_reference`
/// above exactly. Every rank of `MPI_COMM_WORLD` must call this.
void run_and_check(std::size_t n_beam, std::size_t n_fluid, fuse::RankSet beam_ranks,
                   fuse::RankSet fluid_ranks, double vel_a, double vel_b) {
  CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{.name = "beam_mesh", .source = "n/a"});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{.name = "fluid_mesh", .source = "n/a"});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
  const auto beam = g.add_component(
      fuse::ComponentDecl{.name = "beam", .ranks = beam_ranks, .interval = Rational{1}});
  const auto fluid = g.add_component(
      fuse::ComponentDecl{.name = "fluid", .ranks = fluid_ranks, .interval = Rational{1}});
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

  MultiMeshGeometricBackend backend;
  backend.add_mesh_spec("beam_mesh", {n_beam, beam_ranks, 0.0, 1.0});
  backend.add_mesh_spec("fluid_mesh", {n_fluid, fluid_ranks, 0.0, 1.0});
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  const auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  const auto local_count = [&](const CouplingGraph& gg, fuse::FieldId fid) -> std::size_t {
    const auto& fs = gg.field(fid);
    const auto mid = backend.mesh_id(gg.mesh(gg.region(fs.region).mesh).name);
    if (!mid.ok()) throw std::runtime_error("local_count: " + mid.error().message);
    const auto view = backend.view(mid.value());
    if (!view.ok()) throw std::runtime_error("local_count: " + view.error().message);
    const int dim = fs.disc == fuse::Discretization::ContinuousGLL ? 0 : 1;
    return view.value()->entity_count(dim);
  };

  auto b = std::make_unique<Side>("beam", [&](Side& me) {
    auto& v = me.bufs["beam_velocity"];
    const auto mid = backend.mesh_id("beam_mesh").value();
    const auto* view = backend.view(mid).value();
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = vel_a + vel_b * view->centroid(0, i).x;
  });
  auto f = std::make_unique<Side>("fluid", [&](Side& me) {
    auto& q = me.bufs["fluid_force"];
    for (double& x : q) x = 1.0;  // uniform unit density, matching compute_reference()
  });
  b->local_count = local_count;
  f->local_count = local_count;
  // Both raw pointers BEFORE the moves. The one-rank host segfault
  // (EXC_BAD_ACCESS at address 0x30) was `fp->bufs[...]` below, read through
  // a moved-from unique_ptr: `bp` had been taken, `fp` had not, and the
  // fluid buffers were then read through a null `Side*` plus the map's
  // offset. It showed only at one rank because every other rank count died
  // earlier, at initialize(), on E-RT-CONSERVATION-MPI.
  Side* bp = b.get();
  Side* fp = f.get();
  ComponentSet components;
  components.add(std::move(b));
  components.add(std::move(f));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  // A processor set that spans ranks needs a CommGraph-backed Transport, and the
  // Runner rejects to build one on the default 'copy' backend
  // (E-RT-TRANSPORT) -- which is what every cross-rank case of this fixture
  // hit on the host once the earlier rejections were out of the way. Same
  // convention as runner_transport_test.cpp's mpi_comm_setup(): name the
  // backend, and hand the transport its communicator when it is created.
  cfg.transport = "mpi";
  cfg.transport_setup = [](fuse::Transport& tr) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&tr);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
  // check_conservation stays at its default (ON). The Runner's Constrain
  // reduces both integrals over the base communicator (KNOWN_ISSUES.md D10,
  // 2026-09-25) -- before that it rejected a multi-rank target
  // (E-RT-CONSERVATION-MPI) and this fixture had to disable it.
  //
  // `abort_on_violation = true` is NOT the default -- and its absence is
  // exactly why mutation "conservation-check-not-reduced-across-ranks"
  // (build-evidence/mutations/nodal-fields.toml) survived here. Constrain
  // only REPORTS (runner.cpp's own comment: "Phase 4 reports; it does not
  // correct" -- the field values downstream are copied through unchanged
  // either way), so with `abort_on_violation` left at its default `false`,
  // a spurious per-rank violation caused by skipping the cross-rank
  // Allreduce (`ci.reduce == false`, which the mutation forces
  // unconditionally) only increments `runner.stats().conservation_violations`
  // -- which nothing below ever reads -- and `run_period()` still returns
  // `ok()`. Traced by hand and confirmed with a standalone harness linked
  // against this library (DisjointRankSets' own shape, no MPI, so
  // deliberately WITHOUT a base communicator -- the same `ci.reduce ==
  // false` branch the mutation forces): `conservation_violations` came back
  // 2 of 4 checks, `worst_conservation_error` 1.0 (100% relative), while
  // `run_period()` still reported `ok()` -- exactly the silent-survival
  // shape. Setting this true turns that same condition into a hard
  // `E-RT-CONSERVATION` failure (confirmed by the same harness: "integral
  // 10.400000 -> 0.000000" once `abort_on_violation` was flipped on), which
  // is what actually makes this test check the Runner's outcome against its
  // own MPI_Allreduce of the same integrals below -- the claim the comment
  // used to make here without this line being present to back it.
  cfg.abort_on_violation = true;
  {
    const auto r = runner.initialize(components, registry, cfg);
    ASSERT_TRUE(r.ok()) << "initialize: " << r.error().code << ": " << r.error().message;
  }
  // run_period() explicitly, twice -- the wall_force field transfer's own .lag(1)
  // needs a second coupling cycle to have a value to read, exactly like
  // run_consistent_pair's run(components, 2) (interface_transfer_test.cpp).
  for (int period = 0; period < 2; ++period) {
    const auto r = runner.run_period(components);
    ASSERT_TRUE(r.ok()) << "run_period: " << r.error().code << ": " << r.error().message;
  }

  // ---- checks -------------------------------------------------------------
  const auto beam_mid = backend.mesh_id("beam_mesh").value();
  const auto* beam_view = backend.view(beam_mid).value();
  const auto fluid_mid = backend.mesh_id("fluid_mesh").value();
  const auto* fluid_view = backend.view(fluid_mid).value();

  // (1) H exact for a linear field, on every rank's own local fluid cells --
  // pointwise, so ghosts don't even enter into it (fluid has none: cells are
  // a plain, non-overlapping partition).
  const std::vector<double>& fluid_in = fp->bufs["fluid_velocity_in"];
  for (std::size_t j = 0; j < fluid_in.size(); ++j) {
    const double x = fluid_view->centroid(1, j).x;
    EXPECT_NEAR(fluid_in[j], vel_a + vel_b * x, 1e-12) << "rank " << world_rank() << " fluid cell " << j;
  }

  // (2) Each LOCAL beam_load entry (owned or ghost -- apply_distributed_HT's
  // own contract overwrites every local row with the correct global total,
  // so a ghost must agree too) equals the one-rank reference at its GLOBAL
  // id -- "each shared beam vertex's load equals the one-rank result", and
  // implicitly "bit-identical across rank counts" (see file header).
  const ReferenceResult ref = compute_reference(n_beam, n_fluid);
  const std::vector<double>& beam_load = bp->bufs["beam_load"];
  for (std::size_t i = 0; i < beam_load.size(); ++i) {
    const std::size_t gid = beam_view->global_id(0, i);
    EXPECT_NEAR(beam_load[i], ref.beam_load.at(gid), 1e-12 * std::abs(ref.beam_load.at(gid)) + 1e-13)
        << "rank " << world_rank() << " beam local vertex " << i << " (global id " << gid << ")";
  }

  // (3) Global force conserved and interface power balanced to 1e-13
  // relative -- summing only OWNED entries on each side (a shared beam
  // vertex must count once, not once per holder), reduced with MPI_Allreduce
  // across every rank of MPI_COMM_WORLD, exactly like
  // nodal_remap_distributed_test.cpp's own conservation check.
  double local_sent = 0.0, local_received = 0.0, local_p_fluid = 0.0, local_p_beam = 0.0;
  const std::vector<double>& fluid_force = fp->bufs["fluid_force"];
  const double h_fluid = fluid_view->entity_count(1) > 0 ? fluid_view->measure(1, 0) : 0.0;
  for (std::size_t j = 0; j < fluid_force.size(); ++j) {
    local_sent += fluid_force[j] * h_fluid;
    local_p_fluid += fluid_in[j] * fluid_force[j] * h_fluid;
  }
  for (std::size_t i = 0; i < beam_load.size(); ++i) {
    if (!beam_view->is_owned(0, i)) continue;
    local_received += beam_load[i];
    local_p_beam += bp->bufs["beam_velocity"][i] * beam_load[i];
  }
  double total_sent = 0.0, total_received = 0.0, total_p_fluid = 0.0, total_p_beam = 0.0;
  MPI_Allreduce(&local_sent, &total_sent, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_received, &total_received, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_p_fluid, &total_p_fluid, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&local_p_beam, &total_p_beam, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  EXPECT_NEAR(total_sent, total_received, 1e-13 * std::abs(total_sent))
      << "H^T must conserve the total force globally";
  EXPECT_NEAR(total_p_fluid, total_p_beam, 1e-13 * (std::abs(total_p_fluid) + std::abs(total_p_beam)))
      << "interface power must balance globally";
}

}  // namespace

TEST(NodalRemapRunnerMpi, DisjointRankSets) {
  // Round 4 (KNOWN_ISSUES.md D8, defect 3, found by a real host MPI run):
  // `beam_ranks{0, (w+1)/2}` and `fluid_ranks{(w+1)/2, w-(w+1)/2}` are
  // disjoint by construction ONLY when both halves are non-empty; at
  // `w == 1`, `fluid_ranks` has count 0 -- the fluid component would be
  // placed on NO rank at all, the exact two-sided-processor-set-on-one-rank
  // shape commit 2383024 already fixed elsewhere in this codebase. An ad
  // hoc `if (fluid_ranks.size() == 0) GTEST_SKIP()` caught this at `w == 1`
  // specifically, but stated the requirement in terms of THIS test's own
  // arithmetic rather than the rank count a reader can check against
  // parallel-matrix.toml at a glance; `FUSE_REQUIRE_AT_LEAST_RANKS`
  // (test/parallel/runner_transport_test.cpp's own convention for exactly
  // this situation) says it the same way every other multi-rank-shape test
  // in this codebase does.
  FUSE_REQUIRE_AT_LEAST_RANKS(2);  // a disjoint split needs two non-empty halves
  const int w = world_size();
  const fuse::RankSet beam_ranks{0, (w + 1) / 2};
  const fuse::RankSet fluid_ranks{(w + 1) / 2, w - (w + 1) / 2};
  run_and_check(7, 11, beam_ranks, fluid_ranks, 0.3, 2.0);
}

TEST(NodalRemapRunnerMpi, IdenticalRankSets) {
  // `all{0, world_size()}` is never empty (even at w == 1, where both
  // components simply share the one rank the job has) -- so, UNLIKE
  // DisjointRankSets/PartiallyOverlappingRankSets above and below, this
  // case has no empty-RankSet construction to guard against, and none is
  // added here. A real host MPI run (round 4) reported this test failing
  // at world size 1 too, with a cause distinct from the other two (their
  // shared empty-RankSet defect does not apply here); reproducing it needs
  // a genuine MPI launch, which this sandbox cannot provide even at -np 1
  // (`mpirun -np 1` fails in HYDU_sock_listen; running the built MPI test
  // binary directly without mpirun aborts inside MPI_Init's own
  // debugger-attach teardown, `dbginit.c`'s `MPIR_Debugger_set_aborting`) --
  // see KNOWN_ISSUES.md D8 for the exact host command this still needs.
  const fuse::RankSet all{0, world_size()};
  run_and_check(7, 11, all, all, 0.3, 2.0);
}

TEST(NodalRemapRunnerMpi, PartiallyOverlappingRankSets) {
  // `beam_ranks{0, w-1}` and `fluid_ranks{1, w-1}` overlap on `[1, w-1)`,
  // which is only non-empty (and only genuinely "partial", rather than
  // disjoint) once `w >= 3` -- the same class of shape-needs-N-ranks
  // requirement as DisjointRankSets above, stated the same way.
  FUSE_REQUIRE_AT_LEAST_RANKS(3);  // needs a genuine overlap, not just two non-empty sets
  const int w = world_size();
  const fuse::RankSet beam_ranks{0, w - 1};      // ranks [0, w-1)
  const fuse::RankSet fluid_ranks{1, w - 1};     // ranks [1, w)
  run_and_check(8, 6, beam_ranks, fluid_ranks, -0.5, 1.5);
}

// Isolates resolve_remap's `both_on_one_rank` computation from
// `moves_placement` (mutation "nodal-trigger-misses-two-single-rank-
// processor sets", build-evidence/mutations/nodal-fields.toml).
//
// DisjointRankSets above cannot kill that mutation: its two field transfers both
// use a method that matches its nodal side's orientation exactly
// ("consistent" with a nodal source, "consistent-transpose" with a nodal
// target), and `remap_moves_placement()` (fuse/model/remap_kind.hpp) is
// therefore true for both during compilation into the dataflow graph, which makes rules_channel.cpp route
// them through the Epoch-only, processor-set-changing Remap path with NO
// Transport. At runtime that gives `in_rank_set != remap_rank_set`
// (`moves_placement == true`) unconditionally whenever the two components'
// declared ranks genuinely differ -- independent of, and always agreeing
// with, whatever `both_on_one_rank` computes. Traced by hand and confirmed
// with a standalone harness linked against this library (SerialMeshBackend,
// no MPI, no base_communicator; see this track's evidence log): applying
// the mutation to a fresh build left `DisjointRankSets`'s own two field transfers'
// outcome completely unchanged, because `moves_placement` alone already
// forces entry into the nodal-distributed branch.
//
// A field transfer whose method does NOT match its nodal side's orientation
// (`remap_moves_placement() == false`) is compiled into the dataflow graph the OLD way instead: a
// real Transport moves the value to the target's declared ranks first, so
// `in_rank_set == remap_rank_set` and `moves_placement == false`. On such a
// field transfer, `both_on_one_rank` is the ONLY gate deciding whether
// resolve_remap's nodal-parallel check runs at all -- exactly the gap
// DisjointRankSets leaves uncovered. This case needs no cross-rank
// communication to observe: the decision is read entirely from the
// DECLARED graph (component ranks, sizes, kinds), before any Transport or
// collective would run, so it is checked here declaratively, the same way
// `test/verification/interface_transfer_test.cpp`'s
// `BothSidesNodalOnMultipleRanksRefuses` checks an adjacent declared-only
// rejection without a real multi-rank launch -- correct code rejects by name
// (E-RT-REMAP-NODAL-PARALLEL) before ever touching the backend; this file's
// own header already establishes every case here is graph-declared and
// rank-count-independent for that reason.
//
// Confirmed by hand (base build configuration, standalone harness, no MPI): current code
// returns E-RT-REMAP-NODAL-PARALLEL ("source: 1, target: 1"); with the
// mutation applied, `both_on_one_rank` wrongly reports true for this
// disjoint pair, the whole nodal-parallel check is skipped, and control
// falls through to the OLD rank-local path, which -- because
// SerialMeshBackend hands every declared processor set the WHOLE (identically
// ordered) mesh regardless of rank -- reports E-RT-REMAP-SIZE instead ("the
// buffer plan and the mesh backend disagree about the same mesh entity set", 6 vs 5
// values) once it discovers the vertex- and cell-counts collide under a
// coincidence check that never should have run. Both outcomes are
// `!r.ok()`, so a bare `ASSERT_FALSE(r.ok())` cannot tell them apart --
// which is exactly why the mutation survived nowhere else; the code check
// below is the specific error CODE, mirroring
// `BothSidesNodalOnMultipleRanksRefuses`'s own convention for the same
// reason.
TEST(NodalRemapRunnerMpi, DisjointSingleRankPlacementRefusesMismatchedOrientation) {
  fuse::CouplingGraph g;
  const auto m_a = g.add_mesh(fuse::MeshDecl{.name = "a_mesh", .source = "generated:line:5"});
  const auto m_b = g.add_mesh(fuse::MeshDecl{.name = "b_mesh", .source = "generated:line:5"});
  const auto r_a = g.add_region(fuse::Region{.name = "r_a", .mesh = m_a, .dimension = 1});
  const auto r_b = g.add_region(fuse::Region{.name = "r_b", .mesh = m_b, .dimension = 1});
  // Disjoint SINGLE-rank processor sets -- the exact shape `src_ranks == 1U &&
  // dst_ranks == 1U` alone cannot distinguish from a coincident single
  // rank; only the `min_rank()` comparison the mutation removes can.
  const auto comp_a = g.add_component(
      fuse::ComponentDecl{.name = "a", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  const auto comp_b = g.add_component(
      fuse::ComponentDecl{.name = "b", .ranks = fuse::RankSet{1, 1}, .interval = Rational{1}});
  const auto f_a = g.add_field(comp_a, fuse::FieldSpec{
      .name = "a_field", .region = r_a, .disc = fuse::Discretization::ContinuousGLL, .order = 1,
      .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto f_b = g.add_field(comp_b, fuse::FieldSpec{
      .name = "b_field", .region = r_b,  // default disc: FV / non-nodal
      .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Requires});
  const auto ch = g.add_channel("a_to_b", f_a, f_b);
  // "consistent-transpose" needs a nodal TARGET; b_field is not nodal, so
  // remap_moves_placement() is false here -- unlike DisjointRankSets'
  // field transfers, this one is compiled into the dataflow graph via a real Transport, and moves_placement
  // cannot mask both_on_one_rank the way it does there.
  g.channel(ch).remap_method("consistent-transpose");

  fuse::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    ASSERT_TRUE(r.ok()) << r.error().message;
  }
  const auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  class Side final : public fuse::Component {
   public:
    Side(std::string name, std::size_t n) : name_(std::move(name)), n_(n) {}
    [[nodiscard]] std::string name() const override { return name_; }
    fuse::Result<void> register_fields(FieldRegistry& reg, const CouplingGraph& g) override {
      for (const fuse::FieldId f : g.component(*g.find_component(name_)).fields) {
        auto& buf = bufs[g.field(f).name];
        buf.assign(n_, 0.0);
        auto r = reg.bind(f, fuse::FieldRef{buf.data(), fuse::Layout{.stride = 1, .count = buf.size()},
                                            fuse::MemorySpace::Host});
        if (!r.ok()) return r;
      }
      return fuse::Result<void>::success();
    }
    fuse::Result<void> advance(const StepContext&) override { return fuse::Result<void>::success(); }
    std::map<std::string, std::vector<double>> bufs;
   private:
    std::string name_;
    std::size_t n_;
  };
  ComponentSet components;
  components.add(std::make_unique<Side>("a", 6));  // 6 vertices, 5-cell nodal line mesh
  components.add(std::make_unique<Side>("b", 5));  // 5 cells, FV (non-nodal) region
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  const auto r = runner.initialize(components, registry);
  ASSERT_FALSE(r.ok()) << "a disjoint single-rank-each nodal field transfer with a "
                          "mismatched method initialised";
  EXPECT_EQ(r.error().code, "E-RT-REMAP-NODAL-PARALLEL") << r.error().message;
}

// Mutation "distributed-remap-reads-the-default-view" (nodal-fields.toml):
// build_nodal_dist_remap resolving both sides through `backend.view(bid)`
// instead of each side's own `backend.placed_view(bid, who)`. MultiMeshGeometricBackend
// -- what every other test in this file uses -- cannot tell the two apart
// BY CONSTRUCTION: its own `placed_view` forwards straight to `view()`
// regardless of the ComponentId given (see that class's header comment and
// its `placed_view` body, test/support/partitioned_test_backend.hpp), so no
// fixture built from it, at any rank count or processor set shape, can kill
// this mutation. `PartitionedTestBackend` HAS a genuine per-placement
// decomposition (`set_placements`) but carries no geometry
// (`PartitionedTestMesh`'s own comment: centroid/measure are unused stubs),
// so it cannot feed `build_distributed_interp_rows`, which needs real
// coordinates.
//
// `GeometricPlacedLineBackend` (this track's own additive fixture, added
// alongside this test) closes the gap: it reuses `GeometricLineMesh`
// UNCHANGED for both the mesh's default decomposition and a SEPARATE,
// genuinely different decomposition registered per (mesh, component) via
// `set_component_ranks`. Below, `beam_mesh`'s DEFAULT decomposition is
// deliberately declared over `fluid_ranks` (the WRONG ranks for beam) while
// `set_component_ranks` gives the beam component its own, correct
// `beam_ranks` decomposition -- exactly the "one mesh, two processor sets"
// shape a coupler/shared-mesh scenario produces, and exactly what
// `build_nodal_dist_remap`'s own comment (runner.cpp, "Each side as seen by
// ITS OWN component") says must be read.
//
// Verified mechanism (base build configuration, standalone harness, no MPI, see this track's
// evidence log): with the fixture built this way, correct code
// (`placed_view`) returns the exact affine value at every point (H is
// P1-exact); the mutation (`view`) makes the nodal side's local view come
// back EMPTY (the DEFAULT decomposition excludes the very ranks that own
// it), and `build_distributed_interp_rows` fails with E-INTERP-NOGEOMETRY
// ("the source mesh ... has no cells or no vertices at all"). This
// specific TEST (a real 2+-rank MPI shape, using a REAL communicator so it
// is valid at every registered rank count 1-5, not the single-simulated-
// rank/null-communicator harness used to prove the mechanism above) has
// been compiled (MPI-enabled build configuration, compile-only -- this sandbox cannot
// execute MPI) but NOT run; a host run is what confirms it at real rank
// counts. Expected signatures: correct code -- every EXPECT_NEAR below
// passes; mutated code (apply the manifest's old->new by hand, rebuild) --
// `initialize()` fails and `E-INTERP-NOGEOMETRY` appears in the message,
// exactly as the base-build-configuration double showed.
TEST(NodalRemapRunnerMpi, DistributedRemapUsesThePlacedViewNotTheDefaultView) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);  // needs a real second rank the default view can wrongly omit
  const int w = world_size();
  const fuse::RankSet beam_ranks{0, (w + 1) / 2};
  const fuse::RankSet fluid_ranks{(w + 1) / 2, w - (w + 1) / 2};
  const std::size_t n_beam = 7, n_fluid = 11;

  fuse::CouplingGraph g;
  const auto m_beam = g.add_mesh(fuse::MeshDecl{.name = "beam_mesh", .source = "n/a"});
  const auto m_fluid = g.add_mesh(fuse::MeshDecl{.name = "fluid_mesh", .source = "n/a"});
  const auto r_beam = g.add_region(fuse::Region{.name = "r_beam", .mesh = m_beam, .dimension = 1});
  const auto r_fluid = g.add_region(fuse::Region{.name = "r_fluid", .mesh = m_fluid, .dimension = 1});
  const auto beam = g.add_component(
      fuse::ComponentDecl{.name = "beam", .ranks = beam_ranks, .interval = Rational{1}});
  const auto fluid = g.add_component(
      fuse::ComponentDecl{.name = "fluid", .ranks = fluid_ranks, .interval = Rational{1}});
  const auto vel_out = g.add_field(beam, fuse::FieldSpec{
      .name = "beam_velocity", .region = r_beam, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = kMetersPerSecond, .conservation = fuse::Conservation::Intensive,
      .intent = fuse::Intent::Provides});
  const auto vel_in = g.add_field(fluid, fuse::FieldSpec{
      .name = "fluid_velocity_in", .region = r_fluid, .units = kMetersPerSecond,
      .conservation = fuse::Conservation::Intensive, .intent = fuse::Intent::Requires});
  const auto vch = g.add_channel("wall_velocity", vel_out, vel_in);
  g.channel(vch).remap_method("consistent");

  GeometricPlacedLineBackend backend;
  // WRONG default: beam_mesh's own decomposition is declared over
  // fluid_ranks, not beam_ranks -- `view()` gives beam's own ranks nothing.
  backend.add_mesh_spec("beam_mesh", {.n_cells = n_beam, .ranks = fluid_ranks, .x_lo = 0.0, .x_hi = 1.0});
  backend.set_component_ranks("beam_mesh", beam, beam_ranks);  // the CORRECT decomposition
  backend.add_mesh_spec("fluid_mesh", {.n_cells = n_fluid, .ranks = fluid_ranks, .x_lo = 0.0, .x_hi = 1.0});
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto r = backend.load(g.mesh(fuse::MeshId{i}));
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  const auto lowered = fuse::lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{g};
  auto sched = fuse::schedule(g, lowered.dag, clocks, fuse::make_region_sizes(g, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  const auto local_count = [&](const CouplingGraph& gg, fuse::FieldId fid,
                               fuse::ComponentId who) -> std::size_t {
    const auto& fs = gg.field(fid);
    const auto mid = backend.mesh_id(gg.mesh(gg.region(fs.region).mesh).name);
    if (!mid.ok()) throw std::runtime_error("local_count: " + mid.error().message);
    const auto view = backend.placed_view(mid.value(), who);
    if (!view.ok()) throw std::runtime_error("local_count: " + view.error().message);
    const int dim = fs.disc == fuse::Discretization::ContinuousGLL ? 0 : 1;
    return view.value()->entity_count(dim);
  };

  auto b = std::make_unique<Side>("beam", [&](Side& me) {
    auto& v = me.bufs["beam_velocity"];
    const auto mid = backend.mesh_id("beam_mesh").value();
    const auto* view = backend.placed_view(mid, beam).value();
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = 0.3 + 2.0 * view->centroid(0, i).x;
  });
  auto f = std::make_unique<Side>("fluid", [&](Side&) {});
  b->local_count = [&](const CouplingGraph& gg, fuse::FieldId fid) { return local_count(gg, fid, beam); };
  f->local_count = [&](const CouplingGraph& gg, fuse::FieldId fid) { return local_count(gg, fid, fluid); };
  Side* bp = b.get();
  Side* fp = f.get();
  ComponentSet components;
  components.add(std::move(b));
  components.add(std::move(f));
  FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c.register_fields(registry, g);
    ASSERT_TRUE(r.ok()) << r.error().message;
  }

  Runner runner{g, lowered.dag, sched.plan, backend};
  RunnerConfig cfg;
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  cfg.transport = "mpi";
  cfg.transport_setup = [](fuse::Transport& tr) -> fuse::Result<void> {
    auto* mpi = dynamic_cast<fuse::MpiTransport*>(&tr);
    if (mpi == nullptr) return fuse::Failure{"E-TEST", "expected an MpiTransport"};
    return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
  };
  {
    const auto r = runner.initialize(components, registry, cfg);
    ASSERT_TRUE(r.ok()) << "initialize: " << r.error().code << ": " << r.error().message;
  }
  {
    const auto r = runner.run_period(components);
    ASSERT_TRUE(r.ok()) << "run_period: " << r.error().code << ": " << r.error().message;
  }

  const auto fluid_mid = backend.mesh_id("fluid_mesh").value();
  const auto* fluid_view = backend.placed_view(fluid_mid, fluid).value();
  const std::vector<double>& fluid_in = fp->bufs["fluid_velocity_in"];
  for (std::size_t j = 0; j < fluid_in.size(); ++j) {
    const double x = fluid_view->centroid(1, j).x;
    EXPECT_NEAR(fluid_in[j], 0.3 + 2.0 * x, 1e-12) << "rank " << world_rank() << " fluid cell " << j;
  }
  (void)bp;
}
