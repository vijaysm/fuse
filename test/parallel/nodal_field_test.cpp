// "Nodal fields" track, step 2: parallel nodal fields under a real mpiexec.
//
// NOT YET RUN: this sandbox cannot initialize MPI (docs/dev/testing.md's
// harness note; see also test/model/descriptor_mpi_test.cpp's identical
// disclaimer). Every case here is registered at the MPI-enabled build configuration via
// fuse_add_mpi_test and compiles under FUSE_ENABLE_MPI=ON; outcomes are a
// host action.
//
// WHAT THIS FILE COVERS, AND WHAT IT HONESTLY DOES NOT:
//
//   (a) every copy of a shared vertex equals its owner's value after a
//       transfer -- NodalTransport.EveryLocalCopyMatchesTheGlobalIdentity,
//       through a real Runner + Transport (attach_comm_graphs's dim fix,
//       comm_graph.cpp's existing owned-source/every-destination gather).
//   (b) the owner is the lowest holding rank --
//       SharedVertexOwnership.LowestRankOwnsEachSharedVertexAndBoundariesAgree,
//       directly against SharedVertexLineMesh, no Runner involved.
//   (c) an H^T load field transfer conserves the global force exactly, identically
//       across rank counts -- AccumulateSharedRows.* below verifies the
//       STANDALONE reduction primitive (comm_graph.hpp's
//       accumulate_shared_rows) does this correctly for a deliberately
//       non-uniform per-rank partial pattern. It is NOT run through Runner:
//       docs/dev/nodal-fields.md's "H^T on a distributed interface" records
//       why wiring it into Runner::Impl::execute needed a scope decision
//       this track did not make unilaterally (see this task's
//       open_questions). AccumulateSharedRows.ManuallyAssembledPowerBalances
//       demonstrates, by calling the primitive directly rather than through
//       Runner, that WHEN it is wired in, power balances globally -- this is
//       the closest this file comes to property (e), and it is explicitly a
//       manual assembly, not an automatic one.
//   (d) H exact for a linear field on every rank -- NOT exercised here. This
//       file's Transport pair sits on the SAME mesh entity set (no non-matching
//       interface), so its "transfer" is a same-identity copy, exact for
//       ANY field by construction, not a demonstration of `consistent_
//       interp.hpp`'s H operator specifically. A genuine parallel H test
//       needs a distributed non-matching interface, which needs the same
//       missing accumulation this file already reports as open.

#include <gtest/gtest.h>
#include <mpi.h>

// R-50 (REVIEW.md): the documented rank-requirement convention
// (docs/dev/testing.md).
#include "fuse_test_main.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <fuse/ir/lowering.hpp>
#include <fuse/model/component.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#include <fuse/transport/comm_graph.hpp>

#include "../support/partitioned_test_backend.hpp"

using fuse::Rational;
using fuse::Runner;

namespace {

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

/// Binds one nodal field's whole local buffer (`SharedVertexLineMesh`'s
/// `entity_count(0)` values) to storage this test owns and can read back.
class NodalStub final : public fuse::Component {
 public:
  NodalStub(std::string name, fuse::FieldId field, std::size_t count)
      : name_(std::move(name)), field_(field), storage_(count, 0.0) {}
  [[nodiscard]] std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph&) override {
    fuse::Layout layout;
    layout.count = storage_.size();
    return registry.bind(field_,
                         fuse::FieldRef{storage_.data(), layout, fuse::MemorySpace::Host});
  }
  fuse::Result<void> advance(const fuse::StepContext&) override {
    return fuse::Result<void>::success();
  }
  [[nodiscard]] double* data() noexcept { return storage_.data(); }
  [[nodiscard]] std::size_t size() const noexcept { return storage_.size(); }

 private:
  std::string name_;
  fuse::FieldId field_;
  std::vector<double> storage_;
};

}  // namespace

// ===========================================================================
// (a) every copy of a shared vertex equals its owner's value after a
//     transfer, through a real Runner + Transport.
// ===========================================================================

TEST(NodalTransport, EveryLocalCopyMatchesTheGlobalIdentity) {
  const std::size_t n_cells = 23;  // deliberately not a multiple of any rank count tried
  const fuse::RankSet world{0, world_size()};

  fuse::CouplingGraph graph;
  const auto mesh_decl = graph.add_mesh(fuse::MeshDecl{.name = "beam", .source = "test"});
  const auto region = graph.add_region(
      fuse::Region{.name = "r", .mesh = mesh_decl, .dimension = 0});
  const auto src_c = graph.add_component(
      fuse::ComponentDecl{.name = "src", .ranks = world, .interval = Rational{1}});
  const auto dst_c = graph.add_component(
      fuse::ComponentDecl{.name = "dst", .ranks = world, .interval = Rational{1}});
  const auto src_f = graph.add_field(src_c, fuse::FieldSpec{
      .name = "v_out", .region = region, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = fuse::units::kKelvin, .intent = fuse::Intent::Provides});
  const auto dst_f = graph.add_field(dst_c, fuse::FieldSpec{
      .name = "v_in", .region = region, .disc = fuse::Discretization::ContinuousGLL,
      .order = 1, .units = fuse::units::kKelvin, .intent = fuse::Intent::Requires});
  graph.add_channel("v", src_f, dst_f);  // no .remap_method(): plain copy

  SharedVertexLineBackend backend(n_cells, world);
  ASSERT_TRUE(backend.load(graph.mesh(mesh_decl)).ok());

  const auto lowered = lower(graph);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const fuse::ClockSet clocks{graph};
  const auto sched =
      fuse::schedule(graph, lowered.dag, clocks, fuse::make_region_sizes(graph, lowered.dag, backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();

  const auto view = backend.view(fuse::MeshId{0}).value();
  const std::size_t local_n = view->entity_count(0);

  fuse::ComponentSet components;
  auto src = std::make_unique<NodalStub>("src", src_f, local_n);
  auto dst = std::make_unique<NodalStub>("dst", dst_f, local_n);
  NodalStub* dst_p = dst.get();
  // v(gid) = 3.0 + 0.5 * gid -- affine, so a rank comparing its local buffer
  // against this SAME closed form after the transfer is checking exactly
  // "this copy equals the value its owner would have written", without
  // needing a second communication round to ask the owner directly.
  for (std::size_t i = 0; i < local_n; ++i) {
    src->data()[i] = 3.0 + 0.5 * static_cast<double>(view->global_id(0, i));
  }
  components.add(std::move(src));
  components.add(std::move(dst));

  fuse::FieldRegistry registry;
  for (auto& c : components) ASSERT_TRUE(c.register_fields(registry, graph).ok());

  fuse::RunnerConfig cfg;
  cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
  Runner runner{graph, lowered.dag, sched.plan, backend};
  const auto ri = runner.initialize(components, registry, cfg);
  ASSERT_TRUE(ri.ok()) << ri.error().code << ": " << ri.error().message;
  const auto rr = runner.run(components, 1);
  ASSERT_TRUE(rr.ok()) << rr.error().code << ": " << rr.error().message;

  for (std::size_t i = 0; i < local_n; ++i) {
    const double expected = 3.0 + 0.5 * static_cast<double>(view->global_id(0, i));
    EXPECT_DOUBLE_EQ(dst_p->data()[i], expected)
        << "rank " << current_rank() << ", local vertex " << i
        << " (global " << view->global_id(0, i) << "), owned="
        << view->is_owned(0, i);
  }
}

// ===========================================================================
// (b) the owner is the lowest holding rank, and the two neighbours' claims
//     about their shared boundary agree -- directly, no Runner.
// ===========================================================================

TEST(SharedVertexOwnership, LowestRankOwnsEachSharedVertexAndBoundariesAgree) {
  const int size = world_size();
  const int rank = current_rank();
  const std::size_t n_cells = 17;
  SharedVertexLineMesh mesh(n_cells, size, rank);

  const std::int64_t local_lo = static_cast<std::int64_t>(mesh.lo());
  const std::int64_t local_hi = static_cast<std::int64_t>(mesh.hi());
  const int owns_first = mesh.is_owned(0, 0) ? 1 : 0;
  const int owns_last = mesh.is_owned(0, mesh.entity_count(0) - 1) ? 1 : 0;

  std::vector<std::int64_t> all_lo(static_cast<std::size_t>(size));
  std::vector<std::int64_t> all_hi(static_cast<std::size_t>(size));
  std::vector<int> all_owns_first(static_cast<std::size_t>(size));
  std::vector<int> all_owns_last(static_cast<std::size_t>(size));
  ASSERT_EQ(MPI_Allgather(&local_lo, 1, MPI_INT64_T, all_lo.data(), 1, MPI_INT64_T,
                          MPI_COMM_WORLD),
           MPI_SUCCESS);
  ASSERT_EQ(MPI_Allgather(&local_hi, 1, MPI_INT64_T, all_hi.data(), 1, MPI_INT64_T,
                          MPI_COMM_WORLD),
           MPI_SUCCESS);
  ASSERT_EQ(MPI_Allgather(&owns_first, 1, MPI_INT, all_owns_first.data(), 1, MPI_INT,
                          MPI_COMM_WORLD),
           MPI_SUCCESS);
  ASSERT_EQ(MPI_Allgather(&owns_last, 1, MPI_INT, all_owns_last.data(), 1, MPI_INT,
                          MPI_COMM_WORLD),
           MPI_SUCCESS);

  // Every rank performs the SAME check over the SAME gathered arrays, so a
  // failure is symmetric across ranks -- fuse_test_main's end-of-test
  // reduction (docs/dev/testing.md) needs that to fail cleanly rather than
  // hang.
  for (int r = 0; r + 1 < size; ++r) {
    EXPECT_EQ(all_hi[static_cast<std::size_t>(r)], all_lo[static_cast<std::size_t>(r + 1)])
        << "rank " << r << "'s last vertex should be rank " << (r + 1) << "'s first";
    EXPECT_EQ(all_owns_last[static_cast<std::size_t>(r)], 1)
        << "rank " << r << " should own its own last (shared) vertex -- it is the lower rank";
    EXPECT_EQ(all_owns_first[static_cast<std::size_t>(r + 1)], 0)
        << "rank " << (r + 1) << " should NOT own the vertex it shares with lower rank " << r;
  }
  EXPECT_EQ(all_owns_first[0], 1) << "rank 0 has no lower neighbour, so it owns its own first vertex";
}

// ===========================================================================
// (c) accumulate_shared_rows: the standalone cross-rank reduction, on a
//     deliberately non-uniform per-rank partial pattern.
// ===========================================================================

TEST(AccumulateSharedRows, SumsEveryRanksPartialExactlyOnceAndDeliversTheSameTotalToEveryHolder) {
  const int size = world_size();
  const int rank = current_rank();
  const std::size_t n_cells = 13;
  SharedVertexLineMesh mesh(n_cells, size, rank);
  const std::size_t nv = mesh.entity_count(0);

  // Rank r's partial contribution to global vertex g is (r + 1) -- NOT
  // uniform, and NOT keyed by ownership: every rank holding a given vertex
  // (owned or ghost) contributes its own (r + 1), exactly the "several ranks
  // independently compute a nonzero partial for one shared vertex" shape
  // docs/dev/nodal-fields.md's H^T section describes. The correct total at
  // an INTERIOR shared vertex (held by ranks r and r+1) is therefore
  // (r + 1) + (r + 2); at a genuine boundary (mesh ends, held by one rank
  // only) it is just that one rank's own contribution.
  std::vector<double> local_values(nv);
  for (std::size_t i = 0; i < nv; ++i) local_values[i] = static_cast<double>(rank + 1);

  const auto r = fuse::accumulate_shared_rows(mesh, 0, 1, local_values,
                                              MPI_Comm_c2f(MPI_COMM_WORLD));
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;

  for (std::size_t i = 0; i < nv; ++i) {
    const std::size_t gid = mesh.global_id(0, i);
    // This vertex's total is the sum of (rank_holder + 1) over every rank
    // that holds gid -- gid 0 only rank 0, the last global vertex only the
    // last rank, and every internal boundary gid exactly two consecutive
    // ranks (rank == gid's lower holder, and rank + 1).
    double expected = 0.0;
    if (gid == 0) {
      expected = 1.0;  // only rank 0 holds global vertex 0
    } else if (gid == n_cells) {
      expected = static_cast<double>(size);  // only rank (size-1) holds the last vertex
    } else {
      // An interior gid that is a rank boundary is held by exactly two
      // ranks; one that is not a boundary is held by exactly one. Recover
      // "who holds gid" from this rank's own view: if THIS rank holds gid at
      // local index i, and i is this rank's first or last local index (a
      // shared slot), the other holder is rank-1 or rank+1 respectively.
      const bool is_shared_first = (i == 0) && (rank > 0);
      const bool is_shared_last = (i == nv - 1) && (rank + 1 < size);
      if (is_shared_first) {
        expected = static_cast<double>(rank + 1) + static_cast<double>(rank);  // (rank-1)+1
      } else if (is_shared_last) {
        expected = static_cast<double>(rank + 1) + static_cast<double>(rank + 2);
      } else {
        expected = static_cast<double>(rank + 1);
      }
    }
    EXPECT_DOUBLE_EQ(local_values[i], expected)
        << "rank " << rank << ", local vertex " << i << " (global " << gid << ")";
  }
}

/// A load assembled the way a real one is. Each rank's cells are its own, and
/// each cell spreads its share of a known total force equally onto its two end
/// vertices. A vertex on a rank boundary therefore receives a PARTIAL load from
/// each side, and accumulate_shared_rows must sum those onto the owner exactly
/// once. The owned-only sum across ranks then equals the force sent.
///
/// The first version of this test wrote a full per-vertex share into every
/// holder's copy of a shared vertex. That input already counted each shared
/// vertex twice before any reduction ran, so the test failed by exactly
/// (ranks - 1) / (n_cells + 1), which is what the host measured (3/14 at 4
/// ranks, 4/14 at 5). The primitive was right; the input was not a load.
TEST(AccumulateSharedRows, ManuallyAssembledPowerBalances) {
  const int size = world_size();
  const std::size_t n_cells = 13;
  SharedVertexLineMesh mesh(n_cells, size, current_rank());
  const std::size_t nv = mesh.entity_count(0);
  const std::size_t nc = mesh.entity_count(1);
  ASSERT_EQ(nv, nc + 1) << "local cell c joins local vertices c and c+1";

  const double total_force = 1.0;
  const double per_cell = total_force / static_cast<double>(n_cells);
  std::vector<double> local_load(nv, 0.0);
  for (std::size_t c = 0; c < nc; ++c) {
    local_load[c] += 0.5 * per_cell;
    local_load[c + 1] += 0.5 * per_cell;
  }
  // An interface velocity linear in the global vertex id: power at the owned
  // vertices must equal the cellwise trapezoidal power, which is what the
  // assembled load has to reproduce.
  std::vector<double> v(nv);
  for (std::size_t i = 0; i < nv; ++i) {
    v[i] = 0.25 + static_cast<double>(mesh.global_id(0, i)) / static_cast<double>(n_cells);
  }
  double cell_power = 0.0;
  for (std::size_t c = 0; c < nc; ++c) cell_power += per_cell * 0.5 * (v[c] + v[c + 1]);

  const auto sum_owned = [&](const std::vector<double>& x, const std::vector<double>* w) {
    double local = 0.0;
    for (std::size_t i = 0; i < nv; ++i) {
      if (mesh.is_owned(0, i)) local += x[i] * (w != nullptr ? (*w)[i] : 1.0);
    }
    double global = 0.0;
    EXPECT_EQ(MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), MPI_SUCCESS);
    return global;
  };

  // Before the reduction an owner holds only its own side of a shared vertex,
  // so the owned-only sum is short by half a cell per rank boundary: the
  // reason the reduction exists.
  const double before = sum_owned(local_load, nullptr);
  EXPECT_NEAR(before, total_force - 0.5 * per_cell * static_cast<double>(size - 1), 1e-13);

  const auto r = fuse::accumulate_shared_rows(mesh, 0, 1, local_load,
                                              MPI_Comm_c2f(MPI_COMM_WORLD));
  ASSERT_TRUE(r.ok()) << r.error().code << ": " << r.error().message;

  EXPECT_NEAR(sum_owned(local_load, nullptr), total_force, 1e-13)
      << "the assembled load, summed over owned vertices, must equal the force sent";
  double global_cell_power = 0.0;
  ASSERT_EQ(MPI_Allreduce(&cell_power, &global_cell_power, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD),
            MPI_SUCCESS);
  EXPECT_NEAR(sum_owned(local_load, &v), global_cell_power, 1e-13)
      << "power at the owned vertices must equal the power integrated over the cells";
}