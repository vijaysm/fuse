// KNOWN_ISSUES.md D61 (round 2, the lead's own instruction): the asymmetric
// case neither abi_participant_mpi_test.cpp nor any other fixture in the
// tree exercises -- one rank of a MULTI-RANK component's own begin_step()
// call refuses LOCALLY (a malformed fuse_step_request) while its peers'
// own local checks pass, and every rank must come back with the SAME
// agreed status, naming the refusing rank, with no rank left waiting in a
// later collective for a peer that already returned.
//
// WHY A ONE-COMPONENT, SINGLE-FIELD GRAPH. The existing ABI fixtures
// (test/abi/graph_fixture.hpp) declare every component's own RankSet as
// {0, 1} -- one rank, by construction, which is exactly why they cannot
// exercise a MULTI-rank component's own collective entries at all. This
// file declares ONE component, "p", whose RankSet spans the ENTIRE base
// communicator (every world rank hosts it), with ONE Provides field and
// deliberately NO channel: fuse::schedule()'s own compiled plan then
// contains no Transport/Epoch/Constrain instruction that reads or writes
// this field across ranks, so no CROSS-RANK size agreement (Runner::Impl::
// resolve_regions()'s own E-RT-SIZE, KNOWN_ISSUES.md D36) is ever at
// stake -- but the registered count on THIS rank must still match what
// the schedule's own plan expects for THIS rank's own share of the
// region (a LOCAL, per-rank consistency check that fires independently
// of any channel -- KNOWN_ISSUES.md D15 round 3's provider_subset_mpi_
// test.cpp found the identical class of check from the opposite
// direction). The mesh source is parameterised by `world_size` --
// "generated:line:<world_size>" -- specifically so every rank's own
// share divides EVENLY (one cell each, no remainder, no rank left with
// zero): KNOWN_ISSUES.md D61 round 3's own first host run (this file's
// previous version, "generated:line:1" registered uniformly as count=0)
// failed here, not in reconciliation at all -- ctx came back null from
// SetupSpanningComponent() because initialize() refused on a count/plan
// mismatch before begin_step() was ever reached.
//
// WHY my_component IS SET. KNOWN_ISSUES.md D61's own REVERTED paragraph:
// the fix now reconciles over a COMPONENT-LOCAL communicator, split once
// at initialize() time from `fuse_setup_options::my_component` (src/abi/
// participant.cpp's impl_initialize_local(), context.hpp's own comment on
// `collective_comm_f`). Omitting `my_component` here (this file's own
// previous version did) means no split ever happens and begin_step()'s
// own reconciliation silently no-ops -- every rank would return its OWN
// local verdict, not the agreed one, defeating this row's entire point.
//
// Compiled under FUSE_ENABLE_MPI only (test/CMakeLists.txt); MPI cannot
// start in this sandbox (KNOWN_ISSUES.md C1), so this row is BUILT, never
// run here -- see KNOWN_ISSUES.md D61's own entry for the exact host
// commands and what a pass/hang/disagreement each would mean.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstring>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// One component, "p", spanning every rank of `world_size`; one Provides
/// field over a mesh sized so every rank's own share is EXACTLY one cell
/// (see this file's own header comment for why "generated:line:1" was
/// wrong). No channel moves this field anywhere -- see the header
/// comment for what that does and does not protect against.
fuse::CouplingGraph BuildSpanningComponentGraph(int world_size) {
  fuse::CouplingGraph g;
  const auto mesh = g.add_mesh(fuse::MeshDecl{
      .name = "m", .source = "generated:line:" + std::to_string(world_size)});
  const auto region = g.add_region(fuse::Region{.name = "r", .mesh = mesh, .dimension = 1});
  const auto p = g.add_component(fuse::ComponentDecl{
      .name = "p", .ranks = fuse::RankSet{0, world_size}, .interval = fuse::Rational{1}});
  (void)g.add_field(p, fuse::FieldSpec{.name = "out", .region = region, .components = 1,
                                       .units = fuse::units::kKelvin,
                                       .conservation = fuse::Conservation::None,
                                       .intent = fuse::Intent::Provides});
  return g;
}

/// Sets up and initializes `ctx` identically on every rank: the real base
/// communicator (MPI_COMM_WORLD, c2f'd) AND `my_component = "p"` are both
/// what makes D61's own round-3 reconciliation engage at all (`fuse_ctx::
/// communicator_set` and `fuse_ctx::collective_comm_f` respectively) --
/// every earlier ABI MPI fixture in the tree (abi_participant_mpi_test.
/// cpp) never calls fuse_setup() with a nonzero communicator, which is
/// exactly why none of them could exercise this. Returns the live ctx, or
/// nullptr on any setup failure (every rank fails identically here, since
/// this part is symmetric by construction).
fuse_ctx* SetupSpanningComponent(const fuse_participant_abi_v1* t, int world_size) {
  const fuse::CouplingGraph g = BuildSpanningComponentGraph(world_size);
  const auto bytes = fuse::serialize(g);

  fuse_context_out co{.struct_size = sizeof(co)};
  if (t->create(nullptr, nullptr, &co) != FUSE_OK) return nullptr;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  if (t->load_graph(co.ctx, &src, nullptr) != FUSE_OK) return nullptr;

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
  opts.my_component = "p";
  if (t->setup(co.ctx, &opts, nullptr) != FUSE_OK) return nullptr;

  fuse_field_name fname{.struct_size = sizeof(fname)};
  fname.component = "p";
  fname.field = "out";
  fuse_field_ref fref{.struct_size = sizeof(fref)};
  if (t->field_handle(co.ctx, &fname, &fref) != FUSE_OK) return nullptr;
  double value = 0.0;
  fuse_field_view fview{.struct_size = sizeof(fview)};
  fview.handle = fref.handle;
  fview.element_type = FUSE_ELEMENT_F64;
  fview.memspace = FUSE_MEMSPACE_HOST;
  fview.ownership = FUSE_OWNERSHIP_OWNED;
  fview.components = 1;
  fview.count = 1;        // every rank's own share of "generated:line:<world_size>" is 1 cell
  fview.owned_count = 1;
  fview.stride = 1;
  fview.data = &value;
  if (t->register_field(co.ctx, &fview, nullptr) != FUSE_OK) return nullptr;

  if (t->initialize(co.ctx, nullptr, nullptr) != FUSE_OK) return nullptr;
  return co.ctx;
}

}  // namespace

TEST(AbiReconcileMpi, AsymmetricLocalRefusalAgreesAcrossRanksWithNoHang) {
  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  fuse_ctx* ctx = SetupSpanningComponent(t, world_size);
  ASSERT_NE(ctx, nullptr) << "rank " << rank << ": setup/initialize failed (symmetric so far)";

  // Rank 1 alone gets a MALFORMED request: struct_size below the first
  // frozen fuse_step_request layout (entries.def's own growth history --
  // c-abi.md's "16 -> 72" table), refused LOCALLY with FUSE_ERR_VERSION
  // before this rank's own impl_begin_step_local() reads anything past
  // `component`. Every other rank's own request is ordinary and valid.
  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = "p";
  if (rank == 1) {
    req.struct_size = 4;  // well below offsetof(fuse_step_request, component) + sizeof(char*)
  }
  fuse_step_report rep{.struct_size = sizeof(rep)};
  const fuse_status rc = t->begin_step(ctx, &req, &rep);

  // THE CLAIM: every rank's own return value must be IDENTICAL -- rank
  // 1's own refusal, not a per-rank mix of FUSE_OK and FUSE_ERR_VERSION.
  // Gathered with a real MPI_Allgather so a genuine per-rank disagreement
  // is caught directly, independent of whatever reconcile_collective_
  // status() itself does internally.
  std::vector<std::int32_t> all(static_cast<std::size_t>(world_size));
  const std::int32_t mine = rc;
  MPI_Allgather(&mine, 1, MPI_INT32_T, all.data(), 1, MPI_INT32_T, MPI_COMM_WORLD);
  for (int r = 0; r < world_size; ++r) {
    EXPECT_EQ(all[static_cast<std::size_t>(r)], all[0])
        << "rank " << r << " disagrees with rank 0 on the reconciled status "
           "(rank " << rank << " observing)";
  }
  EXPECT_EQ(rc, FUSE_ERR_VERSION)
      << "the agreed status must be rank 1's own local refusal (struct_size "
         "below the first frozen layout), not this rank's own local verdict";

  // No rank may be left holding an open step the others already resolved
  // collectively: end_step() on the SAME (now-reconciled) request must
  // either also agree or refuse cleanly -- never hang. Not asserted
  // beyond "returns", since the exact code here depends on whether this
  // rank's own begin_step() opened a step at all (it did not, on every
  // rank, once reconciliation makes begin_step() itself refuse).
  t->end_step(ctx, &req, &rep);

  t->destroy(ctx, nullptr, nullptr);
}
