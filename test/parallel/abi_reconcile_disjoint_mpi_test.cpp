// Two externally-driven components on DISJOINT rank sets, each calling
// begin_step()/end_step() a DIFFERENT NUMBER OF TIMES. A reconciliation
// that lands every rank on ONE shared communicator deadlocks on the call
// where the counts stop lining up, while a reconciliation split PER
// COMPONENT never can, because each component's own collective calls stay
// on a communicator sized to that component alone.
//
// EXACTLY THREE RANKS, BY CONSTRUCTION. Component "u" is RankSet{0, 1}
// (rank 0 only); component "v" is RankSet{1, 2} (ranks 1 and 2). Rank 0
// calls begin_step("u")/end_step("u") TWICE; ranks 1 and 2 call
// begin_step("v")/end_step("v") ONCE. No channel connects "u" and "v" --
// this row is about the RECONCILIATION communicator alone, the same
// deliberate omission abi_reconcile_mpi_test.cpp documents in its own
// header comment, for the same reason (no Transport/Epoch/Constrain
// instruction in the plan, so no cross-rank SIZE agreement is at stake;
// only the LOCAL per-rank registered count against this rank's own share
// of its region).
//
// WHY A SINGLE SHARED RECONCILIATION COMMUNICATOR FAILS HERE. Suppose
// impl_initialize_local() called
// fuse::linalg::make_participant_set_reducer(c.communicator,
// mc.ranks.ranks()), where `mc` is THIS rank's OWN component's
// ComponentDecl. That reducer is meant for callers that all pass the SAME
// participant set to test membership against ONE shared region. Here each
// of the three ranks passes a DIFFERENT component's own ranks (rank 0
// passes {0}, ranks 1 and 2 pass {1, 2}), and the reducer's own colour
// rule (comm_mpi.cpp: `colour = participating ? 0 : MPI_UNDEFINED`) scores
// every one of them as "participating" in ITS OWN call, landing all three
// in ONE shared communicator of size 3. Rank 0's SECOND begin_step() would
// then issue a second MPI_Allreduce on that size-3 communicator, but ranks
// 1 and 2 have already made their ONE AND ONLY call and moved on, so rank
// 0 waits forever for peers that never come back. This is the deadlock
// seen in verification.brusselator.participants (U in begin_step ->
// reconcile -> MPI_Allreduce while V and the coupler had already moved on
// to MpiTransport::move), reduced to the smallest graph that reproduces it
// without the brusselator's own participant executables or LowFive
// plumbing.
//
// WHAT impl_initialize_local() DOES INSTEAD. It calls
// MPI_Comm_split(base, colour, key, &sub) directly, with colour set to
// THIS rank's own component's ComponentId value (a distinct integer per
// component -- never 0 for every caller the way the reducer's boolean
// colour is) and key set to this rank's position within its own
// component's RankSet. Rank 0 ("u") lands alone on a communicator of
// size 1; ranks 1 and 2 ("v") land together on a communicator of size 2,
// entirely independent of "u". Rank 0's two begin_step() calls each
// reduce over a one-rank communicator (MPI_Allreduce with one
// participant returns immediately, trivially agreeing with itself); V's
// one begin_step() call reduces over its own two-rank communicator,
// unaffected by how many times "u" calls anything. No rank ever waits
// for a peer outside its own component.
//
// Compiled under FUSE_ENABLE_MPI only (test/CMakeLists.txt). A pass means
// no rank hangs; a hang means the reconciliation communicator is shared
// across components again.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <string>
#include <vector>

#include "fuse/abi/participant.h"
#include "fuse/base/units.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"

namespace {

const fuse_participant_abi_v1* Table() {
  return fuse_coupler_abi(FUSE_ABI_VERSION);
}

/// Two components, "u" (RankSet{0, 1}) and "v" (RankSet{1, 2}), each with
/// its own Provides field over its own mesh, sized so each hosting rank's
/// own share is exactly one cell: "u"'s mesh is decomposed over "u"'s own
/// one-rank placement (the whole mesh, on rank 0); "v"'s mesh is
/// decomposed over "v"'s own two-rank placement (one cell each, on ranks
/// 1 and 2) -- the per-placement decomposition rule,
/// not the whole three-rank world. No channel connects them.
fuse::CouplingGraph BuildDisjointComponentsGraph() {
  fuse::CouplingGraph g;
  const auto mesh_u = g.add_mesh(fuse::MeshDecl{.name = "mu", .source = "generated:line:1"});
  const auto mesh_v = g.add_mesh(fuse::MeshDecl{.name = "mv", .source = "generated:line:2"});
  const auto region_u = g.add_region(fuse::Region{.name = "ru", .mesh = mesh_u, .dimension = 1});
  const auto region_v = g.add_region(fuse::Region{.name = "rv", .mesh = mesh_v, .dimension = 1});
  const auto u = g.add_component(fuse::ComponentDecl{
      .name = "u", .ranks = fuse::RankSet{0, 1}, .interval = fuse::Rational{1}});
  const auto v = g.add_component(fuse::ComponentDecl{
      .name = "v", .ranks = fuse::RankSet{1, 2}, .interval = fuse::Rational{1}});
  (void)g.add_field(u, fuse::FieldSpec{.name = "out", .region = region_u, .components = 1,
                                        .units = fuse::units::kKelvin,
                                        .conservation = fuse::Conservation::None,
                                        .intent = fuse::Intent::Provides});
  (void)g.add_field(v, fuse::FieldSpec{.name = "out", .region = region_v, .components = 1,
                                        .units = fuse::units::kKelvin,
                                        .conservation = fuse::Conservation::None,
                                        .intent = fuse::Intent::Provides});
  return g;
}

/// Sets up and initializes `ctx` on every rank, with `my_component` set
/// to whichever of "u" or "v" this rank's own position hosts (rank 0 ->
/// "u"; ranks 1 and 2 -> "v"). Registers BOTH fields on every rank, each
/// sized by this rank's own position in THAT field's own component's
/// RankSet (one cell if hosted, zero otherwise -- "a component
/// that does not live on this rank gets an empty view of the right
/// dimension", applied here to the registering CALLER rather than to
/// NativeMeshBackend's own internal placements, since the raw C
/// participant API registers whatever count the caller supplies and does
/// not independently consult the backend). Returns the live ctx, or
/// nullptr on any setup failure.
fuse_ctx* SetupDisjointComponents(const fuse_participant_abi_v1* t, int rank) {
  const fuse::CouplingGraph g = BuildDisjointComponentsGraph();
  const auto bytes = fuse::serialize(g);

  fuse_context_out co{.struct_size = sizeof(co)};
  if (t->create(nullptr, nullptr, &co) != FUSE_OK) return nullptr;
  fuse_graph_source src{.struct_size = sizeof(src)};
  src.bytes = bytes.data();
  src.byte_count = bytes.size();
  if (t->load_graph(co.ctx, &src, nullptr) != FUSE_OK) return nullptr;

  const fuse::RankSet ranks_u{0, 1};
  const fuse::RankSet ranks_v{1, 2};
  const bool hosts_u = ranks_u.contains(rank);
  const bool hosts_v = ranks_v.contains(rank);

  fuse_setup_options opts{.struct_size = sizeof(opts)};
  opts.communicator = static_cast<std::int64_t>(MPI_Comm_c2f(MPI_COMM_WORLD));
  // Open MPI's Fortran handle for MPI_COMM_WORLD is 0, which the ABI reads as
  // "none" unless has_communicator says the handle is deliberate.
  opts.has_communicator = 1;
  opts.my_component = hosts_u ? "u" : "v";
  if (t->setup(co.ctx, &opts, nullptr) != FUSE_OK) return nullptr;

  static double storage_u = 0.0;
  static double storage_v = 0.0;
  const struct { const char* component; double* data; bool hosted; } kFields[] = {
      {"u", &storage_u, hosts_u},
      {"v", &storage_v, hosts_v},
  };
  for (const auto& f : kFields) {
    fuse_field_name fname{.struct_size = sizeof(fname)};
    fname.component = f.component;
    fname.field = "out";
    fuse_field_ref fref{.struct_size = sizeof(fref)};
    if (t->field_handle(co.ctx, &fname, &fref) != FUSE_OK) return nullptr;
    const std::int64_t count = f.hosted ? 1 : 0;
    fuse_field_view fview{.struct_size = sizeof(fview)};
    fview.handle = fref.handle;
    fview.element_type = FUSE_ELEMENT_F64;
    fview.memspace = FUSE_MEMSPACE_HOST;
    fview.ownership = FUSE_OWNERSHIP_OWNED;
    fview.components = 1;
    fview.count = count;
    fview.owned_count = count;
    fview.stride = 1;
    fview.data = f.hosted ? f.data : nullptr;
    if (t->register_field(co.ctx, &fview, nullptr) != FUSE_OK) return nullptr;
  }

  if (t->initialize(co.ctx, nullptr, nullptr) != FUSE_OK) return nullptr;
  return co.ctx;
}

}  // namespace

TEST(AbiReconcileDisjointMpi, DifferentCallCountsPerComponentNeverHang) {
  ASSERT_EQ([] { int s = 0; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }(), 3)
      << "this row models exactly u={0}, v={1,2}; it must run with 3 ranks";

  const auto* t = Table();
  ASSERT_NE(t, nullptr);

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const bool is_u = (rank == 0);

  fuse_ctx* ctx = SetupDisjointComponents(t, rank);
  ASSERT_NE(ctx, nullptr) << "rank " << rank << ": setup/initialize failed";

  fuse_step_request req{.struct_size = sizeof(req)};
  req.component = is_u ? "u" : "v";
  fuse_step_report rep{.struct_size = sizeof(rep)};

  // Rank 0 ("u") drives TWO begin_step/end_step pairs; ranks 1 and 2
  // ("v") drive exactly ONE. Round 3's bug deadlocks rank 0's SECOND
  // begin_step() here, in the one-shared-communicator reduction, because
  // ranks 1 and 2 never issue a matching second call. Round 4's fix
  // reduces "u"'s own calls on a one-rank communicator (never blocks on
  // anyone) and "v"'s own call on its own two-rank communicator
  // (unaffected by how many times "u" calls anything).
  const int step_count = is_u ? 2 : 1;
  for (int i = 0; i < step_count; ++i) {
    const fuse_status begin_rc = t->begin_step(ctx, &req, &rep);
    EXPECT_EQ(begin_rc, FUSE_OK)
        << "rank " << rank << ": begin_step() call " << i << " refused unexpectedly";
    const fuse_status end_rc = t->end_step(ctx, &req, &rep);
    EXPECT_EQ(end_rc, FUSE_OK)
        << "rank " << rank << ": end_step() call " << i << " refused unexpectedly";
  }

  t->destroy(ctx, nullptr, nullptr);
}
