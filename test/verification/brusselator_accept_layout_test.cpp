// Conformance Unit 2 (Time windows track), accept-mode layout fix
// (KNOWN_ISSUES.md D36, follow-up to D18/D23/D24): a tier-0 regression for
// the host failure that found it -- `verification.brusselator.participants.
// accept.3` (2026-10-12, MPICH 5.0.0a1/ch4:ofi, the first run where
// accept/connect actually completed the handshake): `u_stdout.log` printed
// `FAIL(u_main): initialize: status=-4 ... window member 'U' registered 256
// values for its export, but the plan sizes that field at 128 on this
// rank`.
//
// WHY THIS LEVEL, NOT NativeMeshBackend DIRECTLY. test/backend/
// native_generators_test.cpp's own `NativeBackend.
// APlacementIsDecomposedOverItsOwnRanksNotTheWholeRun` already pins D24's
// fix (a placement is decomposed over its OWN RankSet, not the whole run)
// at the backend's own level. What D36 fixes is one layer up: which ranks
// the ACCEPT-MODE GRAPH declares as U's own RankSet in the first place, so
// this test builds the REAL three-component graph `brusselator_graph_
// tool.cpp`'s own accept branch builds (via the same `add_window_
// coupling()` that tool and test/verification/brusselator_participants_
// test.cpp both call) and reaches `fuse::make_region_sizes()` -- the
// callback `schedule()`/`Runner::Impl::resolve_buffers()` consult for
// exactly the number the host's own E-RT-SIZE message named ("the plan
// sizes that field at N on this rank" IS this function's return value,
// src/backend/serial_backend.cpp's own `make_region_sizes`, dispatched to
// `NativeMeshBackend::placed_region_size()` for a generated mesh).
//
// WHY NOT ALL THE WAY TO `Runner::initialize()`. Tried first, and found
// NOT REACHABLE at tier 0 for this shape, regardless of D36: U and V's
// window-transfer fields are BOTH declared on `u_region` (this graph's own
// file header, brusselator_graph.hpp, explains why), so `NativeMeshBackend`
// decomposes that ONE mesh twice, once per component's own RankSet -- U's
// decomposition and V's are two DIFFERENT partitions of the same global
// mesh the moment their RankSets differ, which is exactly accept mode's
// own point. Moving data between two different partitions of one mesh
// needs a real exchange plan (a CommGraph, built from a live communicator)
// to know which global entity goes where; without one, `Runner::Impl::
// check_transport_extents()` assumes "the whole arc is moved" and refuses
// `E-RT-SIZE` the instant the two ends' sizes differ -- confirmed here by
// hand (not asserted, since asserting a failure this incidental to D36
// would pin the WRONG thing): `Runner::initialize()` with `force_this_
// process = 0` on this exact graph refuses "Transport instruction 1:
// reads 2816 values from arc %0 into 0 at arc %2; without a CommGraph the
// whole arc is moved ... set base_communicator and the \"mpi\" transport".
// This is the SAME class of MPI-only requirement KNOWN_ISSUES.md already
// documents for window coupling across ranks (D19/D25's own host-only
// `parallel.window_runner_mpi.{2,3}`) -- not a new gap D36 introduces, and
// not something this sandbox can close (KNOWN_ISSUES.md C1: MPI cannot
// start here at all). test/runtime/window_hosting_test.cpp's own two-
// member fixture reaches `Runner::initialize()` on DIFFERENT ranks only
// because it uses `SerialMeshBackend`/a one-cell region, where every
// placement is the identical whole-region view everywhere -- no real
// partition, so no cross-rank mismatch for `check_transport_extents()` to
// catch in the first place; that shortcut is unavailable here because
// testing D36 at all REQUIRES a backend that actually partitions
// (`NativeMeshBackend`), which is the entire reason D24's bug -- and D36's
// own fix to the layout that fed it -- existed.
//
// NO MPI ANYWHERE (KNOWN_ISSUES.md C1: this sandbox cannot start it at
// all). `fuse::RunnerConfig::force_this_process` (test/runtime/window_
// hosting_test.cpp's own convention) is unused here for exactly the reason
// above -- `make_region_sizes()` and `schedule()` already take this
// process's role from `NativeMeshBackend`'s own `(rank, num_ranks)`
// constructor arguments, with no communicator needed at any point in this
// file.

#include <gtest/gtest.h>

#include <fuse/backend/native_backend.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/schedule/plan.hpp>
#include <fuse/transport/connect.hpp>

#include <cstdint>
#include <optional>
#include <string>

#include "brusselator_graph.hpp"

using fuse::CouplingGraph;
using fuse::LaunchMode;
using fuse::LaunchSpec;
using fuse::RankSet;
using fuse::Rational;

namespace {

// The D36 shape (brusselator_graph_tool.cpp's own accept branch,
// --ranks-u 1 --ranks-v 1 --ranks-coupler 1, the exact row that timed out
// on the host): U hosts exactly its own physics rank 0, the coupler the
// gap rank 1, V the rank after the gap. Shared by every TEST below so the
// three agree with each other and with the real tool.
struct AcceptLayoutGraph {
  CouplingGraph g;
  fuse::MeshId u_mesh;
  fuse::RegionId u_region;
  brusselator::WindowGraphHandles handles;
};

[[nodiscard]] AcceptLayoutGraph build() {
  AcceptLayoutGraph out;
  out.u_mesh = out.g.add_mesh({.name = "u", .source = "generated:quad:16x16"});
  out.u_region = out.g.add_region({.name = "u_all", .mesh = out.u_mesh, .dimension = 2});

  const RankSet u_ranks{0, 1};
  const RankSet coupler_ranks{1, 1};
  const RankSet v_ranks{2, 1};
  const LaunchSpec u_launch{.mode = LaunchMode::Connect,
                            .is_acceptor = true,
                            .port_file = "/tmp/fuse-accept-layout-test.port"};
  const LaunchSpec v_launch{.mode = LaunchMode::Connect,
                            .port_file = "/tmp/fuse-accept-layout-test.port"};

  out.handles = brusselator::add_window_coupling(
      out.g, out.u_region, Rational{2, 1000}, Rational{3, 1000}, Rational{2, 100}, 1e-11, 60,
      fuse::AccelKind::None, fuse::WaveformIteration::Jacobi, coupler_ranks, u_ranks, v_ranks,
      u_launch, v_launch, LaunchSpec{});
  return out;
}

}  // namespace

TEST(AcceptLayoutPlacement, UsPlannedOwnedSizeIsItsWholeFieldWhenRanksUIsOneAfterD36) {
  AcceptLayoutGraph fx = build();

  const auto lowered = lower(fx.g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();

  // This process is rank 0 of the base communicator the accept-mode row
  // would eventually form (U's own physics rank); num_ranks=3 is
  // ranks_u + ranks_coupler + ranks_v, the row's own world size.
  fuse::native::NativeMeshBackend backend(/*rank=*/0, /*num_ranks=*/3);
  ASSERT_TRUE(backend.load(fx.g.mesh(fx.u_mesh), fx.g).ok());

  // THE RUNNER'S OWN PLACEMENT PLAN, reached directly and by name:
  // make_region_sizes() IS the sizing function schedule()/resolve_
  // buffers() consult, and "the plan sizes that field at N on this rank"
  // in the host's own E-RT-SIZE message is a direct report of this
  // function's return value.
  const fuse::RegionSizes sizes = fuse::make_region_sizes(fx.g, lowered.dag, backend);
  const std::size_t u_size = sizes(fx.u_region, static_cast<int>(fx.handles.u_component.value()));
  // Under the PRE-D36 declaration (U's own RankSet spanning the bundled
  // coupler rank too, RankSet{0, 2} instead of {0, 1}), this call returns
  // 128 -- half of U's real 16x16 = 256-cell field -- because D24's fix
  // decomposes a placement over its component's own declared RankSet
  // COUNT, which used to be 2 (U's own physics rank plus the coupler's
  // bundled one) rather than 1. This is "the plan sizes that field at 128
  // on this rank" from the host's own log, reproduced without MPI.
  EXPECT_EQ(u_size, 256u)
      << "U's planned owned size on rank 0 must be its whole field (16x16 = 256), "
         "not a decomposition over a declared RankSet that includes ranks U does "
         "not itself host";

  // schedule() itself must also succeed against this same callback: this
  // is as far into the Runner's own initialization pipeline as this
  // sandbox can reach for a graph whose window members live on different,
  // genuinely-partitioned ranks (this file's own header comment has the
  // full account of why Runner::initialize() itself cannot be reached
  // here, and that this is independent of D36).
  const fuse::ClockSet clocks{fx.g};
  const auto sched = schedule(fx.g, lowered.dag, clocks, sizes);
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
}

// The mirror on the coupler's own rank (1 of the same 3-rank world): U's
// own declared RankSet is now {0} (D36 -- it no longer spans the
// coupler's bundled rank), so rank 1 is OUTSIDE it, and `register_
// default_placements()`'s own `hosted_here` branch (D24,
// src/backend/native_backend.cpp) must hand U's placement an EMPTY view
// there -- size 0, not a fraction of U's field, and not the coupler's OWN
// placement (which is a different component with a different RankSet,
// and is not what this row checks).
TEST(AcceptLayoutPlacement, UIsNotPlacedAtAllOnTheCouplersOwnRank) {
  AcceptLayoutGraph fx = build();

  fuse::native::NativeMeshBackend backend(/*rank=*/1, /*num_ranks=*/3);
  ASSERT_TRUE(backend.load(fx.g.mesh(fx.u_mesh), fx.g).ok());

  // No lower() call in this test's own scope -- it checks make_region_
  // sizes()'s placement arithmetic directly, not anything a BuildRemapHalo
  // instruction would act on; an empty OperatorDAG is the correct,
  // zero-instruction input for that question.
  const fuse::RegionSizes sizes = fuse::make_region_sizes(fx.g, fuse::OperatorDAG{}, backend);
  const std::size_t u_size_here = sizes(fx.u_region, static_cast<int>(fx.handles.u_component.value()));
  EXPECT_EQ(u_size_here, 0u)
      << "rank 1 is the coupler's own bundled rank, outside U's declared RankSet "
         "{0} since D36 -- U's own mesh must size at 0 there, not a nonzero slice "
         "of a decomposition over a RankSet that no longer includes this rank";

  // The coupler's OWN placement of the same mesh, by contrast, is its
  // whole field: it is a genuine one-rank component (RankSet{1, 1}), so
  // position 0 of 1 is the full 256 cells -- unrelated to whether it
  // hosts any of U's fields (it hosts none; ComponentKind::Coupler owns
  // no fields at all, docs/dev/coupler-placement.md), and unchanged by
  // D36 (the coupler's own declared RankSet is the same before and
  // after this fix).
  const auto coupler_id = fx.g.find_component("coupler");
  ASSERT_TRUE(coupler_id.has_value());
  const std::size_t coupler_size = sizes(fx.u_region, static_cast<int>(coupler_id->value()));
  EXPECT_EQ(coupler_size, 256u)
      << "the coupler is a genuine one-rank placement of u_region wherever its own "
         "rank sits, regardless of whether it hosts any field there";
}

// V's own rank (2): V's window-transfer fields are declared on u_region
// too (brusselator_graph.hpp's own file header explains why), and V's
// declared RankSet is unchanged by D36 ({ranks_u + ranks_coupler,
// ranks_v} before and after) -- this is the row that pins V's own
// placement stays a full, undivided copy of u_region wherever V's one
// rank sits, exactly as it did before this fix touched only U's and the
// coupler's declarations.
TEST(AcceptLayoutPlacement, VsOwnPlannedSizeOnUsMeshIsUnaffectedByD36) {
  AcceptLayoutGraph fx = build();

  fuse::native::NativeMeshBackend backend(/*rank=*/2, /*num_ranks=*/3);
  ASSERT_TRUE(backend.load(fx.g.mesh(fx.u_mesh), fx.g).ok());

  // Same reasoning as UIsNotPlacedAtAllOnTheCouplersOwnRank above.
  const fuse::RegionSizes sizes = fuse::make_region_sizes(fx.g, fuse::OperatorDAG{}, backend);
  const std::size_t v_size = sizes(fx.u_region, static_cast<int>(fx.handles.v_component.value()));
  EXPECT_EQ(v_size, 256u)
      << "V's own RankSet ({ranks_u + ranks_coupler, ranks_v}) is unchanged by D36, so "
         "its planned size on u_region (its own single rank, position 0 of 1) is "
         "still the whole field";
}

// Graph-tool-output invariants (KNOWN_ISSUES.md D36): the shape
// `brusselator_graph_tool.cpp`'s own accept branch now declares --
// checked directly against the SAME `add_window_coupling()` call that
// branch makes (this file's own `build()`), the way test/verification/
// brusselator_graph_tool_lowfive_test.cpp checks its own branch's wiring
// directly rather than only through the byte-for-byte reference-YAML
// ctest row (`tools/verify-graph-tool-reference.cmake`, which this
// track added no new reference file for: no `reference_*accept*.yaml`
// exists in examples/brusselator/participants/ for either world size,
// confirmed by listing that directory -- accept mode has never had a
// committed reference, only mpmd does).
TEST(AcceptLayoutGraphToolOutput, UsAndTheCouplersRankSetsAreDisjoint) {
  AcceptLayoutGraph fx = build();
  const auto coupler_id = fx.g.find_component("coupler");
  ASSERT_TRUE(coupler_id.has_value());
  const RankSet& u_ranks = fx.g.component(fx.handles.u_component).ranks;
  const RankSet& coupler_ranks = fx.g.component(*coupler_id).ranks;
  EXPECT_FALSE(u_ranks.overlaps(coupler_ranks))
      << "U's own declared RankSet must no longer include the coupler's bundled "
         "rank (KNOWN_ISSUES.md D36): before this fix the two were a deliberate "
         "overlapping pair (RankSet's own documented \"overlap is legal\"), which "
         "is exactly what made D24's fix mis-size U's field";
}

TEST(AcceptLayoutGraphToolOutput, VsFirstRankEqualsTheAcceptorsWorldSize) {
  AcceptLayoutGraph fx = build();
  const int ranks_u = 1, ranks_coupler = 1;  // this row's own --ranks-u/--ranks-coupler
  const int acceptor_world_size = ranks_u + ranks_coupler;
  const RankSet& v_ranks = fx.g.component(fx.handles.v_component).ranks;
  EXPECT_EQ(v_ranks.min_rank(), acceptor_world_size)
      << "V's own Connect-mode block must still start exactly where the "
         "acceptor's (U's) own local MPI_COMM_WORLD ends -- unchanged by D36, "
         "which only moved the gap INSIDE that range from being part of U's own "
         "declared block to being the coupler's";
}

TEST(AcceptLayoutGraphToolOutput, UsAndVsBlocksWithTheGapBetweenThemValidate) {
  // The new shape must actually pass validate_launch_layout()'s own D36
  // gap rule -- test/transport/connect_test.cpp's own `ValidateLaunchLayout.
  // AGapExactlyCoveredByAnInProcessComponentValidates` pins the rule in
  // isolation; this row pins it against the REAL graph the tool builds,
  // so a future change to either side (the tool's own ranks, or the
  // rule's own gap check) that stops agreeing is caught here too.
  AcceptLayoutGraph fx = build();
  const auto r = fuse::transport::validate_launch_layout(fx.g);
  EXPECT_TRUE(r.ok()) << (r.ok() ? "" : r.error().message);
}
