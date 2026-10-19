#pragma once

/// D73 (automatic remap halo, owner decision 2026-10-18; execution point
/// moved in ROUND 2, 2026-10-18). See `OpKind::BuildRemapHalo`'s own doc
/// comment (fuse/ir/op.hpp) and `emit_remap_halo_instructions()`'s
/// (fuse/ir/lowering.hpp) for how these instructions get into the IR.
///
/// NOT A PUBLIC STEP THE APPLICATION MUST TAKE. `fuse::make_region_sizes()`
/// (fuse/backend/mesh_view.hpp) is the only caller: it invokes this
/// function on its own `dag` argument BEFORE building the `RegionSizes`
/// callback it returns, so the halo is built before any buffer is sized
/// from the mesh it grows (see `make_region_sizes`'s own doc for why that
/// ordering, not merely "early", is the one that matters). Round 1 of
/// this track had application code call this directly between loading
/// the backend and calling `schedule()`; KNOWN_ISSUES.md D77 records why
/// that could not be `Runner::initialize()` itself and why round 2 moved
/// the call here instead. This function remains declared in a public
/// header (not an internal/detail namespace) only because
/// `make_region_sizes()`'s own implementation, which calls it, lives in
/// a different translation unit (src/backend/serial_backend.cpp) in the
/// same library.

#include <cstddef>

#include "fuse/base/result.hpp"
#include "fuse/ir/dag.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace fuse {

class MeshBackend;

/// Executes every `OpKind::BuildRemapHalo` instruction `lower()` emitted
/// into `dag` (src/ir/rules_channel.cpp's `emit_remap_halo_instructions()`):
/// for each, calls `fuse::build_remap_halo()` (fuse/backend/remap_plan.hpp
/// -- which dispatches to `fuse::native::build_native_remap_halo` or to
/// `MoabMeshBackend::adopt_remap_halo` depending on `backend`'s concrete
/// type, exactly as `fuse::build_remap_halo` already does for a caller
/// that identifies the pair by hand) against the channel's own source and
/// target regions and the placement the instruction's own
/// `attrs.dst_rank_set` names (set once, at lowering, by
/// `emit_remap_halo_instructions()` -- see that function's own comment
/// for why this reads the frozen decision back rather than recomputing
/// `fuse::remap_placement()` a second time here).
///
/// `Runner::Impl::resolve_remap_halo()` stays the second, independent
/// gate: it VERIFIES, at `initialize()` time, that every `BuildRemapHalo`
/// instruction this function was supposed to satisfy actually was
/// (`has_remap_halo()`, a local bookkeeping lookup, no MPI call) --
/// refusing by name (`E-RT-REMAP-HALO-NOT-BUILT`) rather than attempting
/// the exchange itself, which by `initialize()` time would already be too
/// late to size anything correctly (the same ordering constraint that
/// moved this function's OWN call site out of application code and into
/// `make_region_sizes()`).
///
/// COLLECTIVE, per instruction, in instruction order: `build_remap_halo`
/// is itself collective over `backend`'s own communicator (its own header
/// comment), and every rank of that communicator must call this with the
/// SAME `dag` -- which they already have, since every rank compiles the
/// identical `ExecutionPlan`-independent `OperatorDAG` from the identical
/// `CouplingGraph` (ascending `ChannelId`, `lower()`'s own step 2 comment),
/// so the SET and ORDER of `BuildRemapHalo` instructions already agree
/// before this runs.
///
/// Returns the sum of cells adopted across every instruction (mirroring
/// `build_remap_halo`'s own return value; zero cells for an already-
/// aligned or single-rank pair is a legitimate, successful answer, not
/// "nothing happened") -- or the FIRST refusal, named with the channel it
/// was for. A graph with no cross-rank weighted channel at all is a
/// successful no-op (zero instructions, zero cells), so `make_region_
/// sizes()` makes this call unconditionally on every graph, the same way
/// `build_remap_halo` itself is an unconditional call.
[[nodiscard]] Result<std::size_t> execute_remap_halo_instructions(
    const CouplingGraph& graph, const OperatorDAG& dag, MeshBackend& backend);

}  // namespace fuse
