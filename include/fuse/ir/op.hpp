#pragma once

#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/time_level.hpp"
#include "fuse/base/units.hpp"
#include "fuse/model/channel.hpp"  // RemapFlags
#include "fuse/model/merge.hpp"

namespace fuse {

/// Spec Section 5.1's twelve primitives, plus Source/Sink boundary nodes so that
/// every arc has a producer and every value a consumer.
enum class OpKind {
  Sample, Accumulate, Interpolate, Extrapolate, Remap, Merge, Relax,
  Transport, Residual, Accelerate, Constrain, Diagnose,
  Source, Sink,

  /// Unit conversion by an exact rational factor. Separate from `Relax` (which
  /// scales by a double) because a unit factor must be exact: 1/1000 applied as
  /// a double is not 1/1000, and a conservation check downstream would then be
  /// comparing sums that differ in the last bits for no physical reason.
  ///
  /// Emitted by the compiler on the connection checker's authority, never on
  /// the user's -- which is why the dump prints it and the plan names it.
  Scale,

  /// Rank agreement point. Every rank in the coupling reaches this instruction
  /// in the same order, and each one learns the same outcome (all-succeeded, or
  /// which rank failed) before any of them proceeds. It is the only place a
  /// failure on one rank can be turned into a decision on all of them, and it
  /// is what stops a local error from becoming a hang at the next collective.
  Epoch,

  /// Reserved for multiscale. Declared so the enumerator ordinals do not move
  /// when they land, and REJECTED by the scheduler today -- see
  /// include/fuse/model/multiscale.hpp for the contracts, and the same
  /// reasoning: a reserved opcode that silently schedules as a no-op would let
  /// a micro-macro coupling compile and run as though the closure had been
  /// applied.
  Map, Reduce, Closure,

  /// Time windows track: subcycling (docs/user/glossary.md) -- one component
  /// of a time window of a waveform iteration taking several of its own time
  /// steps across the window, reading its partners' waveforms (delivered by
  /// the ordinary field-transfer chain: Remap/data redistribution/
  /// synchronisation, exactly as any other field transfer) and producing its
  /// own outgoing waveform for the next fixed-point sweep. Unlike every
  /// operator above, it is not a stateless function of its input buffers: it
  /// calls the owning component's advance()/save_state()/restore_state()
  /// directly (fuse::WindowCoupler's algorithm) and carries state across
  /// sweeps (the component's own subcycling clock, its waveform history for
  /// extrapolation, a pending step not aligned with the window end) that
  /// outlives one instruction execution. See fuse::WindowCoupler
  /// (fuse/runtime/window_coupler.hpp) for the algorithm and
  /// docs/dev/coupling-windows.md for why a component receives a mean, not a
  /// point sample.
  WindowAdvance,

  /// Unit 3 (S6 phase 3, Conformance): a boundary node, shaped like
  /// `Source` but for a channel whose data movement an APPLICATION
  /// performs outside the graph -- today, the LowFive-bound case
  /// (`FieldSpec::hdf5_path` set on the channel's source field), where the
  /// transfer is an ordinary HDF5 write/read FUSE never touches
  /// (docs/user/lowfive.md, "a LowFive-bound channel is an
  /// application-performed transfer").
  ///
  /// `lower_channel()` (src/ir/rules_channel.cpp) emits this carrying the
  /// SAME single input `Transport` would have had for the identical
  /// channel -- the source field's own producer (its ordinary `Source`
  /// or, for a window-scoped export, its `WindowAdvance`) -- but its own
  /// `execute()` case (src/runtime/runner.cpp) never reads that input: it
  /// is purely an ORDERING EDGE, there so the scheduler places this node
  /// exactly where `Transport` would stand, on every process, the same
  /// way it does for the distributed ordinary case (the stream is static
  /// and identical everywhere). `Transport`'s own input carries a REAL
  /// data dependency (it moves the producer's own value); this node's
  /// input carries none, because the application's own write is what
  /// fills its output -- but the ORDERING that dependency enforces is
  /// still required: whatever the application writes after the
  /// producer's own last substep must be consumed after that substep and
  /// before the importer's own advance, the identical slot `Transport`
  /// occupies. Severing the edge entirely (an earlier version of this
  /// node had no input at all) left the scheduler with nothing forcing
  /// that order, and a LowFive-bound channel's own chain could and did
  /// get scheduled before the exporting member's own `WindowAdvance` in
  /// the SAME sweep -- reading last sweep's state instead of this one's
  /// (MEASURED this session; not yet recorded as its own divergence note
  /// in docs/user/lowfive.md). The rest of the channel's chain (Scale,
  /// Remap, Extrapolate, the window-importing
  /// Sink) is unchanged downstream of it.
  ///
  /// Deliberately NOT one of `acyclicity.cpp`'s Source/Sink endpoint kinds
  /// (`prove_acyclic()`'s own Source side=1 / Sink side=0 classification):
  /// an externally-delivered arc performs no computation and closes no
  /// algebraic loop through this graph, so including it there would make
  /// `validate()` report a cycle across two components that, in the
  /// deployment this exists for, never meet inside one coupled run at all.
  ///
  /// Its own output arc is NOT given `plan_buffers()`'s `component_owned`
  /// treatment (src/schedule/buffers.cpp): there is no registered field
  /// storage to alias it to for a window-scoped channel (the window
  /// Sink's own output is excluded from that aliasing for the identical
  /// reason, `is_window_arc()` there) -- it stays a private, Runner-owned
  /// scratch buffer, reachable by the application only through the new
  /// `field_view` ABI accessor (docs/user/lowfive.md, Unit 3).
  ExternalDelivery,

  /// D73 (automatic remap halo, owner decision 2026-10-18). A real
  /// collective MPI exchange runs once at setup time, but `is_collective()`
  /// below returns false for it (see that function's own comment on this
  /// case) -- NO DATA: emitted by the lowering (src/ir/rules_channel.cpp,
  /// `channel_needs_remap_halo()` / the front pass in src/ir/lowering.cpp)
  /// once per cross-rank weighted (non-nodal, non-consistent) channel --
  /// the branch `fuse::remap_moves_placement` (fuse/model/remap_kind.hpp)
  /// answers false for, so `Runner::Impl::resolve_remap()` takes the
  /// general conservative/MLS/tempest/intx builder rather than the
  /// nodal-distributed one. `Node::inputs` is empty, like `Source`, and
  /// `Node::output` is the invalid `ArcId{}`: this instruction moves no
  /// value through the dataflow graph, it exchanges MESH GEOMETRY on the
  /// backend the Remap that follows will read. `expand()` stamps every
  /// instance at `t = 0`, `TickPhase::BeforeTicks` -- the same instant
  /// `Source` gets -- so it sorts to the front of the instruction stream,
  /// ahead of every per-period chain.
  ///
  /// Its REAL geometry exchange cannot run inside `Runner::execute()`'s
  /// hot loop: the matching "ORDER MATTERS" note (fuse/backend/moab_
  /// backend.hpp, fuse/backend/remap_plan.hpp) says a mesh's remap halo
  /// must be built BEFORE `make_region_sizes`/`plan_buffers`/`schedule()`
  /// size any buffer on it, and by the time a `Runner` exists `schedule()`
  /// already has. `fuse::make_region_sizes(graph, dag, backend)` therefore
  /// runs `fuse::execute_remap_halo_instructions()`
  /// (fuse/backend/remap_halo_instructions.hpp) before it reads any entity
  /// count, satisfying every `BuildRemapHalo` node `lower()` emitted at the
  /// one point every caller already passes `dag` through (owner decision
  /// 2026-10-18, KNOWN_ISSUES.md D73 and D77); `Runner::Impl::resolve_
  /// instructions()` only VERIFIES, collectively, that each one was
  /// (`NativeMeshBackend::has_remap_halo` / `MoabMeshBackend::has_remap_
  /// halo`), and refuses by name if not.
  BuildRemapHalo,
};

/// Whether executing this operator involves the other ranks of its exchange.
///
/// It decides what a rank may skip after it has failed mid-step. A rank that
/// skips a COLLECTIVE leaves its partners inside that collective waiting for
/// it, which is the same hang as returning early; a rank that skips local
/// arithmetic costs nothing, because the run is being abandoned anyway.
///
/// A function rather than a `default:` in the loop so a test can enumerate
/// every operator and fail when a new one is added without a decision.
[[nodiscard]] constexpr bool is_collective(OpKind k) noexcept {
  switch (k) {
    case OpKind::Transport:  // moves values between two rank sets
    case OpKind::Epoch:      // the agreement itself
    // Constrain reduces the two conservation integrals over the base
    // communicator when one is configured (KNOWN_ISSUES.md D10, 2026-09-25).
    // Before that it was rank-local and REFUSED a multi-rank target, because
    // adding a reduction to a non-collective operator is a hang: a rank that
    // failed earlier in the period skips it while its partners wait inside
    // it. Collective is the classification that makes the reduction safe;
    // with no communicator it is the same local arithmetic as before, merely
    // no longer skipped after a failure, which costs nothing.
    case OpKind::Constrain:
      return true;
    case OpKind::Source: case OpKind::Sink: case OpKind::Sample:
    case OpKind::Accumulate: case OpKind::Interpolate: case OpKind::Extrapolate:
    case OpKind::Remap: case OpKind::Merge: case OpKind::Relax:
    case OpKind::Residual: case OpKind::Accelerate:
    case OpKind::Diagnose: case OpKind::Scale:
      return false;
    // D73: the REAL geometry exchange this instruction represents
    // (`fuse::native::build_native_remap_halo`, `fuse::build_remap_halo`,
    // each its own MPI_Allgather/MPI_Allgatherv plus a cell exchange,
    // agreed over the base communicator already) runs once, before
    // `schedule()` ever sizes a buffer -- see `fuse::execute_remap_halo_
    // instructions()` (fuse/backend/remap_halo_instructions.hpp) and this
    // enumerator's own doc comment above. By the time this instruction
    // reaches `Runner::Impl::execute()`'s per-period hot loop, the
    // exchange has already happened (or been refused) at setup time in
    // `Runner::Impl::resolve_instructions()`, which only reads the
    // backend's own local bookkeeping (`has_remap_halo()`, no MPI call).
    // Nothing a rank could skip here touches another rank, so unlike
    // `Epoch`/`Transport`/`Constrain` this is not collective at the point
    // this classification governs (`step_failure`'s skip-after-failure
    // rule, runner.cpp / iteration.cpp).
    case OpKind::BuildRemapHalo:
      return false;
    // Same reasoning as Source: it involves no other rank set of this
    // graph's own exchange (the application-performed transfer that
    // fills it happens entirely outside this instruction stream). Its
    // own input (this enum's own comment on ExternalDelivery) is an
    // ordering edge within ONE process, never a cross-rank exchange.
    case OpKind::ExternalDelivery:
      return false;
    // Reserved for multiscale, refused by the scheduler. Answering "not
    // collective" here would be a guess about an operator nobody has written.
    case OpKind::Map: case OpKind::Reduce: case OpKind::Closure:
      return true;
    // Subcycling runs on the member's own ranks only (the task brief's own
    // wording): it calls that member's component directly and involves no
    // other rank set. The exchange that delivers a PARTNER's waveform to it
    // is the ordinary field-transfer chain (Remap/Transport/Epoch), each
    // classified on its own merits above; WindowAdvance itself moves nothing
    // between rank sets.
    case OpKind::WindowAdvance:
      return false;
  }
  return true;  // an unclassified operator is assumed collective: skipping one
                // that is hangs, executing one that is not merely wastes work
}

[[nodiscard]] constexpr const char* to_cstring(OpKind k) noexcept {
  switch (k) {
    case OpKind::Sample:      return "Sample";
    case OpKind::Accumulate:  return "Accumulate";
    case OpKind::Interpolate: return "Interpolate";
    case OpKind::Extrapolate: return "Extrapolate";
    case OpKind::Remap:       return "Remap";
    case OpKind::Merge:       return "Merge";
    case OpKind::Relax:       return "Relax";
    case OpKind::Transport:   return "Transport";
    case OpKind::Residual:    return "Residual";
    case OpKind::Accelerate:  return "Accelerate";
    case OpKind::Constrain:   return "Constrain";
    case OpKind::Diagnose:    return "Diagnose";
    case OpKind::Source:      return "Source";
    case OpKind::Sink:        return "Sink";
    case OpKind::Scale:       return "Scale";
    case OpKind::Epoch:       return "Epoch";
    case OpKind::Map:         return "Map";
    case OpKind::Reduce:      return "Reduce";
    case OpKind::Closure:     return "Closure";
    case OpKind::WindowAdvance: return "WindowAdvance";
    case OpKind::ExternalDelivery: return "ExternalDelivery";
    case OpKind::BuildRemapHalo:   return "BuildRemapHalo";
  }
  return "Unknown";
}

enum class ValueKind { Instant, Waveform };

[[nodiscard]] constexpr const char* to_cstring(ValueKind k) noexcept {
  return k == ValueKind::Waveform ? "Waveform" : "Instant";
}

/// The type of an SSA value. Spec Section 2: every arc carries its time level, so a
/// staggering error is a type error found at lowering.
struct ValueType {
  ValueKind    kind = ValueKind::Instant;
  RegionId     region;
  int          components = 1;
  Units        units{};
  Conservation conservation = Conservation::None;
  TimeLevel    time{};
  int          rank_set = -1;  // component index the value lives on; -1 = unplaced

  friend bool operator==(const ValueType&, const ValueType&) = default;

  [[nodiscard]] std::string to_string() const {
    std::string s = to_cstring(kind);
    s += "<r";
    s += region.valid() ? std::to_string(region.value()) : std::string{"?"};
    if (components != 1) {
      s += "x";
      s += std::to_string(components);
    }
    s += ", ";
    s += units.to_string();
    if (conservation != Conservation::None) {
      s += ", ";
      s += to_cstring(conservation);
    }
    s += ", ";
    s += time.to_string();
    if (rank_set >= 0) {
      s += ", ranks=";
      s += std::to_string(rank_set);
    }
    s += ">";
    return s;
  }
};

/// Can `consumer` accept a value of type `producer`?
/// allow_time_mismatch comes from Channel::time_level_mismatch_reason and
/// relaxes ONLY the time check (spec Section 4.3).
[[nodiscard]] inline bool assignable_from(const ValueType& producer,
                                          const ValueType& consumer,
                                          bool allow_time_mismatch) noexcept {
  if (producer.kind != consumer.kind) return false;
  if (producer.region != consumer.region) return false;
  if (producer.components != consumer.components) return false;
  if (!(producer.units == consumer.units)) return false;
  if (producer.conservation != consumer.conservation) return false;
  if (!allow_time_mismatch && !compatible(producer.time, consumer.time)) {
    return false;
  }
  return true;
}

/// Flat attribute bag. Each OpKind reads only what it needs; keeping it flat
/// keeps Node cheap to copy and the dump format stable.
struct OpAttrs {
  AccumOp       accum = AccumOp::Instant;
  TemporalBasis basis = TemporalBasis::None;
  int           degree = 0;
  double        omega = 1.0;
  int           order = 0;
  int           lag = 0;
  RemapFlags    remap_flags = RemapFlags::None;
  std::string   method;
  int           src_rank_set = -1;
  int           dst_rank_set = -1;

  /// `Scale`'s exact factor, and `Epoch`'s reason. Both default to values the
  /// dump omits, so no existing golden file moves.
  Rational      scale{1};
  std::string   reason;

  /// `Merge`'s rule. Carried as the enumerator rather than parsed back out of
  /// `reason`, so the interpreter cannot disagree with the graph about which
  /// combination it is performing.
  MergeOp       merge = MergeOp::Sum;

  /// Time windows track: `OpKind::WindowAdvance` (subcycling) has EMPTY
  /// `Node::inputs` -- like `Source` -- because the field transfer that
  /// delivers its import waveform is on the OTHER end of a two-way time
  /// window, and making that delivery a real producer/consumer edge back
  /// into this node creates a genuine cycle in the untimed dataflow graph
  /// (measured directly: `OperatorDAG::topological_order()` on exactly this
  /// shape refuses with only 2 of 12 nodes ordered). This edge id names
  /// that field transfer's own final edge WITHOUT adding a dependency: the
  /// scheduler's window-causality check (src/schedule/scheduler.cpp) reads
  /// it to verify ordering by TIME (`TickPhase`) instead, the same
  /// causality contract every `Source`/`Sample` boundary read already
  /// relies on rather than an SSA edge. `ArcId{}` (invalid, the default) is
  /// every node that is not `WindowAdvance` -- `attrs_string()` omits it
  /// then, so no existing reference dump moves.
  ArcId         window_import{};

  /// Review 2026-10-11 IS-4: true for the Sink `emit_field_sink()`
  /// (src/ir/lowering.cpp) builds for a window-scoped import -- the one
  /// whose output is K samples wide, delivered unchanged from its provider,
  /// and must NOT be aliased onto the component's own native (single-value)
  /// storage the way an ordinary Sink's output is (`plan_buffers()`,
  /// src/schedule/buffers.cpp, is the reader). Previously this was decided
  /// by a `"/window"` label suffix, which an ordinary field legally named
  /// `q/window` would also match, silently stealing its own delivery into
  /// coupler-owned scratch. Not printed by `attrs_string()` below: a boolean
  /// set structurally by lowering changes no existing dump (the label still
  /// carries the same text it always did, for readability only).
  bool          window_import_sink = false;

  /// Review 2026-10-11 IS-9: the channel `type_check()` (src/ir/
  /// lowering.cpp) authorizes a Sink's time-level mismatch against. Set by
  /// `emit_field_sink()` to the channel whose contribution the merge
  /// adopted as newest for a merged field (which need not be `Node::origin`
  /// -- see that function's own comment on why the two must stay
  /// independent), or to `origin` itself for every other Sink, so a reader
  /// of `type_check()` need not special-case "is this field merged" to know
  /// which channel its authorization question is about. Not printed by
  /// `attrs_string()` below, like `window_import_sink` above: it changes no
  /// existing dump.
  ChannelId     merge_auth_channel{};

  friend bool operator==(const OpAttrs&, const OpAttrs&) = default;
};

struct Node {
  NodeId               id;
  OpKind               kind = OpKind::Source;
  std::vector<ArcId>   inputs;
  ArcId                output;
  OpAttrs              attrs;
  ChannelId            origin;  // which channel emitted this node
  std::string          label;
};

struct Arc {
  ArcId       id;
  NodeId      producer;
  ValueType   type;
  std::string name;
};

}  // namespace fuse
