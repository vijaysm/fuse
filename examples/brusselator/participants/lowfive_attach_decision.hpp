#pragma once

// S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track: the
// PURE, MPI-free and HDF5-free half of the participants' programmatic VOL
// attach (lowfive_attach.cpp's own header comment has the full account of
// what wraps this). Decides, from the loaded coupling graph alone, which
// of two outcomes this rank's own attach call must reach, or refuses:
//
//   - every HDF5-bound channel this rank takes part in is Mpmd-launched on
//     both ends -> attach a DistMetadataVOL over an intercommunicator this
//     process still has to form (fuse::plan_lowfive_rendezvous(), already
//     tier-0 tested, supplies the MPI_Comm_split/MPI_Intercomm_create
//     arguments);
//   - no HDF5-bound channel names this rank at all -> refused
//     (E-LOWFIVE-ATTACH-NOCHANNELS): a caller that reaches this file at
//     all already decided (from --transport lowfive) that SOME field
//     needs interception, so a rank with no part in any of them is a
//     graph/launch mismatch, not a legitimate no-op -- the charter's own
//     "no silent fallback";
//   - any HDF5-bound channel this rank takes part in has an ACCEPT-mode
//     (LaunchMode::Connect) component on either end -> refused clearly
//     (E-LOWFIVE-ATTACH-ACCEPT; docs/user/lowfive.md, KNOWN_ISSUES.md
//     D14's own accept-mode intercommunicator question, still open) --
//     never a half-configured VOL;
//   - two or more HDF5-bound channels this rank takes part in do not all
//     partition the world into the SAME two groups -> refused
//     (E-LOWFIVE-ATTACH-INCONSISTENT): src/transport/lowfive_plugin_
//     graph.cpp's own ConfigureDistributed() comment names this exact gap
//     ("a graph with more than one [channel] would need this loop to
//     confirm every ... channel agrees on the SAME writer/reader pair
//     this process belongs to, which is not yet checked here") -- this
//     function is the check that comment asked for, applied to the
//     participants' own bidirectional (u_out/v_out) exchange, where the
//     writer/reader LABELS swap between the two channels but the two
//     GROUPS {U-ranks},{V-ranks} do not.
//
// Nothing here touches MPI or HDF5; every type is fuse::model/fuse::
// transport header-only arithmetic, exercised directly by test/
// verification/brusselator_lowfive_attach_test.cpp with no LowFive, no
// HDF5, and no MPI (mirrors the "force_this_process, no real
// communicator" tier-0 discipline KNOWN_ISSUES.md C1 already documents
// for this codebase's other MPI-shaped pure functions).

#include <cstddef>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/component.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/transport/lowfive.hpp"

namespace fuse_examples::lowfive_attach {

/// One HDF5-bound channel's writer/reader roles, as read from the graph --
/// `field_name` is kept only so a refusal message can name something a
/// human can find in the YAML (the SOURCE field's own qualified name,
/// `fuse::build_lowfive_bindings()`'s own "<component>.<field>"
/// convention).
struct ChannelRoles {
  std::string field_name;
  fuse::RankSet writer_ranks;
  fuse::RankSet reader_ranks;
  fuse::LaunchMode writer_launch = fuse::LaunchMode::InProcess;
  fuse::LaunchMode reader_launch = fuse::LaunchMode::InProcess;
  /// The two COMPONENTS' own declared names (round 2, "role-ordered
  /// serve") -- distinct from `field_name`'s own "<component>.<field>"
  /// qualification, needed by `decide_serve_roles()` below, which
  /// assigns a role per COMPONENT, not per field. Empty by default so
  /// existing `decide_attach_mode()` callers/tests that never set them
  /// are unaffected -- that function never reads either one.
  std::string writer_name;
  std::string reader_name;
};

enum class AttachMode { kSingleProcess, kDistributed };

struct Decision {
  AttachMode mode = AttachMode::kSingleProcess;
  /// Meaningful only when `mode == kDistributed`.
  fuse::LowFiveRendezvousPlan rendezvous;
};

/// `channels` must already be filtered to the ones naming `this_rank` on
/// either side (lowfive_attach.cpp's own job, mirroring the existing
/// env-route plugin's `ConfigureDistributed()` loop in
/// src/transport/lowfive_plugin_graph.cpp) -- an empty list is refused
/// here, not treated as "nothing to do." `this_rank` is the base
/// communicator's own rank number (MPI_COMM_WORLD's, in MPMD launch,
/// exactly as lowfive_plugin_graph.cpp's own comment states and this
/// track's MEASURED EVIDENCE confirms).
[[nodiscard]] inline fuse::Result<Decision> decide_attach_mode(
    const std::vector<ChannelRoles>& channels, int this_rank) {
  if (channels.empty()) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-NOCHANNELS",
        "rank " + std::to_string(this_rank) +
            " has --transport lowfive but no HDF5-bound channel in the "
            "loaded graph names it on either side; refusing rather than "
            "silently skipping interception"};
  }

  for (const ChannelRoles& ch : channels) {
    if (ch.writer_launch == fuse::LaunchMode::Connect ||
        ch.reader_launch == fuse::LaunchMode::Connect) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-ACCEPT",
          "channel '" + ch.field_name +
              "' has an accept-mode (LaunchMode::Connect) component on one "
              "end; accept mode's own merged base communicator is never "
              "exposed back to a caller (KNOWN_ISSUES.md D14), so this "
              "attach has no intercommunicator to form for it and refuses "
              "rather than falling back to a half-configured VOL"};
    }
  }

  // Canonical channel: the first one, exactly as the existing env-route
  // plugin's ConfigureDistributed() picks "the first LowFive-bound
  // channel found."
  const ChannelRoles& canon = channels.front();
  auto plan = fuse::plan_lowfive_rendezvous(canon.writer_ranks, canon.reader_ranks, this_rank);
  if (!plan.ok()) return plan.error();

  // Every OTHER qualifying channel must partition the world into the
  // IDENTICAL two groups -- same two RankSets, regardless of which one is
  // labelled writer and which reader (the Brusselator's own bidirectional
  // u_out/v_out exchange swaps the labels between the two channels; see
  // lowfive_attach.cpp's own header comment for why that is still one
  // intercommunicator, not two).
  for (std::size_t i = 1; i < channels.size(); ++i) {
    const ChannelRoles& other = channels[i];
    const bool same_order = other.writer_ranks.same_as(canon.writer_ranks) &&
                            other.reader_ranks.same_as(canon.reader_ranks);
    const bool swapped = other.writer_ranks.same_as(canon.reader_ranks) &&
                         other.reader_ranks.same_as(canon.writer_ranks);
    if (!same_order && !swapped) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-INCONSISTENT",
          "channel '" + other.field_name + "' and channel '" + canon.field_name +
              "' do not partition the world into the same two groups; this "
              "attach forms exactly one intercommunicator per process and "
              "cannot serve a third group"};
    }
  }

  Decision d;
  d.mode = AttachMode::kDistributed;
  d.rendezvous = plan.value();
  return d;
}


// ---------------------------------------------------------------------
// Round 2 ("role-ordered serve"): MEASURED deadlock, host sample
// 2026-10-13 (verification.brusselator.participants.lowfive.mpmd.3, 120s
// TIMEOUT). Both U and V were found inside H5Fclose -> LowFive::
// VOLBase::_file_close -> DistMetadataVOL::file_close
// (vol-dist-metadata.cpp:355-358 in this host's own LowFive checkout,
// /Users/mahadevan/Code/hdf5/LowFive rev 7f3a58b) -> DistMetadataVOL::
// serve_all (:38-72) -> Index::serve (vol-dist/index.cpp:106-167), which
// opens with a `local.barrier()` (:110) and then polls each intercomm's
// RPC server (`s.probe()`/`s.process()`, :145-161) until EVERY one of
// them reports "finish" -- which only happens once the PEER actively
// queries this file (Query::get_filenames(), vol-dist/query.cpp:106-139,
// itself a LOCAL broadcast, :125, fed by the SAME RPC round trip over the
// intercomm). `DistMetadataVOL::serve_on_close` is a PUBLIC bool member,
// default true (lowfive/vol-dist-metadata.hpp:33), and `file_close()`
// calls `serve_all()` for every local in-memory file whenever it is true
// (vol-dist-metadata.cpp:355, `serve_on_close && match_any(...)`) --
// so round 1's own `bruss_lowfive_serve_after_write()` call, placed
// AFTER H5Fclose, was a NO-OP: the serve had already happened INSIDE
// H5Fclose, before this file's own code ever ran a single line. On the
// row's own graph (`iteration: Jacobi`, `max_iterations 60`,
// `acceleration None`), both components export at the end of each
// iteration before either imports (round 1's own design notes named
// this exact risk; it is no longer hypothetical) -- so both close their
// own write, both block inside `serve_all()` waiting to be QUERIED, and
// neither has yet reached the read that would query the other: mutual
// deadlock, exactly the sample above.
//
// FIX: `lowfive_attach.cpp` sets `vol.serve_on_close = false` on the
// constructed `DistMetadataVOL`, so `H5Fclose` never serves on its own;
// serving becomes an explicit decision this namespace's own pure
// functions make. `decide_serve_roles()` below assigns the two
// components in the exchange one of two roles from their RankSets alone
// (the lower `min_rank()` serves first); `serve_state_step()` is the
// per-process state machine that decides, for each write/read event,
// whether to serve immediately, defer, receive-then-flush a deferred
// serve, or refuse. See lowfive_attach.cpp's own header comment for the
// full account of why this is deadlock-free and lowfive_attach.h for the
// NOT-NOW follow-up (a helper thread cannot be sound without patching
// LowFive itself: the same `local` communicator serves BOTH the barrier/
// ibarrier serve path above and the broadcast consumer path, and HDF5 is
// built Threadsafety: no).

/// Which of the two components in a bidirectional LowFive exchange
/// serves its own export IMMEDIATELY at write time (blocking until the
/// peer reads it), and which one defers its own serve until its own
/// next import, flushing the deferred serve only after receiving the
/// peer's file first. Assigned once, from the two components' own
/// RankSets alone (`component_a`/`component_b`, order-independent) --
/// never from a runtime condition, so every rank of EITHER component
/// reaches the IDENTICAL assignment (the collective-safety requirement
/// `serve_all()`'s own `local.barrier()` places on every rank of a
/// component: the standing rule that a collective must be reached from
/// data every rank shares, never a locally computed condition).
struct ServeRoleAssignment {
  std::string first_server_name;
  std::string first_reader_name;
};

/// Errors: `E-LOWFIVE-ATTACH-SERVE-OVERLAP` (the two components' own
/// RankSets share a rank -- this protocol needs two DISJOINT sides, one
/// per role), `E-LOWFIVE-ATTACH-SERVE-EMPTY` (either RankSet is empty --
/// a component with no ranks can neither serve nor read).
[[nodiscard]] inline fuse::Result<ServeRoleAssignment> decide_serve_roles(
    const std::string& a_name, const fuse::RankSet& a_ranks, const std::string& b_name,
    const fuse::RankSet& b_ranks) {
  if (a_ranks.overlaps(b_ranks)) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-SERVE-OVERLAP",
        "component '" + a_name + "' and component '" + b_name +
            "' own RankSets overlap; the role-ordered serve protocol needs two DISJOINT "
            "sides, one first-server and one first-reader"};
  }
  if (a_ranks.empty() || b_ranks.empty()) {
    return fuse::Failure{"E-LOWFIVE-ATTACH-SERVE-EMPTY",
                        "component '" + (a_ranks.empty() ? a_name : b_name) +
                            "' has an empty RankSet; it can neither serve nor read"};
  }
  ServeRoleAssignment out;
  if (a_ranks.min_rank() < b_ranks.min_rank()) {
    out.first_server_name = a_name;
    out.first_reader_name = b_name;
  } else if (b_ranks.min_rank() < a_ranks.min_rank()) {
    out.first_server_name = b_name;
    out.first_reader_name = a_name;
  } else {
    // UNREACHABLE given RankSet's own semantics, kept only as a
    // deterministic, defensive tie-break rather than an assert: two
    // DISJOINT, NONEMPTY RankSets cannot share a minimum rank (the
    // global minimum of their union belongs to exactly one of them,
    // which is exactly what the overlap check above already rules out
    // sharing). Not exercised by any test -- there is no reachable
    // input that takes this branch (the charter's own "do not write a
    // test that cannot fail").
    if (a_name < b_name) {
      out.first_server_name = a_name;
      out.first_reader_name = b_name;
    } else {
      out.first_server_name = b_name;
      out.first_reader_name = a_name;
    }
  }
  return out;
}

enum class ServeRole { kFirstServer, kFirstReader };

// ---------------------------------------------------------------------
// ROUND 2 REVIEW (rejected before any host run): the first round-2
// attempt above (an event model with only kExportWrittenAndClosed/
// kImportAboutToOpen/kFinalize, flushing a first-reader's own deferred
// serve INSIDE the pre-open hook) was reviewed against LowFive's own
// source and found to deadlock exactly as before, one step later. Two
// more facts, read from /Users/mahadevan/Code/hdf5/LowFive/src/
// vol-dist-metadata.cpp (rev 7f3a58b), that the first attempt missed:
//
//   - The CONSUMER's H5Fopen of a remote in-memory file IS the receive.
//     `DistMetadataVOL::file_open()` (line 230 on) builds a `Query` and
//     calls the producer's "get_file_hierarchy" RPC (line 278); its own
//     comment says plainly: "force the producer to enter serve_all to
//     get file hierarchy" (lines 255-257). This RPC round trip is what
//     actually moves data across the intercomm -- nothing in shim.cpp's
//     own PRE-open hook does or can substitute for it; `H5Fopen` itself
//     performs it, automatically, as an ordinary VOL callback dispatch.
//   - The consumer's H5Fclose of that remote file is what ENDS the
//     producer's serve. `DistMetadataVOL::file_close()`'s `RemoteFile`
//     branch (line ~333 on) calls `f->obj.call<void>("file_close")`
//     (line 337), then `MPI_Barrier(local)` (line 339), then root calls
//     `f->obj.self_->finish(0)` (line 342) -- `finish(0)` is exactly
//     what lets `Index::serve()`'s own probe/process loop (vol-dist/
//     index.cpp:145-161) exit for that client. The producer's `serve_
//     all()` call therefore blocks until the CONSUMER has done BOTH its
//     own open AND its own close of that specific file -- not until the
//     consumer has merely attempted to open it.
//
// The first attempt's `kImportAboutToOpen` hook ran its flush (a second,
// explicit `serve_all()` for the first-reader's OWN deferred file)
// BEFORE this process's own H5Fopen of the PEER's file -- i.e. before
// this process had done anything the peer's own pending `serve_all()`
// could possibly be released by. Both sides then block: the first-
// server is stuck in its original `kServeNow` from its first write,
// waiting for the first-reader's own open+close; the first-reader is
// stuck in the premature flush the pre-open hook just entered, waiting
// for the first-SERVER's open+close of the newly-served file -- which
// the first-server cannot perform, because it is still blocked in its
// own first `kServeNow`. Circular wait, same two LowFive-level stacks
// (both inside `Index::serve`'s probe loop) the host would have measured
// for round 1, one write/read cycle later. The tier-0 tests for the
// first attempt passed because they drove the state machine only
// against its OWN event model (hook call order), never against what the
// PEER actually observes (its serve is released by MY open AND close,
// not by MY merely calling a hook named "before read").
//
// FIX, this round: move the flush from the PRE-open hook to a NEW event
// fired AFTER this process's own H5Fclose of the peer's file --
// `kImportClosed`, action `kFlushDeferred` -- so a first-reader's own
// deferred serve is only entered once this process has actually done
// the open+close that, symmetrically, is what WOULD release a peer's
// own pending serve. `kImportAboutToOpen` becomes a plain, unconditional
// `kReceiveOnly` for every role and every state: it does nothing
// LowFive-level at all now (see the next paragraph for why), kept only
// as a validated event in the sequence for symmetry with the real
// shim.cpp call sites and so a future mutation that reintroduces a flush
// there is caught by this file's own model check below, not just by a
// human re-reading this comment.
//
// `broadcast_files()` does NOT belong in either hook, and this round
// removes the call lowfive_attach.cpp's first attempt made from the
// pre-open hook. Read directly from source
// (src/vol-dist-metadata.cpp:75-103): it operates ENTIRELY over `local`
// (never touches an intercomm), serializing `files` on one local rank
// (`root`, default 0) and broadcasting it to the OTHER ranks of the SAME
// local communicator -- propagating an ALREADY-KNOWN file registry
// across a MULTI-RANK component's own sibling ranks, not communicating
// with the peer component at all. `grep -rn broadcast_files` across this
// LowFive checkout's own `src/`/`include/` finds zero internal callers;
// its only caller anywhere in the repository is the Python binding
// (`bindings/python/lowfive.cpp:126`), a user-facing utility, never
// invoked by `file_open()`/`file_close()` themselves. The Brusselator's
// own `lowfive.mpmd.3` row places exactly one rank per component (U{1},
// V{1}), so even a harmless call would have nothing to propagate to;
// keeping an unjustified collective "for safety" is worse than removing
// it, since every future reader would have to re-derive this same
// analysis to confirm it is inert. Recorded here as a NOT-NEEDED-FOR-
// THIS-ROW gap, not a fix: a future multi-rank-per-component layout may
// need an explicit `broadcast_files()` call so that only one local rank
// pays the RPC round trip per file; not attempted this round.

/// One process's own write/read lifecycle events, in the SAME terms
/// lowfive_attach.cpp's own bruss_lowfive_serve_after_write()/
/// bruss_lowfive_receive_before_read()/bruss_lowfive_after_read_close()/
/// bruss_lowfive_finalize() see them. `kImportClosed` is new this round
/// -- fired once per read, after this process's OWN H5Fclose of the
/// peer's file, which is the event LowFive's own source ties the serve
/// release to (this header's own "ROUND 2 REVIEW" comment above).
enum class ServeEvent { kExportWrittenAndClosed, kImportAboutToOpen, kImportClosed, kFinalize };

/// What `serve_state_step()` decided THIS process must do next. The
/// caller (lowfive_attach.cpp) maps each one onto the real
/// `DistMetadataVOL::serve_all()` call (or onto nothing at all, for
/// `kReceiveOnly`/`kNoopAfterImport`/`kNoopFinalizeClean` -- see this
/// header's own "ROUND 2 REVIEW" comment for why `broadcast_files()` is
/// no longer called from either hook).
enum class ServeAction {
  kServeNow,            // kExportWrittenAndClosed, first-server: serve_all() now, blocking
  kDeferServe,          // kExportWrittenAndClosed, first-reader: remember a serve is owed
  kReceiveOnly,         // kImportAboutToOpen (always, any role/state): no LowFive call at all
  kFlushDeferred,       // kImportClosed, first-reader with a deferred serve: serve_all() now, blocking
  kNoopAfterImport,     // kImportClosed, nothing to flush (first-server always; first-reader with nothing pending)
  kNoopFinalizeClean,   // kFinalize, nothing left pending
};

/// Per-process state the machine threads across calls -- exactly the
/// `deferred_pending` bit `serve_state_step()` needs and nothing else.
/// Default-constructed with `role = kFirstServer`; a real caller always
/// sets `role` from `decide_serve_roles()`'s own answer before the first
/// event.
struct ServeState {
  ServeRole role = ServeRole::kFirstServer;
  bool deferred_pending = false;
};

/// The protocol, event by event (see this header's own "ROUND 2 REVIEW"
/// comment above for the deadlock the FIRST round-2 attempt had, and the
/// LowFive-source citations this one is built from):
///
///   - kExportWrittenAndClosed, first-server: kServeNow (blocks until
///     the peer's own open+close of THIS file releases it -- see the
///     INVARIANT below for why that peer always arrives, and in time).
///   - kExportWrittenAndClosed, first-reader: kDeferServe. Refused
///     (`E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER`) if a PREVIOUS deferred
///     serve was never flushed by an intervening kImportClosed -- the
///     alternation this protocol depends on (see INVARIANT) was
///     violated, and silently replacing the first deferred serve with
///     the second would drop the first file's data with no diagnostic.
///   - kImportAboutToOpen, ANY role or state: kReceiveOnly,
///     unconditionally. Nothing LowFive-level happens at this event at
///     all (this header's own "ROUND 2 REVIEW" comment has the full
///     account of why not, and of why a flush here specifically is what
///     deadlocked); the real receive is `H5Fopen` itself, which this
///     event merely precedes and does not need to assist.
///   - kImportClosed, first-server: kNoopAfterImport (the first-server
///     never defers, so there is never anything of its own to flush).
///   - kImportClosed, first-reader, nothing deferred: kNoopAfterImport
///     (an import with no outstanding export of its own -- not an
///     error; e.g. the very first event of a run on a graph laid out so
///     the first-reader's own first delivery stop is an import).
///   - kImportClosed, first-reader, a deferred serve pending:
///     kFlushDeferred -- THIS process has just completed its own
///     open-and-close of the peer's file (the action that, symmetrically,
///     is what releases a peer's own pending serve -- see INVARIANT), so
///     entering this process's own blocking serve now is safe: the peer
///     is guaranteed to reach ITS next import (an ordinary kReceiveOnly
///     is not blocking, so it moves straight on to its own open+close)
///     and release it in turn.
///   - kFinalize: refused (`E-LOWFIVE-ATTACH-SERVE-UNFLUSHED`) if a
///     first-reader still has a deferred serve with no `kImportClosed`
///     left to flush it -- the data was written and closed but never
///     served, and refusing loudly is the only acceptable outcome (never
///     a silent drop, never a hang nothing will ever resolve). Clean
///     (`kNoopFinalizeClean`) otherwise, including always for a
///     first-server (which never defers).
///
/// INVARIANT (deadlock-freedom), re-derived this round in terms of what
/// the PEER observes, not in terms of this process's own hook-call
/// order (the first attempt's mistake). I1 (progress) rests on the same
/// three pieces of evidence as before -- the host's own measured stack,
/// u_main.c's/v_main.f90's single-branch dispatch, and `include/fuse/
/// runtime/window_coupler.hpp`'s own documented Jacobi snapshot contract
/// -- establishing that on one process, an export is always eventually
/// followed by an import before that SAME process's own next export.
/// I2 (role exclusivity, this state machine's own contract above): the
/// first-server serves at every export and never defers; the
/// first-reader defers at every export and flushes only at its own next
/// `kImportClosed`. THE RELEASE CONDITION, read from LowFive's own
/// source (this header's own "ROUND 2 REVIEW" comment): a process
/// blocked inside `serve_all()` for file K is released exactly when the
/// PEER completes ITS OWN open-and-close of file K -- not by the peer
/// merely reaching an import, and not by anything this process itself
/// does. So: whenever the first-server blocks inside its own `kServeNow`
/// for file K (entered at its own export), I1 guarantees the first-
/// reader's own NEXT local event is an import, which performs an
/// ordinary, non-blocking open-and-close of file K (kImportAboutToOpen
/// does nothing to delay it) -- exactly the release condition for the
/// first-server's pending serve. Symmetrically, the first-reader's own
/// `kImportClosed` for that SAME import is what then flushes ITS own
/// deferred serve (entered only now, after its own open-and-close is
/// already done) -- and I1 guarantees the first-server's own NEXT local
/// event is ITS import, whose open-and-close is exactly the release
/// condition for the first-reader's newly-entered serve. Each side's own
/// single blocking serve call is therefore always matched by the OTHER
/// side's very next open-and-close, in that order -- never by the hook
/// that merely PRECEDES it -- and never left unconsumed except at
/// `kFinalize`, which this function refuses loudly rather than hanging
/// or dropping. This DEPENDS on I1 holding for BOTH components every
/// iteration, exactly as before; the double-defer refusal remains the
/// check that catches a violation instead of hanging silently. See
/// lowfive_attach_model_check.hpp for an explicit-state simulation that
/// checks this argument directly against the blocking semantics above,
/// rather than taking the argument's own word for it.
[[nodiscard]] inline fuse::Result<ServeAction> serve_state_step(ServeState* state,
                                                                 ServeEvent event) {
  switch (event) {
    case ServeEvent::kExportWrittenAndClosed:
      if (state->role == ServeRole::kFirstServer) return ServeAction::kServeNow;
      if (state->deferred_pending) {
        return fuse::Failure{
            "E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER",
            "a second export was written and closed while a FIRST deferred serve was "
            "still pending (no kImportClosed arrived between them) -- the write/read "
            "alternation this protocol depends on (I1) was violated"};
      }
      state->deferred_pending = true;
      return ServeAction::kDeferServe;
    case ServeEvent::kImportAboutToOpen:
      // Unconditional, any role or state: this header's own "ROUND 2
      // REVIEW" comment has the full account of why NOTHING is flushed
      // here, no matter what -- that is precisely the first round-2
      // attempt's own mistake.
      return ServeAction::kReceiveOnly;
    case ServeEvent::kImportClosed:
      if (state->role == ServeRole::kFirstServer) return ServeAction::kNoopAfterImport;
      if (state->deferred_pending) {
        state->deferred_pending = false;
        return ServeAction::kFlushDeferred;
      }
      return ServeAction::kNoopAfterImport;
    case ServeEvent::kFinalize:
      if (state->role == ServeRole::kFirstReader && state->deferred_pending) {
        return fuse::Failure{
            "E-LOWFIVE-ATTACH-SERVE-UNFLUSHED",
            "finalize reached with a deferred serve still pending -- the last export "
            "was written and closed but no matching kImportClosed ever arrived to "
            "flush it; refusing rather than dropping the data silently or leaving the "
            "peer to hang waiting for it"};
      }
      return ServeAction::kNoopFinalizeClean;
  }
  return fuse::Failure{"E-LOWFIVE-ATTACH-SERVE-BADEVENT", "unrecognised ServeEvent"};
}

}  // namespace fuse_examples::lowfive_attach
