#pragma once

// A pure, MPI-free and HDF5-free explicit-state
// simulation of the TWO processes in one LowFive-bound bidirectional
// exchange (one first-server, one first-reader; lowfive_attach_
// decision.hpp's own ServeRole/ServeEvent/ServeAction/ServeState/
// serve_state_step()), checked against the BLOCKING semantics read
// directly from LowFive's own source (lowfive_attach_decision.hpp's own
// "RECEIVE AND RELEASE" comment has the full citations):
//
//   - a process that enters a blocking serve for file K (ServeAction::
//     kServeNow or kFlushDeferred) does not proceed past that point
//     until the PEER completes its own open-AND-close of file K;
//   - a process's own open-and-close of the peer's file K blocks (the
//     open half) until the peer has entered a blocking serve for file K,
//     then immediately (non-blockingly, from this process's own view)
//     releases it.
//
// The blocking semantics above are confirmed against LowFive's source,
// with the mechanism cited. `DistMetadataVOL::file_close()`
// (LowFive/src/vol-dist-metadata.cpp)'s own RemoteFile branch does
// THREE things, not one: `f->obj.call<void>("file_close")` (an RPC to
// the producer), then a COLLECTIVE `MPI_Barrier(local)` across EVERY
// rank of the consumer's own local communicator, then -- ONLY on the
// consumer's own root rank -- `f->obj.self_->finish(0)`. The producer's
// own `Index::serve()` (LowFive/src/vol-dist/index.cpp) processes RPCs
// in a loop until `s.process(source)` has returned true (a `finish`
// was received) for EVERY one of its own registered intercomm servers
// (`all_done_count == servers.size()`), at which point it posts its own
// `ibarrier` and returns once that completes. For this row's own single
// intercomm, this reduces to exactly the two bullets above; the three-
// part breakdown matters only once more than one intercomm/consumer is
// in play, which this row is not. Nothing here changes which process
// blocks on what -- it only names the RPC/barrier/finish sequence
// underneath "close", which the two bullets above treat as a single
// step.
//
// WHAT THIS DOES AND DOES NOT ESTABLISH. This checks the PROTOCOL
// (lowfive_attach_decision.hpp's own state machine) against the
// blocking semantics AS READ from LowFive's source above -- it is NOT a
// measurement of LowFive itself, and it does not run MPI, HDF5, or any
// LowFive code. If the blocking semantics above are themselves wrong (a
// misreading of LowFive's source), this model proves nothing about the
// real row; only a host run of verification.brusselator.participants.
// lowfive.mpmd.3 measures that. What this DOES establish, assuming the
// semantics above are right: whether `serve_state_step()`'s own event
// wiring, as actually implemented (this file drives the REAL function,
// not a reimplementation of it, so a mutation to serve_state_step()
// itself is caught here too), can reach every process's own end of
// program without either process getting stuck.
//
// THE FIXED LOCAL PROGRAM. Each process's own local event sequence, for
// N iterations, is NOT searched for every possible interleaving WITHIN
// one process -- it is the one the measured diagnosis and
// lowfive_attach_decision.hpp's own INVARIANT comment fix:
// per iteration, write-closed, then import-about-to-open, then the
// open-and-close itself, then import-closed; never two writes before a
// read, matching I1. What IS explored is every possible INTERLEAVING of
// the two processes' own fixed local sequences against each other (an
// explicit-state search over the pair of program counters plus the
// shared "which file, if any, is currently open for serving" flags) --
// the state space is small (bounded by 4N positions per process) and
// explored exhaustively, not sampled.
//
// PURITY. The per-process decision at one step (serve now / defer /
// nothing / flush) is a function of that process's own (role, pending,
// pending_k, step kind, iteration) ALONE -- never of the peer's own
// position -- because `serve_state_step()` itself never reads anything
// about the peer. `Decide()` below is written as a PURE function with
// NO state captured across calls, taking pending/pending_k as explicit
// arguments and returning the updated values alongside the action, so
// that exploring the SAME (process, step) position from multiple
// reachable global states (differing only in the peer's own position --
// routine in an interleaving search) never double-counts a
// `serve_state_step()` call against hidden mutable history. An earlier
// version of this file wrapped a single, stateful `ServeState` shared
// across the whole search and called `serve_state_step()` live during
// the search -- WRONG, caught before this file was ever used in a test:
// the same real-world "write-closed at iteration 3" event would have
// been replayed against that one mutable state once per reachable
// global state that happens to reach it, not once.

#include "lowfive_attach_decision.hpp"

#include <array>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fuse_examples::lowfive_attach_model {

using fuse_examples::lowfive_attach::ServeAction;
using fuse_examples::lowfive_attach::ServeEvent;
using fuse_examples::lowfive_attach::ServeRole;
using fuse_examples::lowfive_attach::ServeState;
using fuse_examples::lowfive_attach::serve_state_step;

/// One process's local event kinds, in the SAME order every iteration
/// runs them. `kOpenClose` is pure LowFive mechanics (never consults a
/// policy at all -- see this header's own top comment); the other three
/// correspond exactly to `ServeEvent`'s own three per-read/write members
/// (`kFinalize` is handled separately, once, after the whole script).
enum class StepKind { kWriteClosed, kAboutToOpen, kOpenClose, kImportClosed,
                      kSyncBarrier };  // the collective sync U, V and the coupler share

/// What a process does at one step, translated from the real
/// `ServeAction` (or, for `kOpenClose`, unconditional). `kEnterBlocking`/
/// `kExitBlocking` are two HALVES of one `kServeNow`/`kFlushDeferred`
/// call: entering sets this process's own "I am serving file K" flag
/// (true even while still inside the blocking call); exiting is gated on
/// the PEER having cleared that flag by completing its own open-and-
/// close, and changes nothing else. Splitting the call this way is what
/// lets the search represent "still blocked inside serve_all()" as an
/// ordinary, re-triable program-counter position instead of a special
/// case in the search itself.
enum class MicroAction { kEnterBlocking, kExitBlocking, kOpenCloseStep, kPassThrough,
                        kSyncBarrierStep };

struct MicroStep {
  StepKind kind;
  int k;  // 1-based iteration index
};

inline std::vector<MicroStep> BuildScript(int n_iterations) {
  std::vector<MicroStep> steps;
  steps.reserve(static_cast<std::size_t>(n_iterations) * 4);
  for (int k = 1; k <= n_iterations; ++k) {
    steps.push_back({StepKind::kWriteClosed, k});
    steps.push_back({StepKind::kAboutToOpen, k});
    steps.push_back({StepKind::kOpenClose, k});
    steps.push_back({StepKind::kImportClosed, k});
  }
  return steps;
}

/// Which policy a process runs -- `kShipped` drives the REAL
/// `decide_serve_roles()`/`serve_state_step()` (so a mutation to either
/// changes this policy's own behaviour and is caught by a model-check
/// test, not only by the narrower unit tests in test/verification/
/// brusselator_lowfive_attach_test.cpp); the other two are rejected
/// designs, expressed directly (never through `serve_state_step()`,
/// since they are not what ships), kept only as illustrative
/// evidence that this search is discriminating -- it must flag these --
/// not as evidence about the shipped protocol, which `kShipped` alone
/// speaks for.
enum class PolicyKind {
  kShipped,              // decide_serve_action() decides (decide_serve_action()); flush at kImportClosed
  kOriginalNatural,      // serve at EVERY write, both roles, never defer
  kBothDefer,            // strawman: both roles defer at write, flush at kImportClosed
  kRejectedPreOpenFlush, // rejected design: flush at kAboutToOpen
  // A FALSIFIED model, hand-rolled exactly like the three strawmen above
  // (never through serve_state_step(), which calls decide_serve_action()
  // and so can no longer express this):
  // a first-reader defers EVERY write, unconditionally, flushing only
  // at its own next kImportClosed; a first-server never defers. This
  // is the regression test for the measured host hang, not a design
  // that ships -- see this file's own ModelCheck.
  // MeasuredOrderUnderRegressedPolicyReproducesTheMeasuredHang test.
  kRegressedAlwaysDeferFirstReader
};

struct ProcessPolicy {
  PolicyKind kind;
  ServeRole role;  // meaningful for kShipped and kRejectedPreOpenFlush only
  // When true, a kEnterBlocking action
  // never actually serves anything, modelling `DistMetadataVOL::
  // serve_all()` returning WITHOUT calling `index.serve()` because
  // `files.size() == 0` by the time it runs -- `MetadataVOL::file_
  // close()`'s own `if (!keep) { files.erase(f->name); delete f; }`
  // (lowfive_attach_decision.hpp's own "ROOT CAUSE" comment has the
  // full citation) erased and deleted the file, synchronously inside
  // H5Fclose(), before this process's own explicit serve call -- the
  // ONLY server this protocol ever gives a file -- ever runs a line.
  // Driven, in the tests that use this field, from `fuse_examples::
  // lowfive_attach::decide_vol_settings()`'s own `keep` field (`!keep`
  // => this flag true), not hand-set independently of it, so that a mutation
  // of the setting changes the verdict. Default
  // false (the shipped protocol's own tests, which assert completion,
  // must not accidentally rely on this).
  bool file_deleted_at_close = false;
  // Whether this process's own exported file is dropped
  // (lowfive_attach.cpp's own GuardedDropOwnExportedFile(), called
  // right after each ACTUAL serve) once it has been served. Default
  // true (the shipped protocol). False models the configuration with keep=true, but the drop step itself
  // never runs -- see WouldReuseStaleFileOnRecreate() below for why
  // this is NOT modelled as a blocking/stuck state at all (it is not
  // one): the search's own vocabulary covers whether processes get
  // stuck, and a never-dropped-but-still-served file does not get
  // anyone stuck, it gets the NEXT exchange's own write silently wrong.
  bool drop_after_serve = true;
};

struct DecideResult {
  MicroAction action;
  bool new_pending;
  int new_pending_k;
  // Meaningful only when action == kEnterBlocking: whether this call
  // actually enters a blocking serve the peer's own open-and-close can
  // release (true), or completes as a no-op that never serves anything
  // at all (false, ProcessPolicy::file_deleted_at_close's own case).
  // Always true for every other action.
  bool actually_serves = true;
  // Threaded the SAME way `pending`/`pending_k` are
  // (this header's own "PURITY" paragraph) -- decide_serve_action()'s
  // own "history" input. A fresh, throwaway ServeState built fresh
  // every Decide() call (as this function already does for
  // deferred_pending) would otherwise silently default this to false
  // on EVERY call, never actually recording that a read has completed
  // (scenario (i) would never complete past exchange 1 without it,
  // first-reader deferring at every write forever, never
  // just the cold start).
  bool new_completed_any_read;
};

/// PURE: the action this process takes at `kind`/`k`, given its own
/// policy and its own CURRENT `pending`/`pending_k`/`completed_any_read`
/// (passed in, not retained) -- see this header's own top "PURITY"
/// paragraph for why. `completed_any_read` is threaded the SAME way
/// `pending`/`pending_k` are.
inline DecideResult Decide(const ProcessPolicy& policy, StepKind kind, int k, bool pending,
                           int pending_k, bool completed_any_read) {
  if (policy.kind == PolicyKind::kShipped || policy.kind == PolicyKind::kRejectedPreOpenFlush) {
    // Reconstruct a throwaway ServeState from the explicit (role, pending,
    // completed_any_read) this call was given -- serve_state_step() is
    // itself a pure function of (ServeState, ServeEvent); giving it a
    // FRESH, explicitly-seeded state each call (rather than one mutable
    // object threaded live through the search) is what keeps THIS
    // function pure. completed_any_read MUST be seeded here too:
    // leaving it at
    // ServeState's own default (false) on every call would silently
    // make decide_serve_action() see "no read has ever completed" every
    // single time, defeating the very policy this header exists to
    // simulate.
    ServeState state;
    state.role = policy.role;
    state.deferred_pending = pending;
    state.completed_any_read = completed_any_read;

    const bool flush_here = (policy.kind == PolicyKind::kShipped) ? (kind == StepKind::kImportClosed)
                                                                   : (kind == StepKind::kAboutToOpen);
    ServeEvent event;
    bool call_it = true;
    switch (kind) {
      case StepKind::kWriteClosed:  event = ServeEvent::kExportWrittenAndClosed; break;
      case StepKind::kAboutToOpen:
        if (policy.kind == PolicyKind::kShipped) {
          event = ServeEvent::kImportAboutToOpen;  // always kReceiveOnly in the real function
        } else {
          // kRejectedPreOpenFlush: model the FIRST attempt's own mistake
          // directly (serve_state_step()'s OWN kImportAboutToOpen case
          // can no longer flush at all, so this
          // branch cannot be expressed by calling the real function --
          // it stands for "what the rejected design's hook body did",
          // not for a call into today's serve_state_step()).
          call_it = false;
        }
        break;
      case StepKind::kImportClosed:
        if (policy.kind == PolicyKind::kShipped) {
          event = ServeEvent::kImportClosed;
        } else {
          call_it = false;  // the rejected design never flushes here
        }
        break;
      case StepKind::kOpenClose:
        return {MicroAction::kOpenCloseStep, pending, pending_k, true, completed_any_read};
      case StepKind::kSyncBarrier:
        // Intercepted earlier, in TryStep(), for every REAL simulation
        // (never reaches here in practice) -- a defensive case only,
        // so this switch stays exhaustive over StepKind.
        return {MicroAction::kSyncBarrierStep, pending, pending_k, true, completed_any_read};
    }

    if (!call_it) {
      // kRejectedPreOpenFlush's own non-serve_state_step steps: flush
      // happens at kAboutToOpen instead (handled below via flush_here),
      // and kImportClosed does nothing. Still marks completed_any_read
      // at the import's own close, exactly as serve_state_step() does
      // for the kShipped path -- this rejected design completes an
      // import just as much as the shipped one does, even though it
      // flushes at a different point.
      const bool next_completed_any_read =
          completed_any_read || kind == StepKind::kImportClosed;
      if (flush_here && pending && policy.role == ServeRole::kFirstReader) {
        return {MicroAction::kEnterBlocking, false, pending_k, true, next_completed_any_read};
      }
      return {MicroAction::kPassThrough, pending, pending_k, true, next_completed_any_read};
    }

    auto result = serve_state_step(&state, event);
    if (!result.ok()) {
      throw std::runtime_error("serve_state_step() refused during a nominal write/read cycle "
                               "the model script never violates: [" +
                               result.error().code + "] " + result.error().message);
    }
    switch (result.value()) {
      case ServeAction::kServeNow:
        return {MicroAction::kEnterBlocking, state.deferred_pending, pending_k,
               !policy.file_deleted_at_close, state.completed_any_read};
      case ServeAction::kDeferServe:
        return {MicroAction::kPassThrough, state.deferred_pending, k, true,
               state.completed_any_read};
      case ServeAction::kReceiveOnly:
      case ServeAction::kNoopAfterImport:
        return {MicroAction::kPassThrough, state.deferred_pending, pending_k, true,
               state.completed_any_read};
      case ServeAction::kFlushDeferred:
        return {MicroAction::kEnterBlocking, state.deferred_pending, pending_k,
               !policy.file_deleted_at_close, state.completed_any_read};
      case ServeAction::kNoopFinalizeClean:
        return {MicroAction::kPassThrough, state.deferred_pending, pending_k, true,
               state.completed_any_read};  // not reached mid-script
    }
    return {MicroAction::kPassThrough, pending, pending_k, true, state.completed_any_read};
  }

  // kOriginalNatural / kBothDefer / kRegressedAlwaysDeferFirstReader:
  // hand-rolled, no serve_state_step() involvement at all (these
  // designs predate it, were never wired through it, or -- the
  // regressed one -- can no longer BE expressed by calling it, since
  // it now calls decide_serve_action() for the write decision).
  // None of these three ever CONSULT completed_any_read (they are
  // role-only or unconditional), but it is still threaded through
  // unchanged, marked true at kImportClosed, so a caller that mixes a
  // strawman policy into the event-sequence machinery still
  // gets a consistent value out, never a silently stale one.
  const bool next_completed_any_read = completed_any_read || kind == StepKind::kImportClosed;
  if (kind == StepKind::kOpenClose) {
    return {MicroAction::kOpenCloseStep, pending, pending_k, true, next_completed_any_read};
  }
  if (kind == StepKind::kWriteClosed) {
    if (policy.kind == PolicyKind::kOriginalNatural) {
      return {MicroAction::kEnterBlocking, pending, pending_k, true, next_completed_any_read};
    }
    if (policy.kind == PolicyKind::kRegressedAlwaysDeferFirstReader) {
      if (policy.role == ServeRole::kFirstServer) {
        return {MicroAction::kEnterBlocking, pending, pending_k, true, next_completed_any_read};
      }
      // first-reader: ALWAYS defers -- the falsified rule.
      return {MicroAction::kPassThrough, true, k, true, next_completed_any_read};
    }
    return {MicroAction::kPassThrough, true, k, true, next_completed_any_read};  // kBothDefer: always defer
  }
  if (kind == StepKind::kImportClosed) {
    if ((policy.kind == PolicyKind::kBothDefer ||
         policy.kind == PolicyKind::kRegressedAlwaysDeferFirstReader) &&
        pending) {
      return {MicroAction::kEnterBlocking, false, pending_k, true, next_completed_any_read};
    }
    return {MicroAction::kPassThrough, pending, pending_k, true, next_completed_any_read};
  }
  // kAboutToOpen / kSyncBarrier: no-op for every strawman.
  return {MicroAction::kPassThrough, pending, pending_k, true, next_completed_any_read};
}

/// One process's own mutable position in the search. `std::set`'s own
/// default ordering on `GlobalState` below needs `operator<`; every
/// field here is part of what must match for two states to be "the
/// same" (the memoization `visited` set relies on).
struct ProcPos {
  int pc = 0;
  bool entering = false;
  bool own_serving = false;
  int serving_k = -1;
  bool pending = false;
  int pending_k = -1;
  // decide_serve_action()'s own "history" input, threaded the
  // SAME way pending/pending_k are (this header's own
  // "PURITY" paragraph) -- see DecideResult::new_completed_any_read's
  // own comment. False initially (no read completed yet).
  bool completed_any_read = false;
  // How many sync barriers (StepKind::kSyncBarrier) this
  // process has gotten PAST so far -- monotonic, 0 initially. Always 0
  // for scripts that contain no kSyncBarrier step.
  int sync_passed = 0;
  // Has this process ARRIVED at (but not yet passed) barrier
  // index `sync_passed`? Mirrors `entering`'s own two-phase pattern for
  // serve blocking -- needed because `sync_passed >= sync_passed` is
  // vacuously true for two processes that have BOTH merely never
  // started (both at 0), which would let the FIRST barrier release
  // itself with nobody actually there; checking `other.sync_entering &&
  // other.sync_passed == mine` instead requires an OBSERVED arrival,
  // not just a count that has not yet fallen behind.
  bool sync_entering = false;

  auto Tied() const {
    return std::tie(pc, entering, own_serving, serving_k, pending, pending_k, completed_any_read,
                    sync_passed, sync_entering);
  }
  bool operator<(const ProcPos& o) const { return Tied() < o.Tied(); }
  bool operator==(const ProcPos& o) const { return Tied() == o.Tied(); }
};

struct GlobalState {
  ProcPos a, b;
  // The coupler, a THIRD actor that only ever runs
  // StepKind::kSyncBarrier steps (a third process that only
  // participates in the sync) -- default-constructed (all-zero) and
  // never touched by the two-process RunSimulation() search.
  ProcPos c;
  bool operator<(const GlobalState& o) const {
    if (!(a == o.a)) return a < o.a;
    if (!(b == o.b)) return b < o.b;
    return c < o.c;
  }
};

struct SimulationResult {
  bool completed = false;
  int num_visited_states = 0;
  std::optional<std::string> stuck_example;
};

namespace detail {

struct StepOutcome {
  ProcPos new_pos;
  std::optional<bool> peer_serving_update;
  std::optional<int> peer_serving_k_update;
};

/// Attempts one process's own next step. `peer_own_serving`/
/// `peer_serving_k` are the OTHER process's own current flags (read-only
/// here); `policy` and `pos` belong to THIS process. Returns
/// `std::nullopt` if blocked right now.
inline std::optional<StepOutcome> TryStep(const std::vector<MicroStep>& script, const ProcPos& pos,
                                          bool peer_own_serving, int peer_serving_k,
                                          const ProcessPolicy& policy,
                                          // The OTHER two processes' own sync-arrival state,
                                          // for the kSyncBarrier branch below -- defaulted so every
                                          // two-process call site (never a
                                          // kSyncBarrier step) is unaffected.
                                          bool other1_sync_entering = false,
                                          int other1_sync_passed = 0,
                                          bool other2_sync_entering = false,
                                          int other2_sync_passed = 0) {
  if (pos.pc >= static_cast<int>(script.size())) return std::nullopt;  // already done
  const MicroStep& step = script[static_cast<std::size_t>(pos.pc)];

  if (step.kind == StepKind::kOpenClose) {
    if (peer_own_serving && peer_serving_k == step.k) {
      ProcPos next = pos;
      next.pc += 1;
      return StepOutcome{next, false, std::nullopt};
    }
    return std::nullopt;
  }

  if (step.kind == StepKind::kSyncBarrier) {
    // A symmetric N-of-N rendezvous (a collective
    // sync event shared by U, V and the coupler), two phases like the
    // serve/open-close pair above but symmetric (every one of the
    // THREE actors both "enters" and is released by the SAME kind of
    // step, unlike serve's asymmetric enter-vs-open-close split).
    // Phase 1 (`!pos.sync_entering`): record arrival at this barrier,
    // visible to the other two via their own read of THIS ProcPos --
    // does not advance `pc` yet.
    if (!pos.sync_entering) {
      ProcPos next = pos;
      next.sync_entering = true;
      return StepOutcome{next, std::nullopt, std::nullopt};
    }
    // Phase 2: released once BOTH others have "reached" this SAME
    // barrier index -- either they are currently ARRIVED there
    // (`sync_entering` at the identical `sync_passed`), OR they have
    // ALREADY PASSED it (`sync_passed` strictly greater -- monotonic,
    // so having passed index i implies having reached it earlier).
    // The second half is NOT optional: an earlier version of this
    // check required `other.sync_passed == pos.sync_passed` ONLY,
    // which works for the FIRST process to check, but then the ONE
    // process that releases FIRST clears its own `sync_entering` and
    // bumps `sync_passed` past the other two's own still-current
    // index -- their own NEXT check (`other_passed == mine`) would
    // then see the released peer as "not arrived" (false) instead of
    // "already gone through," permanently stranding whichever two
    // check second and third. Caught before any test used it (every
    // n_exchanges >= 2 scenario got stuck exactly at the first sync).
    const auto reached = [&](bool other_entering, int other_passed) {
      return other_passed > pos.sync_passed || (other_passed == pos.sync_passed && other_entering);
    };
    if (reached(other1_sync_entering, other1_sync_passed) &&
        reached(other2_sync_entering, other2_sync_passed)) {
      ProcPos next = pos;
      next.sync_entering = false;
      next.sync_passed += 1;
      next.pc += 1;
      return StepOutcome{next, std::nullopt, std::nullopt};
    }
    return std::nullopt;
  }

  if (pos.entering) {
    // Inside a blocking serve entered at THIS same pc; exit once the
    // peer's own kOpenClose has cleared `own_serving` (that update is
    // applied to `pos` directly by the search loop below when it
    // happens -- see RunSimulation).
    if (!pos.own_serving) {
      ProcPos next = pos;
      next.entering = false;
      next.pc += 1;
      return StepOutcome{next, std::nullopt, std::nullopt};
    }
    return std::nullopt;
  }

  // Not yet entering: ask the PURE decision function exactly once,
  // using THIS process's own explicit pending/pending_k/
  // completed_any_read -- never the peer's state, which is irrelevant
  // to the decision (this header's own "PURITY" paragraph).
  const DecideResult d =
      Decide(policy, step.kind, step.k, pos.pending, pos.pending_k, pos.completed_any_read);

  if (d.action == MicroAction::kEnterBlocking && !d.actually_serves) {
    // ProcessPolicy::file_deleted_at_close:
    // `serve_all()` returns WITHOUT ever calling `index.serve()` when
    // nothing is indexed -- a complete no-op from BOTH sides' own point
    // of view. This process does NOT block (advances immediately,
    // `entering` never becomes true) and does NOT tell the peer anything
    // is being served (no peer_serving_update at all) -- so the peer's
    // own NEXT OPEN_CLOSE attempt for this exact file will find
    // `peer_own_serving` false forever and simply never become enabled:
    // a genuine stuck state, not a silent pass, exactly as asked.
    ProcPos next = pos;
    next.pc += 1;
    next.pending = d.new_pending;
    next.pending_k = d.new_pending_k;
    next.completed_any_read = d.new_completed_any_read;
    return StepOutcome{next, std::nullopt, std::nullopt};
  }
  if (d.action == MicroAction::kEnterBlocking) {
    // Sets THIS process's OWN serving flag only (captured in `next`,
    // which becomes this process's own entry in the global state) --
    // never the peer's. An earlier version of this function returned
    // `{next, true, step.k}` here, by analogy with OPEN_CLOSE's own
    // peer-clearing update below -- WRONG, caught by this file's own
    // model-check tests failing to complete even the SHIPPED protocol:
    // it told the search the PEER was now serving file K too, which is
    // simply false (only the peer's OWN OPEN_CLOSE step, consuming THIS
    // process's serving flag, ever changes the peer's state at all).
    ProcPos next = pos;
    next.entering = true;
    next.own_serving = true;
    next.serving_k = step.k;
    next.pending = d.new_pending;
    next.pending_k = d.new_pending_k;
    next.completed_any_read = d.new_completed_any_read;
    return StepOutcome{next, std::nullopt, std::nullopt};
  }
  // kPassThrough (kOpenCloseStep unreachable here; handled above).
  ProcPos next = pos;
  next.pc += 1;
  next.pending = d.new_pending;
  next.completed_any_read = d.new_completed_any_read;
  next.pending_k = d.new_pending_k;
  return StepOutcome{next, std::nullopt, std::nullopt};
}

}  // namespace detail

/// Explores every interleaving of processes A and B running the SAME
/// N-iteration script, under `policy_a`/`policy_b`, to exhaustion (DFS
/// with a visited set; the state space for N up to 60 is a few thousand
/// states total, per this header's own top comment).
inline SimulationResult RunSimulation(int n_iterations, ProcessPolicy policy_a,
                                      ProcessPolicy policy_b) {
  const std::vector<MicroStep> script = BuildScript(n_iterations);

  GlobalState init{};
  std::set<GlobalState> visited;
  std::vector<GlobalState> stack;
  visited.insert(init);
  stack.push_back(init);

  std::optional<std::string> stuck_example;

  while (!stack.empty()) {
    GlobalState s = stack.back();
    stack.pop_back();

    const bool a_done = s.a.pc >= static_cast<int>(script.size());
    const bool b_done = s.b.pc >= static_cast<int>(script.size());
    if (a_done && b_done) continue;  // terminal, nothing more to explore from here

    bool moved = false;

    if (auto out = detail::TryStep(script, s.a, s.b.own_serving, s.b.serving_k, policy_a)) {
      moved = true;
      GlobalState ns = s;
      ns.a = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.b.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.b.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }
    if (auto out = detail::TryStep(script, s.b, s.a.own_serving, s.a.serving_k, policy_b)) {
      moved = true;
      GlobalState ns = s;
      ns.b = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.a.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.a.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }

    if (!moved && !stuck_example.has_value()) {
      stuck_example = "A: pc=" + std::to_string(s.a.pc) + " entering=" + std::to_string(s.a.entering) +
                      " serving=" + std::to_string(s.a.own_serving) + "(" + std::to_string(s.a.serving_k) +
                      ") pending=" + std::to_string(s.a.pending) + "(" + std::to_string(s.a.pending_k) +
                      ")  B: pc=" + std::to_string(s.b.pc) + " entering=" + std::to_string(s.b.entering) +
                      " serving=" + std::to_string(s.b.own_serving) + "(" + std::to_string(s.b.serving_k) +
                      ") pending=" + std::to_string(s.b.pending) + "(" + std::to_string(s.b.pending_k) + ")";
    }
  }

  SimulationResult result;
  result.num_visited_states = static_cast<int>(visited.size());
  result.stuck_example = stuck_example;
  result.completed = !stuck_example.has_value();
  return result;
}


// ---------------------------------------------------------------------
// The per-component event sequences are DATA (a
// list of `EventKind`), not RunSimulation()'s own hard-coded
// write-then-read pair -- the host's own measured steady-state orders
// (lowfive_attach_decision.hpp's own "THE PER-EXCHANGE ORDER DIFFERS"
// comment has the derived table and citations) differ between the two roles, which a single
// shared `BuildScript()` can never express. A third actor, the
// coupler, participates in a `kSyncBarrier` event neither physics
// process's own two-process script has any notion of.

/// One top-level event the catalogue of orders is built
/// from -- `kSync` is the collective the coupler also participates in.
enum class EventKind { kWrite, kRead, kSync };

/// The order catalogue: `kMeasured` is the ONE order any real
/// graph can produce under the current channel-order mechanism
/// (lowfive_attach_decision.hpp's own "THE PER-EXCHANGE ORDER DIFFERS"
/// comment); the other two
/// are strawmen with NO corresponding real graph, kept only to show the
/// model discriminates them -- the identical
/// relationship the existing kOriginalNatural/kBothDefer strawmen above
/// already have to the real protocol.
enum class ExchangeOrder { kMeasured, kBothWriteReadSteady, kBothReadWriteSteady };

/// PURE: this role's own STEADY-state (exchange >= 1) [first, second]
/// shape for `order`. `kMeasured`'s own shape matches
/// `lowfive_attach::ExpectedOrderForExchange(role, 1)` exactly (not
/// re-derived independently of it: both read the SAME measured table).
/// The unit here is an "exchange": ONE write-then-read (or
/// read-then-write) pair, not a coupling window (the host measured ~61
/// of these per window -- lowfive_attach_decision.hpp's own "EXCHANGE vs.
/// PERIOD" paragraph has the full account).
[[nodiscard]] inline std::array<EventKind, 2> SteadyShapeForOrder(ExchangeOrder order,
                                                                   ServeRole role) {
  switch (order) {
    case ExchangeOrder::kMeasured:
      return role == ServeRole::kFirstServer
                 ? std::array<EventKind, 2>{EventKind::kWrite, EventKind::kRead}
                 : std::array<EventKind, 2>{EventKind::kRead, EventKind::kWrite};
    case ExchangeOrder::kBothWriteReadSteady:
      return {EventKind::kWrite, EventKind::kRead};
    case ExchangeOrder::kBothReadWriteSteady:
      return {EventKind::kRead, EventKind::kWrite};
  }
  return {EventKind::kWrite, EventKind::kRead};
}

/// PURE: this role's own full per-process event list for `n_exchanges`
/// exchanges. Exchange 0 is the cold start -- write-then-read on
/// EITHER role, unconditionally, with NO sync (u_main.c's/v_main.f90's
/// own pre-loop priming call, a separate mechanism from the
/// IR-scheduled steady state -- lowfive_attach_decision.hpp's own
/// "THE PER-EXCHANGE ORDER DIFFERS" comment has the full account of why). Exchanges
/// 1..n_exchanges-1 are steady state, from `SteadyShapeForOrder()`,
/// each followed by one `kSync`. `n_exchanges <= 0` returns an empty
/// sequence (nothing to simulate).
[[nodiscard]] inline std::vector<EventKind> BuildEventSequence(ExchangeOrder order, ServeRole role,
                                                                int n_exchanges) {
  std::vector<EventKind> out;
  if (n_exchanges <= 0) return out;
  out.push_back(EventKind::kWrite);
  out.push_back(EventKind::kRead);
  for (int p = 1; p < n_exchanges; ++p) {
    const std::array<EventKind, 2> shape = SteadyShapeForOrder(order, role);
    out.push_back(shape[0]);
    out.push_back(shape[1]);
    out.push_back(EventKind::kSync);
  }
  return out;
}

/// PURE: expands one process's own `BuildEventSequence()` output into
/// `MicroStep`s, labelling every step in an exchange with that
/// exchange's own 0-based index -- exchange 0 (cold start) is exactly
/// 2 events (write, read), no sync; every later exchange is exactly 3
/// (the steady shape's own two events, then sync) -- so this function
/// can walk the list two-or-three-at-a-time rather than inferring
/// exchange boundaries from counting, which is
/// ambiguous at the cold-start/exchange-1 boundary when exchanges are
/// counted purely by "two write/read events seen" without knowing in
/// advance whether a sync follows.
[[nodiscard]] inline std::vector<MicroStep> BuildScriptFromEvents(
    const std::vector<EventKind>& events) {
  std::vector<MicroStep> steps;
  int exchange = 0;
  std::size_t i = 0;
  const auto push_one = [&](EventKind e) {
    switch (e) {
      case EventKind::kWrite:
        steps.push_back({StepKind::kWriteClosed, exchange});
        break;
      case EventKind::kRead:
        steps.push_back({StepKind::kAboutToOpen, exchange});
        steps.push_back({StepKind::kOpenClose, exchange});
        steps.push_back({StepKind::kImportClosed, exchange});
        break;
      case EventKind::kSync:
        steps.push_back({StepKind::kSyncBarrier, exchange});
        break;
    }
  };
  while (i < events.size()) {
    const bool has_sync = (i + 2 < events.size()) && events[i + 2] == EventKind::kSync;
    push_one(events[i]);
    push_one(events[i + 1]);
    i += 2;
    if (has_sync) {
      push_one(events[i]);
      i += 1;
    }
    exchange += 1;
  }
  return steps;
}

/// PURE: the coupler's own script -- one `kSyncBarrier` per STEADY
/// exchange (1..n_exchanges-1), labelled identically to the physics
/// processes' own steady syncs (the cold start, exchange 0, never
/// syncs on any process, including the coupler).
[[nodiscard]] inline std::vector<MicroStep> BuildCouplerScript(int n_exchanges) {
  std::vector<MicroStep> steps;
  for (int exchange = 1; exchange < n_exchanges; ++exchange) {
    steps.push_back({StepKind::kSyncBarrier, exchange});
  }
  return steps;
}

/// Explores every interleaving of THREE actors -- U (`policy_a`), V
/// (`policy_b`), and the coupler (sync only, no policy) -- running
/// `order`'s own `n_exchanges`-exchange scripts, to exhaustion (same
/// DFS + visited-set discipline as `RunSimulation()` above, now over a
/// 3-field `GlobalState`). `policy_a.role`/`policy_b.role` select each
/// one's own event sequence via `BuildEventSequence()`.
inline SimulationResult RunSimulationForOrder(ExchangeOrder order, int n_exchanges,
                                            ProcessPolicy policy_a, ProcessPolicy policy_b) {
  const std::vector<MicroStep> script_a =
      BuildScriptFromEvents(BuildEventSequence(order, policy_a.role, n_exchanges));
  const std::vector<MicroStep> script_b =
      BuildScriptFromEvents(BuildEventSequence(order, policy_b.role, n_exchanges));
  const std::vector<MicroStep> script_c = BuildCouplerScript(n_exchanges);
  const ProcessPolicy policy_c{PolicyKind::kShipped, ServeRole::kFirstServer};  // never consulted

  GlobalState init{};
  std::set<GlobalState> visited;
  std::vector<GlobalState> stack;
  visited.insert(init);
  stack.push_back(init);

  std::optional<std::string> stuck_example;

  while (!stack.empty()) {
    GlobalState s = stack.back();
    stack.pop_back();

    const bool a_done = s.a.pc >= static_cast<int>(script_a.size());
    const bool b_done = s.b.pc >= static_cast<int>(script_b.size());
    const bool c_done = s.c.pc >= static_cast<int>(script_c.size());
    if (a_done && b_done && c_done) continue;  // terminal, nothing more to explore from here

    bool moved = false;

    if (auto out = detail::TryStep(script_a, s.a, s.b.own_serving, s.b.serving_k, policy_a,
                                   s.b.sync_entering, s.b.sync_passed, s.c.sync_entering,
                                   s.c.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.a = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.b.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.b.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }
    if (auto out = detail::TryStep(script_b, s.b, s.a.own_serving, s.a.serving_k, policy_b,
                                   s.a.sync_entering, s.a.sync_passed, s.c.sync_entering,
                                   s.c.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.b = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.a.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.a.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }
    // The coupler never serves/writes/reads (its own script is pure
    // kSyncBarrier) -- `peer_own_serving`/`peer_serving_k` are passed
    // as harmless constants (false/-1) since its own script never
    // reaches the kOpenClose branch that would consult them.
    if (auto out = detail::TryStep(script_c, s.c, false, -1, policy_c, s.a.sync_entering,
                                   s.a.sync_passed, s.b.sync_entering, s.b.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.c = out->new_pos;
      if (visited.insert(ns).second) stack.push_back(ns);
    }

    if (!moved && !stuck_example.has_value()) {
      stuck_example = "A: pc=" + std::to_string(s.a.pc) + " entering=" + std::to_string(s.a.entering) +
                      " serving=" + std::to_string(s.a.own_serving) + "(" + std::to_string(s.a.serving_k) +
                      ") pending=" + std::to_string(s.a.pending) + "(" + std::to_string(s.a.pending_k) +
                      ") sync=" + std::to_string(s.a.sync_passed) +
                      "  B: pc=" + std::to_string(s.b.pc) + " entering=" + std::to_string(s.b.entering) +
                      " serving=" + std::to_string(s.b.own_serving) + "(" + std::to_string(s.b.serving_k) +
                      ") pending=" + std::to_string(s.b.pending) + "(" + std::to_string(s.b.pending_k) +
                      ") sync=" + std::to_string(s.b.sync_passed) +
                      "  Coupler: pc=" + std::to_string(s.c.pc) +
                      " sync=" + std::to_string(s.c.sync_passed);
    }
  }

  SimulationResult result;
  result.num_visited_states = static_cast<int>(visited.size());
  result.stuck_example = stuck_example;
  result.completed = !stuck_example.has_value();
  return result;
}

// ---------------------------------------------------------------------
// The accept layout is a
// scenario: U physics, a placeholder rank that does no LowFive calls
// but takes part in every base-communicator collective, V. Read
// against RunSimulationForOrder() just above: its own third actor
// ("the coupler", `script_c`, pure kSyncBarrier, no write/read/serve of
// its own) is ALREADY, at the level this model operates on, the
// IDENTICAL role an accept-mode bundled coupler/placeholder rank plays
// -- the model has no notion of which executable hosts an actor, only
// of each actor's own event sequence, and "pure sync, never a channel
// event" is the SAME sequence whether that actor is coupler_main.c's
// own separate MPMD process or u_main.c's own extra, bundled rank. So
// RunSimulationForOrder() already IS the accept-layout scenario, under a
// reading that is made EXPLICIT (not silently assumed) by the test
// below that exercises it under that name, rather than inventing a
// fourth, duplicate actor type with nothing to distinguish it.
//
// What this model CANNOT distinguish, stated rather than elided: the
// REAL bridge communicator's own rank-numbering difference between mpmd
// and accept (fixed in lowfive_attach_
// decision.hpp's own decide_attach_mode()/this_rank handling, tier-0
// tested directly in test/verification/brusselator_lowfive_attach_
// test.cpp's own WrongThisRankFromLocalWorldInsteadOfBridgeMisclassifiesV)
// -- that is a RankSet-arithmetic property, checked separately, not an
// event-ordering one this file's own search explores.
//
// The known-bad placement: a THIRD actor that
// EXISTS in the real row (the placeholder/coupler rank is really
// there) but never calls bruss_lowfive_join_local_split() at all -- the
// exact bug the join call prevents. Modelled here as
// `third_actor_joins = false`: U's and V's own scripts are UNCHANGED
// (still try to pass every kSyncBarrier, because the real collective
// IS entered on their side), but the third actor's own script is
// forced EMPTY regardless of `n_exchanges` -- it never shows up to any
// barrier at all, the mirror image of "a collective entered without
// every rank's own participation": U and V get stuck waiting for a
// rank that never joins.
inline SimulationResult RunSimulationAcceptLayout(ExchangeOrder order, int n_exchanges,
                                                  ProcessPolicy policy_a, ProcessPolicy policy_b,
                                                  bool third_actor_joins) {
  const std::vector<MicroStep> script_a =
      BuildScriptFromEvents(BuildEventSequence(order, policy_a.role, n_exchanges));
  const std::vector<MicroStep> script_b =
      BuildScriptFromEvents(BuildEventSequence(order, policy_b.role, n_exchanges));
  const std::vector<MicroStep> script_c =
      third_actor_joins ? BuildCouplerScript(n_exchanges) : std::vector<MicroStep>{};
  const ProcessPolicy policy_c{PolicyKind::kShipped, ServeRole::kFirstServer};  // never consulted

  GlobalState init{};
  std::set<GlobalState> visited;
  std::vector<GlobalState> stack;
  visited.insert(init);
  stack.push_back(init);

  std::optional<std::string> stuck_example;

  while (!stack.empty()) {
    GlobalState s = stack.back();
    stack.pop_back();

    const bool a_done = s.a.pc >= static_cast<int>(script_a.size());
    const bool b_done = s.b.pc >= static_cast<int>(script_b.size());
    const bool c_done = s.c.pc >= static_cast<int>(script_c.size());
    if (a_done && b_done && c_done) continue;  // terminal, nothing more to explore from here

    bool moved = false;

    if (auto out = detail::TryStep(script_a, s.a, s.b.own_serving, s.b.serving_k, policy_a,
                                   s.b.sync_entering, s.b.sync_passed, s.c.sync_entering,
                                   s.c.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.a = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.b.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.b.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }
    if (auto out = detail::TryStep(script_b, s.b, s.a.own_serving, s.a.serving_k, policy_b,
                                   s.a.sync_entering, s.a.sync_passed, s.c.sync_entering,
                                   s.c.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.b = out->new_pos;
      if (out->peer_serving_update.has_value()) ns.a.own_serving = *out->peer_serving_update;
      if (out->peer_serving_k_update.has_value()) ns.a.serving_k = *out->peer_serving_k_update;
      if (visited.insert(ns).second) stack.push_back(ns);
    }
    if (auto out = detail::TryStep(script_c, s.c, false, -1, policy_c, s.a.sync_entering,
                                   s.a.sync_passed, s.b.sync_entering, s.b.sync_passed)) {
      moved = true;
      GlobalState ns = s;
      ns.c = out->new_pos;
      if (visited.insert(ns).second) stack.push_back(ns);
    }

    if (!moved && !stuck_example.has_value()) {
      stuck_example = "A: pc=" + std::to_string(s.a.pc) + " sync=" + std::to_string(s.a.sync_passed) +
                      "  B: pc=" + std::to_string(s.b.pc) + " sync=" + std::to_string(s.b.sync_passed) +
                      "  Third: pc=" + std::to_string(s.c.pc) + " sync=" + std::to_string(s.c.sync_passed) +
                      " joins=" + std::to_string(third_actor_joins ? 1 : 0);
    }
  }

  SimulationResult result;
  result.num_visited_states = static_cast<int>(visited.size());
  result.stuck_example = stuck_example;
  result.completed = !stuck_example.has_value();
  return result;
}

/// PURE (the "soundness link"): would the STATIC check
/// (lowfive_attach_decision.hpp's own `CheckGraphOrderValidated()`)
/// accept a graph exhibiting `order`? Only `kMeasured` has a real graph
/// behind it at all (the channel-order mechanism can only ever produce
/// "one side exports-then-imports, the other imports-then-exports" --
/// lowfive_attach_decision.hpp's own "THE PER-EXCHANGE ORDER DIFFERS"
/// comment) -- the other
/// two catalogue entries are strawmen with no corresponding graph, so
/// there is no signature for the static check to ever be asked about;
/// defined false for them so the soundness assertion below
/// (accepts-implies-completes) is checked against ALL three catalogue
/// entries uniformly, not only the one real graph.
[[nodiscard]] inline bool static_check_accepts(ExchangeOrder order) {
  return order == ExchangeOrder::kMeasured;
}

/// PURE (the "soundness link"): does the SHIPPED policy (both
/// roles, `decide_serve_action()`-driven) complete `order`'s own
/// `n_exchanges`-exchange script without either process getting stuck?
[[nodiscard]] inline bool model_completes(ExchangeOrder order, int n_exchanges) {
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader};
  return RunSimulationForOrder(order, n_exchanges, server, reader).completed;
}

/// PURE, no search needed (unlike RunSimulation()'s own interleaving
/// exploration, this is a STATIC property of a configuration, not of
/// any particular interleaving): does `policy` reuse a STALE file on
/// this process's own SECOND (or later) write of it? lowfive_attach_
/// decision.hpp's own "ROOT CAUSE" comment has the full citation on
/// `MetadataVOL::file_create()`: `files.emplace(name_, f)` is a
/// `std::map::emplace()` call, a NO-OP on a duplicate key -- if the
/// file is KEPT resident from a PREVIOUS write that was never dropped,
/// the OLD File* (and its own stale children) stays exactly where it
/// is, and the fresh write goes into an orphaned object `serve_all()`
/// never finds. `!policy.file_deleted_at_close` is this policy's own
/// `keep` -- the two are the SAME fact read from opposite ends (this
/// header's own ProcessPolicy::file_deleted_at_close doc comment: it is
/// driven from `!decide_vol_settings().keep`), so this function reads
/// it that way rather than adding a third, redundant field that could
/// silently disagree with the first. This happens on EVERY write after
/// the first, regardless of which rank serves when or in what order --
/// hence no interleaving search: `!file_deleted_at_close &&
/// !drop_after_serve && n_iterations > 1` is both necessary and
/// sufficient, true for any policy whose OWN write actually keeps a
/// file (both roles write once per iteration in the fixed
/// script, so this applies to either). Returns false for a single
/// iteration (nothing to recreate yet), whenever a drop actually
/// happens, or whenever the file was never kept in the first place
/// (file_deleted_at_close's own case -- nothing to reuse if nothing
/// survived to be reused).
[[nodiscard]] inline bool WouldReuseStaleFileOnRecreate(const ProcessPolicy& policy,
                                                        int n_iterations) {
  return !policy.file_deleted_at_close && !policy.drop_after_serve && n_iterations > 1;
}

}  // namespace fuse_examples::lowfive_attach_model
