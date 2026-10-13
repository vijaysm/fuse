#pragma once

// S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track,
// ROUND 2 REVIEW: a pure, MPI-free and HDF5-free explicit-state
// simulation of the TWO processes in one LowFive-bound bidirectional
// exchange (one first-server, one first-reader; lowfive_attach_
// decision.hpp's own ServeRole/ServeEvent/ServeAction/ServeState/
// serve_state_step()), checked against the BLOCKING semantics read
// directly from LowFive's own source (lowfive_attach_decision.hpp's own
// "ROUND 2 REVIEW" comment has the full citations):
//
//   - a process that enters a blocking serve for file K (ServeAction::
//     kServeNow or kFlushDeferred) does not proceed past that point
//     until the PEER completes its own open-AND-close of file K;
//   - a process's own open-and-close of the peer's file K blocks (the
//     open half) until the peer has entered a blocking serve for file K,
//     then immediately (non-blockingly, from this process's own view)
//     releases it.
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
// one process -- it is the one this track's own measured diagnosis and
// lowfive_attach_decision.hpp's own INVARIANT comment already fixed:
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
enum class StepKind { kWriteClosed, kAboutToOpen, kOpenClose, kImportClosed };

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
enum class MicroAction { kEnterBlocking, kExitBlocking, kOpenCloseStep, kPassThrough };

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
/// since they are not what this track ships), kept only as illustrative
/// evidence that this search is discriminating -- it must flag these --
/// not as evidence about the shipped protocol, which `kShipped` alone
/// speaks for.
enum class PolicyKind {
  kShipped,              // ServeRole decides; flush at kImportClosed (the fix)
  kOriginalNatural,      // round 1: serve at EVERY write, both roles, never defer
  kBothDefer,            // strawman: both roles defer at write, flush at kImportClosed
  kRejectedPreOpenFlush  // the FIRST round-2 attempt this review rejected: flush at kAboutToOpen
};

struct ProcessPolicy {
  PolicyKind kind;
  ServeRole role;  // meaningful for kShipped and kRejectedPreOpenFlush only
};

struct DecideResult {
  MicroAction action;
  bool new_pending;
  int new_pending_k;
};

/// PURE: the action this process takes at `kind`/`k`, given its own
/// policy and its own CURRENT `pending`/`pending_k` (passed in, not
/// retained) -- see this header's own top "PURITY" paragraph for why.
inline DecideResult Decide(const ProcessPolicy& policy, StepKind kind, int k, bool pending,
                           int pending_k) {
  if (policy.kind == PolicyKind::kShipped || policy.kind == PolicyKind::kRejectedPreOpenFlush) {
    // Reconstruct a throwaway ServeState from the explicit (role, pending)
    // this call was given -- serve_state_step() is itself a pure function
    // of (ServeState, ServeEvent); giving it a FRESH, explicitly-seeded
    // state each call (rather than one mutable object threaded live
    // through the search) is what keeps THIS function pure.
    ServeState state;
    state.role = policy.role;
    state.deferred_pending = pending;

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
          // can no longer flush at all, by this round's own fix, so this
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
        return {MicroAction::kOpenCloseStep, pending, pending_k};
    }

    if (!call_it) {
      // kRejectedPreOpenFlush's own non-serve_state_step steps: flush
      // happens at kAboutToOpen instead (handled below via flush_here),
      // and kImportClosed does nothing.
      if (flush_here && pending && policy.role == ServeRole::kFirstReader) {
        return {MicroAction::kEnterBlocking, false, pending_k};
      }
      return {MicroAction::kPassThrough, pending, pending_k};
    }

    auto result = serve_state_step(&state, event);
    if (!result.ok()) {
      throw std::runtime_error("serve_state_step() refused during a nominal write/read cycle "
                               "the model script never violates: [" +
                               result.error().code + "] " + result.error().message);
    }
    switch (result.value()) {
      case ServeAction::kServeNow:
        return {MicroAction::kEnterBlocking, state.deferred_pending, pending_k};
      case ServeAction::kDeferServe:
        return {MicroAction::kPassThrough, state.deferred_pending, k};
      case ServeAction::kReceiveOnly:
      case ServeAction::kNoopAfterImport:
        return {MicroAction::kPassThrough, state.deferred_pending, pending_k};
      case ServeAction::kFlushDeferred:
        return {MicroAction::kEnterBlocking, state.deferred_pending, pending_k};
      case ServeAction::kNoopFinalizeClean:
        return {MicroAction::kPassThrough, state.deferred_pending, pending_k};  // not reached mid-script
    }
    return {MicroAction::kPassThrough, pending, pending_k};
  }

  // kOriginalNatural / kBothDefer: symmetric, hand-rolled, no
  // serve_state_step() involvement at all (these designs predate it /
  // were never wired through it).
  if (kind == StepKind::kOpenClose) return {MicroAction::kOpenCloseStep, pending, pending_k};
  if (kind == StepKind::kWriteClosed) {
    if (policy.kind == PolicyKind::kOriginalNatural) {
      return {MicroAction::kEnterBlocking, pending, pending_k};
    }
    return {MicroAction::kPassThrough, true, k};  // kBothDefer: always defer
  }
  if (kind == StepKind::kImportClosed) {
    if (policy.kind == PolicyKind::kBothDefer && pending) {
      return {MicroAction::kEnterBlocking, false, pending_k};
    }
    return {MicroAction::kPassThrough, pending, pending_k};
  }
  return {MicroAction::kPassThrough, pending, pending_k};  // kAboutToOpen: no-op for both strawmen
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

  auto Tied() const { return std::tie(pc, entering, own_serving, serving_k, pending, pending_k); }
  bool operator<(const ProcPos& o) const { return Tied() < o.Tied(); }
  bool operator==(const ProcPos& o) const { return Tied() == o.Tied(); }
};

struct GlobalState {
  ProcPos a, b;
  bool operator<(const GlobalState& o) const {
    if (!(a == o.a)) return a < o.a;
    return b < o.b;
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
                                          const ProcessPolicy& policy) {
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
  // using THIS process's own explicit pending/pending_k -- never the
  // peer's state, which is irrelevant to the decision (this header's
  // own "PURITY" paragraph).
  const DecideResult d = Decide(policy, step.kind, step.k, pos.pending, pos.pending_k);

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
    return StepOutcome{next, std::nullopt, std::nullopt};
  }
  // kPassThrough (kOpenCloseStep unreachable here; handled above).
  ProcPos next = pos;
  next.pc += 1;
  next.pending = d.new_pending;
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

}  // namespace fuse_examples::lowfive_attach_model
