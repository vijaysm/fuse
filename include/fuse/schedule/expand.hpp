#pragma once

#include <cstdint>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/ir/dag.hpp"
#include "fuse/schedule/clock.hpp"

namespace fuse {

/// WHERE AN INSTANCE SITS RELATIVE TO THE COMPONENT TICKS THAT BEGIN AT THE
/// SAME INSTANT.
///
/// A time alone does not order an exchange against a component's own advance,
/// because both happen "at t". This is the missing half of the timestamp, and
/// it is what decides whether a value delivered at `t` is seen by the step
/// that begins at `t` or only by the one after it. `Runner::Impl::step_period`
/// interleaves the two streams on (time, phase); nothing else reads it.
///
/// Two conventions, and only two:
///
///   * `BeforeTicks` -- the instance runs before the ticks that begin at
///     `time`, so a delivery it makes is consumed by the target step that
///     begins there.
///   * `AfterTicks` -- the instance runs after them. `Sample` is always this:
///     it is stamped at its source step's START and wants the value that step
///     produces, which is why the window a target step closes over is "every
///     source step that BEGINS inside it" (see `expand()`).
///
/// THE CAUSALITY CONTRACT, which is what decides the phase of a delivery:
///
///     A delivery may be `BeforeTicks` -- may be consumed by the target step
///     that begins at `time` -- only if the value it carries is ATTRIBUTED TO
///     AN INTERVAL ENDING AT OR BEFORE `time`.
///
/// "Attributed to", not "computed from", and the difference is the whole of
/// it. The two kinds of chain attribute their value to different intervals:
///
///   * An INSTANT chain (no accumulation, no space-time) carries the source's
///     state at the end of source step `src_step`, an instant: `(src_step + 1)
///     * src_dt`. Its instances are stamped at slab ends, and a slab ends
///     either where its source step does -- then the value is complete and the
///     instance is `BeforeTicks` -- or where its TARGET step does, ahead of
///     the source step it belongs to. In that second case the source state it
///     would hand over is not valid until after the target step it would be
///     handed to has already begun, so it must be `AfterTicks` and be consumed
///     one target step later. It loses nothing by waiting: a target-limited
///     slab carries the same `src_step` as the slab after it, so the value is
///     redelivered unchanged at the next boundary.
///
///   * A WINDOW-CLOSING chain (accumulation, or a space-time projection)
///     carries a quantity attributed to the TARGET STEP it closes,
///     `[dst_step * dst_dt, (dst_step + 1) * dst_dt)`, whose end IS the stamp.
///     It therefore always satisfies the contract and is always `BeforeTicks`.
///     Some of the source steps it reduces or projects extend a little past
///     that edge -- the window is "every source step that begins inside it" --
///     but that is a property of reconstructing a PAST interval from the
///     samples that overlap it, governed by `Instruction::src_step` and, for
///     space-time, by `check_spacetime_schedule()`'s stencil-reach refusal.
///     It is not the target consuming its own future state.
///
/// Reading the contract as "produced from source steps ending at or before
/// `time`" instead would force every window-closing chain to `AfterTicks` for
/// any non-nesting ratio -- 1/5 against 1/3 puts source step 1, [1/5, 2/5),
/// inside target step 0, which ends at 1/3 -- and that is exactly the R2
/// defect this phase exists to close.
///
/// The ordering `BeforeTicks < AfterTicks` is load-bearing: it is the sort key
/// both `expand()` and `schedule()` use after `time`, so within one instant
/// every causally-available delivery precedes every tick, and every tick
/// precedes every `Sample`.
enum class TickPhase : std::uint8_t {
  BeforeTicks = 0,
  AfterTicks  = 1,
};

/// One instance of a DAG node at a specific time within the common period.
///
/// `src_step` / `dst_step` are the step indices of the channel's source and
/// target clocks that this instance belongs to, both counted from the start
/// of the period. For a boundary node (no channel) both hold the owning
/// component's step. They are always filled in: a reduction that runs once
/// per target step still records the last source step that fed it, and a
/// sample that runs on the source clock still records the target step it
/// falls inside.
struct TimedNode {
  NodeId       node;
  Rational     time;
  std::int64_t src_step = 0;
  std::int64_t dst_step = 0;
  std::int64_t instance = 0;  // repetition index of this node within the period
  TickPhase    phase = TickPhase::BeforeTicks;
};

struct TimedDag {
  std::vector<TimedNode> nodes;
  Rational               period{1};
  bool                   lockstep = false;
};

inline constexpr std::int64_t kMaxTimedNodes = 1'000'000;

/// Which expansion algorithm `expand()` may use.
///
/// Test seam. `Auto` is the production setting and the only one any caller
/// outside the test suite should pass. `ForceGeneral` exists because the
/// lockstep fast path (spec Section 8.2) is an *optimisation*, not a second set of
/// semantics: the only way to hold it to that is to run the same graph through
/// both algorithms and compare the results, which needs a way to suppress the
/// shortcut from outside. `TimedDag::lockstep` describes the *graph*, not the
/// path taken, so it is reported identically under either mode -- otherwise
/// the two dumps could not be compared byte for byte.
enum class ExpandMode : std::uint8_t {
  Auto,          ///< take the lockstep fast path when the graph qualifies
  ForceGeneral,  ///< always run the general, slab-based expansion
};

/// Unroll the lowered DAG over one common period of the clock set
/// (spec Section 8, Section 8.3).
///
/// Every node of `dag` is instantiated at least once:
///   - a node whose `origin` names a channel is instantiated once per slab of
///     that channel's (source, target) decomposition, except that when the
///     channel carries a `Sample` -- because it accumulates, or because it is
///     a space-time channel (spec Section 7.3) -- that `Sample` runs once per source
///     tick and everything downstream of it once per target tick;
///   - every other node (`Source`, and any `Sink` no channel wrote) is
///     instantiated once per tick of the component its value lives on.
/// The result is sorted by (time, `TickPhase`, NodeId); NodeId is a
/// topological order because the DAG is built append-only.
///
/// Errors (never exceptions -- all time arithmetic is range-checked in
/// __int128 before it reaches Rational, whose operators throw on overflow):
///   E-SCHED-PERIOD   the common period is unrepresentable or too long, an
///                    interval does not divide it, or a tick value does not
///                    fit an exact Rational
///   E-SCHED-EXPLODE  more than kMaxTimedNodes instances
///   E-SCHED-EXPAND   `clocks` does not describe `g`, or a node's value is
///                    not placed on any component
///   E-SCHED-SPACETIME  a channel asks for both accumulation and a space-time
///                    basis. Each needs its own `Sample` on the source clock
///                    and the two cannot both be scheduled; see the check at
///                    the top of `expand()`
///
/// When every component shares one interval and no channel is space-time
/// (`ClockSet::is_lockstep()`, spec Section 8.2), the common period is that one
/// interval, every clock ticks exactly once in it, and every slab
/// decomposition is the single slab [0, period). `expand()` then takes a
/// shortcut that emits one instance per node directly and never calls
/// `decompose()`. The shortcut is required to produce the *same* `TimedDag`
/// the general path would -- same instances, same times, same step indices,
/// same order -- and `ExpandMode::ForceGeneral` exists so a test can check
/// that by construction rather than by inspection.
[[nodiscard]] Result<TimedDag> expand(const CouplingGraph& g,
                                      const OperatorDAG& dag,
                                      const ClockSet& clocks,
                                      ExpandMode mode = ExpandMode::Auto);

}  // namespace fuse
