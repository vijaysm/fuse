#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "fuse/model/coupling_graph.hpp"
#include "fuse/ir/dag.hpp"
#include "fuse/schedule/expand.hpp"

namespace fuse {

/// Verdict of the spec §8.2 init-time gate.
///
/// WHAT `acyclic == true` PROVES: the schedule that `expand()` produced for
/// **one** coupling period, with every dependence that reaches back across the
/// period boundary removed, contains no dependence cycle. Equivalently: there
/// is an execution order for the instructions of one period in which every
/// value is computed before it is read. That is what rules out a deadlock and
/// an algebraic loop.
///
/// WHAT IT DOES NOT PROVE: nothing about numerics. It says nothing about
/// whether the coupled scheme is stable, whether the explicit (lagged)
/// scheme converges, what its order of accuracy is, or whether an iteration
/// region will reach its tolerance inside `max_iterations`. A perfectly
/// acyclic schedule can diverge on the first step.
struct CycleReport {
  /// False iff a dependence cycle was found in the intra-period graph.
  bool acyclic = true;

  /// One representative cycle, as DAG node ids in dependence order: each
  /// entry must run before the next, and the last before the first again.
  /// Consecutive entries are not always joined by a single arc -- a component
  /// step can sit between them (see `explanation`). The same NodeId can appear
  /// more than once when several *instances* of one node lie on the cycle.
  /// Always complete, however long: it is the witness.
  ///
  /// Empty whenever `acyclic`, but the converse does not hold: the refusals
  /// that report failure without searching for a cycle -- a schedule over
  /// `kMaxTimedNodes`, a malformed `TimedDag`, an edge count over
  /// `kMaxDependenceEdges`, and a channel chain that reaches no `Sink` -- all
  /// leave this empty with `acyclic == false`, because there is no cycle to
  /// exhibit, only a proof that was not completed. `explanation` always says
  /// which case it is.
  std::vector<NodeId> cycle;

  /// Indices into `TimedDag::nodes` for the same cycle, one per element of
  /// `cycle`. This is the precise witness: `cycle` alone loses the time at
  /// which each instance runs.
  std::vector<std::size_t> cycle_instances;

  /// Empty iff `acyclic`. Names every channel on the cycle and walks the
  /// cycle node instance by node instance, with times and the component steps
  /// that close it, so the message is actionable in a graph with fifty
  /// channels rather than two. The walk is elided in the middle past
  /// `kMaxExplainedCycleSteps` entries; `cycle` and `cycle_instances` stay
  /// complete.
  std::string explanation;

  /// Independent of `acyclic`: is `TimedDag::nodes`, in the order `expand()`
  /// emitted it, already a valid execution order for the intra-period
  /// dataflow edges? Task 3.3 emits instructions sorted by (time, NodeId) and
  /// relies on ascending NodeId being a topological order of the untimed DAG.
  /// That happens to hold because every lowering rule emits producers before
  /// consumers, but nothing enforces it. This field checks it instead of
  /// assuming it. False means the *scheduler* is wrong, not the user's graph.
  ///
  /// It is checked only on the exactly-paired dataflow edges. The synthetic
  /// join used when producer and consumer instances cannot be paired is an
  /// over-approximation whose extremal endpoints are not a real
  /// producer/consumer pair, so it deliberately contributes nothing here --
  /// including it produced a ~8% spurious-violation rate on legal multirate
  /// schedules (zero genuine violations in 26,647 fuzzed graphs).
  bool emission_order_valid = true;

  /// Empty iff `emission_order_valid`. Names the offending producer/consumer
  /// pair.
  std::string order_explanation;
};

/// Spec §8.2's hard init-time gate: prove that the time-resolved schedule for
/// one coupling period has no dependence cycle.
///
/// A cycle with a **lag** in it is legal and must be accepted: `A -> B` at
/// time level k with `B -> A` at level k-1 is an ordinary explicit scheme, and
/// it is what most production couplers run. An *unlagged* two-way coupling is
/// an algebraic loop and must be rejected. Distinguishing the two is the whole
/// point of the function, so it operates on the time-resolved graph, where the
/// lag has already been unrolled into a dependence on the previous period.
///
/// THE GRAPH IT SEARCHES is not `dag`'s arcs. Any DAG that `lower()` produces
/// is acyclic in its arcs -- a property of `lower()`, not of `OperatorDAG`,
/// which can be hand-built with an arc cycle because `add_arc()` may create
/// an arc before its producer exists -- so a search over the arcs of a
/// lowered DAG can never fail. The cycle in
/// an unlagged two-way coupling closes *inside the components*: component B
/// cannot produce `b_out` for step k until it has been handed `b_in` for step
/// k, and the DAG does not model that because it only models the coupler's own
/// dataflow. `prove_acyclic` builds a graph over `TimedDag` *instances* whose
/// edges are (i) the dataflow arcs, with the lagged ones removed, and (ii) one
/// edge from every channel-delivered `Sink` of a component at step k to every
/// `Source` of that component at step k. (ii) is what gives the gate teeth,
/// and it treats a component as a black box: every output is assumed to depend
/// on every input, because nothing in the data model says otherwise.
///
/// Throws nothing of its own, and in particular cannot throw out of a
/// `Rational`: every time value it touches is compared with `operator<=>`,
/// which is `noexcept` and cross-multiplies in `__int128` without reducing,
/// and every other computation is an integer index. It performs no `Rational`
/// arithmetic at all. (Allocation failure can of course still propagate.)
///
/// Never recurses: cycle extraction uses an explicit stack, because the graph
/// comes from user input and can be as deep as the schedule is long.
///
/// Complexity: O(N log N + E) time and O(N + E) space, where N is
/// `timed.nodes.size()` (bounded by `kMaxTimedNodes`) and E the number of
/// dependence edges (bounded by `kMaxDependenceEdges`). The log N is one sort
/// used to group node instances by (component, step); everything else is
/// linear. There is no depth limit and no iteration cap that could make the
/// search give up and report success.
///
/// ITERATION REGIONS. An implicit coupling declared with `g.iterate({...})` is
/// a cycle at one time level that Picard iteration resolves, so it should be
/// accepted -- but only when the loop actually carries a value. The annotation
/// alone is not evidence: a region with nothing carried is an ordinary cycle
/// wearing a label, and exempting it would be a false negative, the expensive
/// direction.
///
/// So the test is on `OperatorDAG::IterationRegion::carried`, the checkpoint
/// set derived in Phase 7 Task 7.2 by dataflow analysis over the SSA form. An
/// edge is cut iff its arc is in some region's `carried` set AND the consumer
/// is a `Sink` AND the receiving component also owns the source field of a
/// channel inside that region. The last two conditions are refusals to cut,
/// i.e. they too err towards reporting a cycle. Consequences, each pinned by a
/// test: a two-way coupling with `.relax()` inside a region covering it is
/// accepted; the same region with no relaxation carries nothing and is still
/// rejected; relaxation whose channel lies outside the region is still
/// rejected; and a region covering only part of a ring is still rejected --
/// removing the last condition wrongly accepts that ring.
///
/// Error direction: every judgement call errs towards reporting a cycle.
/// Accepting a real cycle (a false negative) lets an unsolvable setup run and
/// produce plausible-looking numbers; rejecting a graph the gate cannot
/// analyse (a false positive) is a setup error the user can see and work
/// around. Concretely, a malformed `TimedDag`, an unmatched producer/consumer
/// instance pairing, and an edge count over the cap are all resolved by
/// keeping or adding edges, never by dropping them.
[[nodiscard]] CycleReport prove_acyclic(const CouplingGraph& g,
                                        const OperatorDAG& dag,
                                        const TimedDag& timed);

/// Cap on the dependence edges `prove_acyclic` will build. Reaching it makes
/// the proof report failure rather than silently analysing a truncated graph.
inline constexpr std::size_t kMaxDependenceEdges = 8'000'000;

/// How many steps of a cycle `CycleReport::explanation` spells out before it
/// elides the middle. A cycle can be as long as the schedule; a megabyte of
/// diagnostic helps nobody.
inline constexpr std::size_t kMaxExplainedCycleSteps = 40;

}  // namespace fuse
