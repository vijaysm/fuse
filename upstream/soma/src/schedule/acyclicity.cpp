#include "soma/schedule/acyclicity.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

// Spec §8.2's hard init-time gate.
//
// THE PROBLEM THIS SOLVES, AND WHY THE OBVIOUS IMPLEMENTATION IS WRONG.
//
// Any DAG that `lower()` produces is acyclic in its arcs. Note the precise
// claim: this is a property of `lower()`, NOT of `OperatorDAG`. Append-only
// plus single assignment does not imply acyclicity, because `add_arc()` can
// create an arc before its producing node exists, so a node can consume an
// arc that a later node produces -- a hand-built `OperatorDAG` with an arc
// cycle is entirely constructible, and `is_acyclic()` correctly returns false
// for it. What is true is that every lowering rule wires an input to an arc
// whose producer already exists, so node id is a topological order of
// `lower()` output. Measured: `lowered.dag.is_acyclic()` held in 44,288 of
// 44,288 fuzzed graphs. So a cycle search over the arcs of a *lowered* DAG
// can never fail -- it is a tautology, not a gate. The cycle in an unlagged
// two-way coupling does
// not live in the arcs. It closes *inside the components*: component B cannot
// produce `b_out` for step k until it has been handed `b_in` for step k, and
// that dependence is nowhere in the DAG because the DAG only models the
// coupler's own dataflow. This file adds that edge (see `component_edges`) and
// that is the only reason the gate has any teeth.
//
// WHAT THE PROOF ESTABLISHES: see the comment on `CycleReport` in the header.
// One period, dependence order only, no numerics.

namespace soma {
namespace {

using Vertex = std::uint32_t;
constexpr Vertex kNoVertex = 0xFFFFFFFFu;

/// The component a boundary node's value lives on, or -1. `ValueType::rank_set`
/// is a component index (see op.hpp); lowering types both a `Source` and a
/// `Sink` arc with `source_type()`, so it is the owner of the field.
[[nodiscard]] int placed_on(const OperatorDAG& dag, const Node& n) {
  if (n.output.valid()) return dag.arc(n.output).type.rank_set;
  if (!n.inputs.empty()) return dag.arc(n.inputs.front()).type.rank_set;
  return -1;
}

// ---------------------------------------------------------------------------
// The iteration-region rule (spec §5.3, Task 7.2).
//
// Task 7.2 fills `IterationRegion::carried`: the coupler buffers whose value at
// sweep k is formed from their value at sweep k-1 -- today, the output of a
// `Relax` (or `Accelerate`) node inside the region. That is the marker in the
// IR this gate was waiting for. Before it existed, `g.iterate({...})` lowered
// to byte-identical straight-line dataflow, and exempting a cycle on the
// strength of the annotation alone would have been a false negative.
//
// WHAT IS CUT, AND WHY IT IS EXACTLY THAT. The carried value is the iteration
// variable. Picard delivers it to the component at the *start of the next
// sweep*, so the one edge it does not constrain within a sweep is its delivery
// to a component: `Relax -> Sink`. Two guards keep the cut from becoming a
// false negative, both of them refusals to cut:
//
//  * Only a `Sink` consumer. The `Residual` node also reads the carried arc,
//    and it must still run after the `Relax` that produced this sweep's value;
//    cutting that edge would leave the residual unordered inside the sweep.
//    Nothing else consumes it today, and any future consumer keeps its edge
//    until it is understood -- a false positive, the cheap direction.
//
//  * Only when the receiving component also *feeds* the region. The runner
//    re-sweeps the region body and nothing else, so a cycle that leaves the
//    region is not resolved by iterating it. Consider a -> b -> c -> a with an
//    iteration region declared over c->a alone, relaxed: cutting there would
//    break the ring in this graph while the runner would never iterate a->b or
//    b->c, and the run would deadlock or produce nonsense. Requiring the
//    component the value is delivered to (here `a`) to own the source field of
//    some channel *in the region* refuses that case and accepts the two-way
//    FSI case, where the region covers both directions. It is a necessary
//    condition for the cycle to lie inside the region, not a sufficient one,
//    so it is a heuristic that only ever refuses more than a full containment
//    proof would; it never cuts an edge that containment would keep.
//
// Pinned in both directions: IterationRegionWithACarriedValueBreaksTheCycle
// and RelaxedSelfCouplingInsideARegionIsAccepted on the accepting side;
// IterationRegionWithoutACarriedValueDoesNotBreakACycle,
// RelaxationOutsideTheRegionDoesNotBreakTheCycle,
// RelaxedRegionOverPartOfARingIsStillRejected and
// SecondCycleOutsideTheRegionIsStillRejected on the refusing side.
// ---------------------------------------------------------------------------
class CarriedValues {
 public:
  CarriedValues(const CouplingGraph& g, const OperatorDAG& dag) {
    regions_.reserve(dag.region_count());
    for (std::size_t i = 0; i < dag.region_count(); ++i) {
      const OperatorDAG::IterationRegion& r =
          dag.region_at(RegionIdx{static_cast<std::uint32_t>(i)});
      if (r.carried.empty()) continue;  // nothing typed as previous-iterate

      Entry e;
      for (const ArcId a : r.carried) {
        if (a.valid() && a.value() < dag.arc_count()) {
          e.carried.push_back(a.value());
        }
      }
      // Components that feed the region: the owners of the source fields of
      // the channels the region's body was lowered from.
      for (const NodeId n : r.body) {
        if (!n.valid() || n.value() >= dag.node_count()) continue;
        const ChannelId origin = dag.node(n).origin;
        if (!origin.valid() || origin.value() >= g.channel_count()) continue;
        const FieldId src = g.channel_at(origin).source;
        if (!src.valid() || src.value() >= g.field_count()) continue;
        e.feeders.push_back(g.owner_of(src).value());
      }
      std::sort(e.carried.begin(), e.carried.end());
      e.carried.erase(std::unique(e.carried.begin(), e.carried.end()),
                      e.carried.end());
      std::sort(e.feeders.begin(), e.feeders.end());
      e.feeders.erase(std::unique(e.feeders.begin(), e.feeders.end()),
                      e.feeders.end());
      if (!e.carried.empty() && !e.feeders.empty()) {
        regions_.push_back(std::move(e));
      }
    }
  }

  /// Does an iteration region carry `in` from the previous sweep to
  /// `consumer`?
  [[nodiscard]] bool cuts(const OperatorDAG& dag, const Node& consumer,
                          ArcId in) const {
    if (regions_.empty()) return false;
    if (consumer.kind != OpKind::Sink) return false;
    if (!in.valid()) return false;
    const int comp = placed_on(dag, consumer);
    if (comp < 0) return false;
    const auto c = static_cast<std::uint32_t>(comp);
    for (const Entry& e : regions_) {
      if (std::binary_search(e.carried.begin(), e.carried.end(), in.value()) &&
          std::binary_search(e.feeders.begin(), e.feeders.end(), c)) {
        return true;
      }
    }
    return false;
  }

 private:
  struct Entry {
    std::vector<std::uint32_t> carried;  // sorted arc indices
    std::vector<std::uint32_t> feeders;  // sorted component indices
  };
  std::vector<Entry> regions_;
};

// ---------------------------------------------------------------------------
// The lag rule -- the single most consequential judgement in this file.
// ---------------------------------------------------------------------------

/// Is the dependence of `consumer` on the value carried by `in` satisfied by a
/// *previous* period (or a previous iterate) rather than by this one?
///
/// Lowering implements `channel.lag(n)` as a `TimeLevel` shift rather than a
/// node (spec §4.3, rules_channel.cpp step 4): every arc emitted downstream of
/// the shift carries `offset -= n`, and the terminal `Sink`'s own output arc is
/// re-typed back to the field's natural level (that mismatch is exactly what
/// `.allow_time_level_mismatch()` authorises). So along a lagged chain the
/// offsets read, for `lag(1)`:
///
///     Source[0] --a0--> Transport[-1] --a5--> Remap[-1] --a6--> Sink[0]
///                  ^0                    ^-1              ^-1
///
/// The boundary is crossed exactly once, and BOTH halves of the test below are
/// needed to find that one place. An edge is loop-carried iff
///
///   (a) the value it carries belongs to an earlier window (offset < 0, or
///       iterate < 0), AND
///   (b) the consumer produces a later level than the one it reads.
///
/// Drop (a) and the test becomes "the consumer reads a level earlier than the
/// one it produces", which also fires on a *lead* -- `.lag(-1)`, which shifts
/// the offset to +1 and so has a level-0 value feeding a level-+1 consumer.
/// Cutting that edge disconnects the chain from its source and hides a real
/// cycle. That is a false negative, the expensive direction. Pinned by
/// Acyclicity.LeadingChannelIsStillRejectedAsACycle.
///
/// Drop (b) and the test becomes `arc.time.offset < 0`, which is what a naive
/// reading of "a lag is a negative offset" suggests. That cuts every edge in
/// the tail of a lagged chain -- three instead of one above -- asserting that
/// `Remap` may run before `Transport` within a period, and hiding any cycle
/// that lives inside the lagged region. Also a false negative. Pinned by
/// Acyclicity.LagDoesNotDropEdgesInsideTheLaggedRegion.
///
/// Everywhere else this errs towards keeping the edge: incomparable anchors
/// and a consumer with no output arc are treated as intra-period, which can
/// only cost a false positive.
///
/// `Rational::operator<=>` is `noexcept` and compares cross-multiplied
/// products in `__int128` without reducing, so this cannot throw or overflow.
[[nodiscard]] bool is_loop_carried(const OperatorDAG& dag,
                                   const CarriedValues& carried,
                                   const Node& consumer, ArcId in) {
  // Spec §5.3 / Task 7.2: the iteration region carries this value from the
  // previous sweep, so it constrains nothing inside this one. See the comment
  // on CarriedValues for the two guards that keep this from being a false
  // negative. Checked before the `consumer.output.valid()` guard below only
  // because a Sink always has an output arc; the order is immaterial.
  if (carried.cuts(dag, consumer, in)) return true;
  if (!consumer.output.valid()) return false;
  const TimeLevel& from = dag.arc(in).type.time;
  const TimeLevel& to = dag.arc(consumer.output).type.time;
  // A dependence on a *previous* iterate is likewise carried by the iteration
  // loop. No lowering rule types an arc at iterate < 0 yet -- Task 7.2's
  // carried set is the marker that does fire -- but the rule is here so that
  // when one does, the gate does not start rejecting every iteration region.
  if (from.iterate < 0 && from.iterate < to.iterate) return true;
  if (from.iterate != to.iterate) return false;
  // Offsets against different anchors are not ordered. Keep the edge.
  if (from.anchor != to.anchor) return false;
  return from.offset < Rational{0} && from.offset < to.offset;
}

// ---------------------------------------------------------------------------
// Dependence graph over timed instances
// ---------------------------------------------------------------------------

/// Vertices [0, instance_count) are indices into `TimedDag::nodes`. Vertices
/// at and above `instance_count` are synthetic *join* vertices: when a group
/// of producers must all precede a group of consumers, routing through one
/// join keeps the edge count linear instead of quadratic while preserving
/// reachability exactly (u -> J -> v for all u,v is the same relation as the
/// full product, and J is fresh so it introduces no cycle of its own).
class DepGraph {
 public:
  explicit DepGraph(std::size_t instance_count)
      : instance_count_(instance_count), succ_(instance_count) {}

  [[nodiscard]] std::size_t instance_count() const noexcept {
    return instance_count_;
  }
  [[nodiscard]] std::size_t size() const noexcept { return succ_.size(); }
  [[nodiscard]] bool overflowed() const noexcept { return overflowed_; }
  [[nodiscard]] const std::vector<Vertex>& succ(Vertex v) const {
    return succ_[v];
  }

  Vertex add_join(std::string label) {
    join_labels_.push_back(std::move(label));
    succ_.emplace_back();
    return static_cast<Vertex>(succ_.size() - 1);
  }

  void add_edge(Vertex u, Vertex v) {
    if (edges_ >= kMaxDependenceEdges) {
      overflowed_ = true;
      return;
    }
    ++edges_;
    succ_[u].push_back(v);
  }

  [[nodiscard]] const std::string& join_label(Vertex v) const {
    return join_labels_[v - instance_count_];
  }

 private:
  std::size_t              instance_count_;
  std::vector<std::vector<Vertex>> succ_;
  std::vector<std::string> join_labels_;
  std::size_t              edges_ = 0;
  bool                     overflowed_ = false;
};

/// (node, instance) -> vertex, stored as one flat array with per-node offsets.
class InstanceIndex {
 public:
  /// Returns false when `timed` is not the shape `expand()` produces: a node
  /// id out of range, or instance indices that are not exactly 0..count-1.
  /// A malformed input is refused, not analysed.
  [[nodiscard]] bool build(const OperatorDAG& dag, const TimedDag& timed) {
    const std::size_t nodes = dag.node_count();
    begin_.assign(nodes + 1, 0);
    for (const TimedNode& tn : timed.nodes) {
      if (!tn.node.valid() || tn.node.value() >= nodes) return false;
      ++begin_[tn.node.value() + 1];
    }
    for (std::size_t i = 0; i < nodes; ++i) begin_[i + 1] += begin_[i];
    slot_.assign(timed.nodes.size(), kNoVertex);
    for (std::size_t i = 0; i < timed.nodes.size(); ++i) {
      const TimedNode& tn = timed.nodes[i];
      const std::uint32_t node = tn.node.value();
      if (tn.instance < 0 ||
          static_cast<std::size_t>(tn.instance) >= count(node)) {
        return false;
      }
      const std::size_t pos = begin_[node] + static_cast<std::size_t>(tn.instance);
      if (slot_[pos] != kNoVertex) return false;  // duplicate instance index
      slot_[pos] = static_cast<Vertex>(i);
    }
    return true;
  }

  [[nodiscard]] std::size_t count(std::uint32_t node) const {
    return begin_[node + 1] - begin_[node];
  }
  /// Vertex of instance `k` of `node`; `k` must be < count(node).
  [[nodiscard]] Vertex at(std::uint32_t node, std::size_t k) const {
    return slot_[begin_[node] + k];
  }

 private:
  std::vector<std::size_t> begin_;  // size node_count+1
  std::vector<Vertex>      slot_;
};

// ---------------------------------------------------------------------------
// Descriptions for the error message
// ---------------------------------------------------------------------------

std::string channel_name(const CouplingGraph& g, ChannelId c) {
  if (!c.valid() || c.value() >= g.channel_count()) return {};
  return g.channel_at(c).name;
}

std::string component_name(const CouplingGraph& g, std::uint32_t comp) {
  if (comp >= g.component_count()) return "component#" + std::to_string(comp);
  return g.component(ComponentId{comp}).name;
}

std::string describe_instance(const CouplingGraph& g, const OperatorDAG& dag,
                              const TimedDag& timed, Vertex v) {
  const TimedNode& tn = timed.nodes[v];
  const Node& n = dag.node(tn.node);
  std::string s = n.label.empty() ? std::string{"node#"} +
                                        std::to_string(tn.node.value())
                                  : n.label;
  s += " (";
  s += to_cstring(n.kind);
  const std::string ch = channel_name(g, n.origin);
  if (!ch.empty()) {
    s += ", channel '";
    s += ch;
    s += "'";
  }
  s += ") @t=";
  s += tn.time.to_string();
  return s;
}

/// One end of a component edge: an instance of a boundary node, tagged with
/// the component and the step of that component's clock it belongs to.
struct Endpoint {
  std::uint32_t comp = 0;
  std::int64_t  step = 0;
  int           side = 0;  // 0 = sink (an input), 1 = source (an output)
  Vertex        vertex = kNoVertex;

  friend bool operator<(const Endpoint& a, const Endpoint& b) {
    if (a.comp != b.comp) return a.comp < b.comp;
    if (a.step != b.step) return a.step < b.step;
    if (a.side != b.side) return a.side < b.side;
    return a.vertex < b.vertex;
  }
};

}  // namespace

CycleReport prove_acyclic(const CouplingGraph& g, const OperatorDAG& dag,
                          const TimedDag& timed) {
  CycleReport report;
  const std::size_t n_inst = timed.nodes.size();
  if (n_inst == 0) return report;  // nothing scheduled, nothing to order

  // Vertex is 32-bit, and every join vertex costs at least two edges, so with
  // this bound the vertex count cannot exceed kMaxTimedNodes +
  // kMaxDependenceEdges/2 and therefore cannot wrap a Vertex. expand() already
  // refuses to build a longer schedule; an over-long one arriving here did not
  // come from expand() and is refused rather than analysed with truncated ids.
  if (n_inst > static_cast<std::size_t>(kMaxTimedNodes)) {
    report.acyclic = false;
    report.explanation =
        "E-SCHED-CYCLE: the expanded schedule has " + std::to_string(n_inst) +
        " instructions, over the limit of " + std::to_string(kMaxTimedNodes) +
        ", so acyclicity was not proved. An unproved schedule is reported as a "
        "failure, never as a pass.";
    return report;
  }

  const std::size_t n_nodes = dag.node_count();

  InstanceIndex idx;
  if (!idx.build(dag, timed)) {
    report.acyclic = false;
    report.explanation =
        "E-SCHED-CYCLE: the expanded schedule is malformed (a timed node "
        "refers to a node outside the DAG, or its instance indices are not "
        "0..n-1), so acyclicity cannot be proved. Refusing it rather than "
        "assuming it is fine.";
    return report;
  }

  // -------------------------------------------------------------------------
  // 0. Structural completeness: every instantiated channel node's value must
  //    be consumed by something.
  //
  // Defence in depth against a lowering bug, not against the user. A channel
  // exists to deliver a value to a component, and the delivery is modelled by
  // the chain ending in a `Sink`, which step 2 below turns into a component
  // input edge. If a channel-origin node's output arc has no consumer at all,
  // that chain dead-ends: the dependence "the target component at step k needs
  // this channel's value" is simply absent from the graph searched below, and
  // an `acyclic == true` verdict would then be a true statement about a
  // smaller dependence relation than the one the user authored -- which is a
  // false negative, the expensive direction. The header commits to reporting
  // an unproved schedule as a failure and never as a pass; this is where that
  // commitment is honoured for an incomplete DAG.
  //
  // The known way to produce one is two channels writing the same target
  // field: `lower()` binds one `Sink` to the last writer, so the earlier
  // writers' chains reach nothing. `validate()` now rejects that up front
  // (E-MULTI-WRITER), so this check is the second line and fires only for a
  // caller that skipped validation or a rule that grows a new way to drop a
  // chain.
  //
  // Two kinds are terminal by construction and are excluded: a `Sink`'s output
  // arc is the end of the chain, and a `Residual`'s is read by the iteration
  // driver through `IterationRegion::residuals`, not by another DAG node.
  //
  // The node scan alone is NOT sufficient, which is why the field scan below
  // it exists. A channel between co-located components on one region with no
  // accumulate/extrapolate/lag/remap/relax lowers to ZERO nodes -- it just
  // rebinds the target field to the source's `Source` arc. Shadowed by a
  // second writer, such a channel leaves no node behind to be found dangling,
  // and the dropped dependence is invisible to any node-based check.
  // Measured: the node scan alone still let 5 of 19,181 fuzzed multi-writer
  // graphs through as acyclic when they were not (seed 3003411 is one).
  // Counting writers per target field closes that, because `lower()` emits
  // exactly one `Sink` per field and a `Sink` consumes exactly one arc, so at
  // most one writer of a field can ever be represented.
  //
  // When spec §5.1's `Merge` is implemented, lowering will emit a `Merge` node
  // consuming every writer's arc and this second scan must learn about it --
  // until then, more than one writer provably means a dropped dependence.
  for (std::uint32_t ni = 0; ni < n_nodes; ++ni) {
    const Node& n = dag.node(NodeId{ni});
    if (n.kind == OpKind::Sink || n.kind == OpKind::Residual) continue;
    if (!n.origin.valid() || n.origin.value() >= g.channel_count()) continue;
    if (idx.count(ni) == 0) continue;  // not instantiated in this period
    if (!n.output.valid() || n.output.value() >= dag.arc_count()) continue;
    if (!dag.consumers_of(n.output).empty()) continue;
    report.acyclic = false;
    report.explanation =
        "E-SCHED-DANGLING-CHANNEL: channel '" + g.channel_at(n.origin).name +
        "' lowers to a chain that dead-ends at node " + std::to_string(ni) +
        " (" + to_cstring(n.kind) + " \"" + n.label +
        "\"): nothing consumes its output, so the channel's value never "
        "reaches a Sink and the dependence it declares is missing from the "
        "schedule. Acyclicity was therefore not proved, and an unproved "
        "schedule is reported as a failure, never as a pass. The usual cause "
        "is two channels writing the same target field, where lowering keeps "
        "only the last writer -- validate() reports that as E-MULTI-WRITER.";
    return report;
  }

  {
    std::vector<std::uint32_t> writers(g.field_count(), 0);
    for (std::uint32_t ci = 0; ci < g.channel_count(); ++ci) {
      const FieldId t = g.channel_at(ChannelId{ci}).target;
      if (t.valid() && t.value() < g.field_count()) ++writers[t.value()];
    }
    for (std::uint32_t ci = 0; ci < g.channel_count(); ++ci) {
      const Channel& ch = g.channel_at(ChannelId{ci});
      const FieldId t = ch.target;
      if (!t.valid() || t.value() >= g.field_count()) continue;
      if (writers[t.value()] < 2) continue;
      report.acyclic = false;
      report.explanation =
          "E-SCHED-DANGLING-CHANNEL: field '" + g.field(t).name + "' is the "
          "target of " + std::to_string(writers[t.value()]) +
          " channels, including '" + ch.name +
          "'. Lowering emits one Sink per field and a Sink consumes one arc, "
          "so at most one of those channels is represented in the schedule "
          "and the dependences the others declare are missing from it. "
          "Acyclicity was therefore not proved, and an unproved schedule is "
          "reported as a failure, never as a pass. Combining writers needs an "
          "explicit Merge, which is not implemented; validate() reports this "
          "as E-MULTI-WRITER.";
      return report;
    }
  }

  DepGraph graph(n_inst);
  const CarriedValues carried(g, dag);

  // First violation of "producers are emitted before their consumers" in the
  // order expand() laid the instructions out. Recorded, not fatal: it is a
  // statement about the scheduler, not about the user's graph.
  const auto note_order = [&](Vertex producer, Vertex consumer) {
    if (!report.emission_order_valid || producer < consumer) return;
    report.emission_order_valid = false;
    report.order_explanation =
        "E-SCHED-ORDER: expand() emitted " +
        describe_instance(g, dag, timed, producer) + " at position " +
        std::to_string(producer) + ", after its consumer " +
        describe_instance(g, dag, timed, consumer) + " at position " +
        std::to_string(consumer) +
        ". The emitted order is sorted by (time, NodeId) and is only a "
        "topological order while ascending NodeId happens to be one; here it "
        "is not.";
  };

  // -------------------------------------------------------------------------
  // 1. Dataflow edges: producer instance -> consumer instance.
  //
  // Instances have to be paired, not just nodes. `expand()` deliberately does
  // not use one emission convention: a non-accumulating chain node fires once
  // per slab at the slab's end, `Sample` once per *source* tick at the tick's
  // start, and the reduction and everything after it once per *target* tick at
  // the step's end. `TimedNode` carries `src_step`/`dst_step` precisely so the
  // pairing can be recovered; that is what the cases below use. Anything that
  // does not pair exactly falls back to a join, which over-connects.
  // -------------------------------------------------------------------------
  // `dataflow` is retained in the signature (every call site distinguishes a
  // dataflow join from a component join) but is deliberately unused now that
  // the join no longer reports emission order -- see the comment at the end
  // of this lambda.
  const auto join_all = [&](std::uint32_t pn, std::uint32_t cn,
                            const std::string& why, [[maybe_unused]] bool dataflow) {
    const Vertex j = graph.add_join(why);
    Vertex max_p = 0;
    Vertex min_c = kNoVertex;
    for (std::size_t k = 0; k < idx.count(pn); ++k) {
      const Vertex v = idx.at(pn, k);
      graph.add_edge(v, j);
      max_p = std::max(max_p, v);
    }
    for (std::size_t k = 0; k < idx.count(cn); ++k) {
      const Vertex v = idx.at(cn, k);
      graph.add_edge(j, v);
      min_c = std::min(min_c, v);
    }
    // Deliberately NO note_order() here. join_all is an over-approximation:
    // it connects every producer instance to every consumer instance through
    // a synthetic vertex precisely because the true instance pairing is not
    // known. Its extremal endpoints (max_p, min_c) are therefore not a real
    // producer/consumer pair, and reporting them as an emission-order
    // violation is a pure false alarm. Measured over 26,647 graphs through
    // the real lower -> expand -> prove_acyclic path: zero genuine order
    // violations, and 1,114 spurious ones in 13,963 clean multirate graphs
    // -- about 8%. Since emission_order_valid is meant to gate the scheduler,
    // that would have rejected 8% of legal multirate schedules, which is the
    // configuration this phase exists to support. Order checking belongs on
    // the exactly-paired dataflow edges, where it is still performed.
  };

  // dst_step -> instances, for the Sample side of an accumulating channel.
  // Built once per producer node and reused.
  std::vector<std::vector<std::pair<std::int64_t, Vertex>>> by_dst(n_nodes);
  std::vector<char> by_dst_built(n_nodes, 0);
  const auto dst_bucket = [&](std::uint32_t node)
      -> const std::vector<std::pair<std::int64_t, Vertex>>& {
    if (!by_dst_built[node]) {
      by_dst_built[node] = 1;
      auto& v = by_dst[node];
      v.reserve(idx.count(node));
      for (std::size_t k = 0; k < idx.count(node); ++k) {
        const Vertex u = idx.at(node, k);
        v.emplace_back(timed.nodes[u].dst_step, u);
      }
      std::sort(v.begin(), v.end());
    }
    return by_dst[node];
  };

  for (std::uint32_t ci = 0; ci < n_nodes; ++ci) {
    if (idx.count(ci) == 0) continue;  // not instantiated in this period
    const Node& c = dag.node(NodeId{ci});
    for (const ArcId in : c.inputs) {
      if (!in.valid() || in.value() >= dag.arc_count()) continue;
      const NodeId prod = dag.arc(in).producer;
      if (!prod.valid() || prod.value() >= n_nodes) continue;
      const std::uint32_t pi = prod.value();
      // A producer with no instance this period imposes no intra-period
      // order. expand() instantiates every node, so this is defensive only.
      if (idx.count(pi) == 0) continue;
      // THE LAG. Skipping this edge is what makes a lagged two-way coupling
      // legal: the value comes from the previous period, so it constrains
      // nothing inside this one.
      if (is_loop_carried(dag, carried, c, in)) continue;

      const Node& p = dag.node(prod);
      const bool p_ch = p.origin.valid() && p.origin.value() < g.channel_count();
      const bool c_ch = c.origin.valid() && c.origin.value() < g.channel_count();

      if (p_ch && c_ch && p.origin == c.origin) {
        if (p.kind == OpKind::Sample && c.kind != OpKind::Sample) {
          // Accumulating channel: every Sample that fell inside target step k
          // must precede the reduction that closes step k.
          const auto& bucket = dst_bucket(pi);
          bool complete = true;
          for (std::size_t k = 0; complete && k < idx.count(ci); ++k) {
            const Vertex cv = idx.at(ci, k);
            const std::int64_t step = timed.nodes[cv].dst_step;
            const auto lo = std::lower_bound(
                bucket.begin(), bucket.end(),
                std::pair<std::int64_t, Vertex>{step, 0});
            std::size_t matched = 0;
            for (auto it = lo; it != bucket.end() && it->first == step; ++it) {
              ++matched;
            }
            // A consumer instance no producer instance feeds would be an
            // unconstrained reduction; treat the pairing as unusable rather
            // than emit a dependence-free reduction, which could hide a cycle.
            if (matched == 0) complete = false;
          }
          if (complete) {
            for (std::size_t k = 0; k < idx.count(ci); ++k) {
              const Vertex cv = idx.at(ci, k);
              const std::int64_t step = timed.nodes[cv].dst_step;
              const auto lo = std::lower_bound(
                  bucket.begin(), bucket.end(),
                  std::pair<std::int64_t, Vertex>{step, 0});
              for (auto it = lo; it != bucket.end() && it->first == step; ++it) {
                graph.add_edge(it->second, cv);
                note_order(it->second, cv);
              }
            }
          } else {
            join_all(pi, ci, "all instances of '" + p.label + "' before '" +
                                 c.label + "' (unmatched target steps)",
                     /*dataflow=*/true);
          }
        } else if (idx.count(pi) == idx.count(ci)) {
          // Same chain, same emission convention: instance k to instance k.
          for (std::size_t k = 0; k < idx.count(ci); ++k) {
            graph.add_edge(idx.at(pi, k), idx.at(ci, k));
            note_order(idx.at(pi, k), idx.at(ci, k));
          }
        } else {
          join_all(pi, ci, "all instances of '" + p.label + "' before '" +
                               c.label + "' (unequal instance counts)",
                   /*dataflow=*/true);
        }
      } else if (!p_ch && c_ch) {
        // Boundary Source feeding the head of a channel chain. The consumer
        // records which source-clock step it reads in `src_step`.
        bool ok = true;
        for (std::size_t k = 0; ok && k < idx.count(ci); ++k) {
          const std::int64_t s = timed.nodes[idx.at(ci, k)].src_step;
          if (s < 0 || static_cast<std::size_t>(s) >= idx.count(pi)) ok = false;
        }
        if (ok) {
          for (std::size_t k = 0; k < idx.count(ci); ++k) {
            const Vertex cv = idx.at(ci, k);
            const Vertex pv =
                idx.at(pi, static_cast<std::size_t>(timed.nodes[cv].src_step));
            graph.add_edge(pv, cv);
            note_order(pv, cv);
          }
        } else {
          join_all(pi, ci, "all instances of '" + p.label + "' before '" +
                               c.label + "' (source step out of range)",
                   /*dataflow=*/true);
        }
      } else if (!p_ch && !c_ch && idx.count(pi) == idx.count(ci)) {
        // Both on the same component clock (a Bidirectional field's Source
        // feeding its own Sink when no channel writes it).
        for (std::size_t k = 0; k < idx.count(ci); ++k) {
          graph.add_edge(idx.at(pi, k), idx.at(ci, k));
          note_order(idx.at(pi, k), idx.at(ci, k));
        }
      } else {
        // Everything left over: a producer and a consumer whose `origin`s name
        // two different channels, or a channel node feeding a non-channel one,
        // or equal-origin counts that do not line up. Over-connect, which is
        // the conservative direction.
        //
        // What this is NOT is a repair for a dropped writer. When two channels
        // write one field, the pairing that lands here is the *surviving*
        // (last) writer's tail against a `Sink` whose `origin` happens to name
        // the *first* writer -- measured, on the p1 multi-writer reproduction:
        // p = 'ch1_b_to_c/transport', c = 'c.c_f' (Sink). The dropped writer's
        // chain has no edge to any Sink at all, so no amount of
        // over-connecting here can restore the dependence it declared; step 0
        // above refuses the proof instead.
        join_all(pi, ci, "all instances of '" + p.label + "' before '" +
                             c.label + "' (unpaired producer and consumer)",
                 /*dataflow=*/true);
      }
    }
  }

  // -------------------------------------------------------------------------
  // 2. Component edges -- the ones that give the gate teeth.
  //
  // A component is a black box: its step k cannot produce any of its outputs
  // until it has been handed every input that step k needs. So for each
  // component X and each step k of X's clock:
  //
  //     every Sink a channel delivers to X at step k  ->  every Source of X
  //                                                       at step k
  //
  // This is what turns an unlagged A<->B coupling into a cycle, and it is why
  // the same topology with a lag is not one: the lag removes the chain edge
  // into the Sink, so the Sink stops being reachable from the far component.
  //
  // Two judgement calls, both erring towards reporting a cycle:
  //
  //  * Every output is assumed to depend on every input. There is no per-field
  //    dependence declaration in the data model, so the black-box assumption
  //    is the only sound one. A component that genuinely computes one output
  //    without one of its inputs will be rejected and must declare a lag. That
  //    is a false positive: visible, and fixable by the user.
  //
  //  * A `Sink` no channel wrote (a Bidirectional field handed straight back
  //    to its own component, Source -> Sink with nothing between) is excluded
  //    from the input side. It carries no coupled data, so it constrains
  //    nothing; including it would make every such field a spurious 2-cycle
  //    against its own Source. A Sink a channel *did* write is included even
  //    when that channel starts at the same component, because a component
  //    feeding its own input instantaneously is a genuine algebraic loop.
  // -------------------------------------------------------------------------
  std::vector<Endpoint> endpoints;
  for (std::uint32_t ni = 0; ni < n_nodes; ++ni) {
    if (idx.count(ni) == 0) continue;
    const Node& n = dag.node(NodeId{ni});
    int side = -1;
    if (n.kind == OpKind::Source) {
      side = 1;
    } else if (n.kind == OpKind::Sink && n.origin.valid() &&
               n.origin.value() < g.channel_count()) {
      side = 0;
    }
    if (side < 0) continue;
    const int comp = placed_on(dag, n);
    if (comp < 0 || static_cast<std::size_t>(comp) >= g.component_count()) {
      continue;  // unplaced; expand() already refuses to schedule these
    }
    for (std::size_t k = 0; k < idx.count(ni); ++k) {
      const Vertex v = idx.at(ni, k);
      // A Source runs on its own component's clock, so its repetition index is
      // the step. A Sink is the tail of a channel chain, so the step of the
      // component it delivers to is the chain's target step.
      const std::int64_t step =
          (side == 1) ? timed.nodes[v].instance : timed.nodes[v].dst_step;
      endpoints.push_back(
          Endpoint{static_cast<std::uint32_t>(comp), step, side, v});
    }
  }
  std::sort(endpoints.begin(), endpoints.end());
  for (std::size_t i = 0; i < endpoints.size();) {
    std::size_t j = i;
    while (j < endpoints.size() && endpoints[j].comp == endpoints[i].comp &&
           endpoints[j].step == endpoints[i].step) {
      ++j;
    }
    std::size_t mid = i;
    while (mid < j && endpoints[mid].side == 0) ++mid;
    const bool has_sinks = mid > i;
    const bool has_sources = mid < j;
    if (has_sinks && has_sources) {
      const Vertex join = graph.add_join(
          "component '" + component_name(g, endpoints[i].comp) + "' step " +
          std::to_string(endpoints[i].step) +
          " (its outputs depend on its inputs)");
      for (std::size_t k = i; k < mid; ++k) graph.add_edge(endpoints[k].vertex, join);
      for (std::size_t k = mid; k < j; ++k) graph.add_edge(join, endpoints[k].vertex);
    }
    i = j;
  }

  if (graph.overflowed()) {
    report.acyclic = false;
    report.explanation =
        "E-SCHED-CYCLE: the expanded schedule needs more than " +
        std::to_string(kMaxDependenceEdges) +
        " dependence edges, so acyclicity was not proved. Reported as a "
        "failure rather than a pass: an unproved schedule is not an acyclic "
        "one. Reduce the number of slabs per period (coarser interval ratios).";
    return report;
  }

  // -------------------------------------------------------------------------
  // 3. Kahn. O(V + E), no recursion, no depth limit, no iteration cap.
  // -------------------------------------------------------------------------
  const std::size_t n_vert = graph.size();
  std::vector<std::uint32_t> indegree(n_vert, 0);
  for (Vertex v = 0; v < n_vert; ++v) {
    for (const Vertex w : graph.succ(v)) ++indegree[w];
  }
  std::vector<Vertex> ready;
  ready.reserve(n_vert);
  for (Vertex v = 0; v < n_vert; ++v) {
    if (indegree[v] == 0) ready.push_back(v);
  }
  std::vector<char> settled(n_vert, 0);
  std::size_t settled_count = 0;
  while (!ready.empty()) {
    const Vertex v = ready.back();
    ready.pop_back();
    settled[v] = 1;
    ++settled_count;
    for (const Vertex w : graph.succ(v)) {
      if (--indegree[w] == 0) ready.push_back(w);
    }
  }
  if (settled_count == n_vert) return report;  // acyclic

  // -------------------------------------------------------------------------
  // 4. Extract one concrete cycle from the residual subgraph.
  //
  // Every unsettled vertex has at least one unsettled predecessor (that is why
  // its indegree never reached zero), so the residual subgraph contains a
  // cycle. Iterative three-colour DFS with an explicit stack: the graph comes
  // from user input and can be as deep as the schedule is long, so recursion
  // would be a stack-overflow waiting to happen. O(V + E).
  // -------------------------------------------------------------------------
  enum : char { kWhite = 0, kGrey = 1, kBlack = 2 };
  std::vector<char> colour(n_vert, kWhite);
  std::vector<std::pair<Vertex, std::size_t>> stack;
  std::vector<Vertex> cycle;

  for (Vertex root = 0; root < n_vert && cycle.empty(); ++root) {
    if (settled[root] || colour[root] != kWhite) continue;
    stack.clear();
    stack.emplace_back(root, 0);
    colour[root] = kGrey;
    while (!stack.empty()) {
      // Deliberately no reference into `stack`: the push below reallocates it.
      // `graph.succ()` is a reference into `DepGraph`, which is complete and
      // not mutated from here on, so that one is safe to hold.
      const Vertex u = stack.back().first;
      const std::size_t next = stack.back().second;
      const std::vector<Vertex>& out = graph.succ(u);
      if (next >= out.size()) {
        colour[u] = kBlack;
        stack.pop_back();
        continue;
      }
      stack.back().second = next + 1;
      const Vertex w = out[next];
      if (settled[w] || colour[w] == kBlack) continue;
      if (colour[w] == kGrey) {
        // w is on the current stack: the cycle runs from w to the top.
        std::size_t from = 0;
        while (from < stack.size() && stack[from].first != w) ++from;
        cycle.reserve(stack.size() - from);
        for (std::size_t k = from; k < stack.size(); ++k) {
          cycle.push_back(stack[k].first);
        }
        break;
      }
      colour[w] = kGrey;
      stack.emplace_back(w, 0);
    }
  }

  report.acyclic = false;

  // -------------------------------------------------------------------------
  // 5. Explain it. "Cycle detected" is useless in a fifty-channel graph, so
  //    name every channel, every node instance, and every component step.
  // -------------------------------------------------------------------------
  // Channel names, deduplicated in O(1) per step via a seen-flag indexed by
  // ChannelId. A linear scan of an accumulating list would be quadratic in the
  // cycle length, and a cycle can be as long as the schedule.
  std::vector<char>        seen_channel(g.channel_count(), 0);
  std::vector<std::string> channels;
  std::size_t              channels_omitted = 0;
  std::string              path;
  // Head and tail of a long cycle; the middle is elided. `cycle` and
  // `cycle_instances` below are still filled in full -- only the prose is cut.
  const std::size_t head = kMaxExplainedCycleSteps >= 8
                               ? kMaxExplainedCycleSteps - 4
                               : kMaxExplainedCycleSteps;
  for (std::size_t k = 0; k < cycle.size(); ++k) {
    const Vertex v = cycle[k];
    const bool show = cycle.size() <= kMaxExplainedCycleSteps || k < head ||
                      k + 4 >= cycle.size();
    if (show) path += (k == 0) ? "    " : "    -> ";
    if (!show && k == head) {
      path += "    ... " + std::to_string(cycle.size() - kMaxExplainedCycleSteps) +
              " more steps ...\n";
    }
    if (v < graph.instance_count()) {
      report.cycle.push_back(timed.nodes[v].node);
      report.cycle_instances.push_back(v);
      if (show) path += describe_instance(g, dag, timed, v);
      const ChannelId origin = dag.node(timed.nodes[v].node).origin;
      if (origin.valid() && origin.value() < g.channel_count() &&
          !seen_channel[origin.value()]) {
        seen_channel[origin.value()] = 1;
        if (channels.size() < kMaxExplainedCycleSteps) {
          channels.push_back(g.channel_at(origin).name);
        } else {
          ++channels_omitted;
        }
      }
    } else if (show) {
      path += graph.join_label(v);
    }
    if (show) path += "\n";
  }
  if (!cycle.empty()) {
    path += "    -> (back to the first)\n";
  }

  std::string names;
  for (const std::string& c : channels) {
    if (!names.empty()) names += ", ";
    names += "'" + c + "'";
  }
  if (channels_omitted != 0) {
    names += " and " + std::to_string(channels_omitted) + " more";
  }

  report.explanation =
      "E-SCHED-CYCLE: the expanded schedule for one coupling period (period = " +
      timed.period.to_string() +
      ") contains a dependence cycle that no lag breaks, so no execution order "
      "exists and the coupled system is an algebraic loop.\n"
      "  channels on the cycle: " +
      (names.empty() ? std::string{"(none named)"} : names) +
      "\n  cycle:\n" + path +
      "  Fix: add .lag(1) to one channel on the cycle to make it an explicit "
      "scheme, together with .allow_time_level_mismatch(\"reason\") to "
      "authorise the staggering.\n"
      "  Or make it implicit: spec §5.3's iteration region breaks a cycle only "
      "where it carries a previous-iterate value, and declaring the region "
      "alone does not create one. Task 7.2 derives IterationRegion::carried "
      "from the SSA form, and a region whose channels have no .relax(omega) "
      "carries nothing, so it lowers to the same straight-line dataflow and "
      "this gate still rejects it. Add .relax(omega) to a channel inside a "
      "region that covers the whole cycle -- every component on it must both "
      "send into and receive from the region, or iterating the region would "
      "not resolve the dependence and the cut is refused.";
  return report;
}

}  // namespace soma
