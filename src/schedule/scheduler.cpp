#include "fuse/schedule/plan.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include "fuse/ir/dump.hpp"  // detail::attrs_string, detail::pad
#include "fuse/schedule/window.hpp"

namespace fuse {
namespace {

/// Sentinel for "this node belongs to no iteration region". -1 is the value
/// Instruction::region_enter/exit already use, so the two agree.
constexpr std::int32_t kNoRegion = -1;

/// How many operands one Instruction can hold. Tied to the array so the two
/// cannot drift apart.
constexpr std::size_t kMaxInstructionInputs =
    std::tuple_size<decltype(Instruction::in)>::value;
static_assert(kMaxInstructionInputs == 3,
              "spec §5.1: three inputs covers every primitive; Merge is "
              "lowered as a left-fold of pairwise merges");

/// Does `in` carry a value from an earlier coupling period into `consumer`,
/// so that it imposes no ordering constraint *inside* this period?
///
/// The canonical statement of this rule, with the reasoning for both halves
/// and the two tests that pin them, is `is_loop_carried()` in
/// src/schedule/acyclicity.cpp; it lives in an anonymous namespace there, so
/// this is a deliberate second copy rather than a shared declaration. Both
/// halves are needed: `(a)` the value belongs to an earlier window and `(b)`
/// the consumer produces a later level than it reads. Everywhere else this
/// errs towards *keeping* the edge, which can only make the order check below
/// stricter -- never more permissive.
///
/// `Rational::operator<=>` is noexcept and cross-multiplies in __int128, so
/// this cannot throw or overflow.
[[nodiscard]] bool crosses_period(const OperatorDAG& dag, const Node& consumer,
                                  ArcId in) {
  if (!consumer.output.valid()) return false;
  const TimeLevel& from = dag.arc(in).type.time;
  const TimeLevel& to = dag.arc(consumer.output).type.time;
  if (from.iterate < 0 && from.iterate < to.iterate) return true;
  if (from.iterate != to.iterate) return false;
  if (from.anchor != to.anchor) return false;
  return from.offset < Rational{0} && from.offset < to.offset;
}

/// Map every DAG node to the iteration region that owns it, or kNoRegion.
///
/// Membership is `IterationRegion::body` *plus* the producer of every residual
/// arc. `build_iteration_regions()` collects `body` before it appends the
/// Residual nodes, so a residual is not in `body` -- but the residual is the
/// region's convergence measure and is computed once per pass, so it is inside
/// the loop. Leaving it out would put the convergence test outside the loop it
/// controls.
///
/// Returns an error rather than a map when two regions claim the same node:
/// one flat `region_enter`/`region_exit` pair per instruction cannot express
/// nesting, and guessing would produce a loop structure nobody wrote.
[[nodiscard]] Result<std::vector<std::int32_t>> region_membership(
    const OperatorDAG& dag) {
  const std::size_t n_nodes = dag.node_count();
  const std::size_t n_regions = dag.region_count();
  std::vector<std::int32_t> owner(n_nodes, kNoRegion);
  if (n_regions == 0) return owner;

  // region_enter is an int32_t index into the region list.
  if (n_regions > static_cast<std::size_t>(
                      std::numeric_limits<std::int32_t>::max())) {
    return Failure{"E-SCHED-REGION",
                 "the DAG declares " + std::to_string(n_regions) +
                     " iteration regions, more than an instruction's "
                     "region_enter index can hold"};
  }

  const auto claim = [&](NodeId n, std::int32_t r,
                         const std::string& what) -> Result<void> {
    if (!n.valid() || n.value() >= n_nodes) {
      return Failure{"E-SCHED-REGION",
                   "iteration region " + std::to_string(r) + " names " + what +
                       " node " + std::to_string(n.value()) +
                       ", which is outside the DAG"};
    }
    std::int32_t& slot = owner[n.value()];
    if (slot != kNoRegion && slot != r) {
      return Failure{"E-SCHED-REGION",
                   "node " + std::to_string(n.value()) + " (\"" +
                       dag.node(n).label + "\") belongs to iteration regions " +
                       std::to_string(slot) + " and " + std::to_string(r) +
                       "; nested or overlapping regions cannot be expressed "
                       "as one enter/exit pair per instruction"};
    }
    slot = r;
    return Result<void>::success();
  };

  for (std::size_t i = 0; i < n_regions; ++i) {
    const auto r = static_cast<std::int32_t>(i);
    const OperatorDAG::IterationRegion& reg =
        dag.region_at(RegionIdx{static_cast<std::uint32_t>(i)});
    for (const NodeId n : reg.body) {
      const Result<void> c = claim(n, r, "body");
      if (!c.ok()) return c.error();
    }
    for (const ArcId a : reg.residuals) {
      if (!a.valid() || a.value() >= dag.arc_count()) {
        return Failure{"E-SCHED-REGION",
                     "iteration region " + std::to_string(r) +
                         " names residual arc " + std::to_string(a.value()) +
                         ", which is outside the DAG"};
      }
      const NodeId p = dag.arc(a).producer;
      if (!p.valid()) continue;  // an unproduced residual constrains nothing
      const Result<void> c = claim(p, r, "residual-producing");
      if (!c.ok()) return c.error();
    }
  }
  return owner;
}

/// Stamp `region_enter` / `region_exit` on maximal contiguous runs of one
/// region's members.
///
/// A run must be exactly one whole pass over the region: as many instructions
/// as the region has member nodes, each a different node. Anything else --
/// a region whose channels sit on different clocks, so its members interleave
/// with other work -- cannot be expressed as a single flat loop, and is
/// refused. Emitting a plan that quietly runs `max_iterations = 5` exactly
/// once is the silent-wrong-answer direction; a setup error is the visible
/// one.
[[nodiscard]] Result<void> mark_regions(const OperatorDAG& dag,
                                        const std::vector<std::int32_t>& owner,
                                        std::vector<Instruction>& instructions) {
  const std::size_t n_regions = dag.region_count();
  if (n_regions == 0) return Result<void>::success();

  std::vector<std::size_t> member_count(n_regions, 0);
  for (const std::int32_t r : owner) {
    if (r != kNoRegion) ++member_count[static_cast<std::size_t>(r)];
  }

  // "Which run did I last see this node in?" so distinctness inside a run is
  // O(1) per instruction and needs no allocation per run.
  std::vector<std::size_t> last_run(dag.node_count(), 0);
  std::size_t run_id = 0;

  const std::size_t n = instructions.size();
  std::size_t i = 0;
  while (i < n) {
    const std::uint32_t node0 = instructions[i].origin_node.value();
    const std::int32_t r = owner[node0];
    if (r == kNoRegion) {
      ++i;
      continue;
    }
    ++run_id;
    std::size_t j = i;
    std::size_t distinct = 0;
    while (j < n && owner[instructions[j].origin_node.value()] == r) {
      const std::uint32_t nd = instructions[j].origin_node.value();
      if (last_run[nd] != run_id) {
        last_run[nd] = run_id;
        ++distinct;
      }
      ++j;
    }
    const std::size_t run_len = j - i;
    const std::size_t want = member_count[static_cast<std::size_t>(r)];
    if (run_len != want || distinct != want) {
      const OperatorDAG::IterationRegion& reg =
          dag.region_at(RegionIdx{static_cast<std::uint32_t>(r)});
      return Failure{
          "E-SCHED-REGION",
          "iteration region '" + reg.name + "' (" + std::to_string(r) +
              ") has " + std::to_string(want) +
              " member nodes, but instructions " + std::to_string(i) + ".." +
              std::to_string(j - 1) + " (at t=" +
              instructions[i].time.to_string() + ") form a run of " +
              std::to_string(run_len) + " covering " +
              std::to_string(distinct) +
              " of them. A region can only become flat enter/exit boundaries "
              "when one whole pass over its body is contiguous in the "
              "instruction stream; put the region's channels on one clock, or "
              "split it into one region per clock. If they are ALREADY on one "
              "clock, the interleaving instruction reads a value this region "
              "re-advances -- an explicit channel hanging off one of its "
              "components is the usual shape -- and it would run once, on one "
              "sweep's iterate, so its place relative to the loop is a "
              "modelling decision and is not chosen for you. Route it from "
              "the region's output instead. (Declaration order used to matter "
              "here and no longer does: KNOWN_ISSUES.md D1.)"};
    }
    instructions[i].region_enter = r;
    instructions[j - 1].region_exit = r;
    i = j;
  }
  return Result<void>::success();
}

[[nodiscard]] std::string region_label(RegionId r) {
  return r.valid() ? ("r" + std::to_string(r.value())) : std::string{"r?"};
}

/// Position of every node in a topological order of the *untimed* DAG.
///
/// This, not ascending NodeId, is what breaks ties between instructions that
/// `expand()` placed at the same time. `expand()` emits at three different
/// conventions -- chain nodes at the slab end, `Sample` at the source tick
/// start, reductions at the target step end -- so equal times are the normal
/// case, not an edge case, and "sorted by time" says nothing about dependence.
/// Ascending NodeId happens to be a topological order today (OperatorDAG is
/// append-only and every lowering rule emits producers before consumers), but
/// nothing enforces it, so the rank is computed rather than assumed.
///
/// Returns an error if the arcs of `dag` do not admit a topological order at
/// all. `OperatorDAG` is acyclic by construction -- single assignment, a
/// second producer rejected at add_node() -- so this is defensive.
[[nodiscard]] Result<std::vector<std::uint32_t>> topological_rank(
    const OperatorDAG& dag) {
  const std::vector<NodeId> order = dag.topological_order();
  if (order.size() != dag.node_count()) {
    return Failure{"E-SCHED-PLAN",
                 "the operator DAG has " + std::to_string(dag.node_count()) +
                     " nodes but only " + std::to_string(order.size()) +
                     " of them can be topologically ordered, so its arcs "
                     "contain a cycle and no instruction order exists"};
  }
  std::vector<std::uint32_t> rank(dag.node_count(), 0);
  for (std::size_t i = 0; i < order.size(); ++i) {
    // topological_order() only ever returns ids that came from the DAG.
    rank[order[i].value()] = static_cast<std::uint32_t>(i);
  }
  return rank;
}

/// Reorder inside each (time, phase) instant so one region's members come out
/// contiguously — KNOWN_ISSUES D1, fixed 2026-09-21.
///
/// The sort below orders an instant by topological rank, which is a dependence
/// order but says nothing about grouping: two independent chains at the same
/// instant interleave whenever their ranks alternate, and whether they do
/// depended on the order the user happened to declare their components in. A
/// region whose body was cut in half by an unrelated channel was then refused
/// (`E-SCHED-REGION`) even though an order existed that kept it whole — the
/// user's only cure was to move a declaration, which is not a property of
/// their coupling at all.
///
/// The rule is a greedy emission with one preference: while a region has
/// members left in this instant, prefer its ready members; otherwise take the
/// first ready instruction. **Only ready instructions are ever emitted** — an
/// instruction whose operands are all written already, carried in from the
/// previous period, or unproduced — so this cannot invert a dependence, which
/// is why it needs no argument beyond that sentence. `check_dependence_order`
/// re-derives the same property from the finished stream regardless.
///
/// With no region in an instant the result is *identical* to the sort: within
/// one instant an operand always has a lower topological rank than its
/// consumer, so scanning in rank order always finds the lowest-ranked
/// instruction ready, and greedy-lowest-ready is the rank order itself. That
/// is what keeps the golden IR of every region-free plan byte-identical.
///
/// A region can still be refused: if one of its members is blocked by an
/// unrelated instruction that has not run yet, that instruction must be
/// emitted and the run breaks. The difference is that the refusal now means
/// the graph genuinely does not admit a contiguous pass, rather than that the
/// components were declared in an unlucky order.
void group_regions_within_instants(const OperatorDAG& dag,
                                   const std::vector<std::int32_t>& owner,
                                   std::vector<Instruction>& instructions) {
  const std::size_t n_regions = dag.region_count();
  if (n_regions == 0) return;

  std::vector<char> written(dag.arc_count(), 0);
  const auto is_ready = [&](const Instruction& ins) {
    const Node& nd = dag.node(ins.origin_node);
    for (const ArcId in : nd.inputs) {
      if (!in.valid() || in.value() >= dag.arc_count()) continue;
      if (!dag.arc(in).producer.valid()) continue;
      if (crosses_period(dag, nd, in)) continue;
      if (written[in.value()] == 0) return false;
    }
    return true;
  };
  const auto commit = [&](const Instruction& ins) {
    if (ins.out.valid() && ins.out.value() < dag.arc_count()) {
      written[ins.out.value()] = 1;
    }
  };

  const std::size_t n = instructions.size();
  std::size_t lo = 0;
  while (lo < n) {
    std::size_t hi = lo + 1;
    while (hi < n && instructions[hi].time == instructions[lo].time &&
           instructions[hi].phase == instructions[lo].phase) {
      ++hi;
    }

    std::vector<std::size_t> left(n_regions, 0);
    bool any_member = false;
    for (std::size_t k = lo; k < hi; ++k) {
      const std::int32_t r = owner[instructions[k].origin_node.value()];
      if (r != kNoRegion) {
        ++left[static_cast<std::size_t>(r)];
        any_member = true;
      }
    }

    // Group ONLY where the displaced instructions share no data with the
    // region. A non-member that reads what a member produces -- or produces
    // what a member reads -- is not unrelated work that happens to sit in the
    // way: where it runs relative to the loop decides which iterate it sees.
    // An explicit channel hanging off a component the region re-advances is
    // the case that matters: it would run once, on the pre-iteration value,
    // and the converged iterate would never reach its consumer. Until this
    // grouping existed, `E-SCHED-REGION` refused that graph as a side effect
    // of the interleave, and `Iteration.RejectsAnExplicitChannelHangingOff`
    // `AnIteratedComponent` measures the refusal. Separating the interleave
    // would have turned that refusal into a silently-chosen placement, so an
    // entangled instant keeps the order it had -- and with it, whatever
    // verdict it produced before.
    // An arc a member reads that a member does NOT produce: a value coming in
    // from outside the loop. If a non-member reads one of those too, the two
    // read the same iterated value and where the non-member runs decides
    // which sweep's value it sees -- that is the hanging-channel hazard, and
    // choosing silently is the wrong answer to it.
    //
    // Reading an arc a member PRODUCES is the opposite case and must stay
    // groupable: the region's own sink (`Sink "pb.in"`, a non-member, reads
    // the region's converged output) is exactly what interleaved with the
    // body and made contiguity depend on declaration order. Deferring it past
    // the loop is not a silent choice -- it is the only correct placement.
    bool entangled = false;
    {
      std::vector<char> member_produces(dag.arc_count(), 0);
      for (std::size_t k = lo; k < hi; ++k) {
        if (owner[instructions[k].origin_node.value()] == kNoRegion) continue;
        const Instruction& ins = instructions[k];
        if (ins.out.valid() && ins.out.value() < dag.arc_count()) {
          member_produces[ins.out.value()] = 1;
        }
      }
      std::vector<char> from_outside(dag.arc_count(), 0);
      for (std::size_t k = lo; k < hi; ++k) {
        if (owner[instructions[k].origin_node.value()] == kNoRegion) continue;
        for (const ArcId in : dag.node(instructions[k].origin_node).inputs) {
          if (!in.valid() || in.value() >= dag.arc_count()) continue;
          const NodeId prod = dag.arc(in).producer;
          const bool inside =
              prod.valid() && prod.value() < owner.size() &&
              owner[prod.value()] != kNoRegion;
          if (!inside) from_outside[in.value()] = 1;
        }
      }
      for (std::size_t k = lo; k < hi && !entangled; ++k) {
        if (owner[instructions[k].origin_node.value()] != kNoRegion) continue;
        for (const ArcId in : dag.node(instructions[k].origin_node).inputs) {
          if (in.valid() && in.value() < dag.arc_count() &&
              from_outside[in.value()] != 0) {
            entangled = true;
          }
        }
      }
      (void)member_produces;
    }

    if (std::getenv("FUSE_SCHEDULE_DEBUG") != nullptr) {
      fmt::print(stderr,
                 "[instant {}..{}] t={} phase={} members={} entangled={}\n",
                 lo, hi - 1, instructions[lo].time.to_string(),
                 static_cast<int>(instructions[lo].phase), any_member,
                 entangled);
      for (std::size_t k = lo; k < hi; ++k) {
        const Instruction& ins = instructions[k];
        const Node& nd = dag.node(ins.origin_node);
        std::string ins_s;
        for (const ArcId in : nd.inputs) {
          if (in.valid() && in.value() < dag.arc_count()) {
            ins_s += " %" + std::to_string(in.value()) + ":" + dag.arc(in).name;
          }
        }
        fmt::print(stderr, "    [{}] {} \"{}\" region={} reads{} writes %{}\n",
                   k, to_cstring(nd.kind), nd.label,
                   owner[ins.origin_node.value()], ins_s,
                   ins.out.valid() ? std::to_string(ins.out.value()) : "-");
      }
    }

    if (!any_member || entangled || hi - lo < 2) {
      for (std::size_t k = lo; k < hi; ++k) commit(instructions[k]);
      lo = hi;
      continue;
    }

    // Candidates in the sort's order, so "first ready" means lowest rank.
    const std::vector<Instruction> bucket(instructions.begin() + lo,
                                          instructions.begin() + hi);
    std::vector<char>  taken(bucket.size(), 0);
    const std::size_t  none = bucket.size();
    std::int32_t       open = kNoRegion;

    for (std::size_t out = lo; out < hi; ++out) {
      std::size_t pick = none;
      if (open != kNoRegion) {
        for (std::size_t k = 0; k < bucket.size(); ++k) {
          if (taken[k] != 0) continue;
          if (owner[bucket[k].origin_node.value()] != open) continue;
          if (!is_ready(bucket[k])) continue;
          pick = k;
          break;
        }
      }
      if (pick == none) {
        for (std::size_t k = 0; k < bucket.size(); ++k) {
          if (taken[k] == 0 && is_ready(bucket[k])) {
            pick = k;
            break;
          }
        }
      }
      if (pick == none) {
        // Nothing is ready, so the instant's own order was not a dependence
        // order to begin with. Emit what is left as it stands and let
        // check_dependence_order report it against the real stream rather
        // than silently reshuffling a stream that is already wrong.
        for (std::size_t k = 0; k < bucket.size(); ++k) {
          if (taken[k] == 0) {
            pick = k;
            break;
          }
        }
      }

      taken[pick] = 1;
      const std::int32_t r = owner[bucket[pick].origin_node.value()];
      if (r != kNoRegion) {
        open = r;
        if (--left[static_cast<std::size_t>(r)] == 0) open = kNoRegion;
      }
      commit(bucket[pick]);
      instructions[out] = bucket[pick];
    }
    lo = hi;
  }
}

/// The scheduler's own dependence check on the finished stream: every operand
/// must be written by an *earlier* instruction, except one carried in from the
/// previous period.
///
/// Deliberately not `CycleReport::emission_order_valid`. That flag is derived
/// from `prove_acyclic`'s deliberately over-approximated dependence graph --
/// `join_all()` reports the extremal endpoints of a join, which are not a real
/// producer/consumer pair -- and a sweep over 26,647 generated graphs measured
/// ~8% spurious violations on clean multirate schedules and none that were
/// real. Rejecting a plan on it would reject roughly one legal multirate
/// schedule in twelve. This check reads only the SSA arcs and the emitted
/// order, so it has no such failure mode.
[[nodiscard]] Result<void> check_dependence_order(
    const OperatorDAG& dag, const std::vector<Instruction>& instructions) {
  std::vector<char> written(dag.arc_count(), 0);
  for (std::size_t i = 0; i < instructions.size(); ++i) {
    const Instruction& ins = instructions[i];
    const Node& n = dag.node(ins.origin_node);
    for (const ArcId in : n.inputs) {
      if (!in.valid() || in.value() >= dag.arc_count()) continue;
      if (!dag.arc(in).producer.valid()) continue;  // nothing writes it
      if (crosses_period(dag, n, in)) continue;     // last period's value
      if (written[in.value()] != 0) continue;
      return Failure{
          "E-SCHED-ORDER",
          "instruction " + std::to_string(i) + " (" +
              std::string{to_cstring(n.kind)} + " \"" + n.label + "\" at t=" +
              ins.time.to_string() + ") reads arc %" +
              std::to_string(in.value()) + " (\"" + dag.arc(in).name +
              "\"), which no earlier instruction has written. The emitted "
              "order is not a dependence order, which is a scheduler defect, "
              "not a defect in the coupling graph"};
    }
    if (ins.out.valid() && ins.out.value() < dag.arc_count()) {
      written[ins.out.value()] = 1;
    }
  }
  return Result<void>::success();
}

}  // namespace

namespace detail {
/// Time windows track: the ordering proof for the subcycling node
/// (`OpKind::WindowAdvance`). Its import waveform is deliberately NOT a
/// real `Node::inputs` edge (`OpAttrs::window_import`'s own doc comment,
/// fuse/ir/op.hpp, explains why: a real edge closes a cycle in the untimed
/// graph). `check_dependence_order()` above therefore cannot see this
/// dependency at all -- it walks `Node::inputs` only -- so causality here
/// must be established, and checked, some other way: by TIME. Its export
/// waveform needs no separate check: a field transfer reading it names it
/// in a REAL `Node::inputs` edge, which `check_dependence_order()` already
/// covers.
///
/// THE INVARIANT, stated exactly: for every window instance w (identified
/// by `Instruction::dst_step`, which `expand()` gives the same value to
/// every node -- channel or subcycling -- decomposed at one window's
/// length, since both ends of that decomposition are the window length)
/// and every subcycling node with a valid `window_import`, the instruction
/// that writes `window_import`'s arc for window instance w must appear
/// EARLIER in the SCHEDULED STREAM (vector position, not merely time --
/// position is what the runtime's execute-in-order loops actually read)
/// than the subcycling node's own instance for w.
///
/// Enforced by scanning the stream once, tracking each arc's most
/// recently-seen writer's `dst_step`; at a subcycling instance, the
/// tracked step for its `window_import` arc must equal ITS OWN `dst_step`
/// exactly -- not merely "some earlier write happened", which (unlike
/// `check_dependence_order()`'s coarser, sufficient-for-its-purpose flag)
/// would silently accept a subcycling instance reading a STALE window's
/// value if the graph or the phase assignment is wrong.
///
/// Declared in fuse/schedule/plan.hpp's `detail` namespace, not kept
/// anonymous here, so a test can exercise it directly against a hand-built
/// instruction stream (this track's Unit P: the proof must hold before any
/// lowering rule emits the node it is about).
// Time windows track: is the field transfer that delivers `imp` a LAGGED
// one (`Channel::lag != 0`)? Found via `imp`'s producer node's own
// `origin` -- the channel every node in a lowered chain carries
// (fuse/ir/lowering.cpp step 2 / src/ir/rules_channel.cpp) -- rather than
// via region membership: a Sink is deliberately never part of any
// `IterationRegion::body` (it is downstream of the loop, not inside it;
// see group_regions_within_instants()'s own comment), so "loop-carried
// inside the group" is a property of the CHANNEL, not of the writer
// node's region membership. Returns false (today's strict rule) for an
// invalid/out-of-range producer or origin -- a hand-built instruction
// stream with no real CouplingGraph behind it (this track's own Unit P
// tests) as much as a graph with no matching channel at all.
//
// Reads `effective_channel_lag()` (fuse/model/coupling_graph.hpp), not
// `Channel::lag` directly: the compiled graph is the single source of
// truth for which transfers are loop-carried, and `lower()` (src/ir/
// rules_channel.cpp step 4) applies that SAME function's shift to the
// arc this reads. A `WaveformIteration::Jacobi` window's transfers are
// therefore already loop-carried in the compiled graph by the time this
// runs, whatever `Channel::lag` the user wrote -- this function does not
// need to know about `WaveformIteration` at all, and does not.
[[nodiscard]] bool window_import_is_lagged(const CouplingGraph& g,
                                           const OperatorDAG& dag,
                                           ArcId imp) {
  if (!imp.valid() || imp.value() >= dag.arc_count()) return false;
  const NodeId producer = dag.arc(imp).producer;
  if (!producer.valid() || producer.value() >= dag.node_count()) return false;
  const ChannelId origin = dag.node(producer).origin;
  if (!origin.valid() || origin.value() >= g.channel_count()) return false;
  return effective_channel_lag(g, origin) != 0;
}

Result<void> check_window_causality(
    const CouplingGraph& g, const OperatorDAG& dag,
    const std::vector<Instruction>& instructions) {
  // Pass 1: EVERY (arc, dst_step) pair any instruction ever writes,
  // position independent -- what a LAGGED import's relaxed rule checks
  // against, since its writer's own re-execution (one sweep per
  // `StepContext::iteration`) can legitimately come after this read in
  // the one static pass the schedule describes.
  std::vector<std::vector<std::int64_t>> written_steps(dag.arc_count());
  for (const Instruction& ins : instructions) {
    if (ins.out.valid() && ins.out.value() < dag.arc_count()) {
      written_steps[ins.out.value()].push_back(ins.dst_step);
    }
  }
  const auto written_at = [&](ArcId a, std::int64_t step) {
    const std::vector<std::int64_t>& steps = written_steps[a.value()];
    return std::find(steps.begin(), steps.end(), step) != steps.end();
  };

  // Pass 2: most recent dst_step at which each arc was written so far --
  // the UNLAGGED rule, unchanged from before this track added laggedness.
  std::vector<std::int64_t> last_written_step(dag.arc_count(), -1);
  for (std::size_t i = 0; i < instructions.size(); ++i) {
    const Instruction& ins = instructions[i];
    if (ins.kind == OpKind::WindowAdvance) {
      const Node& n = dag.node(ins.origin_node);
      const ArcId imp = n.attrs.window_import;
      if (imp.valid()) {
        const bool lagged = window_import_is_lagged(g, dag, imp);
        const bool ok = imp.value() < dag.arc_count() &&
                       (lagged ? written_at(imp, ins.dst_step)
                               : last_written_step[imp.value()] == ins.dst_step);
        if (!ok) {
          return Failure{
              "E-SCHED-WINDOW-CAUSALITY",
              "instruction " + std::to_string(i) +
                  " (WindowAdvance \"" + n.label + "\" at t=" +
                  ins.time.to_string() + ", window instance " +
                  std::to_string(ins.dst_step) +
                  ") reads its import waveform (arc %" +
                  std::to_string(imp.value()) +
                  ") " + (lagged
                             ? "which is never written for this SAME window "
                               "instance anywhere in the scheduled stream"
                             : "before that arc's own writer for this SAME "
                               "window instance has run earlier in the "
                               "scheduled stream") +
                  ". The field transfer delivering a subcycling node's "
                  "import waveform must be ordered before it by TIME "
                  "(TickPhase) -- or, if loop-carried (Channel::lag != 0), "
                  "written for the same window instance ANYWHERE in the "
                  "group -- since it is deliberately not a real dependency "
                  "edge (OpAttrs::window_import's own doc comment); this is "
                  "either a scheduler defect in that ordering or a graph "
                  "that never delivers the waveform this window at all"};
        }
      }
    }
    if (ins.out.valid() && ins.out.value() < dag.arc_count()) {
      last_written_step[ins.out.value()] = ins.dst_step;
    }
  }
  return Result<void>::success();
}
}  // namespace detail

std::string ExecutionPlan::to_string() const {
  std::ostringstream os;
  // The host application's global locale reaches every default-constructed
  // stream. Under a locale with digit grouping this dump prints
  // "coupler_elements=16.384" instead of "16384", and any double would print
  // "0,7" -- a golden-file artifact must not depend on LC_NUMERIC. Same
  // reasoning as detail::format_double() in ir/dump.hpp.
  os.imbue(std::locale::classic());

  os << "# fuse-plan v1\n";
  os << "period=" << period_.to_string()
     << " lockstep=" << (lockstep_ ? "yes" : "no")
     << " instructions=" << instructions_.size()
     << " buffers=" << buffers_.buffers.size()
     << " coupler_elements=" << buffers_.total_coupler_elements
     << " regions=" << region_count_ << "\n";

  // Buffers: what the plan actually allocates, and what it must not.
  for (std::size_t i = 0; i < buffers_.buffers.size(); ++i) {
    const BufferDecl& b = buffers_.buffers[i];
    os << "buffer " << detail::pad(std::to_string(i), 3) << " %"
       << b.arc.value() << "  elements=" << b.elements
       << "  components=" << b.components
       << "  region=" << region_label(b.region)
       << "  owner=" << (b.is_component_owned ? "component" : "coupler")
       << "  persistent=" << (b.is_persistent ? "yes" : "no");
    if (!b.label.empty()) os << "  \"" << b.label << "\"";
    os << "\n";
  }

  for (std::size_t i = 0; i < instructions_.size(); ++i) {
    const Instruction& ins = instructions_[i];
    os << "  " << i << "  t=" << ins.time.to_string() << "  "
       << to_cstring(ins.kind) << "  node=" << ins.origin_node.value();
    if (ins.n_in > 0) {
      os << "  in=[";
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        if (k) os << ",";
        os << "%" << ins.in[k].value();
      }
      os << "]";
    }
    if (ins.out.valid()) os << "  out=%" << ins.out.value();
    const std::string attrs = detail::attrs_string(ins.kind, ins.attrs);
    if (!attrs.empty()) os << "  {" << attrs << "}";
    // Only on the instructions that carry it, like enter=/exit=: the common
    // phase is the silent one, so a plan with no deferred instruction dumps
    // exactly as it did before TickPhase existed. See Instruction::phase for
    // why this cannot simply be left out.
    if (ins.phase == TickPhase::AfterTicks) os << "  after-ticks";
    if (ins.region_enter >= 0) os << "  enter=" << ins.region_enter;
    if (ins.region_exit >= 0) os << "  exit=" << ins.region_exit;
    os << "\n";
  }
  return os.str();
}

namespace {

/// True when `rank` takes part in an instruction: it owns the source side, the
/// destination side, or the instruction names no ranks at all (a coupler-local
/// operator, which every participating rank runs on its own data).
bool rank_participates(const Instruction& ins, int rank, const CouplingGraph& g) {
  const auto in_set = [&g, rank](int component) {
    if (component < 0) return false;
    const auto id = static_cast<std::uint32_t>(component);
    if (id >= g.component_count()) return false;
    const RankSet& rs = g.component(ComponentId{id}).ranks;
    // Membership, not a range test: a placement may be strided or shared.
    return rs.contains(rank);
  };
  if (ins.attrs.src_rank_set < 0 && ins.attrs.dst_rank_set < 0) return true;
  return in_set(ins.attrs.src_rank_set) || in_set(ins.attrs.dst_rank_set);
}

}  // namespace

std::vector<std::string> epoch_sequence(const ExecutionPlan& plan,
                                        const CouplingGraph& g, int rank) {
  std::vector<std::string> seq;
  for (const Instruction& ins : plan.instructions()) {
    if (ins.kind != OpKind::Epoch) continue;
    if (!rank_participates(ins, rank, g)) continue;
    seq.push_back(ins.time.to_string() + ":" + ins.attrs.reason);
  }
  return seq;
}

ScheduleResult schedule(const CouplingGraph& g, const OperatorDAG& dag,
                        const ClockSet& clocks, const RegionSizes& sizes,
                        ExpandMode mode) {
  ScheduleResult result;

  // Time windows track: the legality gate for a window-scoped iteration
  // region (fuse::CouplingWindowScope, fuse/schedule/window.hpp) must be
  // called from schedule(), never from lower()/ir -- ir sits below
  // schedule (docs/dev/layering.md) and cannot call validate_coupling_
  // window() itself. Run before expand(), which also needs window_length()
  // for its own ticking but does not repeat this gate's POLICY/ORDER/
  // CAPACITY checks, so a misconfigured scope is reported with the precise
  // code this gate assigns it rather than expand()'s more generic
  // E-SCHED-EXPAND. `waveform_capacity` uses validate_coupling_window()'s
  // own default (4096): schedule()'s RegionSizes callback sizes ordinary
  // per-rank buffers, a different concept, and threading a dedicated
  // capacity value through schedule()'s own signature is deferred.
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    const Result<void> legal =
        validate_coupling_window(g, decl, *decl.window_scope);
    if (!legal.ok()) {
      result.diags.add(Severity::Error, legal.error().code,
                       legal.error().message);
      return result;
    }
  }

  const Result<TimedDag> timed = expand(g, dag, clocks, mode);
  if (!timed.ok()) {
    result.diags.add(Severity::Error, timed.error().code,
                     timed.error().message);
    return result;
  }
  const TimedDag& td = timed.value();

  // Spec §8.2: the hard gate. A graph that fails it produces no plan at all.
  const CycleReport cycles = prove_acyclic(g, dag, td);
  if (!cycles.acyclic) {
    result.diags.add(Severity::Error, "E-SCHED-CYCLE", cycles.explanation);
    return result;
  }
  // NOTE: `cycles.emission_order_valid` is deliberately *not* consulted. It is
  // known-unreliable pending a fix to the note_order() call inside
  // acyclicity.cpp's join_all(): a sweep of 26,647 generated graphs found ~8%
  // spurious violations on clean multirate schedules and zero real ones.
  // Ordering is derived below from the dependence relation instead, and
  // verified by check_dependence_order(), which reads only the SSA arcs.
  const Result<std::vector<std::uint32_t>> rank = topological_rank(dag);
  if (!rank.ok()) {
    result.diags.add(Severity::Error, rank.error().code, rank.error().message);
    return result;
  }

  // Every rank must reach the same agreement points, in the same order.
  //
  // This is checked here, once, at compile time, because the alternative is
  // finding out at run time: the ranks that expect one more agreement point
  // than their partners simply wait, and a wait with no timeout on a thousand
  // ranks is indistinguishable from slow physics. The proof is cheap because
  // the plan is a pure function of the graph -- each rank builds this same
  // instruction list -- so disagreement can only come from which instructions
  // a rank takes part in.
  //
  // Checked after the plan is built, at the bottom of this function.

  // Reserved multiscale operators. They exist as enumerators so their ordinals
  // do not move when they land, and they are refused here for the same reason
  // the multiscale contracts refuse (include/fuse/model/multiscale.hpp): an
  // opcode that scheduled as a no-op would let a micro-macro coupling compile
  // and run as though the closure had been applied, which is worse than not
  // having the opcode at all.
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if (n.kind == OpKind::Map || n.kind == OpKind::Reduce || n.kind == OpKind::Closure) {
      result.diags.add(Severity::Error, "E-SCHED-MULTISCALE",
                       std::string("operator '") + to_cstring(n.kind) +
                           "' is reserved for multiscale coupling and cannot be "
                           "scheduled yet",
                       n.label);
      return result;
    }
  }

  // Arity, checked once over the DAG rather than once per instance: a node
  // with four inputs instantiated a million times must not produce a million
  // identical diagnostics. Truncating to three would silently rewire the
  // operator, so this refuses the plan instead.
  bool arity_ok = true;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if (n.inputs.size() > kMaxInstructionInputs) {
      arity_ok = false;
      result.diags.add(Severity::Error, "E-SCHED-ARITY",
                       std::string{"operator "} + to_cstring(n.kind) +
                           " has " + std::to_string(n.inputs.size()) +
                           " inputs, more than the three an instruction holds;"
                           " lower Merge as a left-fold of pairwise merges",
                       n.label);
    }
  }
  if (!arity_ok) return result;

  // An empty RegionSizes, an invalid region, a components count below 1, an
  // arc id out of range, or an elements*components product that does not fit
  // size_t are all init-time setup errors and belong in diags. They are
  // OBTAINED as diagnostics rather than caught: routing them through an
  // exception made this conversion work only in a build with exceptions
  // enabled, and abort in one without.
  auto planned = try_plan_buffers(dag, td, sizes);
  if (!planned.ok()) {
    result.diags.add(Severity::Error, planned.error().code,
                     planned.error().message);
    return result;
  }
  const BufferPlan buffers = std::move(planned).value();
  const std::size_t n_buffers = buffers.buffers.size();

  std::vector<Instruction> instructions;
  instructions.reserve(td.nodes.size());
  for (const TimedNode& tn : td.nodes) {
    // dag.node() is .at()-checked and would throw; prove_acyclic already
    // rejects a TimedDag naming a node outside the DAG, so this is defensive.
    if (!tn.node.valid() || tn.node.value() >= dag.node_count()) {
      result.diags.add(Severity::Error, "E-SCHED-PLAN",
                       "the expansion names node " +
                           std::to_string(tn.node.value()) +
                           ", which is outside a DAG of " +
                           std::to_string(dag.node_count()) + " nodes");
      return result;
    }
    const Node& n = dag.node(tn.node);
    Instruction ins;
    ins.kind = n.kind;
    ins.out = n.output;
    ins.attrs = n.attrs;
    ins.time = tn.time;
    ins.src_step = tn.src_step;
    ins.dst_step = tn.dst_step;
    ins.phase = tn.phase;
    ins.origin_node = tn.node;
    ins.n_in = static_cast<std::uint8_t>(n.inputs.size());  // <= 3, checked
    for (std::uint8_t k = 0; k < ins.n_in; ++k) ins.in[k] = n.inputs[k];

    // plan_buffers() promises one BufferDecl per arc, in arc order, so every
    // operand must index it. Checked rather than trusted: an operand past the
    // end of the buffer table is a heap overread in the hot loop, where there
    // is no bounds check left.
    for (std::uint8_t k = 0; k < ins.n_in; ++k) {
      if (!ins.in[k].valid() || ins.in[k].value() >= n_buffers) {
        result.diags.add(Severity::Error, "E-SCHED-PLAN",
                         "operator " + std::string{to_cstring(n.kind)} +
                             " reads arc " + std::to_string(ins.in[k].value()) +
                             ", which has no buffer (the plan declares " +
                             std::to_string(n_buffers) + ")",
                         n.label);
        return result;
      }
    }
    if (ins.out.valid() && ins.out.value() >= n_buffers) {
      result.diags.add(Severity::Error, "E-SCHED-PLAN",
                       "operator " + std::string{to_cstring(n.kind)} +
                           " writes arc " + std::to_string(ins.out.value()) +
                           ", which has no buffer (the plan declares " +
                           std::to_string(n_buffers) + ")",
                       n.label);
      return result;
    }
    instructions.push_back(ins);
  }

  // Dependence order, not time order. `expand()` sorts by (time, phase,
  // NodeId); the key here replaces NodeId with the node's topological rank, so
  // an operand produced at the *same* time as its consumer still precedes it
  // whatever the node ids happen to be. Across different times nothing has to
  // be moved: every emission convention places a producer instance at or
  // before the time of the instance that reads it (a slab ends after the
  // source tick that feeds it; a target step ends after every Sample inside
  // it). stable_sort keeps expand()'s order for equal keys, so instances of
  // one node stay in ascending instance order and the dump stays reproducible
  // run to run.
  //
  // `phase` outranks the topological rank, and that is a real constraint, not
  // a cosmetic one: within one instant every delivery must precede the
  // component ticks that begin there and every `Sample` must follow them (see
  // TickPhase in expand.hpp), and the interpreter can only honour that if the
  // stream is already grouped that way. It cannot invert a dependence. The
  // only `AfterTicks` instances are `Sample`s, whose consumers -- the
  // `Accumulate` or `Interpolate` that closes a target step -- read source
  // steps whose `Sample` is stamped STRICTLY earlier, and whose first instance
  // of the period is at t=0 where no `BeforeTicks` chain instance exists.
  // `check_dependence_order()` below is what actually holds this to it.
  const std::vector<std::uint32_t>& rk = rank.value();
  std::stable_sort(instructions.begin(), instructions.end(),
                   [&rk](const Instruction& a, const Instruction& b) {
                     if (a.time != b.time) return a.time < b.time;
                     if (a.phase != b.phase) return a.phase < b.phase;
                     const std::uint32_t ra = rk[a.origin_node.value()];
                     const std::uint32_t rb = rk[b.origin_node.value()];
                     if (ra != rb) return ra < rb;
                     return a.origin_node < b.origin_node;
                   });

  // Spec §2 layer 3: iteration regions become explicit loop boundaries here,
  // not something the interpreter rediscovers from node membership. Computed
  // BEFORE the dependence check because the grouping below needs it, and the
  // grouping must be inside the window the check validates -- checking the
  // stream and then permuting it would validate an order nobody runs.
  const Result<std::vector<std::int32_t>> owner = region_membership(dag);
  if (!owner.ok()) {
    result.diags.add(Severity::Error, owner.error().code,
                     owner.error().message);
    return result;
  }

  group_regions_within_instants(dag, owner.value(), instructions);

  const Result<void> ordered = check_dependence_order(dag, instructions);
  if (!ordered.ok()) {
    result.diags.add(Severity::Error, ordered.error().code,
                     ordered.error().message);
    return result;
  }

  const Result<void> window_ordered =
      detail::check_window_causality(g, dag, instructions);
  if (!window_ordered.ok()) {
    result.diags.add(Severity::Error, window_ordered.error().code,
                     window_ordered.error().message);
    return result;
  }

  const Result<void> marked = mark_regions(dag, owner.value(), instructions);
  if (!marked.ok()) {
    result.diags.add(Severity::Error, marked.error().code,
                     marked.error().message);
    return result;
  }

  result.plan = ExecutionPlan{std::move(instructions), std::move(buffers),
                              td.period, td.lockstep, dag.region_count()};

  // Rank agreement, proved before the plan leaves the compiler.
  //
  // NOT "every rank reaches every agreement point": ranks legitimately do not.
  // Two independent couplings on disjoint ranks have two independent agreement
  // points, and forcing all four ranks through both would add synchronization
  // the physics never asked for -- the cost that makes people distrust a
  // coupler at scale.
  //
  // The property that actually prevents a hang is narrower: the ranks that
  // take part in a data move must be exactly the ranks that reach the
  // agreement point guarding it. If a rank moves data and then does not reach
  // the point, its partners wait for a report it never sends; if a rank
  // reaches the point without having moved data, it waits for partners who are
  // not coming. Both are hangs, and both are compile-time detectable here.
  for (std::size_t i = 0; i + 1 < result.plan.instructions().size(); ++i) {
    const Instruction& move = result.plan.instructions()[i];
    if (move.kind != OpKind::Transport) continue;

    const Instruction* guard = nullptr;
    for (std::size_t k = i + 1; k < result.plan.instructions().size(); ++k) {
      const Instruction& candidate = result.plan.instructions()[k];
      if (candidate.kind != OpKind::Epoch) continue;
      if (candidate.n_in == 0 || candidate.in[0] != move.out) continue;
      guard = &candidate;
      break;
    }
    if (guard == nullptr) {
      result.diags.add(Severity::Error, "E-SCHED-EPOCH-MISSING",
                       "a data move at t=" + move.time.to_string() +
                           " is not followed by a rank-agreement point, so a "
                           "failure on one of its ranks would be discovered by "
                           "the others only when they block on the next "
                           "collective");
      return result;
    }
    if (guard->attrs.src_rank_set != move.attrs.src_rank_set ||
        guard->attrs.dst_rank_set != move.attrs.dst_rank_set) {
      result.diags.add(
          Severity::Error, "E-SCHED-EPOCH-PARTIAL",
          "the rank-agreement point after the data move at t=" +
              move.time.to_string() +
              " covers a different set of ranks than the move itself (move " +
              std::to_string(move.attrs.src_rank_set) + "->" +
              std::to_string(move.attrs.dst_rank_set) + ", agreement " +
              std::to_string(guard->attrs.src_rank_set) + "->" +
              std::to_string(guard->attrs.dst_rank_set) +
              "); some rank would wait for a partner that never arrives");
      return result;
    }
  }

  return result;
}

}  // namespace fuse
