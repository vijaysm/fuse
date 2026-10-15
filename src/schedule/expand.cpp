#include "fuse/schedule/expand.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/schedule/slab.hpp"
#include "fuse/schedule/window.hpp"

namespace fuse {
namespace {

using detail::Int128;

[[nodiscard]] Int128 abs128(Int128 v) { return v < 0 ? -v : v; }

bool any_spacetime(const CouplingGraph& g) {
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    if (g.channel_at(ChannelId{i}).spacetime.basis != TemporalBasis::None) {
      return true;
    }
  }
  return false;
}

/// Does this node run on the source clock rather than the target clock?
/// Sample is the only interior op that does: accumulation samples every source
/// tick and reduces once per target tick.
bool runs_on_source_clock(OpKind k) { return k == OpKind::Sample; }

/// Does this channel carry a `Sample`, i.e. does its chain straddle two clocks
/// rather than lying on the slab decomposition?
///
/// Two constructions do. Accumulation is the original one. A SPACE-TIME
/// channel (spec Section 7.3) is the second and was added by Task 6.3: the temporal
/// operator `T` of `M = T (x) W` has one coefficient per (target step, source
/// step) pair over the whole window, and its column index runs over EVERY
/// source step -- including, for any reconstruction degree above zero, source
/// steps that do not overlap the target step being written. Only a node on the
/// source clock sees them all, so lowering gives a space-time channel a
/// `Sample` for exactly the reason an accumulating one has one, and everything
/// downstream of it closes a target step.
///
/// Laying the chain on the slabs instead would be incomplete rather than
/// merely inconvenient: a slab lies inside one target step by construction, so
/// a slab-laid chain can never reach the source steps outside it that `T`'s
/// row names. (It would once also have recorded the WRONG source step at a
/// slab that ends exactly on a source boundary, because such a slab ran after
/// the tick beginning there. `TickPhase` fixed that half -- see runner.cpp --
/// but the reach problem above is structural and is why `Sample` is here.)
bool has_sample(const Channel& ch) {
  return ch.accumulate.op != AccumOp::Instant ||
         ch.spacetime.basis != TemporalBasis::None;
}

/// k * interval, formed and range-checked in __int128.
///
/// Rational::operator* throws std::logic_error when the reduced product does
/// not fit int64_t, and that happens for legal inputs: with intervals P/4 and
/// P (P = 8999999999999999999) the common period is P, tick 3 of the finer
/// clock is 3P/4, and 3P > 2^63. expand() returns a Result, so the condition
/// has to be detected rather than thrown. Same contract, and the same shape,
/// as scaled_tick() in slab.cpp.
[[nodiscard]] Result<Rational> scaled_tick(std::int64_t k,
                                           const Rational& interval) {
  const Int128 n = static_cast<Int128>(k) * interval.num();
  const Int128 d = interval.den();  // always >= 1
  const Int128 g = detail::gcd128(abs128(n), d);
  const Int128 rn = n / g;
  if (!detail::fits_i64(rn)) {
    return Failure{"E-SCHED-PERIOD",
                 "tick " + std::to_string(k) + " of interval " +
                     interval.to_string() +
                     " is not representable as an exact 64-bit Rational time"};
  }
  return Rational{static_cast<std::int64_t>(rn),
                  static_cast<std::int64_t>(d / g)};
}

/// period / interval as an exact tick count, in __int128 so that neither the
/// division nor the check can throw out of Rational.
[[nodiscard]] Result<std::int64_t> tick_count(const Rational& period,
                                              const Rational& interval,
                                              const char* what) {
  if (!(interval > Rational{0})) {
    return Failure{"E-SCHED-INTERVAL", std::string(what) + " interval " +
                                         interval.to_string() +
                                         " must be positive"};
  }
  const Int128 num = static_cast<Int128>(period.num()) * interval.den();
  const Int128 den = static_cast<Int128>(period.den()) * interval.num();
  // den > 0: denominators are positive and interval.num() > 0.
  if (num <= 0 || num % den != 0) {
    return Failure{"E-SCHED-PERIOD",
                 std::string(what) + " interval " + interval.to_string() +
                     " does not divide the common period " +
                     period.to_string() + " an integer number of times"};
  }
  const Int128 n = num / den;
  if (n > kMaxPeriodTicks) {
    return Failure{"E-SCHED-PERIOD",
                 std::string(what) + " interval " + interval.to_string() +
                     " needs more than " + std::to_string(kMaxPeriodTicks) +
                     " ticks per period " + period.to_string()};
  }
  return static_cast<std::int64_t>(n);
}

/// floor(a / b) and ceil(a / b) for a >= 0, b > 0, in __int128 so that the
/// products the callers feed in (up to kMaxPeriodTicks^2 == 1e12) cannot
/// overflow. Both results are step indices, so they fit int64_t comfortably.
[[nodiscard]] std::int64_t floor_div(Int128 a, Int128 b) {
  return static_cast<std::int64_t>(a / b);
}
[[nodiscard]] std::int64_t ceil_div(Int128 a, Int128 b) {
  return static_cast<std::int64_t>((a + b - 1) / b);
}

[[nodiscard]] Failure explode_error() {
  return Failure{"E-SCHED-EXPLODE",
               "time expansion exceeds " + std::to_string(kMaxTimedNodes) +
                   " timed nodes; check component intervals"};
}

/// Does a channel of `g` own this node's instances? Shared by both expansion
/// paths: which nodes the channel loop claims and which fall through to the
/// boundary rule must be one decision, not two that can drift.
[[nodiscard]] bool claimed_by_channel(const CouplingGraph& g, const Node& n) {
  return n.origin.valid() && n.origin.value() < g.channel_count();
}

/// Time windows track: the window-scoped IterationRegionDecl that declares
/// `cid` among its channels, or nullptr if none does. Every node whose
/// origin is such a channel -- the WindowAdvance node finish_window_regions()
/// (src/ir/window_lowering.cpp) built from it, that channel's own ordinary
/// Transport/Remap/Epoch/Sink chain, AND the Residual build_iteration_
/// regions() (src/ir/regions.cpp) still builds for it unconditionally --
/// ticks at the window's own length instead of that channel's ordinary
/// source/target clock.
[[nodiscard]] const IterationRegionDecl* owning_window_region(
    const CouplingGraph& g, ChannelId cid) {
  for (std::uint32_t r = 0; r < g.iteration_region_count(); ++r) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{r});
    if (!decl.window_scope.has_value()) continue;
    for (const ChannelId cid2 : decl.channels) {
      if (cid2.value() == cid.value()) return &decl;
    }
  }
  return nullptr;
}

/// The window-scoped IterationRegionDecl that lists `owner` among
/// `window_region_members()`, or nullptr if none does. A member of a time
/// window is stepped ONLY by its own OpKind::WindowAdvance instruction
/// (Runner::Impl::build_ticks()'s own window_member_components() exclusion,
/// src/runtime/runner.cpp: such a member gets no ordinary per-native-
/// interval Tick at run time, ever). The boundary rule below (every node NO
/// channel claims -- a component's own Source/Sink/etc., unrelated to any
/// window channel's chain) must give that member NO INSTRUCTIONS at its own
/// native interval either, or the compiled stream disagrees with what the
/// runtime actually does: mark_regions() (src/schedule/scheduler.cpp) would
/// see extra, distinct-looking instructions inside what should be one
/// window instance's own contiguous run, and refuse the whole region
/// (E-SCHED-REGION) even though nothing would actually run twice. This is
/// the graph-level equivalent of `owning_window_region()` above, keyed by
/// component rather than channel, because a node this rule places may have
/// no channel origin at all.
[[nodiscard]] const IterationRegionDecl* owning_window_region_for_component(
    const CouplingGraph& g, ComponentId owner) {
  for (std::uint32_t r = 0; r < g.iteration_region_count(); ++r) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{r});
    if (!decl.window_scope.has_value()) continue;
    for (const ComponentId member : window_region_members(g, decl)) {
      if (member.value() == owner.value()) return &decl;
    }
  }
  return nullptr;
}

/// The two components a channel connects, or the error the graph earns for
/// having an endpoint no component owns.
[[nodiscard]] Result<std::pair<ComponentId, ComponentId>> channel_endpoints(
    const CouplingGraph& g, const Channel& ch) {
  const ComponentId src_c = g.owner_of(ch.source);
  const ComponentId dst_c = g.owner_of(ch.target);
  if (!src_c.valid() || src_c.value() >= g.component_count() ||
      !dst_c.valid() || dst_c.value() >= g.component_count()) {
    return Failure{"E-SCHED-EXPAND",
                 "channel '" + ch.name + "' has an unowned endpoint"};
  }
  return std::pair<ComponentId, ComponentId>{src_c, dst_c};
}

/// The component whose clock a node no channel claimed runs on: the rank set
/// its value is placed on. The arc type records which rank set (component) the
/// value lives on.
///
/// Dropping such a node would silently shorten the schedule, which is worse
/// than refusing to build one, so an unplaced value is an error. Shared by
/// both expansion paths so that the set of schedulable graphs -- and the
/// diagnostic an unschedulable one gets -- cannot depend on which path ran.
[[nodiscard]] Result<ComponentId> boundary_owner(const CouplingGraph& g,
                                                 const OperatorDAG& dag,
                                                 std::uint32_t i,
                                                 const Node& n) {
  const ArcId ref = n.output.valid()
                        ? n.output
                        : (n.inputs.empty() ? ArcId{} : n.inputs.front());
  const int rank_set = ref.valid() ? dag.arc(ref).type.rank_set : -1;
  if (rank_set < 0 ||
      static_cast<std::uint32_t>(rank_set) >= g.component_count()) {
    return Failure{"E-SCHED-EXPAND",
                 "node " + std::to_string(i) + " \"" + n.label +
                     "\" belongs to no channel and its value is not placed "
                     "on any component, so it cannot be scheduled"};
  }
  return ComponentId{static_cast<std::uint32_t>(rank_set)};
}

/// The `TickPhase` a Merge's input is attributed at the merge instant
/// `(dst_step + 1) * dst_dt` -- the same causality contract the per-channel
/// loop below applies to that writer's own tail node, re-derived here because
/// a Merge has no single origin channel to look the answer up on (review
/// 2026-10-11 IS-1: the boundary pass previously stamped every Merge
/// `BeforeTicks` unconditionally, so a source-limited writer whose tail is
/// still `AfterTicks` at this instant sorted AFTER a Merge that read it,
/// failing `check_dependence_order()` with `E-SCHED-ORDER` on a legal graph).
///
/// `input` is one of the Merge node's `inputs` arcs. Three shapes reach here:
///   - a window-closing chain (accumulation or space-time): always
///     `BeforeTicks`, per the causality contract in expand.hpp;
///   - an ordinary (Instant) channel chain: re-run its slab decomposition at
///     `dst_dt` and read the phase of the slab that ends at this instant,
///     exactly as the per-channel loop computed it for the writer's own tail;
///   - another `Merge` (the left-fold chain lowering.cpp builds for more than
///     three writers): recurse, since every node in that fold shares the
///     combined field's destination component and therefore the same
///     `dst_dt` and the same `dst_step` numbering.
[[nodiscard]] Result<TickPhase> merge_writer_phase_at(
    const CouplingGraph& g, const OperatorDAG& dag, const ClockSet& clocks,
    const Rational& period, ArcId input, const Rational& dst_dt,
    std::int64_t dst_step) {
  const NodeId producer = dag.arc(input).producer;
  const Node& pn = dag.node(producer);

  if (pn.kind == OpKind::Merge) {
    TickPhase worst = TickPhase::BeforeTicks;
    for (const ArcId in2 : pn.inputs) {
      const Result<TickPhase> p =
          merge_writer_phase_at(g, dag, clocks, period, in2, dst_dt, dst_step);
      if (!p.ok()) return p.error();
      worst = std::max(worst, p.value());
    }
    return worst;
  }

  if (!claimed_by_channel(g, pn)) {
    // No channel owns this writer (e.g. a Source feeding the merge
    // directly, or a constant) -- its value is current at every instant, so
    // it is never the reason a delivery has to wait.
    return TickPhase::BeforeTicks;
  }

  const ChannelId origin{pn.origin.value()};
  const Channel& ch = g.channel_at(origin);
  if (has_sample(ch)) {
    return TickPhase::BeforeTicks;
  }

  const Result<std::pair<ComponentId, ComponentId>> ends =
      channel_endpoints(g, ch);
  if (!ends.ok()) return ends.error();
  Rational src_dt = clocks.clock(ends.value().first).interval();
  const IterationRegionDecl* owning_window = owning_window_region(g, origin);
  if (owning_window != nullptr) {
    const Result<Rational> wl =
        window_length(g, *owning_window->window_scope);
    if (!wl.ok()) return wl.error();
    src_dt = wl.value();
  }

  const Result<SlabDecomposition> d = decompose(src_dt, dst_dt, period);
  if (!d.ok()) return d.error();
  const Result<Rational> boundary = scaled_tick(dst_step + 1, dst_dt);
  if (!boundary.ok()) return boundary.error();
  for (const Slab& s : d.value().slabs) {
    if (s.dst_step != dst_step || s.end != boundary.value()) continue;
    const Result<Rational> complete = scaled_tick(s.src_step + 1, src_dt);
    if (!complete.ok()) return complete.error();
    return s.end < complete.value() ? TickPhase::AfterTicks
                                    : TickPhase::BeforeTicks;
  }
  // No slab of this writer ends exactly at this merge instant: its interval
  // does not divide the merge's, which tick_count() above already refused
  // for the merge's own owner clock -- unreachable for a graph that reached
  // this point.
  return TickPhase::BeforeTicks;
}

/// The order every expansion finishes in: by time, then by `TickPhase`, ties
/// by NodeId. One definition, used by both paths, so the fast path cannot be
/// sorted differently from the general one.
///
/// The phase sits between the two for the reason given in expand.hpp: within
/// one instant every delivery has to precede the component ticks that begin
/// there, and every `Sample` has to follow them. NodeId alone cannot express
/// that, because the two kinds of instance belong to the same channel chain
/// and their ids interleave.
[[nodiscard]] bool by_time_then_node(const TimedNode& a, const TimedNode& b) {
  if (a.time != b.time) return a.time < b.time;
  if (a.phase != b.phase) return a.phase < b.phase;
  return a.node < b.node;
}

/// Spec Section 8.2's lockstep shortcut: expand a whole period without decomposing.
///
/// Precondition (the caller's `out.lockstep`): every component shares one
/// interval and no channel is space-time. The common period is then that one
/// interval, every clock ticks exactly once in it, and `decompose(dt, dt, dt)`
/// is the single slab [0, dt). The general path's three emission conventions
/// therefore collapse to two constant times:
///
///   * a node a channel claims runs at the end of that channel's only slab,
///     `t = period`; the one exception is the `Sample` of an accumulating
///     channel, which runs on the *source* clock at the start of its only
///     tick, `t = 0`, and therefore carries `TickPhase::AfterTicks`;
///   * a `Merge` -- claimed by no single channel, since it combines two or
///     more -- also runs at `t = period`, the same instant as the
///     Epoch/Transport/Remap outputs it reads (N-ir-schedule-2);
///   * every other node no channel claimed (`Source`, an unwritten `Sink`)
///     runs at its component's only tick, `t = 0`.
///
/// Every step index is 0, because there is only step 0 of either clock, and
/// every instance index is 0, because nothing repeats.
///
/// This is not an approximation of the general path's answer, it *is* the
/// general path's answer; `ExpandMode::ForceGeneral` and the byte-for-byte
/// comparison in test/unit/lockstep_test.cpp are what hold it to that.
[[nodiscard]] Result<void> emit_lockstep(const CouplingGraph& g,
                                         const OperatorDAG& dag,
                                         TimedDag& out) {
  // The general path validates every channel's endpoints -- and rejects the
  // graph if one is unowned -- before it emits anything for that channel.
  // The shortcut must reject the same graphs with the same message, or which
  // graphs are schedulable would depend on an optimisation.
  for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
    const Result<std::pair<ComponentId, ComponentId>> ends =
        channel_endpoints(g, g.channel_at(ChannelId{c}));
    if (!ends.ok()) return ends.error();
  }

  // One instance per node, so the general path's per-emission cap reduces to
  // this single check.
  if (static_cast<std::int64_t>(dag.node_count()) > kMaxTimedNodes) {
    return explode_error();
  }

  out.nodes.reserve(dag.node_count());
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node& n = dag.node(nid);
    Rational t{0};
    TickPhase phase = TickPhase::BeforeTicks;
    if (claimed_by_channel(g, n)) {
      const Channel& ch = g.channel_at(ChannelId{n.origin.value()});
      if (has_sample(ch) && runs_on_source_clock(n.kind)) {
        t = Rational{0};
        phase = TickPhase::AfterTicks;
      } else {
        t = out.period;
      }
    } else if (n.kind == OpKind::Merge) {
      // A Merge combining two or more channels has no single owning channel
      // (fuse/ir/lowering.cpp gives it ChannelId{} on purpose -- it is not
      // any one of them), but it is a chain-CLOSING node like the rest of
      // those channels' own non-Sample tails, not a component-tick-opening
      // one. Falling into the `boundary_owner()` branch below placed it at
      // its component's tick START, sorting it before the Epoch/Transport
      // outputs it reads and failing the scheduler's own dependence check --
      // REVIEW.md N-ir-schedule-2.
      t = out.period;
    } else {
      const Result<ComponentId> owner = boundary_owner(g, dag, i, n);
      if (!owner.ok()) return owner.error();
    }
    out.nodes.push_back(TimedNode{nid, t, 0, 0, 0, phase});
  }
  return Result<void>::success();
}

}  // namespace

Result<TimedDag> expand(const CouplingGraph& g, const OperatorDAG& dag,
                        const ClockSet& clocks, ExpandMode mode) {
  // clocks.clock() is .at()-checked and throws std::out_of_range on a
  // mismatch; expand() returns a Result, so the mismatch is rejected here
  // instead of escaping as an exception.
  if (clocks.size() != g.component_count()) {
    return Failure{"E-SCHED-EXPAND",
                 "clock set has " + std::to_string(clocks.size()) +
                     " clocks but the graph has " +
                     std::to_string(g.component_count()) + " components"};
  }

  // checked_common_period(), not common_period(): the latter throws
  // std::logic_error when the LCM overflows (two coprime intervals near 2^31.5
  // are enough), and it does not bound the number of ticks in the period.
  const Result<Rational> period = clocks.checked_common_period();
  if (!period.ok()) return period.error();

  // Accumulation and a space-time projection each give the channel a `Sample`
  // on the source clock, and a channel cannot have two: the second would read
  // a value the first chain only produces at the END of a target step, so the
  // emitted stream would fail `check_dependence_order()` with
  // `E-SCHED-ORDER` -- a message that blames the scheduler for what is a
  // conflict between two temporal operators the user asked for on one channel.
  // Refused here, where the reason is still visible.
  //
  // Not refused in `lower()`: the DAG for such a channel is well formed (it is
  // the *schedule* that does not exist), and `fuse_initialize()`
  // (`impl_initialize`, src/abi/participant.cpp -- review 2026-10-11 IS-11,
  // correcting a stale `soma_initialize` citation) lowers without
  // scheduling, so rejecting it there would make a graph unloadable for a
  // reason that only shows up one layer down.
  for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
    const Channel& ch = g.channel_at(ChannelId{c});
    if (ch.accumulate.op == AccumOp::Instant) continue;
    if (ch.spacetime.basis == TemporalBasis::None) continue;
    return Failure{
        "E-SCHED-SPACETIME",
        "channel '" + ch.name + "' asks for both accumulate(" +
            to_cstring(ch.accumulate.op) + ") and the space-time basis " +
            to_cstring(ch.spacetime.basis) +
            ". Those are two different temporal operators on the same data and "
            "each needs its own sampler on the source clock, which cannot be "
            "scheduled; drop one. A degree-0 space-time projection onto a "
            "single target step already IS the window mean"};
  }

  TimedDag out;
  out.period = period.value();
  out.lockstep = clocks.is_lockstep() && !any_spacetime(g);

  std::unordered_map<std::uint32_t, std::int64_t> instance_of;  // NodeId -> count

  // Returns false when the emission would push the total past kMaxTimedNodes.
  // The cap is enforced on every emission, not once per node: a single node
  // can expand to kMaxPeriodTicks instances, so a per-node check lets the
  // vector overshoot by that much before anyone looks.
  const auto emit = [&](NodeId n, const Rational& t, std::int64_t src_step,
                        std::int64_t dst_step,
                        TickPhase phase = TickPhase::BeforeTicks) -> bool {
    if (static_cast<std::int64_t>(out.nodes.size()) >= kMaxTimedNodes) {
      return false;
    }
    const std::int64_t inst = instance_of[n.value()]++;
    out.nodes.push_back(TimedNode{n, t, src_step, dst_step, inst, phase});
    return true;
  };

  // Spec Section 8.2: lockstep is the degenerate case and must not pay for the slab
  // machinery. One instance per node, `decompose()` never called. The result
  // is sorted by the same comparator as the general path, below, and is
  // required to be identical to it -- see emit_lockstep().
  if (out.lockstep && mode == ExpandMode::Auto) {
    const Result<void> fast = emit_lockstep(g, dag, out);
    if (!fast.ok()) return fast.error();
    std::stable_sort(out.nodes.begin(), out.nodes.end(), by_time_then_node);
    return out;
  }

  // --- channel chains -------------------------------------------------------
  for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
    const ChannelId cid{c};
    const Channel& ch = g.channel_at(cid);
    const Result<std::pair<ComponentId, ComponentId>> ends =
        channel_endpoints(g, ch);
    if (!ends.ok()) return ends.error();
    // Intervals come from the ClockSet, not from ComponentDecl: Clock's
    // constructor is the thing that guarantees they are positive, and the
    // ClockSet is what the runtime will actually advance.
    //
    // Time windows track: a window-scoped region's own channel carries
    // the WHOLE window's waveform as one value (K samples as K field
    // components, the user's chosen design) -- it changes once per
    // window, not once per either endpoint's own native tick, so BOTH
    // ends of its slab decomposition are overridden to the window's own
    // length instead. This is the ENTIRE accommodation this channel's
    // chain needs: with src_dt == dst_dt, decompose() lays out exactly
    // one slab per window instance, and every downstream node (Transport,
    // Remap, Epoch, Sink -- ordinary nodes, unaware any of this
    // happened) is emitted by the SAME per-channel loop below as any
    // other channel's chain, at that slab's end.
    const IterationRegionDecl* owning_window = owning_window_region(g, cid);
    Rational src_dt = clocks.clock(ends.value().first).interval();
    Rational dst_dt = clocks.clock(ends.value().second).interval();
    if (owning_window != nullptr) {
      const Result<Rational> wl =
          window_length(g, *owning_window->window_scope);
      if (!wl.ok()) return wl.error();
      src_dt = wl.value();
      dst_dt = wl.value();
    }

    const Result<std::int64_t> ns = tick_count(out.period, src_dt, "source");
    if (!ns.ok()) return ns.error();
    const Result<std::int64_t> nd = tick_count(out.period, dst_dt, "target");
    if (!nd.ok()) return nd.error();

    const bool sampled = has_sample(ch);

    // Only a chain without a Sample is laid out on slabs, so only it pays
    // for (and can be rejected by) the decomposition.
    SlabDecomposition decomp;
    // TickPhase's causality contract, resolved once per channel rather than
    // once per (channel, node): an Instant chain carries the source state at
    // `(src_step + 1) * src_dt`, so the slab may be consumed by the target
    // step beginning at its end only when the slab ends where its source step
    // does. A slab ends at `min(source step end, target step end)`, so the
    // test is an equality; it is written as a comparison because that is what
    // the contract says and the reader should not have to re-derive the min.
    std::vector<TickPhase> slab_phase;
    if (!sampled) {
      Result<SlabDecomposition> d = decompose(src_dt, dst_dt, out.period);
      if (!d.ok()) return d.error();
      decomp = std::move(d.value());
      slab_phase.reserve(decomp.slabs.size());
      for (const Slab& s : decomp.slabs) {
        const Result<Rational> complete = scaled_tick(s.src_step + 1, src_dt);
        if (!complete.ok()) return complete.error();
        slab_phase.push_back(s.end < complete.value()
                                 ? TickPhase::AfterTicks
                                 : TickPhase::BeforeTicks);
      }
    }

    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const NodeId nid{i};
      // dag is const throughout, so this reference cannot be invalidated;
      // out.nodes is the only vector that grows and nothing points into it.
      const Node& n = dag.node(nid);
      if (n.origin != cid) continue;
      // Time windows track: a WindowAdvance node (finish_window_regions(),
      // src/ir/window_lowering.cpp) has this SAME channel as its origin
      // too, and flows through this loop exactly like the channel's own
      // Transport/Remap/Epoch/Sink chain -- it has no Sample and no
      // accumulation of its own, so (like them) it lands in the "no
      // accumulation: one instance per slab" branch below, at that
      // slab's end. No special case needed: `src_dt`/`dst_dt`'s window-
      // length override above already makes that slab the window
      // instance, which is the only thing that made this node need
      // different treatment in the first place.

      if (sampled && runs_on_source_clock(n.kind)) {
        // One Sample per source tick, inside whichever target step it falls.
        // Stamped at the tick's START and marked `AfterTicks`, which together
        // mean "the value that tick produces": the source step's own value,
        // available only once the source has advanced over it. Stamping it at
        // the tick's END instead would read the same value, but it would also
        // redefine which source steps a target step closes over -- the window
        // would shrink to the steps that have COMPLETED by the target step's
        // end, which is a different (and, for space-time, strictly weaker)
        // operator. See `src_step` below.
        for (std::int64_t k = 0; k < ns.value(); ++k) {
          const Result<Rational> t = scaled_tick(k, src_dt);
          if (!t.ok()) return t.error();
          // t / dst_dt == k * nd / ns exactly, done in integers.
          const std::int64_t dst_step =
              floor_div(static_cast<Int128>(k) * nd.value(), ns.value());
          if (!emit(nid, t.value(), k, dst_step, TickPhase::AfterTicks)) {
            return explode_error();
          }
        }
      } else if (sampled) {
        // The reduction and everything after it close a target step, so they
        // run at its end and carry the last source step that contributed:
        // the largest j with j*src_dt < (k+1)*dst_dt.
        //
        // `TickPhase::BeforeTicks` (the default) is the load-bearing half of
        // this stamp, not a detail, and unlike a slab instance this one always
        // earns it: a window-closing chain carries a quantity attributed to
        // target step k, whose end IS `(k+1)*dst_dt`, so it satisfies the
        // causality contract in expand.hpp unconditionally. That instant is
        // also where the target's step k+1 BEGINS, and the value exists
        // precisely so that step k+1 can consume it. Emitting it after that
        // tick -- which is what a bare timestamp used to mean -- delayed every
        // delivery by two target steps and dropped one write per period
        // unread, because the next write overtook it. That was R2;
        // `SpacetimeRunner.EveryTargetStepProjectionReachesTheTargetComponent`
        // in test/unit/spacetime_conservation_test.cpp fails if it comes back.
        //
        // A `Sample` at this same instant is `AfterTicks` and therefore sorts
        // after this chain, which is correct and not merely tolerated: this
        // chain closes over source steps j with j*src_dt STRICTLY less than
        // `(k+1)*dst_dt`, so it never depends on a `Sample` stamped here.
        for (std::int64_t k = 0; k < nd.value(); ++k) {
          const Result<Rational> t = scaled_tick(k + 1, dst_dt);
          if (!t.ok()) return t.error();
          const std::int64_t src_step =
              ceil_div(static_cast<Int128>(k + 1) * ns.value(), nd.value()) - 1;
          if (!emit(nid, t.value(), src_step, k)) return explode_error();
        }
      } else {
        // No accumulation: one instance per slab, at the slab's end, with the
        // phase the causality contract gave it above. A target-limited slab is
        // `AfterTicks`: its source step is still running at the stamp, so the
        // state it carries is not valid until after the target step beginning
        // there has already started. The next slab redelivers the same
        // `src_step` unchanged, so nothing is lost by waiting -- what would be
        // lost by not waiting is the "explicit, lagged" character of the
        // scheme, silently.
        for (std::size_t si = 0; si < decomp.slabs.size(); ++si) {
          const Slab& s = decomp.slabs[si];
          if (!emit(nid, s.end, s.src_step, s.dst_step, slab_phase[si])) {
            return explode_error();
          }
        }
      }
    }
  }

  // --- everything no channel claimed: one instance per owning component tick.
  // Source always lands here (lowering gives it no origin); so does a Sink for
  // a Bidirectional field that no channel writes. A Sink that *was* written is
  // part of its channel's chain above and must not be emitted twice. A Merge
  // combining two or more channels also has no single origin and lands here,
  // but -- unlike Source/Sink -- it closes its destination component's tick
  // rather than opening it (N-ir-schedule-2, see the loop body below).
  //
  // `BeforeTicks`, like every non-`Sample` instance. `Source` moves no data --
  // its arc already aliases the component's own storage (see
  // `Runner::Impl::execute`) -- so which side of its component's tick it sits
  // on is not observable; what would be observable is the reverse choice,
  // because a `Source` stamped at the same instant as a delivery sorts first
  // (it is a DAG root) and would drag the receiving component's tick forward
  // past the delivery. That is R2 again, reached from the boundary pass.
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node& n = dag.node(nid);
    if (claimed_by_channel(g, n)) continue;

    const Result<ComponentId> owner = boundary_owner(g, dag, i, n);
    if (!owner.ok()) return owner.error();
    Rational dt = clocks.clock(owner.value()).interval();
    // Time windows track: `owner` is stepped ONLY by its own WindowAdvance
    // instruction (build_ticks()'s own window_member_components()
    // exclusion, src/runtime/runner.cpp -- no ordinary per-native-interval
    // Tick ever fires for it), so this boundary-rule node -- unrelated to
    // any window channel's own chain, which already gets the window-
    // length override in the channel loop above -- must not tick at
    // `owner`'s own native interval either: doing so would place extra,
    // distinct-looking instructions inside what should be one window
    // instance's own contiguous run, and mark_regions() (scheduler.cpp)
    // would refuse the region even though nothing would actually run
    // twice at run time.
    const IterationRegionDecl* owning_window =
        owning_window_region_for_component(g, owner.value());
    if (owning_window != nullptr) {
      const Result<Rational> wl = window_length(g, *owning_window->window_scope);
      if (!wl.ok()) return wl.error();
      dt = wl.value();
    }

    const Result<std::int64_t> nk = tick_count(out.period, dt, "component");
    if (!nk.ok()) return nk.error();

    // A Merge combining two or more channels lands here too -- it has no
    // single owning channel by construction (see the twin comment in
    // emit_lockstep()) -- but it CLOSES its destination component's tick
    // rather than opening it, so it is stamped at each tick's END,
    // `(k+1)*dt`, the same instant a channel's own non-Sample tail lands on
    // for that tick. Source and an unwritten Sink genuinely open the tick,
    // `k*dt`, which is the one case this loop was written for.
    // REVIEW.md N-ir-schedule-2: placing Merge at the tick's start sorted it
    // before the Epoch/Transport outputs it reads.
    //
    // Review 2026-10-11 IS-1: the tick's END is one instant, but which SIDE
    // of it the Merge sits on is not fixed at `BeforeTicks` the way Source's
    // is. A multirate writer whose own tail is target-limited at this same
    // instant is `AfterTicks` (its source step has not finished), and the
    // Merge reads that writer, so it cannot be ready any earlier than the
    // writer it depends on: its phase is the LATEST (max) of its writers'
    // phases at this instant, derived by `merge_writer_phase_at()`.
    const bool tick_closing = (n.kind == OpKind::Merge);
    for (std::int64_t k = 0; k < nk.value(); ++k) {
      const Result<Rational> t = scaled_tick(tick_closing ? k + 1 : k, dt);
      if (!t.ok()) return t.error();
      TickPhase phase = TickPhase::BeforeTicks;
      if (n.kind == OpKind::Merge) {
        for (const ArcId in : n.inputs) {
          const Result<TickPhase> p =
              merge_writer_phase_at(g, dag, clocks, out.period, in, dt, k);
          if (!p.ok()) return p.error();
          phase = std::max(phase, p.value());
        }
      }
      if (!emit(nid, t.value(), k, k, phase)) return explode_error();
    }
  }

  std::stable_sort(out.nodes.begin(), out.nodes.end(), by_time_then_node);
  return out;
}

}  // namespace fuse
