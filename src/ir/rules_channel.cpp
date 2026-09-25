#include <string>

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"
#include "fuse/model/connect.hpp"
#include "fuse/model/remap_kind.hpp"
#include "fuse/model/transform.hpp"

namespace fuse {
namespace {

/// Emit one node and return its output arc, whose type is `out_type`.
ArcId emit(LoweringContext& ctx, OpKind kind, ArcId input, ValueType out_type,
           OpAttrs attrs, ChannelId origin, const std::string& label) {
  const ArcId out = ctx.dag.add_arc(std::move(out_type), label);
  ctx.dag.add_node(kind, {input}, out, std::move(attrs), origin, label);
  return out;
}

bool lower_channel(LoweringContext& ctx, ChannelId cid) {
  const CouplingGraph& g = ctx.graph;
  if (!cid.valid() || cid.value() >= g.channel_count()) return false;

  const Channel& ch = g.channel_at(cid);
  const std::string base = ch.name;

  // `source_arc`, NOT `field_arc`. A channel reads what the source field's own
  // component produces. For an Intent::Bidirectional source field `field_arc`
  // has already been rebound by whatever channel targets it, so reading it
  // here would forward the *incoming* coupled value -- and its lag offset --
  // instead of the component's own state, leaving the component's `Source`
  // node with no consumer at all. See LoweringContext for the invariant.
  const auto it = ctx.source_arc.find(ch.source.value());
  if (it == ctx.source_arc.end()) {
    ctx.diags.add(Severity::Error, "E-IR-NOSRC",
                  "source field has no producing arc; is its intent Provides?",
                  "channel '" + base + "'");
    return true;  // claimed, but failed
  }

  ArcId cur = it->second;
  ValueType type = ctx.dag.arc(cur).type;

  // 1. The conversions the connection check authorized.
  //
  // `check_channel` (fuse/model/connect.hpp) decides whether the source and
  // target field descriptions are compatible and, if they are, returns the
  // ordered list of operations that makes them so -- a unit conversion, a
  // component mapping, a spatial transfer, and so on. That list is a decision,
  // not a suggestion: the compiled plan must contain exactly those operations,
  // in that order, or the run does something other than what was checked.
  //
  // So this rule reads the list rather than re-deriving it. Two rules follow:
  //
  //   * an operation the checker authorized and this compiler cannot emit is a
  //     REFUSAL, not a silent omission. Dropping it would produce a plan that
  //     passes every check and computes the wrong thing -- the same failure
  //     mode as a dropped writer.
  //   * an operation emitted here that the checker did not authorize would be
  //     the compiler inventing physics; there is no path that does it.
  {
    const ConnectionCompatibility verdict = check_channel(g, cid, ctx.connect_policy);
    for (const OperatorStep& step : verdict.chain.steps) {
      if (!ir_can_emit(step.kind)) {
        ctx.diags.add(Severity::Error, "E-IR-UNLOWERABLE",
                      std::string("the connection check authorized a '") +
                          std::string(operator_kind_name(step.kind)) +
                          "' operation, which this compiler cannot emit yet. The "
                          "channel is refused rather than compiled without it: a plan "
                          "missing an authorized operation passes every check and "
                          "computes something else.",
                      "channel '" + base + "'");
        return true;  // claimed, but failed
      }
      switch (step.kind) {
        case OperatorKind::UnitScale: {
          // Exact, and recorded even when it is 1: a factor of 1 means the
          // units already agreed, and emitting nothing is the right plan.
          if (step.scale == Rational{1}) break;
          // The output arc carries the TARGET's units. That is what makes the
          // conversion real to the type checker: without the retype, the arc
          // would still claim the source's units and the assignment at the sink
          // would be comparing a value in pascals against a slot in kilopascals
          // while the plan contained a multiply. Type and data must move
          // together.
          ValueType converted = type;
          converted.units = g.field(g.channel_at(cid).target).units;
          cur = emit(ctx, OpKind::Scale, cur, converted,
                     OpAttrs{.scale = step.scale,
                             .reason = step.explicitly_requested ? "requested" : "inferred"},
                     cid, base + "/scale");
          type = converted;
          break;
        }
        case OperatorKind::SpatialTransfer:
        case OperatorKind::TemporalTransfer:
          // Emitted below by the remap and interpolation steps, which know the
          // regions and the clocks. Named here so the two agree.
          break;
        default:
          break;  // unreachable: ir_can_emit() refused it above
      }
    }
  }

  // 2. Sample + Accumulate.
  if (ch.accumulate.op != AccumOp::Instant) {
    ValueType wave = type;
    wave.kind = ValueKind::Waveform;
    cur = emit(ctx, OpKind::Sample, cur, wave, OpAttrs{}, cid, base + "/sample");

    ValueType acc = type;  // back to Instant
    cur = emit(ctx, OpKind::Accumulate, cur, acc,
               OpAttrs{.accum = ch.accumulate.op}, cid, base + "/accumulate");
    type = acc;
  }

  // 2b. Space-time projection (spec §7.3): the `T` of `M = T (x) W`.
  //
  // Two nodes, not one. `Sample` runs on the SOURCE clock and is what makes
  // the whole window of source step values visible at once -- `T`'s column
  // index runs over every source step of the window, not only the ones that
  // overlap the target step being written, so a single node on the target
  // clock could not see its own operands. `Interpolate` runs once per target
  // step and applies row `dst_step` of `T`. `expand()` places them on the two
  // clocks (see `has_sample()` there) and the runtime interprets the pair
  // (src/runtime/runner.cpp).
  //
  // `W`, the spatial half, stays on the separate `Remap` node below. That is
  // exact, not an approximation: `T (x) W == (T (x) I)(I (x) W)`, so applying
  // the two factors as two nodes is the same operator as the tensor product,
  // and it keeps the IR's one-operator-per-node property.
  if (ch.spacetime.basis != TemporalBasis::None) {
    ValueType wave = type;
    wave.kind = ValueKind::Waveform;
    cur = emit(ctx, OpKind::Sample, cur, wave, OpAttrs{}, cid,
               base + "/spacetime_sample");

    ValueType projected = type;  // back to Instant
    cur = emit(ctx, OpKind::Interpolate, cur, projected,
               OpAttrs{.basis = ch.spacetime.basis,
                       .degree = ch.spacetime.degree},
               cid, base + "/spacetime");
    type = projected;
  }

  // 3. Extrapolate.
  if (ch.extrapolate.enabled) {
    cur = emit(ctx, OpKind::Extrapolate, cur, type,
               OpAttrs{.order = ch.extrapolate.order}, cid,
               base + "/extrapolate");
  }

  // 4. Lag: a TimeLevel shift, not a node (spec §4.3).
  if (ch.lag != 0) {
    ValueType shifted = type;
    shifted.time.offset = shifted.time.offset - Rational{ch.lag};
    // Rebind by emitting nothing: retype the value by threading a fresh arc
    // through a no-op Interpolate would be wrong, so instead record the shift
    // on the next emitted arc. If no further op follows, the Sink sees it.
    type = shifted;
  }

  // Transport before Remap: v1 places remap on the target's ranks.
  // type.rank_set/dst_rank_set are component ids (ValueType::rank_set's
  // documented meaning), so a plain integer comparison would fire Transport
  // for every cross-component channel, since source and target are always
  // different components. What determines whether a data move is needed is
  // whether the two components' *physical* placements are the SAME SET of
  // ranks: co-located components (identical ranks, different component ids)
  // need no Transport, because every value is already on the rank that wants
  // it.
  //
  // Identical, not merely overlapping. This condition used to read
  // `!overlaps(...)`, which suppressed the move whenever the placements shared
  // even one rank -- so a channel between ranks {0,1,2,3} and {2,3,4,5} got no
  // Transport at all and ranks 4 and 5 silently received nothing. That was
  // invisible in testing because the only thing that would have caught it
  // lived in transport setup, which only runs when a Transport was emitted: no
  // move meant no check either. Partially shared placements now emit the move,
  // and the runtime delivers it -- a peer-wise plan for the ranks that differ,
  // a local copy for the ranks in common.
  const int dst_rank_set = static_cast<int>(g.owner_of(ch.target).value());
  bool ranks_differ = type.rank_set != dst_rank_set;
  if (type.rank_set >= 0 && dst_rank_set >= 0) {
    const RankSet& src_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(type.rank_set)}).ranks;
    const RankSet& dst_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(dst_rank_set)}).ranks;
    ranks_differ = !src_ranks.same_as(dst_ranks);
  }
  // Whether this channel's Remap moves the data between placements ITSELF
  // (fuse/model/remap_kind.hpp). If it does, a Transport in front of it is
  // not redundant but wrong: it delivers the source values onto the target
  // placement's view of the source mesh, which on a disjoint placement is
  // empty, and the operator then reads its input where nothing lives.
  const bool remap_moves =
      remap_moves_placement(ch.remap.method_override, g.field(ch.source), g.field(ch.target));

  if (ranks_differ && remap_moves) {
    // The agreement point still comes first, for the same reason it follows a
    // Transport: the distributed operator is the first place these ranks meet,
    // and a rank that failed before it must be discovered here, by everyone,
    // rather than by its partners blocking inside the operator's gather.
    OpAttrs ea;
    ea.src_rank_set = type.rank_set;
    ea.dst_rank_set = dst_rank_set;
    ea.reason = "before-distributed-remap";
    cur = emit(ctx, OpKind::Epoch, cur, type, ea, cid, base + "/epoch");
  } else if (ranks_differ) {
    ValueType moved = type;
    moved.rank_set = dst_rank_set;
    OpAttrs ta;
    ta.src_rank_set = type.rank_set;
    ta.dst_rank_set = dst_rank_set;
    cur = emit(ctx, OpKind::Transport, cur, moved, ta, cid, base + "/transport");
    type = moved;

    // A rank-agreement point immediately after the data move.
    //
    // The move is the first place in the channel where ranks that were doing
    // independent work have to meet. If one of them failed on the way here --
    // a component refused, a remap found an uncovered point, an exact
    // conversion overflowed -- the others are already committed to the move
    // and will wait for a message that never comes. That is the hang.
    //
    // So the plan carries an explicit instruction where every participating
    // rank learns the same answer: all succeeded, or which rank did not, and
    // the run stops together. It is an operator rather than something the
    // runtime does implicitly so that it is visible in the plan dump, ordered
    // by the same scheduler as everything else, and checkable -- the scheduler
    // verifies that the ranks reaching it are the same set on every side.
    OpAttrs ea;
    ea.src_rank_set = ta.src_rank_set;
    ea.dst_rank_set = ta.dst_rank_set;
    ea.reason = "after-transport";
    cur = emit(ctx, OpKind::Epoch, cur, type, ea, cid, base + "/epoch");  // after the move
  }

  // 5. Remap: emitted when flags ask for it or the regions differ.
  const RegionId tgt_region = g.field(ch.target).region;
  const bool region_changes = (type.region != tgt_region);
  if (ch.remap.flags != RemapFlags::None || region_changes || (ranks_differ && remap_moves)) {
    ValueType remapped = type;
    remapped.region = tgt_region;
    // The distributed operator writes its result on the TARGET placement; the
    // output type says so, and that is what sizes its buffer
    // (make_region_sizes reads the arc's rank_set).
    if (ranks_differ && remap_moves) remapped.rank_set = dst_rank_set;
    cur = emit(ctx, OpKind::Remap, cur, remapped,
               OpAttrs{.remap_flags = ch.remap.flags,
                       .method = ch.remap.method_override},
               cid, base + "/remap");
    type = remapped;
  } else if (ch.lag != 0) {
    // The lag shift needs somewhere to live when no Remap follows.
    cur = emit(ctx, OpKind::Interpolate, cur, type, OpAttrs{.lag = ch.lag}, cid,
               base + "/lag");
  }

  // insert_derived_ops (Constrain, Diagnose) reads/writes through
  // ctx.field_arc[target], so bind the post-Remap arc there before calling
  // it, then re-read the result. This must run before Relax: Relax is
  // user-authored and belongs after the derived verification chain
  // (spec §5.2's Remap -> Constrain -> Diagnose -> Relax), and the DAG is
  // append-only so there is no way to splice Constrain/Diagnose in front of
  // an already-emitted Relax node.
  ctx.field_arc[ch.target.value()] = cur;
  insert_derived_ops(ctx, cid);
  cur = ctx.field_arc.at(ch.target.value());
  type = ctx.dag.arc(cur).type;

  // 6. Relax.
  if (ch.relax.enabled) {
    cur = emit(ctx, OpKind::Relax, cur, type, OpAttrs{.omega = ch.relax.omega},
               cid, base + "/relax");
  }

  // 7. Record this channel as a writer of the target field.
  //
  // It APPENDS. The old code assigned `field_arc[target] = cur`, so a second
  // channel writing the same field replaced the first and the first's chain
  // dead-ended with no consumer -- a declared dependence deleted in silence.
  // Which writers exist is settled here; how they combine is settled after
  // every channel has lowered, because the combine needs all of them at once.
  ctx.field_writers[ch.target.value()].push_back(cur);
  ctx.field_arc[ch.target.value()] = cur;  // single-writer fast path, fixed up below
  return true;
}

}  // namespace

void register_builtin_rules(LoweringRegistry& reg) {
  reg.add("channel.default", 100, lower_channel);
}

}  // namespace fuse
