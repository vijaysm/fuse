#include <string>

#include "soma/ir/lowering.hpp"
#include "soma/ir/op.hpp"

namespace soma {
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
  // different components. What actually determines whether a data move is
  // needed is whether the two components' *physical* rank ranges overlap:
  // co-located components (same ranks, different component ids) need no
  // Transport, disjoint ones do.
  const int dst_rank_set = static_cast<int>(g.owner_of(ch.target).value());
  bool ranks_differ = type.rank_set != dst_rank_set;
  if (type.rank_set >= 0 && dst_rank_set >= 0) {
    const RankSet& src_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(type.rank_set)}).ranks;
    const RankSet& dst_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(dst_rank_set)}).ranks;
    ranks_differ = !src_ranks.overlaps(dst_ranks);
  }
  if (ranks_differ) {
    ValueType moved = type;
    moved.rank_set = dst_rank_set;
    OpAttrs ta;
    ta.src_rank_set = type.rank_set;
    ta.dst_rank_set = dst_rank_set;
    cur = emit(ctx, OpKind::Transport, cur, moved, ta, cid, base + "/transport");
    type = moved;
  }

  // 5. Remap: emitted when flags ask for it or the regions differ.
  const RegionId tgt_region = g.field(ch.target).region;
  const bool region_changes = (type.region != tgt_region);
  if (ch.remap.flags != RemapFlags::None || region_changes) {
    ValueType remapped = type;
    remapped.region = tgt_region;
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

  // 7. Bind the result to the target field.
  ctx.field_arc[ch.target.value()] = cur;
  return true;
}

}  // namespace

void register_builtin_rules(LoweringRegistry& reg) {
  reg.add("channel.default", 100, lower_channel);
}

}  // namespace soma
