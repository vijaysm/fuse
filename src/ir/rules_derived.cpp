#include <string>
#include <vector>

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"
#include "fuse/model/remap_kind.hpp"

namespace fuse {
namespace {

ArcId append(LoweringContext& ctx, OpKind kind, ArcId input, ValueType type,
             OpAttrs attrs, ChannelId origin, const std::string& label) {
  const ArcId out = ctx.dag.add_arc(std::move(type), label);
  ctx.dag.add_node(kind, {input}, out, std::move(attrs), origin, label);
  return out;
}

/// The chain of nodes this channel emitted, in dependency order, starting from
/// the source field's arc and following single-consumer links.
std::vector<NodeId> channel_chain(const LoweringContext& ctx, ChannelId cid,
                                  ArcId start) {
  std::vector<NodeId> chain;
  ArcId cur = start;
  while (true) {
    const auto& consumers = ctx.dag.consumers_of(cur);
    NodeId next;
    for (const NodeId n : consumers) {
      if (ctx.dag.node(n).origin == cid) { next = n; break; }
    }
    if (!next.valid()) break;
    chain.push_back(next);
    const ArcId out = ctx.dag.node(next).output;
    if (!out.valid()) break;
    cur = out;
  }
  return chain;
}

}  // namespace

void insert_derived_ops(LoweringContext& ctx, ChannelId cid) {
  const Channel& ch = ctx.graph.channel_at(cid);
  const std::string base = ch.name;

  const auto tgt_it = ctx.field_arc.find(ch.target.value());
  if (tgt_it == ctx.field_arc.end()) return;  // the rule failed; nothing to do

  // `chain_entry_arc`, not `source_arc`: this is the head of the chain
  // `lower_channel()` actually emitted FOR THIS CHANNEL -- the source
  // field's own `Source`/`WindowAdvance` arc for an ordinary channel, but
  // the channel's own `ExternalDelivery` node's OWN OUTPUT arc for a
  // LowFive-bound one, since this channel's chain reads what THAT node
  // produces (the application's own write), never `source_arc` directly
  // -- unchanged by `ExternalDelivery` carrying `source_producer` as an
  // ordering-only input of its own (include/fuse/ir/op.hpp's own comment
  // on that node has the full account; `channel_chain()`, below, must
  // still start AT the delivery node, not at ITS OWN input, which is
  // `source_arc` again and would defeat the whole point).
  // Walking from `source_arc` unconditionally would find no node of
  // this channel's origin downstream of a severed `ExternalDelivery`
  // root, so `has_remap` would come back false and Constrain/Diagnose go
  // silently unemitted for every LowFive-bound channel. For an ordinary channel `chain_entry_arc[cid] ==
  // source_arc[ch.source]` always, so nothing changes there; see
  // LoweringContext for the `field_arc`-vs-`source_arc` invariant this
  // still relies on for `Bidirectional` fields.
  const auto entry_it = ctx.chain_entry_arc.find(cid.value());
  if (entry_it == ctx.chain_entry_arc.end()) return;

  const std::vector<NodeId> chain = channel_chain(ctx, cid, entry_it->second);

  ArcId cur = tgt_it->second;
  ValueType type = ctx.dag.arc(cur).type;

  // Find whether this channel emitted a Remap; Constrain and Diagnose are
  // both keyed off of it.
  bool has_remap = false;
  for (const NodeId n : chain) {
    if (ctx.dag.node(n).kind == OpKind::Remap) { has_remap = true; break; }
  }

  // A conservation check needs a conserved integral. A consistent (pointwise)
  // interpolation has none (fuse::is_pointwise_interpolation), so checking it
  // reports a violation that is not one: for example, a consistent velocity
  // transfer from 8 beam vertices to 11 fluid cells "failing" 10.4 -> 1.3,
  // a vertex sum against a cell integral.
  if (has_remap && type.conservation != Conservation::None &&
      !is_pointwise_interpolation(ch.remap.method_override)) {
    cur = append(ctx, OpKind::Constrain, cur, type, OpAttrs{}, cid,
                 base + "/constrain");
  }
  if (has_remap) {
    cur = append(ctx, OpKind::Diagnose, cur, type, OpAttrs{}, cid,
                 base + "/diagnose");
  }

  ctx.field_arc[ch.target.value()] = cur;
}

}  // namespace fuse
