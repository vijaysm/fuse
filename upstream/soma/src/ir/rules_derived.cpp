#include <string>
#include <vector>

#include "soma/ir/lowering.hpp"
#include "soma/ir/op.hpp"

namespace soma {
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

  // `source_arc`, not `field_arc`: this is the head of the chain the channel
  // rule just emitted, and lower_channel starts that chain at the source
  // field's own `Source` arc. For an Intent::Bidirectional source field
  // `field_arc` points at whatever channel last wrote it, and walking from
  // there would find no node of this channel, so `has_remap` would come out
  // false and Constrain/Diagnose would silently not be emitted. See
  // LoweringContext for the invariant.
  const auto src_it = ctx.source_arc.find(ch.source.value());
  if (src_it == ctx.source_arc.end()) return;

  const std::vector<NodeId> chain = channel_chain(ctx, cid, src_it->second);

  ArcId cur = tgt_it->second;
  ValueType type = ctx.dag.arc(cur).type;

  // Find whether this channel emitted a Remap; Constrain and Diagnose are
  // both keyed off of it.
  bool has_remap = false;
  for (const NodeId n : chain) {
    if (ctx.dag.node(n).kind == OpKind::Remap) { has_remap = true; break; }
  }

  if (has_remap && type.conservation != Conservation::None) {
    cur = append(ctx, OpKind::Constrain, cur, type, OpAttrs{}, cid,
                 base + "/constrain");
  }
  if (has_remap) {
    cur = append(ctx, OpKind::Diagnose, cur, type, OpAttrs{}, cid,
                 base + "/diagnose");
  }

  ctx.field_arc[ch.target.value()] = cur;
}

}  // namespace soma
