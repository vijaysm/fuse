#include <string>
#include <unordered_set>
#include <vector>

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"

namespace fuse {

void build_iteration_regions(LoweringContext& ctx) {
  const CouplingGraph& g = ctx.graph;

  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});

    std::unordered_set<std::uint32_t> member_channels;
    for (const ChannelId c : decl.channels) member_channels.insert(c.value());

    OperatorDAG::IterationRegion region;
    region.name = decl.name;
    region.max_iterations = decl.max_iterations;

    for (std::uint32_t n = 0; n < ctx.dag.node_count(); ++n) {
      const Node& node = ctx.dag.node(NodeId{n});
      if (node.origin.valid() && member_channels.count(node.origin.value())) {
        region.body.push_back(NodeId{n});
      }
    }

    if (region.body.empty()) {
      ctx.diags.add(Severity::Error, "E-IR-REGION-EMPTY",
                    "iteration region '" + decl.name +
                        "' covers no lowered nodes");
      continue;
    }

    // One Residual per channel, consuming that channel's final value.
    for (const ChannelId c : decl.channels) {
      const Channel& ch = g.channel_at(c);
      const auto it = ctx.field_arc.find(ch.target.value());
      if (it == ctx.field_arc.end()) continue;

      ValueType rtype = ctx.dag.arc(it->second).type;
      const ArcId res = ctx.dag.add_arc(rtype, ch.name + "/residual");
      ctx.dag.add_node(OpKind::Residual, {it->second}, res, OpAttrs{}, c,
                       ch.name + "/residual");
      region.residuals.push_back(res);
    }

    // Spec §5.3 / Task 7.2: the checkpoint set is derived, never asked for.
    // Run it last, after the Residual nodes exist, so the analysis sees the
    // whole body. Note the residual nodes are deliberately *not* in
    // region.body -- body is fixed above, before they are emitted -- and nor
    // are the Sink nodes, which lower() creates only after this function
    // returns. derive_checkpoint_set() accounts for both (see its rule 4).
    const CheckpointSet cp = derive_checkpoint_set(g, ctx.dag, region);
    region.carried = cp.coupler_arcs;
    ctx.dag.add_region(std::move(region));
  }
}

}  // namespace fuse
