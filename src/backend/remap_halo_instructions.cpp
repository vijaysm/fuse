#include "fuse/backend/remap_halo_instructions.hpp"

#include <cstdint>
#include <string>

#include "fuse/backend/remap_plan.hpp"
#include "fuse/model/component.hpp"  // fuse::RankSet

namespace fuse {

Result<std::size_t> execute_remap_halo_instructions(const CouplingGraph& graph,
                                                     const OperatorDAG& dag,
                                                     MeshBackend& backend) {
  std::size_t total = 0;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node&  n = dag.node(nid);
    if (n.kind != OpKind::BuildRemapHalo) continue;

    // Defensive: `emit_remap_halo_instructions()` (src/ir/lowering.cpp)
    // always stamps a valid channel origin on this kind; a DAG built some
    // other way is the only way this could fire.
    if (!n.origin.valid() || n.origin.value() >= graph.channel_count()) {
      return Failure{"E-IR-REMAP-HALO",
                   "execute_remap_halo_instructions: BuildRemapHalo node " +
                       std::to_string(i) +
                       " carries no valid channel origin"};
    }

    const Channel& ch = graph.channel_at(n.origin);

    RemapPlan plan;
    plan.src = graph.field(ch.source).region;
    plan.tgt = graph.field(ch.target).region;
    plan.flags = ch.remap.flags;
    plan.method = ch.remap.method_override;
    // ROUND 2 (owner decision 2026-10-18): read the placement BACK from
    // the instruction's own `attrs.dst_rank_set` -- the component
    // `emit_remap_halo_instructions()` (src/ir/rules_channel.cpp) already
    // resolved via `fuse::remap_placement()` when it decided to emit this
    // node -- rather than calling `remap_placement()` a second time here.
    // What this instruction acts on is what lowering froze into it, not
    // whatever the graph would answer if asked again at execution time.
    if (n.attrs.dst_rank_set < 0) {
      return Failure{"E-IR-REMAP-HALO",
                   "execute_remap_halo_instructions: BuildRemapHalo node " +
                       std::to_string(i) +
                       " carries no placement in attrs.dst_rank_set"};
    }
    plan.placement = ComponentId{static_cast<std::uint32_t>(n.attrs.dst_rank_set)};

    // OWNER NOTE (2026-10-18, relayed mid-track; reaffirmed in round 2):
    // D5's disjoint-placement weights are settled as HALO BRIDGING --
    // `plan.placement` stays the TARGET placement (as above, now read
    // from `attrs.dst_rank_set`); the MoabHalo track is adding a second
    // placement argument naming which ranks OWN the source cells. That
    // RankSet is ALREADY available here without a graph lookup --
    // `attrs.src_rank_set` names the owning component
    // (`graph.component(ComponentId{n.attrs.src_rank_set}).ranks` is its
    // RankSet, the same pattern `Runner::Impl::attach_comm_graphs()`
    // already uses for `Transport`'s own `src_rank_set`/`dst_rank_set`,
    // src/runtime/runner.cpp) -- not yet threaded through
    // `fuse::build_remap_halo`'s own signature as of this track's export.
    // Once the two-placement overload lands, this call needs
    // `graph.component(ComponentId{n.attrs.src_rank_set}).ranks` added as
    // that second argument -- open_questions names this explicitly.
    const Result<std::size_t> r = build_remap_halo(plan, graph, backend);
    if (!r.ok()) {
      return Failure{r.error().code,
                   "execute_remap_halo_instructions: channel '" + ch.name +
                       "': " + r.error().message};
    }
    total += r.value();
  }
  return total;
}

}  // namespace fuse
