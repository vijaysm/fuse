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
      // A Sink is never an ORDINARY region's member: it is downstream of
      // the loop (reads the region's converged output), not part of what
      // the loop re-executes -- see group_regions_within_instants()'s
      // own comment (src/schedule/scheduler.cpp). Sinks are created early
      // in lower()'s own pipeline, so this is stated explicitly here
      // rather than relying on no Sink existing yet when this scan runs.
      //
      // Time windows: a WINDOW-SCOPED region's own Sink is the
      // opposite case -- every Sink whose origin is one of THIS region's
      // channels is, by construction (finish_window_regions(),
      // src/ir/window_lowering.cpp, only pre-creates a Sink's arc for a
      // field some member of a window-scoped region imports), exactly
      // the delivery a partner member's OWN WindowAdvance node reads
      // back via OpAttrs::window_import -- i.e. INSIDE this same loop,
      // not a converged value leaving it. Excluding it here anyway
      // starves it in group_regions_within_instants(): that pass always
      // prefers a ready REGION MEMBER over a ready non-member in the
      // same instant, and an importing WindowAdvance node (empty
      // `inputs`, so trivially always "ready") is such a member, so the
      // Sink -- excluded, hence never preferred -- was measured running
      // AFTER the WindowAdvance node whose window_import names it,
      // failing check_window_causality()'s strict (unlagged) rule with
      // NO valid schedule existing for either channel's direction.
      // Measured directly: a two-way, one-channel-lagged fixture refused
      // with E-SCHED-WINDOW-CAUSALITY until this member-inclusion; the
      // identical fixture with BOTH channels lagged happened to schedule
      // anyway (the relaxed rule does not care about position), which is
      // what surfaced this as a scheduling-order defect rather than a
      // graph error.
      if (node.kind == OpKind::Sink && !decl.window_scope.has_value()) continue;
      // `BuildRemapHalo` carries the SAME channel origin its qualifying
      // channel's ordinary chain does (so `fuse::execute_remap_halo_
      // instructions()` and `Runner::Impl::resolve_remap_halo()` can find
      // the channel it is for), but it is a ONE-TIME setup instruction
      // (`expand()` stamps a single instance at `t = 0`, `TickPhase::
      // BeforeTicks` -- OpKind::BuildRemapHalo's own doc comment, op.hpp),
      // never one more node the loop below re-executes per sweep. Without
      // this exclusion it would be swept into `region.body` by the SAME origin
      // match that finds the channel's Transport/Remap/.../Sink chain,
      // which both misreports the loop as starting at `t = 0` (`region_
      // enter` lands on it instead of on the real first chain node) and
      // risks `E-SCHED-REGION` on an otherwise valid graph the moment a
      // `t = 0` instruction unrelated to this region's own channels sorts
      // between it and the rest of the body (`schedule()` requires a
      // region's body to be instantiated as CONTIGUOUS whole-body passes).
      if (node.kind == OpKind::BuildRemapHalo) continue;
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

    // Spec Section 5.3: the checkpoint set is derived, never asked for.
    // Run it last, after the Residual nodes exist, so the analysis sees the
    // whole body. Note the residual nodes are deliberately *not* in
    // region.body -- body is fixed above, before they are emitted -- and nor
    // are the Sink nodes, explicitly excluded above regardless of when
    // lower() happens to create them. derive_checkpoint_set() accounts for
    // both (see its rule 4).
    const CheckpointSet cp = derive_checkpoint_set(g, ctx.dag, region);
    region.carried = cp.coupler_arcs;
    ctx.dag.add_region(std::move(region));
  }
}

}  // namespace fuse
