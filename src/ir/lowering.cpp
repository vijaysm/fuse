#include "fuse/ir/lowering.hpp"

#include "fuse/ir/op.hpp"

namespace fuse {

LoweringRegistry& LoweringRegistry::builtin() {
  static LoweringRegistry* reg = [] {
    auto* r = new LoweringRegistry;
    register_builtin_rules(*r);
    return r;
  }();
  return *reg;
}

namespace {

/// The type a field has as it leaves its component.
ValueType source_type(const CouplingGraph& g, FieldId f) {
  const FieldSpec& spec = g.field(f);
  const ComponentId owner = g.owner_of(f);
  return ValueType{.kind = ValueKind::Instant,
                   .region = spec.region,
                   .components = spec.components,
                   .units = spec.units,
                   .conservation = spec.conservation,
                   .time = TimeLevel::current(),
                   .rank_set = static_cast<int>(owner.value())};
}

std::string qualified(const CouplingGraph& g, FieldId f) {
  return g.component(g.owner_of(f)).name + "." + g.field(f).name;
}

/// Step 4 of the driver: every node input must be assignable from its arc.
void type_check(LoweringContext& ctx) {
  const OperatorDAG& d = ctx.dag;
  for (std::uint32_t i = 0; i < d.node_count(); ++i) {
    const Node& n = d.node(NodeId{i});
    if (n.kind != OpKind::Sink) continue;  // interior types are set by rules

    if (n.inputs.empty() || !n.output.valid()) continue;
    const ValueType& produced = d.arc(n.inputs.front()).type;
    const ValueType& required = d.arc(n.output).type;

    bool allow_time = false;
    if (n.origin.valid() && n.origin.value() < ctx.graph.channel_count()) {
      allow_time =
          !ctx.graph.channel_at(n.origin).time_level_mismatch_reason.empty();
    }
    if (!assignable_from(produced, required, allow_time)) {
      ctx.diags.add(Severity::Error, "E-IR-TYPE",
                    "cannot assign " + produced.to_string() + " to " +
                        required.to_string(),
                    "node " + std::to_string(i) + " \"" + n.label + "\"");
    }
  }
}

}  // namespace

LoweringResult lower(const CouplingGraph& g) {
  LoweringResult result;
  LoweringContext ctx{g, result.dag, result.diags, {}, {}};

  // 1. Source node per provided field.
  //
  // Both maps are seeded here; see LoweringContext for the invariant that
  // separates them. `source_arc` is written exactly once per field, right
  // here, and is never rebound: it is what the component itself produces, and
  // no channel can change that. `field_arc` starts as the same arc only so
  // that a Bidirectional field no channel writes still lowers to the
  // degenerate Source -> Sink self-copy in step 3.
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Provides && intent != Intent::Bidirectional) continue;
    const ArcId a = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
    ctx.dag.add_node(OpKind::Source, {}, a, OpAttrs{}, ChannelId{},
                     qualified(g, f));
    ctx.source_arc[i] = a;
    ctx.field_arc[i] = a;
  }

  // 2. One rule application per channel, in id order. The channel rule
  // itself calls insert_derived_ops between Remap and Relax (see
  // rules_channel.cpp): Relax is user-authored and must consume whatever
  // Constrain/Diagnose produced, which is only possible if the derived pass
  // runs before Relax's node is created, not after the whole rule returns.
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    if (!LoweringRegistry::builtin().apply_first_match(ctx, cid)) {
      ctx.diags.add(Severity::Error, "E-IR-NORULE",
                    "no lowering rule matched",
                    "channel '" + g.channel_at(cid).name + "'");
    }
  }

  build_iteration_regions(ctx);

  // 3. Sink node per required field that a channel actually wrote.
  //
  // For a Bidirectional field no channel ever targets, `ctx.field_arc[i]`
  // found below is still the Source arc step 1 seeded it with (never rebound,
  // since no channel wrote it) -- so the Sink built here deliberately
  // consumes the field's own Source output, producing a degenerate
  // Source -> Sink self-copy. This looks unintended from here alone; it is
  // not. See LoweringContext's `field_arc` doc comment in
  // include/fuse/ir/lowering.hpp (lines ~39-46) for why prove_acyclic() and
  // expand() are written to expect exactly this shape, and
  // Lowering.UnwrittenBidirectionalFieldSelfCopiesThroughOneSink in
  // test/unit/lowering_test.cpp for the regression test that pins it.
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Requires && intent != Intent::Bidirectional) continue;
    const auto it = ctx.field_arc.find(i);
    if (it == ctx.field_arc.end()) continue;  // W-NO-WRITER is validate()'s job

    // Find the channel that wrote it, so the Sink can carry its origin.
    ChannelId origin;
    for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
      if (g.channel_at(ChannelId{c}).target == f) { origin = ChannelId{c}; break; }
    }

    const ArcId sink_arc = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
    ctx.dag.add_node(OpKind::Sink, {it->second}, sink_arc, OpAttrs{}, origin,
                     qualified(g, f));
  }

  // 4. Type check.
  type_check(ctx);
  return result;
}

}  // namespace fuse
