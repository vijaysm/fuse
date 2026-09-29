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

/// Step 3 of the driver, factored out so it can run in two passes (see
/// `lower()`'s own comment on `window_imported_fields` for why): a Sink
/// for field `f`, consuming whatever `ctx.field_arc[f]` currently holds.
/// No-op (silently) if `f` has no writer yet -- W-NO-WRITER is validate()'s
/// job, not this function's, and both call sites below rely on that.
void emit_field_sink(LoweringContext& ctx, const CouplingGraph& g, FieldId f) {
  const auto it = ctx.field_arc.find(f.value());
  if (it == ctx.field_arc.end()) return;

  // Find the channel that wrote it, so the Sink can carry its origin.
  ChannelId origin;
  for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
    if (g.channel_at(ChannelId{c}).target == f) { origin = ChannelId{c}; break; }
  }

  const ArcId sink_arc = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
  ctx.dag.add_node(OpKind::Sink, {it->second}, sink_arc, OpAttrs{}, origin,
                   qualified(g, f));
  ctx.sink_arc[f.value()] = sink_arc;
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
      // channel_time_mismatch_is_authorized(), not the channel's own
      // `time_level_mismatch_reason` directly: a `WaveformIteration::
      // Jacobi` window's own auto-lag (effective_channel_lag(),
      // src/ir/rules_channel.cpp step 4) creates this same mismatch on a
      // channel the user may never have called `.lag()` or
      // `.allow_time_level_mismatch()` on at all, and the compiler that
      // chose to apply the shift is what authorizes it.
      allow_time = channel_time_mismatch_is_authorized(ctx.graph, n.origin);
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

bool ir_can_emit(OperatorKind kind) noexcept {
  switch (kind) {
    // Emitted here, as an exact rational multiply.
    case OperatorKind::UnitScale:
      return true;
    // Emitted by the remap and interpolation steps of the channel rule, which
    // know the regions and the two clocks.
    case OperatorKind::SpatialTransfer:
    case OperatorKind::TemporalTransfer:
      return true;
    // No operator yet. Each changes what the value means -- an offset moves the
    // zero of a scale, selection and aggregation discard components, expansion
    // invents them, clipping and conservation enforcement alter the data to
    // satisfy a constraint -- so each needs its own verified kernel, not a
    // generic pass-through.
    case OperatorKind::AffineUnit:
    case OperatorKind::SignFlip:
    case OperatorKind::ComponentSelect:
    case OperatorKind::ComponentReorder:
    case OperatorKind::Aggregate:
    case OperatorKind::Expand:
    case OperatorKind::Clip:
    case OperatorKind::ConservationEnforcement:
      return false;
  }
  return false;  // a new kind is refused until someone decides
}

LoweringResult lower(const CouplingGraph& g, ConnectPolicy policy) {
  LoweringResult result;
  LoweringContext ctx{g, result.dag, result.diags, {}, {}, {}, {}, policy};

  // Time windows track: every field a window-scoped iteration region's
  // own exporting member provides instead of an ordinary Source
  // (seed_window_source_arcs(), called after step 1). Computed once, up
  // front, since step 1 needs it below.
  const std::unordered_set<std::uint32_t> window_fields =
      window_provided_fields(g);

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
    if (window_fields.count(i)) continue;  // seed_window_source_arcs() provides this
    const ArcId a = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
    ctx.dag.add_node(OpKind::Source, {}, a, OpAttrs{}, ChannelId{},
                     qualified(g, f));
    ctx.source_arc[i] = a;
    ctx.field_arc[i] = a;
  }

  // Time windows track: pre-creates the bare output arc (no producing
  // node yet) for every window-scoped region member that owns a
  // channel's source field, seeding ctx.source_arc/ctx.field_arc exactly
  // as the ordinary Source loop above would have -- so step 2's ordinary
  // per-channel rule, run UNCHANGED below for a window-scoped region's
  // own channels too, has something real to read as that channel's
  // source. See seed_window_source_arcs()'s own doc comment
  // (fuse/ir/lowering.hpp) for why the actual WindowAdvance NODE is not
  // created until much later (finish_window_regions(), after step 3).
  seed_window_source_arcs(ctx);

  // 2. One rule application per channel, in id order. The channel rule
  // itself calls insert_derived_ops between Remap and Relax (see
  // rules_channel.cpp): Relax is user-authored and must consume whatever
  // Constrain/Diagnose produced, which is only possible if the derived pass
  // runs before Relax's node is created, not after the whole rule returns.
  //
  // Time windows track: a window-scoped region's own channels run through
  // this SAME rule, unchanged -- the user's chosen design. Their source
  // reads back the arc seed_window_source_arcs() just seeded; their
  // ordinary Transport/Remap/Epoch chain, and the Sink step 3 builds for
  // them below, are exactly what any other cross-placement channel gets.
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    if (!LoweringRegistry::builtin().apply_first_match(ctx, cid)) {
      ctx.diags.add(Severity::Error, "E-IR-NORULE",
                    "no lowering rule matched",
                    "channel '" + g.channel_at(cid).name + "'");
    }
  }

  // 2b. Combine multiple writers of one field.
  //
  // Iterated in ascending field id, and the inputs are in channel declaration
  // order, so the emitted node is identical on every rank -- which matters
  // because each rank compiles this plan independently and they must agree.
  {
    std::vector<std::uint32_t> targets;
    targets.reserve(ctx.field_writers.size());
    for (const auto& entry : ctx.field_writers) targets.push_back(entry.first);
    std::sort(targets.begin(), targets.end());

    for (const std::uint32_t field_value : targets) {
      const std::vector<ArcId>& writers = ctx.field_writers[field_value];
      if (writers.size() < 2) continue;
      const FieldId f{field_value};
      const MergeSpec* spec = g.merge_for(f);
      if (spec == nullptr) {
        // validate() reports this first, with the advice. Refusing again here
        // is deliberate: lowering must never be reachable in a state where it
        // would have to choose a writer, and a compiler that depends on an
        // earlier pass having run is one refactor away from choosing silently.
        ctx.diags.add(Severity::Error, "E-IR-MULTI-WRITER",
                      "field '" + g.field(f).name + "' has " +
                          std::to_string(writers.size()) +
                          " writers and no declared merge; the compiler will not "
                          "choose one",
                      qualified(g, f));
        continue;
      }
      // The combined value's time level is the LATEST of its contributions,
      // not the first writer's.
      //
      // This is not bookkeeping. A lagged contribution is typed one level
      // back, and the acyclicity proof cuts an edge whose input is older than
      // its consumer's output -- correctly, since last period's value
      // constrains nothing in this one. If the combined value inherited a
      // lagged writer's level merely because that writer was declared first,
      // the edge from the combine to the field would be cut, and with it the
      // dependence carried by the writers that are NOT lagged. The sweep in
      // acyclicity_sweep_test.cpp found exactly that: a two-way coupling where
      // each direction had one lagged and one unlagged channel into a combined
      // field was certified acyclic, which is the direction that lets an
      // unsolvable setup run.
      //
      // A value is available when its most recent contribution is, so the
      // latest level is also the physically right answer.
      //
      // Offsets against different anchors have no shared origin (the same
      // reason src/schedule/acyclicity.cpp:218-219 and
      // src/schedule/scheduler.cpp:52 give for "keep the edge" when they hit
      // this same ambiguity) -- but unlike those two call sites, silently
      // keeping whichever writer got here first would mistype the merged
      // output, not just risk an over-cautious cycle report. So this refuses
      // rather than guesses (REVIEW.md R-43; today no lowering rule produces
      // a non-WindowEnd anchor, so this branch is not yet reachable, but the
      // guard has to exist before one does).
      //
      // Two passes, deliberately not one: `iterate` alone orders writers
      // regardless of anchor (it numbers coupling periods the same way under
      // every anchor convention), so the first pass finds the greatest
      // `iterate` across ALL writers, independent of anchor. Only the
      // writers tied at THAT iterate can be ambiguous -- a writer whose
      // iterate is strictly greater is the newest no matter what anchor it
      // or anyone else uses, and must not be shadowed by an anchor clash
      // among older writers a single combined pass would still be visiting.
      int max_iterate = ctx.dag.arc(writers.front()).type.time.iterate;
      for (const ArcId w : writers) {
        max_iterate = std::max(max_iterate, ctx.dag.arc(w).type.time.iterate);
      }
      ArcId newest{};
      bool have_newest = false;
      bool anchor_ambiguous = false;
      for (const ArcId w : writers) {
        const TimeLevel& b = ctx.dag.arc(w).type.time;
        if (b.iterate != max_iterate) continue;
        if (!have_newest) {
          newest = w;
          have_newest = true;
          continue;
        }
        const TimeLevel& a = ctx.dag.arc(newest).type.time;
        if (b.anchor != a.anchor) {
          anchor_ambiguous = true;
        } else if (b.offset > a.offset) {
          newest = w;
        }
      }
      if (anchor_ambiguous) {
        ctx.diags.add(Severity::Error, "E-IR-MERGE-ANCHOR",
                      "field '" + g.field(f).name + "' has writers whose time "
                      "levels agree on iterate but disagree on anchor; there "
                      "is no ordering between them from which to pick the "
                      "newest contribution",
                      qualified(g, f));
        continue;
      }
      const OpAttrs merge_attrs{.reason = std::string(merge_op_name(spec->op)),
                                .merge = spec->op};
      const ValueType newest_type = ctx.dag.arc(newest).type;

      // Merge with more than three sources is lowered as a left-fold of
      // pairwise merges (documented at include/fuse/schedule/plan.hpp's
      // Instruction; that struct's `in` array is what actually holds three,
      // but `ir` does not depend on `schedule`, so the limit is duplicated
      // here rather than shared -- REVIEW.md R-27). Up to that many writers
      // still become one Merge node, unchanged from before this fix.
      constexpr std::size_t kMaxMergeInputs = 3;
      if (writers.size() <= kMaxMergeInputs) {
        const ArcId out =
            ctx.dag.add_arc(newest_type, qualified(g, f) + "/merge");
        ctx.dag.add_node(OpKind::Merge, writers, out, merge_attrs,
                         ChannelId{}, qualified(g, f) + "/merge");
        ctx.field_arc[field_value] = out;
      } else {
        // Every node in the chain carries `newest_type`, computed once above
        // over the FULL writer list -- not recomputed from each pair's own
        // two inputs -- so an intermediate fold's output still reports the
        // correct running-newest time level to the acyclicity prover's
        // lag-cutting rule partway through the chain.
        ArcId acc = writers.front();
        for (std::size_t k = 1; k < writers.size(); ++k) {
          const ArcId out = ctx.dag.add_arc(
              newest_type,
              qualified(g, f) + "/merge" + std::to_string(k));
          ctx.dag.add_node(OpKind::Merge, {acc, writers[k]}, out, merge_attrs,
                           ChannelId{}, qualified(g, f) + "/merge" +
                                            std::to_string(k));
          acc = out;
        }
        ctx.field_arc[field_value] = acc;
      }
    }
  }

  // 3. Sink node per required field that a channel actually wrote.
  //
  // For a Bidirectional field no channel ever targets, `ctx.field_arc[i]`
  // found by `emit_field_sink()` is still the Source arc step 1 seeded it
  // with (never rebound, since no channel wrote it) -- so the Sink built
  // there deliberately consumes the field's own Source output, producing
  // a degenerate Source -> Sink self-copy. This looks unintended from
  // here alone; it is not. See LoweringContext's `field_arc` doc comment
  // in include/fuse/ir/lowering.hpp (lines ~39-46) for why prove_acyclic()
  // and expand() are written to expect exactly this shape, and
  // Lowering.UnwrittenBidirectionalFieldSelfCopiesThroughOneSink in
  // test/unit/lowering_test.cpp for the regression test that pins it.
  //
  // Time windows track: split into two passes rather than moved wholesale,
  // to keep this step's own node-creation ORDER unchanged for every graph
  // with no window-scoped region (which is every graph before this
  // track, and every one after it that does not opt in): only a field a
  // window member actually IMPORTS needs its Sink before
  // build_iteration_regions() (so finish_window_regions() below can read
  // its output arc via ctx.sink_arc, for that importing member's
  // OpAttrs::window_import); every other required field's Sink -- which
  // is every one of them on a graph with no window-scoped region --
  // still lands exactly where step 3 always has, after
  // build_iteration_regions(), so its node ids and the golden IR fixtures
  // that pin them are unaffected.
  const std::unordered_set<std::uint32_t> window_imported =
      window_imported_fields(g);
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Requires && intent != Intent::Bidirectional) continue;
    if (!window_imported.count(i)) continue;
    emit_field_sink(ctx, g, f);
  }

  // Time windows track: must run after the window-imported Sink pass
  // above (an importing member's window_import names a Sink's own output
  // arc, via ctx.sink_arc) and before build_iteration_regions(), which
  // groups the WindowAdvance nodes this creates into the region's body by
  // their origin channel, exactly as it already does for
  // ordinarily-lowered nodes.
  finish_window_regions(ctx);

  build_iteration_regions(ctx);

  // 3 (continued). Every other required field's Sink, in the same order
  // step 3 has always used -- this is the ENTIRE step 3 loop for a graph
  // with no window-scoped region, since window_imported is then empty.
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Requires && intent != Intent::Bidirectional) continue;
    if (window_imported.count(i)) continue;  // already built above
    emit_field_sink(ctx, g, f);
  }

  // 4. Type check.
  type_check(ctx);
  return result;
}

}  // namespace fuse
