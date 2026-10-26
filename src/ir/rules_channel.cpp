#include <optional>
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

  // Unit 3 (S6 phase 3, Conformance): a LowFive-bound channel
  // (`FieldSpec::hdf5_path` set on the source field) is an
  // APPLICATION-performed transfer -- the field's own producer (its
  // ordinary `Source`, or for a window-scoped export its
  // `WindowAdvance`) still runs exactly as it does today, but this
  // channel's own chain does not READ its output: `ExternalDelivery`'s
  // own `execute()` case (src/runtime/runner.cpp) ignores its input, and
  // the application (an ordinary HDF5 write/read, through the installed
  // VOL, or for the tier-0 oracle, a test harness) fills its output from
  // outside this graph entirely. See OpKind::ExternalDelivery
  // (include/fuse/ir/op.hpp) and docs/user/lowfive.md, Unit 3, for why:
  // in the deployment this exists for, the source field's own producer
  // and this channel's consumer may be different processes with no such
  // DATA edge at all.
  //
  // The DAG edge is NOT severed, though -- it is given the SAME input
  // `Transport` would have had for this same channel (`source_producer`,
  // below), an earlier version of this node did not, and that was wrong:
  // with no input at all, nothing forces the scheduler to place this
  // node where `Transport` stands (after the source field's own
  // producer, before the channel's own Remap/Diagnose/Sink), and it was
  // measured, this session, to place a LowFive-bound channel's own
  // chain BEFORE the exporting member's own WindowAdvance in the SAME
  // sweep instead -- reading last sweep's state, not this one's (not
  // yet recorded as its own divergence note in docs/user/lowfive.md).
  // Carrying that SAME edge here is what gives the scheduler the
  // ordering constraint it needs, on every process, exactly as it has
  // for `Transport` in the
  // distributed ordinary case (the stream is static and identical
  // everywhere) -- without making this node READ anything through it.
  //
  // Everything below this point -- unit conversion, Remap, Extrapolate,
  // the window-importing Sink -- is unchanged: it operates on `cur`/
  // `type` exactly as it would for an ordinary channel, unaware of
  // where the value came from.
  const ArcId source_producer = it->second;
  const bool lowfive_bound = g.field(ch.source).hdf5_path.has_value();

  ArcId cur = source_producer;
  ValueType type = ctx.dag.arc(cur).type;
  if (lowfive_bound) {
    // The value the application's own read (H5Dread, or the tier-0
    // harness's stand-in) produces lands on the CONSUMER's own
    // placement, not the producer's: it is that application's own
    // process that performed the read, same as an ordinary field it
    // reads into its own registered storage. Carrying the producer's
    // rank_set forward here would make every step below that checks
    // "does this value need to move to reach its target" (the Transport-
    // before-Remap and coupler-second-hop logic further down) try to
    // insert a move FUSE must not perform for this channel at all.
    ValueType delivered = ctx.dag.arc(source_producer).type;
    delivered.rank_set = static_cast<int>(g.owner_of(ch.target).value());
    // `source_producer` as the ONE input -- the same arc `Transport`
    // would take as its own input for this identical channel (the
    // `emit(ctx, OpKind::Transport, cur, ...)` calls below, `cur` still
    // equal to `source_producer` at that point for an ordinary channel).
    // `execute()`'s own ExternalDelivery case (src/runtime/runner.cpp)
    // never reads it -- this enum's own top comment (include/fuse/ir/
    // op.hpp) has the full account of why an ordering edge with no data
    // read is still the right edge to carry.
    const ArcId out =
        ctx.dag.add_arc(delivered, base + "/external_delivery");
    ctx.dag.add_node(OpKind::ExternalDelivery, {source_producer}, out,
                     OpAttrs{}, cid, base + "/external_delivery");
    cur = out;
    type = delivered;
  }
  // `cur` is now this channel's own entry point -- `source_producer` for an
  // ordinary channel, the fresh `ExternalDelivery` arc for a LowFive-bound
  // one -- and `insert_derived_ops()`'s `channel_chain()` needs exactly
  // this, not `source_arc[ch.source]` unconditionally (LoweringContext's
  // own comment on `chain_entry_arc` has the full account of why).
  ctx.chain_entry_arc[cid.value()] = cur;

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
          // For an `AccumOp::Integral` channel the TARGET's units are the
          // time-integrated ones (connect.hpp's `delivered_units()`), and
          // the integral is only formed by the Accumulate below; the arc
          // between carries the source quantity, scaled, so its units are
          // the target's divided back by time.
          ValueType converted = type;
          converted.units = ch.accumulate.op == AccumOp::Integral
                                ? g.field(ch.target).units / units::kSecond
                                : g.field(ch.target).units;
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
    OpAttrs attrs{.accum = ch.accumulate.op};
    if (ch.accumulate.op == AccumOp::Integral) {
      // `Sample` runs once per SOURCE tick (src/schedule/expand.cpp), so the
      // interval between samples is the source component's own interval:
      // the `dt` of `num::accumulate_finalize_integral` (kernels.hpp). It is
      // carried exactly, as `OpAttrs::scale`, and applied once by the
      // interpreter when the window closes (src/runtime/runner.cpp,
      // `OpKind::Accumulate`); the delivered value is [source] x [time].
      attrs.scale = g.component(g.owner_of(ch.source)).interval;
      acc.units = acc.units * units::kSecond;
    }
    cur = emit(ctx, OpKind::Accumulate, cur, acc, attrs, cid, base + "/accumulate");
    type = acc;
  }

  // 2b. Space-time projection (spec Section 7.3): the `T` of `M = T (x) W`.
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

  // 4. Lag: a TimeLevel shift, not a node (spec Section 4.3).
  //
  // `effective_channel_lag()` (fuse/model/coupling_graph.hpp), not `ch.lag`
  // directly: inside a `WaveformIteration::Jacobi` window this channel's
  // effective lag is 1 whatever `ch.lag` itself says, and this is the ONE
  // place that decision is applied to the compiled graph -- schedule()'s
  // own causality check reads the SAME function, so the two cannot
  // disagree about which transfers are loop-carried.
  const int effective_lag = effective_channel_lag(g, cid);
  if (effective_lag != 0) {
    ValueType shifted = type;
    shifted.time.offset = shifted.time.offset - Rational{effective_lag};
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

  // Whether this channel's Remap moves the data between placements ITSELF
  // (fuse/model/remap_kind.hpp). If it does, a Transport in front of it is
  // not redundant but wrong: it delivers the source values onto the target
  // placement's view of the source mesh, which on a disjoint placement is
  // empty, and the operator then reads its input where nothing lives.
  const bool remap_moves =
      remap_moves_placement(ch.remap.method_override, g.field(ch.source), g.field(ch.target));

  // The coupler-placement track (design C, plans/2026-09-23-coupler-
  // placement-and-migration.md): `RemapSpec::remap_on` says WHERE this
  // channel's Remap runs. THE RULE is one rule, not a second mode: the value
  // moves source -> C, is remapped on C's decomposition of both meshes, and
  // (after the Remap block below) moves C -> target, with any hop between
  // identical placements elided by the same `same_as` test as always.
  // `C = target` -- `remap_on` invalid, the default -- must reproduce
  // today's IR exactly, which is why `coupler_rank_set` below degrades to
  // `dst_rank_set` both when no `remap_on` is declared AND when one is
  // declared but could never affect this channel's lowering: honouring it
  // in that case would insert two data moves (source->C, C->target) where a
  // channel with no coupler needed none, for a setting `validate()`'s
  // `W-CHANNEL-REMAP-ON-UNUSED` already reports as having no effect.
  // `could_ever_emit_remap` (fuse/model/remap_kind.hpp) is the exact gate
  // that warning is keyed on, so the two cannot disagree about which
  // channels a coupler declaration actually touches.
  const bool remap_on_declared = ch.remap.remap_on.valid() &&
                                 ch.remap.remap_on.value() < g.component_count();
  const bool remap_on_could_matter =
      remap_on_declared &&
      could_ever_emit_remap(ch.remap.flags, g.field(ch.source),
                            g.field(ch.target), ch.remap.method_override);
  const int coupler_rank_set =
      remap_on_could_matter ? static_cast<int>(ch.remap.remap_on.value())
                            : dst_rank_set;

  // `dst_ranks`, not `c_ranks`: kept under its original name so the
  // pre-existing mutation "lowering-skips-move-on-any-sharing"
  // (build-evidence/mutations.toml) still anchors on this line. It is the
  // placement this hop moves data TO, which is `coupler_rank_set` here --
  // the coupler when one is honoured, the channel's own target otherwise.
  bool ranks_differ = type.rank_set != coupler_rank_set;
  if (type.rank_set >= 0 && coupler_rank_set >= 0) {
    const RankSet& src_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(type.rank_set)}).ranks;
    const RankSet& dst_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(coupler_rank_set)}).ranks;
    ranks_differ = !src_ranks.same_as(dst_ranks);
  }
  // Unit 3: the application's own read already delivered the value onto
  // the consumer's placement (the ExternalDelivery arc above was typed
  // with dst_rank_set for exactly this reason) -- FUSE performs no data
  // move of its own on this channel, so neither hop below (Transport-
  // before-Remap here, nor the coupler-second-hop Transport at 5b) may
  // fire, whatever a rank-set comparison alone would otherwise say.
  if (lowfive_bound) ranks_differ = false;

  if (ranks_differ && remap_moves) {
    // The agreement point still comes first, for the same reason it follows a
    // Transport: the distributed operator is the first place these ranks meet,
    // and a rank that failed before it must be discovered here, by everyone,
    // rather than by its partners blocking inside the operator's gather.
    OpAttrs ea;
    ea.src_rank_set = type.rank_set;
    ea.dst_rank_set = coupler_rank_set;
    ea.reason = "before-distributed-remap";
    cur = emit(ctx, OpKind::Epoch, cur, type, ea, cid, base + "/epoch");
  } else if (ranks_differ) {
    ValueType moved = type;
    moved.rank_set = coupler_rank_set;
    OpAttrs ta;
    ta.src_rank_set = type.rank_set;
    ta.dst_rank_set = coupler_rank_set;
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
    // The distributed operator writes its result on the placement this
    // channel's Remap runs on -- C, or the target when no coupler is in
    // play -- and that is what sizes its buffer (make_region_sizes reads
    // the arc's rank_set).
    if (ranks_differ && remap_moves) remapped.rank_set = coupler_rank_set;
    cur = emit(ctx, OpKind::Remap, cur, remapped,
               OpAttrs{.remap_flags = ch.remap.flags,
                       .method = ch.remap.method_override},
               cid, base + "/remap");
    type = remapped;
  } else if (effective_lag != 0) {
    // The lag shift needs somewhere to live when no Remap follows.
    cur = emit(ctx, OpKind::Interpolate, cur, type, OpAttrs{.lag = effective_lag},
               cid, base + "/lag");
  }

  // 5b. Coupler -> target: the second hop of the coupler-placement rule.
  // Elided by the same `same_as` test as every other hop, so with `C =
  // target` (no `remap_on`, or one `could_ever_emit_remap` found unused)
  // `type.rank_set` already equals `dst_rank_set` and this is a no-op --
  // today's IR, unchanged. When a real coupler ran the Remap above (or
  // moved the value there via Transport in step 4, with no Remap
  // following), this is what actually delivers the result to the channel's
  // own target placement; `insert_derived_ops` below runs on the delivered
  // value, so Constrain's two conservation integrals stay on the channel's
  // source and target -- never on the coupler, which owns no state to
  // integrate.
  if (type.rank_set >= 0 && type.rank_set != dst_rank_set) {
    const RankSet& cur_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(type.rank_set)}).ranks;
    const RankSet& dst_ranks =
        g.component(ComponentId{static_cast<std::uint32_t>(dst_rank_set)}).ranks;
    if (!cur_ranks.same_as(dst_ranks)) {
      ValueType moved = type;
      moved.rank_set = dst_rank_set;
      OpAttrs ta;
      ta.src_rank_set = type.rank_set;
      ta.dst_rank_set = dst_rank_set;
      cur = emit(ctx, OpKind::Transport, cur, moved, ta, cid,
                 base + "/transport_to_target");
      type = moved;
      OpAttrs ea;
      ea.src_rank_set = ta.src_rank_set;
      ea.dst_rank_set = ta.dst_rank_set;
      ea.reason = "after-coupler-transport";
      cur = emit(ctx, OpKind::Epoch, cur, type, ea, cid, base + "/epoch_to_target");
    }
  }

  // insert_derived_ops (Constrain, Diagnose) reads/writes through
  // ctx.field_arc[target], so bind the post-Remap arc there before calling
  // it, then re-read the result. This must run before Relax: Relax is
  // user-authored and belongs after the derived verification chain
  // (spec Section 5.2's Remap -> Constrain -> Diagnose -> Relax), and the DAG is
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
  // Review 2026-10-11 IS-9: recorded here, not reconstructed later from
  // `cur`'s producer node, because an identity channel (same region, same
  // placement, unlagged -- this loop emitted no node at all for it) leaves
  // `cur` as the bare Source arc, whose producer's own `Node::origin` is
  // deliberately `ChannelId{}` (lower()'s step 1: a Source belongs to its
  // component, not to any one reader). This is the one place that still
  // knows `cid` is the channel that put `cur` into `field_writers`.
  ctx.writer_arc_channel[cur.value()] = cid;
  ctx.field_arc[ch.target.value()] = cur;  // single-writer fast path, fixed up below
  return true;
}

/// D73. Whether channel `cid` is a CROSS-RANK WEIGHTED (non-nodal,
/// non-consistent) channel: one whose Remap, if `lower_channel()` lowers it
/// today, takes `Runner::Impl::resolve_remap()`'s general conservative /
/// MLS / tempest / intx branch rather than the nodal-distributed H / H^T
/// one (`fuse::remap_moves_placement`, fuse/model/remap_kind.hpp, answers
/// false for this channel's method and fields), whose source field's own
/// component and whose Remap placement (`remap_placement()` above) are
/// physically different sets of ranks, and that genuinely lowers to a
/// `Remap` node at all (an explicit `RemapFlags` or a region change --
/// `lower_channel()`'s own step 5 gate).
///
/// MUST STAY IN SYNC with `lower_channel()`'s own `ranks_differ` /
/// `remap_moves` / `coupler_rank_set` / `region_changes` computation
/// (above, this same file): this predicate runs in a SEPARATE, EARLIER
/// pass (`emit_remap_halo_instructions()`, called from `lower()` in
/// src/ir/lowering.cpp before step 1's `Source` loop -- see `OpKind::
/// BuildRemapHalo`'s own doc comment, fuse/ir/op.hpp, for why this must
/// precede every `Source` node in `NodeId` order), not inside
/// `lower_channel()` itself, so the two cannot share one evaluation the
/// way `remap_moves_placement`/`could_ever_emit_remap` already let
/// `validate()` and this file agree from one function. A disagreement
/// between this predicate and `lower_channel()`'s own reasoning is not a
/// silent wrong answer either way it goes: a channel this predicate missed
/// still runs headfirst into `Runner::Impl::resolve_instructions()`'s own
/// structural check (E-RT-REMAP-NO-HALO), built independently of the IR,
/// and a channel it over-emits for costs an idempotent, zero-adopted-cells
/// halo build and nothing else (`build_native_remap_halo`'s own "zero
/// adopted cells is a legitimate result" contract, native_halo.hpp).
///
/// `lowfive_bound` (the source field declares `hdf5_path`) is excluded
/// unconditionally -- the same reason `lower_channel()` forces its own
/// `ranks_differ = false` for it: the application performs this channel's
/// transfer outside the graph, FUSE moves nothing, and a halo neither
/// `Remap` nor `Transport` will ever read would be invented work.
[[nodiscard]] bool channel_needs_remap_halo(const CouplingGraph& g, ChannelId cid) {
  if (!cid.valid() || cid.value() >= g.channel_count()) return false;
  const Channel& ch = g.channel_at(cid);
  if (g.field(ch.source).hdf5_path.has_value()) return false;

  const FieldSpec& source_field = g.field(ch.source);
  const FieldSpec& target_field = g.field(ch.target);
  if (remap_moves_placement(ch.remap.method_override, source_field, target_field)) {
    return false;  // the nodal-distributed H / H^T path, not this one
  }

  const ComponentId source_component = g.owner_of(ch.source);
  const ComponentId placement = remap_placement(g, ch);

  bool ranks_differ = source_component.value() != placement.value();
  if (source_component.valid() && placement.valid()) {
    const RankSet& source_ranks = g.component(source_component).ranks;
    const RankSet& placement_ranks = g.component(placement).ranks;
    ranks_differ = !source_ranks.same_as(placement_ranks);
  }
  if (!ranks_differ) return false;

  const bool region_changes = source_field.region != target_field.region;
  return ch.remap.flags != RemapFlags::None || region_changes;
}

}  // namespace

/// D73 (automatic remap halo, owner decision 2026-10-18). The placement
/// `lower_channel()`'s own Remap step runs on: the coupler channel `ch`
/// names via `RemapSpec::remap_on` when one is declared AND `could_ever_
/// emit_remap()` says it could actually change this channel's lowering
/// (fuse/model/remap_kind.hpp; mirrors `lower_channel()`'s own `coupler_
/// rank_set`, this same file, above), the target's own owning component
/// otherwise. PUBLIC (declared in fuse/ir/lowering.hpp) so `fuse::
/// execute_remap_halo_instructions()` (fuse/backend/remap_halo_instructions.cpp)
/// can resolve the SAME placement a `BuildRemapHalo` instruction's own
/// `channel_needs_remap_halo()` gate (above) used to decide whether to
/// emit it, from one definition rather than two that could drift.
ComponentId remap_placement(const CouplingGraph& g, const Channel& ch) {
  const ComponentId dst_component = g.owner_of(ch.target);
  const bool remap_on_declared =
      ch.remap.remap_on.valid() && ch.remap.remap_on.value() < g.component_count();
  const bool remap_on_could_matter =
      remap_on_declared &&
      could_ever_emit_remap(ch.remap.flags, g.field(ch.source), g.field(ch.target),
                            ch.remap.method_override);
  return remap_on_could_matter ? ch.remap.remap_on : dst_component;
}

/// D73. One `BuildRemapHalo` instruction per channel `channel_needs_remap_
/// halo()` above names, in channel id order, with empty inputs and the
/// invalid `ArcId{}` as output -- see `OpKind::BuildRemapHalo`'s own doc
/// comment (fuse/ir/op.hpp) for why this instruction carries no SSA value
/// at all. Called from `lower()` (src/ir/lowering.cpp) BEFORE step 1's
/// `Source` loop, so every `BuildRemapHalo` node gets a lower `NodeId`
/// than every `Source` node -- `OperatorDAG::topological_order()`'s own
/// "ties broken by ascending NodeId" rule (dag.hpp) is what turns that
/// into "first in the compiled stream" once `expand()` also stamps every
/// instance at `t = 0`, `TickPhase::BeforeTicks` (the same instant
/// `Source` gets).
void emit_remap_halo_instructions(LoweringContext& ctx) {
  const CouplingGraph& g = ctx.graph;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    if (!channel_needs_remap_halo(g, cid)) continue;
    const Channel& ch = g.channel_at(cid);

    // ROUND 2 (lead's design decision 2026-10-18): carry BOTH arcs' owning
    // components through the instruction's own `attrs`, the same
    // `src_rank_set`/`dst_rank_set` fields `Transport`/`Epoch` nodes
    // already use for exactly this purpose (dump.hpp's `attrs_string`
    // only prints them for `OpKind::Transport`, so this adds no visible
    // text to any golden dump). `execute_remap_halo_instructions()`
    // (fuse/backend/remap_halo_instructions.cpp) reads them back rather
    // than re-deriving `g.owner_of(ch.source)`/`remap_placement(g, ch)` a
    // second time at execution: what this instruction acts on is frozen
    // at lowering, the same decision `channel_needs_remap_halo()` just
    // made, not whatever the graph would answer if asked again later.
    OpAttrs attrs;
    attrs.src_rank_set = static_cast<int>(g.owner_of(ch.source).value());
    attrs.dst_rank_set = static_cast<int>(remap_placement(g, ch).value());
    ctx.dag.add_node(OpKind::BuildRemapHalo, /*inputs=*/{}, /*output=*/ArcId{},
                     attrs, cid, ch.name + "/build_remap_halo");
  }
}

void register_builtin_rules(LoweringRegistry& reg) {
  reg.add("channel.default", 100, lower_channel);
}

}  // namespace fuse
