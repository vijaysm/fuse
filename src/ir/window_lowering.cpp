// Time windows: a window-scoped iteration region's field transfers
// compile through the ORDINARY per-channel rule (Remap/Transport/Epoch,
// src/ir/rules_channel.cpp) at the window's own clock, unchanged, carrying
// the waveform as K field components -- the user's chosen design. This
// file only adds the two things the ordinary rule cannot: the exporting
// member's own OpKind::WindowAdvance node (a real SSA root, EMPTY inputs,
// so the channel that reads it as its source cannot close a cycle), and
// the importing member's own such node, whose `OpAttrs::window_import`
// names the arc the field transfer DELIVERS -- the Sink's own output, on
// the importer's mesh and ranks -- never the exporter's raw output
// directly. Kept in its own file, parallel to src/ir/regions.cpp, rather
// than folded into lowering.cpp's steps: the two are independent passes
// over the same LoweringContext, and this one is specific to a single
// opt-in feature.
//
// ORDERING, and why this is two functions, not one. A WindowAdvance
// node's own output arc must exist BEFORE step 2 (the ordinary channel
// rule) runs, so that rule can read it as `ctx.source_arc[[the exporting
// member's own field]]` -- exactly as it already reads any other
// component's Source arc. But a WindowAdvance node's `window_import` --
// the PARTNER channel's Sink output -- does not exist until AFTER step 3
// (lower()'s own Sink-per-required-field loop) has run. `OperatorDAG` is
// append-only with no node-mutation API, so the WindowAdvance NODE itself
// cannot be created until both its own output arc (needed early) and its
// window_import (available late) are both in hand -- which splits into
// `seed_window_source_arcs()` (pre-creates the bare output ARCS, run
// before step 2) and `finish_window_regions()` (creates the actual NODES
// against those arcs and the by-then-available Sink arcs, run after step
// 3 and before build_iteration_regions()).

#include "fuse/ir/lowering.hpp"
#include "fuse/ir/op.hpp"

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace fuse {

namespace {

// Mirrors fuse::window_region_members() (src/schedule/window.cpp) exactly.
// Duplicated rather than called: `ir` sits below `schedule`
// (docs/dev/layering.md, tools/check-layering.py), so this layer cannot
// depend on it.
std::vector<ComponentId> members_in_first_seen_order(
    const CouplingGraph& g, const IterationRegionDecl& region) {
  std::vector<ComponentId> members;
  const auto add = [&](ComponentId c) {
    for (const ComponentId& existing : members) {
      if (existing.value() == c.value()) return;
    }
    members.push_back(c);
  };
  for (const ChannelId ch : region.channels) {
    const Channel& c = g.channel_at(ch);
    add(g.owner_of(c.source));
    add(g.owner_of(c.target));
  }
  return members;
}

// How many `components` THIS member's own exported field carries once it
// crosses a coupling window: `spec.components` scalar/vector/tensor
// entries per element, TIMES `window_sample_capacity()` (fuse/model/
// coupling_graph.hpp) samples across the window -- the user's decision,
// carried through from field width already ("a time window's field
// transfer carries the provider's K-sample waveform as K components of
// one field"). `window_sample_capacity()` is the SAME function
// validate_coupling_window()'s own E-SCHED-WINDOW-CAPACITY calls, so a
// malformed scope that lowering could not resolve above (nullopt) simply
// leaves the field at its own declared width here -- harmless, since
// schedule() refuses the malformed scope before anything reads that
// field's width for real.
std::size_t window_export_width(const CouplingGraph& g,
                                const IterationRegionDecl& decl,
                                ComponentId member, FieldId exported) {
  const FieldSpec& spec = g.field(exported);
  const CouplingWindowScope& scope = *decl.window_scope;
  // fuse::window_length() now lives in the model layer (fuse/model/
  // coupling_graph.hpp) precisely so this call is legal: ir sits above
  // model in docs/dev/layering.md's ordering, and this is the SAME
  // resolution validate_coupling_window() (schedule layer) uses for its
  // own E-SCHED-WINDOW-AMBIGUOUS/COMPONENT/INTERVAL checks -- one
  // function, so lowering and schedule() cannot disagree about this
  // scope's own window length. A malformed scope this call refuses is
  // left unwidened here (today's pre-window behaviour); the actual
  // diagnostic is validate_coupling_window()'s job, reached moments
  // later by schedule().
  const Result<Rational> window_len = window_length(g, scope);
  if (!window_len.ok()) return static_cast<std::size_t>(spec.components);
  const Rational interval = g.component(member).interval;
  const DtPolicy policy =
      scope.policy_for(g.component(member).name).value_or(DtPolicy::Adjustable);
  const std::size_t k =
      window_sample_capacity(window_len.value(), interval, policy);
  return static_cast<std::size_t>(spec.components) * k;
}

// The type a field has as it leaves its component. Duplicated from
// lowering.cpp's own file-local helper of the same shape (anonymous
// namespace, not exported) rather than shared, to keep this pass
// self-contained. `components` is the caller's own choice (today, always
// `window_export_width()`'s result for a window-scoped export) rather
// than `spec.components` directly, so this function stays a pure
// type-assembly helper with no window-width policy of its own.
ValueType window_source_type(const CouplingGraph& g, FieldId f,
                             std::size_t components) {
  const FieldSpec& spec = g.field(f);
  const ComponentId owner = g.owner_of(f);
  return ValueType{.kind = ValueKind::Instant,
                   .region = spec.region,
                   .components = static_cast<int>(components),
                   .units = spec.units,
                   .conservation = spec.conservation,
                   .time = TimeLevel::current(),
                   .rank_set = static_cast<int>(owner.value())};
}

std::string window_qualified(const CouplingGraph& g, FieldId f) {
  return g.component(g.owner_of(f)).name + "." + g.field(f).name;
}

// This member's own exported field in `region` -- the field it owns as
// some channel's SOURCE -- or an invalid FieldId if it owns none (a
// consumer-only role in this region). window_region_members()'s own
// first-seen order can name such a member; today's fixtures always give
// every member a source, but a future one-way fixture may not, so this
// stays a query rather than an assumption.
FieldId member_exported_field(const CouplingGraph& g,
                              const IterationRegionDecl& decl,
                              ComponentId member) {
  for (const ChannelId cid : decl.channels) {
    const Channel& ch = g.channel_at(cid);
    if (g.owner_of(ch.source).value() == member.value()) return ch.source;
  }
  return FieldId{};
}

// This member's own imported field in `region` -- the field it owns as
// some channel's TARGET -- or an invalid FieldId if it owns none (a
// pure-exporter role). Its Sink's output arc (`ctx.sink_arc`, populated by
// lower()'s own step 3) is what that member's WindowAdvance node's
// `window_import` names.
FieldId member_imported_field(const CouplingGraph& g,
                              const IterationRegionDecl& decl,
                              ComponentId member) {
  for (const ChannelId cid : decl.channels) {
    const Channel& ch = g.channel_at(cid);
    if (g.owner_of(ch.target).value() == member.value()) return ch.target;
  }
  return FieldId{};
}

}  // namespace

std::unordered_set<std::uint32_t> window_provided_fields(
    const CouplingGraph& g) {
  std::unordered_set<std::uint32_t> out;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl =
        g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    for (const ComponentId m : members_in_first_seen_order(g, decl)) {
      const FieldId exported = member_exported_field(g, decl, m);
      if (exported.valid()) out.insert(exported.value());
    }
  }
  return out;
}

std::unordered_set<std::uint32_t> window_imported_fields(
    const CouplingGraph& g) {
  std::unordered_set<std::uint32_t> out;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl =
        g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    for (const ComponentId m : members_in_first_seen_order(g, decl)) {
      const FieldId imported = member_imported_field(g, decl, m);
      if (imported.valid()) out.insert(imported.value());
    }
  }
  return out;
}

std::size_t window_import_width(const CouplingGraph& g, FieldId f) {
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    for (const ChannelId cid : decl.channels) {
      const Channel& ch = g.channel_at(cid);
      if (ch.target.value() != f.value()) continue;
      const ComponentId provider = g.owner_of(ch.source);
      return window_export_width(g, decl, provider, ch.source);
    }
  }
  return static_cast<std::size_t>(g.field(f).components);
}

void seed_window_source_arcs(LoweringContext& ctx) {
  const CouplingGraph& g = ctx.graph;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl =
        g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;
    for (const ComponentId m : members_in_first_seen_order(g, decl)) {
      const FieldId exported = member_exported_field(g, decl, m);
      if (!exported.valid()) continue;
      const ArcId out = ctx.dag.add_arc(
          window_source_type(g, exported,
                             window_export_width(g, decl, m, exported)),
          window_qualified(g, exported) + "/window");
      ctx.source_arc[exported.value()] = out;
      ctx.field_arc[exported.value()] = out;
    }
  }
}

void finish_window_regions(LoweringContext& ctx) {
  const CouplingGraph& g = ctx.graph;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl =
        g.iteration_region(IterationRegionIdx{i});
    if (!decl.window_scope.has_value()) continue;

    for (const ComponentId m : members_in_first_seen_order(g, decl)) {
      const FieldId exported = member_exported_field(g, decl, m);
      if (!exported.valid()) continue;  // consumer-only; no node of its own yet

      // seed_window_source_arcs() already created this member's own
      // output arc (needed early, as the exporting field transfer's
      // source_arc); find it back by field id rather than re-creating it
      // -- add_arc() has no lookup, so ctx.source_arc IS that lookup.
      const ArcId out = ctx.source_arc.at(exported.value());

      OpAttrs attrs;
      const FieldId imported = member_imported_field(g, decl, m);
      if (imported.valid()) {
        const auto it = ctx.sink_arc.find(imported.value());
        if (it != ctx.sink_arc.end()) attrs.window_import = it->second;
      }

      // origin: the channel this member sources, so build_iteration_
      // regions() (which groups nodes by origin-in-decl.channels) picks
      // this node up exactly as it already does the field transfer's own
      // chain nodes.
      ChannelId origin{};
      for (const ChannelId cid : decl.channels) {
        if (g.channel_at(cid).source == exported) { origin = cid; break; }
      }

      ctx.dag.add_node(OpKind::WindowAdvance, {}, out, attrs, origin,
                       window_qualified(g, exported) + "/window");
    }
  }
}

}  // namespace fuse
