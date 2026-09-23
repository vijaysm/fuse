#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/connect.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace fuse {
namespace detail {

inline std::string channel_ctx(const CouplingGraph& g, ChannelId c) {
  return "channel '" + g.channel_at(c).name + "'";
}

/// True unless `w` is a FromComponent window naming a component that does
/// not exist in `g`. `Id::valid()` only tests the 0xFFFFFFFF sentinel, not
/// `component_count()`, so an out-of-range (but not sentinel) id — reachable
/// either from direct-API misuse or from a hand-patched descriptor, since
/// `deserialize()` does not itself validate embedded window component ids —
/// must be caught here before any `.at()`-based lookup can throw.
inline bool window_component_ok(const CouplingGraph& g, const Window& w) {
  if (w.kind != Window::Kind::FromComponent) return true;
  return w.component.valid() && w.component.value() < g.component_count();
}

/// The window length a channel's lag is measured against. Caller must have
/// already confirmed window_component_ok(g, ch.accumulate.window); this
/// still degrades to Rational{1} defensively rather than calling .at() on an
/// out-of-range id, so it can never itself be the thing that throws.
inline Rational channel_window(const CouplingGraph& g, const Channel& ch) {
  const Window& w = ch.accumulate.window;
  switch (w.kind) {
    case Window::Kind::Explicit:
      return w.length;
    case Window::Kind::FromComponent:
      return window_component_ok(g, w) ? g.component(w.component).interval
                                       : Rational{1};
    case Window::Kind::CouplingPeriod:
    default: {
      // The target's interval defines the period the value must survive.
      const ComponentId tgt = g.owner_of(ch.target);
      return g.component(tgt).interval;
    }
  }
}

}  // namespace detail

/// Spec Global Constraints: accumulate every problem, report them together.
[[nodiscard]] inline Diagnostics validate(const CouplingGraph& g,
                                          const ConnectPolicy& policy = {}) {
  Diagnostics d;

  // --- name uniqueness -----------------------------------------------------
  std::unordered_set<std::string> comp_names;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    if (!comp_names.insert(c.name).second) {
      d.add(Severity::Error, "E-DUP-COMP",
            "duplicate component name '" + c.name + "'");
    }
    if (!(c.interval > Rational{0})) {
      d.add(Severity::Error, "E-INTERVAL",
            "component '" + c.name + "' has non-positive interval " +
                c.interval.to_string());
    }
    std::unordered_set<std::string> field_names;
    for (const FieldId f : c.fields) {
      if (!field_names.insert(g.field(f).name).second) {
        d.add(Severity::Error, "E-DUP-FIELD",
              "duplicate field '" + g.field(f).name + "' in component '" +
                  c.name + "'");
      }
    }
  }

  // --- reference integrity -------------------------------------------------
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    if (!f.region.valid() || f.region.value() >= g.region_count()) {
      d.add(Severity::Error, "E-BAD-REF",
            "field '" + f.name + "' references an invalid region");
    }
    // A field must carry at least one component per entity. Nothing rejected
    // this before, so a graph with components <= 0 validated clean and only
    // failed much later in plan_buffers -- where the count is multiplied by
    // the region size into a std::size_t, so components == -1 produced a
    // total of 18446744073709551416 rather than a diagnostic. Unsigned
    // overflow is *defined* to wrap, so neither the assert nor the sanitizer
    // leg sees it. It belongs here, at the authoring gate, where it can be
    // reported alongside every other problem instead of throwing from the
    // scheduler.
    if (f.components < 1) {
      d.add(Severity::Error, "E-SHAPE",
            "field '" + f.name + "' has a non-positive component count " +
                std::to_string(f.components));
    }
  }
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    if (!r.mesh.valid() || r.mesh.value() >= g.mesh_count()) {
      d.add(Severity::Error, "E-BAD-REF",
            "region '" + r.name + "' references an invalid mesh");
    }
  }

  // --- per-channel rules ---------------------------------------------------
  std::unordered_map<std::uint32_t, int> writers;  // target field -> count

  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    const auto& ch = g.channel_at(cid);
    const std::string ctx = detail::channel_ctx(g, cid);

    const bool src_ok = ch.source.valid() && ch.source.value() < g.field_count();
    const bool tgt_ok = ch.target.valid() && ch.target.value() < g.field_count();
    if (!src_ok || !tgt_ok) {
      d.add(Severity::Error, "E-BAD-REF",
            "channel references an invalid field", ctx);
      continue;  // every later rule would dereference garbage
    }

    // A channel whose either end belongs to a component placed NOWHERE.
    //
    // An empty placement is legal on its own -- it is how a component is
    // switched off for one run (docs/user/placement.md) -- but a channel out
    // of or into one asks to move data to or from a component that exists on
    // no rank, which is a modelling error. It was previously diagnosed two
    // layers down, as MpiTransport::build_plan refusing an empty rank set
    // (E-TRANSPORT-RANKS): a true refusal, but one that told the user about a
    // rank set when the mistake was in the graph. Caught here, the message can
    // name the component and the channel, and it is caught for every transport
    // rather than only the MPI one.
    for (const bool source_side : {true, false}) {
      const FieldId fid = source_side ? ch.source : ch.target;
      const ComponentId owner = g.owner_of(fid);
      if (!owner.valid() || owner.value() >= g.component_count()) continue;
      const auto& comp = g.component(owner);
      if (!comp.ranks.empty()) continue;
      d.add(Severity::Error, "E-PLACEMENT-EMPTY",
            "component '" + comp.name + "' is placed on no rank, so this " +
                (source_side ? std::string("channel has nothing to read from")
                             : std::string("channel has nowhere to deliver")) +
                "; an empty placement is legal only for a component with no "
                "channels -- give it a placement, or remove the channel for "
                "runs where it is switched off",
            ctx);
    }

    const auto& sf = g.field(ch.source);
    const auto& tf = g.field(ch.target);
    ++writers[ch.target.value()];

    // The unit gap goes through the spine's graded stages (fuse/model/connect.hpp),
    // so the three outcomes stay distinct instead of collapsing into one error:
    //   * different DIMENSIONS      -> E-UNITS, an incompatibility;
    //   * an exact zero-offset SCALE -> an inferred UnitScale operator, reported as
    //     a warning naming the factor, because the user should see an operator they
    //     did not write;
    //   * an OFFSET change, or a scale under a policy that forbids insertion
    //     -> E-UNITS-SCALE, refused. Moving the zero of a scale changes meaning.
    const auto connection = check_channel(g, cid, policy);
    for (const auto& item : connection.diagnostics) {
      if (item.stage != ConnectionStage::Unit) continue;
      switch (item.reason) {
        case ConnectionReason::DimensionMismatch:
          d.add(Severity::Error, "E-UNITS",
                "unit mismatch: source '" + sf.name + "' is " + sf.units.to_string() +
                    ", target '" + tf.name + "' is " + tf.units.to_string(), ctx);
          break;
        case ConnectionReason::UnitsUndeclared:
          // Unknown dimensions, which is NOT dimensionless. Refused rather than
          // assumed, because assuming would connect two unrelated quantities.
          d.add(Severity::Error, "E-UNITS-UNDECLARED",
                "a unit could not be interpreted, so the two cannot be shown to agree: " +
                    item.detail, ctx);
          break;
        default:
          d.add(Severity::Error, "E-UNITS-SCALE",
                std::string("unit conversion refused (") +
                    std::string(connection_reason_name(item.reason)) + "): " + item.detail, ctx);
          break;
      }
    }
    for (const auto& step : connection.chain.steps) {
      if (step.kind == OperatorKind::UnitScale && !step.explicitly_requested) {
        d.add(Severity::Warning, "I-UNIT-SCALE-INSERTED",
              "inserting an exact unit scale of " + step.scale.to_string() + " from '" + sf.name +
                  "' to '" + tf.name + "'", ctx);
      }
    }

    if (!same_shape(sf, tf)) {
      d.add(Severity::Error, "E-SHAPE",
            "component count mismatch: " + std::to_string(sf.components) +
                " vs " + std::to_string(tf.components), ctx);
    }
    if (sf.conservation != tf.conservation) {
      d.add(Severity::Error, "E-CONSERV",
            std::string{"conservation class mismatch: "} +
                to_cstring(sf.conservation) + " vs " +
                to_cstring(tf.conservation), ctx);
    }
    if (sf.intent == Intent::Requires) {
      d.add(Severity::Error, "E-INTENT",
            "source field '" + sf.name + "' is Requires; must Provide", ctx);
    }
    if (tf.intent == Intent::Provides) {
      d.add(Severity::Error, "E-INTENT",
            "target field '" + tf.name + "' is Provides; must Require", ctx);
    }
    if (g.owner_of(ch.source) == g.owner_of(ch.target)) {
      d.add(Severity::Error, "E-SELF",
            "source and target belong to the same component", ctx);
    }

    // A FromComponent window that names a nonexistent component is a
    // dangling reference, same class as the field/region/mesh checks above.
    // Reachable from a byte-patched descriptor (deserialize() does not
    // itself validate embedded window component ids), so this must be
    // checked before channel_window() ever dereferences it.
    const bool accum_window_ok =
        detail::window_component_ok(g, ch.accumulate.window);
    if (!accum_window_ok) {
      d.add(Severity::Error, "E-BAD-REF",
            "channel's accumulate window references an invalid component",
            ctx);
    }
    if (!detail::window_component_ok(g, ch.spacetime.window)) {
      d.add(Severity::Error, "E-BAD-REF",
            "channel's spacetime window references an invalid component",
            ctx);
    }

    // Spec §4.3: OASIS documents lag beyond the coupling period as a
    // restartability and deadlock hazard.
    if (ch.lag != 0 && accum_window_ok) {
      const Rational span =
          Rational{std::abs(ch.lag)} * g.component(g.owner_of(ch.source)).interval;
      const Rational window = detail::channel_window(g, ch);
      if (span > window) {
        d.add(Severity::Error, "E-LAG",
              "lag spans " + span.to_string() + " but the coupling window is " +
                  window.to_string(), ctx);
      }
    }

    if (!ch.time_level_mismatch_reason.empty()) {
      d.add(Severity::Warning, "W-MISMATCH-OK",
            "time-level mismatch permitted: " + ch.time_level_mismatch_reason,
            ctx);
    }
  }

  // --- cross-channel rules -------------------------------------------------
  // Two channels writing one field is an ERROR, not a warning, and the
  // severity is the whole point of the check.
  //
  // Combining writers is now supported, but only when the user says HOW:
  // `graph.merge(field, MergeOp::…)`. Summing two heat fluxes and averaging two
  // temperatures are different physics, and nothing in the field descriptions
  // chooses between them, so the compiler must not guess.
  //
  // Without a declared rule this stays an error, and the severity is the point.
  // The compiler used to bind one arc per target field, so every writer but the
  // last was simply dropped: its chain dead-ended with no consumer and the
  // target component never depended on it. That silently deletes a declared
  // dependence, which is how the parent's acyclicity proof came to certify
  // graphs containing real algebraic loops (232 of 19,181 in a fuzz sweep). A
  // warning let that clear the authoring gate.
  //
  // Iterated in ascending field id rather than in `writers`' unordered_map
  // order, so the diagnostic text is reproducible run to run.
  std::vector<std::uint32_t> multi_written;
  for (const auto& [field_value, count] : writers) {
    if (count > 1) multi_written.push_back(field_value);
  }
  std::sort(multi_written.begin(), multi_written.end());
  for (const std::uint32_t field_value : multi_written) {
    if (g.merge_for(FieldId{field_value}) != nullptr) continue;  // the user said how
    d.add(Severity::Error, "E-MULTI-WRITER",
          "field '" + g.field(FieldId{field_value}).name + "' is written by " +
              std::to_string(writers[field_value]) +
              " channels, and no rule says how to combine them. Summing two heat "
              "fluxes and averaging two temperatures are different physics, and "
              "nothing in the field descriptions chooses between them. Declare one "
              "with graph.merge(field, MergeOp::Sum|Mean|First), or retarget all but "
              "one channel.");
  }

  // A declared merge that combines nothing usually means a channel was
  // renamed or retargeted and the merge was left behind, so a field is now
  // written by one channel under a rule that claims it is being combined.
  for (std::size_t i = 0; i < g.merge_count(); ++i) {
    const auto& entry = g.merge_at(i);
    if (entry.first >= g.field_count()) {
      d.add(Severity::Error, "E-MERGE-FIELD",
            "a merge is declared for field id " + std::to_string(entry.first) +
                ", which does not exist");
      continue;
    }
    const auto it = writers.find(entry.first);
    const int n = it == writers.end() ? 0 : it->second;
    if (n < 2) {
      d.add(Severity::Warning, "W-MERGE-UNUSED",
            "field '" + g.field(FieldId{entry.first}).name + "' declares a " +
                std::string(merge_op_name(entry.second.op)) +
                " merge but is written by " + std::to_string(n) +
                " channel(s); the merge combines nothing");
    }
  }

  std::unordered_set<std::uint32_t> sources;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& ch = g.channel_at(ChannelId{i});
    if (ch.source.valid()) sources.insert(ch.source.value());
  }
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    if (f.intent == Intent::Provides && !sources.count(i)) {
      d.add(Severity::Warning, "W-NO-CONSUMER",
            "field '" + f.name + "' is Provides but no channel consumes it");
    }
  }

  return d;
}

}  // namespace fuse
