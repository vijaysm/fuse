#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/connect.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/remap_kind.hpp"

namespace fuse {
namespace detail {

inline std::string channel_ctx(const CouplingGraph& g, ChannelId c) {
  return "channel '" + g.channel_at(c).name + "'";
}

/// True unless `w` is a FromComponent window naming a component that does
/// not exist in `g`. `Id::valid()` only tests the 0xFFFFFFFF sentinel, not
/// `component_count()`, so an out-of-range (but not sentinel) id -- reachable
/// either from direct-API misuse or from a hand-patched descriptor, since
/// `deserialize()` does not itself validate embedded window component ids --
/// must be caught here before any `.at()`-based lookup can throw.
inline bool window_component_ok(const CouplingGraph& g, const Window& w) {
  if (w.kind != Window::Kind::FromComponent) return true;
  return w.component.valid() && w.component.value() < g.component_count();
}

/// The window length a channel's lag is measured against. Caller must have
/// already confirmed window_component_ok(g, ch.accumulate.window); this
/// still degrades to Rational{1} defensively rather than calling .at() on an
/// out-of-range id, so it can never itself be the thing that throws.
///
/// A channel that belongs to a `CouplingWindowScope`-governed
/// `IterationRegionDecl` measures its lag against THAT region's own window
/// length (`fuse::window_length()`, `fuse/model/coupling_graph.hpp` --
/// the SAME function `window_lowering.cpp` and `validate_coupling_window()`
/// resolve, so this check and the schedule itself cannot disagree about
/// what the window is), never the target component's own `.interval` --
/// that fallback is for a channel outside any time window entirely (the
/// `Window::Kind::CouplingPeriod` case below), where "the period the value
/// must survive" genuinely is the target's own tick. Conflating the two
/// was invisible as long as every window-scoped test graph happened to set
/// dt_u == window: answering "the target's
/// own interval" would only ever match the window's real length by
/// coincidence.
inline Rational channel_window(const CouplingGraph& g, ChannelId cid,
                               const Channel& ch) {
  if (cid.valid() && cid.value() < g.channel_count()) {
    for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
      const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});
      if (!decl.window_scope.has_value()) continue;
      bool member = false;
      for (const ChannelId c : decl.channels) {
        if (c.value() == cid.value()) { member = true; break; }
      }
      if (!member) continue;
      const Result<Rational> wl = window_length(g, *decl.window_scope);
      return wl.ok() ? wl.value() : Rational{1};
    }
  }
  const Window& w = ch.accumulate.window;
  switch (w.kind) {
    case Window::Kind::Explicit:
      return w.length;
    case Window::Kind::FromComponent:
      return window_component_ok(g, w) ? g.component(w.component).interval
                                       : Rational{1};
    case Window::Kind::CouplingPeriod: {
      // The target's interval defines the period the value must survive.
      // Reached only for a channel OUTSIDE any window-scoped region --
      // the loop above already returned for one inside.
      const ComponentId tgt = g.owner_of(ch.target);
      return g.component(tgt).interval;
    }
  }
  // There is deliberately no `default:` above: it would fold an
  // out-of-range `Window::Kind` into the `CouplingPeriod` case, silently
  // reinterpreting a corrupted value rather than refusing it.
  // `deserialize()` (fuse/model/descriptor.hpp) range-checks every
  // byte a `Window::Kind` can be built from, so every graph this
  // function can see has one of the three enumerators above. Reaching
  // here means a `Window` was constructed directly with a `static_cast`
  // outside both the deserializer and the builder factories
  // (`Window::coupling_period()`/`from()`/`explicit_length()`) -- a
  // programmer error, not a trust-boundary input, and `FUSE_ASSERT` is
  // the project's own spelling for "this cannot happen" (fuse/base/
  // error.hpp).
  FUSE_ASSERT(false, "channel_window: Window::Kind {} is out of range",
             static_cast<int>(w.kind));
  return Rational{1};
}

}  // namespace detail

/// Spec Global Constraints: accumulate every problem, report them together.
[[nodiscard]] inline Diagnostics validate(const CouplingGraph& g,
                                          const ConnectPolicy& policy = {}) {
  Diagnostics d;

  // --- name uniqueness -----------------------------------------------------
  std::unordered_set<std::string> comp_names;
  // Whether ANY component in the
  // graph declares LaunchMode::LowFive. Checked against every field's own
  // hdf5_path below (E-LOWFIVE-NO-COMPONENT) -- a field whose bytes move
  // through LowFive-intercepted HDF5 needs at least one component in the
  // graph actually set up to run that interception (fuse_setup()'s own
  // per-component dispatch, docs/user/lowfive.md); this check does not
  // require that component to be the FIELD's own owner, since the single-
  // process interception path (the only one measured end to end) makes
  // the connector a process-wide registration, not a per-component one.
  bool any_lowfive_component = false;
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
    // A coupler is a placement, not a physics participant: it has no
    // register_fields() call and no Source/Sink of its own, so a field
    // declared on it would be silently unreachable -- nothing ever produces
    // or consumes it. Caught here rather than left to surface as an
    // unrelated "no producing arc" failure much later in lowering.
    if (c.kind == ComponentKind::Coupler && !c.fields.empty()) {
      d.add(Severity::Error, "E-COUPLER-FIELDS",
            "coupler '" + c.name + "' owns " +
                std::to_string(c.fields.size()) +
                " field(s); a coupler is a placement, not a physics "
                "component, and owns no fields");
    }
    std::unordered_set<std::string> field_names;
    for (const FieldId f : c.fields) {
      if (!field_names.insert(g.field(f).name).second) {
        d.add(Severity::Error, "E-DUP-FIELD",
              "duplicate field '" + g.field(f).name + "' in component '" +
                  c.name + "'");
      }
    }
    // ---- launch checks: begin ---------------------------------------------
    // Per-member LaunchSpec checks that need no OTHER component's
    // declaration (fuse/model/launch.hpp). Cross-component questions --
    // whether the launched components' RankSets actually partition the
    // declared world, whether an Mpmd appnum collides with another
    // component's, whether a Connect pair agrees on exactly one acceptor --
    // are fuse::transport::validate_launch_layout()'s job instead
    // (include/fuse/transport/connect.hpp), where the message can name the
    // other component or the actual runtime layout it disagrees with.
    if (c.launch.mode == LaunchMode::Mpmd && c.launch.appnum < 0) {
      d.add(Severity::Error, "E-LAUNCH-APPNUM",
            "component '" + c.name +
                "' declares an mpmd launch with no appnum set (appnum " +
                std::to_string(c.launch.appnum) +
                "); the MPI_APPNUM this executable is launched as must be "
                "given explicitly");
    }
    if (c.launch.mode == LaunchMode::Connect && c.launch.port_file.empty() &&
        c.launch.service_name.empty()) {
      d.add(Severity::Error, "E-LAUNCH-CONNECT-TARGET",
            "component '" + c.name +
                "' declares a connect launch with neither port_file nor "
                "service_name set; connect() has no way to publish or find "
                "the rendezvous port");
    }
    // ---- launch checks: end -----------------------------------------------
    // ---- plugin launch checks: begin --------------------------------------
    if (c.launch.mode == LaunchMode::Plugin && c.launch.plugin_path.empty()) {
      d.add(Severity::Error, "E-LAUNCH-PLUGIN-PATH",
            "component '" + c.name +
                "' declares a plugin launch with no plugin_path set; "
                "fuse::abi_detail::Registry has no shared object to dlopen()");
    }
    // ---- plugin launch checks: end ----------------------------------------
    if (c.launch.mode == LaunchMode::LowFive) {
      any_lowfive_component = true;
      // A LaunchMode::LowFive component is a placement, the same role a
      // Coupler already plays for an ordinary remap_on routing -- it has
      // no register_fields() of its own to call, no advance(), and gets
      // no Tick (fuse_initialize()'s own wiring loop, src/abi/
      // participant.cpp, exempts a ComponentKind::Coupler entry for
      // exactly that reason). Declaring kind == Physics (the
      // ComponentDecl default) or owning fields would make it a THIRD
      // kind of participant this graph has no machinery for at all --
      // MEASURED: a marker declared with no explicit .kind gets a spurious
      // externally-driven component (resolved to it at period start, with
      // no step call for it anywhere in the caller's own loop), so the
      // marker must declare .kind = Coupler by hand. This check pins the
      // RULE, not any tool's habit of getting it right.
      if (c.kind != ComponentKind::Coupler || !c.fields.empty()) {
        d.add(Severity::Error, "E-LOWFIVE-COMPONENT-KIND",
              "component '" + c.name +
                  "' declares LaunchMode::LowFive but is not a bare "
                  "placement (kind=Coupler, no fields); it has kind=" +
                  (c.kind == ComponentKind::Coupler ? "Coupler" : "Physics") +
                  " and " + std::to_string(c.fields.size()) +
                  " field(s) -- a LowFive marker component is never "
                  "stepped and owns no fields of its own, the same rule "
                  "E-COUPLER-FIELDS already enforces for an ordinary "
                  "coupler placement");
      }
    }
  }

  // --- reference integrity -------------------------------------------------
  // hdf5_path -> the first field id
  // seen claiming it, so a later field naming the SAME exact path can be
  // reported against the earlier one. Ascending field-id order, like the
  // multi-writer check further down, so the diagnostic text is
  // reproducible run to run. A leading '*' in the file component is a
  // wildcard (LowFiveFieldBindingTable::add's own rule, fuse/transport/
  // lowfive.hpp) and is deliberately excluded here too: two wildcard
  // bindings on different real files are the ordinary multi-field case,
  // and this layer has no more way to know the real file than that one
  // does.
  std::unordered_map<std::string, std::uint32_t> hdf5_path_first_field;
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
    if (f.hdf5_path.has_value()) {
      if (!any_lowfive_component) {
        d.add(Severity::Error, "E-LOWFIVE-NO-COMPONENT",
              "field '" + f.name + "' sets hdf5_path '" + *f.hdf5_path +
                  "', but no component in the graph declares "
                  "LaunchMode::LowFive; nothing would set up the "
                  "interception this field's transfer depends on");
      }
      const bool is_wildcard = !f.hdf5_path->empty() && f.hdf5_path->front() == '*';
      if (!is_wildcard) {
        const auto [it, inserted] =
            hdf5_path_first_field.try_emplace(*f.hdf5_path, i);
        if (!inserted) {
          d.add(Severity::Error, "E-LOWFIVE-DUPPATH",
                "field '" + f.name + "' and field '" +
                    g.field(FieldId{it->second}).name +
                    "' both set hdf5_path '" + *f.hdf5_path +
                    "'; two fields would silently overwrite each other's "
                    "dataset");
        }
      }
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

    // `remap_on` names WHERE this channel's
    // Remap runs. Invalid (the default) means "the target's own placement"
    // and needs no check at all.
    if (ch.remap.remap_on.valid()) {
      const ComponentId ro = ch.remap.remap_on;
      if (ro.value() >= g.component_count()) {
        d.add(Severity::Error, "E-BAD-REF",
              "channel's remap_on references an invalid component", ctx);
      } else {
        const ComponentDecl& roc = g.component(ro);
        const bool names_a_coupler = roc.kind == ComponentKind::Coupler;
        const bool names_an_endpoint =
            ro == g.owner_of(ch.source) || ro == g.owner_of(ch.target);
        if (!names_a_coupler && !names_an_endpoint) {
          d.add(Severity::Error, "E-CHANNEL-REMAP-ON",
                "remap_on names component '" + roc.name +
                    "', which is neither a coupler nor this channel's own "
                    "source or target component",
                ctx);
        } else if (!could_ever_emit_remap(ch.remap.flags, g.field(ch.source), g.field(ch.target), ch.remap.method_override)) {
          // A remap_on that would never see a Remap emitted for it: the
          // setting has no effect. Not an error -- the graph may have named
          // it in anticipation of a flag that got removed, or for symmetry
          // with a sibling channel -- but silently doing nothing is exactly
          // the kind of surprise a warning exists to flag.
          d.add(Severity::Warning, "W-CHANNEL-REMAP-ON-UNUSED",
                "channel names a remap_on placement, but no Remap operator "
                "would be emitted for it (no remap flags, matching regions, "
                "and no distributed-remap placement move); the setting is "
                "unused",
                ctx);
        }
        // Whether the placement itself is usable, but only when a Remap
        // will actually route through it -- an empty coupler placement that
        // nothing uses is not a modelling error (docs/user/placement.md: an
        // empty placement is legal for a component with no channels; a
        // coupler no channel's remap_on names has exactly that shape).
        if (roc.ranks.empty() && could_ever_emit_remap(ch.remap.flags, g.field(ch.source), g.field(ch.target), ch.remap.method_override)) {
          d.add(Severity::Error, "E-PLACEMENT-EMPTY",
                "remap_on names component '" + roc.name +
                    "', which is placed on no rank, so this channel's Remap "
                    "has nowhere to run; give it a placement, or remove "
                    "remap_on",
                ctx);
        }
      }
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
    //
    // This loop maps the Location- and Conservation-stage refusals as well as
    // Unit, so that a graph does not validate clean and then fail much later,
    // inside `lower()`, with no authoring-time explanation. Shape-stage
    // diagnostics are NOT added here: `to_semantics()`
    // derives `FieldSemantics::shape`/`components` from the same
    // `FieldSpec::components` the pre-existing `E-SHAPE` check below already
    // compares, so a Shape-stage refusal from `check_channel()` fires on
    // exactly the same condition `E-SHAPE` already reports -- adding a
    // second code for it would be a duplicate, not a silent gap.
    // Temporal and Capability never fire through this path at all: neither
    // axis is populated from a `FieldSpec` (connect.hpp's own top-of-file
    // comment: continuity, temporal interpretation, ownership, access and
    // update all stay `Unspecified`), so `check_temporal`/`check_capabilities`
    // see no interval, no capability bits and no requirement in a
    // graph-built `ConnectionRequest` and never refuse.
    const auto connection = check_channel(g, cid, policy);
    for (const auto& item : connection.diagnostics) {
      switch (item.stage) {
        case ConnectionStage::Unit:
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
          break;
        case ConnectionStage::Location:
          // LocationChangeRequiresTransfer / ContinuityChangeRequiresTransfer:
          // the two fields are represented differently (location, continuity
          // or basis degree) and the channel declares no remap at all and no
          // region change -- `declared_transfer(g, ch)` (connect.hpp) stayed
          // at `Copy`. A plain copy cannot change where or how a value is
          // represented.
          d.add(Severity::Error, "E-CONN-LOCATION",
                "channel from '" + sf.name + "' to '" + tf.name + "' changes " +
                    std::string(connection_reason_name(item.reason)) +
                    " but declares a plain copy (no remap, same region); declare "
                    "a remap for this channel (ChannelBuilder::remap) or route it "
                    "through a region change",
                ctx);
          break;
        case ConnectionStage::Conservation: {
          // ConservationNotProvidedByTransfer: the target field is
          // `Conservation::Extensive` (`required_conservation()`,
          // connect.hpp) but the channel's declared transfer
          // (`declared_transfer()`, same file) does not carry a strong
          // enough `ConservationGuarantee` -- a plain Copy, an uncoerced
          // Interpolation, or a Remap with no `RemapFlags::Conservative`.
          // Named here rather than left for `lower()`'s own, unrelated
          // `E-IR-UNLOWERABLE`/`E-IR-TYPE` refusals, so the authoring gate
          // says why in terms of the rule the user actually broke.
          const TransferSemantics transfer = declared_transfer(g, ch);
          d.add(Severity::Error, "E-CONN-CONSERVATION",
                "channel into '" + tf.name + "' requires " +
                    std::string(conservation_guarantee_name(required_conservation(tf))) +
                    " (its target field is Conservation::Extensive), but the "
                    "declared transfer (" + std::string(transfer_kind_name(transfer.kind)) +
                    ") provides " + std::string(conservation_guarantee_name(transfer.conservation)) +
                    "; declare the channel conservative "
                    "(ChannelBuilder::conservative()) if its transfer really "
                    "does conserve the field globally, or mark the target "
                    "field Conservation::Intensive if it does not need to",
                ctx);
          break;
        }
        default:
          break;  // Identity/Temporal/Capability: never populated from a FieldSpec (see above)
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

    // Spec Section 4.3: OASIS documents lag beyond the coupling period as a
    // restartability and deadlock hazard.
    //
    // `ch.lag` is a raw `int32_t` read off the wire with no range
    // check (descriptor.hpp), so it is widened to `int64_t` before
    // negating: `std::abs(ch.lag)` would be UB for `ch.lag == INT_MIN`.
    // The multiply uses `try_mul` rather than the throwing `*` operator,
    // which can `FUSE_REFUSE` on overflow and escape `validate()`, whose
    // whole contract is "every problem is reported together", never a
    // throw; `try_mul` reports an unrepresentable span as a diagnostic
    // instead.
    if (ch.lag != 0 && accum_window_ok) {
      const std::int64_t lag_magnitude =
          std::abs(static_cast<std::int64_t>(ch.lag));
      const auto span = Rational::try_mul(
          Rational{lag_magnitude}, g.component(g.owner_of(ch.source)).interval);
      if (!span) {
        d.add(Severity::Error, "E-LAG",
              "channel's lag (" + std::to_string(ch.lag) + ") times the "
                  "source component's interval is not representable as an "
                  "exact rational",
              ctx);
      } else {
        const Rational window = detail::channel_window(g, cid, ch);
        if (*span > window) {
          d.add(Severity::Error, "E-LAG",
                "lag spans " + span->to_string() + " but the coupling window is " +
                    window.to_string(), ctx);
        }
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
  // `graph.merge(field, MergeOp::...)`. Summing two heat fluxes and averaging two
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
