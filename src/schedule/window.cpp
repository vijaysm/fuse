#include "fuse/schedule/window.hpp"

#include <algorithm>
#include <cmath>

namespace fuse {

std::vector<ComponentId> window_region_members(
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

Result<Rational> window_length(const CouplingGraph& g,
                               const CouplingWindowScope& scope) {
  const bool has_component = scope.window_component.has_value();
  const bool has_explicit   = scope.explicit_interval.has_value();
  if (has_component == has_explicit) {
    return Failure{"E-SCHED-WINDOW-AMBIGUOUS",
                 "CouplingWindowScope must set exactly one of "
                 "window_component / explicit_interval"};
  }
  if (has_explicit) {
    const Rational len = *scope.explicit_interval;
    if (!(len > Rational{0})) {
      return Failure{"E-SCHED-WINDOW-INTERVAL",
                   "CouplingWindowScope::explicit_interval (" +
                       len.to_string() + ") must be positive"};
    }
    return len;
  }
  const std::optional<ComponentId> found =
      g.find_component(*scope.window_component);
  if (!found.has_value()) {
    return Failure{"E-SCHED-WINDOW-COMPONENT",
                 "CouplingWindowScope::window_component '" +
                     *scope.window_component +
                     "' does not name a component in the graph"};
  }
  const Rational len = g.component(*found).interval;
  if (!(len > Rational{0})) {
    return Failure{"E-SCHED-WINDOW-INTERVAL",
                 "window-defining component '" + *scope.window_component +
                     "' has a non-positive interval (" + len.to_string() +
                     ")"};
  }
  return len;
}

namespace {

void note(Diagnostics* diags, Diagnostics& local, const std::string& code,
         const std::string& msg) {
  (diags != nullptr ? *diags : local).add(Severity::Error, code, msg);
}

}  // namespace

Result<void> validate_coupling_window(const CouplingGraph& g,
                                      const IterationRegionDecl& region,
                                      const CouplingWindowScope& scope,
                                      std::size_t waveform_capacity,
                                      Diagnostics* diags) {
  Diagnostics local;
  Result<void> first_failure = Result<void>::success();
  const auto fail = [&](std::string code, std::string msg) {
    if (first_failure.ok()) first_failure = Failure{code, msg};
    note(diags, local, code, msg);
  };

  if (region.channels.empty()) {
    fail("E-SCHED-WINDOW-EMPTY",
         "iteration region '" + region.name +
             "' declares no channels; there is no window boundary to "
             "schedule");
  }

  const Result<Rational> len = window_length(g, scope);
  if (!len.ok()) {
    fail(len.error().code, len.error().message);
  }

  if (scope.order < 2) {
    fail("E-SCHED-WINDOW-ORDER",
         "CouplingWindowScope::order (" + std::to_string(scope.order) +
             ") must be >= 2: the interpolant and extrapolant at a window "
             "boundary are required to be at least second order");
  }

  const std::vector<ComponentId> members = window_region_members(g, region);
  const auto is_member = [&](std::string_view name) {
    for (const ComponentId c : members) {
      if (g.component(c).name == name) return true;
    }
    return false;
  };

  // Every declared policy must name an actual member.
  for (const WindowMemberPolicy& p : scope.policies) {
    if (!is_member(p.component)) {
      fail("E-SCHED-WINDOW-COMPONENT",
           "CouplingWindowScope::policies names '" + p.component +
               "', which does not own either end of any channel in "
               "iteration region '" + region.name + "'");
    }
  }
  if (scope.window_component.has_value() &&
      !is_member(*scope.window_component)) {
    fail("E-SCHED-WINDOW-COMPONENT",
         "CouplingWindowScope::window_component '" +
             *scope.window_component +
             "' does not own either end of any channel in iteration "
             "region '" + region.name + "'");
  }

  // Every member must have a policy.
  for (const ComponentId c : members) {
    const std::string& name = g.component(c).name;
    bool found = false;
    for (const WindowMemberPolicy& p : scope.policies) {
      if (p.component == name) { found = true; break; }
    }
    if (!found) {
      fail("E-SCHED-WINDOW-POLICY",
           "component '" + name + "' owns an endpoint of a channel in "
               "iteration region '" + region.name +
               "' but has no DtPolicy entry in CouplingWindowScope::policies");
    }
  }

  if (!len.ok()) {
    // Nothing below can be checked meaningfully without a window length.
    return first_failure;
  }
  const Rational window = len.value();

  for (const WindowMemberPolicy& p : scope.policies) {
    if (!is_member(p.component)) continue;  // already reported above
    const std::optional<ComponentId> cid = g.find_component(p.component);
    if (!cid.has_value()) continue;  // reported by is_member() path above
    const Rational interval = g.component(*cid).interval;

    if (p.policy == DtPolicy::Adjustable && !(interval > Rational{0})) {
      fail("E-SCHED-WINDOW-SHORTEN",
           "component '" + p.component +
               "' is declared Adjustable but its own interval (" +
               interval.to_string() +
               ") is not positive, so its last substep in a window has "
               "nothing to shorten");
      continue;
    }
    if (!(interval > Rational{0})) continue;  // Fixed with interval<=0: not
                                              // this gate's concern.

    // How many samples would this member's own outgoing Waveform need to
    // cover one window? At most ceil(window/interval) substep-end samples,
    // plus the window-start sample, plus one more for a Fixed member's
    // straddling overshoot sample.
    const Rational ratio = window / interval;
    // ratio.to_double() is safe here: it is used only to size a capacity
    // check, never as a scheduling decision (those stay exact Rational
    // arithmetic in WindowCoupler).
    const double substeps_d = std::ceil(ratio.to_double() - 1e-9);
    std::size_t substeps =
        substeps_d > 0.0 ? static_cast<std::size_t>(substeps_d) : 0;
    std::size_t needed = substeps + 1 /* window-start sample */;
    if (p.policy == DtPolicy::Fixed) needed += 1;  // possible straddle
    if (needed > waveform_capacity) {
      fail("E-SCHED-WINDOW-CAPACITY",
           "component '" + p.component + "' would need " +
               std::to_string(needed) +
               " samples to cover one coupling window (window " +
               window.to_string() + " / interval " + interval.to_string() +
               "), more than the configured Waveform capacity (" +
               std::to_string(waveform_capacity) + ")");
    }
  }

  return first_failure;
}

}  // namespace fuse
