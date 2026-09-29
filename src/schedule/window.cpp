#include "fuse/schedule/window.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

namespace fuse {

namespace {

// GaussSeidel needs at least one field transfer in the region's own cycle
// broken by `Channel::lag`, exactly as an ordinary (non-window) iteration
// region does: a cycle with every edge unlagged has no valid sweep order
// (each member would need the OTHER's not-yet-computed current iterate).
// Built directly over the DECLARED graph -- `region.channels`, not the
// lowered/expanded DAG -- so this runs at `validate_coupling_window()`
// time, before any lowering, with a message that names the fix.
//
// Only UNLAGGED channels are edges: a lagged channel already reads its
// partner's PREVIOUS iterate, so it cannot be part of an unresolvable
// cycle whatever else the group contains. Self-loops (a channel whose
// source and target are owned by the same component) contribute nothing
// to a cross-component cycle and are skipped.
[[nodiscard]] bool has_unbroken_cycle(const CouplingGraph& g,
                                      const IterationRegionDecl& region) {
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> adj;
  for (const ChannelId c : region.channels) {
    const Channel& ch = g.channel_at(c);
    if (ch.lag != 0) continue;
    const std::uint32_t from = g.owner_of(ch.source).value();
    const std::uint32_t to   = g.owner_of(ch.target).value();
    if (from == to) continue;
    adj[from].push_back(to);
  }

  std::unordered_set<std::uint32_t> visited;
  std::unordered_set<std::uint32_t> on_stack;
  const auto dfs = [&](std::uint32_t start) {
    // Explicit stack, not recursion: the number of members in one window
    // is small, but nothing here bounds it, and a cycle check must not
    // itself be the thing that overflows the call stack.
    std::vector<std::pair<std::uint32_t, std::size_t>> stack;
    stack.push_back({start, 0});
    visited.insert(start);
    on_stack.insert(start);
    while (!stack.empty()) {
      auto& [v, next] = stack.back();
      const auto it = adj.find(v);
      const std::vector<std::uint32_t>* nbrs =
          (it == adj.end()) ? nullptr : &it->second;
      if (nbrs != nullptr && next < nbrs->size()) {
        const std::uint32_t w = (*nbrs)[next];
        ++next;
        if (on_stack.count(w)) return true;
        if (!visited.count(w)) {
          visited.insert(w);
          on_stack.insert(w);
          stack.push_back({w, 0});
        }
      } else {
        on_stack.erase(v);
        stack.pop_back();
      }
    }
    return false;
  };

  for (const auto& [v, unused] : adj) {
    (void)unused;
    if (!visited.count(v) && dfs(v)) return true;
  }
  return false;
}

}  // namespace

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

  if (scope.iteration == WaveformIteration::GaussSeidel &&
      has_unbroken_cycle(g, region)) {
    fail("E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG",
         "iteration region '" + region.name +
             "' is WaveformIteration::GaussSeidel, but every field "
             "transfer among its members is unlagged: the group closes a "
             "cycle with nothing to break it, exactly as an ordinary "
             "(non-window) iteration region with no lag does. Either lag "
             "one direction (Channel::lag(...)), or set "
             "CouplingWindowScope::iteration = WaveformIteration::Jacobi, "
             "which needs no .lag() at all");
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
