#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/iteration.hpp"

/// Time windows track (spec-in-progress §5.4): an OPT-IN alternative to
/// today's single-clock iteration region, for a region whose channels span
/// two (or more) clocks that do not coincide for a long time -- the 7:13
/// case the multirate probe (`test/schedule/multirate_region_test.cpp`)
/// pins as refused today.
///
/// WHY THIS IS A SIBLING PARAMETER, NOT A FIELD ON `IterationRegionDecl`.
/// The natural spelling is a member on the region declaration itself
/// (`fuse/model/coupling_graph.hpp`). That file is owned by a different
/// track of this port and is not touched here; see the "opt-in on the
/// region declaration" ruling recorded as an open question. `schedule()`'s
/// contiguous-run rule (`mark_regions`, src/schedule/scheduler.cpp) is
/// UNCHANGED and keeps refusing a multi-clock region with `E-SCHED-REGION`
/// exactly as it does today (`MultirateRegionProbe.*`); the message now
/// names this header as where the opt-in lives. `CouplingWindowScope` is the
/// opt-in's payload, carried alongside the graph rather than inside it, and
/// `validate_coupling_window()` is the legality gate a caller (today: a
/// test harness or a model driving `WindowCoupler` directly;
/// designed-but-not-wired: `schedule()` itself, once the region declaration
/// can carry it) must pass before running a window-scoped region.
namespace fuse {

/// One member's declared role inside a coupling window.
struct WindowMemberPolicy {
  std::string component;  ///< must name a `ComponentDecl` in the graph
  DtPolicy    policy = DtPolicy::Adjustable;
};

/// The opt-in payload. Exactly one of `window_component` /
/// `explicit_interval` must be set -- naming a component ties the window to
/// whatever interval that component is later re-declared with (the common
/// case: "couple every time the slow solver steps"); an explicit interval is
/// for a window with no single natural owner.
struct CouplingWindowScope {
  std::optional<std::string> window_component;
  std::optional<Rational>    explicit_interval;

  /// Every component that owns either end of one of the region's channels
  /// must appear here exactly once. `validate_coupling_window()` checks
  /// this; nothing here does, so an incomplete scope can be built
  /// incrementally with `.policies.push_back(...)` before validation.
  std::vector<WindowMemberPolicy> policies;

  /// Order of the temporal interpolant/extrapolant at the window boundary
  /// (`WindowCoupler`'s Waveform basis and its extrapolation degree). Must
  /// be >= 2 per the user's ruling ("interpolation and extrapolation are at
  /// least second order"); `Adjustable`/`Fixed` do not change this floor.
  int order = 2;

  [[nodiscard]] std::optional<DtPolicy> policy_for(
      std::string_view component) const {
    for (const WindowMemberPolicy& p : policies) {
      if (p.component == component) return p.policy;
    }
    return std::nullopt;
  }
};

/// The window's length: the named component's own interval, or the explicit
/// one. `E-SCHED-WINDOW-AMBIGUOUS` if both or neither are set;
/// `E-SCHED-WINDOW-COMPONENT` if `window_component` does not name a
/// component in `g`; `E-SCHED-WINDOW-INTERVAL` if the resulting length is
/// not positive.
[[nodiscard]] Result<Rational> window_length(const CouplingGraph& g,
                                             const CouplingWindowScope& scope);

/// Every distinct component that owns the source or the target field of one
/// of `region`'s channels, in first-seen order. What `scope.policies` is
/// checked against.
[[nodiscard]] std::vector<ComponentId> window_region_members(
    const CouplingGraph& g, const IterationRegionDecl& region);

/// The legality gate for a coupling window, run before any `WindowCoupler`
/// is built from `g`/`region`/`scope`. Every check is independent and all
/// are reported (Global Constraints, like `validate()`); the returned
/// `Result` carries the FIRST one for a caller that wants a single
/// diagnosis, `diags` (out param) every one found.
///
/// Refusals:
///   E-SCHED-WINDOW-EMPTY       `region.channels` is empty: no boundary to
///                               window.
///   E-SCHED-WINDOW-AMBIGUOUS   see `window_length()`.
///   E-SCHED-WINDOW-COMPONENT   `window_component` (or a `policies` entry)
///                               does not name a component in `g`, or names
///                               one that is not a member of `region`
///                               (`window_region_members()`).
///   E-SCHED-WINDOW-INTERVAL    the window length is not positive.
///   E-SCHED-WINDOW-POLICY      a member of `region` has no entry in
///                               `scope.policies`, or `policies` names a
///                               component that is not a member.
///   E-SCHED-WINDOW-ORDER       `scope.order < 2`.
///   E-SCHED-WINDOW-SHORTEN     an `Adjustable` member's own interval is not
///                               positive (there is no substep to shorten).
///   E-SCHED-WINDOW-CAPACITY    a member's interval is small enough,
///                               relative to the window and
///                               `waveform_capacity`, that its own trajectory
///                               would need more than `waveform_capacity`
///                               samples per window -- more than a
///                               `fuse::Waveform` built with that capacity
///                               (spec: "window longer than a Waveform can
///                               hold") can carry.
[[nodiscard]] Result<void> validate_coupling_window(
    const CouplingGraph& g, const IterationRegionDecl& region,
    const CouplingWindowScope& scope,
    std::size_t waveform_capacity = 4096,
    Diagnostics* diags = nullptr);

}  // namespace fuse
