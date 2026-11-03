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

/// An OPT-IN alternative to
/// the single-clock iteration region, for a region whose channels span
/// two (or more) clocks that do not coincide for a long time -- the 7:13
/// case the multirate probe (`test/schedule/multirate_region_test.cpp`)
/// pins as refused without a window scope.
///
/// THE OPT-IN LIVES ON THE REGION DECLARATION. `CouplingWindowScope` and
/// `WindowMemberPolicy` are defined in `fuse/model/iteration.hpp`, not here
/// -- `IterationRegionDecl` (`fuse/model/coupling_graph.hpp`) carries an
/// `std::optional<CouplingWindowScope> window_scope` member directly, and
/// `coupling_graph.hpp` already includes `iteration.hpp`; defining the type
/// in this header instead (which includes `coupling_graph.hpp` for
/// `CouplingGraph`) would be circular. This header only holds the free
/// functions that CONSUME the type once a region carries one:
/// `window_length()`, `window_region_members()` and
/// `validate_coupling_window()`, the legality gate a caller (a test
/// harness, a model driving `WindowCoupler` directly, or `schedule()`
/// itself for a region whose declaration opts in) must pass before running
/// a window-scoped region.
namespace fuse {

/// window_length() moved to fuse/model/coupling_graph.hpp (available here
/// transitively, via this header's own #include above) -- ir sits below
/// schedule, and window_lowering.cpp needs the identical resolution to
/// size a compiled field.

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
///   E-SCHED-WINDOW-GAUSSSEIDEL-NOLAG
///                              `scope.iteration == WaveformIteration::
///                              GaussSeidel` and every field transfer among
///                              `region`'s members is unlagged: the group
///                              closes a cycle with nothing to break it.
///                              Lag one direction, or switch to
///                              `WaveformIteration::Jacobi`.
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
