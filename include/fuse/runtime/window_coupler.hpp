#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/iteration.hpp"
#include "fuse/num/accelerator.hpp"
#include "fuse/num/waveform.hpp"
#include "fuse/runtime/component.hpp"

/// Time windows track: an iteration region whose channels span two (or
/// more) clocks, coupled over a macro WINDOW rather than a single tick.
///
/// `WindowCoupler` is the runtime driver. It is written directly against
/// the production `fuse::Component` ABI (`advance()`/`save_state()`/
/// `restore_state()`, `StepContext`) and against the production
/// `fuse::Waveform` and `fuse::Accelerator`/`fuse::ConvergenceCriterion`, so
/// it is usable today by a model author driving a small set of components
/// directly. `schedule()`/`Runner` wiring it to an arbitrary
/// `fuse::CouplingGraph` declaration -- resolving `WindowMember::provided`/
/// `required` from the `FieldRegistry` the way `RegionBlock` does for a
/// single-clock region -- is DESIGNED ONLY (see the "Five states",
/// docs/dev/testing.md); that graph-level front end is not built in this
/// track. The mechanism below is "Implemented, untested" until
/// test/runtime/window_coupler_test.cpp exists and passes against it --
/// this comment does not itself make it "Verified". See
/// `include/fuse/schedule/window.hpp` for the schedule-time legality gate a
/// future front end would call before constructing one of these.
///
/// ---------------------------------------------------------------------------
/// WHAT ONE `advance()` CALL RECEIVES (why the algorithm looks like it does)
/// ---------------------------------------------------------------------------
/// `Component::advance()`'s contract (fuse/runtime/component.hpp) hands the
/// implementer exactly ONE snapshot of a `Requires` field, valid at
/// `ctx.t_begin` -- not a trajectory. To get second-order accuracy in the
/// EXCHANGED quantity out of that one-value-per-call contract,
/// `WindowCoupler` feeds each substep the sender's WAVEFORM MEAN over
/// exactly that substep's interval (`Waveform::integrate(...) / dt`), not a
/// point sample. This is the same convention `AccumOp::Mean` already uses
/// for a single-clock accumulate channel, and `Waveform`'s own docstring
/// measures exactly this order for the `Constant` basis's INTEGRATED
/// accuracy ("Constant, 1 sample: claimed 1, measured 1.04, integral over
/// the window 2 (2.06)"). The sender's own outgoing trajectory is
/// represented with `TemporalBasis::Linear` (its "in-step interpolant"),
/// which is exact for `integrate()` over any sub-interval of its support, so
/// the substep means handed to different consumers over the SAME sender
/// window always sum to the sender's whole-window integral to round-off
/// (spec: conservative in time).
namespace fuse {

/// One participant of a coupling window.
///
/// FIELD ACCESS. `provided`/`required` are non-owning pointers into storage
/// the `Component` itself registered (mirrors `RegionBlock`'s `value_arc`/
/// `feed_arc` for a single-clock region): `WindowCoupler` never allocates a
/// component's field storage, only reads/writes through these pointers, and
/// they must stay valid and fixed for this `WindowCoupler`'s whole lifetime
/// (`Component::register_fields()`'s own contract).
struct WindowMember {
  Component*  component = nullptr;
  std::string name;
  Rational    interval{1};
  DtPolicy    policy = DtPolicy::Adjustable;
  /// Interpolation/extrapolation order for THIS member's outgoing data.
  /// Must be >= 2 (`WindowCouplerConfig::order` sets every member's default;
  /// this overrides it per member when non-zero).
  int         order = 0;
  double*     provided = nullptr;  ///< `length` doubles; advance() writes here
  double*     required = nullptr;  ///< `length` doubles; advance() reads here
  std::size_t length = 1;
};

/// A directed exchange: `members[provider]`'s outgoing data feeds
/// `members[consumer]`'s `required` buffer.
struct WindowChannel {
  std::size_t provider = 0;
  std::size_t consumer = 0;
};

struct WindowCouplerConfig {
  Rational              window_length{1};
  int                   max_iterations = 50;
  ConvergenceCriterion  convergence;
  AccelConfig           acceleration{.kind = AccelKind::None};
  /// Default interpolation/extrapolation order, used by any `WindowMember`
  /// whose own `order` is 0. Must be >= 2.
  int                   order = 2;
  /// Waveform sample capacity per member per window. See
  /// `validate_coupling_window`'s `E-SCHED-WINDOW-CAPACITY`.
  std::size_t           waveform_capacity = 64;
};

/// Per-window outcome, exposed for the caller (and for the mutation-evidence
/// tests) to inspect without re-deriving it from side effects.
struct WindowIterationStats {
  int                 iterations = 0;
  bool                converged = false;
  std::vector<double> residual_history;  ///< one entry per sweep, ||G(y)-y||
};

/// Constructed once per iteration region; `run_window()` advances every
/// member through exactly one window and may be called repeatedly for
/// consecutive windows (window `w`'s `[T0, T1)` is `[w, w+1) * window_length`
/// -- windows MUST be run back to back, because a `Fixed` member's pending
/// straddle is resolved relative to the immediately preceding window; there
/// is no API to skip or replay one).
class WindowCoupler {
 public:
  /// Validates `members`/`channels`/`cfg` and, on success, returns a
  /// constructed coupler. Refusals (independent of, and narrower than,
  /// `validate_coupling_window`'s graph-level checks -- this constructor has
  /// no `CouplingGraph` to consult):
  ///   E-RT-WINDOW-EMPTY     no members or no channels
  ///   E-RT-WINDOW-NULL      a member's `component`/`provided`/`required`
  ///                          is null while `length > 0`
  ///   E-RT-WINDOW-INTERVAL  a member's `interval` is not positive
  ///   E-RT-WINDOW-ORDER     `cfg.order < 2`, or a member's own `order` is
  ///                          nonzero and < 2
  ///   E-RT-WINDOW-CHANNEL   a channel names a provider/consumer index
  ///                          outside `members`
  ///   E-RT-WINDOW-ROLLBACK  a `Fixed` member's `component->supports_rollback()`
  ///                          is false (a straddling step MUST be rolled back
  ///                          and recomputed; there is no fallback)
  [[nodiscard]] static Result<std::unique_ptr<WindowCoupler>> create(
      std::vector<WindowMember> members, std::vector<WindowChannel> channels,
      WindowCouplerConfig cfg);

  /// Advances every member through the next window, sweeping to convergence
  /// (or `max_iterations`, per `cfg.acceleration`/`cfg.convergence`).
  /// `E-RT-WINDOW-NOCONVERGE` if the window's sweep loop exhausts
  /// `max_iterations` without meeting `cfg.convergence`; every other error
  /// propagates from `Component::advance()`/`save_state()`/`restore_state()`
  /// or from the accelerator.
  [[nodiscard]] Result<void> run_window();

  [[nodiscard]] const WindowIterationStats& stats() const noexcept {
    return stats_;
  }

  /// The elapsed simulated time at which the NEXT `run_window()` call will
  /// begin its sweep-loop checkpoint (after resolving any pending straddle):
  /// window `w`'s nominal start, `w * window_length`. A `Fixed` member may
  /// sit ahead of this (a straddling substep already carried it past the
  /// window boundary; see the class comment).
  [[nodiscard]] Rational next_window_start() const noexcept {
    return window_index_ * cfg_.window_length;
  }

 private:
  /// Per-window-boundary sample, one per element: what this member
  /// COMMITTED at that time. Used only for extrapolation beyond the most
  /// recent window (`extrapolate_boundary_value`); the finer, within-window
  /// trajectory a partner needs for an in-range mean lives in
  /// `last_window_waveform` instead.
  using BoundaryHistory = std::vector<std::pair<Rational, double>>;

  struct MemberState {
    /// Simulated time this member has actually reached (its own state is
    /// valid here). Equal to the window start unless a `Fixed` member's
    /// last substep straddled past it.
    Rational clock{0};
    /// One list per element, ascending in time, seeded with the initial
    /// condition (`t=0`, the value bound at `WindowCoupler::create()` time).
    std::vector<BoundaryHistory> history;
    /// Set when this member's last committed substep overshot the window it
    /// was taken in (`DtPolicy::Fixed` only). Cleared by
    /// `resolve_pending_straddles()` once the overshoot has been replayed
    /// with that window's converged data.
    bool     straddle_pending = false;
    Rational straddle_window_start{0};  ///< checkpoint restore_state() returns to
    Rational straddle_window_end{0};    ///< the window whose data was provisional
    /// This member's own converged trajectory over the window that just
    /// completed (one `Waveform` per element, `TemporalBasis::Linear`); a
    /// partner reads it for an in-window mean, and a partner's replay
    /// (`resolve_pending_straddles`) reads it for the "real data" part of a
    /// straddle recompute. Null before the first window completes.
    std::vector<std::unique_ptr<Waveform>> last_window_waveform;
  };

  WindowCoupler(std::vector<WindowMember> members,
               std::vector<WindowChannel> channels, WindowCouplerConfig cfg);

  [[nodiscard]] Result<void> resolve_pending_straddles(Rational T0);
  [[nodiscard]] Result<void> sweep_window(Rational T0, Rational T1);

  [[nodiscard]] int order_for(std::size_t member_idx) const {
    const int o = members_[member_idx].order;
    return o >= 2 ? o : cfg_.order;
  }

  std::vector<WindowMember>  members_;
  std::vector<WindowChannel> channels_;
  WindowCouplerConfig        cfg_;
  std::vector<MemberState>   state_;
  std::unique_ptr<Accelerator> accel_;
  std::int64_t               window_index_ = 0;
  WindowIterationStats       stats_;
};

/// Extrapolates a scalar boundary-value trajectory to `target`, using the
/// LAST `order` points of `history` (a degree `order - 1` polynomial fit,
/// `fuse::num::fit_polynomial`), reduced to `history.size() - 1` when there
/// are fewer points than `order` needs (mirrors `Waveform::effective_dg_
/// degree`'s reduction, so a startup transient with only two seed points --
/// t=0's initial condition plus the first window's committed value -- still
/// gets a well-defined, if lower-order, prediction rather than a refusal).
/// `history` must have at least one point.
[[nodiscard]] double extrapolate_boundary_value(
    const std::vector<std::pair<Rational, double>>& history, Rational target,
    int order);

}  // namespace fuse
