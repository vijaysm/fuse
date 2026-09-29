#pragma once

#include <cstddef>
#include <functional>
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
/// directly. The mechanism below is "Verified" against
/// test/runtime/window_coupler_test.cpp, `examples/gray_scott` and
/// `examples/brusselator`.
///
/// STATE OF THE GRAPH-LEVEL FRONT END (multirate-window-on-the-Runner
/// track). Three of its four pieces exist; `schedule()`/`Runner` do not yet
/// call any of them, so wiring one arbitrary `fuse::CouplingGraph`
/// declaration all the way through remains DESIGNED ONLY (see the "Five
/// states", docs/dev/testing.md):
///   * the opt-in lives on the declaration itself --
///     `IterationRegionDecl::window_scope`
///     (`fuse::CouplingWindowScope`, `fuse/model/iteration.hpp`), carried
///     through the descriptor (v7) and the YAML round trip;
///   * `fuse::validate_coupling_window()` (`fuse/schedule/window.hpp`) is
///     the legality gate a front end must pass before constructing one of
///     these -- written, tested, not yet CALLED by `schedule()`, which
///     still runs its unconditional contiguous-run check
///     (`mark_regions`, src/schedule/scheduler.cpp) and refuses a
///     multi-clock region with `E-SCHED-REGION` whether or not
///     `window_scope` is set;
///   * the sweep loop below is driven through `WindowChannelIO`
///     (immediately below `WindowChannel`), an exchange seam a future
///     front end can override to route a channel's provider->consumer
///     data through a real `fuse::Remap`/`fuse::Transport` instead of the
///     raw-pointer read the base class still performs by default -- built
///     and exercised by the existing test suite (which never overrides
///     it), NOT YET implemented by anything that does the remap/transport
///     it exists for.
/// What is NOT built: `schedule()` recognizing a window-scoped region and
/// lowering/scheduling it as one window block instead of the per-channel,
/// per-tick nodes `build_iteration_regions()` (src/ir/regions.cpp) emits
/// today; the Runner executing that block by resolving
/// `WindowMember::provided`/`required` from the `FieldRegistry` the way
/// `RegionBlock` resolves value/feed arcs for a single-clock region; and a
/// `WindowChannelIO` override that actually performs the remap/transport.
/// See this track's design notes for why (the arc/Instruction
/// representation this needs does not yet exist) and the concrete shape
/// proposed for it.
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

namespace detail {

/// One substep of a member's own grid within a time window (docs/user/
/// glossary.md: subcycling). Shared (Unit A of the multirate-window-on-
/// the-Runner track) so a future subcycling-node runtime dispatch can plan
/// the same grid `WindowCoupler` does today, from the identical inputs.
struct WindowSubstep {
  Rational t0, t1;
  bool     straddle = false;  ///< true only for the (at most one) last step
};

/// Spec: choose the window so it never waits for the LCM of two intervals.
/// A member's own substeps march on ITS OWN grid, `Adjustable` truncating
/// the last one onto `window_end`, `Fixed` taking it in full even past
/// `window_end` (the straddling step; see the class comment below). Moved
/// here from window_coupler.cpp's own anonymous namespace (Unit A, step
/// 1): a pure function of its arguments, unchanged, now callable by
/// anything else that needs the identical grid.
[[nodiscard]] std::pair<std::vector<WindowSubstep>, Rational> plan_substeps(
    Rational clock, Rational interval, Rational window_end, DtPolicy policy);

/// The mean of `wf` over `[a, b)`, `a < b` required. `wf` must already
/// cover `[a, b]` (its support, per `Waveform`'s own contract). Moved here
/// from window_coupler.cpp's own anonymous namespace (Unit A, step 1),
/// unchanged.
[[nodiscard]] Result<double> mean_over(const Waveform& wf, Rational a,
                                       Rational b);

/// Per-window-boundary sample, one per element: what a member COMMITTED at
/// that time. Used only for extrapolation beyond the most recent window
/// (`extrapolate_boundary_value`); the finer, within-window trajectory a
/// partner needs for an in-range mean lives in `WindowMemberState::
/// last_window_waveform` instead.
using WindowBoundaryHistory = std::vector<std::pair<Rational, double>>;

/// One member's own subcycling state, tracked across windows (docs/user/
/// glossary.md: subcycling). Moved here from `WindowCoupler`'s own private
/// nested `MemberState` (Unit A, step 2): unchanged, now nameable by
/// anything else that needs to hold or advance a member's state the way
/// `WindowCoupler` does today.
struct WindowMemberState {
  /// Simulated time this member has actually reached (its own state is
  /// valid here). Equal to the window start unless a `Fixed` member's
  /// last substep straddled past it.
  Rational clock{0};
  /// One list per element, ascending in time, seeded with the initial
  /// condition (`t=0`, the value bound at `WindowCoupler::create()` time).
  std::vector<WindowBoundaryHistory> history;
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

}  // namespace detail

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

/// Time windows track: the exchange seam a graph-level front end needs, so
/// that a window channel's provider -> consumer data can be routed through
/// a real `fuse::Remap`/`fuse::Transport` (design notes for this track)
/// instead of the direct pointer arithmetic `WindowCoupler` used before
/// this class existed. `WindowCoupler` calls this at every point it needs
/// "this channel's provider value for consumer element `e`"; the default
/// -- what `WindowCoupler` constructs whenever
/// `WindowCouplerConfig::channel_io` is left null -- is EXACTLY the direct
/// read every caller had before: the same `std::min(e, provider.size()-1)`
/// clamp, the same `Waveform::integrate`/`evaluate` calls and the same
/// `extrapolate_boundary_value()`. Every existing direct caller
/// (`examples/gray_scott`, `examples/brusselator`, this header's own test)
/// is unaffected by this seam's existence: none of them set
/// `channel_io`, so none of them observe anything but the base class's own
/// behaviour.
///
/// AN OVERRIDE THAT DOES REAL WORK MUST BATCH IT ITSELF. `mean()`/
/// `boundary()`/`extrapolate()` are called once per (channel, consumer
/// substep, element) -- exactly as often as `WindowCoupler` always needed
/// the value, one query at a time; nothing here changes that call
/// frequency. A remap is spatial and linear, so it commutes with the
/// temporal mean and can legally be applied once to `provider`'s whole
/// trajectory rather than once per query -- but ONLY an override that
/// itself memoizes on `provider`'s identity (rebuilt at most once per
/// sweep by `WindowCoupler::sweep_window()`) turns that legality into the
/// "one message per exchange" property this track is bound to; the
/// straightforward reading of this interface -- remap and transport
/// inside every call -- would send one message per query, which is
/// exactly what that property forbids.
class WindowChannelIO {
 public:
  virtual ~WindowChannelIO() = default;

  /// The provider's mean over `[a, b)`, `a < b`, for consumer element `e`.
  [[nodiscard]] virtual Result<double> mean(
      const WindowChannel& ch, std::size_t e,
      const std::vector<std::unique_ptr<Waveform>>& provider, Rational a,
      Rational b) const;

  /// The provider's point value at `t` for consumer element `e`.
  [[nodiscard]] virtual Result<double> boundary(
      const WindowChannel& ch, std::size_t e,
      const std::vector<std::unique_ptr<Waveform>>& provider,
      Rational t) const;

  /// The provider's history, extrapolated to `target`, for consumer
  /// element `e`. Never fails (matches `extrapolate_boundary_value()`).
  [[nodiscard]] virtual double extrapolate(
      const WindowChannel& ch, std::size_t e,
      const std::vector<std::vector<std::pair<Rational, double>>>&
          provider_history,
      Rational target, int order) const;
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

  /// Time windows track: the exchange seam (`WindowChannelIO` above).
  /// `nullptr` (the default) is every direct caller today -- `create()`
  /// then constructs a plain `WindowChannelIO`, whose direct-read default
  /// behaviour reproduces the pointer arithmetic this class always used.
  std::shared_ptr<WindowChannelIO> channel_io;
};

/// Per-window outcome, exposed for the caller (and for the mutation-evidence
/// tests) to inspect without re-deriving it from side effects.
struct WindowIterationStats {
  int                 iterations = 0;
  bool                converged = false;
  std::vector<double> residual_history;  ///< one entry per sweep, ||G(y)-y||
};

namespace detail {

/// Advances one member across a chunk `[t_a, t_b)` that straddles a split
/// point `t_split` (`t_a <= t_split < t_b`, `t_split == t_b` skips the
/// tail), splitting the `advance()` call exactly at `t_split` instead of
/// taking one `advance()` over the whole chunk with a single blended mean
/// and then interpolating the value AT `t_split` between the two endpoint
/// states -- that interpolation does not recover x(t_split) to the
/// substep's own order, while splitting the call makes the committed
/// t_split value as accurate as an ordinary substep's own endpoint (see
/// `WindowCoupler`'s class comment). Moved here (Unit A, step 3) from its
/// two, previously duplicated, homes: `WindowCoupler::sweep_window()`'s
/// per-member substep loop (its `mean_in_window`/`boundary_at_split`/
/// `extrapolate_tail` read this sweep's live `delivered[]` waveforms) and
/// `WindowCoupler::resolve_pending_straddles()`'s replay (its callbacks
/// read the already-converged `last_window_waveform`/`history` from the
/// window that just completed instead) -- so a future subcycling-node
/// runtime dispatch can drive the identical choreography from either
/// data source.
///
/// The two existing call sites also differ, EACH IN ITS OWN WAY, on what
/// a callback returns when no channel feeds this member's element (one
/// silently defaults to zero, the other leaves `member.required[e]`
/// unchanged) and on whether a failed boundary lookup propagates or is
/// silently absorbed -- this function does not choose a policy: each
/// callback encodes its own call site's existing policy, and this
/// function's own behaviour is the one thing genuinely shared, the two-
/// phase split-advance sequencing itself with an exact (never
/// interpolated) value recorded at the split.
///
/// `mean_in_window(e, a, b)`: the value to feed `member.required[e]` for
/// the FIRST `advance()`, over `[t_a, t_split)`.
/// `boundary_at_split(e)`: element `e`'s value exactly at `t_split`, for
/// the blended tail mean.
/// `extrapolate_tail(e, target)`: element `e`'s extrapolated value at
/// `target` (== `t_b`), for the same blend.
/// `out_at_split` (`member.length` doubles, caller-owned and
/// caller-positioned in whatever layout it needs -- `WindowCoupler`'s
/// canonical flat (member, element, substep) index for `sweep_window()`,
/// a throwaway scratch buffer for `resolve_pending_straddles()`) receives
/// the EXACT value at `t_split` from the first `advance()` call, never an
/// interpolant.
[[nodiscard]] Result<void> advance_straddle_split(
    const WindowMember& member, Rational t_a, Rational t_split, Rational t_b,
    std::size_t step, int iteration,
    const std::function<Result<double>(std::size_t, Rational, Rational)>&
        mean_in_window,
    const std::function<Result<double>(std::size_t)>& boundary_at_split,
    const std::function<double(std::size_t, Rational)>& extrapolate_tail,
    double* out_at_split);

}  // namespace detail

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
  /// Unit A, step 2: `BoundaryHistory`/`MemberState` are now
  /// `fuse::detail::WindowBoundaryHistory`/`WindowMemberState`
  /// (declared above, alongside `plan_substeps`/`mean_over`), unchanged,
  /// so a future subcycling-node runtime dispatch can hold and advance a
  /// member's state the same way `WindowCoupler` does today. These
  /// aliases keep every existing use in this class (`BoundaryHistory`,
  /// `MemberState`) resolving to them without another line changing.
  using BoundaryHistory = detail::WindowBoundaryHistory;
  using MemberState = detail::WindowMemberState;

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
  /// Never null after construction: `WindowCoupler(...)` builds a plain
  /// `WindowChannelIO` when `cfg_.channel_io` is null, so every call site
  /// below can dereference this unconditionally.
  std::shared_ptr<WindowChannelIO> channel_io_;
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
