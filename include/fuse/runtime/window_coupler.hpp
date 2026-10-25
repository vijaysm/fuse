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

/// Time windows: an iteration region whose channels span two (or
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
/// STATE OF THE GRAPH-LEVEL FRONT END (multirate window on the Runner):
///   * the opt-in lives on the declaration itself --
///     `IterationRegionDecl::window_scope`
///     (`fuse::CouplingWindowScope`, `fuse/model/iteration.hpp`), carried
///     through the descriptor and the YAML round trip;
///   * `fuse::validate_coupling_window()` (`fuse/schedule/window.hpp`) is
///     the legality gate a front end must pass before constructing one of
///     these, and `schedule()` calls it for a window-scoped region
///     (`mark_regions`/`expand.cpp`, src/schedule/) instead of running its
///     unconditional contiguous-run check and refusing every multi-clock
///     region with `E-SCHED-REGION`;
///   * `schedule()`/`Runner` compile and execute a window-scoped
///     region as one window block: a per-member `OpKind::WindowAdvance`
///     node with no graph inputs (`src/ir/window_lowering.cpp`), reading
///     its import through `OpAttrs::window_import` and driven every sweep
///     by `Runner::Impl::advance_window()` (src/runtime/iteration.cpp) --
///     resolving `WindowMemberConfig::provided`/`required` from the
///     `FieldRegistry` the way `RegionBlock` resolves value/feed arcs for
///     a single-clock region. Verified bit-for-bit against this file's own
///     `WindowCoupler`, both waveform-relaxation modes, for a genuinely
///     multirate (incommensurate 13:7) pair with `window_component`
///     naming the slower member (`test/runtime/window_runner_straddle_
///     test.cpp`) -- including the Runner's own new-window seed (the SAME
///     `extrapolate_boundary_value()` predictor this class documents
///     below, over a consumer's own `WindowMemberState::history`, which
///     needs no partner state because what arrived through the field
///     transfer at every earlier window's own end IS the provider's own
///     committed trajectory there).
///
///     That seed is `Runner::Impl::advance_window()`'s own import
///     trajectory for a lagged member's substep computation. It is distinct
///     from `Runner::Impl::run_region()`'s `R.x_old` pre-loop seed -- the
///     one value the generic Picard/accelerator machinery (residual norm,
///     `R.accel->update()`) reads as "the previous iterate". `R.x_old`'s
///     seed applies the identical predictor for every window instance
///     after a member's first (`src/runtime/iteration.cpp`,
///     `Runner::Impl::run_region()`'s `k == 0` block), verified
///     bit-for-bit against this class's own `y_fed`, element for element,
///     and against a direct `AccelKind::Aitken` parity case in
///     `test/runtime/window_runner_straddle_test.cpp`. The distinction
///     matters only for accelerators that read `x_old`: under
///     `AccelKind::None` or the refusal-only `AccelKind::Constant`,
///     `R.x_old` never reaches `x_out` at all (`NoneAccel::update()`
///     ignores it outright), whereas `AccelKind::Aitken`/`SteffensenAccel`/
///     `QuasiNewtonAccel` all read `x_old` directly (the relaxed output
///     `x_old + w*(x_new-x_old)`, and the residual difference `x_old`
///     enters a history-dependent `omega`/weight recursion from). A
///     zeroth-order/persistence predictor (the raw, already-delivered
///     buffer) is the alternative this class's own `sweep_window()`
///     comment records trying and reverting for `y_fed`.
///
///     Known open item: this predictor makes the Brusselator's own Aitken
///     configuration converge more slowly (a window that converged in 25
///     sweeps with the persistence seed runs the full 100-sweep cap, and a
///     raised cap does not resolve it), so it is not settled for every
///     accelerator;
///   * the sweep loop below is driven through `WindowChannelIO`
///     (immediately below `WindowChannel`), an exchange seam a future
///     front end can override to route a channel's provider->consumer
///     data through a real `fuse::Remap`/`fuse::Transport` instead of the
///     raw-pointer read the base class still performs by default -- built
///     and exercised by the existing test suite (which never overrides
///     it), NOT YET implemented by anything that does the remap/transport
///     it exists for. The Runner's own compiled field transfer (Source,
///     Transport, Remap, Epoch, Sink) is a SEPARATE path from this seam --
///     it is what carries a window member's K-wide export/import arcs
///     today -- and is itself still an identity map on every test that
///     exercises it (matching meshes throughout); a genuine remap across
///     the window boundary is still open.
/// A `DtPolicy::Fixed` member's own step crossing the window end (the
/// straddling case this class's own `resolve_pending_straddles()`
/// replays) is fully wired into the Runner too, for BOTH roles a
/// straddling member can play: its own self-straddle (`advance_window()`
/// calls the same shared `fuse::detail::advance_straddle_split()` this
/// class does, where nothing crosses a component boundary), and a
/// STRADDLING IMPORT PROVIDER as seen by some OTHER member (`Runner::
/// Impl::resolve_pending_window_straddles()`, `src/runtime/iteration.cpp`
/// -- this class's own mirror, replaying a straddling provider's
/// overshoot against the window's now-converged data, exactly as this
/// class's own function of the same name does, and checked bit-for-bit
/// against it in `test/runtime/window_runner_straddle_test.cpp`).
/// `fuse::window_sample_times()`/`fuse::window_sample_shift()`
/// (`fuse/model/coupling_graph.hpp`) are what let a consumer reconstruct
/// a straddling provider's own sample times and buffer-index shift from
/// the graph alone, no partner state -- the subtlety both resolve: once
/// a provider has straddled at least one window boundary, its own
/// compiled sample buffer's index 0 no longer holds the value AT this
/// window's own T0 (the provider's real clock already sits at the next
/// native grid point past T0 by the time this window's sweep 0 begins),
/// so `advance_window()` reads that one entry from the consumer's own
/// `WindowMemberState::history` instead (recorded when the previous
/// window's straddle resolved), and every remaining entry shifted down
/// by one buffer slot from where it would sit for a non-straddling
/// provider. A LAGGED consumer's own first-sweep-of-a-pass predictor has
/// one further real (non-extrapolated) knot at exactly that same shifted
/// index -- the provider's own real post-replay state, which `resolve_
/// pending_window_straddles()` publishes into every such consumer's own
/// `window_import` arc the moment the replay finishes, before this
/// pass's first sweep ever snapshots `Impl::previous[]` -- matching this
/// class's own `build_delivered_member()`, which reads `start_value[i][e]`
/// directly for the identical reason. That publish writes the value
/// straight into the consumer's own arc, bypassing whatever Transport/
/// Remap the channel's own compiled chain would otherwise run between
/// the provider's export and the consumer's import -- exact ONLY for a
/// same-rank, same-region (identity) transfer, since (unlike this
/// class, one object in one address space) the Runner's own provider
/// and consumer can legitimately be on different ranks or meshes.
/// `resolve_regions()` (`src/runtime/iteration.cpp`) refuses, at setup,
/// any channel where a `DtPolicy::Fixed` provider that CAN straddle
/// (its own interval does not divide the window) feeds a consumer that
/// is not on the same ranks and the same region (`E-RT-WINDOW-STRADDLE-
/// REPUBLISH`, `test/runtime/window_runner_test.cpp`'s own
/// `FixedPolicyStraddleWithGenuineRemapIsRejected`/
/// `FixedPolicyStraddleWithRankMismatchIsRejected`) -- running the
/// publish THROUGH the real compiled chain instead, so a genuine remap
/// or a cross-rank transfer is handled rather than refused, is still
/// open.
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
/// glossary.md: subcycling). Shared so a future subcycling-node runtime
/// dispatch can plan the same grid `WindowCoupler` does, from the identical
/// inputs.
struct WindowSubstep {
  Rational t0, t1;
  bool     straddle = false;  ///< true only for the (at most one) last step
};

/// Spec: choose the window so it never waits for the LCM of two intervals.
/// A member's own substeps march on ITS OWN grid, `Adjustable` truncating
/// the last one onto `window_end`, `Fixed` taking it in full even past
/// `window_end` (the straddling step; see the class comment below). A pure
/// function of its arguments, callable by anything else that needs the
/// identical grid.
[[nodiscard]] std::pair<std::vector<WindowSubstep>, Rational> plan_substeps(
    Rational clock, Rational interval, Rational window_end, DtPolicy policy);

/// The mean of `wf` over `[a, b)`, `a < b` required. `wf` must already
/// cover `[a, b]` (its support, per `Waveform`'s own contract). Declared
/// here so any window dispatch code can reuse it.
[[nodiscard]] Result<double> mean_over(const Waveform& wf, Rational a,
                                       Rational b);

/// Per-window-boundary sample, one per element: what a member COMMITTED at
/// that time. Used only for extrapolation beyond the most recent window
/// (`extrapolate_boundary_value`); the finer, within-window trajectory a
/// partner needs for an in-range mean lives in `WindowMemberState::
/// last_window_waveform` instead.
using WindowBoundaryHistory = std::vector<std::pair<Rational, double>>;

/// One member's own subcycling state, tracked across windows (docs/user/
/// glossary.md: subcycling). Nameable by
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
  /// Meaningful only when `WindowCouplerConfig::
  /// iteration == WaveformIteration::GaussSeidel` (silently unused under
  /// the default `Jacobi`, which never reads it -- see that enumerator's
  /// own docstring, fuse/model/iteration.hpp). `false` (the default): this
  /// channel is unlagged, so `sweep_window()` refreshes `delivered[provider]`
  /// the moment `provider`'s own substep loop finishes THIS sweep, and
  /// `provider`'s own outgoing waveform is not part of the accelerated
  /// vector (its value is already exact, not relaxed). `true`: this
  /// channel is loop-carried, so `consumer` reads `provider`'s PREVIOUS
  /// sweep's value all the way through this one, and `provider`'s
  /// outgoing waveform IS accelerated.
  bool        lagged = false;
  /// A `WindowCouplerConfig::
  /// convergence` measure names a field TRANSFER -- this channel -- not
  /// a raw field. Empty (the default) is fine as long as no measure
  /// names it; `create()` rejects any measure whose `.field` does not
  /// match some channel's own `name` here, by name, with a test.
  std::string name;
};

/// The exchange seam a graph-level front end needs, so
/// that a window channel's provider -> consumer data can be routed through
/// a real `fuse::Remap`/`fuse::Transport`
/// instead of direct pointer arithmetic. `WindowCoupler` calls this at every point it needs
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
/// "one message per exchange" property; the
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
  /// Which classical waveform-relaxation scheme `sweep_window()` runs
  /// (`WaveformIteration`'s own docstring, fuse/model/iteration.hpp).
  /// Default `Jacobi`: every member's substep loop reads the SAME
  /// snapshot of every other member's waveform for the whole sweep --
  /// this class's original, and only, behaviour before this field
  /// existed, and it must stay byte-identical to it.
  WaveformIteration     iteration = WaveformIteration::Jacobi;
  /// Default interpolation/extrapolation order, used by any `WindowMember`
  /// whose own `order` is 0. Must be >= 2.
  int                   order = 2;
  /// Waveform sample capacity per member per window. See
  /// `validate_coupling_window`'s `E-SCHED-WINDOW-CAPACITY`.
  std::size_t           waveform_capacity = 64;

  /// The exchange seam (`WindowChannelIO` above).
  /// `nullptr` (the default) is every direct caller today -- `create()`
  /// then constructs a plain `WindowChannelIO`, whose direct-read default
  /// behaviour reproduces the pointer arithmetic this class always used.
  std::shared_ptr<WindowChannelIO> channel_io;

  /// Default `true`: `run_window()` exhausting `max_iterations` without
  /// meeting `convergence` returns `E-RT-WINDOW-NOCONVERGE`, exactly as
  /// this class has always behaved -- every existing caller relies on
  /// that, so the default preserves it byte-for-byte. `false` mirrors
  /// `RunnerConfig::abort_on_nonconvergence`'s own convention: the window
  /// still commits (history, `last_window_waveform`, clock, straddle
  /// bookkeeping) and `window_index_` still advances, with only
  /// `WindowIterationStats::converged` recording the outcome -- for a
  /// caller that deliberately sets an unreachable tolerance to force
  /// every sweep to run (a fixed-iteration-count comparison against
  /// another implementation, independent of either one's own stopping
  /// rule), not for ordinary use.
  bool                  abort_on_nonconvergence = true;
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
/// `WindowCoupler`'s class comment). Shared by
/// two callers: `WindowCoupler::sweep_window()`'s
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
  ///
  /// `cfg.iteration == WaveformIteration::GaussSeidel` only (Jacobi sweeps
  /// declaration order and needs none of this: every member reads the same
  /// snapshot regardless of when it runs):
  ///   E-RT-WINDOW-GAUSSSEIDEL-NOLAG   the directed graph of UNLAGGED
  ///                          channels (provider -> consumer) has a cycle
  ///                          -- including the "no channel at all is
  ///                          lagged" case, which is that graph with EVERY
  ///                          channel in it. No sweep order can make every
  ///                          unlagged provider run before its consumer,
  ///                          the one thing this mode promises. The exact
  ///                          runtime counterpart of `validate_coupling_
  ///                          window()`'s own `E-SCHED-WINDOW-GAUSSSEIDEL-
  ///                          NOLAG` (fuse/schedule/window.hpp) -- one
  ///                          method (a cycle check over the unlagged-edge
  ///                          graph), one meaning, at both the compiled-
  ///                          graph and the standalone-driver level.
  ///   E-RT-WINDOW-GAUSSSEIDEL-MIXED-LAG   a member provides both a lagged
  ///                          AND an unlagged outgoing channel. `delivered
  ///                          [provider]` is one waveform per provider, not
  ///                          one per channel, so it cannot simultaneously
  ///                          be "this sweep's fresh value" (what the
  ///                          unlagged consumer needs) and "last sweep's
  ///                          value" (what the lagged consumer needs).
  ///                          Refused rather than silently picking one and
  ///                          getting the other's consumer wrong.
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
  /// `BoundaryHistory`/`MemberState` are
  /// `fuse::detail::WindowBoundaryHistory`/`WindowMemberState`
  /// (declared above, alongside `plan_substeps`/`mean_over`),
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
  /// The order `sweep_window()` visits members in, WITHIN one sweep --
  /// what "declaration order" (`members_`'s own index order) meant before
  /// GaussSeidel existed, and still does under Jacobi (identity
  /// permutation, computed once at construction, never touched again).
  /// Under GaussSeidel: a topological order of the UNLAGGED-channel
  /// precedence graph (provider before consumer), ties broken by
  /// declaration order for a deterministic, reproducible sweep -- see
  /// `compute_sweep_order()`. Declaration order alone is not enough: a
  /// caller could declare members in any order, but the compiled
  /// schedule's own ordering is fixed by which channel is lagged, not by
  /// which member the graph happened to add first.
  std::vector<std::size_t>   sweep_order_;
  /// Never null after construction: `WindowCoupler(...)` builds a plain
  /// `WindowChannelIO` when `cfg_.channel_io` is null, so every call site
  /// below can dereference this unconditionally.
  std::shared_ptr<WindowChannelIO> channel_io_;
  std::unique_ptr<Accelerator> accel_;
  std::int64_t               window_index_ = 0;
  WindowIterationStats       stats_;
  /// One entry per `cfg_.convergence.measures()` entry, in the same
  /// order: the index into `channels_` that measure's own `.field`
  /// named. `create()` already rejected any measure naming a channel
  /// that does not exist, so this is always a valid index once the
  /// object exists -- resolved once here rather than re-searching
  /// `channels_` by name every sweep.
  std::vector<std::size_t>   convergence_channel_;
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
