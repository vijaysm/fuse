#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/iteration.hpp"
#include "fuse/model/restart.hpp"

/// Spec Section 5.1 (`Accelerate`: residual history -> correction) and Section 5.3
/// (implicit coupling groups). One implementation family serves Picard, constant
/// relaxation, Aitken, secant, Steffensen, IQN-ILS and IQN-IMVJ, because they
/// differ only in how much of the residual history they keep and how they
/// project onto it. Accelerators know no physics: they see a concatenated
/// vector of doubles and an iteration index.
namespace fuse {

/// The ladder, weakest to strongest.
///   None        plain Picard, x_out = x_new
///   Constant    fixed omega
///   Aitken      one dynamic omega for the whole vector
///   Steffensen  one dynamic omega per component (diagonal secant)
///   Secant      rank-1 vector least squares (IQN with one column)
///   IQN_ILS     rank-k vector least squares, history cleared per window
///   IQN_IMVJ    as IQN_ILS, plus an inverse-Jacobian retained across windows
// AccelKind, FilterKind and AccelConfig are NOT declared here. They moved to
// fuse/model/iteration.hpp during the compiler port, because the coupling
// graph has to name the accelerator an implicit coupling group uses and the model layer must not
// depend on the numerics layer. This header declares the accelerators
// themselves; what to configure them with is part of the graph.

/// Spec Section 5.3. Stateful across the iterations of a window; `reset()` marks a
/// window boundary.
class Accelerator {
 public:
  /// History depth policy, enforced by make_accelerator() and by the
  /// quasi-Newton implementation:
  ///   * the per-window secant history holds at most `AccelConfig::reuse`
  ///     column pairs (oldest dropped first), and `reuse` is capped here;
  ///   * IQN-IMVJ chains at most kMaxJacobianBlocks past windows, each holding
  ///     at most `reuse` columns.
  /// Peak storage is therefore bounded by
  ///   (2 * reuse + 2 * kMaxJacobianBlocks * reuse) * n doubles,
  /// i.e. at most 2 * 32 * (1 + 8) = 576 vectors of length n, independent of
  /// how many iterations or windows the run performs. An unbounded history
  /// would be both a slow leak and a growing O(k^2) cost per iteration.
  static constexpr int kMaxReuse           = 32;
  static constexpr int kMaxJacobianBlocks  = 8;

  virtual ~Accelerator() = default;

  Accelerator(const Accelerator&)            = delete;
  Accelerator& operator=(const Accelerator&) = delete;

  /// Window boundary. Clears the per-window secant history. IQN_IMVJ folds it
  /// into its retained inverse-Jacobian first; every other kind discards it.
  virtual void reset() = 0;

  /// One accelerated iterate.
  ///
  /// `x_old` is the iterate that was fed to the coupled solvers, `x_new` what
  /// they returned, so the residual is `x_new - x_old`. `x_out` receives the
  /// next iterate. `iteration` is the 0-based index within the current window;
  /// 0 means "no history", and is treated as an implicit window boundary
  /// equivalent to reset().
  ///
  /// `x_out` must not overlap either input: the history is recorded from
  /// `x_old` and `x_new` after `x_out` has been written, so an in-place call
  /// would store a secant pair that was never evaluated -- a wrong answer, not
  /// a crash. Any overlap, exact or partial, is rejected with E-ACCEL-ALIAS.
  /// The test is a range comparison through `std::less<const double*>`, which
  /// is a strict total order over arbitrary pointer values and must agree with
  /// `<` wherever `<` is defined; two ranges that really do overlap therefore
  /// always compare as overlapping, so the check has no false negatives on a
  /// conforming implementation.
  ///
  /// Returns an Error rather than throwing for anything a running coupled
  /// simulation can legitimately produce: a null pointer, a negative
  /// iteration, a non-finite input, or an accelerated iterate that came out
  /// non-finite.
  ///
  /// In that last case (E-ACCEL-DIVERGED) the call fails -- a non-finite
  /// iterate must never be reported as progress -- but `x_out` is guaranteed
  /// to hold a *finite* iterate that a caller may keep, log and continue from:
  ///   * the under-relaxed fallback `x_old + omega * (x_new - x_old)`, where
  ///     omega is `initial_relaxation` for every kind that has one and
  ///     `AccelConfig::omega` for AccelKind::Constant; or
  ///   * `x_old` unchanged, if that fallback overflowed too. Both inputs are
  ///     validated finite on entry, so this last resort always exists.
  /// The error message says which of the two `x_out` holds. Nothing else in
  /// the family can fail after `x_out` has been written, so "E-ACCEL-DIVERGED
  /// implies x_out is finite" is a postcondition of `update`, not a
  /// best-effort.
  ///
  /// WHAT A DIVERGENCE COSTS THE HISTORY, precisely, because "the history is
  /// dropped" is three different statements for this family:
  ///   * the current window's secant columns go, every time and for every
  ///     kind: `columns()` reads 0 afterwards, and Aitken's dynamic omega
  ///     returns to `initial_relaxation`. They are what the failed
  ///     extrapolation was built from.
  ///   * the previous iterate and residual are KEPT by the quasi-Newton kinds
  ///     (Aitken and Steffensen drop theirs with the rest of their state).
  ///     Nothing is wrong with them: both endpoints were evaluated by the
  ///     coupled solvers and validated finite, and only the least-squares
  ///     extrapolation over them blew up, so the next call can form a genuine
  ///     secant pair immediately instead of spending a sweep re-deriving one.
  ///   * IQN-IMVJ's retained inverse Jacobian is dropped when it is what made
  ///     the step non-finite -- decided by recomputing the same step without
  ///     it (the plain IQN-ILS step, which is this expression with
  ///     Jprev^{-1} = -I) and asking whether THAT is finite -- and, in any
  ///     case, once it has been present for more than one divergence it was
  ///     not blamed for. `retained_blocks()` reports what survived.
  ///
  ///     Both halves are load bearing. Dropping the chain on any divergence
  ///     would discard every past window because of one bad extrapolation in
  ///     the current one, which is the whole of IMVJ's advantage over
  ///     IQN-ILS. But the test is one-sided: a finite step without the chain
  ///     proves the chain is what diverged, while a non-finite one proves only
  ///     that the current window's own least squares would have overflowed
  ///     too, NOT that the chain is sound. Both can be at fault at once, and
  ///     then the probe exonerates a chain that will keep overflowing. The
  ///     second rule is what bounds that: **any one retained Jacobian is
  ///     present for at most two divergences**, whatever the probe says and
  ///     whatever the caller does in between, because only folding a window in
  ///     -- which makes it a different operator -- starts the count again.
  ///     An unbounded run of failures with a poisoned chain, which an earlier
  ///     version of this code did permit, is therefore not reachable.
  ///
  /// n == 0 is a success no-op. A rank that owns no interface elements is
  /// ordinary in a parallel run, not an error.
  ///
  /// ALLOCATION: none, in steady state. Every buffer an implementation needs
  /// is a member, sized on the first call for a given `n` and reused
  /// thereafter (a growing `n` resizes, a shrinking one keeps the larger
  /// buffer). An accelerated implicit coupling group calls this from inside the sweep
  /// loop, which spec Section 2 layer 3 requires to allocate nothing;
  /// Accelerator.SteadyStateUpdateAllocatesNothing and
  /// Iteration.TheAcceleratorAllocatesNothingPerUpdate measure it for every
  /// kind rather than asserting it here. Three things are outside that
  /// steady state, all of them one-off or off the sweep path: the first call
  /// for each `n`; the `std::string` an error message is built from, on the
  /// failure path only; and IQN-IMVJ's window boundary, which allocates while
  /// its chain is still filling to its cap and again whenever a window closes
  /// with more columns than the block its chain evicts, since the evicted
  /// storage is recycled but not grown in advance. A window boundary is not
  /// the sweep loop, and that last case settles at zero as soon as the window
  /// sizes do.
  virtual Result<void> update(const double* x_old, const double* x_new,
                              double* x_out, std::size_t n, int iteration) = 0;

  [[nodiscard]] virtual std::string name() const = 0;

  /// Retained secant columns; 0 for the relaxation methods, and 0 immediately
  /// after reset() for every method.
  [[nodiscard]] virtual std::size_t columns() const = 0;

  /// Low-rank blocks of the inverse Jacobian retained across window
  /// boundaries; non-zero only for IQN-IMVJ. Exists so that the history depth
  /// policy stated on kMaxJacobianBlocks is observable, and therefore
  /// testable, rather than merely asserted in a comment.
  [[nodiscard]] virtual std::size_t retained_blocks() const noexcept {
    return 0;
  }

  /// Multi-field support (spec Section 5.3: per-field scaling is part of the
  /// accelerator interface, not an add-on). `offsets` gives the start index of
  /// each field block inside the concatenated vector. It must begin at 0 and
  /// must not decrease; equal neighbours are an empty block, which a rank
  /// owning none of that field has and which is not an error. An empty
  /// vector restores the default, which is no
  /// partition and therefore no row scaling at all -- not a single block. See
  /// AccelConfig::scale_fields for why the two differ.
  ///
  /// Throws std::invalid_argument on a malformed partition. That is a
  /// programmer error, not a runtime outcome, and it must still be caught
  /// under -DNDEBUG -- a silently mis-partitioned vector produces a
  /// plausible-looking but wrongly weighted least-squares fit.
  void set_field_partition(std::vector<std::size_t> offsets);

  [[nodiscard]] const std::vector<std::size_t>& field_partition()
      const noexcept {
    return field_offsets_;
  }

  /// Round 2 fix 1 (docs/dev/restart.md, NO SILENT RESET). Whether
  /// `export_restart_state()`/`import_restart_state()` below are a REAL
  /// implementation for this accelerator kind -- even a trivial one,
  /// because the kind provably has no learned state to restart at all.
  /// `NoneAccel`/`ConstantAccel` override this to return `true` for
  /// exactly that reason (`ConstantAccel::omega_` is a FIXED run option,
  /// never learned history); `AitkenAccel`/`SteffensenAccel` override it
  /// to `true` because their restart methods below are real. Default:
  /// `false` -- a kind that inherits the base classes' no-op
  /// export/import (`QuasiNewtonAccel` today; any future kind that adds
  /// learned state without also overriding these three methods) reports
  /// `false`, and `fuse::Runner::restore_restart_fields()`
  /// (src/runtime/runner.cpp) checks this BEFORE calling
  /// `import_restart_state()` at all: `false` with the run's restore
  /// options not opted into resetting an unimplemented accelerator is a
  /// collective refusal (E-RESTART-ACCEL-NOT-IMPLEMENTED), never a
  /// silent `reset()`. This check is collective BY CONSTRUCTION, not by
  /// an added reduction: every rank's graph assigns the SAME `AccelKind`
  /// to the SAME region (`validate()`'s own precondition, identical
  /// input on every rank), so `restart_state_implemented()` returns the
  /// identical boolean on every rank without any communication -- the
  /// same reasoning already used for the version-mismatch and
  /// missing-component refusals (docs/dev/restart.md Section 7.iii).
  [[nodiscard]] virtual bool restart_state_implemented() const noexcept {
    return false;
  }

  /// Restart track (docs/dev/restart.md), owner decision 4: accelerator
  /// memory is keyed by (field, global id) so it MOVES across a
  /// rank-count change rather than being reset. Round 2's own EXACT-KEY
  /// fix: `blocks` is the region's own stacked interface vector broken
  /// into the SAME per-field sub-ranges `fuse::Runner` already tracks
  /// (`detail::RegionBlock`, src/runtime/runner_impl.hpp) -- block `k`'s
  /// `ids` are THAT block's OWNING component's own global ids, used
  /// UNMODIFIED (never combined with another block's ids into one flat
  /// id space, and never hashed). The sum of every block's
  /// `ids.size()` must equal the `n` the most recent `update()` call
  /// used, in the SAME block order `update()`'s own flat buffers are
  /// laid out in. `name_prefix` qualifies every dataset this call
  /// writes with the calling region's own name, so two regions' own
  /// accelerators never collide in one restart point; each block's own
  /// `field_name` further qualifies within that, so two blocks of the
  /// SAME region's accelerator never collide either (RestartRowBlock's
  /// own doc comment, fuse/model/restart.hpp).
  ///
  /// Default: a successful no-op. Correct ONLY for a kind whose
  /// `restart_state_implemented()` is `true` and which has genuinely
  /// no learned state (`NoneAccel`/`ConstantAccel`); for any OTHER kind
  /// this default must never run, because `restart_state_implemented()`
  /// reporting `false` is what stops the Runner from calling it in a
  /// context where silence would be mistaken for success (see that
  /// method's own doc comment).
  virtual Result<void> export_restart_state(
      RestartWriter& /*writer*/, std::string_view /*name_prefix*/,
      std::span<const RestartRowBlock> /*blocks*/) const {
    return Result<void>::success();
  }

  /// The read-back counterpart. `allow_reset_on_mismatch` is the owner's
  /// explicit, off-by-default run option (docs/dev/restart.md's
  /// `@written_by_stacked_vector_layout_hash` refusal): `false` (the
  /// default) means a row with no matching entry in the file is
  /// `Failure` (E-RESTART-BAD-NUMBERING); `true` means `reset()` runs
  /// instead and `did_reset` is set to `true`, for the caller to report
  /// at restart (owner decision 4: "when used it is reported at
  /// restart").
  ///
  /// Default: UNCONDITIONAL `Failure` (E-RESTART-NOT-IMPLEMENTED) --
  /// round 2 fix 1's own "the default no-op must not be able to look
  /// like success" requirement, stated here as literally as the
  /// interface allows: the base class itself refuses rather than
  /// returning a success a caller could mistake for "nothing to
  /// restore". `fuse::Runner::restore_restart_fields()` already checks
  /// `restart_state_implemented()` before reaching this call at all (so
  /// this Failure is normally dead code, reached only if some OTHER
  /// caller invokes `import_restart_state()` directly without that
  /// guard) -- defense in depth, not the primary mechanism.
  virtual Result<void> import_restart_state(
      RestartReader& /*reader*/, std::string_view /*name_prefix*/,
      std::span<const RestartRowBlock> /*blocks*/,
      bool /*allow_reset_on_mismatch*/, bool& did_reset) {
    did_reset = false;
    return Failure{"E-RESTART-NOT-IMPLEMENTED",
                   "Accelerator::import_restart_state(): " + name() +
                       " does not override export_restart_state()/"
                       "import_restart_state()/restart_state_implemented(), "
                       "so this call cannot be trusted to mean \"nothing to "
                       "restore\" -- the caller must check "
                       "restart_state_implemented() before ever reaching "
                       "this default"};
  }

 protected:
  Accelerator() = default;

  std::vector<std::size_t> field_offsets_;
};

/// Validates the configuration and returns the matching accelerator.
/// Rejects a non-finite or zero relaxation factor, a reuse outside
/// [1, Accelerator::kMaxReuse], and a limit outside [0, 1).
[[nodiscard]] Result<std::unique_ptr<Accelerator>> make_accelerator(
    const AccelConfig& cfg);

namespace detail {

/// Shared by src/num/accel_relaxation.cpp (where they are defined) and
/// src/num/accel_quasi_newton.cpp. Not part of the supported surface.
[[nodiscard]] std::string format_accel_double(double v);
[[nodiscard]] Result<void> check_update_args(const double* x_old,
                                             const double* x_new,
                                             const double* x_out,
                                             std::size_t n, int iteration);
/// Enforces the E-ACCEL-DIVERGED postcondition documented on
/// Accelerator::update: succeeds if `x_out` is already finite everywhere,
/// otherwise repairs it -- first with the under-relaxed fallback at `w`, then,
/// if that overflows too, with `x_old` -- and reports E-ACCEL-DIVERGED naming
/// which of the two it left behind. `x_out` is finite on every return path.
[[nodiscard]] Result<void> finish_or_fallback(const char* who, double w,
                                              const double* x_old,
                                              const double* x_new,
                                              double* x_out, std::size_t n);
void relax_into(double w, const double* x_old, const double* x_new,
                double* x_out, std::size_t n);
void field_weights(const std::vector<std::size_t>& offsets, const double* r,
                   std::size_t n, std::vector<double>& out);

/// Round 2 EXACT-KEY fix (docs/dev/restart.md). Shared by every
/// Accelerator implementation that restarts a flat, stacked-interface-
/// vector buffer (AitkenAccel::r_prev_, SteffensenAccel::r_prev_/
/// x_old_prev_, QuasiNewtonAccel's own buffers if a future round
/// implements them): writes `flat` -- one value per row, `components`
/// doubles per row, laid out in the SAME block order as `blocks` --  as
/// one `put_field()` call PER BLOCK, under `name_prefix + "/" +
/// block.field_name`, using that block's own unmodified ids. Two
/// blocks can never collide because they are written under two
/// different qualified names, never the same name distinguished only by
/// a hoped-for-unique id.
[[nodiscard]] Result<void> put_field_per_block(
    RestartWriter& writer, const std::string& name_prefix,
    std::span<const RestartRowBlock> blocks, std::span<const double> flat,
    int components);

/// The read-back counterpart of put_field_per_block(). Returns the
/// FIRST block's own error unmodified (missing-id/missing-field error
/// codes are identical to the single-call case, so an existing caller's
/// `if (!r.ok()) { check error().code; ... }` pattern needs no change
/// beyond looping) -- not a reduction across blocks, since the first
/// failing id already ends the attempt.
[[nodiscard]] Result<void> get_field_per_block(
    RestartReader& reader, const std::string& name_prefix,
    std::span<const RestartRowBlock> blocks, std::span<double> flat,
    int components);

/// Sum of every block's own row count -- the `n` an accelerator's flat
/// buffer must have for the SAME `blocks` list.
[[nodiscard]] std::size_t total_rows(std::span<const RestartRowBlock> blocks);

// The dense least-squares solve is not declared here: it moved to
// fuse/linalg/least_squares.hpp, which is where a backend can be substituted
// for it. The accelerator calls linalg::least_squares_into() with its own
// scratch.

}  // namespace detail

}  // namespace fuse
