#pragma once

/// S7.1's Newton-Krylov seam: the coupled fixed-point residual over one
/// iteration region, the thing a Newton loop (S7.2, a separate, not-yet-
/// landed track) drives to zero instead of letting `Runner` run Picard's
/// own sweep loop for that region.
///
/// `G` is one Picard sweep of the region's body: deliver the stacked
/// interface vector `x` into every member's `Requires` storage, re-advance
/// every checkpointed member over the SAME interval, and read back what
/// the region's own dataflow (Transport/Remap/Accumulate/...) produces at
/// each block's residual arc. `F(x) = x - G(x)` is the quantity a Newton
/// iteration zeroes; see `Runner::region_fixed_point_residual()`
/// (fuse/runtime/runner.hpp) for how one is obtained and the exact
/// save/restore contract that makes evaluating it side-effect-free.

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"

namespace fuse {

/// A callable evaluating `F(x) = x - G(x)` over one iteration region's
/// stacked interface vector, in the calling convention the JFNK track can
/// wrap directly: a `std::function` over `std::span<const double> in /
/// std::span<double> out`, matching `fuse::linalg::MatrixFreeOperator`'s
/// own `apply()` shape but over spans rather than a `DistVector` -- the
/// JFNK track adapts one into the other for its own distributed Krylov
/// solve; this seam stays serial and span-based because that is all S7.1
/// itself needs to prove.
struct FixedPointResidual {
  /// `F(x) = x - G(x)`. `in.size() == out.size() == n`. Evaluating this
  /// leaves every region member's own state, and every coupler-owned
  /// loop-carried buffer the sweep touches, exactly as it found them --
  /// see `Runner::region_fixed_point_residual()`'s own comment for the
  /// save/restore bracket that makes that true, and for the scope this is
  /// restricted to (a single participant; no window-scoped or externally
  /// driven member).
  std::function<Result<void>(std::span<const double> in,
                             std::span<double> out)>
      evaluate;

  /// Length of the stacked interface vector `evaluate` takes and returns.
  std::size_t n = 0;

  /// One entry per block (one entry per residual the region carries), in
  /// the SAME order `evaluate`'s vector is laid out: `offsets[i]` is where
  /// block i starts, `lengths[i]` how long it is -- offsets a caller
  /// (e.g. the JFNK track's own block-Jacobi preconditioner) needs to
  /// scatter a global Newton iterate into each member's own slice, or
  /// gather per-block Jacobian actions back. `channel_names[i]` is
  /// `RegionBlock::channel_name` (src/runtime/runner_impl.hpp) for that
  /// block, so a caller can report which physical channel a Newton
  /// divergence traces to.
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> lengths;
  std::vector<std::string> channel_names;
};

/// S7.2 WindowNewton (newton-krylov.md Sec. 9.1-9.2, 9.7's own test plan):
/// the window-scoped sibling of `FixedPointResidual`, letting
/// `window_fixed_point_residual_test.cpp` exercise `Runner::Impl::
/// evaluate_window_region_residual()` the same way `FixedPointResidual`
/// lets S7.1's own test exercise the ordinary seam -- see
/// `Runner::window_region_fixed_point_residual()` (fuse/runtime/
/// runner.hpp) for how one is obtained. TEST/DIAGNOSTIC-ONLY: no
/// production caller drives a window region this way (`run_region_
/// newton_window()` calls `evaluate_window_region_residual()` directly).
struct WindowFixedPointResidual {
  /// `x`/`f_out` are `n` long, in the SAME (element,sample)-major,
  /// GaussSeidel-excluded-blocks-never-appear-here layout Sec. 9.1
  /// describes. `g_out`, when non-empty, additionally receives `G(x)`
  /// (`evaluate_window_region_residual()`'s own `g_out`); pass an empty
  /// span when only `F` is wanted. `iteration` is `advance_window()`'s
  /// own parameter, forwarded verbatim -- `0` exactly once per window
  /// instance (the predictor-seeding call, `x` ignored), `1` for every
  /// other call; see `evaluate_window_region_residual()`'s own comment
  /// for the full two-call convention and the `E-RT-WINDOW-SWEEP`
  /// refusal a third value earns. `commit`, when true, makes this
  /// window instance's own result real (the `.history` refresh) instead
  /// of restoring every snapshot this call took.
  std::function<Result<void>(int iteration, std::span<const double> x,
                              std::span<double> f_out, std::span<double> g_out,
                              bool commit)>
      evaluate;

  /// Length of `x`/`f_out` -- `RegionRuntime::newton_n`, the Newton-
  /// included blocks only.
  std::size_t n = 0;
  std::vector<std::size_t> offsets;
  std::vector<std::size_t> lengths;
  std::vector<std::string> channel_names;

  /// The GaussSeidel-excluded blocks (`RegionBlock::newton_offset ==
  /// RegionBlock::kNotInNewtonVector`) -- never appear in `x`/`f_out`,
  /// listed here so a test can confirm WHICH channel was excluded, not
  /// only that the total length dropped by the right amount (Sec. 9.7's
  /// own `GaussSeidelUnlaggedBlockIsExcludedFromX`).
  std::vector<std::size_t> excluded_lengths;
  std::vector<std::string> excluded_channel_names;
};

/// TEST/DIAGNOSTIC-ONLY (Sec. 9.7's own `TheIterationZeroCallSeedsThePass
/// AndIterationOneDoesNotReseedIt` and the mutation rows for Sec. 9.2's
/// table): one window member's own checkpoint state, the fields
/// `evaluate_window_region_residual()`'s own save/restore bracket reads
/// and writes -- `Runner::debug_window_member_snapshots()` (fuse/
/// runtime/runner.hpp) returns one of these per `RegionRuntime::
/// window_member_nodes`, in that order.
struct WindowMemberDebugSnapshot {
  /// This member's own `ComponentSet` slot (`WindowMemberConfig::
  /// component_slot`) -- resolve to a name via `components.at(slot).
  /// name()` using the SAME `ComponentSet` the region was initialized
  /// with; there is no node-to-name map on the public side of this
  /// seam otherwise.
  std::size_t component_slot = 0;

  /// `WindowMemberState::clock`/`straddle_pending`/`straddle_window_
  /// start`/`straddle_window_end` (fuse/runtime/window_coupler.hpp) --
  /// everything `evaluate_window_region_residual()`'s own save/restore
  /// bracket touches for this member EXCEPT `history`/`last_window_
  /// waveform` (neither copyable nor needed by Sec. 9.7's own tests).
  Rational clock{0};
  bool     straddle_pending = false;
  Rational straddle_window_start{0};
  Rational straddle_window_end{0};

  /// `Runner::Impl::window_pass_start_clock`/`window_start_value` for
  /// this member -- absent (both flags false, `pass_start_value` empty)
  /// until this pass's own `iteration == 0` sweep writes them; THE SAME
  /// value, bit for bit, on every later call within the SAME pass (Sec.
  /// 9.3's own two-call convention) -- that invariant is what Sec. 9.7's
  /// own test 3 pins directly.
  bool                has_pass_start_clock = false;
  Rational            pass_start_clock{0};
  std::vector<double> pass_start_value;
};

}  // namespace fuse
