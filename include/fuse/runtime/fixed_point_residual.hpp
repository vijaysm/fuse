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

}  // namespace fuse
