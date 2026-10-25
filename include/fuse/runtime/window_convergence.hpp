#pragma once

/// AGREED CONVERGENCE for a window-scoped
/// iteration region. Separated from the MPI call that gathers the input,
/// for the same reason `epoch_schedule.hpp`/`rank_agreement.hpp` are: "a
/// collective produces an array, and a rule reads it" -- a mistake in the
/// DECISION is a wrong number, checkable in a plain unit test, while a
/// mistake in the COLLECTIVE is a hang, observable only under `mpiexec`.
/// Keeping them apart means the decision itself is tested at tier 0, the
/// platform most of this project's suite runs under.
///
/// A convergence check of the form `R.convergence.
/// all_satisfied(R.residual_norms, R.reference_norms)` over `R.block_
/// residual`/`R.block_value` computed entirely from THIS rank's own
/// buffers is fine for an ordinary iteration region, whose own cross-rank
/// agreement is `reduce_region_residuals()` (a binomial reduce-then-
/// broadcast, `epoch_schedule.hpp`). A window-scoped region's own blocks
/// are tied to `WindowMemberConfig::hosted`, not
/// `RegionBlock::length > 0` -- a non-hosted rank's local `block_residual`/
/// `block_value` are computed from whatever stale buffer sits in its own
/// `slots[]` (nothing writes there; `advance_window()` skips a non-hosted
/// member entirely), not a legitimate "I own zero of this" the existing
/// combine formula already tolerates. `decide_window_convergence` is
/// therefore a flat Allgather-and-sum, not a tree-combine: every
/// participant's own (possibly-zero, possibly-garbage-but-EXCLUDED)
/// contribution is gathered whole, and only the HOSTED contributions are
/// ever added.

#include <cmath>
#include <cstddef>
#include <vector>

#include "fuse/model/iteration.hpp"

namespace fuse {

/// One participating rank's own contribution to one window-region block's
/// convergence measure for one sweep, already squared (so agreement is a
/// plain sum, never a combine formula) -- gathered from every participant
/// via a single, fixed-size `MPI_Allgather` (every rank's own array is the
/// SAME length: one entry per `RegionRuntime::blocks`, computed from the
/// SAME compiled stream on every rank), in block order.
struct WindowBlockSample {
  /// This rank's own `||G(y)-y||^2` for this block, meaningful only when
  /// `owned` is true -- a non-owning rank reports exactly 0.0 here, never
  /// whatever its own stale local buffer happens to hold (reading that
  /// into a sum this rank does not own, even multiplied by zero
  /// elsewhere, risks propagating a NaN from truly uninitialized memory;
  /// the caller excludes it BEFORE it reaches this struct).
  double residual_sq = 0.0;
  /// This rank's own `||G(y)||^2` for this block, same convention.
  double reference_sq = 0.0;
  /// Whether this rank hosts (window-scoped: `WindowMemberConfig::
  /// hosted`) or owns (ordinary: `RegionBlock::length > 0`) any part of
  /// this block this sweep.
  bool owned = false;
};

/// The outcome of agreeing one sweep's convergence decision across every
/// gathered rank.
struct WindowConvergenceResult {
  /// `measures.all_satisfied(residual_norms, reference_norms)` against the
  /// AGREED per-measure norms -- identical by construction on every
  /// participant, since every participant computed it from the identical
  /// `gathered` array.
  bool converged = false;
  /// The index of the first block no participating rank owns any part of,
  /// or -1 when every block had at least one owner. Mirrors `reduce_
  /// region_residuals()`'s own "no participant owns this block" check
  /// (E-RT-REGION-UNMEASURED) -- a setup defect, not a converged zero; the
  /// caller turns this into a Failure rather than trusting `converged`.
  int unmeasured_block = -1;
  /// The agreed per-block residual/value norms this sweep -- `sqrt` of the
  /// summed squares across every owning rank. The caller copies these into
  /// `RegionRuntime::block_residual`/`block_value` so every later reader
  /// (the residual history, the accelerator, the NEXT sweep's own
  /// `block_first` capture) sees the agreed value, not this rank's own
  /// local one.
  std::vector<double> block_residual;
  std::vector<double> block_value;
};

/// Pure function, no MPI (testable at tier 0 with hand-built `gathered`
/// data, including a single "rank" -- the single-process case, where
/// `gathered.size() == 1` and every block's lone sample is `owned` by
/// construction, reproduces today's un-agreed local computation exactly,
/// bit for bit, since there is nothing else to sum).
///
/// `gathered[rank][block]` is every participating rank's own
/// `WindowBlockSample`, already Allgathered by the caller. Sums each
/// block's own `residual_sq`/`reference_sq` across ranks in ASCENDING
/// RANK-INDEX order -- fixed regardless of how many ranks participate, so
/// a 1-rank, 2-rank and 3-rank run covering the same blocks sum
/// identically. `measure_block[mi]` names which block measure `mi` reads
/// (`RegionRuntime::measure_block`'s own convention); `block_first[bi]` is
/// block `bi`'s own sweep-0 residual, used for a `MeasureKind::
/// ResidualRelative` reference and (when `first_sweep`) OVERWRITTEN here
/// with this sweep's own agreed residual, exactly mirroring `run_region()`'s
/// own pre-fix capture (`if (k == 0) R.block_first[bi] = R.block_residual
/// [bi];`, itself downstream of that sweep's own agreement) -- so a
/// caller no longer needs to repeat that assignment itself.
[[nodiscard]] inline WindowConvergenceResult decide_window_convergence(
    const std::vector<std::vector<WindowBlockSample>>& gathered,
    const ConvergenceCriterion& measures,
    const std::vector<std::size_t>& measure_block,
    std::vector<double>& block_first, bool first_sweep,
    std::vector<double>& residual_norms_out,
    std::vector<double>& reference_norms_out) {
  WindowConvergenceResult out;
  const std::size_t nblocks = gathered.empty() ? 0 : gathered.front().size();
  out.block_residual.assign(nblocks, 0.0);
  out.block_value.assign(nblocks, 0.0);

  for (std::size_t bi = 0; bi < nblocks; ++bi) {
    double rs = 0.0, vs = 0.0;
    bool any_owner = false;
    for (std::size_t r = 0; r < gathered.size(); ++r) {
      if (bi >= gathered[r].size()) continue;
      const WindowBlockSample& s = gathered[r][bi];
      if (!s.owned) continue;
      any_owner = true;
      rs += s.residual_sq;
      vs += s.reference_sq;
    }
    if (!any_owner) {
      out.unmeasured_block = static_cast<int>(bi);
      return out;
    }
    out.block_residual[bi] = std::sqrt(rs);
    out.block_value[bi] = std::sqrt(vs);
  }

  if (first_sweep) {
    block_first.assign(nblocks, 0.0);
    for (std::size_t bi = 0; bi < nblocks; ++bi) {
      block_first[bi] = out.block_residual[bi];
    }
  }

  residual_norms_out.assign(measure_block.size(), 0.0);
  reference_norms_out.assign(measure_block.size(), 0.0);
  for (std::size_t mi = 0; mi < measure_block.size(); ++mi) {
    const std::size_t bi = measure_block[mi];
    if (bi >= nblocks) continue;
    residual_norms_out[mi] = out.block_residual[bi];
    reference_norms_out[mi] =
        measures.measures()[mi].kind == MeasureKind::ResidualRelative
            ? (bi < block_first.size() ? block_first[bi] : 0.0)
            : out.block_value[bi];
  }
  out.converged = measures.all_satisfied(residual_norms_out, reference_norms_out);
  return out;
}

}  // namespace fuse
