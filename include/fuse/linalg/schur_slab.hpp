#pragma once

/// An exact distributed direct solver for `A = I - coefficient * L_h`, `L_h`
/// the periodic 5-point Laplacian on an `nx * ny` cell grid
/// (`examples/brusselator/brusselator_physics.hpp`'s own `periodic_laplacian`
/// builds the identical operator, canonical index `j*nx+i`), rows
/// partitioned into contiguous ROW BANDS of `j`
/// (`fuse::native::rank_slice(ny, rank, nranks)`,
/// `src/backend/native_generators.cpp` -- the same split
/// `examples/brusselator/brusselator_physics.hpp`'s own `NativeGrid` already
/// names a "row band" for exactly this reason: the domain-decomposition
/// literature's term is SUBDOMAIN, and this project's existing code already
/// settled on "row band" for one rank's contiguous range of `j`, keeping
/// `docs/user/glossary.md`'s own "slab" entry (an interval of the common
/// refinement, a TIME-domain concept) unambiguous. `SchurSlabSolver`'s own
/// name and this file's own name predate this note; every doc comment below says "row band",
/// never "slab", for the partition itself.
///
/// `docs/dev/schur-slab-solver.md` has the full derivation: which rows are
/// INTERFACE rows (a row band's first and last row, including the periodic
/// wrap between the last band and the first), the block system
/// `[A_II A_IG; A_GI A_GG]`, the Schur complement
/// `S = A_GG - A_GI A_II^-1 A_IG` assembled by every rank computing its own
/// contribution and one `Reducer::allgather`, solved identically (dense
/// LDLT) on every rank, then local back-substitution -- and the operation
/// count each step costs.
///
/// PIMPL: Eigen (`SimplicialLDLT`, dense `LDLT`) does the actual
/// factorisations, entirely in `src/linalg/schur_slab.cpp`; this header
/// names neither Eigen nor `mpi.h` (`docs/dev/layering.md`: `linalg`'s
/// public headers carry no third-party type at all), the same boundary
/// `fuse::search::distributed`'s own header already draws for its own
/// MPI-only translation unit.

#include <cstdint>
#include <memory>

#include "fuse/base/result.hpp"
#include "fuse/linalg/comm.hpp"              // Reducer
#include "fuse/linalg/dist_sparse_matrix.hpp"  // RowOwnership
#include "fuse/linalg/dist_vector.hpp"        // DistVector

namespace fuse::linalg {

/// One rank's factored, ready-to-solve operator for its own row band.
/// `solve()` reads and writes OWNED-ONLY vectors in the band's own local
/// canonical order `(j - j_lo)*nx + i` -- no ghosts, since the interface
/// exchange this solver needs (one dense-row allgather) is entirely
/// internal to `factor()`/`solve()`, never exposed through
/// `DistVector::update_ghosts()`.
class SchurSlabSolver {
 public:
  SchurSlabSolver(SchurSlabSolver&&) noexcept;
  SchurSlabSolver& operator=(SchurSlabSolver&&) noexcept;
  SchurSlabSolver(const SchurSlabSolver&) = delete;
  SchurSlabSolver& operator=(const SchurSlabSolver&) = delete;
  ~SchurSlabSolver();

  /// Factors `A = I - coefficient * L_h` restricted to this rank's own row
  /// band `[j_lo, j_hi)` of an `nx * ny` periodic grid. `row_ownership`
  /// carries every rank's own `j_lo` in the SAME convention
  /// `DistSparseMatrix::RowOwnership` already uses for a row-indexed
  /// partition (ascending, one entry per participant, the last
  /// participant's own upper bound being `ny` rather than a further
  /// entry) -- reinterpreted here over the grid's `ny` ROWS rather than its
  /// `nx*ny` CELLS, which is what lets this rank resolve which of its two
  /// neighbours in the periodic ring of row bands owns the row just
  /// outside its own `[j_lo, j_hi)`, with no communication (`reducer`'s
  /// `rank()`/`size()` alone fix the ring: predecessor `(rank-1+nranks) %
  /// nranks`, successor `(rank+1) % nranks`, exactly as `NativeGrid`'s own
  /// `owner_of_row` resolves a row's band from `rank_slice` alone).
  ///
  /// Refactor when `coefficient` (`dt*D`) changes: this call factors once
  /// for a fixed `coefficient` and reuse across repeated `solve()` calls at
  /// that step size is the caller's job -- deciding WHEN to refactor is a
  /// caller policy (`DirectDiffusionSolver::factor`'s own
  /// `factored_step_`-guarded re-factor, `examples/brusselator/
  /// species_core.hpp`, is the precedent this mirrors), not this solver's.
  ///
  /// Errors, never throws:
  ///   * `E-LINALG-SCHUR-SLAB` -- this rank's own row band owns fewer than
  ///     2 rows (`j_hi - j_lo < 2`) while `reducer->size() > 1`: a row band
  ///     needs 2 DISTINCT rows to have a first and a last interface row at
  ///     all; a 1-row (or empty) band would have to use its one row as
  ///     both at once, double-counting it between the interior block (zero
  ///     rows either way) and an interface block that is supposed to carry
  ///     exactly 2 row-widths. Refused rather than handled: a legitimate
  ///     caller sizes `nranks` so that `ny >= 2*nranks`, which this grid
  ///     and partition were always meant to allow; collapsing to a
  ///     1-row-wide interface silently would hide that misconfiguration
  ///     instead of surfacing it (see docs/dev/schur-slab-solver.md's own
  ///     "what degenerates" section for the count this bounds).
  ///   * `E-LINALG-SCHUR-SLAB-FACTORIZE` -- a factorisation this call
  ///     performs (the local interior `SimplicialLDLT`, the interior solve
  ///     that builds the Schur correction, or the assembled Schur
  ///     complement's own dense `LDLT`) reported `Eigen::Success` did not
  ///     hold. Not expected for `A = I - coefficient*L_h` at any
  ///     `coefficient` this operator is actually used at (a diagonally
  ///     dominant, symmetric positive definite matrix for `coefficient <=
  ///     0` and Laplacian magnitudes this grid produces), reported rather
  ///     than asserted because a factorisation's numerical success is a
  ///     property of the INPUT, not an invariant of this library's own
  ///     logic.
  [[nodiscard]] static Result<SchurSlabSolver> factor(int nx, std::int64_t ny, std::int64_t j_lo,
                                                      std::int64_t j_hi, double coefficient,
                                                      const RowOwnership& row_ownership,
                                                      std::shared_ptr<const Reducer> reducer);

  /// `x_owned <- A^-1 rhs_owned`, both this rank's own row band, in local
  /// canonical order `(j - j_lo)*nx + i`, `owned_size() == (j_hi-j_lo)*nx`.
  /// COLLECTIVE under MPI: every rank named in this solver's own
  /// `row_ownership` must call this when any one of them does, the same
  /// discipline `GhostExchanger::update`'s own documentation states for
  /// the same reason (one `Reducer::allgather` inside).
  ///
  /// Errors, never throws: `E-LINALG-SCHUR-SLAB-SOLVE`, the `solve()`-time
  /// sibling of `E-LINALG-SCHUR-SLAB-FACTORIZE` above (a factored solve
  /// reporting its own numerical failure, never expected, reported rather
  /// than asserted for the same reason).
  [[nodiscard]] Result<void> solve(const DistVector& rhs_owned, DistVector& x_owned) const;

 private:
  SchurSlabSolver();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse::linalg
