#pragma once

/// The PETSc substrate's own driver: `fuse::petsc::solve()`, a `SNES`
/// (`SNESNEWTONLS`) solve over the SAME `fuse::num::ResidualFn` shape
/// (`fuse/num/newton_krylov.hpp`) the native `fuse::num::solve()` already
/// takes, returning the SAME `fuse::num::NewtonResult` -- a Newton-region
/// caller (the graph-level substrate selection) can call
/// either substrate behind one call shape, and
/// `test/petsc/snes_test.cpp` runs both over the IDENTICAL problem and
/// options to compare them directly.
///
/// NO PETSC TYPE NAMES THIS FILE -- see fuse/petsc/token.hpp's own header
/// comment for the layering rule this shares.
///
/// `SNESSetFromOptions()` is deliberately NEVER called: every option this
/// driver sets comes from `SnesOptions`/`fuse::num::NewtonKrylovOptions`,
/// which is FUSE's own configuration (a `NewtonConfig`, read from a YAML
/// graph or a C++-authored one -- fuse/model/iteration.hpp), not the
/// process's command line. A caller that also wants `-snes_*` command-line
/// overrides is free to call `SNESSetFromOptions()` on the handle -- not something this header
/// exposes, since the aim is to drive the SAME options the native solver
/// accepts, not a second, parallel options surface.

#include <cstddef>
#include <memory>

#include "fuse/base/result.hpp"
#include "fuse/linalg/dist_vector.hpp"
#include "fuse/num/newton_krylov.hpp"

namespace fuse::petsc {

using fuse::Result;
using fuse::linalg::DistVector;

/// `fuse::num::NewtonKrylovOptions` plus the one thing the native solver's
/// own options struct has no notion of at all: WHICH communicator the
/// `SNES` (and every `Vec`/`Mat`/`KSP`/`PC` it owns) is built on. Every
/// other field is read IDENTICALLY to the native solver's own meaning
/// (`docs/dev/newton-krylov.md` Sec. 10's own option table) -- this is a
/// deliberate reuse of the native struct, not a parallel declaration that
/// could drift from it: `rtol`/`atol`/`stol`/`divergence_tolerance`/
/// `max_nonlinear_iterations`/`max_function_evaluations` map onto
/// `SNESSetTolerances`/`SNESSetDivergenceTolerance`/
/// `SNESSetMaxFunctionEvaluations` directly; `forcing` maps onto
/// `SNESKSPSetUseEW`/`SNESKSPSetParametersEW` with the SAME six
/// constants; `line_search.kind` selects `SNESLINESEARCHBT`/`CP`/`NONE`;
/// `gmres.restart`/`gmres.max_iterations` map onto
/// `KSPGMRESSetRestart`/`KSPSetTolerances`'s own `max_it`;
/// `mffd.error_rel` maps onto `MatMFFDSetFunctionError` (`mffd.
/// min_direction_norm` has no PETSc counterpart -- `MATMFFD_WP` has no
/// floor of its own, `fuse/petsc/token.hpp`'s own `MffdTokenOptions`
/// comment); `preconditioner_lag` maps onto a caller-driven refresh
/// schedule identical in spirit to `-snes_lag_jacobian`, implemented in
/// `src/petsc/solve.cpp` rather than handed to SNES's own (Jacobian,
/// not preconditioner-callback) lag option, since this driver's own
/// Jacobian is always the matrix-free MFFD action (never lagged) and
/// only the PRECONDITIONER is what `preconditioner_lag` controls, exactly
/// the native solver's own split.
struct SnesOptions {
  fuse::num::NewtonKrylovOptions newton{};

  /// `MPI_Comm_c2f()` handle the `SNES` and every PETSc object it owns is
  /// built on. `fuse::petsc::comm_self_f()` (fuse/petsc/token.hpp) for
  /// the single-rank case.
  int comm_f = 0;
};

/// `fuse::num::ResidualFn` is `fuse::linalg::DistVector`-based
/// (fuse/num/newton_krylov.hpp) -- this driver copies `x`'s owned block
/// into a PETSc `Vec` once per call and back, the SAME boundary
/// `fuse::petsc::MffdToken`'s own span-based `ResidualSpanFn` crosses
/// (src/petsc/solve.cpp adapts one into the other with a one-line
/// lambda), rather than inventing a second residual-function shape for
/// this driver alone.
///
/// MULTI-RANK over `SnesOptions::comm_f`: `X`/
/// `F` are built `VecCreate`+`VecSetSizes(x.owned_size(), PETSC_
/// DETERMINE)`+`VecSetType(VECSTANDARD)` on `comm_f`, so PETSc's own
/// `VecDot`/`VecNorm` (and therefore every GMRES inner product and the
/// outer convergence test SNES itself runs) reduce across EVERY rank of
/// `comm_f`, not just this one -- a rank owning zero unknowns
/// (`x.owned_size() == 0`) is legal as long as `comm_f` has more than
/// one rank (refused, `E-PETSC-SOLVE-SHAPE`, only when the WHOLE
/// communicator is empty). STILL refuses `x` carrying a nonzero GHOST
/// region (`E-PETSC-SOLVE-GHOSTED`) -- that is a property of the vector
/// shape, orthogonal to rank count: every window- or ordinary-Newton `x`
/// this library builds (`run_region_newton_window()`/`run_region_
/// newton()`) carries zero ghosts regardless of how many ranks it is
/// split across (`docs/dev/newton-krylov.md` Sec. 9.4's own analysis of
/// why), so this refusal is not expected to fire for any caller inside
/// this library; a caller with a genuinely ghosted vector would need
/// this driver to synchronise it through PETSc's own `VecGhost`
/// mechanism, which it does not implement. Refuses `E-PETSC-SOLVE-
/// NOCOMM` when `comm_f` resolves to `MPI_COMM_NULL` -- a caller must
/// always supply a real communicator (`fuse::petsc::comm_self_f()` for
/// a single-rank region, the region's own participant communicator
/// otherwise), never leave `comm_f` at its zero default and expect a
/// silent fallback.
///
/// `precond_apply`/`refresh_precond` may both be `nullptr`, identically
/// to `fuse::num::solve()`'s own meaning (no preconditioning, `PCNONE`).
[[nodiscard]] Result<fuse::num::NewtonResult> solve(
    const fuse::num::ResidualFn& f, DistVector& x,
    const fuse::num::PreconditionerApplyFn& precond_apply,
    const fuse::num::RefreshPreconditionerFn& refresh_precond,
    const SnesOptions& opt = {});

// ===========================================================================
// Registering this substrate with fuse::num's
// Newton registry.
// ===========================================================================

/// Registers `fuse::num::NewtonSubstrate::Petsc` (fuse/num/
/// newton_krylov.hpp) against this file's own `solve()`, so a Newton-
/// selected region whose graph names the PETSc substrate
/// (`fuse::NewtonSubstrate::Petsc`, fuse/model/iteration.hpp) dispatches
/// here through the registry rather than through a direct call this
/// layer's own placement (`tools/check-layering.py`'s topmost `petsc`
/// layer, ABOVE `fuse::runtime`) would make impossible for the Runner to
/// make itself.
///
/// COMMUNICATOR. The registered entry
/// point reads the communicator `fuse::num::NewtonSolveFn`'s own
/// `comm_f` parameter carries (`newton_krylov.hpp`'s own header comment
/// on that parameter) -- the SAME split `RegionRuntime::newton_reducer`
/// reduces over, built once in `resolve_regions()` and threaded through
/// `run_region_newton()`/`run_region_newton_window()`'s own dispatch
/// call. `comm_f == 0` (a single-rank region, or no communicator
/// configured -- `check_region_supports_newton()` already refuses the
/// opposite case, more than one participant with none) falls back to
/// `comm_self_f()` (`MPI_COMM_SELF`): that is the honest
/// communicator for a problem this seam already guarantees is rank-
/// local in that shape, and it still sidesteps the real ownership
/// question `fuse::petsc::ensure_initialized()`'s own comment flags for
/// `MPI_COMM_SELF` specifically (several different single-rank regions'
/// first calls sharing one communicator need not agree on which one
/// initialises PETSc on it, since each is really MPI_COMM_SELF
/// regardless of which region's call reaches `ensure_initialized()`
/// first). A MULTI-RANK region's own `comm_f` is a REAL, per-region
/// split, never shared across two different regions' own participant
/// sets, so the identical coordination question does not arise for it
/// either -- `ensure_initialized()`'s own first-caller-wins contract is
/// unaffected either way.
///
/// Idempotent: calling this more than once simply re-registers the same
/// entry point (`fuse::num::register_newton_substrate()`'s own
/// "registers OR REPLACES" contract). Never called by this library
/// itself -- a binary that links `fuse-coupler-petsc` and wants the
/// PETSc substrate reachable through the registry calls this once,
/// typically at startup, before `Runner::initialize()`.
void install();

}  // namespace fuse::petsc
