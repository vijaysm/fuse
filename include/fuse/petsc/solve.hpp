#pragma once

/// The PETSc substrate's own driver: `fuse::petsc::solve()`, a `SNES`
/// (`SNESNEWTONLS`) solve over the SAME `fuse::num::ResidualFn` shape
/// (`fuse/num/newton_krylov.hpp`) the native `fuse::num::solve()` already
/// takes, returning the SAME `fuse::num::NewtonResult` -- a Newton-region
/// caller (a later step's graph-level substrate selection) can call
/// either substrate behind one call shape, and this track's own
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
/// overrides is free to call `SNESSetFromOptions()` on the handle (a later
/// step's own concern, if ever needed) -- not something this header
/// exposes, since this track's own task is driving the SAME options the
/// native solver already accepts, not a second, parallel options surface.

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
  /// this track's own single-rank tier-0 scope (the window/multi-rank
  /// lift this track does not attempt, `KNOWN_ISSUES.md` D31's own
  /// scope).
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
/// Single-rank ONLY (`x.owned_size() == x.size()`, no ghost region) --
/// this track's own scope, matching `fuse::num::solve()`'s own
/// `Runner::region_fixed_point_residual()` restriction and
/// `check_region_supports_newton()`'s single-participant gate
/// (`docs/dev/newton-krylov.md` Sec. 4's own "Scope" paragraph); refuses
/// `E-PETSC-SOLVE-MULTIRANK` otherwise rather than silently dropping the
/// ghost region.
///
/// `precond_apply`/`refresh_precond` may both be `nullptr`, identically
/// to `fuse::num::solve()`'s own meaning (no preconditioning, `PCNONE`).
[[nodiscard]] Result<fuse::num::NewtonResult> solve(
    const fuse::num::ResidualFn& f, DistVector& x,
    const fuse::num::PreconditionerApplyFn& precond_apply,
    const fuse::num::RefreshPreconditionerFn& refresh_precond,
    const SnesOptions& opt = {});

}  // namespace fuse::petsc
