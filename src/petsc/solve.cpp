// src/petsc/solve.cpp -- fuse::petsc::solve(), the SNES driver
// (fuse/petsc/solve.hpp's own header comment has the design; this file's
// own doc comment on the implementation lives at each mapping below,
// collected into docs/dev/newton-krylov.md Sec. 10's own option table).

#include "fuse/petsc/solve.hpp"

#include <petsc.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "fuse/linalg/comm.hpp"
#include "fuse/petsc/token.hpp"  // ensure_initialized

namespace fuse::petsc {

namespace {

Result<void> check(PetscErrorCode ierr, const std::string& what) {
  if (ierr) {
    return Failure{"E-PETSC-CALL", what + ": PETSc error code " + std::to_string(ierr)};
  }
  return Result<void>::success();
}

struct SnesFnContext {
  const fuse::num::ResidualFn* f = nullptr;
  fuse::linalg::DistVector* x_scratch = nullptr;
  fuse::linalg::DistVector* fx_scratch = nullptr;
  std::size_t n = 0;
  bool failed = false;  // sticky: SNESSolve's own return code is checked
                        // separately, but this records WHICH call failed.
};

PetscErrorCode snes_function(SNES, Vec x, Vec Fx, void* raw_ctx) {
  auto* ctx = static_cast<SnesFnContext*>(raw_ctx);
  const PetscScalar* xarr = nullptr;
  PetscScalar* farr = nullptr;
  PetscErrorCode ierr = VecGetArrayRead(x, &xarr);
  if (ierr) return ierr;
  ierr = VecGetArray(Fx, &farr);
  if (ierr) {
    VecRestoreArrayRead(x, &xarr);
    return ierr;
  }
  std::copy(xarr, xarr + ctx->n, ctx->x_scratch->data());
  Result<void> r = (*ctx->f)(*ctx->x_scratch, *ctx->fx_scratch);
  std::copy(ctx->fx_scratch->data(), ctx->fx_scratch->data() + ctx->n, farr);
  ierr = VecRestoreArrayRead(x, &xarr);
  if (ierr) return ierr;
  ierr = VecRestoreArray(Fx, &farr);
  if (ierr) return ierr;
  if (!r.ok()) {
    ctx->failed = true;
    return static_cast<PetscErrorCode>(PETSC_ERR_USER);
  }
  return PETSC_SUCCESS;
}

struct PrecondContext {
  const fuse::num::PreconditionerApplyFn* apply = nullptr;
  fuse::linalg::DistVector* r_scratch = nullptr;
  fuse::linalg::DistVector* z_scratch = nullptr;
  std::size_t n = 0;
};

PetscErrorCode snes_pc_apply(PC pc, Vec r, Vec z) {
  void* raw_ctx = nullptr;
  PetscErrorCode ierr = PCShellGetContext(pc, &raw_ctx);
  if (ierr) return ierr;
  auto* ctx = static_cast<PrecondContext*>(raw_ctx);
  const PetscScalar* rarr = nullptr;
  PetscScalar* zarr = nullptr;
  ierr = VecGetArrayRead(r, &rarr);
  if (ierr) return ierr;
  ierr = VecGetArray(z, &zarr);
  if (ierr) {
    VecRestoreArrayRead(r, &rarr);
    return ierr;
  }
  std::copy(rarr, rarr + ctx->n, ctx->r_scratch->data());
  Result<void> res = (*ctx->apply)(*ctx->r_scratch, *ctx->z_scratch);
  std::copy(ctx->z_scratch->data(), ctx->z_scratch->data() + ctx->n, zarr);
  ierr = VecRestoreArrayRead(r, &rarr);
  if (ierr) return ierr;
  ierr = VecRestoreArray(z, &zarr);
  if (ierr) return ierr;
  if (!res.ok()) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
  return PETSC_SUCCESS;
}

struct MonitorContext {
  fuse::num::NewtonResult* result = nullptr;
  const fuse::num::RefreshPreconditionerFn* refresh = nullptr;
  SNES snes = nullptr;
  Vec x_tmp = nullptr;
  Vec f_tmp = nullptr;
  fuse::linalg::DistVector* x_scratch = nullptr;
  fuse::linalg::DistVector* fx_scratch = nullptr;
  std::size_t n = 0;
  int preconditioner_lag = 1;
  int steps_since_refresh = 0;
};

PetscErrorCode snes_monitor(SNES snes, PetscInt its, PetscReal fnorm, void* raw_ctx) {
  auto* ctx = static_cast<MonitorContext*>(raw_ctx);
  fuse::num::NewtonIterationRecord rec;
  rec.iteration = static_cast<int>(its);
  rec.norm_f = fnorm;
  // SNESLineSearchGetLambda() after step `its` reports the step just
  // taken to REACH this iterate (undefined/1.0 at its==0, before any
  // step) -- recorded as a measured, best-effort analogue of the native
  // solver's own NewtonIterationRecord::line_search_lambda, not an exact
  // reproduction: PETSc's own SNESLineSearchReason distinguishes several
  // failure modes this mapping collapses to "FullStep" (lambda == 1) vs.
  // "Converged" (lambda < 1, a backtrack occurred), never "MaxStepsReached"/
  // "Failed" (an actual line-search failure already surfaces as the
  // SNES's own SNES_DIVERGED_LINE_SEARCH, mapped independently below).
  {
    SNESLineSearch ls = nullptr;
    if (SNESGetLineSearch(snes, &ls) == 0 && ls != nullptr) {
      PetscReal lambda = 1.0;
      if (SNESLineSearchGetLambda(ls, &lambda) == 0) {
        rec.line_search_lambda = lambda;
        rec.line_search_status = (lambda >= 1.0) ? fuse::num::LineSearchStatus::FullStep
                                                 : fuse::num::LineSearchStatus::Converged;
      }
    }
  }
  ctx->result->trace.push_back(rec);

  // Preconditioner refresh, lagged: "rebuild every preconditioner_lag
  // Newton steps" (fuse/num/newton_krylov.hpp's own NewtonKrylovOptions::
  // preconditioner_lag comment) -- `its == 0` is the INITIAL point
  // (before any step has been taken), the native solver's own "always
  // before the first" case; every later multiple of the lag refreshes
  // again, read from the SNES's own current iterate.
  if (ctx->refresh != nullptr && *ctx->refresh) {
    const bool is_first = (its == 0);
    const bool is_lag_boundary =
        ctx->preconditioner_lag > 0 && (its % ctx->preconditioner_lag == 0);
    if (is_first || is_lag_boundary) {
      Vec x = nullptr, f = nullptr;
      PetscErrorCode ierr = SNESGetSolution(snes, &x);
      if (ierr) return ierr;
      ierr = SNESGetFunction(snes, &f, nullptr, nullptr);
      if (ierr) return ierr;
      const PetscScalar* xarr = nullptr;
      const PetscScalar* farr = nullptr;
      ierr = VecGetArrayRead(x, &xarr);
      if (ierr) return ierr;
      ierr = VecGetArrayRead(f, &farr);
      if (ierr) {
        VecRestoreArrayRead(x, &xarr);
        return ierr;
      }
      std::copy(xarr, xarr + ctx->n, ctx->x_scratch->data());
      std::copy(farr, farr + ctx->n, ctx->fx_scratch->data());
      VecRestoreArrayRead(x, &xarr);
      VecRestoreArrayRead(f, &farr);
      Result<void> r = (*ctx->refresh)(*ctx->x_scratch, *ctx->fx_scratch);
      if (!r.ok()) return static_cast<PetscErrorCode>(PETSC_ERR_USER);
      rec.preconditioner_refreshed = true;
      ctx->result->trace.back().preconditioner_refreshed = true;
    }
  }
  return PETSC_SUCCESS;
}

// ---- the Eisenstat-Walker floor SNES does not have -------------------------
//
// fuse::num clamps every forcing term to [eta_min, eta_max]
// (src/num/newton_krylov.cpp, `std::clamp(eta, opt.eta_min, opt.eta_max)`).
// SNESKSPSetParametersEW has rtol_0, rtol_max, gamma, alpha, alpha2 and the
// safeguard threshold -- no floor. Without one the Brusselator window on
// this substrate reports DivergedLinearSolve on its FIRST window (with
// --driver runner --accel newton --substrate petsc --end 1.0:
// "DivergedLinearSolve after 3 Newton iteration(s), ||F|| = 0.000012"),
// the same stall the native solver showed before eta_min = 1e-4 (docs/dev/
// newton-krylov.md Sec. 9.9.2): the forcing term tightens past the accuracy
// a finite-difference Jacobian action can deliver, and GMRES exhausts its
// budget chasing it.
//
// SNES's EW sets the KSP's relative tolerance in a pre-solve hook, so the
// one place the floor can be applied to the value EW chose is the KSP's own
// convergence test, which GMRES calls first at iteration 0 before any
// residual is judged: read the tolerance EW set, raise it to eta_min if it
// is below, then hand over to KSPConvergedDefault unchanged. EW's post-solve
// reads the (clamped) tolerance back as rtol_last, exactly as the native
// solver carries the clamped eta forward.
struct ForcingFloorContext {
  PetscReal eta_min = 0.0;
  void* default_context = nullptr;  // KSPConvergedDefaultCreate's
};

PetscErrorCode converged_with_forcing_floor(KSP ksp, PetscInt n, PetscReal rnorm,
                                            KSPConvergedReason* reason, void* raw) {
  auto* ctx = static_cast<ForcingFloorContext*>(raw);
  if (n == 0) {
    PetscReal rtol = 0, abstol = 0, dtol = 0;
    PetscInt maxits = 0;
    PetscCall(KSPGetTolerances(ksp, &rtol, &abstol, &dtol, &maxits));
    if (rtol < ctx->eta_min) PetscCall(KSPSetTolerances(ksp, ctx->eta_min, abstol, dtol, maxits));
  }
  return KSPConvergedDefault(ksp, n, rnorm, reason, ctx->default_context);
}

// PetscCtxDestroyFn's argument is declared `void*` but is, by PETSc's own
// documentation of PetscCtxRt (petscsys.h), the ADDRESS of the context
// pointer -- "psychologically a void**" -- and the callee nulls it. The
// default context is destroyed the same way.
PetscErrorCode destroy_forcing_floor(void* raw) {
  auto** slot = static_cast<ForcingFloorContext**>(raw);
  ForcingFloorContext* ctx = *slot;
  PetscErrorCode ierr = KSPConvergedDefaultDestroy(&ctx->default_context);
  delete ctx;
  *slot = nullptr;
  return ierr;
}

/// `SNESConvergedReason` -> `fuse::num::NewtonStatus` (docs/dev/
/// newton-krylov.md Sec. 10's own table). Every reason the tests
/// can reach maps to a named native status; the six reasons that
/// have no native counterpart at all (`SNES_CONVERGED_ITS`,
/// `SNES_CONVERGED_USER`, `SNES_DIVERGED_INNER`, `SNES_DIVERGED_LOCAL_MIN`,
/// `SNES_DIVERGED_TR_DELTA`, `SNES_DIVERGED_USER`) fall to
/// `DivergedMaxIterations` as the closest "did not reach a named
/// convergence test" fallback -- recorded here rather than silently
/// reused, and never reached by `SNESNEWTONLS` with no trust region, no
/// user convergence/divergence test, and no inner solve, which is every
/// configuration this driver builds.
fuse::num::NewtonStatus map_reason(SNESConvergedReason reason) {
  switch (reason) {
    case SNES_CONVERGED_FNORM_ABS:
      return fuse::num::NewtonStatus::ConvergedAbsolute;
    case SNES_CONVERGED_FNORM_RELATIVE:
      return fuse::num::NewtonStatus::ConvergedRelative;
    case SNES_CONVERGED_SNORM_RELATIVE:
      return fuse::num::NewtonStatus::ConvergedStep;
    case SNES_DIVERGED_FUNCTION_COUNT:
      return fuse::num::NewtonStatus::DivergedFunctionCount;
    case SNES_DIVERGED_LINE_SEARCH:
      return fuse::num::NewtonStatus::DivergedLineSearch;
    case SNES_DIVERGED_DTOL:
      return fuse::num::NewtonStatus::DivergedResidualGrowth;
    case SNES_DIVERGED_LINEAR_SOLVE:
      return fuse::num::NewtonStatus::DivergedLinearSolve;
    case SNES_DIVERGED_MAX_IT:
      return fuse::num::NewtonStatus::DivergedMaxIterations;
    case SNES_DIVERGED_FUNCTION_DOMAIN:
    case SNES_DIVERGED_FUNCTION_NANORINF:
    case SNES_DIVERGED_JACOBIAN_DOMAIN:
    case SNES_DIVERGED_OBJECTIVE_DOMAIN:
      return fuse::num::NewtonStatus::DivergedFunctionError;
    default:
      return fuse::num::NewtonStatus::DivergedMaxIterations;
  }
}

const char* line_search_type_for(fuse::num::LineSearchKind kind) {
  switch (kind) {
    case fuse::num::LineSearchKind::BT:
      return SNESLINESEARCHBT;
    case fuse::num::LineSearchKind::CP:
      return SNESLINESEARCHCP;
    case fuse::num::LineSearchKind::Basic:
      return SNESLINESEARCHNONE;
  }
  return SNESLINESEARCHBT;
}

}  // namespace

Result<fuse::num::NewtonResult> solve(const fuse::num::ResidualFn& f, DistVector& x,
                                      const fuse::num::PreconditionerApplyFn& precond_apply,
                                      const fuse::num::RefreshPreconditionerFn& refresh_precond,
                                      const SnesOptions& opt) {
  // This refusal is about a NONZERO GHOST
  // region, never about rank count -- a window- or ordinary-Newton `x`
  // built by `run_region_newton_window()`/`run_region_newton()` always
  // carries `ghost_size() == 0` regardless of how many ranks it is
  // split across (Sec. 9.4/9.5's own analysis: every cross-rank
  // dependency this seam's F needs is already resolved by the compiled
  // Transport instructions the Runner's own body executes, never by a
  // ghost copy of a PEER's owned entries sitting in `x` itself) -- a
  // caller handing this driver a vector with real ghosts would need this
  // driver to synchronize them through PETSc's own `VecGhost` mechanism,
  // which it does not implement.
  if (x.owned_size() != x.size()) {
    return Failure{"E-PETSC-SOLVE-GHOSTED",
                   "fuse::petsc::solve: x carries a nonzero ghost region (owned " +
                       std::to_string(x.owned_size()) + ", total " + std::to_string(x.size()) +
                       "); this driver has no VecGhost synchronisation for one"};
  }
  const std::size_t n = x.owned_size();

  if (auto r = ensure_initialized(opt.comm_f); !r.ok()) return r.error();
  MPI_Comm comm = MPI_Comm_f2c(opt.comm_f);
  if (comm == MPI_COMM_NULL) {
    return Failure{"E-PETSC-SOLVE-NOCOMM",
                   "fuse::petsc::solve: SnesOptions::comm_f resolves to "
                       "MPI_COMM_NULL -- the caller must supply a real "
                       "communicator (fuse::petsc::comm_self_f() for a "
                       "single-rank region, or the region's own "
                       "participant communicator for a multi-rank one)"};
  }
  int comm_size = 1;
  if (auto r = check(MPI_Comm_size(comm, &comm_size), "MPI_Comm_size"); !r.ok()) return r.error();
  // A rank owning zero unknowns is legal: refused only when
  // the WHOLE communicator has nothing (comm_size == 1, the single-rank
  // shape this check already covered before the multi-rank lift) --
  // never refused here just because THIS rank's own local share happens
  // to be zero while a peer rank's is not; `VecCreate`/`VecSetSizes`
  // below accept a zero local size on an MPI Vec by PETSc's own design.
  if (n == 0 && comm_size <= 1) {
    return Failure{"E-PETSC-SOLVE-SHAPE", "fuse::petsc::solve: x is empty"};
  }

  auto exchanger = fuse::linalg::make_serial_exchanger();
  auto reducer = fuse::linalg::make_serial_reducer();
  fuse::linalg::DistVector x_scratch(n, exchanger, reducer);
  fuse::linalg::DistVector fx_scratch(n, exchanger, reducer);
  fuse::linalg::DistVector pc_r_scratch(n, exchanger, reducer);
  fuse::linalg::DistVector pc_z_scratch(n, exchanger, reducer);
  // `x_scratch`/`fx_scratch`/`pc_r_scratch`/`pc_z_scratch` stay built on
  // the SERIAL reducer/exchanger regardless of `comm`'s own rank count:
  // they are purely local adapters between PETSc's own raw `Vec` arrays
  // (`VecGetArray`) and `f`/`precond_apply`'s `DistVector&` call shape
  // (`snes_function`/`snes_pc_apply` below) -- neither callback ever
  // calls `.dot()`/`.norm2()`/`.sum()` on them, so their own Reducer is
  // never invoked; every REAL cross-rank reduction GMRES/SNES need goes
  // through `comm` itself, via the properly-distributed `Vec X`/`Vec F`
  // built below, not through this library's `fuse::linalg::Reducer`
  // abstraction at all.
  SnesFnContext fn_ctx{&f, &x_scratch, &fx_scratch, n, false};
  PrecondContext pc_ctx{&precond_apply, &pc_r_scratch, &pc_z_scratch, n};
  // Declared here, before any `goto fail` below, so every jump to `fail`
  // (none of which touch `result`) does not jump over its initialization
  // -- a plain C++ rule, not a PETSc one: a `goto` may not skip a
  // non-trivial variable's initializer still in scope at the label.
  fuse::num::NewtonResult result;

  // `VecCreate`+`VecSetSizes`+`VecSetType(...,
  // VECSTANDARD)`, not `VecCreateSeq` -- `VecCreateSeq` ignores `comm`'s
  // own size entirely (every rank would build its OWN disconnected
  // n-length vector, with VecDot/VecNorm reducing only LOCALLY, so SNES/
  // GMRES would run `comm_size` independent single-rank solves with no
  // real cross-rank Krylov coupling at all -- a silently wrong answer
  // for any communicator with more than one rank, not a refusal).
  // `VECSTANDARD` resolves to `VECSEQ` on a one-rank communicator and
  // `VECMPI` otherwise, automatically, with NO command-line option
  // parsing (`VecSetFromOptions()` would also read `-vec_type` and
  // similar, which this driver's own header comment already forbids for
  // `SNESSetFromOptions()`, for the identical reason: every choice here
  // comes from `SnesOptions`, never the process's command line) --
  // this is the single-rank case's own UNCHANGED behaviour (`VECSEQ`
  // is what `VecCreateSeq` would have built) plus the real multi-rank
  // one, from the SAME two calls. `PETSC_DETERMINE` for the global size:
  // PETSc sums every rank's own local `n` via its own collective, the
  // SAME total `RegionRuntime::newton_n`/`R.n` would give summed across
  // `R.participants` -- never computed redundantly here.
  Vec X = nullptr, F = nullptr;
  PetscErrorCode ierr = VecCreate(comm, &X);
  if (auto r = check(ierr, "VecCreate(X)"); !r.ok()) return r.error();
  ierr = VecSetSizes(X, static_cast<PetscInt>(n), PETSC_DETERMINE);
  if (auto r = check(ierr, "VecSetSizes(X)"); !r.ok()) { VecDestroy(&X); return r.error(); }
  ierr = VecSetType(X, VECSTANDARD);
  if (auto r = check(ierr, "VecSetType(X)"); !r.ok()) { VecDestroy(&X); return r.error(); }
  ierr = VecDuplicate(X, &F);
  if (auto r = check(ierr, "VecDuplicate(F)"); !r.ok()) {
    VecDestroy(&X);
    return r.error();
  }
  if (n > 0) {
    PetscScalar* arr = nullptr;
    VecGetArray(X, &arr);
    std::copy(x.data(), x.data() + n, arr);
    VecRestoreArray(X, &arr);
  }

  SNES snes = nullptr;
  ierr = SNESCreate(comm, &snes);
  if (auto r = check(ierr, "SNESCreate"); !r.ok()) {
    VecDestroy(&X);
    VecDestroy(&F);
    return r.error();
  }
  ierr = SNESSetType(snes, SNESNEWTONLS);
  if (auto r = check(ierr, "SNESSetType"); !r.ok()) goto fail;
  ierr = SNESSetFunction(snes, F, snes_function, &fn_ctx);
  if (auto r = check(ierr, "SNESSetFunction"); !r.ok()) goto fail;

  {
    Mat J = nullptr;
    ierr = MatCreateSNESMF(snes, &J);
    if (auto r = check(ierr, "MatCreateSNESMF"); !r.ok()) goto fail;
    ierr = MatMFFDSetType(J, const_cast<char*>(MATMFFD_WP));
    if (auto r = check(ierr, "MatMFFDSetType"); !r.ok()) { MatDestroy(&J); goto fail; }
    ierr = MatMFFDSetFunctionError(J, opt.newton.mffd.error_rel);
    if (auto r = check(ierr, "MatMFFDSetFunctionError"); !r.ok()) { MatDestroy(&J); goto fail; }
    ierr = SNESSetJacobian(snes, J, J, MatMFFDComputeJacobian, nullptr);
    if (auto r = check(ierr, "SNESSetJacobian"); !r.ok()) { MatDestroy(&J); goto fail; }
    // SNESSetJacobian retains its own reference; this token no longer
    // needs to hold one.
    MatDestroy(&J);
  }

  ierr = SNESSetTolerances(snes, opt.newton.atol, opt.newton.rtol, opt.newton.stol,
                           static_cast<PetscInt>(opt.newton.max_nonlinear_iterations),
                           static_cast<PetscInt>(opt.newton.max_function_evaluations));
  if (auto r = check(ierr, "SNESSetTolerances"); !r.ok()) goto fail;
  ierr = SNESSetDivergenceTolerance(snes, opt.newton.divergence_tolerance);
  if (auto r = check(ierr, "SNESSetDivergenceTolerance"); !r.ok()) goto fail;

  {
    SNESLineSearch ls = nullptr;
    ierr = SNESGetLineSearch(snes, &ls);
    if (auto r = check(ierr, "SNESGetLineSearch"); !r.ok()) goto fail;
    ierr = SNESLineSearchSetType(ls, line_search_type_for(opt.newton.line_search.kind));
    if (auto r = check(ierr, "SNESLineSearchSetType"); !r.ok()) goto fail;
    if (opt.newton.line_search.kind == fuse::num::LineSearchKind::Basic) {
      ierr = SNESLineSearchSetDamping(ls, opt.newton.line_search.damping);
      if (auto r = check(ierr, "SNESLineSearchSetDamping"); !r.ok()) goto fail;
    }
  }

  {
    const auto& ew = opt.newton.forcing;
    if (ew.choice == fuse::num::EisenstatWalkerChoice::Constant) {
      ierr = SNESKSPSetUseEW(snes, PETSC_FALSE);
      if (auto r = check(ierr, "SNESKSPSetUseEW(false)"); !r.ok()) goto fail;
    } else {
      const PetscInt version = (ew.choice == fuse::num::EisenstatWalkerChoice::Choice1) ? 1 : 2;
      ierr = SNESKSPSetUseEW(snes, PETSC_TRUE);
      if (auto r = check(ierr, "SNESKSPSetUseEW(true)"); !r.ok()) goto fail;
      // alpha2 has no counterpart in fuse::num::EisenstatWalkerOptions
      // (that struct names only `alpha`, choice 2's own exponent); passed
      // as `alpha` again here, a documented approximation
      // (docs/dev/newton-krylov.md Sec. 10) rather than a silent gap --
      // SNES's own alpha2 affects choice 1's safeguard term specifically,
      // which the test problems (quadratic, atan) select
      // choice 2 for, so the approximation is never exercised by anything
      // the tests measure.
      ierr = SNESKSPSetParametersEW(snes, version, ew.eta_initial, ew.eta_max, ew.gamma, ew.alpha,
                                    ew.alpha, ew.safeguard_threshold);
      if (auto r = check(ierr, "SNESKSPSetParametersEW"); !r.ok()) goto fail;
    }
  }

  {
    KSP ksp = nullptr;
    ierr = SNESGetKSP(snes, &ksp);
    if (auto r = check(ierr, "SNESGetKSP"); !r.ok()) goto fail;
    ierr = KSPSetType(ksp, KSPGMRES);
    if (auto r = check(ierr, "KSPSetType"); !r.ok()) goto fail;
    ierr = KSPGMRESSetRestart(ksp, static_cast<PetscInt>(opt.newton.gmres.restart));
    if (auto r = check(ierr, "KSPGMRESSetRestart"); !r.ok()) goto fail;
    ierr = KSPSetPCSide(ksp, PC_RIGHT);
    if (auto r = check(ierr, "KSPSetPCSide"); !r.ok()) goto fail;
    ierr = KSPSetTolerances(ksp, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT,
                            static_cast<PetscInt>(opt.newton.gmres.max_iterations));
    if (auto r = check(ierr, "KSPSetTolerances"); !r.ok()) goto fail;
    if (opt.newton.forcing.choice == fuse::num::EisenstatWalkerChoice::Constant) {
      ierr = KSPSetTolerances(ksp, opt.newton.forcing.eta_constant, PETSC_CURRENT, PETSC_CURRENT,
                              static_cast<PetscInt>(opt.newton.gmres.max_iterations));
      if (auto r = check(ierr, "KSPSetTolerances(constant eta)"); !r.ok()) goto fail;
    }
    if (opt.newton.forcing.eta_min > 0.0) {
      auto* floor_ctx = new ForcingFloorContext{};
      floor_ctx->eta_min = static_cast<PetscReal>(opt.newton.forcing.eta_min);
      ierr = KSPConvergedDefaultCreate(&floor_ctx->default_context);
      if (auto r = check(ierr, "KSPConvergedDefaultCreate"); !r.ok()) { delete floor_ctx; goto fail; }
      ierr = KSPSetConvergenceTest(ksp, converged_with_forcing_floor, floor_ctx, destroy_forcing_floor);
      if (auto r = check(ierr, "KSPSetConvergenceTest(forcing floor)"); !r.ok()) goto fail;
    }

    PC pc = nullptr;
    ierr = KSPGetPC(ksp, &pc);
    if (auto r = check(ierr, "KSPGetPC"); !r.ok()) goto fail;
    if (precond_apply) {
      ierr = PCSetType(pc, PCSHELL);
      if (auto r = check(ierr, "PCSetType(SHELL)"); !r.ok()) goto fail;
      ierr = PCShellSetContext(pc, &pc_ctx);
      if (auto r = check(ierr, "PCShellSetContext"); !r.ok()) goto fail;
      ierr = PCShellSetApply(pc, snes_pc_apply);
      if (auto r = check(ierr, "PCShellSetApply"); !r.ok()) goto fail;
      ierr = PCShellSetName(pc, "fuse-petsc-precond");
      if (auto r = check(ierr, "PCShellSetName"); !r.ok()) goto fail;
    } else {
      ierr = PCSetType(pc, PCNONE);
      if (auto r = check(ierr, "PCSetType(NONE)"); !r.ok()) goto fail;
    }
  }

  {
    MonitorContext mon_ctx{&result,
                           refresh_precond ? &refresh_precond : nullptr,
                           snes,
                           nullptr,
                           nullptr,
                           &x_scratch,
                           &fx_scratch,
                           n,
                           opt.newton.preconditioner_lag,
                           0};
    ierr = SNESMonitorSet(snes, snes_monitor, &mon_ctx, nullptr);
    if (auto r = check(ierr, "SNESMonitorSet"); !r.ok()) goto fail;

    // SNESSetFromOptions() is deliberately never called -- see this
    // file's own header comment and fuse/petsc/solve.hpp's.
    ierr = SNESSolve(snes, nullptr, X);
    // SNESSolve itself does not fail (return nonzero) on nonconvergence --
    // it fails only on a genuine PETSc error (an out-of-memory, a
    // malformed matrix); nonconvergence is reported through
    // SNESGetConvergedReason below, the same split
    // fuse::num::solve()'s own Result<NewtonResult> (never Failure, for a
    // DivergedXxx status) uses.
    if (auto r = check(ierr, "SNESSolve"); !r.ok()) goto fail;

    SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
    ierr = SNESGetConvergedReason(snes, &reason);
    if (auto r = check(ierr, "SNESGetConvergedReason"); !r.ok()) goto fail;
    PetscInt its = 0;
    ierr = SNESGetIterationNumber(snes, &its);
    if (auto r = check(ierr, "SNESGetIterationNumber"); !r.ok()) goto fail;
    PetscInt nfevals = 0;
    ierr = SNESGetNumberFunctionEvals(snes, &nfevals);
    if (auto r = check(ierr, "SNESGetNumberFunctionEvals"); !r.ok()) goto fail;
    PetscReal fnorm = 0.0;
    ierr = SNESGetFunctionNorm(snes, &fnorm);
    if (auto r = check(ierr, "SNESGetFunctionNorm"); !r.ok()) goto fail;

    result.status = fn_ctx.failed ? fuse::num::NewtonStatus::DivergedFunctionError
                                  : map_reason(reason);
    result.iterations = static_cast<int>(its);
    result.function_evaluations = static_cast<std::size_t>(nfevals);
    result.initial_residual_norm = result.trace.empty() ? 0.0 : result.trace.front().norm_f;
    result.final_residual_norm = fnorm;

    Vec Xsol = nullptr;
    ierr = SNESGetSolution(snes, &Xsol);
    if (auto r = check(ierr, "SNESGetSolution"); !r.ok()) goto fail;
    const PetscScalar* xarr = nullptr;
    ierr = VecGetArrayRead(Xsol, &xarr);
    if (auto r = check(ierr, "VecGetArrayRead(Xsol)"); !r.ok()) goto fail;
    std::copy(xarr, xarr + n, x.data());
    ierr = VecRestoreArrayRead(Xsol, &xarr);
    if (auto r = check(ierr, "VecRestoreArrayRead(Xsol)"); !r.ok()) goto fail;
  }

  SNESDestroy(&snes);
  VecDestroy(&X);
  VecDestroy(&F);
  return result;

fail:
  SNESDestroy(&snes);
  VecDestroy(&X);
  VecDestroy(&F);
  return Failure{"E-PETSC-CALL", "fuse::petsc::solve failed, see PETSc error code above"};
}

// ===========================================================================
// Registering this substrate.
// ===========================================================================

void install() {
  fuse::num::register_newton_substrate(
      fuse::num::NewtonSubstrate::Petsc,
      [](const fuse::num::ResidualFn& f, fuse::linalg::DistVector& x,
         const fuse::num::PreconditionerApplyFn& precond_apply,
         const fuse::num::RefreshPreconditionerFn& refresh_precond,
         const fuse::num::NewtonKrylovOptions& opt, int comm_f) {
        // `comm_f` is the caller's own region
        // communicator (`RegionRuntime::newton_comm_f`, runner_impl.hpp
        // -- the SAME split `R.newton_reducer` reduces over) when the
        // registry's own call passed one; `0` (a single-rank region, or
        // no communicator configured at all -- `NewtonSolveFn`'s own
        // header comment, newton_krylov.hpp) falls back to
        // `comm_self_f()`, the single-rank behaviour of
        // this entry point. Never a SILENT
        // substitute for a real communicator a multi-rank region
        // actually has: `resolve_regions()` is the ONLY writer of
        // `newton_comm_f`, and it writes a real split's own handle on
        // every rank for which `R.newton_reducer` is non-null -- the
        // only shape in which this lambda is ever called with `comm_f
        // == 0` for a GENUINELY multi-rank region is a rank this
        // region's own early return (`run_region_newton()`/
        // `run_region_newton_window()`) already kept from reaching here
        // at all.
        SnesOptions snes_opt;
        snes_opt.newton = opt;
        snes_opt.comm_f = comm_f != 0 ? comm_f : comm_self_f();
        return solve(f, x, precond_apply, refresh_precond, snes_opt);
      });
}

}  // namespace fuse::petsc
