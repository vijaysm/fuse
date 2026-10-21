#pragma once

/// S7.4 (owner decision 2026-10-21, Option C). `NeumannHalfPetsc`: the SAME
/// physics as `NeumannHalfNative` (reaction_diffusion_component.hpp), but
/// the INNER backward-Euler Newton solve runs on real PETSc objects:
///
///   - the residual/update in a real PETSc `Vec`
///     (`VecCreate`+`VecSetSizes(..., PETSC_DETERMINE)`+
///     `VecSetType(VECSTANDARD)`, `src/petsc/solve.cpp`'s own convention --
///     NOT `VecCreateSeq`, so the SAME two calls would already be correct
///     if this component's single-rank region were ever handed a
///     multi-rank communicator; this rank's own portion only, through
///     `VecGetArray`, never a global-index `VecSetValues`);
///   - the Newton Jacobian in a real, assembled PETSc `Mat`
///     (`MatCreateAIJ` with 3 as the per-row nonzero estimate -- this
///     rank's own tridiagonal nonzero COUNT, not a dense over-estimate --
///     assembled by `MatSetValues`/`ADD_VALUES` the same way
///     `src/petsc/token.cpp`'s own `make_assembled_token` already does for
///     its `MatCreateSeqAIJ`; chosen over a `MatShell` because the
///     stencil's sparsity is fixed and known up front, a `KSP`+`PCLU`
///     direct factorization is cheap at this component's own small `m`,
///     and an EXPLICIT matrix is what the implicit-differentiation
///     sensitivity below needs to reuse via one more `KSPSolve` against
///     the SAME factored operator -- a `MatShell` would still need a
///     real, invertible representation for that second solve, so nothing
///     is saved by deferring the assembly);
///   - `KSP` (`KSPPREONLY`+`PCLU`): a DIRECT solve, matching this
///     project's standing rule (robustness over efficiency) and making
///     the native (Thomas algorithm) and PETSc substrates solve the
///     IDENTICAL linear system to the IDENTICAL (round-off-limited)
///     accuracy every Newton step -- the native-vs-PETSc agreement bound
///     (docs/user/petsc-reaction-diffusion.md) is derived from exactly
///     that equivalence, not from two iterative solvers that might
///     disagree by their own, unrelated tolerances.
///
/// The OUTER `LinearizationToken` (`dG/dq_I`, 1x1) reuses
/// `fuse::petsc::make_assembled_token()` directly (`fuse/petsc/token.hpp`)
/// rather than hand-rolling a second `Mat` -- the smallest-footprint reuse
/// of an already-reviewed library primitive for exactly this shape (a
/// real, assembled, single-entry `Mat` with `MatMult`/`MatMultTranspose`,
/// never `MATMFFD`).
///
/// HOST-PENDING (KNOWN_ISSUES.md D32, unchanged by this track): this
/// sandbox cannot complete a bare `MPI_Init()`, and `PetscInitialize()`
/// (as of `src/petsc/init.cpp`, commit 70b3d16) deliberately lets
/// `MPI_Init()` run rather than skip it -- so EVERY path through this
/// file SEGFAULTS the instant `ensure_initialized()` runs in this
/// sandbox, the same already-documented, already-measured constraint
/// every other `petsc.*` row carries, not a new one. This file is
/// therefore BUILT and ctest-REGISTERED here, never RUN here; see
/// design_notes.md Section 1 for the direct measurement this track took
/// of that constraint before writing a line of this file, and
/// KNOWN_ISSUES.md's own D32 addendum for the record.

#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <fuse/petsc/token.hpp>
#include <fuse/runtime/component.hpp>
#include <petsc.h>

#include "reaction_diffusion_physics.hpp"

namespace petsc_reaction_diffusion {

namespace detail {
[[nodiscard]] inline fuse::Result<void> check_petsc(PetscErrorCode ierr, const char* what) {
  if (ierr) {
    return fuse::Failure{"E-PETSC-CALL", std::string(what) + ": PETSc error code " +
                                             std::to_string(ierr)};
  }
  return fuse::Result<void>::success();
}
}  // namespace detail

/// One backward-Euler Newton solve of the Right half's own `m`-point
/// system, entirely on real PETSc objects -- the PETSc-substrate analogue
/// of `right_newton_solve` (reaction_diffusion_physics.hpp), SAME
/// discretization, SAME ghost-point boundary treatment, SAME manufactured
/// forcing, assembled and solved through `Vec`/`Mat`/`KSP` instead of
/// `thomas_solve`. Returns the identical `RightSolveResult` shape so a
/// caller (the component below, the petsc-gated structural test) cannot
/// tell which substrate produced it except by comparing the two.
[[nodiscard]] inline fuse::Result<RightSolveResult> right_newton_solve_petsc(
    const std::vector<double>& u_old, double q_I, double dt, double dx, double t_new,
    const PhysicsParams& p, double tol = 1e-13, int max_iterations = 40) {
  using detail::check_petsc;
  const PetscInt m = static_cast<PetscInt>(u_old.size());
  MPI_Comm comm = PETSC_COMM_SELF;

  Vec u_vec = nullptr, du_vec = nullptr, r_vec = nullptr, sens_rhs = nullptr, sens_sol = nullptr;
  Mat J = nullptr;
  KSP ksp = nullptr;
  PC pc = nullptr;

  auto cleanup = [&]() {
    if (ksp) KSPDestroy(&ksp);
    if (J) MatDestroy(&J);
    if (u_vec) VecDestroy(&u_vec);
    if (du_vec) VecDestroy(&du_vec);
    if (r_vec) VecDestroy(&r_vec);
    if (sens_rhs) VecDestroy(&sens_rhs);
    if (sens_sol) VecDestroy(&sens_sol);
  };

  PetscErrorCode ierr = VecCreate(comm, &u_vec);
  if (auto r = check_petsc(ierr, "VecCreate(u)"); !r.ok()) { cleanup(); return r.error(); }
  ierr = VecSetSizes(u_vec, m, PETSC_DETERMINE);
  if (auto r = check_petsc(ierr, "VecSetSizes(u)"); !r.ok()) { cleanup(); return r.error(); }
  ierr = VecSetType(u_vec, VECSTANDARD);
  if (auto r = check_petsc(ierr, "VecSetType(u)"); !r.ok()) { cleanup(); return r.error(); }
  VecDuplicate(u_vec, &du_vec);
  VecDuplicate(u_vec, &r_vec);
  VecDuplicate(u_vec, &sens_rhs);
  VecDuplicate(u_vec, &sens_sol);

  ierr = MatCreateAIJ(comm, m, m, PETSC_DETERMINE, PETSC_DETERMINE, 3, nullptr, 0, nullptr, &J);
  if (auto r = check_petsc(ierr, "MatCreateAIJ"); !r.ok()) { cleanup(); return r.error(); }

  {
    PetscScalar* arr = nullptr;
    ierr = VecGetArray(u_vec, &arr);
    if (auto r = check_petsc(ierr, "VecGetArray(u, seed)"); !r.ok()) { cleanup(); return r.error(); }
    for (PetscInt i = 0; i < m; ++i) arr[i] = u_old[static_cast<std::size_t>(i)];
    VecRestoreArray(u_vec, &arr);
  }

  std::vector<double> u(u_old);
  int it = 0;
  for (; it < max_iterations; ++it) {
    const ResidualJacobian rj = right_residual_jacobian(u, u_old, q_I, dt, dx, t_new, p);
    double nrm2 = 0.0;
    for (double rv : rj.R) nrm2 += rv * rv;
    if (std::sqrt(nrm2) < tol) break;

    // Residual into the real PETSc Vec: owned entries only, via
    // VecGetArray -- never a global-index VecSetValues (this component's
    // own header comment; the owner's Section 4 wording, verbatim).
    PetscScalar* rarr = nullptr;
    ierr = VecGetArray(r_vec, &rarr);
    if (auto r = check_petsc(ierr, "VecGetArray(r)"); !r.ok()) { cleanup(); return r.error(); }
    for (PetscInt i = 0; i < m; ++i) rarr[i] = -rj.R[static_cast<std::size_t>(i)];
    VecRestoreArray(r_vec, &rarr);

    // Jacobian into a real, assembled PETSc Mat -- MatSetValues one
    // triplet at a time, ADD_VALUES, the SAME convention
    // src/petsc/token.cpp's own make_assembled_token already uses
    // (owner's own correction to this track's Stage A claim).
    MatZeroEntries(J);
    for (PetscInt i = 0; i < m; ++i) {
      PetscInt cols[3];
      PetscScalar vals[3];
      PetscInt nz = 0;
      if (i > 0) { cols[nz] = i - 1; vals[nz] = rj.J.sub[static_cast<std::size_t>(i - 1)]; ++nz; }
      cols[nz] = i; vals[nz] = rj.J.diag[static_cast<std::size_t>(i)]; ++nz;
      if (i + 1 < m) { cols[nz] = i + 1; vals[nz] = rj.J.sup[static_cast<std::size_t>(i)]; ++nz; }
      ierr = MatSetValues(J, 1, &i, nz, cols, vals, ADD_VALUES);
      if (auto r = check_petsc(ierr, "MatSetValues(J)"); !r.ok()) { cleanup(); return r.error(); }
    }
    MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);

    if (ksp) { KSPDestroy(&ksp); ksp = nullptr; }
    ierr = KSPCreate(comm, &ksp);
    if (auto r = check_petsc(ierr, "KSPCreate"); !r.ok()) { cleanup(); return r.error(); }
    KSPSetOperators(ksp, J, J);
    KSPSetType(ksp, KSPPREONLY);
    KSPGetPC(ksp, &pc);
    PCSetType(pc, PCLU);  // direct factorization: robustness over efficiency
    KSPSetUp(ksp);
    ierr = KSPSolve(ksp, r_vec, du_vec);
    if (auto r = check_petsc(ierr, "KSPSolve"); !r.ok()) { cleanup(); return r.error(); }

    PetscScalar* duarr = nullptr;
    VecGetArray(du_vec, &duarr);
    for (PetscInt i = 0; i < m; ++i) u[static_cast<std::size_t>(i)] += duarr[i];
    VecRestoreArray(du_vec, &duarr);
  }

  // Implicit-differentiation sensitivity: ONE more KSPSolve against the
  // SAME (converged, still-factored) J -- the identical trick
  // reaction_diffusion_physics.hpp's own thomas_solve-based
  // right_newton_solve uses, now through PETSc's own KSP.
  const ResidualJacobian rj = right_residual_jacobian(u, u_old, q_I, dt, dx, t_new, p);
  PetscScalar* brr = nullptr;
  VecGetArray(sens_rhs, &brr);
  for (PetscInt i = 0; i < m; ++i) brr[i] = (i == 0) ? (1.0 / dx) : 0.0;
  VecRestoreArray(sens_rhs, &brr);
  ierr = KSPSolve(ksp, sens_rhs, sens_sol);
  if (auto r = check_petsc(ierr, "KSPSolve(sensitivity)"); !r.ok()) { cleanup(); return r.error(); }
  PetscScalar sens0 = 0.0;
  {
    PetscScalar* sarr = nullptr;
    VecGetArray(sens_sol, &sarr);
    sens0 = sarr[0];
    VecRestoreArray(sens_sol, &sarr);
  }

  RightSolveResult out;
  out.u = u;
  out.iterations = it + 1;
  out.u_interface = u[0] + q_I * dx / p.D;
  out.dG_dq_I = sens0 + dx / p.D;
  cleanup();
  return out;
}

/// `fuse::ImplicitComponent` wiring: byte-for-byte the SAME
/// register_fields/advance/save_state/restore_state/evaluate_residual
/// shape as `NeumannHalfNative`, only the inner solve differs (and the
/// outer `LinearizationToken`, which wraps `dG/dq_I` in a REAL PETSc Mat
/// instead of a plain double).
class NeumannHalfPetsc final : public fuse::Component, public fuse::ImplicitComponent {
 public:
  NeumannHalfPetsc(std::string name, int m, PhysicsParams params)
      : name_(std::move(name)),
        m_(m),
        params_(params),
        dx_(params.L_right / static_cast<double>(m + 1)),
        q_in_(1, 0.0),
        u_out_(1, 0.0),
        profile_(static_cast<std::size_t>(m), 0.0),
        saved_profile_(static_cast<std::size_t>(m), 0.0) {
    fuse::petsc::ensure_initialized(fuse::petsc::comm_self_f());
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return fuse::Failure{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const fuse::FieldId f : g.component(*cid).fields) {
      const bool provides = g.field(f).intent == fuse::Intent::Provides;
      double* base = provides ? u_out_.data() : q_in_.data();
      auto r = reg.bind(f, fuse::FieldRef{base, fuse::Layout{.stride = 1, .count = 1},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }

  void seed_initial_condition() {
    for (int i = 0; i < m_; ++i) {
      profile_[static_cast<std::size_t>(i)] = manufactured_u(
          params_.L_left + static_cast<double>(i + 1) * dx_, 0.0, params_);
    }
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    auto r = right_newton_solve_petsc(profile_, q_in_[0], dt, dx_, t_new, params_);
    if (!r.ok()) return r.error();
    profile_ = r.value().u;
    u_out_[0] = r.value().u_interface;
    last_sensitivity_ = r.value().dG_dq_I;
    ++advance_count;
    ++signature_;
    return fuse::Result<void>::success();
  }

  fuse::Result<void> save_state() override {
    saved_profile_ = profile_;
    ++save_count;
    return fuse::Result<void>::success();
  }
  fuse::Result<void> restore_state() override {
    profile_ = saved_profile_;
    ++restore_count;
    return fuse::Result<void>::success();
  }

  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/true, fuse::DeterminismLevel::Bitwise,
                                   /*hidden_state=*/true) |
           fuse::kCapabilityResidual | fuse::kCapabilityLinearization;
  }

  fuse::Result<void> evaluate_residual(const fuse::StepContext& ctx,
                                       std::span<const double> interface_in,
                                       std::span<double> residual_out) override {
    if (interface_in.size() != 1 || residual_out.size() != 1) {
      return fuse::Failure{"E-EX-COMP", "NeumannHalfPetsc::evaluate_residual: expected 1 in/out"};
    }
    const double dt = ctx.dt().to_double();
    const double t_new = ctx.t_end.to_double();
    auto r = right_newton_solve_petsc(profile_, interface_in[0], dt, dx_, t_new, params_);
    if (!r.ok()) return r.error();
    residual_out[0] = r.value().u_interface;
    last_sensitivity_ = r.value().dG_dq_I;
    ++signature_;
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::LinearizationToken* linearization() override {
    auto token = fuse::petsc::make_assembled_token(
        1, 1, {fuse::petsc::MatrixTriplet{0, 0, last_sensitivity_}},
        fuse::petsc::comm_self_f());
    if (!token.ok()) {
      token_.reset();
      return nullptr;  // a caller sees no token; evaluate_residual's own error already fired
    }
    token_ = std::move(token.value());
    return token_.get();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }

  [[nodiscard]] const std::vector<double>& profile() const { return profile_; }
  [[nodiscard]] double u_out() const { return u_out_[0]; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string name_;
  int m_;
  PhysicsParams params_;
  double dx_;
  std::vector<double> q_in_;
  std::vector<double> u_out_;
  std::vector<double> profile_;
  std::vector<double> saved_profile_;
  double last_sensitivity_ = 0.0;
  std::uint64_t signature_ = 0;
  std::unique_ptr<fuse::LinearizationToken> token_;
};

}  // namespace petsc_reaction_diffusion
