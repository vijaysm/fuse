#pragma once

/// Preconditioned conjugate gradient, right-preconditioned-by-construction
/// since the Hestenes-Stiefel recurrence below precondition a SELF-ADJOINT
/// operator the same way on either side: `alpha = (r, z) / (p, A p)`,
/// `beta = (r_new, z_new) / (r, z)`, `p <- z + beta p` -- the exact
/// recurrence Eigen's `ConjugateGradient` uses
/// (Eigen/src/IterativeLinearSolvers/ConjugateGradient.h), which is what
/// makes a one-rank comparison against it an ITERATE-FOR-ITERATE one and
/// not merely "converges to the same answer eventually".
///
/// WHY THE STOPPING TEST RECOMPUTES RATHER THAN TRUSTS THE RECURRENCE.
/// `r` is updated by `r <- r - alpha * (A p)` every iteration -- one
/// axpy, no matrix-vector product of its own. For a well-conditioned
/// operator this tracks `b - A x` to machine precision for as many
/// iterations as the problem needs; for an ill-conditioned one, rounding
/// error accumulates in the recurrence faster than in `x` itself, and the
/// two can disagree by orders of magnitude by the time the recurrence
/// reports convergence (Greenbaum, "Behavior of slightly perturbed Lanczos
/// and conjugate-gradient recurrences", Lin. Alg. Appl. 113, 1989) --
/// `test/linalg/pcg_test.cpp`'s own ill-conditioned case measures exactly
/// this gap. So a candidate convergence (the recurrence's `r` meets the
/// tolerance) is never reported directly: `x` is recomputed against a
/// FRESH `apply(x, _)` first, and only the relative norm of THAT
/// recomputed residual decides whether to return converged, and is the
/// value `SolverReport::residual_norm` reports. If the recomputed residual
/// disagrees, the recurrence is resynchronised to it (a residual
/// replacement step, in the sense of van der Vorst and Ye, "Residual
/// replacement strategies for Krylov subspace iterative methods for the
/// convergence of true residuals", Numer. Lin. Alg. Appl. 7, 2000) and the
/// search direction restarts from the corrected residual -- robustness
/// over the one extra `apply` a replacement costs, per the task's own
/// standing rule.
///
/// NO SILENT CAP. Exhausting `PcgOptions::max_iterations` without either
/// the recurrence or a recomputed residual meeting the tolerance returns
/// `E-LINALG-NO-CONVERGENCE`, carrying the iteration count and the last
/// known relative residual, rather than returning a `SolverReport` whose
/// `converged` field a caller might not check.

#include <cstddef>
#include <functional>

#include <fmt/format.h>

#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/preconditioner.hpp"

namespace fuse::linalg {

struct PcgOptions {
  /// The stopping test is `||r||_2 <= rel_tol * ||b||_2`, `r` always the
  /// residual recomputed fresh at the point of deciding, never the
  /// recurrence's own running value (see this file's header comment).
  double rel_tol = 1e-10;
  std::size_t max_iterations = 1000;
  /// Called after every accepted iteration (including iteration 0, the
  /// initial residual, before any step is taken) with `(iteration,
  /// relative_residual_norm)`. A caller that wants a convergence history
  /// -- a log, or a test comparing iterate-for-iterate against a
  /// reference solver -- sets this; a solve that does not touches it at
  /// all. The SAME values a caller observes through this callback are
  /// what `SolverReport::residual_norm` reports at the end.
  std::function<void(std::size_t, double)> monitor = nullptr;
};

/// Solves `A x = b` by preconditioned conjugate gradient, `A` self-adjoint
/// positive definite. `x` is both the initial guess on input and the
/// solution on output; `b`, the preconditioner `m`, and `x` must all be
/// built against the same distribution (the same owned size), a caller
/// mistake `DistVector`'s own `axpy`/`dot` already assert on, so nothing
/// further is asserted here beyond what those calls assert on their own.
template <class Op, class Precond>
  requires LinearOperator<Op> && Preconditioner<Precond>
[[nodiscard]] Result<SolverReport> pcg(const Op& a, const Precond& m, DistVector& x,
                                        const DistVector& b, const PcgOptions& opt = {}) {
  const double b_norm = b.norm2();
  if (b_norm == 0.0) {
    // No right-hand side: the only solution a residual-based stopping test
    // can certify is x = 0, with nothing to divide a relative residual by.
    x.set_zero();
    if (opt.monitor) opt.monitor(0, 0.0);
    return SolverReport{0, 0.0, true};
  }

  DistVector ax(b);  // scratch for A*x, same distribution as b (copy ctor)
  if (Result<void> s = a.apply(x, ax); !s.ok()) return s.error();
  DistVector r(b);
  r.copy_from(b);
  r.axpy(-1.0, ax);  // r <- b - A x0

  double residual_norm = r.norm2();
  if (opt.monitor) opt.monitor(0, residual_norm);
  if (residual_norm <= opt.rel_tol * b_norm) {
    return SolverReport{0, residual_norm, true};
  }

  DistVector z(b);
  if (Result<void> s = m.apply(r, z); !s.ok()) return s.error();
  DistVector p(z);
  DistVector ap(b);

  double rz_old = r.dot(z);

  for (std::size_t iter = 1; iter <= opt.max_iterations; ++iter) {
    if (Result<void> s = a.apply(p, ap); !s.ok()) return s.error();
    const double pAp = p.dot(ap);
    const double alpha = rz_old / pAp;

    x.axpy(alpha, p);
    r.axpy(-alpha, ap);

    const double recurrence_norm = r.norm2();
    if (recurrence_norm <= opt.rel_tol * b_norm) {
      // Candidate convergence: verify against a freshly recomputed
      // residual before reporting it (this file's header comment).
      if (Result<void> s = a.apply(x, ax); !s.ok()) return s.error();
      DistVector r_true(b);
      r_true.copy_from(b);
      r_true.axpy(-1.0, ax);
      const double true_norm = r_true.norm2();
      if (true_norm <= opt.rel_tol * b_norm) {
        if (opt.monitor) opt.monitor(iter, true_norm);
        return SolverReport{iter, true_norm, true};
      }
      // The recurrence drifted: resynchronise to the true residual and
      // restart the search direction, rather than report a convergence
      // the recomputed residual does not support.
      r.copy_from(r_true);
      residual_norm = true_norm;
      if (opt.monitor) opt.monitor(iter, residual_norm);
      if (Result<void> s = m.apply(r, z); !s.ok()) return s.error();
      p.copy_from(z);
      rz_old = r.dot(z);
      continue;
    }

    residual_norm = recurrence_norm;
    if (opt.monitor) opt.monitor(iter, residual_norm);

    if (Result<void> s = m.apply(r, z); !s.ok()) return s.error();
    const double rz_new = r.dot(z);
    const double beta = rz_new / rz_old;
    p.scale(beta);
    p.axpy(1.0, z);
    rz_old = rz_new;
  }

  return Failure{"E-LINALG-NO-CONVERGENCE",
                fmt::format("pcg did not converge within {} iterations; last relative "
                           "residual {:.6e} against tolerance {:.6e}",
                           opt.max_iterations, residual_norm / b_norm, opt.rel_tol)};
}

}  // namespace fuse::linalg
