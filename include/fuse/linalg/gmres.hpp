#pragma once

/// Right-preconditioned GMRES(m): modified Gram-Schmidt Arnoldi, Givens
/// rotations applied to the Hessenberg as each column is produced, restart
/// every `GmresOptions::restart` steps -- the structure
/// unsupported/Eigen/src/IterativeSolvers/GMRES.h uses (Saad and Schultz,
/// "GMRES: A Generalized Minimal Residual Algorithm for Solving
/// Nonsymmetric Linear Systems", SIAM J. Sci. Stat. Comput. 7, 1986), so
/// that a one-rank, identity-preconditioned comparison against it is
/// ITERATE-FOR-ITERATE. Eigen's own implementation orthogonalises with
/// Householder reflections and preconditions on the LEFT (`v <- M^-1 A v`,
/// mixing `M^-1` into the Krylov basis itself); this one orthogonalises
/// with modified Gram-Schmidt and preconditions on the RIGHT (`w <- A M^-1
/// v`, so the Arnoldi recurrence stays over `A` and the basis vectors
/// stay the TRUE residual's Krylov subspace, which is what the "recompute
/// before reporting convergence" discipline below needs). The two
/// orthogonalisation methods compute the same Krylov subspace in exact
/// arithmetic, so the comparison test measures agreement to a stated
/// floating-point bound, not bit-identity.
///
/// WHY THE STOPPING TEST RECOMPUTES RATHER THAN TRUSTS THE ROTATION
/// ESTIMATE. `|g[k+1]|` (the last entry of the Givens-rotated right-hand
/// side) is the STANDARD GMRES residual-norm estimate -- Eigen's own
/// `tol_error = abs(w(k)) / r0Norm` is the same quantity -- and it tracks
/// the true residual well for a well-conditioned basis. Modified
/// Gram-Schmidt loses orthogonality faster than a Householder basis as a
/// cycle's Arnoldi vectors become ill-conditioned (Greenbaum, Rozloznik
/// and Strakos, "Numerical behaviour of the modified Gram-Schmidt GMRES
/// implementation", BIT Numerical Mathematics 37, 1997), which is exactly
/// when the rotation estimate and the true residual can disagree. So,
/// same as pcg.hpp: a candidate convergence is verified by one fresh
/// `apply` before being reported, `SolverReport::residual_norm` is always
/// that recomputed value, and a restart boundary ALREADY recomputes `r0`
/// fresh unconditionally (never from the rotation estimate), the same
/// Eigen's own restart does (`p0 = rhs - mat * x`).
///
/// NO SILENT CAP. Exhausting `GmresOptions::max_iterations` (the total
/// Arnoldi-step budget across every restart cycle) without a verified
/// convergence returns `E-LINALG-NO-CONVERGENCE`, carrying the iteration
/// count and the last known relative residual. `GmresOptions::restart <
/// 1` refuses `E-LINALG-OPTIONS` at entry rather than livelocking: with
/// `restart == 0` the Arnoldi loop below never executes a single step,
/// so `total_iter` never advances and the outer restart loop spins
/// through `a.apply` (a collective under MPI) with no exit -- the one
/// configuration `max_iterations` cannot cap.

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <fmt/format.h>

#include "fuse/linalg/dist_vector.hpp"
#include "fuse/linalg/operator.hpp"
#include "fuse/linalg/preconditioner.hpp"

namespace fuse::linalg {

struct GmresOptions {
  /// The stopping test is `||r||_2 <= rel_tol * ||b||_2`, `r` always a
  /// freshly recomputed residual, never the Givens-rotation estimate
  /// alone (this file's header comment).
  double rel_tol = 1e-10;
  std::size_t max_iterations = 1000;
  /// Arnoldi steps per cycle before a restart (`GMRES(m)`'s `m`).
  std::size_t restart = 30;
  /// Called after every accepted Arnoldi step (and once per restart
  /// boundary's fresh `r0`) with `(total_iterations_so_far,
  /// relative_residual_norm)` -- same contract as `PcgOptions::monitor`.
  std::function<void(std::size_t, double)> monitor = nullptr;
};

/// Solves `A x = b` by restarted, right-preconditioned GMRES. `x` is both
/// the initial guess on input and the solution on output. Preconditions
/// are the same `DistVector` caller-mistake checks `axpy`/`dot` already
/// assert on; nothing further is asserted here.
template <class Op, class Precond>
  requires LinearOperator<Op> && Preconditioner<Precond>
[[nodiscard]] Result<SolverReport> gmres(const Op& a, const Precond& m, DistVector& x,
                                          const DistVector& b, const GmresOptions& opt = {}) {
  if (opt.restart < 1) {
    return Failure{"E-LINALG-OPTIONS",
                  fmt::format("gmres: GmresOptions::restart must be >= 1, got {} (restart == 0 "
                             "never executes an Arnoldi step, so the restart loop never exits)",
                             opt.restart)};
  }
  const double b_norm = b.norm2();
  if (b_norm == 0.0) {
    x.set_zero();
    if (opt.monitor) opt.monitor(0, 0.0);
    return SolverReport{0, 0.0, true};
  }

  const std::size_t mrestart = opt.restart;
  std::size_t total_iter = 0;
  double last_residual_norm = b_norm;

  DistVector ax(b);
  DistVector r0(b);

  while (true) {
    // Restart boundary: r0 is ALWAYS a fresh `b - A x`, never derived from
    // the previous cycle's rotation estimate (this file's header comment).
    if (Result<void> s = a.apply(x, ax); !s.ok()) return s.error();
    r0.copy_from(b);
    r0.axpy(-1.0, ax);
    const double beta = r0.norm2();
    last_residual_norm = beta;
    if (opt.monitor) opt.monitor(total_iter, beta);
    if (beta <= opt.rel_tol * b_norm) {
      return SolverReport{total_iter, beta, true};
    }
    if (total_iter >= opt.max_iterations) {
      return Failure{"E-LINALG-NO-CONVERGENCE",
                    fmt::format("gmres did not converge within {} iterations; last relative "
                               "residual {:.6e} against tolerance {:.6e}",
                               opt.max_iterations, last_residual_norm / b_norm, opt.rel_tol)};
    }

    std::vector<DistVector> v;  // Arnoldi basis, orthonormal, v[0..k]
    v.reserve(mrestart + 1);
    v.emplace_back(r0);
    v.back().scale(1.0 / beta);

    std::vector<std::vector<double>> h(mrestart + 1, std::vector<double>(mrestart, 0.0));
    std::vector<double> cs(mrestart, 0.0), sn(mrestart, 0.0);
    std::vector<double> g(mrestart + 1, 0.0);
    g[0] = beta;

    DistVector z(b);
    DistVector w(b);
    bool restart_cycle = false;

    for (std::size_t k = 0; k < mrestart && total_iter < opt.max_iterations; ++k) {
      ++total_iter;

      if (Result<void> s = m.apply(v[k], z); !s.ok()) return s.error();  // z <- M^-1 v_k
      if (Result<void> s = a.apply(z, w); !s.ok()) return s.error();     // w <- A z

      // Modified Gram-Schmidt against every previous basis vector.
      for (std::size_t i = 0; i <= k; ++i) {
        h[i][k] = w.dot(v[i]);
        w.axpy(-h[i][k], v[i]);
      }
      const double h_subdiag = w.norm2();
      h[k + 1][k] = h_subdiag;
      const bool breakdown = h_subdiag <= 1e-14 * beta;
      if (!breakdown) {
        v.emplace_back(w);
        v.back().scale(1.0 / h_subdiag);
      }

      // Apply the k EXISTING Givens rotations to the new column.
      for (std::size_t i = 0; i < k; ++i) {
        const double h_ik = h[i][k];
        const double h_i1k = h[i + 1][k];
        h[i][k] = cs[i] * h_ik + sn[i] * h_i1k;
        h[i + 1][k] = -sn[i] * h_ik + cs[i] * h_i1k;
      }

      // The new rotation that eliminates h[k+1][k].
      const double h_kk = h[k][k];
      const double h_k1k = h[k + 1][k];
      const double denom = std::sqrt(h_kk * h_kk + h_k1k * h_k1k);
      const double c_k = denom == 0.0 ? 1.0 : h_kk / denom;
      const double s_k = denom == 0.0 ? 0.0 : h_k1k / denom;
      cs[k] = c_k;
      sn[k] = s_k;
      h[k][k] = denom;
      h[k + 1][k] = 0.0;

      const double g_k = g[k];
      g[k] = c_k * g_k + s_k * g[k + 1];
      g[k + 1] = -s_k * g_k + c_k * g[k + 1];

      const double residual_estimate = std::fabs(g[k + 1]);
      last_residual_norm = residual_estimate;
      const bool at_restart_boundary = (k + 1 == mrestart);
      const bool candidate = residual_estimate <= opt.rel_tol * b_norm || breakdown ||
                             at_restart_boundary;
      if (!candidate) {
        if (opt.monitor) opt.monitor(total_iter, residual_estimate);
        continue;
      }

      // Solve the (k+1) x (k+1) upper-triangular system H y = g by back
      // substitution (small and dense: no third-party solver needed).
      std::vector<double> y(k + 1, 0.0);
      for (std::size_t ii = k + 1; ii-- > 0;) {
        double s_val = g[ii];
        for (std::size_t j = ii + 1; j <= k; ++j) s_val -= h[ii][j] * y[j];
        y[ii] = s_val / h[ii][ii];
      }

      DistVector correction(b);
      correction.set_zero();
      for (std::size_t j = 0; j <= k; ++j) correction.axpy(y[j], v[j]);
      DistVector dx(b);
      if (Result<void> s = m.apply(correction, dx); !s.ok()) return s.error();

      DistVector x_trial(x);
      x_trial.axpy(1.0, dx);

      if (Result<void> s = a.apply(x_trial, ax); !s.ok()) return s.error();
      DistVector r_true(b);
      r_true.copy_from(b);
      r_true.axpy(-1.0, ax);
      const double true_norm = r_true.norm2();

      if (true_norm <= opt.rel_tol * b_norm) {
        x.copy_from(x_trial);
        if (opt.monitor) opt.monitor(total_iter, true_norm);
        return SolverReport{total_iter, true_norm, true};
      }
      if (opt.monitor) opt.monitor(total_iter, residual_estimate);

      if (at_restart_boundary || breakdown) {
        // Commit the (unconverged but genuine) progress and restart the
        // cycle from the corrected residual.
        x.copy_from(x_trial);
        restart_cycle = true;
        break;
      }
      // Otherwise the rotation estimate under- or over-reported the true
      // residual without the cycle being exhausted: keep growing this
      // cycle's Krylov subspace rather than committing an unverified step.
    }
    (void)restart_cycle;

    if (total_iter >= opt.max_iterations) {
      return Failure{"E-LINALG-NO-CONVERGENCE",
                    fmt::format("gmres did not converge within {} iterations; last relative "
                               "residual {:.6e} against tolerance {:.6e}",
                               opt.max_iterations, last_residual_norm / b_norm, opt.rel_tol)};
    }
  }
}

}  // namespace fuse::linalg
