# Native Krylov solvers: PCG and GMRES(m)

`fuse::linalg::pcg` and `fuse::linalg::gmres` (`include/fuse/linalg/{pcg,gmres}.hpp`) are
header-only template functions built on `DistVector`/`DistSparseMatrix`
([](linalg-distributed.md)): a native, dependency-free preconditioned conjugate gradient and a
restarted, right-preconditioned GMRES(m), with no third-party header anywhere under
`include/fuse/linalg/` ([](layering.md) -- Eigen appears only in the tier-0 tests that check
agreement against it, never in a public header).

## What the solvers are built on

Two structural (duck-typed) concepts, `LinearOperator` and `Preconditioner`
(`include/fuse/linalg/operator.hpp`, `preconditioner.hpp`), both the same shape: `Result<void>
apply(DistVector& x, DistVector& y) const` computing `y <- A x` or `y <- M^-1 x`. `DistSparseMatrix`
satisfies `LinearOperator` without naming it; `MatrixFreeOperator` wraps a caller-supplied callable
with the same shape for a Jacobian-free Newton-Krylov step (`y <- J(u) v` from a residual function's
action, never an assembled matrix). `IdentityPreconditioner` and `JacobiPreconditioner` (built from
`DistSparseMatrix::diagonal()`, which needs no communication) satisfy `Preconditioner`. `pcg`/`gmres`
take both through independent template parameters, so any type with the right shape is accepted,
with no base class and no virtual dispatch.

Every dot and norm the solvers compute goes through `DistVector::dot`/`norm2`, which already reduce
across whatever `Reducer` the vector was built with -- the solvers themselves never issue a
collective of their own, so they are parallel by construction. `SolverReport {iterations,
residual_norm, converged}` is returned alongside the solution.

**The claim above is about the CONVERGED path only.** `pcg`/`gmres`
return on the first `!apply(...).ok()` with no cross-rank reconciliation of that decision --
`LinearOperator`'s/`Preconditioner`'s own header comments now state the precondition this
relies on: every `apply` the solver is handed must fail IDENTICALLY on every rank of a call, or
the failing rank skips every later `dot`/`norm2` collective its peers still enter --
[](rank-agreement.md)'s own hang class. Every operator/preconditioner this library builds
itself (`DistSparseMatrix`, `IdentityPreconditioner`, `JacobiPreconditioner`) only fails on an
MPI error, in practice process-fatal, so this is latent, not exercised, today; a caller-supplied
`MatrixFreeOperator` wrapping a residual function that CAN refuse data-dependently on one rank
alone is the one path this library does not itself close.

## PCG: the Hestenes-Stiefel recurrence, verified before being reported

`pcg`'s recurrence is exactly Eigen's `ConjugateGradient`
(`Eigen/src/IterativeLinearSolvers/ConjugateGradient.h`): `alpha = (r, z) / (p, A p)`, `x <- x +
alpha p`, `r <- r - alpha (A p)`, `z <- M^-1 r`, `beta = (r, z)_new / (r, z)_old`, `p <- z + beta p`.
A one-rank, identity- or Jacobi-preconditioned comparison against Eigen's own algorithm is therefore
iterate-for-iterate, not merely "converges to the same answer eventually"
(`test/linalg/pcg_test.cpp`).

The one deliberate departure from a textbook PCG loop: **a candidate convergence is never reported
directly from the recurrence**. `r` is updated by one axpy per iteration, with no matrix-vector
product of its own; for an ill-conditioned operator this can drift from the true residual by orders
of magnitude by the time the recurrence reports convergence (Greenbaum, "Behavior of slightly
perturbed Lanczos and conjugate-gradient recurrences", Linear Algebra and its Applications 113,
1989). So when `||r||_2 <= rel_tol ||b||_2` on the recurrence's own `r`, `pcg` recomputes `b - A x`
fresh from a new `apply` before deciding anything: if that recomputed residual also satisfies the
tolerance, `pcg` returns `converged = true` with `SolverReport::residual_norm` set to the
*recomputed* value, never the recurrence's. If it does not, the recurrence is resynchronised to the
recomputed residual and the search direction restarts from it (a residual-replacement step, in the
sense of van der Vorst and Ye, "Residual replacement strategies for Krylov subspace iterative methods
for the convergence of true residuals", Numerical Linear Algebra with Applications 7, 2000) -- one
extra `apply` per false candidacy, traded for never reporting a convergence the true residual does
not support, per the project's standing rule that robustness outweighs efficiency.
`test/linalg/pcg_test.cpp`'s `PcgRecomputesTrueResidual` case constructs a diagonal SPD matrix with
condition number `1e12` at `rel_tol = 1e-13`, a regime where the gap between the recurrence and the
recomputed residual is large enough to measure directly, and asserts the reported convergence
survives an independent recompute done outside the solver entirely.

No silent cap: exhausting `PcgOptions::max_iterations` without either the recurrence or a recomputed
residual meeting the tolerance returns `E-LINALG-NO-CONVERGENCE`, carrying the iteration count and
the last known relative residual.

## GMRES(m): modified Gram-Schmidt Arnoldi, right-preconditioned

`gmres` restarts every `GmresOptions::restart` steps (`GMRES(m)`'s `m`), orthogonalising the Krylov
basis with modified Gram-Schmidt and applying Givens rotations to the Hessenberg as each column is
produced -- the same overall structure
`unsupported/Eigen/src/IterativeSolvers/GMRES.h` uses (Saad and Schultz, "GMRES: A Generalized
Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems", SIAM Journal on Scientific and
Statistical Computing 7, 1986), with two differences from Eigen's own implementation, both
deliberate:

* **Orthogonalisation.** Eigen's unsupported GMRES orthogonalises with Householder reflections;
  this one uses modified Gram-Schmidt. The two compute the same Krylov subspace in exact arithmetic,
  so a one-rank comparison against Eigen (`test/linalg/gmres_test.cpp`) agrees to a measured
  floating-point bound, not bit identity.
* **Preconditioning side.** Eigen's own `gmres()` preconditions on the LEFT (`v <- M^-1 A v`, mixing
  `M^-1` into the Krylov basis itself). This implementation preconditions on the RIGHT (`w <- A M^-1
  v`), so the Arnoldi recurrence stays over `A` and the basis vectors stay in the TRUE residual's
  Krylov subspace -- which the "recompute before reporting convergence" discipline below needs.
  Because the two conventions solve a genuinely different recurrence whenever `M != I`, the
  iterate-for-iterate comparison against Eigen in `test/linalg/gmres_test.cpp` uses the identity
  preconditioner on both sides; a Jacobi-preconditioned case is checked for convergence and solution
  accuracy against `Eigen::SparseLU`, not iterate-for-iterate.

Same discipline as PCG: `|g[k+1]|` (the last entry of the Givens-rotated right-hand side) is the
standard GMRES residual-norm estimate, and modified Gram-Schmidt loses orthogonality faster than a
Householder basis as a cycle's Arnoldi vectors become ill-conditioned (Greenbaum, Rozloznik and
Strakos, "Numerical behaviour of the modified Gram-Schmidt GMRES implementation", BIT Numerical
Mathematics 37, 1997) -- exactly when the rotation estimate and the true residual can disagree. A
candidate convergence is verified by one fresh `apply` before being reported, and a restart boundary
*already* recomputes `r0` fresh unconditionally (never from the rotation estimate), the same Eigen's
own restart does. `SolverReport::residual_norm` is always the recomputed value.

No silent cap, same as PCG: exhausting `GmresOptions::max_iterations` (the total Arnoldi-step budget
across every restart cycle) without a verified convergence returns `E-LINALG-NO-CONVERGENCE`.

## What is and is not verified

Measured at tier 0, one rank, against Eigen (`test/linalg/{pcg,gmres}_test.cpp`):

* PCG, identity and Jacobi preconditioned, iterate-for-iterate against `Eigen::ConjugateGradient` on
  a periodic Laplacian and a random SPD matrix, and solution agreement against
  `Eigen::SimplicialLDLT`.
* GMRES(30), identity preconditioned, iterate-for-iterate against Eigen's unsupported GMRES on a
  random nonsymmetric, diagonally dominant matrix, and solution agreement against `Eigen::SparseLU`.
  A Jacobi-preconditioned case is checked for convergence and solution accuracy only (the
  preconditioning-side argument above).
* The recompute-before-reporting-convergence discipline, on an ill-conditioned diagonal SPD matrix
  constructed specifically to make the recurrence and a recomputed residual disagree.
* Both refusals: `E-LINALG-NO-CONVERGENCE` on a deliberately tiny iteration cap, and
  `E-LINALG-ZERO-DIAGONAL` (`JacobiPreconditioner::from_matrix` on a matrix with a structurally
  absent diagonal entry).
* `test/parallel/krylov_mpi_test.cpp` (tier 1, `RANKS 2`, built here, never run here -- MPI cannot
  start in this sandbox, [](testing.md)) repeats convergence and an independently recomputed
  residual check for both solvers over a real `MPI_COMM_WORLD`. `parallel.krylov_mpi.2` passes on
  the host (0.31 s).

Not verified here: breakdown (an exactly rank-deficient Krylov subspace) is handled defensively
(`gmres.hpp`'s own comments) but no test constructs a genuine breakdown case, since the random test
matrices this file uses essentially never produce one. A caller relying on breakdown handling for a
structurally singular or near-singular operator should treat that path as unexercised.
