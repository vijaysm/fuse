# The derivative gate

`include/fuse/linalg/derivative_gate.hpp` is the one check every later "Newton" or "monolithic"
claim in this project must cite. It verifies three properties, each against a bound that is
*derived* from a measurement rather than chosen, per this project's own standing rule that a
tolerance is never picked by hand without the measurement that justifies it recorded next to it.
The ctest row is `linalg.derivative_gate`
(`test/linalg/derivative_gate_test.cpp`); the tier-1 build-only sibling is
`parallel.derivative_gate_mpi` (`test/parallel/derivative_gate_mpi_test.cpp`).

## 1. The adjoint identity

For an operator with an `apply` and an `apply_transpose` (or, for an operator this project only
ever builds self-adjoint -- `fuse::linalg::SchurSlabSolver` over `A = I - coefficient*L_h`, `L_h`
symmetric -- the same `apply` used on both sides), `<A x, y>` and `<x, A^T y>` must agree to within

```
bound = c * eps * ||A||_est * ||x|| * ||y||
```

`||A||_est` is estimated by power iteration on `A^T A` (`estimate_operator_norm`): a fixed-seed
random unit vector, `v <- normalize(A^T A v)` repeated (20 iterations by default), with the final
reading `||A v||` the estimate. This needs no assumption about the operator's sign pattern --
unlike a row/col-sum bound, which only bounds `||A||_2` for a non-negative matrix, and the random-
nonsymmetric `DistSparseMatrix` fixture below has negative entries by construction.
`NormEstimateMatchesClosedFormForASmallSymmetricLaplacian` validates the estimator itself: for a
small symmetric periodic Laplacian, the estimate is checked against the largest `|eigenvalue|` from
a dense eigensolve (`Eigen::SelfAdjointEigenSolver`), not merely asserted plausible.

`c` (`kAdjointGateConstant` in the header) is the one free constant. **Measured** (Debug
build, AppleClang 21, this project's Eigen linalg backend), over every operator case
`adjoint_cases()` builds -- `fuse::linalg::DistSparseMatrix` at one and two in-process partitions
(a symmetric periodic 16x16 Laplacian and a random nonsymmetric 50x50 matrix, the latter
load-bearing: a transpose bug is invisible on a symmetric operator), `fuse::WeightsOperator` over
three hand-built matrices plus a real consistent-interpolation `H` (`generated:line` meshes) plus a
real supermesh map plus a real moving-least-squares map, their composition
(`fuse::ComposedOperator`), and `fuse::linalg::SchurSlabSolver` at one rank and at two in-process
partitions -- the worst observed ratio `defect / (eps * ||A||_est * ||x|| * ||y||)` was **0.235**
(the one-partition periodic 16x16 Laplacian, the operator with the most floating-point terms per
dot product among the cases exercised). `kAdjointGateConstant = 50` carries roughly 200x headroom
over that measurement. `AdjointIdentityHoldsForEveryOperator` re-measures and prints this ratio
every time it runs (`stderr`, `[derivative-gate] max observed adjoint ratio = ...`) and asserts it
stays below the constant, so a future operator case that erodes the headroom is caught rather than
silently passing on borrowed margin.

`ADeliberatelyWrongAdjointIsDetected` is the control: the random-nonsymmetric case with its own
`apply_transpose` replaced by its own (forward) `apply` -- the exact mutation
`build-evidence/mutations/linalg.toml`'s `derivative-gate-adjoint-transpose-swapped-for-apply` row
describes -- produces a defect more than 100x its own bound, confirming the check can fail and is
not vacuously true for every operator shape.

"Nodal weights" (one of the four backend operator families named for this check): `H`
(`build_consistent_interp_weights`, `consistent_interp.hpp`) interpolates a field whose DOFs are the
source mesh's *vertices* -- nodal DOFs, by that header's own description -- so the "consistent"
case above is also the "nodal weights" case. `nodal_remap_distributed.hpp`'s own
`apply_distributed_H`/`apply_distributed_HT` are a different, MPI-distributed code path that never
produces a `RemapWeights` or a `fuse::LinearOperator`; they are not separately adapted to this
gate's `apply`/`apply_transpose` shape (a deliberate deviation).

## 2. Finite-difference derivative convergence

For a residual `F` and a claimed Jacobian action `J v`, `taylor_remainder_order` sweeps
`h = h0 * 10^-k` (`k = 0..ndecades`, largest `h` first), measures the Taylor remainder

```
remainder(h) = || F(x + h v) - F(x) - h * J(x) v ||_2
```

at each, and fits the log-log slope ("order") over the points *before* the point where the
remainder stops dropping by at least a factor of 2 from the previous point (`plateau_drop = 0.5`,
chosen because a genuine order-2 remainder drops by about 100x per decade and a genuine order-1 one
by about 10x -- a mere 2x drop is never mistaken for either, so the plateau detector only ever
triggers on the genuine round-off floor, where the remainder has stopped decreasing because
`F(x+hv) - F(x)` has lost precision to cancellation, not because the Taylor expansion says anything
new). Fewer than two points before the plateau (including the plateau starting at the very first
step) means there is nothing to fit at all: `at_floor_everywhere` is set, `order` is `NaN`, and
`floor` is the mean of every measured remainder.

Three cases, all residuals built directly from this test's own fixtures rather than through
`examples/dn_heat` or the Runner-level residual:

- **A hand-written cubic map**, `F(x)_i = x_i^3`, `J(x) v_i = 3 x_i^2 v_i` (exact) --
  `CubicMapWithAnalyticJacobianMeasuresOrderTwo` measures order in `(1.8, 2.3)`. The same map with a
  deliberately wrong coefficient (`2 x_i^2` instead of `3 x_i^2`) --
  `WrongJacobianMeasuresOrderOneNotTwo` -- measures order in `(0.7, 1.3)`: the test's own test,
  proving a wrong Jacobian is caught rather than also reading as order 2.
- **A Brusselator species residual** at a frozen `v`, `r(u) = u - u_old - dt*(D*Lap*u + u^2 v - A
  u)` (`D = 10`, `A = 3.4`, `brusselator::PhysicsConfig`'s own defaults), diffusion from
  `brusselator::periodic_laplacian` and the reaction Jacobian `d/du (u^2 v - A u) = 2 u v - A`
  written by hand -- `BrusselatorSpeciesResidualMeasuresOrderTwo` measures order in `(1.8, 2.3)`.
- **A backward-Euler-shaped AFFINE residual**, `r(u) = (I - dt*D*Lap) u - rhs` (the "dn_heat-shaped"
  case, built directly from the same `periodic_laplacian` rather than
  from the `dn_heat` example itself) -- exactly linear in `u`, so its exact Jacobian's remainder is
  mathematically zero at every `h`. `BackwardEulerAffineResidualIsAtTheFloorForEveryH` asserts
  `at_floor_everywhere` and every measured remainder below `1e-8`.

## 3. Assembled-vs-matrix-free agreement

`matrix_free_agreement` applies two independent implementations of the same operator to
`n_vectors` fixed-seed random vectors and requires `||a(x) - b(x)||_2 <= c * eps * ||A||_est *
||x||_2` for every one, reporting the worst case. `kMatrixFreeAgreementConstant = 20`: **measured**: the Brusselator diffusion operator at 16x16 (`DistSparseMatrix::apply`, assembled from
the same triplets `species_core.hpp`'s `KrylovDiffusionSolver::factor` builds, vs. a hand-written
periodic-stencil kernel implementing the identical coefficients directly on arrays) reached a ratio
of about 0.36 at `c = 1`, so `c = 20` carries about 55x headroom.

Two cases: `AssembledDiffusionMatrixAgreesWithEigenProductOnOneRank` (the `DistSparseMatrix` against
a plain `Eigen::SparseMatrix` product, one rank) and
`AssembledDiffusionMatrixAgreesWithHandWrittenStencilApply` (against the hand-written stencil
kernel). **Deviation from the originating task's wording**: that task describes the matrix-free
reference as "the Kokkos stencil kernel", but `examples/brusselator/brusselator_physics.hpp`'s only
Kokkos kernel (`reaction_kokkos_span`) computes the pointwise *reaction* term, `u^2 v - A u`, not
diffusion -- there is no existing Kokkos diffusion stencil in the codebase for this check to call.
The "matrix-free kernel" compared against is therefore a plain hand-written function
(`periodic_laplacian_stencil_apply`, `derivative_gate_test.cpp`) implementing the same stencil
arithmetic `brusselator::periodic_laplacian`/`KrylovDiffusionSolver::factor` both already use, not a
Kokkos kernel.

## What "verified" means for a derivative-bearing operator

Per [](testing.md#the-five-states): a Newton step, a monolithic Jacobian block, or any other
operator whose correctness depends on its own adjoint or its own linearisation is **Verified**
only once it has a `linalg.derivative_gate`-style case citing it -- an operator with an `apply` and
no corresponding case in `adjoint_cases()` is **Implemented, untested** for this specific property,
however well-tested its forward `apply` alone is.

## Mutation coverage

`build-evidence/mutations/linalg.toml` carries three rows for this gate, each HAND-PROVEN against a
byte-identical restored copy of the mutated file (apply the change, rebuild
`fuse_linalg_derivative_gate_test`, run the whole binary unfiltered, record which cases fail,
restore, `cmp`-verify):

- **`derivative-gate-adjoint-transpose-swapped-for-apply`**: `adjoint_gate`'s own
  `apply_transpose` call replaced by `apply` -- killed by the random-nonsymmetric `DistSparseMatrix`
  case (a symmetric operator cannot distinguish this mutation at all, which is why that case is
  load-bearing for this row specifically, not only for the adjoint identity in general).
- **`derivative-gate-plateau-detection-removed`**: `taylor_remainder_order`'s plateau scan
  short-circuited so it never fires, fitting the log-log slope through the round-off floor -- killed
  by the two order-2 cases (`CubicMapWithAnalyticJacobianMeasuresOrderTwo`,
  `BrusselatorSpeciesResidualMeasuresOrderTwo`), whose `h` sweep reaches the floor within its 9
  decades.
- **`derivative-gate-wrong-jacobian-expectation-inverted`**: the *test's* own expectation in
  `WrongJacobianMeasuresOrderOneNotTwo` inverted to the bounds a *correct* Jacobian would need
  (`(1.8, 2.3)` instead of `(0.7, 1.3)`) -- fails, because the wrong Jacobian's own measured order
  (~1) does not satisfy them, confirming the original test's assertion direction is load-bearing and
  not a vacuous pass either way.
