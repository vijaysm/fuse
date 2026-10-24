# Linear algebra and Newton-Krylov for users

The distributed vectors, sparse matrices, Krylov solvers, derivative checks and Jacobian-free
Newton-Krylov solver in `fuse::linalg` and `fuse::num`. The coupler uses them to solve a Newton
iteration region; you can also call them directly from your own component or driver. This page
is for people who want to do that, or who want to understand what `.newton(cfg)` runs. Each snippet
below is from one single-process program; the output quoted after it is real (exact digits can vary
by platform).

```cpp
#include <fuse/linalg/comm.hpp>             // Reducer, GhostExchanger, serial and MPI factories
#include <fuse/linalg/dist_vector.hpp>      // DistVector
#include <fuse/linalg/dist_sparse_matrix.hpp>
#include <fuse/linalg/operator.hpp>         // LinearOperator, MatrixFreeOperator, SolverReport
#include <fuse/linalg/preconditioner.hpp>   // IdentityPreconditioner, JacobiPreconditioner
#include <fuse/linalg/pcg.hpp>
#include <fuse/linalg/gmres.hpp>
#include <fuse/linalg/derivative_gate.hpp>
#include <fuse/num/newton_krylov.hpp>       // solve(), NewtonKrylovOptions, MffdJacobianAction
#include <fuse/num/line_search.hpp>
```

Link `fuse::linalg` and `fuse::num` (or `fuse::runtime`, which depends on both). Every call that can
fail returns a `fuse::Result`; nothing throws for a numerical condition.

## Communication: `Reducer` and `GhostExchanger`

The containers do not know about MPI. They are given two small objects at construction:

- a `Reducer`, which performs `sum`, `max`, `allgather` and reports `rank()` / `size()`;
- a `GhostExchanger`, which refreshes a vector's ghost entries from their owners.

```cpp
auto reducer   = fuse::linalg::make_serial_reducer();     // one process
auto exchanger = fuse::linalg::make_serial_exchanger();   // no ghosts
```

With MPI enabled (`FUSE_ENABLE_MPI`), the same roles are filled by `make_mpi_reducer(int comm_f)`,
`make_mpi_exchanger(int comm_f, const GhostPattern&, std::size_t owned_n)` and
`make_participant_set_reducer(base_comm_f, participants)`, which splits a sub-communicator for a
subset of ranks and returns it with its reducer. Communicators are passed as Fortran handles
(`MPI_Comm_c2f`), so no public header includes `<mpi.h>`. A `GhostPattern` lists, for each ghost,
its `owner_rank` and the owner's `remote_local_index`. `mpi_is_gpu_aware()` reports whether the
linked MPI can take device pointers. These need an MPI build; the code below is identical on one
rank and on many.

## `DistVector`

A row-distributed vector: owned entries first, ghosts after.

```cpp
const std::size_t n = 8;
auto reducer = make_serial_reducer();            // one process: sums are local
auto exchanger = make_serial_exchanger();        // no ghosts to refresh
DistVector x(n, exchanger, reducer), b(n, exchanger, reducer);
for (std::size_t i = 0; i < n; ++i) b.data()[i] = 1.0;   // owned entries first, then ghosts
x.set_zero();
std::cout << "||b||_2 = " << b.norm2() << "  ||b||_inf = " << b.norm_inf() << "\n";
```

```text
||b||_2 = 2.82843  ||b||_inf = 1
```

| Member | Meaning |
|---|---|
| `DistVector(owned_n, exchanger, reducer)` | the ghost count comes from `exchanger->ghost_size()` |
| `data()`, `owned_size()`, `ghost_size()`, `size()` | raw storage and its partition |
| `update_ghosts()` | refresh ghosts from their owners; returns `Result<void>` |
| `dot(y)`, `norm2()`, `norm_inf()` | reduced across ranks over owned entries |
| `axpy(a, y)`, `scale(a)`, `set_zero()`, `copy_from(y)` | local updates |

## `DistSparseMatrix`

Built once from triplets; each rank passes the rows it owns.

```cpp
// A = tridiag(-1, 2, -1): one rank owns global rows [0, n).
std::vector<Triplet> t;
for (std::int64_t i = 0; i < std::int64_t(n); ++i) {
  t.push_back({i, i, 2.0});
  if (i > 0) t.push_back({i, i - 1, -1.0});
  if (i + 1 < std::int64_t(n)) t.push_back({i, i + 1, -1.0});
}
const RowOwnership owners{{0}};                    // row_lo of every participant
auto a = DistSparseMatrix::from_triplets(/*row_lo=*/0, /*row_hi=*/n, /*global_n=*/n, t,
                                         reducer, owners);
```

`Triplet{row, col, value}` uses global indices; duplicates are summed. `RowOwnership::row_lo` holds
the first row of every participant in ascending order, which is how the matrix derives its ghost
pattern (`ghost_pattern()`) for off-rank columns. A triplet in a row this rank does not own is
`E-LINALG-FOREIGN-ROW`; a column outside `[0, global_n)` is `E-LINALG-COL-RANGE`. `apply(x, y)`
computes y = A x (refreshing `x`'s ghosts first), `diagonal(d)` extracts the diagonal, and
`local_rows()` / `global_rows()` report the shape.

## Operators and preconditioners

The solvers are templates over two concepts, so a matrix, a lambda or your own class all fit:

```cpp
template <class T>
concept LinearOperator = requires(const T& op, DistVector& x, DistVector& y) {
  { op.apply(x, y) } -> std::same_as<Result<void>>;
};
```

`Preconditioner` has the same shape. Ready-made implementations:

| Type | Use |
|---|---|
| `MatrixFreeOperator(std::function<Result<void>(DistVector&, DistVector&)>)` | wrap any action as an operator |
| `IdentityPreconditioner{}` | no preconditioning |
| `JacobiPreconditioner::from_matrix(a)` | inverse diagonal; a zero diagonal entry is `E-LINALG-ZERO-DIAGONAL` |
| `fuse::num::BlockJacobiPreconditioner::from_blocks(blocks, n)` | block-diagonal from `PreconditionerBlock{offset, length, apply}` entries that must tile `[0, n)` exactly (`E-NEWTON-PRECOND-COVERAGE`) |

Every solver returns a `SolverReport{iterations, residual_norm, converged}`.

## Krylov solvers: `pcg` and `gmres`

```cpp
PcgOptions pcg_opt;
pcg_opt.rel_tol = 1e-12;
auto jac = JacobiPreconditioner::from_matrix(a.value());   // E-LINALG-ZERO-DIAGONAL if a_ii == 0
auto rep = pcg(a.value(), jac.value(), x, b, pcg_opt);
std::cout << "pcg: " << rep.value().iterations << " iterations, residual "
          << rep.value().residual_norm << ", converged " << rep.value().converged << "\n";
```

```text
pcg: 4 iterations, residual 0, converged 1
```

```cpp
MatrixFreeOperator op([&](DistVector& v, DistVector& w) { return a.value().apply(v, w); });
GmresOptions g_opt;
g_opt.rel_tol = 1e-12;
g_opt.restart = 30;
g_opt.monitor = [](std::size_t it, double r) { std::printf("  gmres it %zu  |r| = %.3e\n", it, r); };
x.set_zero();
auto grep = gmres(op, IdentityPreconditioner{}, x, b, g_opt);
std::cout << "gmres: " << grep.value().iterations << " iterations, x[0] = " << x.data()[0] << "\n";
```

```text
  gmres it 0  |r| = 2.828e+00
  gmres it 1  |r| = 2.449e+00
  gmres it 2  |r| = 2.000e+00
  gmres it 3  |r| = 1.414e+00
  gmres it 4  |r| = 3.553e-15
gmres: 4 iterations, x[0] = 4
```

| | `pcg` | `gmres` |
|---|---|---|
| Signature | `pcg(op, precond, x, b, PcgOptions = {})` | `gmres(op, precond, x, b, GmresOptions = {})` |
| Options | `rel_tol = 1e-10`, `max_iterations = 1000`, `monitor` | the same plus `restart = 30` |
| Requires | symmetric positive definite operator, symmetric preconditioner | any operator; right-preconditioned (z = M⁻¹v, w = Az) |
| Convergence | relative to ‖b‖; the true residual is recomputed before convergence is accepted | the same; `restart < 1` is `E-LINALG-OPTIONS` |
| Failure | `E-LINALG-NO-CONVERGENCE` after `max_iterations`, with the last relative residual in the message | the same |

A zero right-hand side returns immediately with x = 0 and `converged = true`. `monitor(iteration,
residual_norm)` is called at every iteration when set.

## Schur slab solver

`fuse::linalg::SchurSlabSolver` (`fuse/linalg/schur_slab.hpp`) is an exact distributed direct
solver for one specific operator: `A = I − coefficient · L_h` with `L_h` the periodic five-point
Laplacian on an `nx × ny` cell grid, rows partitioned into contiguous row bands of `j`. Each rank
calls `SchurSlabSolver::factor(nx, ny, j_lo, j_hi, coefficient, RowOwnership, reducer)` once and
`solve(rhs_owned, x_owned)` per right-hand side. It needs the Eigen linear-algebra backend: with
`FUSE_LINALG_BACKEND=builtin`, `factor()` refuses:

```text
E-LINALG-SCHUR-BACKEND: SchurSlabSolver needs the Eigen linear-algebra backend (FUSE_LINALG_BACKEND=eigen); this build selected 'builtin'
```

Other refusals are `E-LINALG-SCHUR-SLAB-FACTORIZE`, `E-LINALG-SCHUR-SLAB-SOLVE` and
`E-LINALG-SCHUR-SLAB` (a band or shape that does not fit the operator). The
[Brusselator example](brusselator.md) uses it as its decomposed diffusion solver (`--solver schur`).

## The derivative gate: test an operator before Newton trusts it

A Jacobian or transpose that is slightly wrong does not crash a Newton solve; it slows it down or
makes it converge to the wrong place. `fuse/linalg/derivative_gate.hpp` holds three host-side checks
you can run on your own operator in a unit test. They work on plain `std::span<double>` /
`std::vector<double>` through `HostApplyFn`, `HostVectorFn` and `HostJacobianActionFn`.

**Adjoint identity.** Checks ⟨A x, y⟩ = ⟨x, Aᵀ y⟩ to within c · ε · ‖A‖ · ‖x‖ · ‖y‖, with c =
`kAdjointGateConstant` (50) by default and ‖A‖ estimated by power iteration on AᵀA
(`estimate_operator_norm`):

```cpp
HostApplyFn apply = [&](std::span<const double> in, std::span<double> out) {
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = 2.0 * in[i];
    if (i > 0) out[i] -= in[i - 1];
    if (i + 1 < n) out[i] -= in[i + 1];
  }
  return fuse::Result<void>::success();
};
HostApplyFn apply_transpose = apply;             // symmetric, so the transpose is itself
const auto norm_est = estimate_operator_norm(n, n, apply, apply_transpose);
std::vector<double> xv(n), yv(n);
// ... fill xv, yv with any non-trivial vectors ...
const auto adj = adjoint_gate(n, n, apply, apply_transpose, xv, yv, norm_est.value());
std::cout << "adjoint gate: defect " << adj.value().defect << " <= bound " << adj.value().bound
          << " -> " << (adj.value().passed ? "passed" : "FAILED") << "\n";
```

```text
adjoint gate: defect 0 <= bound 1.83511e-13 -> passed
```

`AdjointGateResult` carries `defect`, `bound`, `norm_est`, `x_norm`, `y_norm` and `passed`. A
shape mismatch is `E-GATE-ADJOINT-SHAPE` (`E-GATE-NORM-SHAPE` for the norm estimate).

**Taylor remainder.** Measures the order of ‖F(x + h v) − F(x) − h J v‖ over `ndecades` decades of
h starting at `h0`. A correct Jacobian action gives order 2 until round-off takes over; a wrong one
gives order 1. The round-off plateau is detected (`plateau_drop`) and excluded from the fit:

```cpp
HostVectorFn F = [](const std::vector<double>& u) { /* F(u)_i = u_i^3 - 1 */ };
HostJacobianActionFn Jv = [](const std::vector<double>& u, const std::vector<double>& v) {
  /* (J v)_i = 3 u_i^2 v_i */ };
const auto tr = taylor_remainder_order(F, Jv, x0, dir, /*h0=*/1e-1, /*ndecades=*/8);
std::cout << "taylor remainder: observed order " << tr.order << " from " << tr.fit_points
          << " points, round-off floor " << tr.floor << "\n";
```

```text
taylor remainder: observed order 2.0323 from 8 points, round-off floor 1.57967e-15
```

`TaylorRemainderResult` holds the sampled `h`, `remainder`, the fitted `order`, the `floor`,
`at_floor_everywhere` (every sample sat in round-off, so no order could be measured) and
`fit_points`.

**Matrix-free agreement.** `matrix_free_agreement(rows, cols, a, b, n_vectors, norm_est, c, seed)`
applies two operators that should be equal (an assembled matrix and a matrix-free action, say) to
`n_vectors` random vectors and reports the worst defect against c · ε · ‖A‖ · ‖x‖, with c =
`kMatrixFreeAgreementConstant` (20). Shape mismatch is `E-GATE-AGREE-SHAPE`.

:::{tip}
Run the Taylor remainder test on every `LinearizationToken` you write ([](component-interface.md))
and the adjoint identity on every token whose `apply_jacobian_transpose()` is more than a stub.
[](petsc-reaction-diffusion.md) ships both as tests next to its component, including two deliberately
wrong Jacobians that measure order 1.
:::

## Newton-Krylov: `fuse::num::solve`

The native solver is Jacobian-free Newton-Krylov: each Newton step solves J δ = −F with GMRES, where
J v is approximated by a finite difference of F (`MffdJacobianAction`), the GMRES tolerance follows
an Eisenstat-Walker forcing sequence, and the step length comes from a line search.

```cpp
using namespace fuse::num;
// F(x)_i = atan(x_i) - atan(c_i): a scalar problem per entry, solved as one system.
const std::vector<double> c = {0.5, -1.0, 2.0, 0.0, 1.0, -0.5, 3.0, 0.25};
ResidualFn residual = [&](DistVector& u, DistVector& fu) -> fuse::Result<void> {
  for (std::size_t i = 0; i < c.size(); ++i) fu.data()[i] = std::atan(u.data()[i]) - std::atan(c[i]);
  return fuse::Result<void>::success();
};
DistVector u(c.size(), exchanger, reducer);
for (std::size_t i = 0; i < c.size(); ++i) u.data()[i] = 20.0;   // far from the root
NewtonKrylovOptions nopt;
nopt.rtol = 1e-10;
nopt.atol = 1e-12;
nopt.max_nonlinear_iterations = 60;
nopt.line_search.kind = LineSearchKind::BT;        // backtracking; CP and Basic also exist
nopt.forcing.choice = EisenstatWalkerChoice::Choice2;
nopt.forcing.eta_min = 1e-4;
auto nr = solve(residual, u, /*precond_apply=*/nullptr, /*refresh_precond=*/nullptr, nopt);
const NewtonResult& res = nr.value();
std::cout << "newton: " << to_cstring(res.status) << " in " << res.iterations << " steps, "
          << res.function_evaluations << " F evaluations, |F| " << res.initial_residual_norm
          << " -> " << res.final_residual_norm << ", u[2] = " << u.data()[2] << "\n";
for (const NewtonIterationRecord& rec : res.trace) {
  std::printf("  it %d  |F| %.3e  eta %.2e  gmres %zu  lambda %.3f\n", rec.iteration, rec.norm_f,
              rec.eta, rec.gmres_iterations, rec.line_search_lambda);
}
```

```text
newton: ConvergedAbsolute in 21 steps, 1006 F evaluations, |F| 3.88603 -> 5.07585e-14, u[2] = 2
  it 0  |F| 3.886e+00  eta 3.00e-01  gmres 1  lambda 0.045
  it 1  |F| 2.576e+00  eta 3.95e-01  gmres 4  lambda 0.017
  ...
  it 17  |F| 2.683e-02  eta 1.98e-01  gmres 3  lambda 1.000
  it 18  |F| 3.110e-03  eta 1.21e-02  gmres 6  lambda 1.000
  it 19  |F| 1.724e-05  eta 1.00e-04  gmres 5  lambda 1.000
  it 20  |F| 5.951e-10  eta 1.00e-04  gmres 5  lambda 1.000
```

The arctangent from a starting point of 20 is the classic case where the full Newton step
overshoots: the line search takes short steps (λ ≈ 0.02–0.3) for a dozen iterations, then the
quadratic phase takes over and `eta` drops to its floor.

### The call

```cpp
Result<NewtonResult> solve(const ResidualFn& f, DistVector& x,
                           const PreconditionerApplyFn& precond_apply,
                           const RefreshPreconditionerFn& refresh_precond,
                           const NewtonKrylovOptions& opt = {});
```

- `ResidualFn` is `std::function<Result<void>(DistVector& x, DistVector& Fx)>`. An error from it
  ends the solve with `NewtonStatus::DivergedFunctionError`.
- `precond_apply` (`Result<void>(DistVector& r, DistVector& z)`) and `refresh_precond`
  (`Result<void>(const DistVector& x, const DistVector& fx)`) may both be `nullptr` for
  unpreconditioned JFNK. The refresh runs before the first step and every
  `preconditioner_lag` steps after.
- On return `x` is always a point at which F was evaluated. Divergence is reported in `status`,
  not as an error.

### `NewtonKrylovOptions`

| Field | Default | Meaning |
|---|---|---|
| `atol`, `rtol`, `stol` | `1e-50`, `1e-8`, `1e-8` | stop on ‖F‖ ≤ atol, ‖F‖ ≤ rtol ‖F₀‖, or a step shorter than stol relative to ‖x‖ |
| `divergence_tolerance` | `1e4` | `DivergedResidualGrowth` when ‖F‖ exceeds this multiple of ‖F₀‖ |
| `max_nonlinear_iterations` | `50` | `DivergedMaxIterations` |
| `max_function_evaluations` | `10000` | `DivergedFunctionCount`; MFFD evaluations count |
| `mffd` | `MffdOptions{}` | `error_rel` (√ε) and `min_direction_norm` for the finite-difference step h |
| `forcing` | `EisenstatWalkerOptions{}` | see below |
| `gmres` | `GmresOptions{}` | the inner solver's restart and iteration budget; `rel_tol` is overridden by the forcing term |
| `line_search` | `LineSearchOptions{}` | see below |
| `preconditioner_lag` | `1` | rebuild the preconditioner every this many Newton steps |

`NewtonResult` carries `status`, `iterations`, `function_evaluations`, `initial_residual_norm`,
`final_residual_norm` and a `trace` of `NewtonIterationRecord`s (`norm_f`, `eta`,
`gmres_iterations`, `gmres_residual_norm`, `mffd_steps`, `line_search_status`,
`line_search_lambda`, `preconditioner_refreshed`). `is_converged(status)` and
`is_diverged(status)` classify the `NewtonStatus` values `ConvergedAbsolute`, `ConvergedRelative`,
`ConvergedStep`, `DivergedFunctionCount`, `DivergedLineSearch`, `DivergedResidualGrowth`,
`DivergedLinearSolve`, `DivergedMaxIterations` and `DivergedFunctionError`.

### Eisenstat-Walker forcing

The GMRES relative tolerance η_k at step k is chosen from the nonlinear progress rather than fixed:

| `EisenstatWalkerChoice` | η_k |
|---|---|
| `Constant` | `eta_constant` (0.1) every step |
| `Choice1` | `|‖F_k‖ − ‖F_{k−1} + J_{k−1} d_{k−1}‖| / ‖F_{k−1}‖`, how well the last linear model predicted the new residual |
| `Choice2` (default) | `gamma · (‖F_k‖ / ‖F_{k−1}‖)^alpha`, with `gamma = 0.9`, `alpha = 2` |

`eta_initial = 0.3` starts the sequence, `eta_max = 0.9` caps it and `eta_min = 0` floors it. The
over-solving safeguard raises η_k to at least `gamma · η_{k−1}^alpha` whenever that value exceeds
`safeguard_threshold = 0.1`, so one lucky step cannot demand an unreasonably tight linear solve on
the next. A floor of
`eta_min = 1e-4` keeps GMRES from chasing the noise floor of the finite-difference Jacobian; the
Brusselator and PETSc reaction-diffusion examples both set it. `eisenstat_walker_eta(opt, k,
eta_prev, norm_fk, norm_fkm1, linear_residual_prev)` computes one value if you want the formula on
its own.

### Line search

`fuse/num/line_search.hpp` offers three kinds, selected by `LineSearchOptions::kind`:

| `LineSearchKind` | Function | Behaviour |
|---|---|---|
| `BT` (default) | `backtracking_line_search(f0, MeritFn, opts)` | Armijo backtracking on f(λ) = ½‖F(x + λd)‖²; `armijo_alpha = 1e-4`, `lambda_min = 1e-12`, `max_backtracks = 40`, safeguarded quadratic/cubic interpolation within `[safeguard_low, safeguard_high]` = [0.1, 0.5] |
| `CP` | `critical_point_line_search(g0, DirectionalFn, opts)` | secant iteration on ⟨F(x + λd), d⟩ = 0; `secant_tol = 1e-10`, `max_secant_steps = 40` |
| `Basic` | `basic_line_search(opts)` | no evaluation: λ = `damping` (1.0) |

A `MeritFn` returns `Result<double>` for a given λ, already reduced across ranks. The result
`LineSearchResult{lambda, steps, status, trials}` has a `LineSearchStatus` of `FullStep`,
`Converged`, `MaxStepsReached` or `Failed`.

```cpp
// Merit f(lambda) = 1/2 ||F(x + lambda d)||^2 for a scalar F(x) = x^3 - 1 at x = 2 with d = -4.
const double xs = 2.0, d = -4.0;
auto Fs = [](double q) { return q * q * q - 1.0; };
const double f0 = 0.5 * Fs(xs) * Fs(xs);
MeritFn merit = [&](double lambda) -> fuse::Result<double> {
  const double fq = Fs(xs + lambda * d);
  return 0.5 * fq * fq;
};
LineSearchOptions ls;
ls.kind = LineSearchKind::BT;
auto lr = backtracking_line_search(f0, merit, ls);
std::cout << "line search: lambda = " << lr.value().lambda << " after " << lr.value().steps
          << " backtracks, status " << to_cstring(lr.value().status) << "\n";
```

```text
line search: lambda = 0.376923 after 2 backtracks, status Converged
```

### Matrix-free Jacobian action on its own

```cpp
MffdJacobianAction mffd(residual, MffdOptions{});
mffd.set_point(u, fx);                              // F(x) is already known; no extra evaluation
mffd.apply(v, jv);                                  // one new F evaluation
std::cout << "mffd: h = " << mffd.trace().front().h << ", (Jv)[2] = " << jv.data()[2] << "\n";
```

```text
mffd: h = 1.17154e-08, (Jv)[2] = 0.2 (exact 1/(1+c^2) = 0.2)
```

`apply()` before `set_point()` is refused with `E-NEWTON-MFFD-NO-POINT`. `trace()` records
`MffdRecord{h, x_norm, v_norm}` per application and `evaluations()` the running count the solver
charges against `max_function_evaluations`.

### Substrates: Native and PETSc

`solve()` is the native substrate. A second implementation can be registered under
`NewtonSubstrate::Petsc`, and a Newton region picks one through `NewtonConfig::substrate`:

```cpp
using namespace fuse::num;
const NewtonSolveFn* native = newton_substrate(NewtonSubstrate::Native);   // always registered
const NewtonSolveFn* petsc = newton_substrate(NewtonSubstrate::Petsc);     // nullptr until fuse::petsc::install()
std::cout << "native substrate registered: " << (native != nullptr)
          << ", petsc: " << (petsc != nullptr) << "\n";
```

```text
native substrate registered: 1, petsc: 0
```

`register_newton_substrate(kind, NewtonSolveFn)` registers or replaces an entry. A `NewtonSolveFn`
has the signature of `solve()` plus a trailing `int comm_f`. In a build with `FUSE_ENABLE_PETSC`,
calling `fuse::petsc::install()` once at start-up registers PETSc's `SNESNEWTONLS` as the `Petsc`
substrate; a region selecting it in a process that never called `install()` is refused at
`Runner::initialize()` with `E-NEWTON-PETSC-UNAVAILABLE`. `fuse::petsc::make_assembled_token(rows,
cols, triplets, comm_f)` wraps an assembled PETSc `Mat` as a `LinearizationToken`
([](petsc-reaction-diffusion.md)). [](dn-heat.md#the-petsc-substrate) runs the same graph on both
substrates and bounds their disagreement.

### Debugging a Newton solve

Set `FUSE_NEWTON_DEBUG=1` to have a Newton iteration region print, on stderr, the region's stacked
vector layout (`[newton-debug] region=... newton_n=...`, one line per block with its channel, length
and offset) and then one line per Newton iteration (`norm_f`, `eta`, `gmres_it`, `gmres_res`,
`ls_lambda`, `precond_refreshed`, `mffd_evals`) followed by the final status. From a driver, the same
trace is in `NewtonResult::trace`; `Runner::debug_newton_gmres_iterations(region_index)` returns the
GMRES iteration counts of a region's most recent solve.

## How a Newton iteration region uses all this

When a region is declared with `.newton(cfg)` ([](convergence.md#newton-at-a-glance)), the Runner:

1. Stacks every exchanged field in the region into one interface vector x, one block per channel,
   and builds a `ResidualFn` that evaluates F(x) = x − G(x) by delivering x to the members,
   advancing each one under `save_state()` / `restore_state()` (or calling `evaluate_residual()`
   directly when `NewtonConfig::residual_path = Direct`), and reading back what they provide.
2. Translates `NewtonConfig` field by field into `NewtonKrylovOptions` (`line_search_kind`,
   `forcing_*`, `atol`/`rtol`/`stol`, `gmres_restart`, `preconditioner_lag`, ...).
3. If `use_block_jacobi_preconditioner` is set, asks each member's `LinearizationToken` for
   `apply_preconditioner()` and composes them with `BlockJacobiPreconditioner::from_blocks()`.
   Otherwise the preconditioner is the identity.
4. Under `NewtonNorm::MeasureWeighted`, scales the residual by a fixed diagonal before the
   substrate sees it.
5. Builds the region's own reducer over exactly the ranks that host its members
   (`make_participant_set_reducer`) and dispatches to the selected substrate through the registry.

The members never see GMRES or the line search; they see `advance()` or `evaluate_residual()`
calls at frozen time levels. [](component-interface.md#joining-a-newton-region) is the contract on
that side.

## Limitations

- `SchurSlabSolver` is not available with `FUSE_LINALG_BACKEND=builtin`; it refuses with
  `E-LINALG-SCHUR-BACKEND`.
- The derivative gate functions are host-only and single-process; they take plain vectors.
- `BlockJacobiPreconditioner` recombines blocks over the owned part of the vector only.
- `NewtonResidualPath::Direct` applies to ordinary regions; a window-scoped region selecting it is
  refused with `E-NEWTON-DIRECT-WINDOW`.

## See also

- [](convergence.md): choosing the Newton norm and tolerances for a region.
- [](component-interface.md): `ImplicitComponent` and `LinearizationToken`.
- [](dn-heat.md) and [](petsc-reaction-diffusion.md): Newton regions on the native and PETSc substrates.
- [](brusselator.md): PCG, GMRES and the Schur solver as a component's own diffusion solvers.
- [](build.md): `FUSE_LINALG_BACKEND` and `FUSE_ENABLE_PETSC`.
