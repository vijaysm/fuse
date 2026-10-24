# Brusselator: multirate implicit coupling and separately built executables

`examples/brusselator/` couples a multirate 2D Brusselator reaction–diffusion model on two
independent native meshes. Read this page if you want a worked example of a multirate implicit
(Picard, Aitken, quasi-Newton or Newton) coupling with conservative remap between non-matching
meshes, of a grid-decomposed participant with a pluggable diffusion solver, or of a coupling split
across separately built C, Fortran and C++ executables.

The directory holds three things:

- `fuse_example_brusselator`: an in-process, single-executable example that runs the whole coupled
  model directly, with either `fuse::WindowCoupler` or `fuse::Runner`.
- `brusselator_graph_tool`: a small tool that writes the identical coupling specification as text
  YAML (`fuse::to_yaml()`).
- `brusselator_u_main` (C), `brusselator_v_main` (Fortran) and `brusselator_coupler_main` (C):
  three separate executables that load that YAML with `fuse_load_graph()` and run the same model as
  three independently built programs. See [](multiple-executables.md) for the launch modes.

## The model

On the periodic unit square,

```
U_t = a*Laplacian(U) + B - U + q + f
V_t = a*Laplacian(V) - q
q   = U^2*V - A*U
```

with `A = 3.4`, `B = 1`, `a = 10` by default. Initial conditions sample `U0 = 22*[y(1-y)]^1.5` and
`V0 = 27*[x(1-x)]^1.5` at cell midpoints. The forcing `f` is 5 inside the radius-0.1 circle centred
at `(0.3, 0.6)` for `t >= 1.1`, and zero elsewhere and before. Defaults: `nx_u=ny_u=16`,
`nx_v=ny_v=24`, `dt_u=0.002`, `dt_v=0.003`, `window=0.02`, `end=11.5`, Picard tolerance `1e-11`,
60 maximum iterations.

U and V each run their own backward-Euler time loop at their own step size on their own mesh. They
exchange one shared reaction rate `q` once per coupling window, through the conservative
(area-overlap, mean-preserving) supermesh remap `P[T,S] = area(T∩S)/area(T)`.

## Run it in one process

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_brusselator \
  --nx-u 16 --ny-u 16 --nx-v 24 --ny-v 24 \
  --dt-u 0.002 --dt-v 0.003 --window 0.02 --end 11.5 \
  --out results
```

The example needs Eigen and Kokkos (a build with `FUSE_LINALG_BACKEND=builtin` skips it; see
[](build.md)). A quick run that also checks the invariants below:

```sh
<build-dir>/examples/fuse_example_brusselator --end 0.5 --check
```

Outputs under `--out`: `u.csv` and `v.csv` (final fields, `time,x,y,value`) and `diagnostics.txt`
(parameters and the measured residuals). `--save-history` also writes `windows.csv`
(`t0,t1,mean_u,mean_v,iterations` per window).

### Command-line flags

| Flag | Meaning |
|---|---|
| `--nx-u --ny-u --nx-v --ny-v` | Grid sizes of U and V |
| `--dt-u --dt-v` | Time steps of U and V |
| `--window` | Coupling window length |
| `--end` | End time |
| `--alpha` | Diffusion coefficient `a` |
| `--tolerance`, `--max-iterations` | Picard convergence tolerance and iteration cap |
| `--no-forcing` | Switch the forcing `f` off |
| `--accel none\|constant\|aitken\|secant\|steffensen\|iqn-ils\|iqn-imvj\|newton` | Sets `fuse::WindowCouplerConfig::acceleration`; default `none` (plain Picard). `iqn` is an alias of `iqn-ils`. An unknown name is refused with `E-EXAMPLE-ACCEL` before any setup. `constant`, `secant`, `steffensen` and `iqn-imvj` use the library's default settings for that kind |
| `--driver window\|runner` | Use `fuse::WindowCoupler` (default) or `fuse::Runner` directly. Both pass `--check` for every `--accel` value that `--driver window` supports |
| `--iteration-mode jacobi\|gauss-seidel`, `--no-block-jacobi` | Newton window iteration options (see [Newton acceleration](#newton-acceleration)) |
| `--substrate native\|petsc` | Newton solver substrate; read only with `--accel newton` |
| `--save-history`, `--out`, `--check` | Output controls and the pass/fail check below |

### `--check`

Exits non-zero unless: every window converged; the final U and V fields are `>= -1e-12`
everywhere; `max_window_balance_error <= 1e-11` and `|global_balance_error| <= 1e-9`;
`max_window_exchange_error <= 1e-9`. The thresholds are round-off scale.

## How the coupling is built

**Mesh and remap.** Cell geometry comes from `fuse::native::NativeMeshBackend`'s
`"generated:quad:NxM"` generator. `fuse::build_weights` dispatches to the dependency-free order-1
simplex supermesh path for any non-MOAB backend. The example recovers the native-index to
canonical-index correspondence by reading back every cell's centroid, and refuses with
`E-BRUSS-MESH-INDEX` if that is not a clean bijection (this cannot happen with the built-in
generator).

**One component owns `q`.** A `Component` receives one snapshot per `advance()` call, valid at
`ctx.t_begin`. `UComponent` therefore computes `q` from its own state and the just-received,
remapped V mean, and consumes over its step the trapezoid `0.5*dt*(q_prev + q_new)`, which is
exactly what `WindowCoupler` delivers to V from the same two consecutive samples
(`TemporalBasis::Linear`). Because `q_new` depends on the state being solved for, U's step is a
short fixed point against one factorisation of the diffusion operator; it converges in a handful of
solves and refuses with `E-BRUSS-U-REACTION` if fifty do not suffice. `VComponent` provides
`P_VU(V)` directly. Both converge to the same discrete-in-space, first-order-in-time solution.

**The forcing switch is a window boundary.** `WindowCouplerConfig::window_length` is fixed per
instance, so `main.cpp` builds the list of window boundaries with `t = 1.1` inserted if absent, and
uses a fresh `WindowCoupler` for each maximal run of equal-length windows. Boundaries are
`fuse::Rational`, so with the default `--window 0.02` (`55 * 0.02 == 1.1`) the switch is detected
by exact equality.

**Kokkos.** The elementwise reaction rate `q = U^2*V - A*U` runs as a `Kokkos::parallel_for`
(`ctest -L omp` exercises it at 1 and at every host core). The diffusion solve uses Eigen.

## Grid decomposition and solvers

U's and V's components each run on however many ranks the graph gives them.

- **Mesh and halo.** `NativeMeshBackend` slabs a generated mesh by rows; `NativeGrid` exposes the
  rank's owned row band plus one periodic ghost row above and below (a
  `fuse::linalg::GhostPattern`). `SlabField` stores owned-then-ghost cells in one buffer, the same
  layout as `fuse::linalg::DistVector`, refreshed through a `fuse::linalg::GhostExchanger`. With one
  rank the band is the whole grid and the local index equals the global one.
- **Means are global.** Every mean (`SlabField::owned_mean`) is an owned partial sum reduced across
  the component's communicator through a `fuse::linalg::Reducer`. Convergence of U's reaction fixed
  point is decided from a global `max()` so that all ranks leave the collective diffusion solve
  together.
- **Diffusion solvers.** `PhysicsConfig::solver` selects `Auto`, `Direct`, `Schur`, `Pcg` or
  `Gmres` (the `--solver auto|direct|schur|pcg|gmres` flag of `brusselator_u_main` and
  `brusselator_v_main`). `Direct` is Eigen `SimplicialLDLT` on the whole grid and is valid only on
  one rank (otherwise `E-BRUSS-DIFFUSION-DIRECT-DECOMPOSED`). `Schur` wraps
  `fuse::linalg::SchurSlabSolver`, an exact Schur-complement solve for slab-decomposed grids. `Pcg`
  and `Gmres` assemble a `fuse::linalg::DistSparseMatrix` and use `fuse::linalg::pcg`/`gmres` with a
  Jacobi preconditioner. `Auto` resolves to `Direct` on one rank and `Schur` on more. The Krylov
  tolerance (`solver_tolerance`) defaults to `1e-13`.
- **Remap between decomposed meshes.** The U-to-V and V-to-U remaps go through the library's
  remap-halo machinery (`fuse::build_remap_halo`, `fuse::build_weights`), so U and V may be split
  over different, even misaligned, rank sets. See [](placement.md).

## Newton acceleration

`--accel newton` (with `--driver runner` only; `--driver window` refuses it with `E-ACCEL-KIND`)
solves each coupling window as one Newton–Krylov problem instead of a Picard/Aitken/IQN sweep.
`--iteration-mode gauss-seidel` and `--no-block-jacobi` (the per-species physics-based
preconditioner, on by default) are available under every `--accel` value. For this example
`NewtonConfig::eta_min` is set to `1e-4`; without a floor on the Eisenstat–Walker forcing term the
first window's linear solve can stall once the requested accuracy falls below the noise of the
finite-difference Jacobian action.

Expected output of `--driver runner --accel newton --end 1.0` (50 windows; exact digits can vary
by platform):

| Quantity | Value |
|---|---:|
| Windows converged (Jacobi with block-Jacobi preconditioner; also with `gauss-seidel`; also with `--no-block-jacobi`) | 50/50 |
| `final_mean_u` / `final_mean_v` (all three variants) | 0.668994004315 / 2.8189403867 |
| Outer Newton iterations, first window (block-Jacobi / identity preconditioner) | 4 / 3 |
| Outer Newton iterations, later windows | 3 |
| `max_window_balance_error` / `global_balance_error` | 1.158e-13 / 4.663e-12 |

With the default parameters Aitken acceleration under `--driver window` becomes unstable at the
third window; Newton converges every window.

:::{warning}
Window Newton runs on a single rank; a Newton-selected region with more than one cross-rank
participant is refused.
:::

### PETSc substrate

`--substrate petsc` (read only under `--accel newton`) solves the identical window problem with
PETSc's `SNESNEWTONLS` (`fuse::petsc::solve()`) instead of the native Jacobian-free Newton–Krylov
coupler. It needs a build configured with `-DFUSE_ENABLE_PETSC=ON`; without it the flag is refused
with `E-NEWTON-PETSC-UNAVAILABLE`. The PETSc solve always runs on `MPI_COMM_SELF`.

Expected output for `--end 1.0`:

| Quantity | native | PETSc |
|---|---:|---:|
| Windows converged | 50/50 | 50/50 |
| Outer Newton iterations, first window | 4 | 5 |
| `final_mean_u` / `final_mean_v` | 0.668994004315 / 2.8189403867 | 0.668994004315 / 2.8189403867 |
| `max_window_balance_error` / `global_balance_error` | 1.158e-13 / 4.663e-12 | 1.144e-13 / 4.692e-12 |

SNES applies its convergence test after the step, so it takes one more outer iteration than the
native solver on the first window. The test `petsc.brusselator_newton` asserts element-wise
agreement of the final fields between the two substrates, with a bound calibrated from the native
solver's own error at the test tolerance and the two solvers' terminal residuals. See
[](petsc-reaction-diffusion.md) and [](convergence.md) for the Newton and PETSc interfaces.

## Run it as separately built executables

This path is for a code that is a separate executable built by its own build system, runs its own
time loop, calls FUSE when it wants to exchange, hands FUSE its existing arrays, and runs on its own
set of processes. `u_loop.c`'s `u_loop_step()` is U's own time loop, calling `fuse_begin_step()` and
`fuse_end_step()` whenever it wants to exchange; `u_main.c` binds the FUSE field registration
directly to its own storage with no copy. U and V's coupling-facing code is pure C and pure
Fortran; only the shared physics (`shim.cpp`) is C++, reached through a plain C wrapper.

Build (needs MPI, a Fortran compiler for V, and the Eigen backend):

```sh
cmake --preset tier1 && cmake --build --preset tier1
```

Author the graph once, then launch. The launch mode comes from the loaded YAML, not from a flag:

```sh
<build-dir>/examples/brusselator_graph_tool --ranks-u 1 --ranks-v 1 --ranks-coupler 1 \
    --launch mpmd --out graph.yaml

# mpmd, world size 3: one mpiexec, three ':'-separated segments
mpiexec -n 1 <build-dir>/examples/brusselator_u_main --graph graph.yaml --log u.log --periods 2 \
    : -n 1 <build-dir>/examples/brusselator_v_main --graph graph.yaml --log v.log --periods 2 \
    : -n 1 <build-dir>/examples/brusselator_coupler_main --graph graph.yaml

<build-dir>/examples/brusselator_check --u-log u.log --v-log v.log
```

`brusselator_graph_tool` flags: `--ranks-u N`, `--ranks-v N`, `--ranks-coupler N`,
`--launch mpmd|accept`, `--out FILE` (required), `--port-file`, `--transport fuse|lowfive`.
`brusselator_u_main` accepts `--graph`, `--log`, `--ranks-u`, `--periods`, `--transport
fuse|lowfive` and `--solver`; `brusselator_coupler_main` accepts `--graph` and `--periods`.
`brusselator_check` reads both executables' per-window logs and reproduces the in-process
`--check` verdict.

`test/parallel/brusselator_participants/run_participants.sh` wraps the whole sequence (graph
authoring, launch, check):

```sh
bash test/parallel/brusselator_participants/run_participants.sh mpmd <mpiexec> 1 1 1 \
    <build-dir>/examples/brusselator_graph_tool <build-dir>/examples/brusselator_u_main \
    <build-dir>/examples/brusselator_v_main <build-dir>/examples/brusselator_coupler_main \
    <build-dir>/examples/brusselator_check <scratch-dir> 2 60
```

The arguments are the mode, the `mpiexec` path, `ranks_u ranks_v ranks_coupler`, the five
executables, a scratch directory, the number of coupling periods and a timeout in seconds. It is
registered with `ctest` as `verification.brusselator.participants.{mpmd,accept}.{3,5}`, where the
number is the world size: 3 is `1+1+1` ranks (U, V, coupler) and 5 is `2+2+1`.

For **accept** mode, U and V are launched as two independent jobs that rendezvous over MPI dynamic
process management:

```sh
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 &
wait
```

`fuse::transport::connect_pairwise()` supports exactly two connect-mode components, so in accept
mode the coupler's ranks are extra ranks of U's own `MPI_COMM_WORLD` rather than a separate
executable. `--ranks-u N` tells `brusselator_u_main` how many of its local ranks are U-physics
ranks; the remaining local ranks are coupler placeholder ranks. In `mpmd` mode `--ranks-u` defaults
to the whole local world. If the accept rows are absent or `connect()` fails with
`E-CONNECT-ACCEPT-UNAVAILABLE`, see "Which launch modes this build offers" in
[](multiple-executables.md).

### Expected agreement

The `mpmd.3` run's final `mean_u` and `mean_v` are bit-identical to the single-process example; the
`mpmd.5` run (U on two ranks, V on two, coupler on one) differs from it by about 9e-14 absolute in
each mean, consistent with the decomposed Schur solve's round-off.

## Measured results

Expected values (exact digits can vary by platform). "Reference" means an independently written
implementation of the same model with its own mesh and remap.

| Category | Quantity | Value |
|---|---|---:|
| Weights vs. an independent oracle | max\|FUSE − oracle\|, V→U / U→V | 4.707e-14 / 1.061e-13 |
| | `P.1=1` and `area_target^T.P=area_source^T` | hold in both directions |
| Defaults, `--end 0.1` | `final_mean_u`: FUSE / reference / \|diff\| | 1.52801298656 / 1.5280505999004665 / 3.761e-5 |
| | `final_mean_v`: FUSE / reference / \|diff\| | 2.02340201528 / 2.0233654055333425 / 3.661e-5 |
| Defaults, `--end 11.5` | `final_mean_u`: FUSE / reference | 4.07486930721 / 4.0437736127620596 |
| | `final_mean_v`: FUSE / reference | 1.18944993515 / 1.2358694937952479 |
| | `max_window_balance_error` / `global_balance_error` | 7.576e-13 / 4.251e-11 |
| | windows / maximum iterations | 575 / 17 |
| Cross-code convergence, `--end 0.1`, 1x to 8x refinement | \|diff\| ratios, `mean_u` | 1.657, 1.696, 1.812 |
| | \|diff\| ratios, `mean_v` | 1.623, 1.670, 1.797 |
| Self-convergence in `dt`, `--end 0.1` | successive-difference order, `mean_u` / `mean_v` | 1.17, 1.13 / 1.19, 1.14 |
| Exchange conservation, `--end 0.1`, 1x/2x/4x | `max_window_exchange_error` | 7.984e-14, 7.501e-14, 4.147e-15 |
| | `max_window_balance_error` | 1.703e-13, 8.970e-14, 8.802e-15 |
| | `global_balance_error` | 5.300e-13, 2.112e-13, 7.964e-14 |
| Decomposed diffusion vs. single-rank `Direct` | Schur, U, 2 partitions (8x8, 20 steps) | 2.420e-14 |
| | Schur, U, 3 partitions (8x12) | 4.485e-14 |
| | Schur, V, 2 partitions | 2.665e-15 |
| | `Pcg` / `Gmres` at `rel_tol=1e-13` | 5.862e-14 / 5.462e-14 |
| Decomposed two-sided remap vs. single-rank | aligned split, 2 partitions, Schur | 1.821e-14 |
| | aligned split, 3 partitions, Schur | 7.994e-15 |
| | misaligned split, 3 partitions, Schur | 1.732e-14 |
| | misaligned split, 3 partitions, Pcg (20 sub-steps) | 5.262e-14 |
| `--end 0.1 --check` | `final_mean_u` / `final_mean_v` | 1.52801298656 / 2.02340201528 |
| `--end 0.5 --check` (the `ctest` default; same with `--driver runner`) | `final_mean_u` / `final_mean_v` | 1.10791238505 / 2.31637697544 |

The one-rank results of the decomposed code are byte-identical to the undecomposed code. The large
absolute difference from the reference at `--end 11.5` (about 0.03 in `mean_u`, 0.05 in `mean_v`) is
expected: two first-order schemes on a stiff nonlinear problem need not stay close in absolute terms
over 575 windows and the forcing switch, even though each converges. The convergence-order study is
therefore reported at `--end 0.1`.

## Limitations

- Window Newton (`--accel newton`) runs on one rank, and only with `--driver runner`.
- `--substrate petsc` needs a PETSc-enabled build (`E-NEWTON-PETSC-UNAVAILABLE` otherwise).
- The Fortran participant `brusselator_v_main` is built only when a Fortran compiler is found; all
  separately built executables need the Eigen linear-algebra backend.
- Accept-mode launches need working MPI dynamic process management between two jobs (see
  [](multiple-executables.md)).

Covered by the automated test suite, including multi-rank runs.

## See also

- [](multiple-executables.md): launch modes for splitting a coupling across executables
- [](coupling-windows.md): `WindowCoupler` and multirate windows
- [](convergence.md): accelerators and Newton
- [](placement.md): rank sets and decomposition
- [](c-abi.md), [](fortran.md): the C and Fortran interfaces the participants use
- [](lowfive.md): the `--transport lowfive` option
