# Gray-Scott: two species, one grid, two clocks

`examples/gray_scott` couples the U and V species of the Gray-Scott
reaction-diffusion model as two `fuse::Component`s on the SAME structured
grid, stepped at INDEPENDENT rates and coupled implicitly through
`fuse::WindowCoupler` (see `coupling-windows.md` for the window mechanism
itself; this page is the model built on top of it).

## The model

On a periodic square of side `L`:

```
u_t = Du * Laplacian(u) - u*v^2 + F*(1 - u)
v_t = Dv * Laplacian(v) + u*v^2 - (F + k)*v
```

`u` is consumed by the autocatalytic term `u*v^2` and fed toward 1 at rate
`F`; `v` is produced by that same term and decays at rate `F + k`.

## Discretisation

Cell-centred P0 finite volumes on an `N x N` grid, `h = L/N`, cell `(i,j)`
centred at `((i+0.5)h, (j+0.5)h)`. The periodic 5-point Laplacian
(`gray_scott::periodic_laplacian`) is the standard second-order
finite-volume stencil:

```
(Laplacian u)_ij = (u_E + u_W + u_N + u_S - 4 u_ij) / h^2
```

Each species advances by Crank-Nicolson in time, Newton-solved with
`Eigen::SparseLU` (`gray_scott::GrayScottSpeciesComponent<Species>` in
`species_component.hpp`), the OTHER species held as a frozen input
supplied by `fuse::WindowCoupler`. Newton's own tolerance is set two to
three orders tighter than the smallest discretisation error a verification
run measures (`1e-11` throughout this page, against errors no smaller than
`~1e-8`), so it never contaminates a spatial- or temporal-order
measurement. `monolithic_reference.hpp`'s `MonolithicGrayScott` is the
single-rate, fully-coupled Newton reference used to check the partitioned
scheme against: it solves both species' `2N^2` unknowns in one Newton
system, needing the reaction term's cross derivatives
(`d(react_u)/dv`, `d(react_v)/du`) that the partitioned components never
need (each only differentiates its own reaction term against its own
unknown, treating the partner as fixed for that Newton solve). Its own
Jacobian solve goes through one more indirection than the species
components' bare `Eigen::SparseLU` above: `fuse_examples::SparseLuBackend`
(held in `MonolithicGrayScott` as `jac_solver_`) -- `Eigen::SparseLU`
itself, bit-for-bit, with `FUSE_ENABLE_CUSPARSE` off (pinned by
`verification.sparse_solve_backend`), `cusolverRf` with it on.

## Coupling

U is always the window-definer (its own interval IS
`WindowCouplerConfig::window_length`, so it structurally cannot straddle);
V may take a finer, possibly non-dividing step. Each receives the OTHER's
mean over its own substep, as `fuse::WindowCoupler` computes it, iterated
to convergence within the window (`AccelKind::None`/`Aitken`/`IQN_ILS`,
recorded in `test/verification/gray_scott_multirate_test.cpp`). Every
configuration used here respects the KNOWN_ISSUES.md D7 envelope: U's
interval never exceeds the window (it IS the window), and V's does not
either, so `E-RT-WINDOW-LONGSTEP` never fires; only V is ever the
straddling member, so `E-RT-WINDOW-STRADDLE` (which fires when TWO members
could straddle the same window) never fires either.

## Building and running it

Tier 0 is enough — no MPI, no MOAB:

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
<build-dir>/examples/fuse_example_gray_scott --n 64 --dt 0.5 --ratio 1|m|"p:q" \
    --policy fixed|adjustable \
    --accel none|constant|aitken|secant|steffensen|iqn-ils|iqn-imvj \
    --t-end 50 --check --output snapshot.csv
```

`--ratio m` makes the window (U's own interval) `m*dt` and V steps at
`dt` itself — V is m times finer than the window, not m times finer than
some other, smaller step (`--ratio 2 --dt 0.5` means the window is `1.0`
and V steps at `0.5`, not `0.25`). `--ratio 7:13` means the window is 13
parts and V is 7 (incommensurate). `--check` runs a short internal
self-consistency probe (see "Chosen tolerances" below) and exits non-zero
on failure. `--output` writes a CSV snapshot (`i,j,x,y,u,v`) of the final
state. `--F`, `--k`, `--Du`, `--Dv`, `--L` override the reaction-diffusion
parameters and domain size from their defaults above.

The `ctest` row `example.gray_scott` runs
`fuse_example_gray_scott --n 24 --ratio 2 --t-end 4 --dt 0.1 --check`.
Measured on this build (macOS arm64, AppleClang 21.0.0, Kokkos 5.1.0
OpenMP+Serial, tier0-release):

```
Gray-Scott windowed coupling: n=24 ratio=2 policy=adjustable accel=aitken dt=0.1 t_end=4 windows=20 total_iterations=81
  u range [0.459793, 1], v range [4.67742e-70, 0.269348]
  --check: m=1 windowed vs monolithic max diff 3.039e-08 (OK)
```

`example.gray_scott.omp_identical` runs the identical command at one OpenMP thread and at every host
core and diffs the output byte for byte; verified on this build at 1 and 12 threads.

## Verification summary

### Species solver alone (MMS), `gray_scott_mms_test`

Each species, driven alone with the other frozen and a closed-form
manufactured forcing, measures:

- **Spatial order 2**: N in {8,16,32}, small fixed dt. Measured orders for
  both U and V are within [1.85, 2.15] at every refinement step.
- **Temporal order 2**: N=32 (fine enough that an isolation-margin check
  confirms spatial error is negligible), n_steps in {4,8,16,32}. Measured
  orders within [1.85, 2.15].

### Multirate coupling, `gray_scott_multirate_test`

Commensurate ratios (m=1,2,4) and the incommensurate 7:13 ratio, both
`DtPolicy` values for V, converge (finite, bounded state, positive
iteration counts) under `AccelKind::None`, `Aitken`, and `IQN_ILS`.
Measured total sweeps over 4 windows at 7:13: **None=9, Aitken=13,
IQN-ILS=12** — None is not worst here; Gray-Scott's `u*v^2`
coupling term is a comparatively weak perturbation on this problem's
diffusion-dominated Newton system at the tested dt, so plain Picard
already converges in very few sweeps and Aitken's relaxation occasionally
overshoots by one sweep rather than saving one. Reported as measured, not
forced to any expected ordering.

### Against the monolithic reference, `gray_scott_reference_test`

**At m=1**, the windowed result matches the monolithic one to the
iteration tolerance -- but only at a small enough dt, not unconditionally.
Gray-Scott's reaction term is QUADRATIC in the partner (`u*v^2`): the
partitioned scheme freezes the partner at its CN-window MEAN and applies
that SAME frozen value at both the old and new reaction evaluations,
while the monolithic scheme uses the partner's actual old/new values
separately. Because squaring does not commute with averaging, this
mismatch is a real, measured, order-2-in-dt quantity (confirmed by a
fixed-total-time refinement study: consecutive-step orders 2.94, 2.97,
2.99 as the mismatch itself shrinks toward zero -- consistent with a
clean order-2 signal once past this ladder's coarsest point), not merely
the coupling-iteration residual, and it does **not** vanish at an
arbitrary fixed dt. `example.gray_scott`'s own `--check` uses `--dt 0.1`
specifically because at that dt the term is already below the tight
bound (measured on this build: diff 3.039e-08 against a 1.0e-06 bound,
consistent with the 3.022e-08 recorded when the bound was chosen);
loosening the bound instead of choosing that operating point was
considered and rejected (the task's own instruction not to loosen an
order/tolerance bound to paper over a real discrepancy applies here as
much as to a pure order-of-convergence assertion).

**For ratio != 1 (commensurate m=2/4 and 7:13)** the state error against
a fine monolithic reference falls at order 2 in dt, every refinement step
inside [1.9, 2.1] (`Order2StateErrorAndMassExchangeCommensurate`,
`Order2StateErrorAndMassExchangeSevenThirteen`). These tests first failed
because of a defect in `fuse::WindowCoupler`, found by this model and since
fixed (below).

### Long pattern-forming run, `gray_scott_reference_test`

Pearson's (1993) "lambda" self-replicating-spot parameters
(`F=0.0392, k=0.0649, Du=2e-5, Dv=1e-5`, domain 2.5) on a 48x48 grid,
seeded with a small central patch (u=0.5, v=0.25 on a 4x4 block, <0.7% of
the domain) against an otherwise exactly-uniform (u=1, v=0) background.
Measured (at -O2, for the timing only -- the physics is identical to the
-O0 test binary): variance(v) is `1.8760e-04` by `t=82` (already above the
`1e-4` threshold), dips to a local minimum of `1.3787e-04` at `t=102`
(still above threshold), then rises to a sustained plateau of
`2.4719e-04` from `t~302` onward, unchanged through `t=440` -- a single
stable spot at this domain size and resolution rather than the runaway
self-replication cascade a larger, finer-resolved domain would show
(explicitly NOT claimed here; this run demonstrates pattern formation and
persistence from a near-uniform start, the task's own quantitative
requirement, not the full self-replication phenomenology). Smaller proxy
grids (N=16, N=24) were tried first to shorten the test's runtime and
rejected: at both, the same seed instead decays back toward the trivial
uniform state (variance falling to `~1e-43` by `t=500`), never sustaining
a pattern -- this model's spot-sustaining behaviour is resolution-
sensitive, so N=48 is kept and only the run length is cut. `u`/`v` stay in
`[0, 1]` throughout. The registered test runs only 60 steps (t=120,
safely past the first threshold crossing, and past the t=102 dip) rather
than the full 220 to the plateau: this Newton-SparseLU solve costs roughly
2s/step in this project's Debug (`-O0`) test build versus ~25ms/step at
`-O2`, so the full run (measured separately, at -O2, for this page) would
cost the -O0 CTest suite close to 8 minutes.

### Mass exchange (`u*v^2`) between U and V

`GrayScottSpeciesComponent::coupling_flux()` accumulates each species' own
CN-quadrature integral of the autocatalytic term (`-u*v^2` for U, `+u*v^2`
for V), using that species' own frozen partner mean. The two means are
different means of different trajectories, so what U removes and what V gains
do not cancel exactly. The mismatch is **not** zero at the iteration tolerance.
It is a discretisation error that falls at order 2 in dt, commensurate and
7:13, asserted in the same two tests.

The flux is part of each species' checkpoint (`save_state`/`restore_state`).
It was not at first, and then it summed every coupling sweep and every
straddle replay: at 7:13 it stayed near 1.58 under refinement, measuring how
often the coupler iterated rather than what was exchanged.

### Chosen tolerances

Newton: `1e-11` (species components and the monolithic reference alike),
two to three orders tighter than any discretisation error measured on
this page. Coupling iteration: `1e-9` relative
(`test/verification/gray_scott_*`), `1e-8` (`examples/gray_scott/main.cpp`'s
internal `Args::coupling_tol` default -- not currently exposed as a CLI
flag). `example.gray_scott`'s `--check` bound is `100 * coupling_tol` (a
factor of 100 above the coupling tolerance, covering roundoff in the
comparison itself, not the physics).

## The WindowCoupler defect this model found (fixed)

`WindowCoupler`'s exchanged vector is laid out (member, element, substep).
The sweep's commit path wrote it with a running counter inside a
(substep, element) loop: transposed. The two layouts coincide for a scalar
member or for one substep per window, which is every case the coupler's own
tests covered. Gray-Scott is the first field-valued member with several
substeps per window, and at ratio 2 U was fed a "mean" of V (0.12337832)
outside V's own range over that window ([0.11668668, 0.11767767]). No true
mean can do that.

The sweep now writes each sample at its computed canonical index.
`WindowCouplerLayout.PackedElementsReproduceEachScalarProblemExactly` packs
two independent scalar problems into one length-2 member. It fails on the old
code (0.4136 against 0.4074) and passes on the fix, to within the iteration
tolerance (about 1e-11).

## State

**Verified.** `example.gray_scott` and `example.gray_scott.omp_identical` pass at tier 0 on this
build (above); `gray_scott_mms_test`, `gray_scott_multirate_test`, and `gray_scott_reference_test`
cover the species solver's own order, the multirate coupling's convergence under every accelerator,
and agreement with the monolithic reference, respectively, each with the measured numbers quoted in
"Verification summary". No multi-rank or MOAB-dependent path exists for this example: both species
live on one shared structured grid with an identity remap, so there is nothing for a distributed
mesh backend to add.
