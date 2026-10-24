# Gray-Scott: two species, one grid, two clocks

`examples/gray_scott` couples the U and V species of the Gray-Scott reaction-diffusion model as two
`fuse::Component`s on the same structured grid, stepped at independent rates and coupled implicitly
through `fuse::WindowCoupler`. Read it to see how a windowed multirate coupling is built and checked;
[](coupling-windows.md) explains the window mechanism itself.

## What it demonstrates

Two field-valued components with different time steps, iterated to convergence within each coupling
window, with a choice of acceleration and of step policy for the finer component, validated against a
fully coupled monolithic Newton reference. **This example requires the Eigen linear-algebra backend**
(`FUSE_LINALG_BACKEND=eigen`, the default); it is not built under `FUSE_LINALG_BACKEND=builtin`
because its Newton solves use `Eigen::SparseLU`.

## Build and run

```sh
cmake --preset tier0-release && cmake --build --preset tier0-release
# or, without presets:
#   cmake -S . -B build -DFUSE_ALLOW_FETCH=ON && cmake --build build

<build-dir>/examples/fuse_example_gray_scott --n 24 --ratio 2 --t-end 4 --dt 0.1 --check
```

See [](build.md) for the presets and the linear-algebra backend choice. Flags (defaults in
parentheses):

| Flag | Meaning |
|---|---|
| `--n N` (64) | Grid is `N x N` cells. |
| `--dt D` (0.5) | The fine time unit. |
| `--ratio 1\|m\|p:q` ("1") | Ratio of U's interval to V's, see below. |
| `--policy fixed\|adjustable` ("adjustable") | `DtPolicy` for the finer or straddling member. |
| `--accel none\|constant\|aitken\|secant\|steffensen\|iqn-ils\|iqn-imvj` ("aitken") | Coupling-iteration acceleration. An unknown name is refused with `E-EXAMPLE-ACCEL`. |
| `--t-end T` (50) | End time. |
| `--check` | Run a short internal self-consistency probe (see "What the example checks") and exit non-zero on failure. |
| `--output FILE` | Write a CSV snapshot (`i,j,x,y,u,v`) of the final state. |
| `--F`, `--k`, `--Du`, `--Dv`, `--L` | Override the reaction-diffusion parameters and domain size (defaults `F=0.0392, k=0.0649, Du=2e-5, Dv=1e-5, L=2.5`). |

`--ratio m` makes the window (U's own interval) `m*dt` and V steps at `dt` itself, so V is `m` times
finer than the window. `--ratio 2 --dt 0.5` means the window is `1.0` and V steps at `0.5`, not
`0.25`. `--ratio 7:13` means the window is 13 parts and V is 7 (incommensurate).

## The problem

On a periodic square of side `L`:

```
u_t = Du * Laplacian(u) - u*v^2 + F*(1 - u)
v_t = Dv * Laplacian(v) + u*v^2 - (F + k)*v
```

`u` is consumed by the autocatalytic term `u*v^2` and fed toward 1 at rate `F`; `v` is produced by
that same term and decays at rate `F + k`.

**Discretisation.** Cell-centred P0 finite volumes on an `N x N` grid, `h = L/N`, cell `(i,j)` centred
at `((i+0.5)h, (j+0.5)h)`. The periodic 5-point Laplacian (`gray_scott::periodic_laplacian`) is the
standard second-order stencil:

```
(Laplacian u)_ij = (u_E + u_W + u_N + u_S - 4 u_ij) / h^2
```

Each species advances by Crank-Nicolson in time, Newton-solved with `Eigen::SparseLU`
(`gray_scott::GrayScottSpeciesComponent<Species>` in `species_component.hpp`), the other species held
as a frozen input supplied by `fuse::WindowCoupler`. The Newton tolerance is `1e-11`, two to three
orders tighter than the smallest discretisation error measured here, so it does not contaminate an
order measurement.

**Monolithic reference.** `MonolithicGrayScott` (`monolithic_reference.hpp`) is a single-rate,
fully-coupled Newton solve of both species' `2N^2` unknowns in one system. It needs the reaction
term's cross derivatives that the partitioned components never use. Its Jacobian solve goes through
`fuse_examples::SparseLuBackend`, which is `Eigen::SparseLU` unless the library is built with
`FUSE_ENABLE_CUSPARSE`, in which case it is `cusolverRf`.

**Coupling.** U is always the window-definer: its own interval is `WindowCouplerConfig::window_length`,
so it cannot straddle a window. V may take a finer, possibly non-dividing step. Each species receives
the other's mean over its own substep, as `fuse::WindowCoupler` computes it, iterated to convergence
within the window. Every configuration here keeps both intervals within the window and has only one
possible straddling member, so neither `E-RT-WINDOW-LONGSTEP` nor `E-RT-WINDOW-STRADDLE` fires. See
[](coupling-windows.md) for what those mean.

## How the graph is authored

There is no `CouplingGraph` here: the example uses the window API directly. `main.cpp` builds one
`GrayScottSpeciesComponent` per species and declares a `WindowMember` for each (U's interval is the
window, V's follows `--ratio` and `--policy`). It declares a `WindowChannel` for each direction
(`u_out`, `v_out`) with its own `Relative` convergence measure at the coupling tolerance, and sets the
acceleration from `--accel` in `WindowCouplerConfig`. It then calls `WindowCoupler::create` and
`run_window` once per window. The coupling-iteration tolerance is `1e-8` relative (`Args::coupling_tol`,
not exposed as a flag). See [](coupling-windows.md) for the API and [](convergence.md) for
convergence measures.

## Expected output

The default test command, exact digits can vary by platform:

```
Gray-Scott windowed coupling: n=24 ratio=2 policy=adjustable accel=aitken dt=0.1 t_end=4 windows=20 total_iterations=81
  u range [0.459793, 1], v range [4.67742e-70, 0.269348]
  --check: m=1 windowed vs monolithic max diff 3.039e-08 (OK)
```

The output is byte-identical at any OpenMP thread count; the species components declare
`DeterminismLevel::Bitwise`.

## What the example checks

`--check` runs a separate sub-run of 5 windows at `m = 1` (internally fixed, regardless of `--ratio`
and `--t-end`) and compares it with the monolithic reference. It fails if:

- the windowed and monolithic results differ by more than `100 * coupling_tol` (1e-6), or
- a concentration goes negative.

That bound is tight on purpose, so `--dt` matters. The reaction term is quadratic in the partner
(`u*v^2`). The partitioned scheme freezes the partner at its mean over the window and uses that same
value at both the old and new reaction evaluations, while the monolithic scheme uses the partner's
actual old and new values. Squaring does not commute with averaging, so there is a real mismatch of
order 2 in `dt`, over and above the iteration residual. At `--dt 0.1` it is already below the bound
(about 3e-08 above); at a much larger `--dt`, `--check` can fail without anything being wrong.

## Verification behind the example

The automated test suite measures the following, with the numbers you can expect:

- **Species solver alone**, driven with the other species frozen and a manufactured forcing: spatial
  order 2 (N in {8,16,32}) and temporal order 2 (N = 32, 4 to 32 steps), each measured order within
  [1.85, 2.15].
- **Multirate coupling.** Commensurate ratios (1, 2, 4) and the incommensurate 7:13, with both
  `DtPolicy` values for V, converge under `AccelKind::None`, `Aitken` and `IQN_ILS`. Total sweeps over
  4 windows at 7:13 are 9 (None), 13 (Aitken) and 12 (IQN-ILS). Plain Picard is not the worst because
  the `u*v^2` coupling is a weak perturbation of this diffusion-dominated problem at the tested `dt`,
  so Picard already converges in few sweeps and Aitken's relaxation can overshoot by one sweep.
- **Against the monolithic reference at ratio != 1** (2, 4 and 7:13): the state error against a fine
  monolithic reference falls at order 2 in `dt`, every refinement step within [1.9, 2.1].
- **Mass exchange.** `GrayScottSpeciesComponent::coupling_flux()` accumulates each species' own
  Crank-Nicolson integral of the autocatalytic term (`-u*v^2` for U, `+u*v^2` for V) using its own
  frozen partner mean. The two means are different means of different trajectories, so what U removes
  and what V gains do not cancel exactly. The mismatch is not zero at the iteration tolerance; it is a
  discretisation error that falls at order 2 in `dt`. The flux is part of each species' checkpoint
  (`save_state`/`restore_state`), so repeated coupling sweeps and straddle replays do not
  double-count it.
- **Pattern formation.** With Pearson's "lambda" parameters (`F=0.0392, k=0.0649, Du=2e-5, Dv=1e-5`,
  domain 2.5) on a 48x48 grid seeded with a small central patch (u=0.5, v=0.25 on a 4x4 block) in an
  otherwise uniform background (u=1, v=0), the variance of `v` rises above `1e-4` by `t = 82` and
  settles at about `2.47e-04` from `t ~ 302` on: a single stable spot at this size and resolution,
  with `u` and `v` staying in `[0, 1]`. Spot-sustaining behaviour is resolution-sensitive: on 16x16 and
  24x24 grids the same seed decays back to the uniform state.

Tolerances used: Newton `1e-11`; coupling iteration `1e-9` relative in the tests and `1e-8` in the
example.

## What to try next

- Vary `--ratio` (`1`, `2`, `4`, `7:13`) with `--check` off and compare `total_iterations` in the
  header line. The coupling still converges at the incommensurate ratio.
- Vary `--accel`. On this problem plain `none` is already competitive, so do not expect Aitken or
  IQN to win by a large factor; see the sweep counts above.
- Try `--policy fixed` against `adjustable` at a non-dividing ratio to see the two `DtPolicy` choices
  for the straddling member.
- Raise `--dt` with `--check` on. The order-2 mismatch against the monolithic reference grows with
  `dt` and can eventually exceed the tight bound, without anything being wrong with the coupling.
- Run a longer pattern-forming case with `--n 48 --output snapshot.csv` and plot `v` from the CSV.
  The pattern is resolution-sensitive; very coarse grids decay to the uniform state.

## See also

- [](coupling-windows.md): the window mechanism this model is built on.
- [](convergence.md): convergence measures and acceleration.
- [Brusselator](brusselator.md): a related multirate model on two independent non-nested meshes.
- [FSI channel](fsi-channel.md): another Eigen-backend example.
