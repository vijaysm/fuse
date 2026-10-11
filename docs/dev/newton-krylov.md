# The Newton-Krylov seam (S7.1)

This page documents the residual/linearisation seam a native Jacobian-free Newton-Krylov
coupler (S7.2, a separate track) drives. S7.1 gates no correctness result of its own: it is
the interface the solver will call, proved by the capability-rejection tests firing before
any Newton code exists and by the fixed-point residual computing the right answer on a known
coupled problem. The solver itself -- MFFD Jacobian action, right-preconditioned GMRES,
Eisenstat-Walker forcing, line search, convergence/divergence tests, the lagged and
block-Jacobi preconditioners -- is appended to this page by that track, not here.

## 1. Two questions a component may additionally answer

Every `fuse::Component` answers `advance()`, `save_state()`/`restore_state()` and
`capabilities()` (`fuse/runtime/component.hpp`). A component that is also willing to
participate in a Newton-solved iteration region answers two more, through a SEPARATE
interface, `fuse::ImplicitComponent`, reached through `Component::implicit()`:

- **The residual.** `evaluate_residual(ctx, interface_in, residual_out)`: this component's own
  contribution to the coupled residual at a frozen time level, given the interface values it
  `Requires` inside the region, *without advancing its state*. Must be a pure function of
  (restored state, inputs) -- the same purity `advance()` itself must have inside an ordinary
  iteration region -- and must leave the component bit-identical to how it found it; a test
  (`test/runtime/implicit_component_test.cpp`) proves this by comparing state before and
  after the call.
- **The linearisation.** `linearization()` returns a `fuse::LinearizationToken*` (or
  `nullptr`) for the point of the most recent `evaluate_residual()` call: `apply_jacobian`,
  `apply_jacobian_transpose`, and `apply_preconditioner` -- this component's own
  physics-based preconditioner for its diagonal block, the thing a generic algebraic
  preconditioner cannot know and a component author can. All three act on this component's
  OWN block of the stacked interface vector, `cols()` long in, `rows()` long out, never on
  the whole region's vector -- the same duck-typed shape `fuse::linalg::MatrixFreeOperator`
  already uses for a single operator, applied once per region member.

**Why a query, not a second base class.** `Component::implicit()` mirrors the existing
`is_externally_driven()` precedent: a nullable virtual query, not multiple inheritance from
`ImplicitComponent`. Most components never implement it, and a second base would force every
one of them to override two pure virtuals with do-nothing bodies, or leave `Component` itself
uninstantiable. A query costs one vtable slot and one branch and needs no `dynamic_cast`.

## 2. The capability bits

`fuse::kCapabilityResidual` (bit 4) and `fuse::kCapabilityLinearization` (bit 5),
`fuse/model/capabilities.hpp`, narrow `kCapabilityReservedMask` from bits 4-63 to bits 6-63.
A component sets `kCapabilityResidual` in `capabilities()` when `implicit()` returns
non-null and `evaluate_residual()` can be called; it sets `kCapabilityLinearization` when
`linearization()` can return a usable token once a residual has been asked for. The two are
independent: a component may answer the residual question without ever linearising it, in
which case the Newton coupler falls back to a purely matrix-free directional derivative for
that component's own block.

These bits are **checked, not trusted**, the same pattern `kCapabilityRollback` already
follows for `supports_rollback()`: a Newton region's own validation reads the bit, not
`implicit()`'s return value, so the refusal below fires at `initialize()` time regardless of
whether the author's `implicit()` override and `capabilities()` override actually agree.

## 3. The model-side requirement, and the two refusals

`fuse::IterationRegionDecl` (`fuse/model/coupling_graph.hpp`) carries two booleans,
`requires_residual_capability` and `requires_linearization_capability`, both defaulting to
`false`. They are fields on the declaration, not a new `fuse::AccelKind` enumerator: the
Newton solver itself (S7.2) adds its own `AccelKind` value and sets these flags when that
value is chosen, which lets this track's own refusal be tested before any Newton code exists.

`fuse::verify_newton_capability_support()` (declared in `fuse/ir/lowering.hpp`, implemented
in `src/runtime/checkpoint_support.cpp`, alongside `verify_rollback_support()` -- a
`ComponentSet`-aware runtime check, the IR layer's own layering rule keeps it out of `ir`)
checks every member of the region's derived checkpoint set against the two bits and refuses
**`E-NEWTON-CAPABILITY`**, naming the component, when a required bit is unset. It is called
from `verify_iteration_region()`'s own single call site (`Runner::Impl::resolve_regions()`,
`src/runtime/iteration.cpp`) alongside the existing rollback and convergence checks, so a
Newton region's capability gap is reported at `initialize()`, in the same diagnostics pass as
every other iteration-region refusal -- never discovered only once the solver tries to call a
missing interface.

A Newton region's existing members still go through the UNCHANGED determinism check
(`verify_rollback_support()`): a component that declares `DeterminismLevel::Nondeterministic`
is refused with `E-ITER-NONDETERMINISTIC` whether or not Newton is selected for the region.
Newton gains no separate, weaker determinism rule of its own -- confirmed directly by
`test/model/newton_capability_test.cpp`'s
`NondeterministicComponentInNewtonRegionUsesTheExistingRefusal`.

The second refusal, **`E-NEWTON-SHAPE`**, is a pure function over already-known sizes,
`fuse::verify_linearization_shape()` (`fuse/runtime/component.hpp`): whether a
`LinearizationToken`'s own `rows()`/`cols()` agree with the block sizes a region's
interface-vector assembly derived for the component that returned it. It needs no graph, no
`ComponentSet` and no running region, which is why it is a free function rather than folded
into the model-level check above -- the Newton solver (S7.2) calls it once it has both a
token and the block sizes to check it against; this track's own test calls it directly
against a hand-built token of the wrong shape.

## 4. The fixed-point residual over the Runner

`Runner::region_fixed_point_residual(region_index, components)` (`fuse/runtime/runner.hpp`)
returns a `fuse::FixedPointResidual` (`fuse/runtime/fixed_point_residual.hpp`): a
`std::function<Result<void>(std::span<const double>, std::span<double>)> evaluate` computing
`F(x) = x - G(x)` over the region's stacked interface vector, plus `n` and the per-block
`offsets`/`lengths`/`channel_names` a caller needs to scatter a global Newton iterate into
each member's own slice. `G` is one Picard sweep of the region's body: deliver `x` into every
member's own `Requires` storage, re-advance every checkpointed member over the region's own
interval, and read back what the region's dataflow produces at each block's residual arc --
the same thing `Runner::Impl::run_region()`'s own sweep loop computes for the PREVIOUS
sweep's delivered value, except available for an arbitrary `x`, which a Newton line search or
a Krylov iteration chooses, rather than only the sequence Picard itself would have chosen.

**The no-side-effect contract.** Evaluating `F` leaves every member's own state, and every
coupler-owned buffer the sweep touches, exactly as it found it: the region may be re-entered
afterward, by another call to `evaluate` or by the Runner's own Picard loop, and reaches the
same fixed point it would have without the evaluation, bit for bit
(`test/runtime/fixed_point_residual_test.cpp`'s
`EvaluatingFDoesNotPerturbTheRunnersOwnPicardIteration`). This is not merely a documented
intent: the first implementation of `Runner::Impl::evaluate_region_residual()` restored only
component state (`save_state()`/`restore_state()`) and the region's loop-carried coupler
buffers (`RegionRuntime::carried_arcs`) -- everything `derive_checkpoint_set()`
(`src/ir/checkpoint.cpp`) proves survives sweep to sweep inside an ORDINARY Picard loop -- and
that test failed: delivering an arbitrary `x` directly into each block's feed arc (component
`Requires` storage) is not something an ordinary sweep ever does with a caller-chosen value,
so the derived checkpoint set does not cover it. The fix additionally snapshots and restores
every block's own feed and value arc; `build-evidence/mutations/runtime.toml`'s
`fixed-point-residual-feed-value-arc-not-restored` row reproduces the defect on demand.

**Scope.** `evaluate_region_residual()` refuses `E-NEWTON-UNSUPPORTED-REGION` for a
window-scoped region, a region with more than one cross-rank participant, or a region with an
externally driven member: none of those have a Picard sweep this seam can run standalone
without reimplementing `run_region()`'s own window/parallel machinery, which is this track's
explicit non-goal. Lifting either is the JFNK track's own work. `Runner::
region_fixed_point_residual()` additionally refuses `E-RT-REGION` for an out-of-range region
index or before a successful `initialize()`.

Verified end to end against `examples/dn_heat`'s real Dirichlet-Neumann pair
(`dn::DirichletSlab`/`dn::NeumannSlab`): with the example's own constants (`kL1 = kL2 = kK2 =
1`), `F` is affine with a per-station 2x2 Jacobian of exactly `{{1, r_i}, {-1, 1}}`, `r_i`
the documented contraction factor `k1_at(i,n) * kL2 / (kK2 * kL1)`; measured by one-sided
finite difference at an arbitrary nonzero step (exact to round-off for a genuinely affine map,
not a truncation-error estimate), and the off-diagonal product recovers `r_i` directly -- the
loop gain around dirichlet -> neumann -> dirichlet that the contraction factor names.

## 5. Mapping to the participant ABI

No ABI entry is added by this track -- the exposure is a later step -- but nothing in the
C++ shape above blocks it. Per `docs/dev/abi-design.md`'s versioning rule, a future
`fuse_evaluate_residual`/`fuse_apply_jacobian`/`fuse_apply_preconditioner` entry would each
take a `struct_size`-headed argument struct (the same convention every existing ABI entry
uses: a `uint32_t struct_size` the caller sets to `sizeof` of the struct it compiled against,
so a field can be appended later without breaking an older caller) carrying the raw
`const double*`/`double*`/length triples `std::span` already wraps on the C++ side, plus the
`StepContext` fields already passed by value through `fuse_step_context` today. The capability
bits cross the descriptor the same way `kCapabilityRollback` and `kCapabilityHiddenState`
already do -- no version bump, since an old binary reading a newer one simply sees the new
bits unset, which is the honest "does not support Newton" answer.

## 6. S7.2: the native Jacobian-free Newton-Krylov solver

This section is this track's own work: `fuse::num::solve()` (`fuse/num/newton_krylov.hpp`,
`src/num/newton_krylov.cpp`), a matrix-free, inexact Newton-Krylov iteration on the residual
`F` the seam above (§4) produces, with `fuse::num::line_search` (`fuse/num/line_search.hpp`,
`src/num/line_search.cpp`) as its own pluggable strategy module. It replicates exactly six
mechanisms from PETSc's `SNESNEWTONLS` -- the ones that make JFNK robust in practice without
pulling in the rest of SNES's own machinery (trust region, NGMRES, nonlinear preconditioning;
none of this track's test problems needed any of those, and none was pre-built against a need
that was never measured).

### 6.1 The six mechanisms, and their SNES counterpart

| # | This project | PETSc SNES counterpart | Option name here |
|---|---|---|---|
| 1 | `fuse::num::MffdJacobianAction` -- matrix-free Jacobian action by one-sided finite difference, `J(x) v ~= (F(x+hv)-F(x))/h`, `h` from the Walker-Pernice rule `h = error_rel * sqrt(1+\|\|x\|\|) / max(\|\|v\|\|, min_direction_norm)` (Pernice & Walker, "NITSOL", SIAM J. Sci. Comput. 19, 1998) | `MATMFFD_WP` (the default `MatCreateSNESMF` differencing routine; WP's own `wp.c` has NO `umin` parameter at all -- that belongs to the DIFFERENT routine `MATMFFD_DS`/Dennis-Schnabel, a distinct formula this project does not implement; WP's own tunable besides `error_rel` is `-mat_mffd_compute_normu`, unrelated to any floor) | `NewtonKrylovOptions::mffd` (`MffdOptions::error_rel`, `::min_direction_norm`) |
| 2 | `fuse::linalg::gmres` over a `MatrixFreeOperator` wrapping (1), right-preconditioned by (6) or the identity | `KSPGMRES` with the matrix-free `Mat` from `MatCreateSNESMF`, right preconditioning (`-ksp_pc_side right`, SNES's own default for a matrix-free Jacobian) | `NewtonKrylovOptions::gmres` (`restart`, `max_iterations`) |
| 3 | `fuse::num::eisenstat_walker_eta` -- Eisenstat-Walker forcing, choice 1 or 2, with the over-solving safeguard and a cap/floor | `SNESKSPSetUseEW` / `-snes_ksp_ew`, choices 1/2/constant (`-snes_ksp_ew_version`) | `NewtonKrylovOptions::forcing` (`EisenstatWalkerOptions`) |
| 4 | `fuse::num::backtracking_line_search` / `critical_point_line_search` / `basic_line_search` | `SNESLINESEARCHBT` / `SNESLINESEARCHCP` / `SNESLINESEARCHBASIC` | `NewtonKrylovOptions::line_search` (`LineSearchOptions`, `LineSearchKind`) |
| 5 | `fuse::num::NewtonStatus` -- named convergence/divergence codes, never a bare bool | `SNESConvergedReason` | `NewtonKrylovOptions::{atol,rtol,stol,divergence_tolerance,max_nonlinear_iterations,max_function_evaluations}` |
| 6 | `fuse::num::BlockJacobiPreconditioner`, rebuilt every `preconditioner_lag` Newton steps | `-snes_lag_jacobian <k>` (the rebuild interval) composed with a user `PCSHELL`/physics-based `PC` (SNES has no built-in per-component block-Jacobi "physics-based" preconditioner; a caller assembles one from each component's own linearisation the same way `SNESSetJacobian`'s own preconditioning matrix callback would) | `NewtonKrylovOptions::preconditioner_lag`, `BlockJacobiPreconditioner::from_blocks` |

### 6.2 Defaults

Every numeric default in `NewtonKrylovOptions`/`EisenstatWalkerOptions`/`LineSearchOptions`
is SNES's own default for the matching option (`-snes_atol 1e-50`, `-snes_rtol 1e-8`,
`-snes_stol 1e-8`, `-snes_divergence_tolerance 1e4`, `-snes_max_it 50`,
`-snes_max_funcs 10000`; `-snes_ksp_ew_version 2`, `gamma=0.9`, `alpha=2.0`,
`-snes_ksp_ew_rtol0 0.3`, `-snes_ksp_ew_rtolmax 0.9`; `-snes_linesearch_alpha 1e-4`,
`-snes_linesearch_minlambda 1e-12`; `-snes_lag_jacobian 1`) -- including
`EisenstatWalkerOptions::eta_min`/`NewtonConfig::eta_min`, defaulted to `0.0` (CORRECTED: an
earlier draft of this page defaulted it to `1e-4` as a judgement call; SNES's own forcing
term has no nonzero floor at all -- it can reach an exact linear solve, `eta = 0` -- so `0.0`
is what makes this default actually BE SNES's default, with the option itself still present
for a caller who wants a nonzero floor the task's own brief named as a possibility).

`MffdOptions::min_direction_norm` is NOT an SNES default and does not claim to be one
(CORRECTED: an earlier draft of this page named this field `umin` and presented its default
as PETSc's own `-mat_mffd_umin`; that option belongs to `MATMFFD_DS`, a differencing routine
this project does not implement, with its own, different formula -- `MATMFFD_WP`'s own
`wp.c` has no `umin` parameter at all). It is a pure zero-division safeguard on `\|\|v\|\|`,
the Walker-Pernice step's own divisor, reached ROUTINELY rather than only in a corner case:
`fuse::linalg::gmres()` evaluates its own first matrix-vector product at the initial guess
`d = 0`, so `apply()`'s first call on every Newton step sees `v = 0` exactly, and without a
floor `h` would be `+inf` (a finite numerator over an exact-zero denominator) and `x + h*v`
would be IEEE 754's NaN (`inf * 0`). Defaulted to `std::numeric_limits<double>::min() * 1e8`
-- the smallest NORMAL double times a margin, chosen only to stay away from zero and from
the denormal range. The value does not affect CORRECTNESS at `v = 0`: the true action there
is exactly the zero vector regardless of which positive `h` the floor produces, since
`x + h*0 == x` identically, proved directly by `Mffd.ZeroDirectionIsSafeAndReturnsExactlyZero`
(test/num/newton_krylov_test.cpp).

### 6.3 What integrates where

`fuse::num::solve()` depends on `fuse::linalg` (`DistVector`, `gmres`, `MatrixFreeOperator`,
the `Preconditioner` concept) and on nothing above it: no `fuse::runtime`, no `Component`, no
`LinearizationToken` (`src/CMakeLists.txt`'s own dependency graph keeps `fuse::num` below
`fuse::runtime`). `BlockJacobiPreconditioner::PreconditionerBlock::apply` is the SAME
span-based `(std::span<const double> r, std::span<double> z) -> Result<void>` shape
`LinearizationToken::apply_preconditioner` already uses (`fuse/runtime/component.hpp`), so a
caller in `fuse::runtime` -- the graph-level wiring a later step of this track adds -- adapts
a real token into one of these blocks with a one-line lambda; `fuse::num` itself never needs
to see the token type. The six mechanisms, their tests, and the seven hand-proven mutation
rows (`build-evidence/mutations/num.toml`) are this file's own deliverable; wiring the
solver into `Runner::Impl::run_region()`'s sweep loop and the dn_heat/Brusselator measured
results are later sections of this same track, appended as they land. §7 below covers the
graph-selection half of this same step.

## 7. Selectable from the graph

`AccelKind` gains `Newton` rather than a sibling `IterationSolver` enum on
`IterationRegionDecl`, decided from `AccelKind`'s own declaration site
(`fuse/model/iteration.hpp`): the header's comment on `AccelKind::Newton` states the
reasoning in full, repeated here because this is the place a reader comparing the two
options would look first. In short: this page's own §1 design page (S7.1) had already
anticipated exactly this choice -- "the Newton solver itself (S7.2) adds its own AccelKind
value and sets these flags when that value is chosen" -- so a second enum would have
contradicted a decision the seam's own track already made and documented; and every
consumer that already switches on `IterationRegionDecl::acceleration.kind` (the YAML
emitter/reader's `kAccelNames` table, `fuse::descriptor`'s binary encoding, `to_cstring`,
`make_accelerator`) only has to add ONE new case each, rather than learn a second, parallel
field that means "which solver" the same way `acceleration.kind` already does. The cost of
that choice -- Newton does not fit `Accelerator`'s own interface (`fuse::num::
make_accelerator()` refuses `AccelKind::Newton` with `E-ACCEL-KIND` and a message explaining
why, rather than silently misusing the quasi-Newton machinery) -- is confined to one
`case` arm, not a second enum's worth of plumbing.

**`NewtonConfig`** (`fuse/model/iteration.hpp`) mirrors `fuse::num::NewtonKrylovOptions`
field for field, the same "WHAT was asked for, not how it is computed" split `AccelConfig`
already keeps between this header and `fuse::num::Accelerator`. Every numeric default is
SNES's own (§6.2's table). Read only when `acceleration.kind == AccelKind::Newton`.

**The auto-trigger.** `IterationBuilder::accelerate()` and the new `IterationBuilder::
newton()` both set `requires_residual_capability = true` the moment `AccelKind::Newton` is
selected (and `requires_linearization_capability` too, when
`NewtonConfig::use_block_jacobi_preconditioner` is set) -- a graph author cannot select
Newton and forget the capability gate S7.1 built. The YAML reader (`fuse::from_yaml()`)
applies the IDENTICAL auto-trigger after parsing a region's `acceleration`/`newton_config`
blocks, so a hand-written YAML file gets the same protection a C++-authored graph gets from
the builder, proven by `build-evidence/mutations/model.toml`'s own
`newton-builder-accelerate-auto-trigger-removed` and
`newton-yaml-deserialize-auto-trigger-removed` rows (each one disables one side of the
trigger and is caught only by `test/runtime/newton_region_test.cpp` or
`test/model/yaml_io_test.cpp` respectively, confirming the two are independent code paths
that could otherwise drift apart). `IterationBuilder::requires_residual()`/
`::requires_linearization()` -- the fluent setters S7.1's own brief named but left for this
track to add -- exist independently of Newton, for a region that wants the residual seam
without selecting Newton at all.

**The binary descriptor: `kGraphVersion` 11 -> 12.** `AccelKind::Newton` and `NewtonConfig`
now cross `include/fuse/model/descriptor.hpp`'s wire format too -- the gap the previous
revision of this section recorded as open is closed. Each iteration region record gains,
appended after its window-scope block (the same append-only placement every version bump
since v7 uses): the two capability bytes
(`IterationRegionDecl::requires_residual_capability`/`::requires_linearization_capability`)
and `NewtonConfig`'s own 25 fields in full, in the struct's declared order -- two enum bytes
(`line_search_kind`, `forcing_choice`), 15 doubles, 5 signed 32-bit words, 2 unsigned 64-bit
counts, and a final presence byte for `use_block_jacobi_preconditioner`. Both are written
UNCONDITIONALLY, matching the existing unconditional-write convention `acceleration` itself
already follows -- so `requires_residual_capability`/`::requires_linearization_capability`
survive a round trip even when set independently of Newton selection, via
`IterationBuilder::requires_residual()`/`::requires_linearization()`. The accelerator-kind
bound check `deserialize()` applies widens from `AccelKind::IQN_IMVJ` to `AccelKind::Newton`
(ordinal 7); a v11 reader handed a v12 buffer would read the first region's
residual-capability byte as the start of whatever comes next, so v12 bytes are refused by a
v11 reader (and v11 bytes, in turn, by this v12 reader's own version check) -- exactly the
class of silent misread every prior bump exists to prevent, which the version check turns
into `E-DESC-VERSION` instead. `test/golden/waist-v12.bin` is the new frozen fixture
(`freeze_fixture()` gained a second iteration region selecting `AccelKind::Newton` with every
`NewtonConfig` member away from its default, for the same reason every other optional member
in that fixture is pinned non-default: a member that round-trips to its own default proves
nothing about whether it was actually read back or silently dropped); `waist-v11.bin` moved
to the superseded, refused list alongside v1 and v4-v10.
`build-evidence/mutations/model.toml`'s `descriptor-accel-kind-bound-not-widened` row
hand-proves the bound widening itself is load-bearing: reverting it to the pre-v12 bound makes
`Descriptor.RoundTripPreservesEveryNonDefaultMember` fail with `E-DESC-ENUM` ("unknown
acceleration kind 7"), because the fixture's own Newton region can no longer be read back.

## 8. dn_heat by Newton

### 8.1 The runtime wiring

A Newton-selected region bypasses `run_region()`'s own Picard sweep loop entirely, rather than
replacing just the quasi-Newton update inside it: `Runner::Impl::run_region_newton()`
(src/runtime/iteration.cpp) hands the WHOLE solve to `fuse::num::solve()` in one call, over an
`evaluate` built from `Runner::Impl::evaluate_region_residual()` (the SAME side-effect-free
`F(x) = x - G(x)` primitive S7.1 built, wrapped into `fuse::num::ResidualFn`'s `DistVector`
calling convention via `linalg::make_serial_exchanger()`/`make_serial_reducer()` -- Newton is
restricted to a single cross-rank participant, so no real communication is ever injected).
`resolve_regions()` never calls `make_accelerator()` for a Newton-selected region -- it has no
`case AccelKind::Newton` and would refuse with `E-ACCEL-KIND` -- and instead eagerly validates
the SAME three conditions `evaluate_region_residual()` itself checks lazily (non-window, single
participant, no externally driven member) through one shared function,
`check_region_supports_newton()`, so a Newton-selected region this Runner cannot yet solve is
refused with `E-NEWTON-UNSUPPORTED-REGION` at `initialize()` rather than at the first period.

**The verification sweep.** Once `fuse::num::solve()` returns, `evaluate_region_residual()`
gains a `commit` parameter (default `false`, S7.1's own original meaning: restore everything
this call touched, so repeated evaluations are side-effect free). `run_region_newton()` calls
it exactly once more, with `commit = true`, at the converged iterate -- the SAME deliver/
advance/execute-body sequence every other evaluation performed, except the restore at the end
is skipped, so Newton's own converged point becomes the region's real, persisted state (the
components' registered storage, the loop-carried buffers) instead of one more probe. A FAILURE
during that commit still restores regardless of the flag: nothing here is allowed to leave a
partial advance in place.

**Non-convergence** is reported the same way the Picard path's own `E-RT-NOCONVERGE` is, naming
the region and, in addition, Newton's own `NewtonStatus` code
(`build-evidence/mutations/runtime.toml`'s `newton-nonconvergence-not-reported` row proves this
check is load-bearing). The block-Jacobi preconditioner hook (Step 1's own mechanism) is wired
as far as `NewtonConfig::use_block_jacobi_preconditioner` -- neither dn_heat test in this track
sets it, so `precond_apply`/`refresh_precond` both stay `nullptr` (the plain, unpreconditioned
matrix-free iteration); composing one from each region member's own `LinearizationToken` across
blocks that are not one-per-component, in general, is open scope, recorded in this track's own
report.

### 8.2 Measured: one Newton step is not what SNES's own defaults give

The task's own assumption -- "Newton converges in ONE Newton step" on dn_heat's affine map --
does not hold under SNES's own default Eisenstat-Walker forcing (`eta_initial = 0.3`). MEASURED
(`NewtonRegion.ConvergesInThreeNewtonStepsOnTheAffineProblemUnderEwForcing`,
test/runtime/newton_region_test.cpp): `eta_0 = 0.3` deliberately stops the FIRST linear solve at
30% relative accuracy (GMRES needs only 2 of a possible many iterations), so the first Newton
direction is not the exact affine root even though the Jacobian itself is exact; three total
Newton steps are needed (`||F||`: 765.78 -> 68.77 -> 0.288 -> 2.6e-8). This is textbook inexact-
Newton behaviour -- trading a cheap, loose early linear solve against one or two more nonlinear
steps overall -- not a defect, and it is exactly what Eisenstat & Walker's own forcing exists to
do. A `NewtonConfig` with a tighter/near-exact constant forcing term would converge in fewer
steps, at the cost of more GMRES work on each one; that trade is not this track's own default.

### 8.3 The nonlinear variant, and the quadratic rate measured

`DirichletSlab` (examples/dn_heat/dn_component.hpp) gained a nonlinear conductivity,
`k1_eff(i, T) = k1_at(i, n) * (1 + beta * (T - t_ref) / t_ref)`, `T` the local wall temperature
being solved for (`u_in_[i]`, or `interface_in[i]` inside `evaluate_residual()` -- the two
compute the identical formula, duplicated rather than shared, because `evaluate_residual()`'s
own contract forbids the mutation `advance()` is free to do). `beta = 0.0`, every existing call
site's own default, makes `k1_eff` reduce to `k1_at(i, n) * 1.0` EXACTLY -- measured directly,
`Dn.NonlinearAtBetaZeroIsBitIdenticalToTheAffineMap` -- so every mode this project already ships
(picard, aitken, iqn-ils, iqn+qr2, severed) is unaffected.

MEASURED (`NewtonRegion.QuadraticRateMeasuredOnTheNonlinearProblem`), at `beta = 0.1` (a 10%
conductivity swing, mild enough that Picard's own per-station contraction stays below 1 even at
the strongest strip: `k1_eff`'s own multiplier stays within roughly [0.90, 1.03] over dn_heat's
own working temperatures, so the worst case is `0.96 * 1.03 < 1`) and a near-exact forcing term
(pinned tight specifically to isolate Newton's own order from Eisenstat-Walker's own deliberate
under-solving, §8.2 above): `||F||` = 7.284e+02, 8.823e+01, 7.762e-01, 5.659e-05. Fitting
`q_k = log(||F_{k+1}||/||F_k||) / log(||F_k||/||F_{k-1}||)` over the two triples both still above
a `1e-10 * ||F_0||` round-off floor: `q_1 ~= 2.24`, `q_2 ~= 2.01` -- **order 2.0 +- 0.24**, the
spread this track measured rather than assumed. (The run stops at `rtol = 1e-4` rather than
SNES's own default `1e-8`: reaching `1e-8` from here needs a near-machine-precision LAST linear
solve this unpreconditioned, matrix-free iteration cannot complete within a sane GMRES budget --
not a defect in Eisenstat-Walker or GMRES, the ordinary cost of the last couple of orders of
magnitude with no preconditioner, recorded as an open question: a physics-based block-Jacobi
preconditioner is the mechanism that would make that last solve cheap.)

### 8.3b A bookkeeping-only defect found and fixed while measuring 8.3

Measuring §8.3 surfaced a genuine, if numerically harmless, defect in
`run_region_newton()`'s very first version: `step_period()`'s own top comment
(src/runtime/runner.cpp) states that by the time control reaches ANY region,
every participant has already been advanced once by the generic per-period
tick walk, and `checkpoint_regions()` saves BEFORE that tick specifically so
a Picard sweep loop's own repeated `restore_state()` calls -- which all
target `checkpoint_regions()`'s own save point -- undo it along with every
sweep's own advance. `evaluate_region_residual()`'s own save/restore is
self-referential: every call restores to what IT most recently saved, never
to `checkpoint_regions()`'s earlier point. A Newton-selected region therefore
never undid that first generic tick, and the verification sweep's one
committed advance landed on top of it: two real advances baked into
persisted state per period instead of one. MEASURED directly: `dn::
DirichletSlab::history()`/`dn::NeumannSlab::history()` (private, checkpointed
accumulators the project's own contract -- dn_component.hpp -- states read
"P exactly" after P coupling cycles) read 40 after 20 periods, not 20.

Numerically harmless: `q_`/`u_` are overwritten from scratch by the LAST
(committed) `evaluate_region_residual()` call regardless of what the extra,
un-rolled-back tick left them at, and the L2-error and beta=0 bit-identical
gates already in this track's own tests did not move. But `history()`'s own
contract is a real invariant examples/dn_heat/main.cpp's own picard/aitken/
iqn-ils gates already read for every other row, and a Newton-selected
region silently breaking it is exactly the kind of defect a byte-level
rollback/bookkeeping discipline exists to catch. Fixed by adding one
`restore_state()` call over `R.component_slots`, right after seeding `x`
(already a COPY at that point, so the restore cannot perturb Newton's own
seed) and before calling `fuse::num::solve()` -- `NewtonRegion.
HistoryAccumulatesExactlyOncePerPeriodNotTwice` (test/runtime/
newton_region_test.cpp) is the regression test, and `build-evidence/
mutations/dn-heat.toml`'s own `dn-heat-newton-generic-tick-restore-removed`
row hand-proves the fix is load-bearing.

### 8.4 The line search, shown to engage and recover on dn_heat itself

Step 1's own `NewtonKrylovLineSearch.BtEngagesAndRecoversOnAPoorInitialGuess`
(test/num/newton_krylov_test.cpp) proves the MECHANISM on a synthetic problem designed to need
it (`atan`'s own saturating tail). This track's own
`NewtonRegion.LineSearchEngagesAndRecoversFromADeliberatelyPoorGuess`
(test/runtime/newton_region_test.cpp) shows the mechanism is correctly REACHED through the real
runtime wiring on dn_heat itself: `beta = 2.0` and `NeumannSlab`'s own initial interface
temperature overridden to 2000 K (dn_heat's own physical range is a few hundred K) is far enough
from the root that a full Newton step overshoots. MEASURED (a trace printed by hand and removed
once read, reproduced twice): the first step (`k = 0`) is a full step (`lambda = 1`,
`||F||` 765.78 -> 193.05); the SECOND step's own full Newton step is rejected by Armijo and BT
backtracks to `lambda = 0.025` after 4 trials (`||F||` -> 189.01 -- genuine recovery, not a free
ride); every step after that takes a full step again, and the iteration converges in 7 total
steps (`||F||` -> 7.98e-4). `IterationStats` (fuse/runtime/runner.hpp, outside this track's own
file ownership) carries `residual_history` but not a per-step accepted lambda, so the lambda
values themselves are recorded here as a measurement rather than asserted on directly through
the public `Runner` API; what the test DOES assert is that the run converges (a line-search
failure the solver cannot recover from reports `NewtonStatus::DivergedLineSearch` instead,
caught the same way the non-convergence test above catches `DivergedMaxIterations`) and needs
more than the affine problem's own one-Newton-step case.
