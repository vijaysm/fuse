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

## 9. Newton over a coupling window -- design (Stage A), IMPLEMENTED (Stage B, 2026-10-11)

This section began as the WindowNewton track's own Stage A: a design,
bounded by measurement, for lifting `check_region_supports_newton()`'s
`E-NEWTON-UNSUPPORTED-REGION` refusal on a window-scoped region
(`fixed_point_residual.cpp`'s own header comment: "Lifting either is
follow-on work for the JFNK track (S7.2), not this seam"; `newton_krylov.hpp`'s
`BlockJacobiPreconditioner` comment names the identical boundary: "a
cross-rank block-Jacobi preconditioner is a later step's work, once the
fixed-point residual itself is lifted to a window-scoped, multi-participant
region"). Sec. 9.1-9.8 below are Stage A's own text, UNCHANGED (the lead's
own instruction: "keep measurements") -- every claim in them was either a
line-numbered citation of the window path as it stood before Stage B, or a
number measured against the UNMODIFIED Brusselator example at Stage A time.
**Stage B (2026-10-11) built the design Sec. 9.1-9.7 describes**, against
the SAME checkout (`bbe1a1a` at Stage A time); Sec. 9.9 below is the new,
Stage-B-only section recording what was actually built, the four open
questions' own resolutions, and the measurements taken against the NOW
Newton-capable Brusselator example.

### 9.1 What `x` is

The ordinary-region seam's `x` is the stacked FEED values of every
`RegionBlock` (`run_region_newton()`'s own seed loop,
`iteration.cpp:1686-1688`: `num::copy(slots[b.feed_arc].data, x.data() +
b.offset, b.length)`). A window-scoped region's accelerator already
consumes the SAME shape of vector -- `R.accel->update(R.x_old.data(),
R.x_new.data(), R.x_out.data(), R.n, k)` (`iteration.cpp:2327-2328`) runs
over the full, per-block concatenation `R.blocks` -- so `x` for a
window-Newton region is that same concatenation, with one exclusion:

**Layout.** One contiguous slice per included `RegionBlock`, each slice
itself `(element, sample)`-major: `b.length` doubles, where sample 0 is the
window-START value and samples `1..window_export_capacity-1` are the
per-substep END values (`RegionBlock::window_export_capacity`'s own comment,
`runner_impl.hpp:363-377`; the write pattern is `advance_window()`'s own
`out[e * cfg.capacity + s]`, `iteration.cpp:3217-3220,3309,3361`). This is
`WindowCoupler`'s own `accel_members`/`member_base` layout exactly
(`window_coupler.cpp:775-789`: `at(e, s) = base + e * n_sub + s`), modulo
the ordinary driver's own exclusion of the window-start sample from its
`accel_total` tally (`window_coupler.cpp:738`: `steps[i].size()`, no `+1`)
-- the Runner's own `R.x_old`/`R.n` keeps sample 0 in the vector because
`RegionBlock::length`/`offset` are the FULL arc (`RegionBlock::
window_export_capacity`'s own comment: "never touching `length`/`offset`
themselves, which the accelerator ... must keep seeing exactly what
`advance_window()` wrote"). A Newton `x` should follow `R.x_old`'s own
convention (full per-block length, sample 0 included): `evaluate_region_
residual()`'s ordinary-region analogue delivers `x` wholesale into
`b.feed_arc` and reads `G(x)` wholesale back from `b.value_arc`, and
`advance_window()` reads sample 0 (`start_value`) as a SEPARATE, cached
quantity it never re-derives from the rest of the vector
(`iteration.cpp:3212-3220`) -- so the window-start sample is logically part
of `x` even though no part of `F(x) = x - G(x)`'s own RESIDUAL depends on
it (`run_region()`'s own residual-norm computation strips it,
`iteration.cpp:2087-2106`, for exactly that reason). Carrying it in `x`
costs nothing (it stays fixed under `advance_window()`'s own `first_sweep_
of_pass` convention, Sec. 9.3) and avoids inventing a second, shorter
vector shape nothing else in the Runner uses.

**Exclusion.** `R.blocks` today is NOT restricted to lagged transfers --
the `E-RT-WINDOW-ACCEL` refusal exists PRECISELY because the accelerator
has no such restriction and would therefore be wrong under GaussSeidel
(`iteration.cpp:1347-1373`: "R.blocks below is NOT YET restricted the same
way ... Running a real accelerator over that undifferentiated vector would
accelerate a quantity the method does not define an iterate history for").
A window-Newton region's `x` MUST NOT inherit that gap: it is built from
`R.blocks` restricted to the blocks whose `feed_arc` is a member of `R.
window_lagged_import_arcs` ("every LAGGED window_import arc this region
owns", `runner_impl.hpp:455-463`, populated at `iteration.cpp:550-552` from
`cfg.import_lagged`). This is exactly `WindowCoupler`'s own `accel_members`
restriction translated to the Runner's own bookkeeping:
`provides_unlagged_channel(i)` (`window_coupler.cpp:715-720`) excludes a
member that provides an UNLAGGED channel; `cfg.import_lagged` is the
CONSUMER-side mirror of the identical fact (`effective_channel_lag()`,
`fuse/model/coupling_graph.hpp`, is the one function both
`window_import_is_lagged()` at schedule time and `cfg.import_lagged` at
runtime read, per `WindowMemberConfig::import_lagged`'s own comment,
`runner_impl.hpp:760-767`) -- a block whose `feed_arc` is NOT in `R.window_
lagged_import_arcs` is precisely the GaussSeidel unlagged direction
`WindowCoupler` excludes, and under `WaveformIteration::Jacobi` EVERY
channel is forced lagged (`cfg.import_lagged`'s own comment: "`effective_
channel_lag()` ... forces lag=1 for every channel in a Jacobi window"), so
every block is included -- "every member is accelerated under Jacobi"
reproduced exactly.

**Measured length, Brusselator at its own defaults** (`--dt-u 0.002 --dt-v
0.003 --window 0.02`, `--nx-u 16 --ny-u 16`; both transfer fields live on
the U grid, `n_u = 256`, per `species_component.hpp`'s own "both channels
live on the U grid" comment and `main.cpp`'s `mv.length = n_u`):
`window_sample_capacity(window_length, interval, Adjustable)` = `ceil(window/
interval) + 1` (`coupling_graph.hpp:505-515`, no `DtPolicy::Fixed` `+1`
since both members are `Adjustable`, `main.cpp`'s own `mu.policy`/`mv.
policy`). U: `ceil(0.02/0.002)+1 = 11`. V: `ceil(0.02/0.003)+1 = ceil(6.667)
+1 = 8`. The Brusselator example's own graph is Jacobi by default (`add_
window_coupling()` is called with no `iteration_mode` override,
`brusselator_graph.hpp:112`'s own default `WaveformIteration::Jacobi`), so
BOTH blocks are lagged and both are included:

| block | consumer grid elements | provider capacity | block length |
|---|---|---|---|
| V->U (U's own `required_`) | 256 | 8 (V) | 2048 |
| U->V (V's own `required_`) | 256 | 11 (U) | 2816 |
| **total `n`** | | | **4864** |

### 9.2 What one evaluation must snapshot and restore

The ordinary seam's checkpoint set (`evaluate_region_residual()`,
`fixed_point_residual.cpp:102-134`) is: component state
(`save_state()`/`restore_state()`), `R.carried_arcs`, and every block's
`feed_arc`/`value_arc` -- added only after a measured failure
(`FixedPointResidual.EvaluatingFDoesNotPerturbTheRunnersOwnPicardIteration`)
showed the FIRST version (component state + carried arcs alone, the
`derive_checkpoint_set()`-proved sweep-to-sweep survivors) was insufficient
for an ARBITRARY caller-chosen `x`. A window sweep mutates strictly more
Runner-owned state than an ordinary one, none of it visible to
`derive_checkpoint_set()` (that analysis is about a component's own
checkpointed state and the region's declared `carried` buffers -- it has no
notion of `Runner::Impl`'s own per-node maps at all). Enumerated against
`advance_window()`'s own read/write sites:

| state | mutated by | covered today by component save/restore? | must the window-Newton seam snapshot/restore it? |
|---|---|---|---|
| component hidden state | `comp.advance(ctx)` | YES (`save_state()`/`restore_state()`, `component_slots`) | reuse as-is |
| `window_member_state[node].clock` | `st.clock = final_clock` every call, unconditionally (`iteration.cpp:3398`) | NO | YES -- read by `resolve_pending_window_straddles()` on the NEXT window instance (`st.clock` as `target_clock`, `iteration.cpp:2609`) and, within THIS pass, discarded in favour of the cached `window_pass_start_clock` (Sec. 9.3) -- but a non-committing probe must not leave a trial value sitting where the NEXT pass's straddle replay would read it |
| `.straddle_pending`/`.straddle_window_start`/`.straddle_window_end` | set when `steps.back().straddle` (`iteration.cpp:3407-3425`) | NO | YES, same reason -- read by `resolve_pending_window_straddles()` at the TOP of the NEXT window instance's pass (`iteration.cpp:2574`), before this region is ever re-entered |
| `.history` | **not** mutated inside `advance_window()` at all (explicit design: "intentionally NOT touched here, mid-sweep", `iteration.cpp:3377-3396`); mutated exactly once, by `run_region()`'s own post-loop refresh, from the LIVE `window_import` buffer, after the Picard loop exits (`iteration.cpp:2488-2553`) | NO | Only the COMMITTING evaluation may perform this refresh (mirrors `run_region()`'s own one-time, post-convergence timing exactly); every non-committing evaluation must leave `.history` untouched, which it already is "for free" as long as the window-Newton residual function reuses `advance_window()` unchanged |
| `.last_window_waveform` | **never**, in the Runner path -- `WindowCoupler`-only field (`window_coupler.cpp:1041`, `channel_io_`'s own straddle handling); `src/runtime/iteration.cpp` never reads or writes it | n/a | n/a -- not part of the Runner's own window state at all |
| `window_last_dst_step[node]` | every call, unconditionally, BEFORE the `first_sweep_of_pass` branch (`iteration.cpp:2895`) | NO | Idempotent within one pass (same `absolute_instance` every call) -- no restore needed as long as every evaluation of the SAME pass passes the SAME `ins.dst_step`; a restore IS needed across passes if a probe is ever issued against the wrong pass, which the seam must refuse rather than allow |
| `window_pass_start_clock[node]` | ONLY at `first_sweep_of_pass` (`iteration.cpp:2940-2942`); read, never rewritten, by every later evaluation of the SAME pass | NO | NOT restored between evaluations of the SAME pass -- it is this pass's own equivalent of `R.x_old`'s seed: fixed once, read by every `F` evaluation Newton performs. Must be reset (or allowed to re-seed) only when a NEW window instance's pass begins |
| `window_start_value[node]` | ONLY at `first_sweep_of_pass` (`iteration.cpp:3212-3215`) | NO | same as `window_pass_start_clock` -- a pass-scoped cache, not a per-evaluation one |
| `previous[arc]` for `arc` in `R.window_lagged_import_arcs` | `run_region()`'s own k-loop top, once per Picard SWEEP (`iteration.cpp:1905-1910`), from the live `slots[arc]` | NO | YES, and differently from the ordinary seam: this is where `x`'s own lagged-import slice must be DELIVERED (Sec. 9.1's block selection is exactly these arcs) -- `advance_window()` reads `previous[cfg.window_import.value()]` for a lagged import (`iteration.cpp:2983-2985`), never `slots[]` directly, so a window-Newton evaluation writes `x`'s block into `previous[]`, not into `slots[b.feed_arc]` the way the ordinary seam does |
| `slots[b.feed_arc]` (the consumer's own `window_import` storage, i.e. `cfg.required`'s backing arc) | the compiled Sink, inside the region body (`b.sink_instruction`) | NO (not a component) | YES, mirrored from the ordinary seam's identical snapshot/restore of `feed_arc`/`value_arc` -- this is where `G(x)` is read back after the body executes |
| `R.carried_arcs` | region body | same as ordinary | reuse as-is |

**The Seam track's own lesson applies again, pre-emptively.** The first
version of `evaluate_region_residual()` under-covered the checkpoint set by
one category (feed/value arcs) and a test caught it; this table identifies
SIX categories of Runner-owned, non-component state a window sweep touches
that the ordinary seam's checkpoint set has no notion of at all
(`clock`/`straddle_*`/`window_last_dst_step` need restore;
`window_pass_start_clock`/`window_start_value` need deliberate
NON-restore, scoped to the pass rather than the evaluation; `previous[]`
needs to become a DELIVERY target, not a restore-only snapshot). Missing
any one of the restore-needing three reproduces the ordinary seam's own
historical defect in a new place: the Runner re-entering the region (the
NEXT window instance, or the Runner's own Picard path for the SAME region
if Newton's capability check ever becomes a runtime choice rather than a
setup-time one) would see a trial `x`'s leftover effect instead of the
committed one.

### 9.3 The coroutine structure's own constraint

`advance_window()` is re-entered once per Picard SWEEP of ONE window
instance (`run_region()`'s k-loop, `iteration.cpp:1896` onward, dispatches
`co_await advance_window(components, stream[i], resolved[i], k)` at
`iteration.cpp:1936` for every `k`). Its `iteration` parameter drives
exactly three decisions, ALL gated on `first_sweep_of_pass = (iteration ==
0)` (`iteration.cpp:2902`):

1. **New-instance detection.** `first_sweep_of_pass` must AGREE with the
   dst_step-derived `new_instance` test, or the call is refused
   (`E-RT-WINDOW-SWEEP`, `iteration.cpp:2903-2910`) -- this is a
   consistency check between "the caller's own claim about where in the
   pass this is" and "what the compiled stream's own `dst_step` says",
   and it is NOT negotiable: a window-Newton residual function cannot
   invent its own `iteration` numbering independent of this check.
2. **`pass_start_clock`/`window_start_value` seeding vs. reading**
   (`iteration.cpp:2912-2945`, `3212-3220`) -- computed from `st.clock`
   and `cfg.provided` ONLY at `iteration == 0`, cached, and read
   UNCONDITIONALLY (never recomputed) at `iteration > 0`.
3. **The lagged-import predictor** (`use_predictor = cfg.import_lagged &&
   first_sweep_of_pass`, `iteration.cpp:2982`) -- extrapolates
   (`extrapolate_boundary_value()`) from `st.history` ONLY on the pass's
   first sweep; every later sweep reads the live/shadow buffer directly.

**Consequently: a window sweep CAN be re-run from a provided `x` without
re-entering the pass, but only as `iteration > 0`.** `iteration == 0` is
not "the first Newton evaluation of this window instance" -- it is
specifically "there is no genuine prior sweep to extrapolate from, use the
predictor instead", a case that applies ONCE, at the moment this window
instance is entered, regardless of which solver (Picard or Newton) drives
it. The design: the window-Newton residual function for one window instance
calls `advance_window(..., iteration = 0)` EXACTLY ONCE, to seed
`window_pass_start_clock`/`window_start_value` and apply the predictor to
the FIRST `x` it evaluates at (the ordinary seam's own analogue: `run_
region_newton()` seeds `x` from `slots[b.feed_arc]`, "what the components
were ACTUALLY advanced with", before ever calling `f`; here that
role is played by the predictor's own extrapolated value, which IS "what
the components would have been advanced with" absent a real prior sweep).
Every SUBSEQUENT call -- the Jacobian action's own perturbed evaluation,
every line-search trial, every later Newton iterate, and the final
committing evaluation -- passes `iteration = 1` (a fixed, non-zero
constant; nothing downstream of `first_sweep_of_pass` distinguishes
`iteration = 1` from `iteration = 7`): `pass_start_clock`/`window_start_
value` read the cached values from the SAME `iteration == 0` call, and the
lagged import reads `previous[]`/`slots[]` directly rather than
extrapolating. This is why Sec. 9.2 lists `previous[arc]` as the lagged
import's OWN delivery target rather than a restore-only snapshot: at
`iteration > 0`, `advance_window()`'s read of `previous[cfg.window_import.
value()]` (`iteration.cpp:2984`) IS the read of `x`'s own lagged-import
slice, so the residual function writes `x`'s block there before calling
`advance_window()`, exactly the way `evaluate_region_residual()` writes
`x`'s block into `slots[b.feed_arc]` for an ordinary region.

**One real gap this surfaces, not papered over.** The very first Newton
evaluation of a window instance is seeded by the PREDICTOR
(`extrapolate_boundary_value()`), not by a caller-chosen `x` -- there is no
"what the components were actually advanced with" to read for a window
instance's own first sweep the way the ordinary seam reads `slots[b.feed_
arc]`, because nothing has been delivered into `previous[]` yet for THIS
window instance (it still holds the PREVIOUS window's final delivery, by
design, `iteration.cpp:2950-2955`). Newton's own first iterate `x_0` is
therefore NOT free to be "the predictor's value" if the solver wants to
start somewhere else (e.g. the previous window's own converged state, with
no extrapolation): the predictor path and the "deliver an arbitrary x0"
path are two different code shapes in `advance_window()` today
(`use_predictor` true vs. false), and only the FIRST is reachable at
`iteration == 0`. The design choice this track proposes (open question,
Sec. 9.8): seed Newton's `x_0` from the SAME predictor-extrapolated vector
`advance_window(iteration=0)` already computes internally -- i.e., call it
once with `iteration = 0` and `x` not yet known, read back what it wrote
into the export buffer as `x_0`'s block for that member -- rather than
inventing a second "deliver a caller-chosen x on the very first sweep of a
pass" code path alongside the predictor. This keeps `advance_window()`
itself unchanged; the window-Newton seam only ever calls it with `iteration`
in `{0, 1}`, never writes `x` before the `iteration = 0` call, and extracts
`x_0` from its output afterward.

### 9.4 Rank agreement

`run_region()`'s own window-scoped branch computes a PER-RANK
`WindowBlockSample{residual_sq, reference_sq, owned}` per block
(`iteration.cpp:2140-2161`), gathers every participant's array via
`gather_window_block_samples()` (point-to-point over the region's own
`R.participants`, `runner.cpp:2746-2839` -- NOT a true `MPI_Allgather`,
for the same reason `reduce_region_residuals()` isn't one: `R.participants`
is a subset of the base communicator's ranks, `runner.cpp:2793-2799`), and
reduces with `decide_window_convergence()` -- a flat sum of OWNED
contributions, in ascending rank-index order, pure and unit-testable at
tier 0 with a hand-built `gathered` array (`window_convergence.hpp:81-85`).

A window-Newton solve's OWN two reductions -- `F`'s norm (for the outer
Newton convergence test) and GMRES's own inner products (for the Krylov
iteration itself) -- need the IDENTICAL agreement: every rank that hosts
any part of this window region's `x` must see the same scalar, summed over
exactly `R.participants`, in the same fixed order. `fuse::linalg::Reducer`
(`comm.hpp:35-62`) is already the single point every norm/dot product in
`fuse::num::solve()` goes through -- `DistVector::dot()`/`norm2()` call
through the `Reducer` the vector was CONSTRUCTED with
(`dist_vector.hpp:48-50`), and `run_region_newton()` already builds `x`
with a specific, swappable choice: `linalg::make_serial_reducer()`
(`iteration.cpp:1679-1681`). **The single-rank design already does what the
multi-rank lift needs**: nothing in `fuse::num::solve()`, `MffdJacobianAction`,
or `gmres()` ever reduces a scalar directly -- every one goes through the
`DistVector` it was handed, which goes through whatever `Reducer` that
vector carries. Lifting rank agreement is therefore confined to ONE
decision at the call site in a future `run_region_newton_window()`: which
`Reducer` to construct `x` with.

**What that `Reducer` must be is the one open design point here, not a
reducer that already exists.** `make_mpi_reducer(int comm_f)`
(`comm.hpp:141`) takes a REAL `MPI_Comm` handle and reduces over EVERY rank
of that communicator -- exactly wrong for a region whose `R.participants`
is a subset of the base communicator (the same reason `reduce_region_
residuals()`/`gather_window_block_samples()` use point-to-point addressing
over `part[]` rather than a sub-communicator built from it,
`runner.cpp:2596-2601`'s own "WHY NOT A SUB-COMMUNICATOR" --
`MPI_Comm_free` is collective and ties a communicator's lifetime to the
Runner's own, which this track inherits rather than re-litigates). The
window-Newton lift therefore needs a NEW `Reducer` implementation --
call it `ParticipantSetReducer` for this discussion -- that performs
`sum`/`max`/`allgather` via the SAME `epoch_schedule()`-driven
point-to-point exchange `gather_window_block_samples()` already proves
correct over `R.participants`, not a call to `make_mpi_reducer()`. This is
a NEW class, not a config choice between two existing `Reducer`s -- flagged
here rather than left implicit, because it is schedulable, bounded work
(one class implementing four virtual methods against an already-proven
communication pattern) but it is not nothing, and Stage B's own scope
should name it as a deliverable if the lift is approved for this step
rather than deferred.

### 9.5 The block preconditioner

**What the species block of the Jacobian actually is.** `F(x) = x - G(x)`
where `G`, for one block (one directed channel over one window), is the
composition of: (a) the window's own temporal reconstruction of `x`'s
samples into a `Waveform` (`advance_window()`'s own `import_wf`,
`Waveform(1, cfg.provider_capacity, TemporalBasis::Linear, 1)`,
`iteration.cpp:3030`), (b) that waveform's MEAN over each of the
IMPORTING species' own substeps (`detail::mean_over`, read as `cfg.
required[e]` immediately before each `comp.advance(ctx)` call,
`iteration.cpp:3315-3326`), and (c) the importing species' own chain of
backward-Euler substeps (`UCore::step()`/`VDiffusionCore::step()`,
`species_core.hpp`, each one `DiffusionSolver::solve()` against a
FACTORED `(1 + extra_diagonal)*I - dt*diffusion*Laplacian` system,
`species_core.hpp:182-220`), run ONCE PER SUBSTEP, serially, each one
depending on the previous substep's own output state. `dG_i/dx_j` (species
`i`'s own exported block with respect to species `j`'s own imported
samples) is therefore NOT block-diagonal in TIME: substep `k`'s own state
depends on every earlier substep, so the TRUE Jacobian action chains `n_sub`
applications of (one substep's own sensitivity to its own mean input) times
(the mean-over-substep operator's own sensitivity to the waveform's
control points) -- a dense, lower-triangular-in-substep-index structure
per species block, exactly what `MffdJacobianAction`'s own one-sided finite
difference already computes CORRECTLY (it differences the WHOLE `G`, chain
and all) but a hand-assembled preconditioner cannot cheaply reproduce
exactly.

**What a per-species preconditioner can honestly approximate.** The
DOMINANT linear operator inside that chain, substep by substep, is exactly
`DiffusionSolver::solve()`'s own already-factored system -- the reaction
term is EXPLICIT in both species (`UComponent`'s own file header: "q...
enters explicitly... keeping U's own linear system LINEAR"; `species_core.
hpp`'s own `DiffusionSolver` comment: "a right-hand side that changes every
step", the FACTORIZATION itself depending only on `dt`/`diffusion`, never
on the coupling term). The honest answer the task anticipates: **the
preconditioner acts PER SAMPLE, through the species' own already-factored
solve, treating each of the window's `n_sub` output samples as
independently preconditioned by ONE application of that species' own
`DiffusionSolver::solve()`** -- a block-DIAGONAL-IN-TIME approximation of a
Jacobian that is genuinely lower-triangular-in-time, i.e. it drops the
substep-to-substep CHAINING (sample `k`'s sensitivity to sample `k-1`'s own
perturbation) and keeps only each sample's own LOCAL diffusion response.
This is a real approximation, not the exact diagonal block `Linearization
Token::apply_preconditioner`'s own ordinary-region contract describes (that
contract is ONE `evaluate_residual()` call at a frozen time level; a window
member's own residual is `n_sub` chained calls, which S7.1's
`ImplicitComponent` interface -- built for the single-clock case -- does
not describe at all). Concretely: `PreconditionerBlock::apply(r, z)` for
species `i`'s own block calls the identical `DiffusionSolver::solve()`
already built and factored for that species (`UCore`/`VDiffusionCore`'s own
diffusion-solver member, unchanged), once per `(element-block, sample)`
slice of `r`, reusing the CURRENT factorization (same `dt`, same
`diffusion`) -- no new factorization, no new assembly.

**Cost.** One `DiffusionSolver::solve()` call is one sparse triangular
back-substitution against an ALREADY-FACTORED matrix (`Eigen::
SimplicialLDLT::solve()`, `species_core.hpp:213-220`) -- the SAME cost
`advance()` already pays once per substep, with no reaction/coupling-term
evaluation, no remap, no waveform construction. Applying the preconditioner
once therefore costs roughly `n_sub` solves per included block (11 for U,
8 for V at the Brusselator's own defaults) -- MUCH cheaper than one `G`
evaluation, which pays the SAME `n_sub` solves PLUS the nonlinear reaction
term, the remap (`fuse::apply_weights`), and the window's own waveform
book-keeping, for BOTH species. A single GMRES iteration needs one
preconditioner application and one Jacobian action (one MFFD evaluation of
`F`, i.e. one full `G`); a Newton step needs the forcing-term-selected
number of GMRES iterations (dn_heat's own measured range: 2 for a loose
`eta_0 = 0.3`, more as `eta_k` tightens, Sec. 8.2) plus the line search's
own trial evaluations. The preconditioner's own cost is therefore a SMALL
ADD-ON to each already-necessary Jacobian action, not a new expense of the
same order as a full window advance -- consistent with its purpose
(reducing the NUMBER of GMRES iterations needed per Newton step), not with
replacing Picard's own per-sweep cost.

### 9.6 Measurements taken, no library change

All of the following were measured this session against the unmodified
`examples/brusselator` binary, built at `/tmp/winnewton/b0` (tier 0,
`-DCMAKE_BUILD_TYPE=Debug`) from this checkout; every number below is from
that binary's own stdout/`windows.csv`, not estimated.

**Picard (`--accel none`) sweep counts, `--driver runner`, defaults
(`--dt-u 0.002 --dt-v 0.003 --window 0.02`), first five windows (`--end
0.1`):** 12, 12, 9, 9, 9 sweeps -- decreasing, never growing, as the system
approaches its own slowly-decaying transient (confirmed over `--end 1.0`,
50 windows: the count decreases monotonically from 12 to 7 and stays
there). Byte-identical between `--driver window` and `--driver runner` at
every window (both print `max_iterations=12`, identical `windows.csv`
columns) -- the two drivers agree on Picard exactly, as Conformance
Units 1/2 already established.

**Aitken (`--accel aitken`) reproduces KNOWN_ISSUES D17 exactly.**
`--driver window --accel aitken --end 0.1`: converges windows 0-1 (16, 16
sweeps), then FAILS at window index 2 (`t=0.060000`) after the full 60-
sweep budget -- the identical failure D17 records at `--end 0.5`.
`--driver runner --accel aitken --end 0.1`: converges throughout, but the
sweep count GROWS every window rather than settling -- 16, 17, 24, 28, 31
-- where Picard's own count over the SAME five windows DECREASES (12, 12,
9, 9, 9). Aitken is measurably getting WORSE, window over window, on a
problem Picard itself finds easier over time. This is the clearest
evidence this session has for Newton's OWN value proposition on this
example: not "fewer sweeps than Picard" (Picard is already cheap and
improving here) but "robust where a quasi-Newton accelerator is not" --
Aitken's own instability (not merely slowness) on the IDENTICAL graph and
parameters is a real, measured failure mode a Jacobian-based method is not
expected to share (JFNK's own convergence theory ties its local rate to the
TRUE Jacobian, not to a rank-1 secant update's own history-dependent
coefficient).

**No Picard-diverges-or-crawls case was found within the CLI-exposed
parameter space.** Swept this session: `--window` from 0.02 to 1.0 (12,
18 [0.08], 20 [0.16], 26 [0.32], 28 [0.5], 32 [1.0] sweeps -- sub-linear
growth, no divergence); `--alpha` (diffusion) from 10 to 200 at the
default window (NO measurable change in sweep count at all -- 12, 12, 9,
9, 9 identically at every value tried, because the coupling that drives
Jacobi's own contraction is the EXPLICIT reaction term, not the implicit
diffusion solve); `--no-forcing` (no change). **This is a genuine
limitation of this measurement, stated plainly rather than papered over**:
the Brusselator example's own CLI does not expose `WaveformIteration`
(`brusselator_graph.hpp`'s own `iteration_mode` parameter defaults to
`Jacobi` and is never threaded through `main.cpp`'s argument parser), and
GaussSeidel -- or a deliberately stiffer reaction constant (`A`/`B` are
compiled in at `3.4`/`1`, not CLI-tunable either) -- is the more likely
place Picard's own contraction factor would exceed 1, given that D17
already shows GaussSeidel-adjacent instability (Aitken diverging under
the direct `WindowCoupler` driver). **Proposed Newton test case** (for
Stage B, pending a code change this Stage does not make): add a
`--iteration-mode jacobi|gauss-seidel` flag to `examples/brusselator/
main.cpp` (a two-line change, threading `WaveformIteration` through to
`add_window_coupling()`'s existing parameter) and measure Picard/Aitken/
Newton under GaussSeidel at the SAME defaults -- if GaussSeidel's own
unlagged-direction substitution pushes the LAGGED direction's own
contraction past 1 (plausible: GaussSeidel's own per-sweep update is a
STRONGER, not weaker, coupling than Jacobi's, since the unlagged member
sees the OTHER member's freshest value within the same sweep), that is
the named Newton case; if it does not, the search continues from there
rather than from a parameter this session already ruled out.

**F-evaluation count, relative to Picard's own measured sweep counts.** A
Newton step needs one `F(x_k)` evaluation, `GMRES`'s own iteration count
(via `MffdJacobianAction`) evaluations of `F` for the Jacobian action,
and the line search's own trial evaluations when the full step is
rejected. dn_heat's own measured counts (Sec. 8.2-8.4): 3 Newton steps for
the forcing-dominated affine case (`eta_0 = 0.3`, 2 GMRES iterations on
the first linear solve); 3-4 Newton steps for the quadratic-rate nonlinear
case under near-exact forcing (more GMRES work per step, traded for fewer
steps); 7 Newton steps with one line-search backtrack (4 trials) on a
deliberately poor initial guess. Applying the SAME order of magnitude to
the window case -- unpreconditioned, so GMRES needs more iterations per
Newton step than the preconditioned design of Sec. 9.5 would need -- a
plausible total is on the order of 15-30 `F` evaluations for one window
solve (3-5 Newton steps x 3-6 GMRES iterations each, plus occasional line
search trials), each `F` evaluation costing roughly what one Picard sweep
costs (one full window advance of both species). Against Picard's OWN
measured 9-12 sweeps at the SAME defaults, Newton is therefore PLAUSIBLY
MORE EXPENSIVE in raw evaluation count on this well-behaved problem --
consistent with Sec. 9.5's own conclusion and with the task's own
expectation: Newton's value here is robustness where Picard's contraction
factor is poor (Sec. 9.6's own GaussSeidel proposal) or where a
quasi-Newton accelerator is unstable (Aitken, measured above), not raw
sweep count on a problem Picard already handles cheaply.

### 9.7 Test plan, and the files Stage B would touch

**Serial tests pinning Sec. 9.2's re-enterability claim** (new file,
`test/runtime/window_fixed_point_residual_test.cpp`, mirroring `test/
runtime/fixed_point_residual_test.cpp`'s own structure):

1. `WindowFixedPointResidual.RepeatedEvaluationAtTheSamePointIsBitIdentical`
   -- two `F` evaluations at the identical `x` (`iteration = 1` both times)
   produce bit-identical `F(x)`, mirroring the ordinary seam's own test of
   the same name.
2. `WindowFixedPointResidual.AWindowPassWithAnInterleavedProbeIsBitIdenticalToOneWithout`
   -- evaluate `F` once (non-committing) at an
   arbitrary `x`, then run the SAME window instance through the Runner's
   own Picard path from its genuine starting point; the result must be
   bit-identical to a window instance that never saw the out-of-band
   evaluation at all -- the direct window-region analogue of
   `EvaluatingFDoesNotPerturbTheRunnersOwnPicardIteration`.
3. `WindowFixedPointResidual.TheIterationZeroCallSeedsThePassAndIterationOneDoesNotReseedIt`
   -- pins Sec. 9.3's own two-call convention directly:
   call with `iteration = 0` once, then `iteration = 1` twice at two
   DIFFERENT `x`, and assert `window_pass_start_clock`/`window_start_value`
   (exposed to the test the same way `window_member_state` already is,
   through a test-only accessor or a white-box test in `src/runtime/`)
   are identical across all three calls.
4. `WindowFixedPointResidual.CommittingEvaluationRefreshesHistoryExactlyOnceNonCommittingDoesNotTouchIt`
   -- pins the Sec. 9.2 table's own
   `.history` row: several non-committing evaluations, then one committing
   one; `st.history` must show exactly the committing call's own effect,
   matching `run_region()`'s own post-loop refresh bit for bit.
5. `WindowFixedPointResidual.GaussSeidelUnlaggedBlockIsExcludedFromX` --
   builds a GaussSeidel two-member window with one unlagged direction, and
   asserts `x`'s own length/offsets cover only the lagged block -- the
   direct test of Sec. 9.1's own exclusion rule, which also proves
   `E-RT-WINDOW-ACCEL`'s own gap (Sec. 9.1's opening citation) is not
   silently reproduced in the Newton seam.
6. `WindowFixedPointResidual.MutationRowsForEachUnrestoredStateCategory`
   -- one test per Sec. 9.2 table row marked "YES": disable that row's own
   restore and show a SPECIFIC, named test fails (`build-evidence/
   mutations/runtime.toml`, new rows `window-newton-clock-not-restored`,
   `window-newton-straddle-state-not-restored`, hand-proven per the
   standing mutation discipline).

**Brusselator comparison table** (extends `test/verification/brusselator_
*_test.cpp`, Stage B's own addition): Picard/Aitken/Newton sweep (or
Newton-step) counts and final `mean_u`/`mean_v` at the SAME defaults this
session measured, plus the GaussSeidel case Sec. 9.6 proposes once its
own CLI flag exists, with Newton's own quadratic rate measured the same
way Sec. 8.3 measures it for dn_heat (`q_k` fit over the trailing triples
of `||F||` still above the `1e-10 * ||F_0||` floor).

**Files Stage B would touch**, if the lead approves this design:
`src/runtime/iteration.cpp` (`check_region_supports_newton()`'s
window-scoped refusal narrowed to multi-participant/externally-driven only;
a new `run_region_newton_window()` sibling to `run_region_newton()`; the
`x`-block selection of Sec. 9.1 inside a new `evaluate_window_region_
residual()` sibling to `evaluate_region_residual()`), `src/runtime/
runner_impl.hpp` (declarations for the above, plus whatever accessor Sec.
9.7's white-box tests need), `src/linalg/comm_mpi.cpp` and `include/fuse/
linalg/comm.hpp` (the `ParticipantSetReducer` of Sec. 9.4, if the lead
wants the multi-rank hook built now rather than left for a later step --
Sec. 9.8's own open question), `examples/brusselator/species_component.hpp`
(the `ImplicitComponent` seam, Sec. 9.5 -- note this needs a WINDOW-aware
residual/linearisation shape S7.1's own `ImplicitComponent` does not
provide today, Sec. 9.8's own second open question), `examples/brusselator/
species_core.hpp` (exposing `DiffusionSolver` through a const accessor for
the preconditioner block), `examples/brusselator/main.cpp` (`--accel
newton`, and the `--iteration-mode` flag Sec. 9.6 proposes),
`test/runtime/window_fixed_point_residual_test.cpp` (new, Sec. 9.7),
`test/verification/brusselator_species_core_decomposition_test.cpp` (one
Newton case at 2 partitions, per the task's own brief), `build-evidence/
mutations/{runtime,brusselator}.toml` (new rows), `docs/user/
brusselator.md` (Newton results).

### 9.8 Open questions for the lead

1. **Newton's `x_0` on a window instance's first sweep** (Sec. 9.3): seed
   it from `advance_window(iteration=0)`'s own predictor output (this
   track's proposal -- no new code path in `advance_window()` itself,
   only in the new residual-function wrapper around it) rather than adding
   a second, caller-chosen-`x0` code path inside `advance_window()`
   alongside the existing predictor. The alternative (a real second path)
   would let Newton start from an arbitrary seed on the very first sweep,
   at the cost of a second branch in already-dense, heavily-commented
   code the Seam/Time-windows tracks fought hard to get right.
2. **The `ParticipantSetReducer` of Sec. 9.4** is new code, not a
   configuration choice between two existing `Reducer`s -- should Stage B
   build it now (so the multi-rank lift truly is "flip the Reducer" when
   it comes), or defer it and ship Stage B single-rank only (`R.
   participants.size() <= 1`, mirroring `check_region_supports_newton()`'s
   OWN existing restriction for an ordinary Newton region), leaving the
   Reducer for the step that actually exercises 2-3 ranks on the host?
   This track's recommendation is to defer it: nothing in Sec. 9.1-9.3's
   design depends on it, and building a communication primitive with no
   host-side test able to exercise it (this sandbox cannot run `mpiexec`,
   per the standing rules) risks the SAME kind of unverified-until-the-
   host-runs defect `KNOWN_ISSUES.md` D24 and D20 both describe.
3. **S7.1's `ImplicitComponent` interface does not describe a window
   member's own residual/linearisation** (Sec. 9.5): `evaluate_residual()`
   is one call, at a frozen time level, for an ordinary region; a window
   member's own residual is `n_sub` chained `advance()` calls. This design
   does NOT propose extending `ImplicitComponent` itself (out of this
   track's own file ownership, `fuse/runtime/component.hpp`) -- it
   proposes that `species_component.hpp` builds its OWN `LinearizationToken`-
   shaped object directly, outside the `Component::implicit()` query
   entirely, read only by the window-Newton seam's own block-Jacobi
   wiring (a narrower, example-local interface, not a library-wide one).
   The lead may prefer the opposite: extend `ImplicitComponent` with a
   window-aware residual method now, so a FUTURE window-Newton component
   (not just the Brusselator's own species) gets the same seam S7.1 built
   for the ordinary case. This track did not have evidence (a second
   window-Newton use case) to justify that generalisation yet.
4. **The GaussSeidel/stiff-parameter Newton test case** (Sec. 9.6) needs a
   code change (an `--iteration-mode` CLI flag) this Stage does not make.
   Confirm before Stage B whether adding that flag is in scope for this
   step, or whether the Newton case should instead be a NEW, purpose-built
   test fixture (smaller, hand-constructed, not the Brusselator example)
   where a divergent Picard contraction factor can be set directly rather
   than searched for.

**RESOLVED (lead's go-ahead, 2026-10-11) -- see Sec. 9.9 for what each
decision actually produced:**
1. ACCEPTED as proposed: `x_0` seeds from `advance_window(iteration=0)`'s
   own predictor output.
2. DEFERRED at Stage B time: single-rank only (`R.participants.size() <=
   1`); the Reducer SEAM was built (`RegionRuntime::newton_reducer`) so a
   later multi-rank step would be a Reducer swap, not a re-plumb.
   **Built, Sec. 9.10 (2026-10-12):** the reducer's own cross-rank body
   (`fuse::linalg::ParticipantSetReducer`) -- exactly the swap Stage B's
   own seam was designed for, confirmed by how little of `run_region_
   newton_window()` itself needed to change.
3. `species_component.hpp` builds its own `fuse::LinearizationToken`
   directly on `UComponent`/`VComponent` (the library-wide `fuse::
   LinearizationToken`, not a narrower example-local shape) -- NOT an
   `ImplicitComponent` extension; `ImplicitComponent::evaluate_residual()`
   on a window member is interpreted as computing `G` alone (this
   member's own exported block from its own window of substeps), not
   `x - G`, following `dn_component.hpp`'s own `DirichletSlab::
   evaluate_residual()` precedent -- never called by this seam's own
   runtime wiring either way (Sec. 9.9.3).
4. No Picard-divergent case was found in CLI-reachable parameter space
   (Stage A's own finding, Sec. 9.6); the `--iteration-mode` CLI flag WAS
   added (in scope after all), and the Newton test case became "every
   window converges where Aitken's own D17 instability does not"
   (Sec. 9.9.4), not a Picard-divergent one.

### 9.9 Stage B record (2026-10-11): what was built, and what was measured

**Implementation.** `Runner::Impl::run_region_newton_window()`/
`evaluate_window_region_residual()` (src/runtime/iteration.cpp, new,
alongside the existing `run_region_newton()`/`evaluate_region_residual()`
the ordinary seam already had) do exactly what Sec. 9.1-9.5 describe:
`check_region_supports_newton()`'s window-scoped refusal narrowed to the
two checks it shares with the ordinary seam (participant count,
externally-driven members); the window-Newton interface vector built
per Sec. 9.1's layout, GaussSeidel-excluded blocks never appearing in it;
the Sec. 9.2 checkpoint table's save/restore bracket; the Sec. 9.3
two-call (`iteration = 0` once, `1` thereafter) convention, driven
synchronously (`Task::resume()`/`take_result()`, Sec. 9.3's own
coroutine-constraint finding held); the Sec. 9.5 block preconditioner via
a `fuse::LinearizationToken` on `UComponent`/`VComponent` directly
(`apply_jacobian`/`apply_jacobian_transpose` refuse `E-NEWTON-NO-JACOBIAN-
ACTION`, never called by this seam's own MFFD-based Jacobian action).
`examples/brusselator/main.cpp` gained `--accel newton`, `--iteration-mode
{jacobi,gauss-seidel}`, and `--no-block-jacobi`.

**File list correction against Sec. 9.7's own prediction.** Two files
needed touching beyond that list, for reasons only visible once the wiring
was actually built: `examples/brusselator/brusselator_graph.hpp`
(`add_window_coupling()` needed a trailing `NewtonConfig` parameter and
the `requires_residual_capability`/`requires_linearization_capability`
auto-trigger `IterationBuilder::newton()` already does for a graph built
through that builder -- `add_window_coupling()` builds its
`IterationRegionDecl` directly and has no builder to inherit the trigger
from) and `include/fuse/runtime/runner.hpp`/`include/fuse/runtime/
fixed_point_residual.hpp` (two new TEST/DIAGNOSTIC-ONLY public accessors,
`Runner::window_region_fixed_point_residual()`/`debug_window_member_
snapshots()`, mirroring `region_fixed_point_residual()`'s own S7.1
precedent for the window seam -- Sec. 9.7's own test plan needed SOME way
to drive `evaluate_window_region_residual()` from outside `Runner::Impl`,
and no such door existed yet).

**9.9.1 Convergence, measured against the real Brusselator.** At the
example's own defaults (`dt_u=0.002, dt_v=0.003, window=0.02`, `--driver
runner --accel newton`), every window instance converges:
- `--end 1.0` (50 windows), Jacobi, WITH the block-Jacobi preconditioner:
  50/50 converged (`NewtonStatus::ConvergedRelative`), 0 divergences. The
  FIRST window (the hardest -- furthest from the limit cycle) needs 4
  outer Newton iterations; by the last few windows (the trajectory
  settled into its own oscillatory regime) that drops to 3, with the
  per-step GMRES iteration count falling from ~70 (first window, last
  Newton step) to 2-6 (late windows).
- The SAME run with `--iteration-mode gauss-seidel`: 50/50 converged,
  IDENTICAL final `mean_u`/`mean_v` (`0.668994004315`/`2.8189403867`, to
  every printed digit) to the Jacobi run -- the physical answer does not
  depend on which waveform-iteration mode reached it.
- `--no-block-jacobi` (the `fuse::num::IdentityPreconditioner` path): also
  50/50 converged, SAME final means, fewer outer iterations on the first
  window (3, not 4) -- a real preconditioner changes the PATH to
  convergence, never whether convergence happens, on this problem.
- A longer run, `--end 11.5` (the example's own default `end`, 575
  windows): cut short at 222 consecutive windows for runtime (each window
  at `eta_min = 1e-6`, see 9.9.2, costs noticeably more GMRES work than at
  the chosen default) -- 222/222 converged, 0 divergences, before the run
  was interrupted. Not a claim about windows 223-575; a claim about the
  222 actually observed.

**9.9.2 A genuine numerical finding: `NewtonConfig::eta_min` needed a
floor for this problem.** At `eta_min`'s own SNES-matching default
(`0.0`), the FIRST window's Newton solve diverges:
`DivergedLinearSolve` after 3 steps, `||F|| = 1.15e-5` (already ~9e-8
relative to `||F_0|| = 125.2`) -- `FUSE_NEWTON_DEBUG=1` tracing (a new,
permanent env-gated diagnostic, `src/runtime/iteration.cpp`, matching the
codebase's own `FUSE_SCHEDULE_DEBUG`/`FUSE_TRANSPORT_DEBUG`/
`FUSE_NODAL_DEBUG`/`FUSE_CONSERVATION_DEBUG` convention) shows the 4th
step's own Eisenstat-Walker forcing term reaching `eta ~= 6.6e-9` --
below the finite-difference noise floor the MFFD Jacobian action can
actually supply when `F` itself composes MANY chained substeps (11 for U,
8 for V) rather than one `G` evaluation. `eta_min = 1e-4` (two orders of
magnitude above the observed stagnation point) removes the stall in
EVERY window tried, at a small fraction of the GMRES cost `eta_min =
1e-6` also removes it at (`examples/brusselator/main.cpp`'s own comment
has the two measurements side by side). This is exactly the scenario
`NewtonConfig::eta_min`'s own doc comment names as the reason the field
exists, not a defect in this track's own wiring -- see that comment
(`fuse/model/iteration.hpp`) and `main.cpp`'s own citation of it.

**9.9.3 The `x` layout, measured exactly as Sec. 9.1 predicted.**
`RegionRuntime::newton_n` reads `4864` under Jacobi at the example's own
defaults -- the EXACT total Sec. 9.1 computed by hand (U-capacity-11 ×
256 = 2816, "u2v"; V-capacity-8 × 256 = 2048, "v2u"). Under GaussSeidel,
`newton_n` drops to `2048`: the "u2v" block (length 2816) is excluded,
"v2u" (length 2048) remains -- confirmed directly (a debug dump of every
`RegionBlock::channel_name`/`length`/`newton_offset`), not inferred from
the total alone, closing the possibility that the WRONG block was
excluded while the arithmetic happened to still add up.

**9.9.4 The Newton test case: Aitken's own D17 instability, not a
Picard-divergent one.** Stage A found no Picard-divergent case in
CLI-reachable parameter space (Sec. 9.6); Stage B's own Newton run at the
SAME defaults where `--accel aitken` reproduces `KNOWN_ISSUES.md` D17's
instability (window index 2, `--driver window`) converges cleanly every
window instead -- the comparison this track's test case actually is.

**9.9.5 Tests added** (`test/runtime/window_fixed_point_residual_test.cpp`,
new, 6 cases on a synthetic two-member linear-ODE window region -- NOT
the Brusselator, so these stay fast and isolated from the Brusselator's
own straddling/remap complexity): re-enterability (`F` twice at the same
`x`, bit-identical), the Sec. 9.3 two-call convention (`window_pass_
start_clock`/`window_start_value` identical across an `iteration = 0`
call and two different-`x` `iteration = 1` calls), an interleaved extra
probe not perturbing the real committed result, the committing-evaluation
`.history` refresh happening exactly once, the GaussSeidel exclusion rule
(by channel name, not just by count), and bit-for-bit agreement between
two full Newton runs from the same inputs. `test/runtime/newton_region_
test.cpp`'s own pre-existing `AWindowScopedRegionSelectingNewtonIsRefused
AtSetup` -- which pinned the NOW-LIFTED blanket refusal -- is renamed and
re-asserted as `AWindowScopedRegionWithNoConfiguredWindowMembersIsANoOp`
(a window-scoped region with no real window members resolves `newton_n =
0` and converges as a no-op, rather than either refusing or crashing).
`test/verification/brusselator_species_core_decomposition_test.cpp`
gained a 2-partition case for `apply_diffusion_preconditioner()`
specifically (not just `step()`, already covered) --
`TwoPartitionVDiffusionCoreApplyPreconditionerMatchesSingleRank`,
max_abs_diff `2.168e-18`. `test/verification/brusselator_runner_test.cpp`
gained `NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPrecondit
ioner` (the pinning test `build-evidence/mutations/brusselator.toml`'s
own `brusselator-preconditioner-solve-replaced-by-identity` row kills).
Full tier-0 suite: 187/187 (186 + this track's own one new ctest entry),
100% passing, including every pre-existing Picard/Aitken/IQN-ILS/
LowFive/omp_identical case.

**9.9.6 What Stage B did NOT verify.** Multi-rank (`KNOWN_ISSUES.md` D31,
new at Stage B time) -- the `ParticipantSetReducer` was a comment, not
code, and no test exercised more than one rank. The window_member_state
`clock`/`straddle_*` restore (Sec. 9.2's table): mutating the `clock`
restore specifically was tried against both the synthetic fixture and
the real (straddling) Brusselator and did NOT change either's observed
output -- consistent with the table's own note that `clock` is "within
THIS pass, discarded in favour of the cached `window_pass_start_clock`"
for every sweep's own restart point, so a leaked trial value is
overwritten by the next sweep's own fresh computation regardless; the
restore is still correct defensive practice for the ONE place that DOES
read it directly (`resolve_pending_window_straddles()`, the NEXT window
instance), but proving that specific path needs a two-window-instance,
Fixed-policy (genuinely straddling), post-commit-probe fixture this
track did not build. Reported here rather than claimed via a fabricated
mutation `result`. **Superseded by Sec. 9.10's own multi-rank lift,
below** for the `ParticipantSetReducer` item specifically; the
window_member_state restore gap is unchanged and still open.

### 9.10 Stage C (2026-10-12): lifted to multi-rank (KNOWN_ISSUES.md D31)

**What was built.** `fuse::linalg::ParticipantSetReducer`
(`src/linalg/comm_mpi.cpp`, declared `include/fuse/linalg/comm.hpp`) --
the class Sec. 9.4 named and deferred. Built via a real `MPI_Comm_split`
over exactly `RegionRuntime::participants`, collective over the Runner's
own `base_communicator` and called in the SAME ascending-region-index
order on every rank (`resolve_regions()`'s own loop, mirroring
`Runner::Impl::attach_comm_graphs()`'s "EVERY RANK CALLS EVERY SPLIT, in
component-id order" discipline) -- a rank outside `participants` gets
`MPI_UNDEFINED` as its own colour and no `Reducer` object at all
(`make_participant_set_reducer()` returns `reducer == nullptr` for it),
so "a rank outside the participant set must not enter a reduction"
holds by construction, not by a runtime guard. The split's own lifetime
is NOT point-to-point addressing over the base communicator the way
`reduce_region_residuals()`/`gather_window_block_samples()` use (Sec.
9.4's own original recommendation) -- the task's own explicit choice
for this reducer, reasoned through in `comm.hpp`'s header comment on the
factory and `runner_impl.hpp`'s comment on `RegionRuntime::
newton_reducer`: a Newton solve's `Reducer` is reused across many
GMRES iterations and line-search trials within one window instance,
where a true `MPI_Allreduce`/`MPI_Allgather` is simpler to reason about
than a fresh point-to-point exchange proved correct per call. The held
communicator's own lifetime hazard (`runner.cpp`'s own "WHY NOT A SUB-
COMMUNICATOR" comment on `reconcile_epoch()`) is handled the way
`src/backend/moab_backend.cpp`'s own `OwnedComm` already handles an
identical one for `MoabMeshBackend`'s per-placement sub-communicators:
freed in the Reducer's own destructor, guarded by `MPI_Finalized()`,
never copyable -- tied to `resolve_regions()`'s own `regions.clear()`
(destroying the OLD `RegionRuntime` vector, and therefore every OLD
`newton_reducer`, before building new ones on a re-`initialize()`),
which runs identically, in the SAME order, on every rank.

`check_region_supports_newton()`'s blanket `R.participants.size() > 1`
refusal is GONE (`fixed_point_residual.cpp`) -- and nothing narrower
replaces it, because `R.participants` is populated with more than the
empty set ONLY when `RunnerConfig::base_communicator` has a value in
the first place (`resolve_regions()`'s own invariant on that
construction), so "more than one participant" already implies a real
communicator is configured, which is exactly what building a real
`ParticipantSetReducer` needs. The EARLIER `E-RT-REGION-MPI` refusal
(the one that blocks a dynamic accelerator on a multi-rank region,
`resolve_regions()`, same file as the `R.participants` construction)
now also exempts `AccelKind::Newton` alongside `None`/`Constant`, for
the identical reason that refusal names for those two: Newton's own
cross-rank agreement is already routed through a `Reducer` the region
owns, not a local-history inner product with no hook to add one.

`RegionBlock::newton_offset`'s own window-Newton vector-layout
computation (`resolve_regions()`) now excludes a block THIS rank does
not host (`!b.window_hosted`) from the SAME exclusion mechanism the
GaussSeidel-unlagged exclusion already used -- `R.newton_n` becomes the
LOCAL, per-rank owned total (the exact `owned_n` `linalg::DistVector`
already expects), with ZERO ghost region: Sec. 9.1's own layout
analysis, re-examined for this lift, shows no block's own F computation
or x-delivery ever needs another rank's OWNED entry directly -- the
compiled Transport instructions inside the window's own body (already
proven correct for Picard's multi-rank case, `window_runner_mpi_test.
cpp`) carry every cross-rank dependency, so `make_serial_exchanger()`'s
zero ghosts, paired with the REAL `ParticipantSetReducer`, is the
correct pairing (`comm.hpp`'s own `DistVector` constructor takes the
exchanger and reducer as independent arguments). GATING WAS MEASURED,
NOT ASSUMED, TO NEED `!b.window_hosted` ALONE: a first version also
required `R.participants.size() > 1` before excluding anything
(mirroring `run_region()`'s own Picard-path `gate_by_hosted`), and every
`EXPECT_EQ`/`EXPECT_FALSE` in `test/runtime/window_newton_hosting_test.
cpp` failed against the real binary until that extra condition was
dropped -- `R.participants` stays EMPTY under `force_this_process`
alone (no communicator), so gating on its size silently disabled the
exclusion for the exact tier-0 shape meant to exercise it; `window_
hosted` itself already defaults to `true` whenever `this_process` has
no value at all, which is why dropping the extra gate does not change
the single-process/no-`this_process` case at all.

`run_region_newton()`/`run_region_newton_window()` (`iteration.cpp`)
gained the identical early return `reduce_region_residuals()` already
has: a rank not in `R.participants` returns success immediately, before
touching `R.newton_reducer` or building a `DistVector` at all
(`Runner::Impl::region_is_local_participant()`, new, `runner.cpp` --
kept out of `iteration.cpp` so that file still names no MPI type of its
own). `run_region_newton()`'s own ordinary (non-window) seam ALSO now
reads `R.newton_reducer` instead of constructing a fresh serial reducer
every call -- the one place this track found that had not yet been
lifted to the Sec. 9.4-established "the region owns the Reducer"
convention even in the single-rank era.

**fuse::petsc::solve() (D32), lifted alongside.** `fuse::num::
NewtonSolveFn`'s signature gained a trailing `int comm_f` (Native
ignores it; PETSc's own registered entry point uses it, falling back to
`comm_self_f()` when `0` -- the unchanged single-rank behaviour, never a
silent substitute for a real communicator a multi-rank region actually
has). `RegionRuntime::newton_comm_f`, new, carries the SAME split's own
raw handle alongside the `Reducer` object (`ParticipantSetReducerResult`
bundles both from ONE `MPI_Comm_split`, not two). `solve()` itself:
`VecCreate`+`VecSetSizes(x.owned_size(), PETSC_DETERMINE)`+
`VecSetType(VECSTANDARD)`, never `VecCreateSeq` (which ignored `comm`'s
own size entirely -- every rank would have built its OWN disconnected
vector, each running an independent single-rank solve with no real
Krylov coupling, a silently wrong answer rather than a refusal, for any
communicator with more than one rank); `VECSTANDARD` resolves to
`VECSEQ` on a one-rank communicator (the UNCHANGED prior behaviour) and
`VECMPI` otherwise, with no command-line option parsing either way. A
rank owning zero unknowns is legal (refused, `E-PETSC-SOLVE-SHAPE`, only
when the WHOLE communicator is empty); a named refusal,
`E-PETSC-SOLVE-NOCOMM`, fires when `comm_f` resolves to
`MPI_COMM_NULL`. §11.3's own "Open for a future multi-rank lift" is this
section.

**What this stage did NOT verify, stated rather than left implicit.**
Every one of the above is BUILD-VERIFIED only (tier-0, tier-1 MPI+
Fortran, and a PETSc-enabled tier-1 tree, 0 compiler errors in each) --
the sandbox this stage worked in cannot complete a bare `MPI_Init()`
(the SAME constraint D31's own Stage B entry and D32's own PetscToken
entry both name), so none of `ParticipantSetReducer`'s own `sum`/`max`/
`allgather` correctness, the PETSc substrate's own distributed `Vec`/
`SNES` behaviour, or the two new host-only tests
(`parallel.window_newton_mpi.{2,3}`, `parallel.participant_reducer_mpi.
{2,3}`, plus `petsc.window_newton_mpi.{2,3}` in its own dedicated
binary) have been RUN anywhere. Sec. 9.9.6's own window_member_state
restore gap is unchanged by this stage. The per-window-instance measure
loop in `run_region_newton_window()` (`R.residual_norms`/`R.
reference_norms`, read by `Runner::region_residual_norms()`-style
callers) sums a measure's own block ONLY on the rank that hosts it,
REDUCED across `R.newton_reducer` the same way the block-total itself
now needs to be, for the diagnostic to read correctly on every
participant, not only the hosting one -- built, not independently
mutation-proved beyond the one tier-0 row this stage's own
`build-evidence/mutations/newton-parallel.toml` records.

## 10. The PETSc substrate (PetscToken track, S7.2b, 2026-10-11)

This section is the second substrate behind the SAME Newton seam §1-§5
describe: a `fuse::LinearizationToken` whose Jacobian action is PETSc's
own `MATMFFD` (Walker-Pernice differencing, `MATMFFD_WP` -- the identical
rule `fuse::num::MffdJacobianAction` (§6.1) implements natively) and whose
preconditioner is a PETSc `PC`, plus a `SNES` (`SNESNEWTONLS`) driver over
the identical `fuse::num::ResidualFn`/`NewtonResult` shapes the native
`fuse::num::solve()` already uses. Nothing in §1-§9's own seam is
PETSc-shaped, and nothing in the native Newton solver (§6-§9) knows which
substrate a `LinearizationToken` comes from -- this track adds a SECOND
implementation behind the same interface, never a change to the first.
`FUSE_ENABLE_PETSC` has no hard dependency anywhere in the library: a
build without it carries zero `fuse::petsc::*` symbols (the library target
`fuse-coupler-petsc`, src/CMakeLists.txt, does not exist at all in that
configuration) and the existing 187 tier-0 rows are byte-for-byte
unaffected (verified this track: 187/187 pass, identical row count, with
`FUSE_ENABLE_PETSC` OFF).

### 10.1 The token: `fuse::petsc::MffdToken`, `make_assembled_token`

`include/fuse/petsc/token.hpp` / `src/petsc/token.cpp`. No PETSc type
names appear in the public header at all (`tools/check-layering.py`'s
`FORBIDDEN_PUBLIC` rule matches `petsc` in any `#include` line, at any
layer, with no exception carved out for this one) -- every `<petsc.h>`
call lives in the two private `.cpp` files, behind plain
`std::span`/`DistVector` shapes.

`fuse::petsc::MffdToken` extends `fuse::LinearizationToken` with two
methods the base interface has no notion of: `set_base_point(x, fx = {})`
(the linearisation point `apply_jacobian()` differences around -- `fx`,
when supplied, is `F(x)` already known by the caller, avoiding one extra
evaluation `MATMFFD` would otherwise spend internally recomputing it) and
`set_block_preconditioner(blocks)` (installs a `PCSHELL` composing the
SAME `fuse::num::BlockJacobiPreconditioner::from_blocks` the native
solver's own §6.1 mechanism 6 uses -- reused, not reimplemented, so the
tiling-coverage check (`E-NEWTON-PRECOND-COVERAGE`) and the per-block
apply are identical code on both substrates). `apply_jacobian_transpose`
unconditionally refuses `E-PETSC-NO-TRANSPOSE`: `MATMFFD` has no
transpose (a one-sided finite-difference action is not
invertible-by-transpose), so this token refuses rather than asking PETSc
for an operation that would itself fail less legibly.
`apply_jacobian`/`apply_preconditioner` both refuse `E-PETSC-MFFD-NO-POINT`
before the first `set_base_point()` call, the identical "programmer
error, refused" shape `MffdJacobianAction::apply`'s own
`E-NEWTON-MFFD-NO-POINT` uses.

`fuse::petsc::make_assembled_token` wraps a real, assembled `MATAIJ`
(`MatCreateSeqAIJ` from a caller-supplied triplet list) instead of an
`MATMFFD` -- the counterpart the adjoint-identity check needs, since the
MFFD token above has no transpose to test against at all. Both
`apply_jacobian`/`apply_jacobian_transpose` are genuine `MatMult`/
`MatMultTranspose` calls against the same matrix.

**Initialisation ownership.** `fuse::petsc::ensure_initialized(comm_f)`
initialises PETSc on the communicator `comm_f` (an `MPI_Comm_c2f()`
handle, `fuse::linalg::make_mpi_reducer`'s own convention) ONLY IF
`PetscInitialized()` currently reads false, and records that THIS call
was the one that did it; every later call, from anywhere in the process,
is a no-op success. `finalize_if_owned()` calls `PetscFinalize()` ONLY
when this library's own `ensure_initialized()` was the one that actually
initialised PETSc, and relies on PETSc's OWN internal tracking of
whether IT called `MPI_Init()` to decide whether `PetscFinalize()` should
call `MPI_Finalize()` -- this library's own flag only needs to answer
"did I call `PetscInitialize()` at all", never "did I call `MPI_Init()`",
since the two can legitimately disagree (a process that already had MPI
up before any PETSc call) and PETSc's own bookkeeping already gets that
right. `test/petsc/init_test.cpp` is its own, separate ctest binary
(`petsc.init`), never sharing a process with `petsc.token`/
`petsc.derivative_gate`/`petsc.snes`: `MPI_Finalize()` is irreversible
within a process, so the full init-then-finalize cycle is exercised in
ONE process that does nothing PETSc-related afterward, while every other
petsc test calls `ensure_initialized()` (idempotently) and never
finalises, relying on process exit.

**CMake.** `fuse-coupler-petsc` (src/CMakeLists.txt, the topmost layer
section) builds only under `FUSE_ENABLE_PETSC`, links `PkgConfig::PETSc`
and `MPI::MPI_C` both PRIVATE (PETSc's own `.pc` file's `Cflags` carries
only its own include directory, never MPI's, even though `<petsc.h>`
unconditionally includes `<mpi.h>` -- `cmake/FuseDependencies.cmake`'s own
comment on the `FUSE_ENABLE_PETSC` block has the measured detail: PETSc
links MPI regardless of `FUSE_ENABLE_MPI`, since the PETSc install itself
was configured with MPI). `cmake/FuseDependencies.cmake`'s old
`FUSE_ENABLE_PETSC requires FUSE_ENABLE_MOAB` gate (tier 3 requiring tier
2) is REMOVED: that bundling was for a PETSc+MOAB physics kit (a
mesh-aware solver integration), never for the solver plugin this track
builds, which needs neither MOAB nor a mesh at all -- a tier-0 build
(`FUSE_ENABLE_MPI=OFF`, hence `FUSE_ENABLE_MOAB=OFF` too) can therefore
carry `FUSE_ENABLE_PETSC=ON` alone, confirmed by configuring and building
exactly that combination this track. `tools/check-layering.py` gained one
new layer, `petsc`, placed LAST (topmost, after `abi`): it depends on
`runtime` (`fuse::LinearizationToken`) and `num`
(`BlockJacobiPreconditioner`), both already far below `abi`, and nothing
in the library depends on it, so placing it after every existing entry
needs no argument about exactly where between `runtime` and `abi` it
belongs and guarantees nothing already-ranked could be made to depend on
it by a future insertion between the two. `ALLOWED_PUBLIC_THIRD_PARTY["petsc"]`
is the empty set, like `linalg`'s own entry -- nothing third-party is
named in this layer's public surface at all.

### 10.2 The SNES driver: `fuse::petsc::solve()`

`include/fuse/petsc/solve.hpp` / `src/petsc/solve.cpp`. `SnesOptions`
wraps `fuse::num::NewtonKrylovOptions` directly (not a parallel
declaration that could drift from it) plus the one thing the native
struct has no notion of, `comm_f`. `SNESSetFromOptions()` is NEVER
called: every option comes from `SnesOptions`, FUSE's own configuration,
never the process command line.

| Native (`fuse::num::NewtonKrylovOptions`) | PETSc (`fuse::petsc::solve()`) |
|---|---|
| `atol`/`rtol`/`stol`/`max_nonlinear_iterations`/`max_function_evaluations` | `SNESSetTolerances()` |
| `divergence_tolerance` | `SNESSetDivergenceTolerance()` |
| `mffd.error_rel` | `MatMFFDSetFunctionError()` on the `MatCreateSNESMF()`-built Jacobian |
| `mffd.min_direction_norm` | no PETSc counterpart -- `MATMFFD_WP` itself has no floor on `\|\|v\|\|` (§6.2's own correction) |
| `gmres.restart`/`gmres.max_iterations` | `KSPGMRESSetRestart()`/`KSPSetTolerances()`'s own `maxits`, `KSPSetPCSide(PC_RIGHT)` |
| `forcing` (choice, gamma, alpha, eta_initial, eta_max) | `SNESKSPSetUseEW()` + `SNESKSPSetParametersEW()`, SAME six constants; `alpha2` (an SNES-only parameter this project's struct has no field for) passed as `alpha` again, an approximation never exercised by Choice2 (this track's own test problems) |
| `forcing.eta_min` | no PETSc counterpart at all -- SNES's own forcing term has no floor; a measured dn_heat/Brusselator finding (§8.2, §9.9.2) that the native solver needed `eta_min` to avoid stalling near convergence has NO equivalent mitigation on this substrate, an open question below |
| `line_search.kind` | `SNESLINESEARCHBT`/`CP`/`NONE` (SNES's 3.26 deprecation renamed `BASIC` to `NONE`; same mechanism) |
| `line_search.damping` (Basic only) | `SNESLineSearchSetDamping()` |
| `line_search.armijo_alpha`/`lambda_min`/`max_backtracks`/the safeguard clamps | left at SNES's own defaults (which this project's own defaults were derived FROM, §6.2) -- not individually propagated; a caller that changes one of these away from its default sees no effect on the PETSc substrate, a known, documented gap rather than a silent one |
| `preconditioner_lag` | a `SNESMonitorSet()` callback refreshing the caller's `RefreshPreconditionerFn` every `preconditioner_lag` steps (and once before the first), mirroring the native solver's own split between a matrix-free Jacobian (never lagged) and a lagged preconditioner |

**SNESConvergedReason -> NewtonStatus.** `SNES_CONVERGED_FNORM_ABS` ->
`ConvergedAbsolute`; `SNES_CONVERGED_FNORM_RELATIVE` -> `ConvergedRelative`;
`SNES_CONVERGED_SNORM_RELATIVE` -> `ConvergedStep`;
`SNES_DIVERGED_FUNCTION_COUNT` -> `DivergedFunctionCount`;
`SNES_DIVERGED_LINE_SEARCH` -> `DivergedLineSearch`; `SNES_DIVERGED_DTOL`
-> `DivergedResidualGrowth`; `SNES_DIVERGED_LINEAR_SOLVE` ->
`DivergedLinearSolve`; `SNES_DIVERGED_MAX_IT` -> `DivergedMaxIterations`;
`SNES_DIVERGED_FUNCTION_DOMAIN`/`FUNCTION_NANORINF`/`JACOBIAN_DOMAIN`/
`OBJECTIVE_DOMAIN` -> `DivergedFunctionError`. Every reason with no
native counterpart at all (`SNES_CONVERGED_ITS`, `SNES_CONVERGED_USER`,
`SNES_DIVERGED_INNER`, `SNES_DIVERGED_LOCAL_MIN`, `SNES_DIVERGED_TR_DELTA`,
`SNES_DIVERGED_USER`) falls to `DivergedMaxIterations`, the closest
"did not reach a named convergence test" fallback -- never reached by
`SNESNEWTONLS` with no trust region, no user convergence test and no
inner solve, every configuration this driver builds.

**Scope.** Single-rank only (`x.owned_size() == x.size()`, no ghost
region), refusing `E-PETSC-SOLVE-MULTIRANK` otherwise -- the identical
restriction §4's "Scope" paragraph states for the native seam itself.

*Superseded 2026-10-12 (`000b01c`, KNOWN_ISSUES.md D31/D32): the driver is
multi-rank over `SnesOptions::comm_f`, and only a vector carrying a nonzero
ghost region (`x.owned_size() != x.size()`) is refused, now as
`E-PETSC-SOLVE-GHOSTED`; `E-PETSC-SOLVE-MULTIRANK` no longer exists. The
paragraph above describes the driver as first measured (§10.3).*

### 10.3 Measured on the host (2026-10-11)

The sandbox this track built in cannot complete a bare `MPI_Init()` (MPICH's
debugger-attach mutex assertion in `dbginit.c`), so every number below was
obtained on the development host, in two trees: `build-all` (the `all` preset,
MPI on, the harness calling `MPI_Init`) and a tier-0 tree with PETSc alone
(`FUSE_ENABLE_PETSC=ON`, no FUSE MPI, nothing calling `MPI_Init` before PETSc).
All four rows pass in both: `petsc.init`, `petsc.token` (13 cases),
`petsc.derivative_gate` (5), `petsc.snes` (4).

**Initialisation ownership, both branches measured.** In the tier-0 tree
`ensure_initialized()` finds MPI down, leaves `PETSC_COMM_WORLD` at its
default and `PetscInitialize` calls `MPI_Init` itself; `finalize_if_owned()`
then releases both (`MPI_Finalized` reads 1 afterwards). In the MPI tree it
finds MPI up, sets `PETSC_COMM_WORLD` from the caller's handle and
`PetscFinalize` leaves MPI running. The first draft set `PETSC_COMM_WORLD`
unconditionally, which made `PetscInitialize` skip `MPI_Init` and fail with
`PETSC_ERR_SUP` (56) in any tier-0 process -- on every host, not only in the
sandbox. A third state matters: `MPI_Initialized` stays true after
`MPI_Finalize`, so a second `ensure_initialized()` after a release would
walk into a finalised MPICH and abort; it now refuses `E-PETSC-MPI-FINALIZED`,
and the tier-0 init binary's second case makes that refusal fire.

**Derivative gate on the PETSc tokens.** Adjoint identity on the assembled
`MATAIJ` token: defect 8.9e-16 against a bound of 6.1e-13 (ratio 0.0015).
Taylor-remainder order of the MFFD token's action: 2.018 (quadratic), 2.007
(atan), no plateau reached in the step range. Agreement with the native
`MffdJacobianAction` on the same `F`, `x`, `a`: both rules produced
**exactly the same step**, `h = 2.7191341248866288e-08`, and the two actions
agree to a maximum defect of **0** (bit-identical) against a derived bound of
1.1e-13 -- PETSc's `wp.c` and `fuse::num::MffdJacobianAction` implement the
same formula with the same `error_rel`, and on this problem the same
operations in the same order. The control case for the adjoint gate first
used `x == y`, for which the identity holds for every operator; it now uses
distinct vectors and fails the deliberately wrong transpose as intended.

**SNES against the native solver**, same problems, same `x0`, same options
(Eisenstat-Walker choice 2, BT line search):

| problem | substrate | steps | final ‖F‖ |
|---|---|---|---|
| quadratic | native | 5 | 3.19e-12 |
| quadratic | PETSc  | 5 | 3.37e-12 |
| atan | native | 5 | 2.75e-11 |
| atan | PETSc  | 5 | 2.75e-11 |

Step-1 residual, quadratic: native 6.1571776626130736, PETSc
6.1571779027533156 -- a relative difference of 3.9e-8, consistent with the
two GMRES implementations' different Arnoldi arithmetic at a forcing term of
0.3. The line-search factors agree step for step (0.2576 then 1 on the
quadratic; 0.0293, 0.4186 then 1 on atan); the two traces record the factor
against different steps (the native trace stores the λ that produced the
next iterate, SNES's monitor the λ that produced the current one), which is
a convention difference, not a solver one.

**Mutations.** Four rows in `build-evidence/mutations/petsc.toml`, each
hand-proved on the host's tier-0 tree: the swapped transpose fails exactly
the adjoint case; the base point not communicated to PETSc fails exactly the
Jacobian-agreement case; the swapped preconditioner arguments fail exactly
the block-Jacobi case; the dropped `MPI_Finalized` check aborts the init
binary after its first case.

### 10.4 Open questions for the lead

1. ~~**`forcing.eta_min` has no PETSc counterpart**~~ — **resolved 2026-10-11, by
   measurement.** The Brusselator window on the PETSc substrate did stall exactly
   as the native one had before its floor (`DivergedLinearSolve after 3 Newton
   iterations` on the first window of `--driver runner --accel newton --substrate
   petsc`). `SNESKSPSetParametersEW()` has no floor, and SNES sets the KSP's
   relative tolerance in a pre-solve hook, so the floor is applied in the one place
   that sees EW's chosen value before GMRES judges a residual: a KSP convergence
   test installed when `eta_min > 0` that, at iteration 0, reads the tolerance EW
   set, raises it to `eta_min` if below, and then defers to `KSPConvergedDefault`.
   EW's post-solve reads the clamped value back, as the native solver carries its
   clamped η forward. With it, 50/50 windows converge and the final means match the
   native path to every printed digit (§11.6); the mutation that removes the
   installation fails exactly `example.brusselator.newton_petsc`
   (`petsc-forcing-floor-not-applied`).
2. **`MffdToken::set_base_point()`'s own interface shape** (the concrete
   type carries `set_base_point`/`set_block_preconditioner` beyond the
   base `fuse::LinearizationToken` interface, rather than the factory
   returning the base type with the point fixed at construction) --
   this track's own choice, to let a caller re-linearise a retained token
   in place (what a Newton loop repeating over many steps needs); the
   lead may prefer a narrower interface if nothing beyond this track's
   own tests ever needs the richer one.
3. **The line-search tolerance fields left at PETSc's own defaults**
   (10.2's own table row) -- propagating `armijo_alpha`/`lambda_min`/
   `max_backtracks`/the safeguard clamps individually is mechanical
   (`SNESLineSearchSetTolerances()`) but was not done this track, since
   no test exercises a non-default value on this substrate yet.
4. **The host run itself**: this track cannot name a date by which
   `KNOWN_ISSUES.md` D32 closes, since it depends on a host whose
   `/opt/mpich` (or an alternative MPI) can complete `MPI_Init()` --
   outside this track's own scope to arrange.

## 11. Per-region substrate selection (S7.2b, PetscSelect track, 2026-10-11)

§10 built a SECOND implementation of the Newton seam; this track makes it
*selectable*, per region, through the SAME `NewtonConfig` every graph already
carries -- a `fuse::NewtonSubstrate` (`fuse/model/iteration.hpp`: `Native`,
`Petsc`), mirrored in `fuse::num` (`fuse/num/newton_krylov.hpp`) for the
identical layering reason `LineSearchKind`/`EisenstatWalkerChoice` are
mirrored rather than shared between the two layers (§7's own paragraph on
this). `NewtonConfig::substrate` defaults to `Native`, so every graph written
before this field existed keeps running the library's own JFNK coupler
unchanged.

### 11.1 The registry

`fuse::num` (`newton_krylov.hpp`/`.cpp`) owns a small substrate registry:

```cpp
using NewtonSolveFn = std::function<Result<NewtonResult>(
    const ResidualFn&, DistVector&, const PreconditionerApplyFn&,
    const RefreshPreconditionerFn&, const NewtonKrylovOptions&)>;
void register_newton_substrate(NewtonSubstrate kind, NewtonSolveFn fn);
const NewtonSolveFn* newton_substrate(NewtonSubstrate kind);
```

`NewtonSubstrate::Native` is registered **deterministically on first use** --
a function-local `static` inside `newton_substrate_registry()`
(`src/num/newton_krylov.cpp`), seeded with a trivial lambda adapting
`fuse::num::solve()` itself -- never by a static initialiser in a `.cpp` that
might not be linked into a given binary at all. `NewtonSubstrate::Petsc` is
registered only when `fuse::petsc::install()` (`fuse/petsc/solve.hpp`) has
run, which registers a lambda wrapping `fuse::petsc::solve()`. `install()` is
never called by the library itself; a binary that wants the PETSc substrate
reachable calls it once, before `Runner::initialize()`.

**Why `fuse::num`, not `fuse::runtime`.** `tools/check-layering.py`'s own
`petsc` layer sits ABOVE `runtime` (§10's own header comment: it depends on
`runtime` for `LinearizationToken` and on `num` for `BlockJacobiPreconditioner`,
and nothing depends on it) -- the Runner (`fuse::runtime`) therefore cannot
name `fuse::petsc` at all, in any form, including a registration call. `num`
sits BELOW `runtime`, so the Runner already depends on it (every
`run_region_newton()`/`run_region_newton_window()` call); putting the registry
there is what lets the Runner dispatch through a function pointer it never
has to know the PETSc-ness of, while `fuse::petsc::install()` -- which DOES
know about both `num` and its own substrate -- is the only thing that ever
populates the `Petsc` entry.

### 11.2 The runtime dispatch, and the eager refusal

`src/runtime/iteration.cpp` translates `fuse::NewtonSubstrate` to
`fuse::num::NewtonSubstrate` at one function, `to_num_newton_substrate()`
(the same translation-function pattern `to_num_line_search_kind()`/
`to_num_ew_choice()` already set), and looks up the registry through
`resolve_newton_substrate(region_name, substrate)`, which refuses
`E-NEWTON-PETSC-UNAVAILABLE`, naming the region, when the registry has
nothing for the requested substrate.

This lookup runs TWICE for every Newton-selected region: **eagerly**, inside
`resolve_regions()` (called from `Runner::initialize()`), right after
`check_region_supports_newton()` and before `R.newton = true` -- so a graph
naming an unavailable substrate is refused before the first period, the same
timing every other Newton-region refusal in this file already has; and
**again**, defensively, inside `run_region_newton()`/
`run_region_newton_window()` themselves, immediately before dispatching
through the returned function pointer -- mirroring `check_region_supports_
newton()`'s own "checked anyway, not trusted" posture (its own comment).
Both `run_region_newton()` and `run_region_newton_window()` dispatch through
the SAME lookup and the SAME `NewtonSolveFn` call shape; a region's own
participant count plays no part in which substrate gets looked up, before
or after Sec. 9.10's lift. **Lifted, Sec. 9.10.** `check_region_supports_
newton()`'s own former single-rank restriction (the sentence this paragraph
used to end on) is gone, not narrowed: `RegionRuntime::participants` holds
more than one rank only when a real communicator built it in the first
place (Sec. 9.10's own invariant), so the restriction had nothing left to
guard against once `ParticipantSetReducer` existed to reduce over that
set. Removing it changes neither this lookup nor its two call sites --
substrate selection and rank count are independent decisions on EITHER
substrate, both before and after.

### 11.3 The communicator: `fuse::petsc::install()`'s own decision

**Lifted, Sec. 9.10 (KNOWN_ISSUES.md D31/D32, 2026-10-12).** `fuse::num::
NewtonSolveFn`'s own signature gained a trailing `int comm_f` carrying the
region's own participant communicator (`0` when there is none); the
registered PETSc entry point (`fuse::petsc::install()`, `src/petsc/
solve.cpp`) reads it and falls back to `comm_self_f()` (`MPI_COMM_SELF`)
only when it is `0` -- the UNCHANGED single-rank behaviour described
below, which this paragraph used to describe as the UNCONDITIONAL one.
`fuse::petsc::solve()` itself now builds its `Vec`/`Mat`/`SNES` on
whichever communicator it is handed (`VecCreate`+`VecSetSizes`+
`VecSetType(VECSTANDARD)`, not `VecCreateSeq`), so a multi-rank region's
own PETSc solve reduces for real across its own participants, not just
the native substrate's. See Sec. 9.10 for the full account, including
what remains build-verified only (no host MPI run yet) -- this section's
own history, unchanged below, explains WHY `MPI_COMM_SELF` was the
correct and complete answer before that lift, not merely the easy one.

`fuse::num::NewtonSolveFn`'s own signature carried no communicator
parameter at Stage B/PetscSelect time -- a deliberate choice then, not an
omission. The registered PETSc entry point always built its own
`SnesOptions` with `comm_f = comm_self_f()` (`MPI_COMM_SELF`), never one
threaded in from the Runner's own `RunnerConfig::base_communicator`. That
was sound at the time because every Newton-selected region, on EITHER
substrate, was restricted by `check_region_supports_newton()` to a single
cross-rank participant (§4's own "Scope" paragraph; `KNOWN_ISSUES.md`
D31's own single-rank restriction for the window path, before Sec. 9.10's
own lift) -- there was no cross-rank reduction for a Newton SOLVE to
perform regardless of which rank of the base communicator happened to own
the region, so `MPI_COMM_SELF` was the honest communicator for a problem
this seam guaranteed was rank-local, not a scope-reducing shortcut.

### 11.4 The descriptor: `kGraphVersion` 12 -> 13

`NewtonConfig::substrate` crosses the binary descriptor the same way every
other `NewtonConfig` field does: one more byte (an enum: `Native` = 0,
`Petsc` = 1), appended at the very END of the per-region v12 block (after
`use_block_jacobi_preconditioner`), written and read UNCONDITIONALLY, the
same append-only placement every version bump since v7 uses. The block's own
trailing length grows from 161 to 162 bytes (including the two capability
bytes); every earlier `NewtonConfig` field keeps its wire offset exactly.
`deserialize()`'s own bound check refuses `E-DESC-ENUM` for a substrate byte
past `NewtonSubstrate::Petsc` (ordinal 1), the same `REVIEW.md` R-04 pattern
every other enum on this wire already follows. `test/golden/waist-v13.bin`
is the new frozen fixture (`freeze_fixture()`'s own "newton" region now also
pins `substrate = NewtonSubstrate::Petsc`, a non-default value, alongside
every other `NewtonConfig` member it already pinned); `waist-v12.bin` moves
to the superseded, refused list.

### 11.5 YAML

`newton_config.substrate: Native|Petsc` (the enumerator name verbatim, as every
other enum key on this page; the examples' `--substrate native|petsc` flag is a
separate, lowercase spelling). An unrecognised
name is refused `E-YAML-PARSE`, the identical shape `line_search_kind`'s own
unknown-name refusal already uses.

### 11.6 dn_heat and the Brusselator by the PETSc path

`examples/dn_heat/main.cpp` gained `--newton-petsc`: a SEPARATE code path
(not one more row in the default table), run ONLY when that flag is passed,
and compiled in at all only under `FUSE_HAVE_PETSC` -- so the default
invocation's output (every existing ctest row) is byte-identical whether or
not the binary was built with `FUSE_ENABLE_PETSC`. It calls
`fuse::petsc::install()`, then runs the SAME affine graph the native
`newton` row does (`beta = 0.0`) with `NewtonConfig::substrate = Petsc`,
plus the nonlinear `newton-nl-petsc` row (`beta = 0.1`, informational, like
`newton-nl`), and checks the PETSc result against the native one to a bound
derived from both solvers' shared `rtol` and dn_heat's own affine Jacobian
structure (the per-station 2x2 `{{1, r_i}, {-1, 1}}`, `fixed_point_residual_
test.cpp`'s own derivation) plus the §10.3-measured 3.9e-8 relative
inter-solver difference -- see that file's own comment for the full
derivation. Registered as `example.dn_heat.newton_petsc`
(`examples/CMakeLists.txt`), gated by `FUSE_ENABLE_PETSC`.

`examples/brusselator/main.cpp` gained `--substrate native|petsc` (default
`native`, read only under `--accel newton`) and an unconditional, idempotent
`fuse::petsc::install()` call early in `main()` (guarded by
`FUSE_HAVE_PETSC`, a no-op for every `--substrate native` invocation).
`test/petsc/brusselator_newton_test.cpp` (NEW, its own dedicated binary --
NOT one more `TEST()` in `test/verification/brusselator_runner_test.cpp`,
since a PETSc call that segfaults in this sandbox, §11.7 below, takes
the whole PROCESS down, which would otherwise take every other,
PETSc-independent case sharing that binary down with it, the SAME reason
every `petsc.*` test is already its own binary) gained
`BrusselatorNewtonPetsc.ConvergesAgreeingWithNative`, comparing the
PETSc-substrate run against the native one (the SAME Rig/graph shape
`NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner`
uses, duplicated rather than shared per that file's own "near-duplicate"
convention) to a derived bound, WITHOUT asserting the two take the same
number of outer Newton iterations (SNES's own GMRES/line-search
implementation is not required to follow the identical path, §10.3's own
measured step-1 difference). Registered as `petsc.brusselator_newton`
(`test/CMakeLists.txt`), gated by `FUSE_ENABLE_PETSC`.

### 11.7 Measured on the host (2026-10-11)

All `petsc.*` rows and both example rows pass in `build-all` (PETSc on, MPI on)
and the serial suite is 191/191 there. `example.dn_heat.newton_petsc`: the PETSc
row converges in four steps per period like the native one and agrees with Picard
to 2.717e-10 K (native 6.321e-11 K), max final ‖F‖ 1.649e-10 (native 4.159e-11).
`example.brusselator.newton_petsc` (five windows, `--check`) and the 50-window run:
50/50 converge on both substrates, final means 0.668994004315 / 2.8189403867 on
both, balance errors 1.14e-13 / 4.69e-12 (PETSc) against 1.16e-13 / 4.66e-12
(native); first-window residual history identical to six digits, SNES taking one
more outer iteration by its convergence-test placement. The forcing floor of
§10.4(1) was required for the Brusselator and is proved load-bearing by mutation;
dn_heat does not reach the regime. The tier-0 refusal
`E-NEWTON-PETSC-UNAVAILABLE` fires without PETSc
(`NewtonRegion.ARegionSelectingAnUnregisteredSubstrateIsRefusedAtSetup`).

## 12. NewtonNorm and NewtonResidualPath (NewtonModel track, 2026-10-12)

Two independent, opt-in `NewtonConfig` fields, both appended at the
struct's own declared end (the append-only convention §7's own
`kGraphVersion` history follows): `norm` (`Plain` default, `Measure
Weighted`) scales the stacked residual before every Newton stopping test
in §6/§9/§10 reads it; `residual_path` (`Advance` default, `Direct`)
changes HOW `G(x)` is obtained for those same tests, never what they
compute once obtained. Both default to their pre-existing behaviour, so
every region declared before this section existed keeps running
unchanged -- confirmed by the full pre-existing suite (descriptor,
yaml_io, newton_region) passing unmodified against every new byte
`kGraphVersion` 14 appends.

### 12.1 `NewtonNorm::MeasureWeighted`: the diagonal `D`

Built once per region, at `resolve_regions()` (same point §7's own
descriptor-driven setup runs), never per Newton step: `D_i = sqrt(w_i /
sum_block_w) / s_b`, where `w_i` is the entity measure (length, area or
volume) entry `i` lives on -- the IDENTICAL per-entity weight
`Runner::Impl::weights_for()` already derives for the Constrain/
conservation machinery, zero for an unowned or ghost entry -- `sum_block_w`
the SAME weights reduced over the block's own owned entries through the
region's own `Reducer` (today always `make_serial_reducer()`: §9.4's own
single-participant restriction applies here too, so a cross-rank sum is
this track's own open question, §12.5 below), and `s_b` the tolerance
scale `fuse::ConvergenceCriterion`'s own measures declare for that block.

**Which measure names a block's own scale.** `E-ITER-MEASURE`
(`match_convergence_measures()`, `src/ir/checkpoint.cpp`) already accepts
THREE equally valid names for one block: its channel's own name, its
provider's source field, or its consumer's target field -- because
dn_heat's own convention (and every test fixture in this repository)
names a convergence measure after a channel's PROVIDER-side export (for
example `"q_wall"`), never its consumer-side import (`"q_wall_in"`).
`NewtonNorm::MeasureWeighted`'s own tolerance-scale lookup matches the
IDENTICAL three names, for the identical reason: a narrower match (the
consumer field name alone) would refuse every graph already written this
way for no actual gain in safety. What it does refuse, eagerly, with the
new `E-NEWTON-NORM-SCALE`: no match at all (defensive -- `E-ITER-MEASURE`
already guarantees at least one via the builder, so this is unreachable
except through a hand-built `IterationRegionDecl`), or two-or-more
matches with DIFFERING tolerances, a genuine ambiguity this option will
not silently resolve by picking one (`NewtonNorm.AmbiguousTolerance
ScaleRefusesAtSetup`, `test/runtime/newton_norm_test.cpp`, fires it on a
block covered twice, once by its channel name and once by its consumer's
target field, with two different tolerances).

**The preconditioner's own composition.** A region ALSO requesting
`NewtonConfig::use_block_jacobi_preconditioner` builds a block-Jacobi `M`
approximating `J^-1` (§9.5's own construction, reused here for an
ordinary, non-window region -- this track's own extension; §9.5's region
was the only caller before this). With the scaled operator `D*J`, the
right preconditioner the Krylov solve actually needs is `(D*J)^-1 = J^-1
* D^-1`, not a bare `M`: composing `precond_apply` as `v -> M(D^-1 * v)`
is the fix (`compose_measure_weighted_preconditioner()`,
`src/runtime/iteration.cpp`); leaving it bare makes the preconditioned
operator `~D` instead of `~I`, degrading the Krylov solve by `D`'s own
spread rather than removing it.

MEASURED (`NewtonNorm.BlockJacobiPreconditionerWithExactTokensConverges
UnderMeasureWeighted`, two synthetic components whose own
`LinearizationToken` is EXACT for their own affine map, so any
degradation observed is the composition's own, not an approximate
token's error): correctly composed, the solve reaches its fixed point in
ONE Newton step, residual history `[1000000.005, 0.02969315636]`.
Removing the composition (`build-evidence/mutations/newton-model.toml`'s
own `newton-norm-precond-not-composed-with-inverse-diagonal` row) changes
this to TWO steps, `[1000000.005, 126.4799515, 0.01571608272]` -- both
runs still converge (this problem's own exactly linear Jacobian reaches
the fixed point via GMRES breakdown either way), the mutation changes the
PATH, matching §9.6's own point about a real preconditioner changing the
path to convergence rather than whether convergence happens, never
asserted of this specific case before being measured.

### 12.2 Scale invariance and unit independence, measured

**(i) Tight-forcing agreement (informational only, Round 2: see (vi)
below for the test that replaced this one's own assertion).** Solving
the SAME affine dn_heat problem twice, once `Plain` once
`MeasureWeighted`, both under a near-exact constant Eisenstat-Walker
forcing term, should give a CLOSE Newton step up to the forcing term's
own achieved GMRES accuracy: `(D*J) dx = -(D*F)` is `J dx = -F`
left-multiplied by the invertible `D`, the SAME equation, so an EXACT
linear solve gives the identical `dx` regardless of `D` -- but GMRES at a
finite, nonzero forcing tolerance does not solve exactly, and no
closed-form bound on the resulting `dx` difference as a function of that
forcing tolerance alone was ever derived (the Round 2 review's own
finding: an earlier version of this test asserted `max diff < 1e-2`
against the measured `6.78e-4` below with no derivation connecting the
two numbers). Kept as `NewtonNorm.
WeightedIteratesAgreeWithPlainUnderTightForcingInformationalOnly` --
printed, not asserted on. MEASURED (`forcing_eta_constant = 1e-3`, two
orders of magnitude above the forcing floor this problem's own
unpreconditioned, matrix-free GMRES solve can actually verify -- `1e-10`
and even `1e-6` both made the FIRST linear solve fail outright,
`DivergedLinearSolve`, `iterations = 0`, the Krylov solver unable to
VERIFY the requested relative accuracy within its own
1000-iteration/restart-30 budget on this unpreconditioned problem, a
tighter floor than the naive `sqrt(eps_mach) ~= 1.49e-8` MFFD estimate
predicts, for the same class of reason §9.9.2 documents on a different
problem): `max |q_plain - q_weighted|` over both fields, `0.00067754`.

**(ii) Unit independence of the stopping norm.** Rescaling one field's
own units by `1e4` (a pure unit conversion: the TRUE fixed point is
unchanged, only the NUMBERS representing that field's value and
tolerance are `1e4` bigger) changes the PLAIN norm's own `||F(x=0)||` by
roughly the rescale factor (MEASURED: `2.82843` at scale 1, `20000` at
scale `1e4` -- order `1e4`, not exactly it, because the OTHER field's own
fixed contribution stops being negligible only once the rescaled one
dominates) but leaves the WEIGHTED norm UNCHANGED to round-off
(`1.41421e+06` at both scales) -- each block is divided by its own
tolerance, scaled by the SAME factor as the field it measures, before
being combined with the other, so the SAME physical accuracy in
different units reads the same combined number
(`NewtonNorm.InitialResidualNormIsUnitIndependentOnlyUnderMeasureWeighted`).
An end-to-end comparison of Newton's own ITERATION COUNT under this
rescaling (rather than the stopping norm directly) was tried first and
found NOT to isolate this effect on dn_heat's own affine map: this
problem's per-entry affine structure is close enough to an exactly linear
Jacobian that a handful of Eisenstat-Walker-forced steps closes most of
the remaining error regardless of scale up to `1e4`, and pushing the
scale to `1e6` to force a difference instead hits the SAME matrix-free
precision floor on both norms, a different effect than the one under
test -- recorded here as the reason this measurement targets the
stopping norm directly rather than the iteration count it drives.

**(iii) The scaled norm equals the criterion's own measure-weighted RMS
divided by its tolerance.** `D_i * F_i = sqrt(w_i) * F_i / s_b`,
confirmed to agree with `fuse::num::field_norm(MeasureWeightedRms,
F).value / s_b` to `1e-15` relative
(`NewtonNorm.ScaledResidualNormMatchesFieldNormDividedByTolerance`) -- the
SAME root-mean-square error §4 of this doc already defines, just divided
by a declared tolerance rather than left in the field's own physical
units.

**(iv) Mesh refinement.** Refining the interface mesh `4x` (uniform
`generated:line:N`, confirmed `1/N` per entry, `src/backend/
serial_backend.cpp`) MEASURED: the plain norm's own `||F(0)||` scales by
`sqrt(4) = 2` (within 5%), the weighted one stays within 5% of
UNCHANGED (`NewtonNorm.MeshRefinement4xLeavesWeightedResidualStable`) --
the same resolution-independence §2/§3 of this doc already establish for
`field_norm()`, carried through to Newton's own stopping test.

**(v) Closing a gap the above tests leave open.** Tests (iii) and (iv)
both recompute this track's own `D` formula INDEPENDENTLY of
`build_newton_measure_weights()`, by design, as a cross-check of the
MATH -- neither one calls it, so neither exercises the PRODUCTION array a
real Newton solve actually scales `F` by, and a mutation to that function
would reproduce itself in the test's own hand-derived comparison and go
uncaught (measured directly: `build-evidence/mutations/newton-model.
toml`'s own `newton-norm-weights-not-applied` and `newton-norm-block-
scale-dropped` rows do not fail (i)/(iii)/(iv) at all). `Runner::debug_
newton_scale()` (new, TEST/DIAGNOSTIC-ONLY, the same convention `debug_
window_member_snapshots()` already established, `fuse/runtime/
runner.hpp`) reads `RegionRuntime::newton_scale` directly;
`NewtonNorm.ProductionDiagonalMatchesTheFormulaEntryForEntry` compares it
against the formula entry for entry and is what those two mutation rows
actually fail against.

**(vi) Round 2 (GAP 3): a derivable bound, replacing (i)'s own
unjustified one.** The Round 2 review found (i)'s original `max diff <
1e-2` bound, set above the `6.78e-4` it was actually measured against
with no derivation connecting the two, unacceptable. `NewtonNorm.
ExactDenseSolveAgreesToRoundoffTimesConditionNumber` (new) replaces it
with a bound that IS derivable: the exact `n=8` Jacobian of the affine
`DiagonalFixture` pair, assembled through `DiagonalComponent::
ExactToken::apply_jacobian()` (production `LinearizationToken`
machinery, not a hand-written shortcut) rather than approximated by
GMRES/MFFD, solved independently for `J dx = -F` and `(D*J) dx = -(D*F)`
via a hand-written dense Gaussian-elimination solver
(`dense_solve()`, this test file's own anonymous namespace) -- an EXACT
linear solve, so the two `dx` must agree to round-off times the
operator's own condition number: `max|dx_plain - dx_weighted| <=
safety_factor * eps_mach * cond_inf(J) * max(||dx||, 1)`, `cond_inf`
computed by self-inverting `J` through the same solver, `safety_factor =
100` chosen BEFORE running the test (not tuned to the observed result).
MEASURED: `max|dx_plain - dx_weighted| = 2.84217e-14`, `cond_inf(J) =
1724.8`, bound `= 6.28091e-09` -- satisfied with roughly 4.5 orders of
magnitude of margin (the measured difference is ~4.5e-6 of the bound,
not a near miss).

**(vii) Round 2 (GAP 2): Plain's own masking failure, shown end-to-end
against an analytically known root.** §12.1's own derivation motivates
`MeasureWeighted`; this closes the gap between that motivation and an
actual demonstration that Plain's OWN stopping test is satisfied while
a badly-scaled field is still far from converged -- not just that the
INITIAL residual looks unit-dependent (test (ii) above), but that the
SOLVE ITSELF stops too early. A synthetic two-field region,
`MixedScaleFixture` (`test/runtime/newton_norm_test.cpp`, new): field
"a" linear (`DiagonalComponent`, re-used), field "b" GENUINELY nonlinear
(`QuadraticComponent`, new: `B_out = coeff*A_in^2 + offset`), "a"'s own
units `1e4` times "b"'s (`tol_a = 1.0`, `tol_b = 1e-4`), coupled into a
fixed point whose exact root is known analytically (the smaller root of
the quadratic the two update rules substitute into one another into).
`NewtonNorm.PlainNormMisjudgesConvergenceOnTheNonlinearTwoFieldRegion`
(new): with `NewtonConfig::atol = 1.0` applied to each norm's own
combined residual, MEASURED -- Plain stops at Newton iteration 1 (its
own huge, A-dominated residual already satisfies `atol` there) with "b"
still `320.037` times its OWN declared tolerance away from the true
root; MeasureWeighted continues to iteration 3, where "b" is
`0.000126888` times its own tolerance away -- actually converged, not
merely reported as such. (A first attempt at this fixture undercounted
`DiagonalComponent::coeff_at(i)`'s own `(0.5 + i/n)` per-entry multiplier
-- a fixed `0.5` at this fixture's `n=1` -- in the hand-derived exact
root; found by re-reading that class's own source after the fixture's
two norms converged to a value neither matched the (wrongly-derived)
root, fixed by doubling the constructor argument passed to it, not by
loosening the test.)

### 12.3 `NewtonResidualPath::Direct`

`Advance` (default, unchanged): `G(x)` is obtained the way every region
before this option existed obtains it -- deliver `x`, re-advance every
checkpointed member under `save_state()`/`restore_state()`, run the
region's own body. `Direct` instead calls each providing member's own
`ImplicitComponent::evaluate_residual()` in place of `advance()`, with NO
`save_state()`/`restore_state()` anywhere in the call (that method's own
contract already guarantees it leaves the component bit-identical to how
it found it, so there is nothing to roll back) --
`Runner::Impl::evaluate_region_residual_direct()`
(`src/runtime/fixed_point_residual.cpp`), a close structural twin of
`evaluate_region_residual()` (§4), writing each provider's own
`residual_out` into the registered Source-node storage `advance()` would
otherwise have written, so the region's existing body instructions carry
it downstream exactly as they already carry `advance()`'s own output.
Ordinary (non-window) regions only: a window-scoped region's own sweep
has coroutine-resident per-member state (§9.2's own table: clock,
straddle flags, the pass-scoped predictor cache) that `advance_window()`
mutates regardless of how a member's own physics is evaluated, and
`Direct` never calls that function at all, so it is refused eagerly, at
setup, with `E-NEWTON-DIRECT-WINDOW`
(`NewtonDirect.AWindowScopedRegionSelectingDirectIsRefusedAtSetup`). The
committing verification sweep at the converged iterate ALWAYS advances
real state through the REAL `evaluate_region_residual()` (Advance),
regardless of which path the solve itself used -- the Runner's own
persisted state is always reached by at least one genuine `advance()`
call, per period.

**Bit-identity, measured.** `Runner::region_fixed_point_residual_direct()`
(new, TEST/DIAGNOSTIC-ONLY, the `Direct` sibling of `region_fixed_point_
residual()` -- §4 -- built for exactly this comparison) against the
Advance seam, at several `x` (all-ones, a per-entry ramp, a negative
vector), on both the affine map and the nonlinear `k(T)` variant
(`beta = 0.1`): `F(x)` agrees to the bit, `EXPECT_EQ`, not a tolerance
(`NewtonDirect.FMatchesAdvanceBitIdenticallyOnTheAffineMap`,
`...OnTheNonlinearConductivity`). The full Newton trajectory (iterations,
residual history) run through `run_period()` under each path separately
also agrees to the bit
(`NewtonDirect.FullNewtonTrajectoryIsIdenticalBetweenAdvanceAndDirect`),
as does the persisted state (`q`/`u`) the committing sweep leaves behind.

**The rollback question, measured rather than assumed.** This track's own
plan asked whether a member WITHOUT rollback could run Newton on `Direct`
specifically, since `Direct` itself never calls `save_state()`/
`restore_state()` on anything. MEASURED: no -- `verify_rollback_support()`
(`src/runtime/checkpoint_support.cpp`) unconditionally refuses `E-ITER-
NOROLLBACK` for EVERY checkpointed member of ANY iteration region, at the
single call site inside `resolve_regions()`, BEFORE any Newton-specific
code (Advance's own setup, `Direct`'s own `build_newton_direct_members()`)
is ever reached -- because `step_period()`'s own generic first tick
(§4's own top comment: "By the time control reaches the region the
participants have already advanced once") needs that SAME rollback to
undo itself regardless of which residual path the region's own Newton
solve will use. A synthetic component with `kCapabilityResidual` set and
`capabilities()`'s own rollback bit CLEARED fires `E-ITER-NOROLLBACK` for
BOTH `residual_path: Direct` AND `residual_path: Advance`, identically
(`NewtonDirect.AMemberWithoutRollbackRefusesNewtonOnDirect`,
`...TheSameRefusalFiresForAdvanceTooProvingItIsPathIndependent`) --
proving this is `verify_rollback_support()`'s own pre-existing,
path-independent rule, not a limitation `Direct` introduces, and NOT a
benefit `Direct` grants either. Lifting this restriction, if ever wanted,
is a change to the generic first-tick's own rollback requirement, not to
`NewtonResidualPath` -- out of this track's own scope, recorded as an
open question (§12.5) rather than attempted.

**Cost, measured.** One `F` evaluation, Advance vs Direct, median of 11
repetitions after a warm-up call, `n = 8` (dn_heat's own affine map):
`advance = 212000.0 ns`, `direct = 210375.0 ns`, ratio (direct/advance)
`0.9923`, `load1 = 6.72` at the time of this run (this host's own
12-core machine, with two other tracks building concurrently -- the
task's own standing condition). An earlier run under heavier concurrent
load (`load1 = 8.41`) measured a DIFFERENT ratio, `1.1485`, flagged
`NOISY` by this same test's own `load1 > 8.0` threshold -- recorded
honestly rather than discarded: on this shared, concurrently-building
host, the measured ratio moves with system load, and the two numbers
above do not by themselves establish that Direct is reliably faster or
slower than Advance; both evaluations do the SAME work (deliver `x`, run
one component call, run the region's own body) minus or plus a
save/restore bracket over a handful of doubles, so a difference much
larger than measured here, in either direction, would be the surprising
result.

### 12.4 Descriptor v14

Both fields appended at `NewtonConfig`'s own declared end, after
`substrate` (v13's own last field) -- `kGraphVersion` 13 -> 14, following
v12's and v13's own append-only pattern exactly: `write()`/`read()` add
two bytes (`norm`, `residual_path`), each bound-checked against its own
enum's cardinality with `E-DESC-ENUM`; every earlier version's own byte
layout is untouched. `test/golden/waist-v14.bin` (new, generated via
`fuse_write_golden_descriptor`) replaces `waist-v13.bin` as the frozen
byte comparison (`Descriptor.MatchesTheFrozenWaistBytes`); `waist-v13.bin`
moves to the superseded list alongside v12/v11/..., still decodable, per
the SAME backward-compatibility guarantee v13 itself established over
v12. `freeze_fixture.hpp`'s own second Newton region (the one exercising
every non-default `NewtonConfig` field, per v13's own convention) selects
`norm: MeasureWeighted, residual_path: Direct` -- the two non-default
values this version adds, proving they round-trip through the descriptor
alongside every earlier non-default field. YAML: `norm`/`residual_path`
keys, enumerators spelled verbatim (`Plain`/`MeasureWeighted`,
`Advance`/`Direct`), an unknown name refused `E-YAML-PARSE`
(`YamlIo.RejectsUnknownNewtonConfigNorm`, `...ResidualPath`).

### 12.5 Open questions for the lead

**Window-scoped `MeasureWeighted`, Round 2 GAP 1: verified against the
real Brusselator, and a real bug found and fixed in the process.**
`decide_window_convergence()` (`fuse/runtime/window_convergence.hpp`)
is, read directly rather than assumed, an UNWEIGHTED flat sum of squared
per-block samples across owning ranks -- no spatial measure weighting of
its own, and no weighting by which substep within the window a sample
belongs to; this still directly contradicts this track's own ORIGINAL
plan, which assumed window regions already use "the weights it already
uses" -- they do not use any, and `build_newton_measure_weights()`'s own
`window_scoped` branch repeats each element's own spatial measure
UNIFORMLY across every sample for exactly that reason, unchanged from
the Stage A framing above.

What Round 1 could not verify, Round 2 did, with a new file
(`test/verification/brusselator_newton_norm_test.cpp`), built from the
same Rig/NewtonConfig conventions as `brusselator_runner_test.cpp`'s own
`NewtonConvergesInAPinnedIterationCountWithTheBlockJacobiPreconditioner`,
against two grid sizes: that file's own small 5x5/7x7 one (fast, for the
convergence and GMRES-count checks) and the EXAMPLE's own real defaults,
16x16/24x24 (the one configuration §9.1's own `n=4864` is about).
Running `NewtonNorm::MeasureWeighted` against that real grid for the
FIRST time (not the ordinary-region-only tests §12.2 ran) immediately
refused every window with `E-NEWTON-NORM-SCALE`: `weights_for()` is
called with `elements = slots[b.feed_arc].elements` (the mesh's own
entity count) and `components = slots[...].components`, and for a
window-scoped consumer arc `src/ir/window_lowering.cpp` sets
`components = field_components * window_sample_capacity()` -- so
`weights_for()` had ALREADY replicated each element's own measure across
every sample column (`w_base.size() == b.length`, not `b.length /
capacity`) before `build_newton_measure_weights()` ever saw it; the
Round 1 code treated `w_base` as purely spatial and re-expanded it by
`capacity` a SECOND time, double-counting the window dimension in both
the per-entry normalisation (`local_sum *= capacity`, now removed) and
the length check it failed. The fixture Round 1 verified against
(`NewtonDnHeatFixture`'s own window-scoped test, where this doc's prior
revision said "only the ordinary-region tests... ran") evidently never
exercised a window with `capacity > 1` for this norm -- EVERY real
multi-sample window (which is to say, every window this feature would
ever actually be used on) was broken from the day it was written, until
this gap's own verification run caught it. Fixed in
`build_newton_measure_weights()` (`src/runtime/iteration.cpp`): `w_base`
is now read as already including the window-sample dimension when
`window_scoped`, with a defended (not merely assumed) check that it really
is uniform across each element's own sample run, refusing
`E-NEWTON-NORM-SCALE` rather than silently misweighting if a future
change to `weights_for()`'s own replication order ever breaks that.

MEASURED, after the fix, at the real grid (`n_u = 256`, `cap_u = 11`,
`cap_v = 8`, `newton_n = 4864` -- §9.1's own hand-derived total,
confirmed exactly):
- **(a) Convergence and agreement.** Plain and MeasureWeighted both
  converge every window (small grid: 4 Newton iterations each);
  `max|u_plain - u_weighted| = 1.332268e-15`, `max|v_plain - v_weighted|
  = 2.220446e-15` -- round-off, not a derived bound (none was attempted
  for this nonlinear reaction-diffusion problem; a sanity bound of
  `1e-6` is asserted instead, three orders looser than
  `brusselator_runner_test.cpp`'s own `1e-9` pin for its different
  (preconditioner-choice) comparison).
- **(b) The production `D`, entry for entry, at `n=4864`.** Checked
  against a closed form derived from the SAME uniform-mesh exploit
  `NewtonNorm.ProductionDiagonalMatchesTheFormulaEntryForEntry` uses for
  dn_heat (`NativeGrid`'s own unit-square grid makes every cell's
  measure identical, cancelling out of `D_i = sqrt(w_i/sum_w)/s_b`
  regardless of that measure's own value): `D_i = sqrt(1/(capacity *
  n_u))/s_b` for every entry, uniform across every sample at a fixed
  element -- PASSED, entry for entry, both blocks ("u2v", "v2u").
- **(c) GMRES cost, composed vs. no preconditioner at all.** MEASURED,
  not assumed: block-Jacobi WITH the correct `D^-1` composition costs
  MORE total GMRES work here (42) than no preconditioner at all (8), and
  needs one more outer Newton iteration (4 vs. 3) -- the SAME direction
  `brusselator_runner_test.cpp`'s own header already measured at the
  full-size grid ("fewer outer iterations... (3, not 4)... without" the
  preconditioner); an initial assertion written the OTHER way (composed
  cannot cost more) was wrong and was corrected to a regression pin
  reporting both numbers as measured, before the mutation row below was
  proved.
- **The window path's own `D^-1` composition, mutation-proved.** A NEW
  row, `newton-norm-window-precond-not-composed-with-inverse-diagonal`
  (`build-evidence/mutations/newton-model.toml`), at `run_region_newton_
  window()`'s own call site (distinct from `run_region_newton()`'s,
  proved in §12.2(v)): dropping the composition moves the composed
  case's own pinned numbers (4 Newton iterations / 42 GMRES -> 3 / 24),
  confirming this call site is live and this composition matters to it
  on a REAL window, not only on the ordinary-region synthetic fixture
  §12.2(v) proved it against.

No further open question remains for this gap: the window D was wrong,
it is now fixed, and the fix is verified against the real example and
mutation-proved on the window path specifically, not merely inferred
from the ordinary-region proof.

**Cross-rank reduction of `sum_block_w`.** §9.4's own single-participant
restriction (`check_region_supports_newton()`) means `build_newton_
measure_weights()`'s own `reducer.sum()` call is, today, unconditionally
`make_serial_reducer()` -- never a reduction over more than one rank's own
contribution. Lifting §9.4's own restriction (this doc's own long-stated
next step for the window track) would need `R.newton_reducer` to become a
real cross-rank reducer for `MeasureWeighted` to keep computing `D`
correctly over a genuinely multi-participant region; this track did not
need to make that change and did not.

**The rollback restriction's own scope.** §12.3's own measured finding
(`E-ITER-NOROLLBACK` fires identically for Advance and Direct) is a
property of `step_period()`'s own generic first tick, not of either
residual path -- lifting it, if ever wanted, means teaching that generic
tick itself to skip the rollback bracket for a region whose OWN residual
path never needs to undo it (`Direct` specifically), which is a change to
shared scheduling code well outside this track's own owned files, and is
recorded here rather than attempted.
