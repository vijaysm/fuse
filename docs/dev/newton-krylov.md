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
