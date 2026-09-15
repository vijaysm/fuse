# SOMA — Scalable Operators for Multiphysics Applications

**Design specification, v1**
**Date:** 2026-09-02
**Status:** approved for implementation planning
**Location:** `modules/coupler/`
**Namespace:** `soma::`

---

## 1. Purpose and scope

SOMA is a C++20 library for coupling multiphysics components. It represents a
coupled simulation as a **network graph that flows in time**: nodes are
components, edges carry the semantics of how fields are accumulated in time,
merged in space, extrapolated to resolve nonlinearity, and iterated to reduce
numerical error.

It replaces nothing. The existing `modules/coupe` (PETSc + MOAB, autotools,
C-style API) is not reused, extended, or ported.

### 1.1 Goals

- **Robust** — conservation, accuracy order, and unit consistency are checked by
  the framework, not by convention.
- **Flexible** — new coupling numerics are new operators plus lowering rules; the
  core does not change.
- **Scalable** — MPI with disjoint component rank sets, n→m redistribution,
  precomputed remap weights, allocation-free hot loop.
- **Portable** — the core builds and its full authoring/lowering/scheduling path
  is testable with no third-party dependency at all.

### 1.2 Non-goals for v1

Explicitly deferred, with the design seam left open:

- JFNK / monolithic residual assembly (§5.5)
- True 4D intersection for moving meshes (§7.2, Tier 2)
- MPMD port-connect transport (§6, mode 3)
- A running coupled Fortran demo (the C ABI and shim ship and are unit-tested)
- GPU execution beyond a device-pointer-clean design

### 1.3 Name

SOMA collides with `613.soma_s` in SPEChpc 2021 (SOft coarse-grained Monte-carlo
Acceleration, Schneider & Müller, *Comput. Phys. Commun.* **235** (2019) 463).
The collision is known and accepted: different domain, no linkage conflict.

---

## 2. Architecture: three representations, two passes

```
   Authored              Lowered                  Compiled
┌──────────────┐     ┌──────────────┐        ┌──────────────┐
│CouplingGraph │ ──▶ │ OperatorDAG  │  ──▶   │ExecutionPlan │ ──▶ Runtime
│ components   │lower│ primitive    │schedule│ flat instr.  │
│ channels     │     │ ops, SSA     │        │ buffers      │
│ semantics    │     │ regions      │        │ comm plan    │
│ time config  │     │ time-typed   │        │ immutable    │
└──────────────┘     └──────────────┘        └──────────────┘
   YAML ⇄ C++ ⇄ Python      analysis              built once
```

**Layer 1 — `CouplingGraph`.** The only representation users author, the only
one serialized. Pure data; no execution, no MPI, no MOAB. Round-trips:
YAML → graph → YAML, Python → graph → YAML.

**Layer 2 — `OperatorDAG`.** Single-assignment primitive operators with typed
arcs. Two additions beyond plain SSA: **regions** (an iteration region owns a
convergence measure, an accelerator, and a *computed* checkpoint set) and
**time-level typing** (every arc carries the time level of its value).

**Layer 3 — `ExecutionPlan`.** Flat, immutable, rank-local instruction stream.
Buffers pre-allocated, communication schedules pre-built, remap weights
precomputed, iteration regions as explicit loop boundaries. Built once at init;
the hot loop interprets a vector of instructions with no virtual dispatch per
DOF, no map lookup per exchange, no allocation.

### 2.1 Why three layers

Each layer is the right shape for a different job: declarative ergonomics for
authoring, explicit ordering for correctness analysis, flat pre-planned streams
for performance. Collapsing any two forces one job to be done badly.

The decisive payoff: because operator ordering is explicit at Layer 2,
`Remap ∘ Accumulate` and `Accumulate ∘ Remap` are **different DAGs**. For a
monotone limiter or a nonlinear flux these are different answers, and a
verification pass can report which one was built and whether it conserves.

### 2.2 Why this survives four integration modes

`Transport` is one primitive with four implementations (in-process copy, MPI,
MPMD port, LowFive interception). A component in a separate executable differs
from an in-process one only in *which `Transport` implementation the lowering
pass selects*. Nothing else in the DAG changes. This is the entire mechanism by
which intrusive and non-intrusive coupling share a runtime.

### 2.3 Introspection

Every layer dumps to Graphviz and JSON: `--soma-dump-graph`, `--soma-dump-dag`,
`--soma-dump-plan`. The plan dump is the coupled scheme made visible, and is a
deliverable in its own right (cf. NUOPC, which needs a runtime Compliance
Checker to reconstruct what SOMA prints).

---

## 3. Header-only tiering and the narrow waist

### 3.1 Tiers

| Tier | Contents | Form | TUs |
|---|---|---|---|
| A. Vocabulary | IDs, enums, units, `Result`, `Span`, time types, `FieldSpec` | header-only, no deps | 0 |
| B. Authoring | `CouplingGraph`, `Channel`, builder, `OperatorDAG`, validation | header-only, STL only | 0 |
| C. Kernels | `Accumulate`, `Relax`, `Extrapolate`, remap matvec, norms | header-only templates | 0 |
| D. Core | lowering, scheduler, plan interpreter, diagnostics | compiled | ~6 |
| E. Backends | MOAB mesh, MBTR remap, transports, PETSc, accelerators | compiled, pimpl | ~10 |
| F. Front-ends | C ABI, Fortran shim, YAML, pybind11 | compiled | ~4 |

Target: ~20 translation units. Full header-only is impossible and undesirable —
the C ABI for Fortran is by definition a compiled artifact, the LowFive VOL
plugin is a runtime-loaded shared object, and hiding MOAB/PETSc/MPI types out of
public headers is directly opposed to header-only for anything that touches them.

Techniques: `extern template` with explicit instantiation of common
(value type × execution space) pairs; one pimpl per *backend*, not per class;
table-driven operator registration; fine-grained headers with an optional
umbrella `soma/soma.hpp` that nothing internal includes.

C++20 modules are deliberately not used in v1 — Cray/NVHPC/oneAPI support is the
weak link on target machines. The layout is module-compatible later.

### 3.2 The narrow waist

**Decision: the serialized graph is the ABI boundary.**

The header-only authoring layer builds a `CouplingGraph` and emits a flat POD
descriptor (`soma_graph_v1`). The compiled core consumes only that descriptor,
through the same C ABI that Fortran uses.

Consequences:

- One entry path shared by C++, Fortran, Python, and YAML. The C ABI stops being
  an extra parallel surface to maintain and becomes *the* mechanism.
- ABI-stable across compilers and stdlib versions; graph-structure changes never
  break existing binaries.
- Costs one serialize/deserialize hop at setup only. Nothing hot crosses it:
  per-step field access is by integer handle resolved once to a pointer.

```c
int soma_register_field(soma_ctx ctx, const char* comp, const char* name,
                        void* data, soma_layout layout, soma_memspace space,
                        int* handle);
```

---

## 4. Data model

### 4.1 Meshes and regions

The coupler never owns topology. `MeshId` is opaque; `MeshView` is the abstract
interface the core sees; `MoabMeshBackend` is the first implementation. The core
knows only entity counts, ownership (owned vs. ghost), a global-ID map, and
entity sets.

**The connectable unit is a `Region` = (mesh, entity set, dimension), not a
mesh.** Coupling happens on interfaces: a boundary sideset for CHT, an overlap
for domain decomposition, a volume for volumetric coupling. Channels connect
regions. This makes surface and volume coupling one code path and maps directly
onto MOAB entity sets and the iMOAB app-mesh model.

A component's ranks and the coupler's ranks may be **disjoint**, requiring n→m
redistribution. That is the `Transport` operator (§5.1), matching what
`iMOAB_ApplyScalarProjectionWeights` already does around its matvec.

### 4.2 Fields

```cpp
struct FieldSpec {
  std::string    name;
  Region         region;
  Discretization disc;             // FV | cGLL(p) | dGLL(p) | PointCloud
  int            components = 1;
  Units          units;            // dimension exponent vector
  Conservation   conservation;     // Extensive | Intensive | None
  Intent         intent;           // Provides | Requires | Bidirectional
};
```

Three members earn their place:

- **`disc`** selects the remap operator. FV→FV, SE→FV, SE→SE are different
  weight computations in TempestRemap. The *pair* of discretizations plus the
  conservation class determines the method; users never name a remap method by
  hand, though they may override (and the override is recorded).
- **`units`** is a vector of seven small integer exponents, `constexpr`-comparable
  — not a string. Connecting W/m² to K is a lowering-time error rather than a
  plausible wrong answer. String units cannot compose; exponent vectors can.
- **`conservation`** drives verification: extensive fields get global-sum checks
  across every remap, intensive fields get min/max bound checks. This makes
  conservation checking a framework property rather than a user discipline.

**Storage ownership: components own their data.** A component registers a
pointer plus layout (stride, component stride, memory space) once at init; the
coupler wraps it in a non-owning `FieldRef`. The coupler allocates only its own
intermediates: accumulators, waveform buffers, remap targets, previous iterates.

Rationale: nek and diablo already own their arrays and will not surrender them;
a mandatory copy per exchange is pure loss. It also keeps the GPU path honest —
a registered device pointer stays on device and `Transport` decides whether host
staging is required.

### 4.3 Time levels

```cpp
struct TimeLevel {          // value at t_ref + offset·dt, from iterate k
  Anchor   anchor;          // WindowStart | WindowEnd | Absolute
  Rational offset;          // exact: 0, 1, 1/2, -1 …
  int      iterate = 0;     // 0 = current, -1 = previous iterate
};
```

Every DAG arc carries one. `Rational` (exact integer numerator/denominator) is
mandatory, not stylistic: 1/3 and 1/7 subcycling must produce a slab
decomposition with no drift over 10⁶ steps, and floating-point time accumulation
would destroy that silently.

`lag(1)` *is* an offset of −1. OASIS's LAG keyword becomes a typed value rather
than a special case, and a lag exceeding the coupling period — which OASIS
documents as a deadlock hazard the user must avoid — becomes a validation error.

**Time-level typing is strict by default.** A mismatch is an error at lowering.
Per-channel escape hatch: `.allow_time_level_mismatch("reason")`, which records
the reason in the plan dump and the provenance log. Deliberate hand-tuned
staggering is legal; silent staggering is not.

### 4.4 Waveforms

A `Waveform` is a field sampled at several points across a window plus an
interpolation policy (constant / linear / cubic B-spline / DG(p)). It is the
base abstraction for coupled data in time; **explicit coupling is the degenerate
case** — a one-sample waveform with constant interpolation — not a separate path.

This is the object that unlocks multirate and higher-order coupling. Both the
quasi-Newton waveform iteration line (Rüth & Uekermann, *IJNME* 2021,
doi:10.1002/nme.6443) and the time-adaptive multirate extension (*IJNME* 2025,
doi:10.1002/nme.70063) depend on exactly this representation, and it is what
defeats the order degradation that flat end-of-window exchange suffers.

---

## 5. Numerics

### 5.1 Primitive operators

Everything is one of these. There is no `MiscOp`.

| Op | Signature | Meaning |
|---|---|---|
| `Sample` | field@t → waveform | record a sample into a window buffer |
| `Accumulate` | waveform → value | sum / mean / min / max / integral (OASIS `LOCTRANS`) |
| `Interpolate` | waveform × t → value | evaluate *inside* the window |
| `Extrapolate` | history × t → value | predict *beyond* known data |
| `Remap` | value@regionA → value@regionB | apply precomputed sparse weights |
| `Merge` | {values} → value | combine several sources into one target |
| `Relax` | (old, new) → value | ω-blend, Aitken |
| `Transport` | value@ranksA → value@ranksB | n→m redistribution; four backends |
| `Residual` | (target, source) → residual | the quantity iteration drives to zero |
| `Accelerate` | residual history → correction | IQN-ILS / IQN-IMVJ / secant / Steffensen |
| `Constrain` | value → value + report | positivity, bounds, conservation fixup (CAAS) |
| `Diagnose` | value → metrics | norms, global sums, NaN detection |

Two separations are load-bearing:

- **`Interpolate` vs `Extrapolate`.** Different error behavior and different
  legality. Interpolation inside a converged window is order-preserving;
  extrapolation beyond it is a predictor that must be corrected by iteration or
  accepted as explicit-scheme error. Keeping them distinct lets a verification
  pass answer *where a scheme loses temporal order*.
- **`Transport` vs `Remap`.** Interpolation in space and movement between rank
  sets are independent choices. Conflating them prevents placing remap flexibly
  (on source ranks, target ranks, or dedicated coupler ranks) — a real
  performance knob when weight matrices are dense.

### 5.2 Lowering

```cpp
g.channel(fluid, "T_wall", solid, "T_bc")
   .accumulate(Accum::Mean, Window::CouplingPeriod)
   .remap(Remap::Conservative | Remap::Monotone)
   .relax(0.7);
```

lowers to

```
Sample(T_wall@fluid, each fluid substep)     → w
Accumulate(w, Mean)                          → v0 @[WindowEnd, k]
Transport(v0, fluid_ranks → coupler_ranks)   → v1
Remap(v1, W_fluid→solid)                     → v2
Constrain(v2, conservation = Extensive)      → v3
Diagnose(v3)                                 → metrics
Relax(T_bc@[k-1], v3, omega = 0.7)           → v4
Transport(v4, coupler_ranks → solid_ranks)   → T_bc @[WindowEnd, k]
```

Three operators appear that the user did not write: the two `Transport`s (from
rank layouts), `Constrain` (from `conservation = Extensive`), and `Diagnose`
(always, cheaply). Ordering is explicit and inspectable.

Lowering rules live in a **registry keyed on (semantic, context)**, not a switch
statement. A new scheme registers a rule; the core is untouched.

### 5.3 Iteration regions

```cpp
g.iterate({fsi_channels})
   .until(Converged::Relative(1e-6, "T_wall") & Converged::Absolute(1e-8, "flux"))
   .max_iterations(30)
   .accelerate(Accel::IQN_ILS{.reuse = 8, .filter = Filter::QR2, .limit = 1e-2})
   .initial_relaxation(0.1);
```

An `IterationRegion` is a subgraph annotation carrying a convergence measure, an
accelerator, and a **computed** checkpoint set. Dataflow analysis over the SSA
form determines exactly which buffers are mutable state crossing the region
boundary. preCICE pushes this onto the user via `requiresWritingCheckpoint()`;
SOMA derives the coupler-side set, tells the component precisely which of its
state must be checkpointed, and verifies at runtime that it was.

Regions nest. **Schwarz waveform relaxation is an iteration region whose
exchanged quantity is a `Waveform` rather than a value** — no new machinery.
DNWR/NNWR (Meisrimel, Monge & Birken, *ZAMM* 2023, doi:10.1002/zamm.202100328)
and QN waveform iteration fall out of the same construct that gives plain Picard.

Accelerators operate on residual history and know no physics; one implementation
serves Picard, Aitken, secant, Steffensen, IQN-ILS, and IQN-IMVJ. Column
filtering (QR1/QR2/QR3) and per-field scaling are required for parallel
multi-field quasi-Newton and are therefore part of the accelerator interface,
not an add-on.

### 5.4 Remap

`RemapPlan` = (source region, target region, discretization pair, method) →
sparse weights. Built once at setup via MOAB + TempestRemap (advancing-front
supermesh; Mahadevan, Grindeanu, Jacob & Sarich, *Geosci. Model Dev.* **13**
(2020) 2355), cached and hash-keyed so identical channels share one plan. Per
step it is only a matvec.

Method defaults derive from (source disc, target disc, conservation). Explicit
override is permitted and recorded in provenance.

### 5.5 JFNK / monolithic coupling — deferred

Monolithic residual assembly requires components to expose *residual evaluation*
rather than *time advance*. That is a genuinely different component contract, and
it is PETSc-dependent.

The design accommodates it: a `Residual`-typed iteration region driven by a PETSc
`SNES`, where each component contributes to a global residual. It is **not built
in v1**. This is a scoping choice — shipping iterated partitioned coupling that
works is preferred to a half-built monolithic path — not an oversight.

---

## 6. Integration modes

All four are `Transport` implementations selected by the lowering pass.

1. **In-process C++** — components implement the C++ interface, one executable,
   communicator split per component. `Transport` is a copy or an MPI exchange.
2. **Fortran/C via the stable C ABI** — the narrow waist (§3.2) is the mechanism.
   An `ISO_C_BINDING` module shim wraps it. Ships and is unit-tested in v1; no
   coupled Fortran demo in v1.
3. **Separate executables (MPMD)** — port connect/accept over a coupling
   communicator. Design accommodates; **not implemented in v1**.
4. **Non-intrusive interception** — LowFive HDF5-VOL plugin (Peterka et al.,
   IPDPS'23) under Wilkins (Yildiz et al., *Front. HPC* 2024). Components are
   unmodified; coupling happens by intercepting HDF5 writes. Wilkins's
   `inport`/`outport` model maps directly onto graph channels.

In-process Python / ML surrogates are peer components through the pybind11
boundary, subject to the same `Constrain` and `Diagnose` contracts as physics.

---

## 7. Space-time remap

### 7.1 Motivation

With space and time handled separately, a transfer is conservative in space and
separately accurate in time, but the **space-time flux integral is not exact**
when both the meshes and the time steps are non-matching.

The fix is the same mechanism that worked in space. Supermesh projection is
exact because intersection removes quadrature error when integrating
discontinuous functions across a boundary (Farrell et al., *CMAME* 2009). Extend
the intersection to the time axis and the time-integrated transfer weights
become exact. This is the DGiT framework of Connors, Owen, Kuberry & Bochev,
*CMAME* **426** (2024) 116975 — DOE work on air–sea and solid–solid coupling.

The user-facing consequence: **each component names its own coupling interval and
the transfer between them still conserves.**

### 7.2 Two tiers

**Tier 1 — tensor-product space-time (v1).** Spatial supermesh weights `W` from
MBTR ⊗ temporal projection `T` onto the target's time partition. The spatial
mesh is static over the window, so `M = T ⊗ W` factorizes and the operator is the
existing matvec applied to time-projected coefficients. Reuses the entire MBTR
path. Delivers exact space-time flux conservation for static meshes with
arbitrary independent time steps — covering CHT, air–sea, and solid–solid.

**Tier 2 — true 4D intersection (deferred).** Pentatope/4D-simplex intersection
of space-time slabs, required when the spatial mesh *moves* within the window
(FSI with deforming interfaces, ALE). The operator does not factorize. The
interface admits it; v1 does not build it.

### 7.3 Representation

No new concepts — a generalization of two:

```cpp
struct RemapPlan {
  Region        src, tgt;
  DiscPair      disc;
  SpatialMethod spatial;   // conservative | monotone | high-order
  TemporalBasis temporal;  // None | Constant | Linear | DG(p) | Spline(p)
  //  temporal == None  ⇒  pure spatial remap
};
```

`SpacetimeRemap` is the general operator; `Remap` is `temporal = None`;
`Accumulate` + `Remap` is `temporal = Constant`. `Waveform` (§4.4) is already
the carrier — it holds samples plus an interpolation policy, which is exactly the
polynomial-in-time representation DGiT requires. What is new is the temporal
projection operator and conservation accounting across it.

```cpp
g.channel(ocn, "sst", atm, "T_surf")
   .spacetime(Temporal::DG(1), Window::from(ocn))
   .conservative();
```

### 7.4 Conservation check

For extensive fields the invariant becomes the space-*time* integral over the
window, ∬ q dA dt, matched between sides — strictly stronger than a per-exchange
spatial sum.

---

## 8. Scheduling with independent intervals

### 8.1 Independent by default

**Per-component intervals are the default; lockstep is the degenerate case.**
No channel implies a global synchronization point.

### 8.2 Consequences

- **No global time step.** The runtime advances a partially ordered set of
  component clocks; the scheduler emits transfers on slab intersections.
- **Exact rational clocks are mandatory** (§4.3).
- **Deadlock becomes a static property.** OASIS handles cyclic waits reactively
  via `SEQ` (detect and abort before hanging). SOMA has the whole graph before it
  runs, so the scheduler **proves acyclicity of the lowered, time-resolved DAG at
  setup**. This is a hard init-time gate: a scheduling error at initialization
  rather than a hang at hour six.
- **Lockstep stays a first-class fast path.** When all intervals are equal and no
  channel is space-time, the scheduler recognizes it and emits the simple plan.
  The common case does not pay slab-machinery overhead.

### 8.3 Slab decomposition

Per channel, at setup: compute the least common window, intersect the two time
partitions, generate temporal projection weights, pair them with the cached
spatial weights. Lag, subcycling, and multirate become special cases of one
mechanism rather than three.

---

## 9. Build

CMake ≥ 3.20, C++20, following KARMA conventions (single `CMAKE_PREFIX_PATH`
PETSc prefix; MOAB via pkg-config).

Options: `SOMA_ENABLE_{MPI, MOAB, PETSC, KOKKOS, LOWFIVE, PYTHON, FORTRAN, TESTS}`.

**The core builds with none of them.** A serial in-memory mesh backend and a copy
transport make the whole authoring → lowering → scheduling path testable with
nothing but a compiler. This is the payoff of the abstract-core decision and what
keeps CI fast.

### 9.1 Layout

```
modules/coupler/
  include/soma/{graph,ir,schedule,runtime,mesh,field,remap,transport,numerics,diag}/
  src/…                       # mirrors include/
  python/                     # pybind11 bindings + YAML front-end
  test/{unit,integration,verification}/
  examples/{mms_pair,lowfive_demo}/
  docs/specs/
```

---

## 10. Verification

Four levels, each a CTest target.

1. **Unit** (GoogleTest) — graph validation, lowering rules, rational clock
   arithmetic, slab decomposition, accelerator algebra against hand-computed
   values, unit-dimension checks.
2. **Golden IR** — canonical graphs lower to canonical DAGs, compared against
   checked-in text dumps. A lowering refactor cannot silently change the
   numerics; the diff is the review.
3. **Analytic operators** — remap of a constant field is exact; conservation of a
   random extensive field is machine precision; temporal projection of a degree-p
   polynomial in a DG(p) basis is exact.
4. **Order of accuracy (MMS)** — the coupled scheme runs at refined Δt and Δx and
   the observed order is fit and asserted. This is the test that catches order
   degradation and distinguishes a coupler that *claims* second order from one
   that *has* it.

Levels 1–3 run serially in seconds. Level 4 runs on 1, 2, and 4 ranks; results
must be rank-count independent to tolerance.

**Provenance.** The plan dump plus library versions, options, and every recorded
`allow_time_level_mismatch` reason form the run's reproducibility record.

---

## 11. v1 deliverables

**Demo A — MMS pair.** Two in-process C++ components (heterogeneous coupled heat
equations, the DNWR benchmark) on non-matching meshes with an analytic solution.
Exercises conservative remap, independent intervals with space-time transfer, the
explicit → Picard → Aitken → IQN-ILS ladder, checkpoint/rollback, MPI on disjoint
rank sets, and measured spatial and temporal order.

**Demo B — LowFive/Wilkins.** Two unmodified HDF5-writing executables coupled by
interception. Exercises the `Transport` abstraction's extreme case, the YAML
front-end, and the claim that intrusive and non-intrusive coupling share one
runtime.

---

## 12. Phasing

| # | Phase | Delivers |
|---|---|---|
| 0 | Skeleton, CMake, CI, vocabulary types | builds and tests with zero dependencies |
| 1 | `CouplingGraph`, validation, YAML round-trip | authoring layer, fully tested |
| 2 | `OperatorDAG`, lowering registry, golden-IR tests | front half of the compiler |
| 3 | Rational clocks, slab decomposition, scheduler, `ExecutionPlan` | acyclicity proof; lockstep fast path |
| 4 | Runtime, in-process transport, serial mesh backend | **first coupled run** (explicit, matching meshes) |
| 5 | MOAB backend, MBTR remap, MPI transport | non-matching meshes, disjoint ranks, conservation |
| 6 | Waveforms, space-time projection, independent intervals | multirate; temporal order tests |
| 7 | Iteration regions, convergence, accelerators, checkpointing | **Demo A complete** |
| 8 | C ABI, Fortran shim, pybind11, LowFive transport | **Demo B complete** |

Every phase ends with green CTest. Phases 4 and 7 are the honest milestones:
first run, and first converged iterated run.

---

## 13. References

- Farrell, Piggott, Pain, Gorman & Wilson. Conservative interpolation between
  unstructured meshes via supermesh construction. *CMAME* **198** (2009) 2632.
- Mahadevan, Grindeanu, Jacob & Sarich. Improving climate model coupling through
  a complete mesh representation: E3SM (v1) and MOAB (v5.x). *GMD* **13** (2020)
  2355.
- Connors, Owen, Kuberry & Bochev. A conservative discontinuous-Galerkin-in-time
  (DGiT) multirate time integration framework for interface-coupled problems.
  *CMAME* **426** (2024) 116975.
- Rüth, Uekermann, Mehl, Birken, Monge & Bungartz. Quasi-Newton waveform
  iteration for partitioned surface-coupled multiphysics applications. *IJNME*
  **122** (2021) 5236.
- Kotarsky et al. A time-adaptive multirate quasi-Newton waveform iteration for
  coupled problems. *IJNME* **126** (2025) e70063.
- Meisrimel, Monge & Birken. A time-adaptive multirate Dirichlet–Neumann waveform
  relaxation method for heterogeneous coupled heat equations. *ZAMM* **103**
  (2023) e202100328.
- Peterka, Morozov, Nigmetov, Yildiz, Nicolae & Davis. LowFive: in situ data
  transport for high-performance workflows. *IPDPS* (2023).
- Yildiz, Morozov, Nigmetov, Nicolae & Peterka. Wilkins: HPC in situ workflows
  made easy. *Front. High Perform. Comput.* **2** (2024).
- Valcke, Craig & Coquart. OASIS3-MCT user guide. CERFACS TR.
- preCICE documentation: coupling schemes, acceleration, implicit coupling.
- ESMF/NUOPC reference manual: driver, connector, run sequence, timekeeping.
- MOOSE documentation: MultiApp system, fixed-point iteration algorithms.
