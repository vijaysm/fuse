# Extensible coupling design review

## Objective and supported operating regime

The intended library connects existing simulation codes, new numerical models,
and emulators through explicit scientific contracts, with transactional execution
and distributed ownership. The initial integration target is **in-process
components within one MPI job**. Independently launched
executables and general MPMD transport are outside this initial target.

The proposed separation of semantic descriptions, storage bindings, component
calls, graph compilation, and execution is appropriate. A common C component
contract for static and dynamic discovery avoids separate language runtimes with
different scientific semantics. Compatibility must describe units, spatial and
temporal meaning, conservation, ownership, and capabilities; matching array shapes
alone is insufficient.

The design is suitable for incremental implementation. It does **not yet justify
claims of general multiphysics support or large-machine scalability**. Those
claims require integrated numerical examples and measured distributed costs,
respectively. A stable C ABI should be declared only after native, Fortran, and
Python implementations exercise the same ownership and error contracts.

## Evidence from the current implementation

The active source list includes the field abstraction, compiled FEM connection,
operator-split scheduler, fixed-point iterator, and safe-point checkpoint code.
These implementations, rather than historical documentation, define the baseline.

| Area | Verified source behavior | Consequence for extension |
| --- | --- | --- |
| Component descriptions | `karma_component.C` requires two ports, scalar FE coefficients, one input and one output | Introduce a general semantic model through explicit adapters; do not broaden the old validator silently |
| Graph compiler | `karma_operator_split_graph.C` requires two reciprocal connections and equal component/instance IDs | Separate implementation contract identity from instance identity; permit multiple instances of one contract |
| Graph signature | Canonical container sorting followed by delimiter-based string concatenation | Preserve existing signatures for existing checkpoints; new serialization needs length prefixes and a distinct version |
| Field identity | `DofKey` has 32 words; `FieldEntryKey` wraps a tag global ID or DoF key | Migrate persistence through explicit conversions; retain dense indices for execution |
| Map persistence | `karma_transfer_map_io.C` writes native-endian integer words and binary64 weights | Do not describe the existing format as endian-independent; introduce a versioned encoding or explicit rejection |
| Restart | Scheduler metadata is little-endian, records a graph signature and two component payload digests, and validates committed times | Preserve component payload ownership and safe-point validation; publication is not atomic across the three files |
| Distributed lookup | `karma_distributed_point_locator.C` gathers rank bounds; architecture documentation identifies further distributed routing work | Establish cost and memory budgets before claiming scalability |

The compatibility oracle exercises graph signatures, schedule order, stable-key
ordering, transfer-map records, fixed-point outcomes, and safe-point metadata.
It complements the existing partitioned mesh, transfer, scheduler rollback, and
restart trajectory tests. Its two-rank case is a replicated-mesh compatibility
check, not a large-machine or independent-partition benchmark.

A compiled probe confirms a signature collision: component metadata
`mesh_id="a:layout=b", layout_id="c"` and
`mesh_id="a", layout_id="b:layout=c"` produce equal graph signatures while
`diff()` reports different metadata. Existing signature preservation must be
limited to the supported legacy representation; new contracts need unambiguous
encoding. Do not freeze this ambiguity into a new schema or rely on the old
signature alone as authoritative mesh identity.

## Issues to resolve before freezing interfaces

### Transaction publication and failure scope

“Validate, then commit” requires a precise failure model. A local host-buffer swap
can be made nonthrowing after allocation, but PETSc/MOAB operations and component
callbacks can fail. Separate preparation, collective agreement, publication, and
cleanup. Keep old state alive until publication succeeds everywhere. Define a
terminal failed state when rollback cannot restore the old state, and preserve
the original error together with the rollback error.

Do not describe rollback of arbitrary external code as infallible. Components
must declare which state is checkpointed, whether callbacks perform external
side effects, and whether replay restores random generators and hidden solver
history. A failed MPI process is a different failure class from a caught
rank-local callback exception; ordinary reconciliation assumes every rank remains
alive and able to participate.

For field transactions, specify aliasing between inputs and outputs, borrowed
view invalidation, duplicate sparse updates, summation order, and how a complete
replacement proves coverage. Define whether deterministic accumulation means
repeatability on a fixed partition or identical results across rank counts.

### Communicators and execution resources

In-process integration still needs framework-owned communicator lifetimes,
participant sets, communicator duplication/release rules, and a policy for
legacy components that use global state. The compiler must know which operations
are collective on which communicator. Nonparticipants need a defined parent-job
control path, including failures originating in a child communicator.

A parallel schedule group cannot imply concurrent MPI callbacks without a
verified MPI thread-support level and component reentrancy declarations. Start
with serialized, deterministic execution of independent nodes if those
capabilities are absent. Keep error reconciliation outside Python GIL-held or
application-lock-held regions where possible. Cancellation polling cannot safely
interrupt an arbitrary blocking collective; expose it as cooperative cancellation.

### ABI negotiation and language memory ownership

Define table-size units, alignment, status widths, overflow-safe extent/stride
arithmetic, optional null pointers, empty arrays, and error-buffer termination.
Every callback needs the same exception/status boundary, including factory,
destroy, checkpoint release, and diagnostics. Component state schema identity
must include its owning contract; unrelated plugins can both have state version
`1.0.0` without their payloads being interchangeable.

Keep communicator tokens opaque in the C ABI. A Fortran bridge must be verified
against the selected MPI and compiler stack. Avoid exposing compiler-specific
Fortran descriptors as a portable binary promise between arbitrary compilers.

Call-lifetime NumPy views alone cannot prevent a Python plugin from retaining an
array or a derived view. NumPy requires external storage to remain alive for the
array lifetime and supports attaching an owning base object. Use owned staging
buffers or an explicit lease tied to storage lifetime, and test retained slices,
reference cycles, module shutdown, and destruction after runtime teardown. This
is an ownership inference from the [NumPy C API contract](https://numpy.org/doc/stable/reference/c-api/array.html#c.PyArray_SetBaseObject).

Optional metadata preservation belongs to the format parser and serializer.
A version negotiation helper should not consume, discard, or reinterpret opaque
bytes. Similarly, allowing a newer optional minor tail does not validate its
size or make unknown required capabilities available.

### Scientific compatibility and temporal semantics

Specify the unit vocabulary, dimension exponents, scale representation, coordinate
frames, tensor component ordering, field location, continuity, and reference
configuration. Automatic unit conversion needs an exact eligibility rule;
floating-point evaluation of a scale factor is not generally numerically exact.
Treat affine units and changes of conservation meaning as explicit operators.

Preserve operator order. Temporal averaging, spatial remapping, nonlinear
transforms, and constraints need not commute. A conservative transfer of cell
averages and a transfer of extensive cell integrals use different normalizations.
Require measure and coverage semantics at the connection boundary.

Define physical stage times independently from splitting composition fractions.
Clarify the supported IMEX tableau and stage interface before claiming general
IMEX support. Nonintegral rendezvous rejection should use explicit rational grids
or an identified numerical tolerance, not accidental floating-point equality.
Waveform relaxation must declare its interpolation basis, time-window residual,
and history reset behavior. Event reproducibility needs more than tie-breaking:
define crossing direction, root-location tolerance, simultaneous events, and
whether events can change topology.

Anderson and IQN-ILS need independent reference updates, rank-revealing history
filtering, and history reset after changes in layout, parameters, or time-step
regime. One successful nonlinear example establishes neither universal convergence
nor superiority over Aitken.

### Conservative remapping and adaptivity

Specify whether partial coverage produces rejection, an explicit fill, or a
reported partially covered result. Do not renormalize incomplete overlap while
claiming conservation over the original domain. Check local balance as well as
global balance, since cancellation can hide row errors. Scale-aware geometric
tolerances must distinguish coordinate scale, cell measure, and accumulated
integral error. Unsupported nonplanar or nonconvex geometry must fail explicitly
unless a decomposition algorithm is provided and verified.

Conforming refinement requires closure rules and may refine unmarked neighbors.
Stable identity assignment must avoid reuse and collisions across refine/coarsen
cycles, and preserve global IDs when externally repartitioned. Geometry,
topology, partition, and field-layout epochs need independent invalidation rules.
An epoch integer alone is not proof that two meshes have identical content.

For prolongation and restriction, state separately which spaces preserve
constants, linear fields, and integrals. A map that is appropriate for a nodal
field is not automatically appropriate for a conserved density or integral.

### Emulators and multiscale closures

A Python adapter is a language capability, not an emulator validation contract.
Record model artifact digests, training/normalization metadata references, input
regime, batching rules, seed ownership, state evolution, and numerical tolerance.
Require explicit policy for out-of-regime predictions and unavailable models;
never silently substitute a physics solver or another model.

Fixed-capacity instance sets provide a tractable starting point for map/reduce
and lift/evolve/restrict closures. They do not provide dynamic population growth
or load balancing by themselves. Test inactive participants, empty reductions,
overlap weights, capacity limits, and rollback of hidden microstate. Specify the
reduction algebra and its reproducibility guarantees.

### Canonical configuration and provenance

Choose exact canonicalization rules before hashing configurations. “RFC 8785
principles” is insufficient for cross-language byte equivalence: JCS sorts keys
by UTF-16 code units, preserves Unicode strings without normalization, and uses
specific binary64 number serialization. Large exact integers may need string
encoding. Pin an exact JSON dependency version rather than an open-ended minimum,
and add conformance vectors before calling the format JCS-compatible.
See [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785.html).

Keep semantic graph identity separate from execution provenance. A graph hash
should not accidentally change with container order, while implementation
versions, solver settings, model digests, and configuration changes must be
visible in provenance and restart validation. Dry-run compilation should not
initialize numerical state or execute plugin side effects.

## Dependency corrections

Several dependencies must be explicit before concurrent implementation is safe:

| Capability | Required predecessor or boundary |
| --- | --- |
| Graph-based language substitution | Executable general graph compiler, not just the registry and language host |
| General DAG execution | An execution loop and merge evaluator must exist; instruction records alone do not execute a graph |
| General runtime field transfer | Ordered pipeline execution must precede integrated graph tests, or those tests must declare a limited identity-transfer fixture |
| Fortran/Python graph examples | Shared component compliance plus graph compilation and the supported schedule runtime |
| Adaptive graph reload | Mesh epochs, persisted stable identities, compiled connection invalidation, and history/acceleration reset |
| Physics components in general graphs | Mixed-field layout and assembly plus the registry adapter and graph executor |
| Integrated restart | Integrated component state/provenance plus the graph-aware prepare/apply protocol |
| Dynamic loading | Finalized ownership contract and static registry compliance; loading must change discovery only |

The semantic header also cannot introduce another `KARMA::FieldSpec`, `PortSpec`,
or `ComponentSpec` alongside the existing definitions. Choose a versioned
namespace or distinct new types, then use explicit adapters. Avoid implicit
compatibility aliases while meanings differ.

## Scalability acceptance criteria

Move performance instrumentation into development rather than waiting until
packaging. Record setup time separately from warm execution, bytes copied per
exchange, rank-local peak memory, history storage, message counts, collective
counts, and load imbalance. Establish bounded memory for transfer setup and map
reload. A rank-bounds gather can be acceptable in one operating regime but still
requires measurement as process count grows.

Require representative strong/weak scaling runs on actual target machines before
using “large-scale verified.” Small rank counts validate distributed correctness
and asymmetric errors, not throughput or scaling. The local double-real
PETSc/MOAB stack establishes one configuration only; it supplies no evidence for
GPU execution, other scalar ABIs, other Fortran compilers, or MPI implementations.

The inspected local Release compile flags contain `-O3` followed by `-O0` and
`-std=gnu++20` imported from `MOAB_CFLAGS_OTHER`. Thus the CMake build-type label
alone does not establish optimized execution or strict C++20 mode. Audit
dependency-exported flags and actual compile commands before benchmarking. This
review does not change dependency flags, so existing numerical behavior remains
the validation baseline.

## Reviewable implementation boundary

The initial code change adds numeric version-domain negotiation and a behavior
oracle for the existing coupler. It does not migrate existing persistence,
declare a stable C ABI, or implement plugin discovery. The version API has an
explicit conservative policy: reject different domains/majors, reject unsupported
required bits, accept older minor contracts, require an explicit optional-tail
policy for newer minors, and require the reader patch to meet the contract patch
within one minor. Owning formats remain responsible for payload validation.

Success for this boundary means a fresh build, registered serial/MPI oracle tests,
adversarial version tests, relevant existing integration tests, sanitizer checks,
and warning-as-error documentation. Subsequent implementation can migrate stable
identity and semantic descriptions against this oracle before introducing
transactions and language-neutral component calls.

## Correspondence with the adjacent coupler

The adjacent coupler already implements portions of the general-graph,
language-adapter, configuration, and conservative-remap surface. The following
comparisons identify which KARMA contracts can be shared and which require an
explicit adapter.

**Version domains.** This library carries seven independent numeric domains. The
adjacent implementation carries a single integer ABI version plus a tagged
serialized graph descriptor. A port would have to decide, per domain, whether the
domain becomes a field of that descriptor, a separate negotiated value, or is
dropped. The domains most likely to survive intact are the C ABI and the graph
schema; transfer map and component state are owned by formats this library
already persists and the adjacent implementation does not.

**Field-entry identity.** `FieldEntryKey` fuses a MOAB global identity with a
32-word algebraic `DofKey`, and the canonical encoding is a fixed 274-byte
record. Any adopted identity model must be able to carry both meanings without
narrowing, and must preserve the sort order the compatibility oracle pins,
because persisted transfer maps and safe-point checkpoints are addressed by it.
A narrower identity is not a substitute, and a wider one needs a migration for
existing files.

**Compatibility checking.** The checks defined here are unit exactness, shape and
component matching, location and space, temporal meaning, and required-capability
closure, each returning the first failing stage. A port must establish which of
these the adopted validator already performs, which are additional, and where an
additional check would run, before any lane is declared reusable.

These comparisons do not select an implementation. They preserve the choice
of a native or adapted implementation without weakening the current contracts.

## Validation: what SOMA already checks, and what is KARMA-only

Read from `coupe/modules/coupler/include/soma/graph/validate.hpp` at SOMA
`HEAD 9902f82`, 2026-09-09. SOMA's `validate()` accumulates diagnostics over a
whole graph and returns them together, which is the same discipline
`check_connection()` follows per connection.

**Checks SOMA already performs.** Name uniqueness for components and for fields
within a component; a strictly positive coupling interval; reference integrity
for field-to-region, region-to-mesh, and window-to-component ids, including ids
that are in range of the sentinel test but out of range of the container;
`components >= 1`, with a recorded history of a negative count wrapping to
1.8e19 in a later size computation; **exact unit equality**; shape equality by
component count; **conservation-class equality between source and target**;
intent direction, so a source may not be `Requires` and a target may not be
`Provides`; self-coupling detection; a lag that spans more than the coupling
window, cited as a restartability and deadlock hazard; a nonempty
`time_level_mismatch_reason`; **multiple channels writing one field**, which is
an error rather than a warning because lowering binds one arc per target and
silently drops every writer but the last - the recorded consequence was that
`prove_acyclic` certified 232 graphs containing real algebraic loops in a
19,181-graph sweep; and a `Provides` field that no channel sources.

**Checks that are KARMA-only.** Graded verdicts at all: SOMA's unit rule is
exact equality, so it has no notion of a connection that *could* compile with a
declared operator, and therefore no lossless-versus-lossy distinction and no
ambiguity verdict. Unit *scaling* and affine conversion, which SOMA cannot
express because its `Units` carries dimensions but no scale or offset. The
automatic-insertion rule and its site policy. Location, continuity, and basis
compatibility with a required spatial transfer - SOMA's `disc`/`order` pair is
declarative and is consumed when selecting a remap operator, not validated as a
compatibility question. Temporal *interpretation* checking: SOMA validates
`TimeLevel` placement and lag, which is when a value lives, not whether an
instantaneous value may be read as an interval average. Per-port required
capability closure. And the ordered operator chain as an output artifact with
its own identity.

**Checks SOMA has that KARMA does not, and should acquire.** Multi-writer
detection is the important one, and SOMA's comment explains why it must be an
error: it is the failure that let an acyclicity proof certify graphs containing
loops. KARMA's two-component graph cannot express fan-in today, so the check has
nowhere to live yet, but any general graph must carry it from the first commit
rather than discover it in a fuzz sweep. Self-coupling detection and the
lag-versus-window hazard are in the same class.

**Consequence for integration.** Adopting SOMA's validator wholesale would lose the
graded verdicts, the transform inference, and the capability closure; adopting
KARMA's would lose the graph-level checks SOMA learned the hard way. The two are
complementary rather than competing, and the honest ruling is likely to be per
check rather than per file.
