# Architecture

## Active Build Boundary

`src/CMakeLists.txt` explicitly enumerates every production `.C` file in `KARMA_SOURCES`. This is deliberate: the repository contains legacy and partially migrated code, so source discovery by glob would silently change the library surface. New implementations must be added explicitly.

All headers under `include/` are currently installed, including historical ones. Consumers and documentation must not infer implementation support from header installation alone.

## Core Layers

### Mesh and Geometry

`KARMAMesh` owns MOAB and is authoritative for topology, coordinates, material/boundary sets, rank ownership, shared entities, and ghost layers. The raw MOAB interface is intentionally public.

`KARMAReferenceTopology` defines reference-cell connectivity. `KARMAGeometryMap` maps points and derivatives to physical cells. `KARMABasisRegistry` owns built-in and custom bases. `KARMAFEBase` composes these pieces and maintains address-stable evaluation buffers used by assembly kernels.

### DoFs and Algebra

`KARMADofMap` derives stable keys, chooses owners, assigns contiguous owner-major indices, builds local-to-global and ghost mappings, selects once-only assembly elements, and computes exact sparsity. `KARMASystem` owns the map, variables, PETSc state vectors, and matrix creation. `KARMAAlgebraicSystem` provides a mesh-free distributed layout for ODEs.

### Physics and I/O

Active physics comprises the generic mass operator, scalar cG diffusion-reaction, scalar dG advection-reaction, and mesh-free ODEs. Implementations stage local work, normalize errors collectively, and assemble PETSc objects.

`KARMAMoabIO` delegates mesh I/O to MOAB and owns exact checkpoint schema validation, coefficient layouts, bounded owner-only gathers, transactional publication, and stable-key restore.

## FEM Transfer Layer

The active coupling layer is separate from excluded legacy interface sources. `KARMAField` identifies location, components, scalar representation, and non-owning MOAB/PETSc storage. `KARMAPointLocator` separates candidate search from geometry inversion; the verified implementation is `KARMAMoabPointLocator` using MOAB's `AdaptiveKDTree`. This native path does not link or call the MOAB Coupler backend. `KARMADistributedPointLocator` reconciles stable source element identities across ranks. `KARMAFEMFieldEvaluator` refreshes PETSc ghosts and evaluates registered bases, while `KARMAFEMTransfer` provides nodal sampling, element-local L2/FV projection, and global cG projection.

Stable `GLOBAL_ID` and `DofKey` identities are the only persisted or cross-rank associations. MOAB handles are local runtime observations and are never used as serialized identity. Coverage is explicit, with uncovered points distinct from invalid geometry and backend failures.

### Transfer TODOs

- Replace all-rank distributed locator reconciliation with bounding-box candidate routing and bounded all-to-all exchanges.
- Reassociate remote source elements to local ghost handles or exchange source coefficient packets before evaluation; the batched owner-request API is now present, but partitioned fixture integration remains.
- Partitioned batch projection orchestration is implemented through one batched locator/evaluator phase over each caller-supplied owned-element batch; fixture coverage verifies independent source/target partitions at 2/4/5 ranks.
- PETSc coefficient and MOAB tag commits are staged/validated APIs; extend mutation coverage to partitioned target fields.
- Persist cross-operator rows as stable-key `B` factors, recompute `M_t` after target-signature validation, and verify cross-rank-count reload with current target-row ownership and source-column availability checks.
- Future optimization: stream/redistribute canonical map rows directly to current target owners rather than parsing the full map on every rank.
- Implement and benchmark ArborX/Kokkos when a compatible standalone stack is available.
- Add validated intersection quadrature for exact nonmatching FV/modal projection and conservative remapping.

## Native operator-split coupling foundation

The active coupling foundation is deliberately narrower than the eventual
plugin architecture. `compile_operator_split_graph()` canonicalizes the two
component instances, ports, reciprocal connections, and instruction stream;
`CompiledOperatorSplitGraph::signature()`, `report()`, and `diff()` expose that
identity and categorized changes. Component-declared `mesh_id`, `layout_id`,
`basis_id`, and solver-policy identifiers are metadata: their default is
`unknown`, and the foundation does not infer globally authoritative mesh or
layout identities for callers.

`KARMACouplingScheduler` owns persistent `CompiledFEMConnection` instances,
transaction-local staging, macro-step rollback, and either Lie/Strang advance
or `advance_fixed_point()`. The fixed-point path is collective and scheduler
integrated: it stages committed component fields as iterates, evaluates each
sweep through the persistent connections, measures residuals with each
component's consistent mass matrix using RMS normalization, and publishes only
after convergence. It supports constant relaxation and bounded Aitken
relaxation, deterministic rank-qualified failures, and rollback outcomes.
Field/mass preflight and initial iterate staging finish before checkpoint
acquisition; source-field lookup, exchange, mapped-state collection, converged
iterate staging, and commit are separate reconciled epochs. A recoverable
rank-local failure after checkpoint acquisition restores the macro-step state
and leaves the scheduler position unchanged.
Anderson acceleration and generalized graph iteration are deferred. The
current graph compiler remains a two-component/two-reciprocal-edge foundation;
arbitrary topology, fan-in, dynamic plugins, Python/Fortran adapters, MPMD
communicators, executable ArborX, and conservative remapping are not active.

The restart layer is a two-phase safe-point protocol. It validates graph
signature, declared metadata, field layout, canonical geometry/topology, and
stable `DofKey` sets before applying staged H5M values. The U checkpoint, V
checkpoint, and scheduler metadata are separate files; publication is not an
atomic multi-file durable transaction.

## Compatibility contracts

The coupling compatibility oracle (`test/coupling_compatibility_oracle_test.C`,
run serial and at 2 ranks) records existing graph signatures, Lie/Strang
instruction order, stable field/DoF key ordering, transfer-map schema metadata,
safe-point scheduler metadata, and fixed-point outcome semantics. A replacement
must preserve the behavior or reject an unsupported persisted representation
with an explicit migration diagnostic.

The recorded fixtures are migration invariants in exactly this sense:

| Invariant | Recorded form |
| --- | --- |
| Graph identity | canonical signature text, unchanged by component, connection, or port container reordering |
| Split semantics | the literal compiled instruction stream for Lie and Strang, including instruction kind, endpoints, and advance-interval code |
| Field identity order | `DofKey` width 32 and its sort order; `FieldEntryKey` sort order across kind, global id, component, and DoF |
| Persisted transfer maps | schema version, scalar width, component count, and source/target signatures, with rows addressed by stable key |
| Safe-point metadata | magic word, schema version, split scheme, committed time and step, and the graph signature length prefix |
| Fixed-point outcomes | status name, iteration count, and checkpoint/restore/publication counts for each of the six terminal scenarios |

The oracle records no PETSc ordinal, MOAB handle, dense index, or payload file
digest, because none of those is a semantic identity. Everything else is free to
change: C++ type and function names (`CompiledOperatorSplitGraph`,
`KARMACouplingScheduler`, `KARMAFixedPointIterator`,
`TransientReactionDiffusionNativeAdapter`, and the `karma_operator_split_graph.h`
record types), header layout, dense runtime index assignment, storage bindings,
and the internal spelling of diagnostics.

`karma_semantic_version.h` defines independent numeric version domains for the
semantic model, graph schema, C ABI, plugin configuration, transfer maps,
component state, and graph checkpoints. Numeric versions use canonical
`major.minor.patch` spelling with unsigned 32-bit words. Prerelease/build suffixes
are not supported. Negotiation returns reason codes and missing required bits;
it does not infer capability from version numbers or validate payload bytes.

Those seven domains are the complete set. Adding a domain is a minor-version
change of the semantic model domain, never a silent extension: `VersionDomain`
values are stable numeric identifiers, and `version_domain_name()` returns no
name for an unrecognized value so that an unknown domain is rejected as
`InvalidDomain` rather than defaulted.

A reader accepts older minor contracts within its major, including major zero.
Within the same minor, the contract patch is the minimum required reader patch.
Newer minors require explicit permission for skippable optional tails.
Negotiation rejects unknown required capability bits here, in
`check_version_compatibility()`, and reports exactly which bits were missing.
Preservation of unknown *optional* bytes is deliberately not implemented at this
layer, which sees no payload: it belongs to the owning format, and for the C ABI
it is the size-prefixed table tail rule. Existing map and checkpoint formats are
unchanged, and these evolving C++ types do not establish a public C ABI.

Mutation evidence for this layer is under `build-evidence/version-mutations`: an
inverted major comparison, an inverted minor comparison, and a build that
ignores unknown required bits each fail the focused test.

## Stable field-entry encoding

`karma_field_entry_encoding.h` gives the existing `FieldEntryKey` one canonical
byte spelling: a fixed 274-byte little-endian record carrying a payload length
prefix, an encoding version, the kind discriminant, the global id, the component
ordinal, and all 32 `DofKey` words, plus an FNV-1a hash over those bytes. It
introduces no new identity. `FieldEntryKey` already wraps both supported
meanings; what was missing was one endian-stable spelling that every new
contract can share.

Every new subsystem that needs to name a field entry across a process, a file,
or a language boundary uses this encoding: semantic-model hashing, C ABI field
tables, and graph checkpoint metadata. The existing transfer-map and safe-point
checkpoint formats keep their own schema versions and word layouts; migrating
them onto this encoding is a separate change and is not implied by this one.

Both kinds encode every field, so a MOAB key carries zeroed DoF words rather
than a shorter record. The encoded size is therefore a constant and a reader
never branches on kind to find the end of a record. Decoding is nonthrowing and
returns explicit reason codes for truncation, declared-length mismatch,
unsupported version, unknown kind, and trailing bytes.

The distributed contract is that a key encoded on one rank decodes to the same
key on every other rank, and that its hash agrees bit-for-bit across ranks, so a
field entry can be re-associated after an arbitrary change of ownership without
consulting a PETSc ordinal, a MOAB handle, or the rank that produced it.
`karma_field_entry_encoding_reassociation_{2,3,4,5}_mpi` proves this over a
canonical 37-key set whose ownership varies with rank count and which leaves one
rank owning nothing.

## Versioned semantic model

`karma_semantics.h` describes what a coupled quantity *means*, separately from
the container that holds it. The header is deliberately free of PETSc, MOAB, and
MPI: semantic identity must not be defined by storage. Adapting the current
two-component specification records into the model is the job of
`karma_semantics_adapter.h`, which is the only place the storage-aware headers
appear, and the direction is one-way - the existing records are the behavioral
oracle and are not modified.

A field's meaning is its identifier and version, its units as seven SI
dimension exponents plus an exact rational scale and offset, its frame, its
shape and component count, its discrete location and continuity, its temporal
interpretation, its role, its ownership, and the access and update the component
performs on it. Ports add direction and a requirement carrying an accepted
version interval and required capability bits. Transforms record whether they
were explicitly requested, because only exact zero-offset unit scaling may ever
be inserted automatically; every other transform changes the meaning of a value
rather than its spelling.

### Time is exact rational

Every time value in the model and in every contract built on it is a normalized
`Rational{int64 numerator, int64 denominator}`. A `double` cannot represent a
one-third subcycle and cannot support the exact equality that a multirate
rendezvous needs. Comparison forms its cross products in 128 bits, so it is
exact even where those products exceed 64 bits; addition, subtraction, and
multiplication report overflow rather than wrapping.

Two conversions exist for interoperating with the current `double`-valued
scheduler, and the distinction between them matters. Every finite `double` is
already a dyadic rational, so `rational_from_dyadic()` never rounds: it converts
`0.125` to `1/8` and the `double` nearest `0.1` to its true value
`3602879701896397/36028797018963968`. Exactness is therefore not the useful
property. That second value is unusable as an exact time because its denominator
is near 2^55 and adding two such values overflows immediately, which is why
`rational_from_dyadic_bounded()` refuses a denominator wider than a caller-given
bound, and why the specification adapter applies a 32-bit bound. The correct response
to a refused value is to declare the step size as a rational, not to widen the
bound.

### Canonical identity and validation

Serialization sorts ports and metadata by identifier, length-prefixes every
string and byte sequence, and writes every integer fixed-width little-endian.
Container order therefore never changes identity and no delimiter needs
escaping. Unknown *optional* metadata is preserved verbatim and participates in
identity; unknown *required* capability bits are a different question, rejected
by version negotiation rather than here.

Validation is purely local and total: it reports every diagnostic it finds
rather than the first, so a caller can reconcile one complete local verdict
across ranks in a single epoch instead of discovering failures one collective at
a time. Mutation evidence is under `build-evidence/semantics-mutations`: eleven
mutants of the model and five of the specification adapter are each killed, including
unsorted containers, unprefixed strings, 64-bit cross multiplication, wrapping
addition, and an ignored denominator bound.

## Collective outcome reconciliation

`karma_collective_outcome.h` is the one failure-reconciliation primitive for
coupling code. The hazard it removes: one rank discovers a local problem and
throws while its peers proceed into the next MPI or PETSc collective, which then
hangs or aborts naming the wrong cause. Every path that can fail locally records
a local outcome, reconciles it across the transaction communicator, and acts on
one agreed verdict.

Selection is deterministic - the lowest failing rank decides, and within one
rank the lowest reason code decides. Determinism matters more than picking the
most interesting failure, because a nondeterministic choice produces diagnostics
that change between runs of the same job. The rule is a pure function over
gathered outcomes (`select_outcome`), so it is testable without a communicator,
and the reduction itself is a single unsigned minimum over a key that packs rank
into the high 32 bits and reason into the low 32. That packing is why no custom
MPI operator is needed, and it is why the success key must sort above every
representable failure key rather than merely above the named ones.

The reduction carries a numeric code, never a string: each rank maps the code to
text locally, which avoids a second collective to broadcast a message from the
failing rank. A caller's own `rank` field is ignored and replaced by the
communicator's rank, so a rank cannot misreport its identity.

## Transactional field publication

`karma_field_view.h` gives storage-neutral read access and setup-time resolution
of stable keys to dense indices. Two rules shape it: a hot path must not look up
a stable key, and the map counts lookups so a test can prove a warmed path
performed none; and ownership is positional, with owned entries preceding ghosts,
so no per-entry predicate is needed to know what may be published.

`karma_field_transaction.h` stages updates in transaction-private memory and
publishes all or nothing. Three modes exist because three distinct things are
being expressed. `Replace` requires every value to be staged exactly once, since
a partial replacement would publish a mixture of this step's values and the
previous step's. `SparseReplace` overwrites staged slots and leaves the rest as
published. `Accumulate` adds to the published value and may leave slots unstaged.
In both replacement modes a second value for one slot is rejected as ambiguous
rather than resolved by arrival order.

Accumulation is canonicalized rather than applied in place. Floating-point
addition is not associative, so summing contributions as they arrive makes the
published value depend on arrival order, and therefore on rank counts and
scheduling. Contributions are collected and summed at commit in ascending
magnitude with the bit pattern breaking ties, which is a total order independent
of staging order; a test stages the same contributions forwards and backwards and
requires bitwise-identical results.

Discarding staged state and restoring a previous commit are deliberately
separate operations. `rollback()` touches only transaction-private memory and
therefore cannot fail; `restore_committed()` republishes the snapshot taken
before the last successful commit and can fail, because it is a publication.
Collapsing the two would make the infallible operation fallible.

`commit_collective()` reconciles local validation across the communicator before
any rank publishes, so a validation failure anywhere means no rank publishes. A
publication failure after a successful reconciliation is reported with the
failing rank; the distributed state is then inconsistent and the caller must
restore. That is the one case this API cannot make atomic, and it is documented
rather than hidden.

Mutation evidence is under `build-evidence/transaction-mutations`: five outcome
mutants and seven transaction mutants are killed, including a reason-primary
sort key, a sign-extended negative rank, a success key that sorts below
failures, publication before validation, accumulation in arrival order, and a
snapshot taken after publication instead of before.

## Component registry and runtime

`karma_registry.h` is a table of what is installed. Two rules give it its
shape.

**It never substitutes.** A request naming an implementation that is not
registered is an error, even when another implementation of the same contract
is sitting right there; a request matching two implementations with no stated
preference is also an error. Answering either with a plausible alternative
would change the physics while reporting success, and the caller would have no
way to notice. `SelectionPolicy::RefuseAmbiguous` is the default, and
`HighestVersion` and `HighestPriority` exist only so that a caller can say out
loud which policy it wants. `priority` is consulted by nothing else, because an
implicit priority is a hidden policy.

Filters are applied after identity, so a refusal names the filter that refused
rather than reporting the record as absent, and a capability refusal reports
*which* bits were missing. An unset version maximum is unbounded, not a refusal
of everything - the same trap the connection checker documents.

**It is re-entrancy safe.** Records live behind stable pointers
(`vector<unique_ptr<...>>`), `visit()` iterates a snapshot taken before a guard
is raised, and a registration attempted from inside a visit is refused with
`Reentrant` rather than allowed to corrupt the iteration. The shape being
avoided is a container iterated by reference while a callback may append to and
re-sort it: `soma::ir::LoweringRegistry::apply_first_match` iterates
`std::vector<Entry> rules_` by reference while calling each rule, and its
`register` both `push_back`s and `stable_sort`s, so a rule that registers
during the callback invalidates the loop mid-iteration. The guard is scoped,
not sticky - registration works again once the visit ends.

The inventory is sorted by implementation identifier rather than left in
registration order, because static initialization order can differ between
builds of the same program and an inventory that changes with it cannot be
compared across runs.

### One path, not a fast path

`ComponentInstance` (`karma_component_runtime.h`) is an RAII handle over a
`karma_component_table`. It calls the same C schema whether the
implementation is C++, a dynamically loaded object, or a language adapter.
There is deliberately **no** second, faster path for native components: a
native component taking a shortcut would not be exercising the contract the
others must satisfy, and the contract is the thing under test. The handle is
move-only, since two handles to one instance would destroy it twice.

`NativeComponent` is the C++ interface an in-tree component implements, and
`register_native_component()` wraps it in that shared table - the only place
native code is special. Factories are keyed by implementation identifier, which
is what the ABI `create` call carries; an identifier no factory claims is
refused rather than answered with whichever factory happens to be first. The
registry validates the record before the factory is stored, so a rejected
record leaves no orphan factory under an identifier nothing can select.

Two gates live in the runtime rather than in each component, because they are
properties of the request and not of the physics: a step whose length is not a
whole multiple of the clock requires the fractional-step capability, and
checkpointing requires the checkpoint capability.

### Why the ABI record operations are a header

`libkarma_abi` links `libkarma` privately, so `libkarma` cannot call back into
the ABI library - that would be a dependency cycle. But the host-side runtime
lives in `libkarma` and needs exactly the same field-table validation the
exported C entry point performs. `karma_abi_records.h` therefore holds those
operations as `inline` functions over plain records, and the exported
`karma_validate_field_table` and `karma_diagnostic_set` are thin forwards
to them. One implementation, two callers, no cycle, and no second copy of the
rules to drift.

### The compliance harness is production code

`check_component_compliance()` lives in the library, not in a test, because the
Fortran and Python adapters must pass the same function rather than a
re-derived approximation of it. It checks what every component must do whatever
it computes: that work before `initialize` is refused rather than undefined,
that the registry record does not overstate its component's capabilities or
version, that fields round-trip, that the component *copies* rather than
retaining the caller's buffer, that checkpoint and restore reproduce state
exactly and a released token is no longer restorable, that a deliberate error
leaves a diagnostic, and that two identical runs produce bitwise-identical
bytes - not approximately identical, because a coupled run that is only
approximately reproducible cannot be restarted.

A harness is only evidence if its checks are live, so the test suite includes
two deliberately non-conforming components: one whose registry record claims a
capability it does not provide, and one that retains the caller's buffer
instead of copying it. Each must be *caught*, with the failure named. Deleting
a check from the harness fails those tests.

### The reference components

`karma_reference_components.h` registers `karma.reference.decay.baseline` and
`.v2`, both satisfying `karma.contract.scalar-decay`. Two of them on purpose:
one implementation cannot demonstrate that selection refuses to substitute.
They are reference implementations rather than physics - an exponential decay
with an analytic step, so a step is one multiplication and two runs are bitwise
identical. They need neither PETSc nor MPI, which is what makes the compliance
harness runnable anywhere the library builds.

### What is not registered yet

The PETSc-backed transient reaction-diffusion adapter is **not** registered
through the runtime. It cannot be until the zero-copy PETSc field binding
exists, and that binding is blocked on the same finding the transaction layer
records: `commit_petsc_values()` writes by global index through `VecSetValues`
and assembly, routing values for entries the committing rank does not own,
which a dense local field layout cannot express. Registering it would also make
the compliance checks unrunnable wherever PETSc and MPI cannot initialize. The
existing fast path remains the executing route for that component and remains
the oracle.

## One reconciliation rule, five callers

`reconcile_outcome()` is the collective reconciliation primitive, and the
transaction layer has one shared implementation in the coupling sources.
That was not true: five files carried their own copy of the same rule — a
max-reduce for "did anyone fail", a min-reduce for "which rank", then a
broadcast of the failing rank's text. `karma_compiled_fem_connection.C`,
`karma_fem_field_evaluator.C`, `karma_fem_transfer.C`, `karma_fixed_point.C`
and `karma_operator_split_graph.C`, and they were not quite identical, which is
the usual way duplicated collective code drifts.

`reconcile_failure_message()` is now the single implementation. It reconciles
through the primitive and returns the agreed message, or empty when no rank
failed — it **returns rather than throws**, so each caller keeps its own
exception type and wording and nothing about how a failure is reported moved
into the shared layer. Three callers throw `InvalidArgument`, the graph throws
`std::runtime_error` naming the instruction, and the fixed-point path returns
the message as a string, exactly as before.

Two things changed in behaviour, both deliberate:

- **The message reports how many ranks failed.** One rank failing and every
  rank failing are different problems with the same verdict, and none of the
  five copies said which.
- **An MPI failure no longer broadcasts.** The primitive reports that case with
  a *local* rank rather than an agreed one, so using it as a broadcast root
  would have the ranks disagree about the root. The caller still gets a
  nonempty message and still fails.

The success path now costs one extra small reduction, because the primitive
also counts failures. Stated rather than hidden: these functions are called per
instruction.

**Not verified in the sandbox.** The message path needs a real communicator, so
no test here reaches it — mutating its success predicate kills nothing locally.
It is exercised by the `_mpi` legs and the behavioral oracle on a host, and
those must be re-run.

## Units are parsed, not matched

`parse_unit_expression()` reads a product or quotient of symbols with integer
exponents: `*` and a space both multiply, `/` divides the single term that
follows it, `^` takes a signed integer, and `1` is the dimensionless unit so
`1/s` parses. It recognizes the seven SI base units and the derived units whose
dimensions are fixed by definition (`N`, `Pa`, `J`, `W`, `Hz`, `C`, `V`, `Ohm`,
`F`, `T`, `Wb`, `rad`, `sr`), each of which expands to base SI and therefore
compares equal to its expansion.

It replaced a lookup against eight spellings, which could not read `kg/m^3` at
all and called `m/s` and `m s^-1` different quantities.

**Metric prefixes are deliberately refused.** `m` is both metre and milli-, so
a prefix grammar would be ambiguous exactly where being wrong is expensive —
reading `km` as metres is an error of a thousand. A prefixed unit therefore
fails to parse and is reported as undeclared, which is honest.

**Failure is the point.** Anything the grammar cannot read returns no value,
and the adapter records the literal under `unrecognized_units_key` rather than
letting it become dimensionless. Dimensionless is a claim about the quantity;
unreadable is an admission of ignorance, and the compatibility checker refuses
the second rather than silently connecting it to a bare scalar. Two mutants
that matter here both concern refusing rather than wrapping: an exponent
magnitude beyond the dimension's range, and an accumulated exponent that would
overflow across a product. The magnitude bound is the only thing that can
refuse `1^300`, because every exponent of a dimensionless unit is still zero —
which the mutation matrix found by removing the bound and watching nothing
fail.

`TimeSemantics` was widened from one enumerator to four, mirroring
`SemanticTemporal`. A specification that could only say
`InstantaneousCommitted` could not express a disagreement the model checks for,
so connecting an interval average to an instantaneous value was inexpressible
rather than refused. Both cases are now refused by the graph's admission check.

## The executing path consults the semantic model

`compile_operator_split_graph()` admits a connection only after
`check_connection()` agrees that the two ports mean the same thing. This is
where the semantic model stops being a description the library carries and
becomes something a running coupled problem depends on.

It replaced a string comparison. The graph used to require that the two
fields' unit *strings* were equal, which is the defect the dimensional model
exists to remove: `m/s` and `m s^-1` are the same physical quantity and were
refused as different, while string equality could say nothing about scale,
offset, role, or the accepted version window. Structural facts that are not
about meaning - that a connection runs from an output to an input - are still
checked directly.

Two things had to be got right, and one of them was a real defect.

**An explicitly wired connection is its own identity statement.** A
`PortSpec` carries no requirement of its own, so the adapter defaults
`requirement.field_id` to the port's own field id. Left in place, that refuses
every connection whose two ports spell their field differently - which is
every connection in this graph. The wiring names both endpoints and is the
authority on which field satisfies the requirement; everything else the
requirement carries still applies.

**A field whose units could not be interpreted is not dimensionless.** The
adapter recognizes eight unit symbols and records anything else as a
`karma.units.unrecognized` metadata entry holding the literal - precisely so a
downstream check can tell "undeclared" from "dimensionless". The checker was
not looking, so a field declared `kg/m^3` degraded to zero dimensions and
connected happily to a bare scalar. It now refuses unless both sides carry the
same literal, under its own reason rather than as a dimension mismatch: the
dimensions are unknown, not known and different. This is the kind of defect
that only appears when contracts are composed, which is the argument for
composing them.

Each component's report carries a `contract_fingerprint`, the semantic hash of
its canonical contract. It moves when meaning moves and not when a
caller-declared mesh, layout, basis or solver identifier changes, so an
introspection report distinguishes a contract change from a relabelling.

The check adds refusals and computes nothing, so a graph that was already
meaningful compiles to the same instructions. That is what makes the
behavioral oracle the right instrument for confirming it, and the oracle needs
a host run.

## One encoder for canonical bytes

`karma_byte_writer.h` holds the rules that turn an integer, a string, a byte
sequence and a rational into canonical bytes, plus the hash over them. The
field-entry key, the semantic model and the operator chain all use it.

They used to each carry their own copy, and the problem with that was not the
duplication but the asymmetry it produced. The length-prefix rule was
load-bearing in the semantic model - a mutation removing it was killed - and
vestigial in the operator chain, whose record has only one variable-length
field, so no collision was constructible and the same mutation survived by
construction. The rule was therefore enforced in one copy and merely present in
the other, and adding a second string field to the chain would have made it
load-bearing again with no test noticing. `byte_writer_test` now enforces it
once, where it is implemented once: six mutants, including both dropped
prefixes, are killed.

Consolidating also surfaced a defect none of the three copies could reveal on
its own. All of them named FNV-1a and used 1469598103934665603 as the offset
basis, which is the published constant with its last digit dropped. As an
internal identity that was harmless - a perfectly good multiplicative hash,
consistent with itself. It stops being harmless as soon as a Fortran or Python
adapter has to compute the same identity for the same bytes: a real FNV-1a
returns a different number and the two disagree about identity with nothing to
say why. The shared encoder uses the published basis and checks itself against
published vectors.

The four pre-existing copies in the I/O and checkpoint sources keep the old
constant on purpose. Their hashes are written into files, so correcting them
would invalidate every existing checkpoint; they carry their own schema
versions and are a separate migration. None of the three encoders sharing this
header persists its hash, which is why they could be corrected now.

## Naming: version markers belong in data, not in names

No type, function, macro, file or component identifier carries a version or a
project or release label. The ABI headers are `karma_abi.h` and
`karma_component_abi.h`; the exported symbols are `karma_abi_version`,
`karma_negotiate_abi`, `karma_status_name`, `karma_diagnostic_set`,
`karma_validate_field_table`, `karma_abi_supported_capabilities` and
`karma_plugin_query`.

Nothing was lost by dropping the suffixes, because the ABI never depended on
them. Its versioning is *in band* and always was: every record opens with
`struct_size`, the negotiation records carry `abi_major`, `KARMA_ABI_MAJOR`
names the current major, and `karma_negotiate_abi` is the function that
compares them. A suffix on the symbol duplicated that and could drift from it.
The same holds in the registry, where `SemanticVersion` is a field on the
record and the implementation identifier is just a name.

The line to hold when adding code: a version marker is legitimate where it
**is** the mechanism — a dedicated field, or a token that is serialized so a
reader can reject bytes it does not understand. It is not legitimate as
decoration on an identifier. Three markers are kept under that rule, all of
them serialized schema tokens rather than names: a basis's
`compatibility_id()` (written into checkpoints, and pinned by the oracle, so
changing it would change persisted bytes) and the two
`configuration_schema_id` values, whose field name says exactly what they are.

## The stable C ABI

`include/abi/karma_abi.h` and `karma_component_abi.h` are the only stable
surface KARMA offers. They are C11, include nothing but `<stdint.h>` and
`<stddef.h>`, and include each other as siblings in the same directory - unlike
the rest of KARMA's headers, which include their siblings flat and therefore
oblige a consumer to add every `include/karma/<area>` to its include path. A C
consumer needs one include path and one library.

Permanently excluded from the boundary: C++ types, exceptions, templates, PETSc
and MOAB objects, and the platform's native MPI communicator. A communicator
crosses as a framework-owned 64-bit token, because `MPI_Comm` is a pointer in
some implementations and an integer in others, and neither Fortran nor Python
can portably spell it.

### A separate shared object

The ABI is built as `libkarma_abi`, not as part of `libkarma`, and it is the
only target with `CXX_VISIBILITY_PRESET hidden` and an explicit export list
(`-exported_symbols_list` on Mach-O, `--version-script` on ELF). The reason is
practical: 70 test executables and the examples link against libkarma's C++
symbols, so hiding those would break the build, and exporting the ABI's
implementation details would turn them into compatibility promises. `libkarma`
is linked PRIVATE, so a C consumer inherits no PETSc, MOAB, or MPI flags -
verified by inspecting the consumer's own link list, which names only
`libkarma_abi`.

`karma_abi_exports` asserts that the exported set *equals* the documented list,
currently seven symbols. Nothing else would catch an accidentally exported
symbol.

### Negotiation

Every record begins with `struct_size`. A record smaller than this version can
read is a `TRUNCATED_PREFIX` - the caller is older. A record *larger* is a
future tail, read up to the part we understand: that is the whole
minor-extension mechanism. A size of zero, or above 64 KiB, is
`MALFORMED_SIZE`, reported separately because a wild size is what a corrupted
record looks like, not a very new one. The out parameter is size-checked too;
writing past a caller's smaller record would corrupt its stack.

A foreign major version is refused, and the refusal still reports the library's
own major so the caller learns what to ask for. A newer minor talks down. An
unknown *required* capability bit is a refusal with the offending bits named,
because ignoring it would tell a participant it received a guarantee it did
not; an unknown *optional* bit is simply not granted. An undersized host table
is answered with the size required, so discovery replaces guesswork.

### Time

`advance` takes `karma_exact_interval`, a pair of
`{int64 numerator, int64 denominator}`. There is an optional tail function,
`advance_seconds`, for a participant with no rational arithmetic; it converts
through the bounded dyadic conversion and *refuses* a value it cannot represent
exactly rather than rounding. `0.5` is accepted; the double nearest `0.1` is
refused, because its denominator is near 2^55 and the first addition would
overflow.

### Exception containment, and one honest limitation

No exception may cross the boundary: unwinding through a C, Fortran, or Python
frame is undefined. Every ABI function body is guarded, and the guard is
unconditional rather than keyed to `KARMA_ENABLE_EXCEPTIONS`, because that
option does not add `-fno-exceptions` - it only changes what `KARMA_THROW`
does. A fixture component (`karma.reference.throwing`) throws from
`advance`, and a test requires a status and a diagnostic rather than a crash.

The limitation, stated rather than hidden: with `KARMA_ENABLE_EXCEPTIONS` OFF,
`KARMA_THROW` calls `std::abort()`, so an internal KARMA failure terminates the
process before any guard can convert it. The ABI implementation therefore
validates locally and returns statuses instead of relying on KARMA internals to
report errors, which keeps the boundary well-behaved in both configurations for
everything it validates itself. Both configurations are built and the ABI tests
run in each.

### The record schema is golden

`language_adapter_schema_test` asserts literal byte offsets and sizes for every
record. A Fortran or Python adapter cannot include a C header - it declares the
records itself at hard-coded offsets - so those offsets are an interface. If one
changes, the failure should be in that test rather than in a participant's
memory.

### Installed ABI smoke test

`karma_abi_install_consumer` installs KARMA into a scratch prefix and then
configures, builds, and runs a standalone pure-C CMake project against it, with
`-I<prefix>/include` as its only include path and `<karma/abi/karma_abi.h>`
as its include spelling. It negotiates, creates a component, imports a host
array, checkpoints, advances over an exact interval, exports, restores, and
receives a named deterministic error. That program is the ABI's executable
compatibility check; `abi-soma-gap-table.md` records the interface differences
that a future adapter must preserve.

## Connection compatibility and the operator chain

A connection between two semantic fields is either one explicit operator chain
or a refusal. There is no runtime fallback, because a fallback that substitutes
an operator changes the meaning of the coupled quantity while reporting success.

`check_connection()` evaluates seven stages - identity, unit, shape, location,
temporal, conservation, capability - in that order, and **every stage runs**. One
call therefore reports every problem rather than the first, while
`first_incompatible_stage` names the earliest refusal, which is the one to fix
first. Required capabilities are collected even when an earlier stage already
refused, so a caller fixing units does not then discover a missing capability on
the next attempt. A refused connection carries no chain at all: a partial chain
is what a fallback would reach for.

Five verdicts are distinguished, and the distinction is about who decided:

| Verdict | Meaning |
| --- | --- |
| `Exact` | The fields already agree; the chain is empty. |
| `LosslessExplicit` | Compiles, and no operator discards, invents, or resamples. |
| `LossyExplicit` | Compiles, and every information-changing operator in it was explicitly requested. |
| `Ambiguous` | An operator would be needed and was not requested, or two operators could close the same gap and the choice is the caller's. |
| `Unsupported` | No chain can express the connection. |

Exactly one operator may be inserted without being asked for: an exact
zero-offset unit scale, and only where site policy allows it. Everything else
changes what the value means rather than how it is spelled. An offset difference
is an affine conversion - Celsius to Kelvin moves the zero of the scale - and is
never automatic. Reducing three components to one is refused as *ambiguous*
rather than resolved, because selection and aggregation are different physics;
requesting both leaves the choice unmade and is also refused.

Reason codes are stable numbers that are never reused or renumbered. The enum is
named `ConnectionReason` rather than `CompatibilityReason` because
`karma_semantic_version.h` already owns the latter for version negotiation: the
two layers answer different questions - may this reader accept this contract,
versus may these two fields be connected - and collapsing them into one name
would force one to be renamed later.

Time is compared exactly. Two intervals spelled `1/3` and `2/6` are the same
interval and compare equal; a tolerance here would make rendezvous equality
depend on how a step size was written.

Chain order is a semantic property, not formatting. Unit and component
transforms act in the source's own space and precede spatial transfer; spatial
precedes temporal, because projecting in time a field not yet mapped in space
projects the wrong samples; constraints act last, on the values that will be
published. `chain_is_ordered()` is therefore a correctness check, and within the
transform phase independent operators commute and are left unordered.

The chain's transform-phase operators are *defined as* `TransformKind`'s values
rather than repeated, with static assertions that fail if either is renumbered,
so the vocabulary a caller declares and the vocabulary the compiler emits cannot
drift apart.

Two things this checker deliberately does not infer. A sign convention has no
representation in `FieldSemantics`, so `SignFlip` exists in the vocabulary and
requires an explicit request but is never inferred - there is nothing to infer
it from. And declaring a transfer other than a copy *is* the explicit request
for a spatial transfer; there is no second place for a caller to say the same
thing.

Mutation evidence is under `build-evidence/compatibility-mutations`: 16 mutants killed
across both layers, including a reordered stage sequence, a silent conservation
downgrade, an inverted conservation comparison, a retained chain after refusal,
the last rather than first stage blamed, loss ignored in the verdict, an affine
conversion inserted automatically, a dropped capability closure, structural
rather than exact rational interval comparison, an unset version maximum treated
as a bound, and operator identity dropped from the chain hash. One mutant is
recorded as *equivalent* rather than survived, with the reason.

### What still publishes through the old path

`KARMAField::commit_petsc_values()` and `commit_moab_values()` have not been
rewired to forward through this API, and the reason is specific rather than
scheduling. The MOAB path is positional and dense and would map cleanly. The
PETSc path does not: it writes by *global* index through `VecSetValues` followed
by assembly, which routes values for entries this rank does not own. A dense
local layout cannot express that. Unifying it therefore requires either
restricting the commit to owned entries, which is a behavior change to verified
code, or a scatter-aware storage backend that assembles. Neither can be
exercised where PETSc and MPI cannot run, so both are left for a host-verified
step rather than changed on an untested assumption.

`semantic-model-soma-mapping.md` maps every record onto SOMA's graph
descriptor as a correspondence record, marking each row same,
narrower, absent, or a placement difference against a read of SOMA's source.

See the [extensible coupling design review](plugin-design-review.md) for integration
boundaries, ownership requirements, numerical contracts, and scalability criteria.
