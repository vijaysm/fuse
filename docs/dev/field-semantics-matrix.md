# Field semantics: KARMA and SOMA against the current FUSE tree

Track "Field semantics", Stage S8.1. Scratch-only, read-only against the owner's live
checkout (`/tmp/s8fields`, unpacked from `git archive HEAD` at `8820aed`); nothing under
`/Users/mahadevan/Code/AI/coupling` was touched. No code was written for this track; every
claim below is READ (file:line) or MEASURED (a grep/count run in this task), and is marked
as such. Nothing is INFERRED without being labeled.

Subject: the eight KARMA coupling headers whose porting.md successors could not be matched
by symbol name in `docs/dev/physics-kit-disposition.md` Section 1.2 (physics-kit-disposition.md:118-130):
`karma_field_coverage.h`, `karma_field_entry_encoding.h`, `karma_field_identity.h`,
`karma_field_transaction.h`, `karma_field_transfer.h`, `karma_semantics_adapter.h`,
`karma_transfer_diagnostics.h`, `karma_reference_components.h`, each read in full together
with its `.C` file (`karma_field_identity.h` has no `.C`, confirmed by a MEASURED grep:
`find upstream/karma -iname "karma_field_identity*"` returns only the header).

This document does not use `docs/dev/porting.md`'s historical successor names as evidence;
every claim of presence or absence in FUSE is a grep and a read of the CURRENT tree.

## 1. Plain verdict (8 rows)

| # | KARMA header | Verdict | One-line reason |
|---|---|---|---|
| 1 | `karma_field_coverage.h` | GAP | No declared, validated fill-value for an under-covered transfer target anywhere in FUSE; FUSE's own coverage audit (`min_src_coverage`/`min_tgt_coverage`, remap_plan.hpp:122-129) is a global minimum, not a per-entity classification a caller can act on. |
| 2 | `karma_field_entry_encoding.h` | GAP | No canonical, versioned, endian-stable, FNV-1a-hashed byte encoding of a field-entry identity exists in FUSE; FUSE hashes whole payloads for file integrity, never one entry's stable identity for cross-rank reassociation. |
| 3 | `karma_field_identity.h` | GAP | No `FieldEntryKey`/`DofKey`-equivalent variant identity (MOAB-global-id vs DOF-basis identity) exists; FUSE's `FieldRef`/`Layout` is purely a dense positional index, and the mesh layer's `global_id()` cannot name a continuous-Galerkin degree of freedom shared by two elements. |
| 4 | `karma_field_transaction.h` | GAP | No staged, all-or-nothing field publication (`FieldStorage`/`UpdateMode`/commit/rollback/`restore_committed`/`commit_collective`) exists anywhere in FUSE; field read/write is direct `FieldRef` memory access with no transactional boundary and no canonical (arrival-order-independent) accumulation discipline. |
| 5 | `karma_field_transfer.h` | GAP | No probed-capability registry over transfer backends (`TransferCapabilities`, a diagnostic string naming why one is unavailable, an explicit `TransferState` machine) exists; FUSE's `TransferSemantics`/`ConnectionCompatibility` is a richer DECLARATIVE classification of what a channel promises, not a query of what is actually buildable. |
| 6 | `karma_semantics_adapter.h` | GAP (narrow) | The field/port/component adaptation core (`to_semantics`, units-undeclared-not-dimensionless, temporal totality) is SUPERSET-CONFIRMED with FUSE tests; the one missing piece is the bounded-dyadic-double-to-Rational conversion with named refusal codes (`exact_time`/`rational_from_dyadic_bounded`) -- `fuse::Rational` has no double-ingesting constructor at all. |
| 7 | `karma_transfer_diagnostics.h` | GAP | The measure-weighted relative-conservation check is SUPERSET-CONFIRMED and collectively verified (`Constrain`, runner.cpp:5112-5230) with one verified numerical DIFFERENCE (zero-reference floor); the PETSc mass-matrix residual-based transfer-accuracy check (`transfer_residual_report`, backward-error scale) has no FUSE equivalent at all. |
| 8 | `karma_reference_components.h` | GAP | No component registry with named-contract selection, ambiguous-selection refusal, duplicate-registration refusal, or a reusable compliance harness that cross-checks a record's claims against the running component exists in FUSE; this is the same gap `physics-kit-disposition.md`'s own open question 6 names for `karma_registry.h`. |

No header in this set is PORT: in every case the SEMANTIC the header carries is well-defined
and independently motivated (not merely a container KARMA happened to need), so the right
unit of work is the semantic addition named in Section 4 below, built in FUSE's own
vocabulary, not a transplant of the KARMA type.

## 2. Semantics matrix

Rows are every semantic KARMA or SOMA carries, grouped by header/concern. Columns are
KARMA, SOMA, FUSE. A FUSE cell names the file:line that carries the semantic and, where one
exists, the FUSE test that pins it; `MISSING` means a MEASURED grep found nothing.

### 2.1 Field entry identity (karma_field_identity.h, karma_field_entry_encoding.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A field entry's identity is either a mesh-global id or a DOF-basis identity, one closed variant | `FieldEntryKey{Kind::MoabGlobalId\|Dof}` (karma_field_identity.h:49-62) | MISSING (SOMA has no per-entry key at all; `FieldId` (base/ids.hpp:41) names a whole field, not one entry) | MISSING (grep for `FieldEntryKey`, `DofKey` under include/fuse, src: zero hits) |
| DOF identity is stable across ranks: variable, component, entity dimension, mesh entity type, canonical slot, then identity words (vertex ids for an intermediate cG entity) | `DofKey` layout, karma_field_identity.h:29-47; `karma_field_entry_encoding.h:14-41`'s layout table | MISSING | MISSING -- FUSE's mesh layer exposes one `global_id(dim, index)` per entity (backend/transfer_map_io.cpp:112-113), with no basis-function-level identity for a shared higher-order DOF |
| A canonical, fixed-width, little-endian, versioned byte encoding of that identity, shared by every new contract rather than reinvented per subsystem | `encode_field_entry_key`/`decode_field_entry_key`, karma_field_entry_encoding.h:62-71; versioned via `field_entry_encoding_version` (:42) | MISSING | MISSING |
| The encoding round-trips exactly, decodes identically on every rank regardless of who produced it, and the kind/component/word position all participate in a schema-version- and length-checked decode | KARMA test `field_entry_encoding_test.C` (round_trip :59-88, canonical_bytes :90-112, distinct_encodings :114-123, malformed_input :125-154) and MPI test `field_entry_encoding_mpi_test.C` (reassociation across ranks, :80-146, collective `MPI_Allreduce(MAX)` failure agreement :142-144) | MISSING | MISSING |
| A stable FNV-1a hash of that encoding, usable as a distributed lookup key, pinned against a literal value so a change of hash function is visible | `field_entry_key_hash`, karma_field_entry_encoding.h:76; pinned in `field_entry_encoding_test.C:27,165-166` (`expected_moab_hash = 0xe7f96370571789f1ULL`) | MISSING | `fuse::bytes::fnv1a` exists (base/bytes.hpp) and is used for WHOLE-PAYLOAD file-integrity hashing (`src/backend/weights_operator.cpp:176`, `src/model/semantics.cpp`, restart descriptor hash `fuse/model/restart.hpp` "restart_descriptor_hash"), never for one field entry's own identity |

### 2.2 Transaction / staged commit (karma_field_transaction.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A publication target is an abstract `FieldStorage` interface (`value_count`, `read_into`, `publish`), implemented over a host array, a PETSc `Vec`, or a MOAB tag, all driven through the same contract | karma_field_transaction.h:40-65; PETSc impl `karma_petsc_field_storage.h:37` (`PetscVectorStorage final : public FieldStorage`) | MISSING | MISSING -- `fuse::FieldRef`/`FieldRegistry` bind a raw pointer (field/field_ref.hpp:125-182, field/registry.hpp:25-107); there is no `FieldStorage`-like polymorphic publish/read interface at all |
| No caller-visible value changes until every staged value has passed LOCAL validation ("stage" is transaction-private memory) | karma_field_transaction.h:11-23, `FieldTransaction::stage` (karma_field_transaction.C:69-83) keeps values in `_staged`/`_contributions`, not the target | MISSING | MISSING |
| Three update modes with distinct ambiguity rules: `Replace` (every slot once, a second value for one slot refused), `SparseReplace` (unstaged slots keep the published value, duplicates still refused), `Accumulate` (slots may be left unstaged, contributions summed at commit) | `UpdateMode`, karma_field_transaction.h:26-37; tested `field_transaction_test.C`: replace_commits :78-101, rejects_bad_staging :103-122, sparse_insertion :124-144, canonical_accumulation :146-178 | MISSING (SOMA's `Channel`/`AccumOp` (graph/channel.hpp:37-40) names a temporal reduction on a VALUE already computed; it is not a staged multi-writer commit discipline over a field's own storage) | MISSING |
| Accumulation is canonicalized at commit (sorted by slot then by a deterministic magnitude-then-bit-pattern total order), so the published value is bitwise independent of staging/arrival order | karma_field_transaction.h:18-22; `canonical_before`, karma_field_transaction.C:45-54; `build_published_values`'s accumulate branch, karma_field_transaction.C:126-150; proved bitwise by `canonical_accumulation`, `field_transaction_test.C:146-178` (`std::bit_cast<std::uint64_t>` equality forward vs reverse staging order) | MISSING | MISSING -- FUSE's own `Constrain`/reduction code (runner.cpp) uses `num::weighted_sum`/`deterministic_dot` for a COLLECTIVE integral reduction, a different concern (one scalar sum per rank, reduced across ranks) with its own determinism discipline, not a per-slot multi-writer commit canonicalization |
| `commit()` validates then publishes; on ANY failure (including a storage backend failing mid-write) nothing is published and no caller-visible value changes | karma_field_transaction.C:153-174, 108-151; proved by `storage_failure_leaves_no_partial_mutation`, `field_transaction_test.C:203-216`, using `FailingHostArrayStorage` (karma_field_transaction.h:70-85) which fails mid-write on purpose | MISSING | MISSING |
| `rollback()` (discard staged state, infallible) is a DIFFERENT operation from `restore_committed()` (republish the pre-commit snapshot, fallible because it is itself a publication) | karma_field_transaction.h:107-115; tested `rollback_and_restore`, `field_transaction_test.C:180-201` | MISSING | MISSING |
| A collective commit agrees first (every rank's local validation), then every rank publishes or none do; a publication failure after agreement is reported by the failing rank, not silently | `commit_collective`, karma_field_transaction.C:194-205, via `reconcile_outcome` | MISSING | The SAME ordering discipline (agree first via a reduction, then every rank takes the same branch) IS present elsewhere in FUSE's runtime for a DIFFERENT boundary: `Constrain`'s own collective verdict (runner.cpp: `MPI_Allreduce` of `{before, after, step_failure flag}` BEFORE the tolerance check, so every rank sees identical totals) and `finish_point()`'s restart global-id verification (`fuse/model/restart.hpp:30-52`). There is no field-IMPORT/commit analogue of it. |
| A `FieldIndexMap` resolves a stable key to a dense index once at setup, counting lookups so a test can prove a warmed hot path performs none; ownership is positional (owned entries precede ghost entries) | `FieldIndexMap`, karma_field_view.h:55-92; tested `index_map`, `field_transaction_test.C:41-64` | MISSING (ownership in SOMA is a mesh/`Region` concept, not encoded in `FieldRef`/`Layout` at all) | MISSING -- `fuse::Layout` (field/field_ref.hpp:63-116) has no owned/ghost split; `FieldSpan::owned(n)` (field/pipeline.hpp:139-141) takes the split point FROM THE CALLER rather than deriving or enforcing it, and there is no stable-key-to-index resolution layer anywhere (MEASURED grep for "IndexMap" under include/fuse, src: zero hits) |

### 2.3 Transfer capability probing (karma_field_transfer.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A named transfer FAMILY (pointwise interpolation, quasi-interpolation, variational projection, conservative remap) independent of the specific scheme | `TransferFamily`, karma_field_transfer.h:12-18 | MISSING (SOMA names a method by a free-text `RemapSpec::method_override`, graph/channel.hpp:44, or derives one from discretization + conservation) | Declaratively present as `TransferKind::Copy/Interpolation/ConservativeRemap` (model/semantics.hpp:201), a coarser 3-way classification of what a channel PROMISES, derived in `declared_transfer` (model/connect.hpp:99-112); tested `Connect.DeclaredTransferIs*`, test/model/connect_test.cpp:96-130 |
| A specific SCHEME within a family (12 named schemes from element-constant to conservative-supermesh), each with its own options variant, validated for finiteness/range at construction | `TransferScheme`, `TransferOptions` variant, karma_field_transfer.h:20-74; `validate_options`, karma_field_transfer.C:42-62 | MISSING (method naming is a free string) | Present as separate backend TYPES/FUNCTIONS (supermesh.hpp, consistent_interp.hpp, general_interp.hpp, remap_weight_normalize.hpp), not a closed enumeration a caller can switch over before committing to one |
| A probed `TransferCapabilities` struct (available / supports_parallel / supports_coverage / can_build_map / guarantees_conservation / unavailable_reason) a caller can query BEFORE attempting setup, including for schemes that are not yet implemented | karma_field_transfer.h:76-107; tested `field_transfer_contract_test.C:1-33` (Shepard with an invalid exponent refuses and reports why; the FEM capability struct asserts available+can_build_map+not-conservative; the conservative-supermesh struct asserts unavailable with a named reason) | MISSING | MISSING -- there is no struct or function anywhere in FUSE a caller can call to ask "is scheme X available on this build, and if not, why" before attempting the transfer; a missing backend fails at the point of use, not at a query point |
| An explicit `TransferState` machine (`Uninitialized -> Ready -> Failed`), with `setup()` atomic on failure (state/diagnostic rolled back) and `apply()` refusing outside `Ready` | karma_field_transfer.h:36-41, 71-96 | MISSING | MISSING |

### 2.4 Semantic adaptation and exact time (karma_semantics_adapter.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A legacy spec-record (`FieldSpec`/`PortSpec`/`ComponentSpec`) adapts into the versioned semantic model (`FieldSemantics`/`PortSemantics`/`ComponentSemantics`), stating only what the spec declared and leaving the rest genuinely unspecified rather than defaulted | `field_semantics_from_spec`/`port_semantics_from_spec`/`component_semantics_from_spec`, karma_semantics_adapter.h:27-31, .C:56-119 | MISSING (SOMA's `FieldSpec`, graph/field_spec.hpp:12-21, IS the only semantic record; there is no second, richer semantic layer it adapts into) | `fuse::to_semantics(const FieldSpec&)`, model/connect.hpp:46-76, with the SAME "leave what the graph cannot state as Unspecified" discipline stated explicitly at connect.hpp:14-22; tested `Connect.AxesTheGraphCannotStateStayUnspecified`, test/model/connect_test.cpp:42-49 |
| An unrecognized unit literal is preserved verbatim under a named metadata key and is NOT treated as dimensionless | karma_semantics_adapter.C:69-72, `unrecognized_units_key` (karma_semantics.h:315); tested `specification_adaptation`, coupling_semantics_test.C:394-409 | MISSING (SOMA's `Units` (base/units.hpp:22-77) is pure SI exponents with no provenance for an unparsed literal; there is no unit-STRING layer in SOMA's graph at all -- `FieldSpec::units` is already a `Units` value) | SAME discipline: `unrecognized_units_key` (model/semantics.hpp:235), `ConnectionReason::UnitsUndeclared` (model/compatibility.hpp:76), surfaced as `E-UNITS-UNDECLARED` (model/validate.hpp:433); tested `Compatibility.UndeclaredUnitsAreNotDimensionless`, test/model/compatibility_test.cpp:464 |
| A unit COMPARISON is dimension-only (scale/offset ignored) for one named check, while a SEPARATE exact affine conversion (scale, offset) is tracked for the cases that need it | `unit_dimensions_match` (karma_semantics.h:193) vs `UnitDimensions::scale/offset` (karma_semantics.h:159-160) | SOMA compares units by EXACT STRUCT EQUALITY only (`sf.units == tf.units`, graph/validate.hpp:131), with no separate scale/offset axis at all -- a strictly coarser check than either KARMA or FUSE | FUSE graduates this into a full 7-stage connection check (`ConnectionStage::Unit`, model/compatibility.hpp:29, 63-93) distinguishing dimension mismatch, an automatic zero-offset scale, an explicit affine conversion, and a refused sign-flipping scale; tested `Compatibility.UnitRules`, `Compatibility.AffineConversionRespectsTargetScale`, test/model/compatibility_test.cpp:465,473 |
| Temporal interpretation (`InstantaneousCommitted`/`InstantaneousTrial`/`IntervalAverage`/`IntervalIntegral`) adapts totally: every legacy case reaches a distinct semantic value, never collapsing two real distinctions into one | `adapt_temporal`, karma_semantics_adapter.C:41-53; tested totally over all 4 cases, `temporal_adaptation_is_total`, coupling_semantics_test.C:342-357 | MISSING (SOMA has no temporal-interpretation axis on a field at all; temporal reduction is a CHANNEL property, `AccumOp`, base/ids.hpp:51, not a field semantic) | `SemanticTemporal` enum exists verbatim (model/semantics.hpp:54-60) but `to_semantics()` leaves it `Unspecified` (connect.hpp:16-22, since `FieldSpec` carries no temporal-interpretation field at all, matching SOMA's own `FieldSpec`) -- the GRAPH-DECLARED axis is absent in both SOMA and FUSE; FUSE merely carries the vocabulary to express it once a field spec grows one |
| An exact conversion from a legacy DOUBLE scheduler position into a rational, bounded so a denominator too wide for exact arithmetic is REFUSED (`NotDyadic`) rather than silently admitted to later overflow | `exact_time`/`exact_interval`, karma_semantics_adapter.h:33-53, .C:121-133, calling `rational_from_dyadic_bounded` (karma_semantics.h:98-99); tested `specification_adaptation` (coupling_semantics_test.C:416-422) and the bound itself in `dyadic_conversion` (coupling_semantics_test.C:74-105, in particular :89-96: a 0.1-literal's 55-bit denominator is refused at a 32-bit bound, and two such values would overflow on the first addition) | MISSING (SOMA's `Rational` (base/rational.hpp, referenced by `ComponentDecl::interval`, graph/component.hpp:31) is constructed directly from integers; there is no double-ingesting path to adapt at all) | **MISSING.** `fuse::Rational` (base/rational.hpp:86-107) has `to_double()` (one-way, diagnostic only) but NO double-to-Rational constructor or free function of any kind, bounded or not (MEASURED grep for "from_dyadic"/"NotDyadic"/"maximum_denominator" under include/fuse/base/rational.hpp: zero hits). This is the one piece of `karma_semantics_adapter.h` with no present FUSE counterpart; see Section 4. |

### 2.5 Transfer-accuracy diagnostics (karma_transfer_diagnostics.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A relative integral error with an explicit zero-reference convention (reference 0, value 0 gives 0; reference 0, value nonzero gives `+infinity`, never a finite-but-meaningless ratio) | `relative_integral_error`, karma_transfer_diagnostics.C:12-18 | MISSING | **DIFFERENT, not superset.** `Constrain`'s own relative error (runner.cpp, around the `scale`/`rel` computation) uses `scale = |before| if nonzero else 1.0`, so a zero-reference, nonzero-after case reports a FINITE number (`|after|`) rather than KARMA's `+infinity`. Both are defensible backward-error-style conventions; they are NOT the same number for the same inputs. Flagged as a verified numerical difference, not assumed equivalent. |
| A per-component integral conservation report (source / target-full / target-covered / fill-or-extension integrals, and the relative error of each against the source) | `TransferIntegralReport`, `make_transfer_integral_report`, karma_transfer_diagnostics.h:13-21,32-34, .C:20-40 | MISSING | Collapsed into a single collectively-reduced scalar per `Constrain` instruction (one before/after pair, `ConstrainInfo`, runner.cpp): no per-component breakdown, no separate "covered" vs "full" vs "fill-or-extension" integral -- a coarser report of the same underlying measured-integral concern |
| The conservation check is COLLECTIVE and reduced before the verdict is formed, so every rank reaches the same accept/refuse decision from the same totals, and a partner's already-failed period does not turn into a spurious local conservation error | Not explicit in `karma_transfer_diagnostics.h` itself (the header's functions are local, PETSc-collective-norm-based, not multi-rank-reduced by this code) | MISSING | **Superset.** `Constrain`'s `MPI_Allreduce` of `{before, after, step_failure flag}` (runner.cpp) happens BEFORE the tolerance comparison, and a failed-elsewhere flag suppresses a spurious local violation -- a stronger collective-agreement discipline than KARMA's own diagnostics header provides, matching this project's standing rule that a collective path never returns on a locally computed condition |
| A mass-matrix RESIDUAL check for a projected/transferred field: recompute `residual = mass*solution - rhs`, test `\|\|residual\|\|_inf <= tolerance * scale` with `scale = max(\|\|rhs\|\|_inf, \|\|mass\|\|_inf*\|\|solution\|\|_inf, eps)` -- a backward-error-style bound, not a flat tolerance on the raw residual | `transfer_residual_report`/`transfer_residual_within_tolerance`, karma_transfer_diagnostics.h:36-42, .C:42-75; used with a flat `1.0e-11` at `cg_transfer_accuracy_test.C:290` (itself flagged elsewhere, `physics-kit-disposition.md:244`, as "no derivation shown" for that constant) | MISSING | **MISSING.** MEASURED grep for "residual_within_tolerance", "transfer_residual", "MatMult.*residual" under include/fuse, src: zero hits. There is no PETSc-mass-matrix-residual-based transfer-accuracy check anywhere in the current FUSE tree. |

### 2.6 Reference component registry and compliance (karma_reference_components.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A component registry maps a named, versioned implementation id to a factory, grouped under a CONTRACT id; registering the same implementation id twice is refused by name (`DuplicateImplementationId`) rather than silently replacing it | `register_native_component`/`ComponentRegistry`, exercised via `register_reference_components`, karma_reference_components.C:198-205; tested `static_registration`, component_compliance_test.C:48-61 | MISSING | **MISSING.** MEASURED grep for "ComponentRegistry", "SelectionRequest", "SelectionPolicy" under include/fuse, src: zero hits. `fuse::Component` (runtime/component.hpp:265) is a single abstract base with no registry, no implementation-id namespace, and no selection layer above it. This is the same gap `physics-kit-disposition.md` Section "open questions" item 6 names for `karma_registry.h` (not one of this track's 8 headers, but the hard dependency `karma_reference_components.h` sits on): still open as of this analysis. |
| Selecting by CONTRACT id alone, when two implementations satisfy it, is refused as `Ambiguous` rather than resolved by registration order; selecting an absent implementation id is `NotFound`, never substituted | component_compliance_test.C:62-68 (`by_contract` request -> `Ambiguous`; absent id -> `NotFound`) | MISSING | MISSING |
| A reusable COMPLIANCE HARNESS drives a component through the full ABI lifecycle (create, bind_host, initialize, import_field, advance, commit, export_field) and checks the RUNNING component's behavior against what its OWN registry record claims, catching a record that overstates its capabilities or disagrees about its version | `check_component_compliance`, exercised in `compliance_harness`/`harness_detects_a_lying_record`, component_compliance_test.C:64-108 (checks > 20 per component; an overstated `KARMA_CAPABILITY_CANCELLATION` bit and a forged version are both caught, each with a named failure message) | MISSING | **MISSING.** No compliance-harness function exists anywhere in FUSE; `fuse::Component::capabilities()` (model/capabilities.hpp) is declared BY the component and consumed directly (`check_capabilities()`, src/model/compatibility.cpp, per model/semantics.hpp:142-149's own comment), with no independent runtime check that a component's declared bits match what it actually does. |
| At least one dependency-free, analytically-steppable, bitwise-reproducible reference component exists in-tree purely to exercise the runtime contract end-to-end, with no PETSc or MPI dependency, so the compliance harness is runnable anywhere the library builds | `ReferenceDecayComponent` (exponential decay, exact per-step factor `exp(-rate*dt)`), karma_reference_components.C:15-116; two registered instances on purpose (`baseline`, `extended`) specifically so a selection test can prove ambiguity is refused rather than resolved silently, karma_reference_components.C:198-205 | MISSING | **MISSING.** No in-tree "reference" component exists purely to exercise `fuse::Component`'s own contract; FUSE's existing minimal examples (dn_heat, Brusselator, etc, per `physics-kit-disposition.md`'s own S7 citation) are physics examples, not contract-exercise fixtures, and none is registered twice under one contract to prove an ambiguity refusal. |
| The capability-bit vocabulary a component declares and a caller's `required_capabilities` demand are the SAME 64-bit space, so a selection and a requirement never silently disagree about what a bit means | `KARMA_CAPABILITY_*` bits shared between `ComponentSemantics::capabilities` and a port's requirement (karma_reference_components.C:118-120) | MISSING | Present and explicitly documented as shared: `FieldRequirement::required_capabilities` and `ComponentSemantics::capabilities` are "the SAME 64-bit space ... `check_capabilities()` ... tests this against a connection's `available_capabilities` bit for bit" (model/semantics.hpp:142-149,168-169); the bit NAMES themselves (`kCapabilityRollback`, `kCapabilityResidual`, etc, model/capabilities.hpp:128-183) are FUSE's own fixed-point/Newton/restart vocabulary, not KARMA's, but the ONE-SPACE discipline is the same semantic and is present |

### 2.7 Field coverage and fill policy (karma_field_coverage.h)

| Semantic | KARMA | SOMA | FUSE |
|---|---|---|---|
| A per-POINT coverage classification (`Full`/`Partial`/`Uncovered`) naming which target entries a transfer actually reached | `PointCoverage`, karma_field_coverage.h:10-14 | MISSING | MISSING |
| A per-ELEMENT coverage classification carrying the MEASURED covered fraction (not just a 3-way class) | `ElementCoverage`, karma_field_coverage.h:16-21 | MISSING | MISSING, though FUSE's `min_src_coverage`/`min_tgt_coverage` (backend/remap_plan.hpp:122-129) measure the SAME underlying quantity (fraction of an entity's measure actually accounted for by the supermesh) -- reduced to a single GLOBAL MINIMUM over all entities, never a per-entity value a caller can inspect or act on |
| A declared, per-component FILL VALUE for an entry a transfer did not cover, validated for finiteness and for matching the field's own component count | `FillPolicy`, karma_field_coverage.h:23-31, validated .C:9-14; tested `transfer_map_test.C:7-13` (a valid 2-component fill passes; a 1-component fill against a 2-component field throws) | MISSING | **MISSING.** FUSE's documented policy for an under-covered target cell is the OPPOSITE of a declared fill: the value is "diluted toward zero" by construction (remap_plan.hpp:86-91), because renormalizing or filling would make every matrix report perfect conservation while quietly moving mass onto an arbitrary fraction of a boundary cell (remap_plan.hpp's own stated rejection of that alternative). There is no mechanism to opt into KARMA's declared-fill behavior instead. |

Note on `PointCoverage`/`ElementCoverage` specifically: a MEASURED grep of KARMA's own test
tree (`grep -rn "PointCoverage\|ElementCoverage" upstream/karma/test`) finds **zero** hits;
production use is limited to one field declaration, `karma_fem_transfer.h:17`
(`std::vector<PointCoverage> coverage;`), itself not independently exercised by any test this
grep found. These two structs are weakly-evidenced even within KARMA; a FUSE addition should
target the underlying CONCERN (what fraction of each target entity was covered, and what
should fill the rest) rather than porting these two specific, untested structs.

## 3. Numtypes (CoupledValue / value_traits / BasicFieldRef / dual) against the matrix

MEASURED: `grep -rln "CoupledValue\|value_traits\|BasicFieldRef\|dual<" include/fuse src/`
returns zero hits on direct re-check of each pattern (one earlier combined grep -l hit on
`src/runtime/runner.cpp` was traced to the substring "resi"+"dual<Space>" inside
`num::residual<Space>(...)` -- not a real match, confirmed by re-running each pattern
separately against that file). These names appear ONLY in
`docs/dev/physics-kit-disposition.md` (lines 14, 389, 561, 732) and
`docs/dev/integrators-design.md`, describing the numtypes track (D-E) as landing IN
PARALLEL, not yet in this tree.

Consequence for this matrix: there is currently no value-typing layer to compare against
KARMA's or SOMA's field semantics, because every field in the current tree is a `double`
(`fuse::FieldRef`/`FieldSpan` are built on `Kokkos::View<double**, ...>`, field/field_ref.hpp,
field/pipeline.hpp) exactly as KARMA's `Real` and SOMA's `double*` are. No current FUSE
semantic is "expressed differently" by value-typing, because value-typing has not landed.

The one forward-looking risk this track's own semantics surface, worth naming as an open
question now rather than after `BasicFieldRef<T>` lands: KARMA's `FieldTransaction`
accumulation canonicalization (Section 2.2 above) is defined in terms of a TOTAL ORDER on
`double` (magnitude, then bit pattern, karma_field_transaction.C:48-54) specifically so a
sum is bitwise independent of arrival order. A `dual<S,N>` (an AD-forward value carrying a
primal plus N tangent components) has no single scalar magnitude to order by across its
tangent slots, so "canonical accumulation order" is not a definition that transfers
unchanged -- it would need either a per-slot ordering (primal, then each tangent
independently) or an explicit decision that bitwise order-independence is a primal-only
guarantee for a `dual` field. Since FUSE has no accumulation-canonicalization mechanism at
all today (Section 2.2: GAP), this is not a currently-broken invariant, only a dependency to
record before anyone builds the gap-closing addition on a scalar-only assumption.

## 4. Proposed additions (one per GAP, smallest first)

Each names the SMALLEST addition and the layer it belongs in (`model`, `field`, or
`runtime`, per this project's own layering document, `docs/dev/layering.md`, referenced at
`include/fuse/model/restart.hpp`'s own layering comment). None of this is a decision to
build; each is an `open_questions` entry below, for the owner.

1. **Bounded double-to-Rational conversion** (closes the narrow gap in Section 2.4). Smallest:
   a free function in `fuse/base/rational.hpp`, `Result<Rational> rational_from_dyadic_bounded
   (double, unsigned max_denominator_bits)`, returning a `Failure` (FUSE's own `Result`
   discipline, not a new error-code enum) on a nonfinite value or a denominator exceeding the
   bound. Layer: `base` (same layer `Rational` itself already lives in).
2. **Per-entity coverage and a declared fill policy** (closes Section 2.7). Smallest: extend
   `RemapWeights` (backend/remap_plan.hpp) with a per-ROW covered-fraction vector alongside
   the existing scalar minimums, and a separate, explicitly-requested `FillSpec{std::vector
   <double> values}` a caller passes to the apply step for entries below a stated coverage
   threshold, validated for finiteness and component count exactly as `FillPolicy::validate`
   does. Layer: `backend` (same layer `RemapWeights` lives in) for the per-row vector; `model`
   for the `FillSpec` declaration, parallel to `TransferSemantics`.
3. **Transfer capability probing** (closes Section 2.3/row 5). Smallest: one free function per
   backend family, `TransferCapabilities probe(TransferKind, std::string_view scheme)`,
   returning FUSE's own `available`/`unavailable_reason` shape; a thin registry mapping scheme
   name to probe function, NOT a new polymorphic class hierarchy, since FUSE's remap backends
   are already free functions over `RemapWeights`. Layer: `backend`.
4. **Mass-matrix residual transfer-accuracy check** (closes the second half of Section 2.5,
   row 7). Smallest: a free function beside FUSE's existing PETSc-facing code (`fuse/petsc/`),
   computing the same backward-error-style bound (`||residual||_inf <= tolerance*scale`)
   KARMA's `transfer_residual_report` does, with the tolerance DERIVED per this project's own
   standing rule (not the flat `1e-11` KARMA used uncommented) -- this needs the owner's
   derivation, not a copied constant. Layer: `petsc` (gated, matching D-B's "PETSc optional").
5. **Field-entry identity and canonical encoding** (closes Section 2.1, rows 2-3). Smallest: a
   closed variant type in `fuse/model` (or `fuse/field`) naming a mesh-global identity vs a
   DOF-basis identity, with a canonical fixed-width encode/decode pair and an FNV-1a hash,
   mirroring `karma_field_entry_encoding.h`'s own layout but under FUSE's own naming and
   `Result`-based error discipline. This is the one addition that is NOT purely additive: it
   would need a DOF identity scheme for FUSE's own higher-order (`Discretization::
   ContinuousGLL`, order > 1) fields, which do not yet have one at all (MEASURED: no DOF-level
   identity of any kind exists in the current tree). Layer: `model` (identity is a semantic,
   independent of storage, exactly KARMA's own stated reason for separating
   `karma_field_identity.h` from `karma_field.h`, karma_field_identity.h:10-26).
6. **Staged, all-or-nothing field commit** (closes Section 2.2, the largest gap). Depends on
   (5): a transaction needs a stable per-entry key to stage against (`stage_by_key`) as well as
   a dense-index hot path (`stage`). Smallest: a `FieldTransaction`-shaped type over FUSE's own
   `FieldRef`/`FieldRegistry`, in `fuse/field`, implementing the three update modes and the
   canonical accumulation order (Section 3's open dependency on numtypes applies here: build
   it over `double` now, and decide the `dual<S,N>` extension when the numtypes track lands,
   not before). Layer: `field`.
7. **Component registry, selection, and a compliance harness** (closes Section 2.6, the second-
   largest gap). This is the one addition `physics-kit-disposition.md`'s own open question 6
   already asks the owner about for a DIFFERENT KARMA header (`karma_registry.h`); this track's
   finding is that `karma_reference_components.h`'s own semantics depend on exactly that same
   missing mechanism, which sharpens the open question rather than answering it. No smallest-
   addition proposal is offered here beyond pointing at the existing open question: building a
   registry is a design decision (ambiguity-refusal policy, priority semantics) the owner
   should make once, not twice.

## 5. What was NOT verified

- The FUSE test counts cited (test/model/semantics_test.cpp, connect_test.cpp,
  compatibility_test.cpp, runtime/field_ref_test.cpp, test/model/validate_test.cpp) were
  grepped for `TEST(...)` names and read for the specific assertions cited; `validate_test.cpp`
  (1060 lines) was not read past the sections grepped for `E-UNITS`/`E-CONSERV`/
  `E-MULTI-WRITER`/`E-SELF`/`E-SHAPE` context (model/validate.hpp:385-608) and the matching
  test names were not individually opened -- the citations above are to the PRODUCTION
  validate.hpp lines, confirmed present, not to a read of every corresponding test assertion.
- KARMA's `karma_registry.h` (ComponentRegistry/SelectionRequest/RegistryReason) was read only
  through its USE in `karma_reference_components.C`/`component_compliance_test.C`, not as a
  header in its own right (it is not one of the 8 headers this track covers).
- SOMA's `coupling_graph.hpp`, `descriptor.hpp`, `yaml_io.hpp` were not read; the SOMA columns
  above rest on `field_ref.hpp`, `registry.hpp`, `field_spec.hpp`, `region.hpp`, `component.hpp`,
  `channel.hpp`, `validate.hpp` (partial, lines 1-232), `base/units.hpp`, `base/ids.hpp`, and
  `test/unit/field_ref_test.cpp`, all read in full or to the cited line range.

---

## 6. Built (Stage B, 2026-10-28)

The owner's 2026-10-28 decision closed gap (1) (`karma_field_transaction.h`, Section 2.2) and
narrow gaps (a) and (b) (Section 2.4's double-to-Rational conversion, and half of Section 2.5's
PETSc residual diagnostic) as follows. Gaps (2) and (3) (Section 2.6's component registry/
compliance harness, Section 2.1's field-entry identity) are DEFERRED; their own KNOWN_ISSUES.md
entries (D95, D96) carry this page's smallest-addition text and name the consumer that would force
each one, per the owner's instruction, rather than repeating that text here.

### 6.1 FieldTransaction (gap 1, BUILT; round 2, 2026-10-29)

`include/fuse/field/transaction.hpp` + `src/field/transaction.cpp`: `FieldStorage` (an abstract
publish/read-into contract), `FieldRefStorage` (the one implementation, over the existing
`fuse::FieldRef`), and `FieldTransaction` itself, carrying KARMA's own invariant
(karma_field_transaction.h:11-23) -- no caller-visible value change until every staged value has
passed local validation, and in the collective form until every rank has agreed. `UpdateMode`
keeps KARMA's three modes, renaming `SparseReplace` to `Insert` per the owner's decision.

**The numtypes seam.** Every value in this file is `double`; the header's own file comment names
exactly where a future `template <class Value>` parameter would go (`FieldStorage`,
`FieldRefStorage`, `FieldTransaction`, all three) and records that the canonical accumulation
order (below) will NOT transfer unchanged to a `dual<S,N>` value, since it is defined on a
`double`'s own magnitude, sign and (as a tie-break) raw bit pattern, none of which has a single
meaning for a multi-component value type.

**Layering: DECIDED, round 2 (the lead's own instruction, 2026-10-29).** `field` sits below
`transport`, the layer this project's own `docs/dev/layering.md` names as where MPI "enters", and
`tools/check-layering.py`'s `ALLOWED_PUBLIC_THIRD_PARTY["field"]` permits only `Kokkos_Core.hpp` in
a public `field` header -- `mpi.h` is forbidden there at every layer. `commit_collective()`
therefore takes `std::optional<int>`, the same Fortran-handle convention `fuse::RunnerConfig::
base_communicator` (include/fuse/runtime/runner.hpp) already uses, never a raw `MPI_Comm`; the
`MPI_Allreduce` itself lives in `src/field/transaction.cpp` behind `#if FUSE_HAVE_MPI`, in the same
`MPI_2INT`/`MPI_MAXLOC` shape `reconcile_plugin_agreement()` (src/abi/participant.cpp) already uses
for a different collective agreement. Round 1 flagged keeping MPI in `field`'s own `.cpp` (never
its public header) as an open question rather than deciding it silently; the lead's round-2
instruction decided it, NOT restructured: `docs/dev/layering.md`'s new "MPI outside `transport`"
section (2026-10-29) names both `field/transaction.cpp` and `abi/participant.cpp` as the two
translation units this applies to, and why -- a collective agreement belongs to the layer whose own
invariant it protects, not to the layer that merely routes values between ranks.

**The canonical accumulation order: changed, round 2 (the owner's standing rule -- accuracy over
efficiency).** Round 1 summed a slot's contributions in a VALUE-determined but accuracy-indifferent
order (ascending raw IEEE-754 bit pattern). Round 2 replaces it with KARMA's own magnitude-first
scheme (karma_field_transaction.C:48-54), precisely stated in `src/field/transaction.cpp`'s own
comment and summarised here: ascending `|value|` first, then (equal magnitude) sign with the
NEGATIVE value first, then (an exact duplicate) the raw bit pattern as a final, formally-complete
tie-break. This bounds the accumulated rounding by the classical recursive-summation error bound
(Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Theorem 4.2) more tightly than
an accuracy-indifferent order, by keeping each running partial sum close to the magnitude of the
terms still to be added rather than letting one disproportionately large term swamp many smaller
ones one at a time -- `FieldTransaction.MagnitudeFirstOrderBeatsANaiveArrivalOrderSumOnACaseWhereIt
Matters` (test/field/transaction_test.cpp) derives and asserts one concrete case where this changes
the EXACT published bits (worked below). This IS still independent of staging order (the order is a
function of WHICH VALUES were staged to a slot, never of WHEN) -- the arrival-dependence mutation
stays killable, since swapping the comparator's keys for raw push order breaks the sort just as
surely as it broke round 1's simpler one. It is NOT a correctly-rounded sum (still a plain
left-to-right fold, no Kahan compensation, no pairwise tree) and NOT pairwise/tree summation.

**The worked case (derivation).** The field starts at `0.0` (Accumulate mode allows starting
fresh); three contributions are STAGED, in this order: `huge = 2^60` (`ulp(huge) = 2^8 = 256`,
`half-ulp = 128`), then `100.0`, then `100.0` again -- deliberately the order a naive,
staging-order sum would fold them in, and deliberately not the canonical order
`build_published_values()` actually uses (it always starts its running sum from the field's
EXISTING value, `0.0` here, then folds the STAGED contributions in ascending-magnitude order,
`huge` last since it is the largest of the three, regardless of the order they were staged in). A
single `100.0 < half-ulp`, so adding it alone to a `huge`-sized running sum rounds back to that
running sum exactly, losing it: a NAIVE staging-order sum (`0 + huge -> huge`, `+100 -> huge`
rounded away, `+100 -> huge` rounded away again) loses the ENTIRE `200.0` contribution, publishing
exactly `huge`. The ACTUAL canonical (magnitude-first) order sums the two smaller,
equal-magnitude contributions with EACH OTHER first, while the running sum is still `0.0` (`0 +
100 + 100 = 200` exactly, far below double precision's own limit at this magnitude), and only THEN
adds the large one: `0 + 200 + huge`; `200 > half-ulp`, so this rounds to the nearest representable
double, `huge + 200` evaluated directly -- which IS the correctly-rounded value of the true
mathematical sum (MEASURED, Python, IEEE-754 binary64, the same arithmetic the C++ test exercises:
`huge + 200.0` rounds to the double exactly one bit-pattern step above `huge`, landing on `huge`'s
own next representable neighbour at this magnitude's `ulp`, not back on `huge` itself). Naive
staging-order sum gives exactly `huge`; magnitude-first order gives exactly `huge + 200.0`'s
correctly-rounded value, one full representable step above `huge` -- the two differ, and the test
asserts both exact values, not merely that they differ.

**Collective agreement: fixed, round 2 (the lead's own instruction).** Round 1's
`commit_collective()` let the ORIGINATING rank -- the one whose own local validation or publication
is what failed -- return its own raw local `Result`, while every OTHER rank got a uniform,
rewritten `E-FIELD-TXN-AGREEMENT-REFUSED` naming that rank; a caller comparing every rank's return
value therefore saw a MIX of one detailed local diagnostic and N-1 identical rewrites, which is
also a `collective path ... return[ing] on a locally computed condition`, the standing rule this
function's own `MPI_Allreduce` comment already invokes and the pre-fix code violated on exactly the
one rank that condition belonged to. Round 2 drops the special case: every rank, failing one
included, returns the IDENTICAL `Failure{"E-FIELD-TXN-AGREEMENT-REFUSED", ...}` naming the failing
rank, for both the validation-agreement and the publication-agreement phase. This is a DIFFERENT
convention from `reconcile_plugin_agreement()`'s own (which keeps the originating rank's raw
message, KNOWN_ISSUES.md D85) -- a deliberate difference for THIS collective, not an inconsistency
to reconcile with the ABI layer's.

**Superseded 2026-10-30 (owner's rule, KNOWN_ISSUES.md D97):** the uniform status above dropped the
failing rank's own code from what the other ranks saw. Both phases of `commit_collective()` now run
through `fuse::agree_outcome()` (`include/fuse/field/collective_agreement.hpp`): after the verdict is
reduced, the originating rank broadcasts its own `Failure`, and every rank returns the identical
`Failure` with `code = E-FIELD-TXN-AGREEMENT-REFUSED`, `origin_rank`, `origin_code` (the origin's own
code, e.g. `E-FIELD-TXN-INCOMPLETE`) and a message that names the rank, the number of refusing ranks
and the origin's own message verbatim. The ABI's `reconcile_plugin_agreement()` and
`reconcile_collective_status()` run through the same helper, so there is one convention, not two:
ranks that refused keep their own record; every other rank gets the origin's.

**Tested red-then-green and mutation-killed.** `test/field/transaction_test.cpp` (16 cases, 14 from
round 1 plus 2 new this round) mirrors `field_transaction_test.C`'s own cases one-for-one where one
exists (cited per case in the test file); `test/parallel/field_transaction_mpi_test.cpp` (new this
round, 3 cases, RANKS 2 3) is the real-communicator twin of the `CommitCollectiveWith*` cases,
closing the one hole round 1 left: `build-evidence/mutations/field-transaction.toml`'s
`field-transaction-collective-agreement-skipped` row is KILLED by this file's own
`ValidationDisagreementAgreesTheSameRefusalOnEveryRankAndPublishesNowhere` case (every non-failing
rank would publish under the mutation, which that case's own "unchanged on every rank" assertion
catches) -- `tools/mutate.py --area field-transaction` still reports it NOT-ASSESSED in THIS
sandbox (no working multi-rank MPI launcher here), per `FUSE_MUTATE_MPI=1` on a host that has one;
see `host_pending`. `tools/mutate.py --area field-transaction`, this sandbox: 3 of 4 KILLED (one
row's own anchor text updated this round for the new comparator function name), 1 NOT-ASSESSED --
see `measured_results` for the exact run.

### 6.2 `rational_from_double` (narrow gap (a), BUILT)

`include/fuse/base/rational.hpp` gains `Result<Rational> rational_from_double(double x,
std::int64_t max_denominator)`: exact whenever `x`'s reduced denominator fits `max_denominator`
(every finite `double` IS a dyadic rational, so exactness itself is never the question -- see the
function's own doc comment), refused by name otherwise. Three refusal codes, matching the task's
own naming: `E-RATIONAL-NONFINITE` (NaN/infinity), `E-RATIONAL-INEXACT` (the exact value's
denominator exceeds the caller's bound -- KARMA's own `NotDyadic`, renamed because every finite
double genuinely IS dyadic; the bound, not dyadicity, is what fails), `E-RATIONAL-OVERFLOW` (the
exact numerator/denominator does not fit `int64_t` even after reduction). No best-approximation
mode, matching KARMA's own `rational_from_dyadic_bounded` discipline (karma_semantics.h:90-99) and
this project's own standing rule against silently admitting an inexact time. Reuses
`Rational::from_i128` for the reduction and the int64 fit check rather than re-deriving either.
Tested in `test/base/rational_test.cpp` (8 new cases: round-trip exactness on `2^-k` for k in
[0,20], negative/integer values, each of the three refusal codes by name, the no-best-
approximation claim, and the `max_denominator <= 0` precondition throw).

### 6.3 Transfer-accuracy residual bound (narrow gap (b)): DERIVATION, interface signature only, NO implementation

The owner's instruction for this round: derive the bound, name every quantity, propose the
interface, and put the derived bound to the owner in `open_questions` -- do not implement it.

**What KARMA checks (karma_transfer_diagnostics.h:36-42, .C:42-75).** Given a mass-matrix solve
`M u = b` for a field transfer (a consistent/variational projection: `M` the target mesh's own
mass matrix, `b` assembled from the source field against the target's test functions, `u` the
projected field), recompute `r = M u - b` and test

```
||r||_inf <= tolerance * max(||b||_inf, ||M||_inf ||u||_inf, eps)
```

The scale `max(||b||_inf, ||M||_inf ||u||_inf, eps)` is the Rigal-Gaches/Oettli-Prager NORMWISE
BACKWARD ERROR denominator (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
Theorem 7.1's componentwise/normwise backward error for a linear system): the smallest normwise
relative perturbation of `(M, b)` for which `u` is an EXACT solution is `eta(u) = ||r|| /
(||M|| ||u|| + ||b||)`, and `max(a, b) <= a + b <= 2 max(a, b)` relates KARMA's own `max`-based
scale to this quantity within a factor of 2. KARMA's own test (`cg_transfer_accuracy_test.C:290`)
calls this with a flat, uncommented `tolerance = 1.0e-11` -- already flagged, independently of this
track, in `physics-kit-disposition.md:244` as "stated as a flat relative bound, no derivation
shown". This section supplies the derivation that row names as missing.

**The derivation.** Let:
- `n` -- the number of degrees of freedom in the target field (the mass matrix's own dimension).
  MEASURED at run time from the target mesh/solve, not a compile-time constant.
- `rtol_solver` -- the Krylov solver's own configured relative-residual stopping tolerance for the
  mass-matrix solve (e.g. PETSc's `KSPSetTolerances` `rtol`). KNOWN at solve-setup time (a
  configuration input the caller chose), not measured after the fact.
- `||r||_2`, `||b||_2` -- the solve's own stopping norms, in whatever norm the solver's own
  `KSPNormType` tests (PETSc's default for an unpreconditioned or left-preconditioned KSP is the
  **2-norm of the (preconditioned) residual**; a solver configured with a different `KSPNormType`
  tests a different quantity, and this derivation's bridging step below would need re-deriving
  against THAT norm instead -- a fact about the solver's own configuration to confirm before this
  bound is trusted, not assumed here).
- `||r||_inf`, `||b||_inf`, `||M||_inf`, `||u||_inf` -- the same infinity-norm quantities KARMA's
  own `transfer_residual_report` already computes via `VecNorm(..., NORM_INFINITY, ...)` and
  `MatNorm(..., NORM_INFINITY, ...)`. MEASURED at run time, exactly as that function already does.
- `eps` -- `std::numeric_limits<PetscReal>::epsilon()`, the same zero-problem floor KARMA's own
  code already uses. A constant, not measured.

Step 1. The solver's own stopping criterion gives `||r||_2 <= rtol_solver * ||b||_2` directly (the
usual relative-residual convergence test, `||r_k||/||r_0|| <= rtol` with `r_0 = b` for a zero
initial guess).

Step 2. Norm equivalence on an `n`-vector: `||r||_inf <= ||r||_2` and `||b||_2 <= sqrt(n)
||b||_inf`. Chaining: `||r||_inf <= ||r||_2 <= rtol_solver * ||b||_2 <= rtol_solver * sqrt(n) *
||b||_inf`.

Step 3. Since KARMA's own denominator is a MAX that already includes `||b||_inf` as one of its
terms, `max(||b||_inf, ||M||_inf ||u||_inf, eps) >= ||b||_inf`, so

```
rel = ||r||_inf / max(||b||_inf, ||M||_inf ||u||_inf, eps)
    <= ||r||_inf / ||b||_inf
    <= rtol_solver * sqrt(n)
```

**The derived bound**: `tolerance = rtol_solver * sqrt(n)`, not a flat constant independent of
problem size. `sqrt(n)` is the one "conditioning factor" this derivation needs -- and it is a
norm-equivalence factor bridging the solver's own 2-norm stopping test to KARMA's infinity-norm
backward-error check, not a matrix condition number (`kappa(M)`): the matrix's own conditioning
does NOT enter this particular bound, because the denominator's `||b||_inf` term alone already
dominates `||r||_inf` once Step 1/2 are applied, before `||M||_inf ||u||_inf` or `kappa(M)` are
ever needed. (A matrix-conditioning-dependent bound WOULD be needed for a tighter statement about
`||u||_inf` alone, e.g. `||u||_inf <= ||M^{-1}||_inf ||b||_inf`, but nothing in KARMA's own check,
or in this derivation's conclusion, requires bounding `||u||_inf` that way.)

**Proposed interface (signature only; not implemented this round):**

```cpp
// fuse/petsc, gated like every other PETSc-facing signature in that layer
// (fuse/petsc/token.hpp's own posture: no PETSc type escapes past src/petsc/*.cpp).
struct TransferResidualReport {
  double residual_norm = 0.0;   // ||r||_inf, measured
  double scale = 0.0;           // max(||b||_inf, ||M||_inf ||u||_inf, eps), measured
  double tolerance = 0.0;       // rtol_solver * sqrt(n), DERIVED per this section, not a literal
  bool within_tolerance = false;
};
[[nodiscard]] Result<TransferResidualReport> transfer_residual_report(
    /* mass matrix, solution, rhs, scratch -- fuse::linalg types, not PETSc's own,
       matching fuse/petsc's existing posture */
    const DistSparseMatrix& mass, const DistVector& solution, const DistVector& rhs,
    DistVector& residual_scratch, double solver_rtol);
```

`solver_rtol` is a REQUIRED parameter (not defaulted to a literal): the derivation above shows the
tolerance is a function of the solve's own configured `rtol` and the problem's own `n`, both
available to the caller that set up the solve, so there is no honest default to supply here.

Put to the owner in `open_questions`: does `rtol_solver * sqrt(n)` read as the right bound, or
should the `sqrt(n)` norm-equivalence factor instead be derived against the ACTUAL `KSPNormType`
the physics-kit tier-0-with-PETSc-optional solves will use (D-B), once one is chosen, rather than
assumed here against PETSc's documented default?

### 6.4 Constrain's finite floor (decision (c)): documented, not changed

The owner's decision: FUSE's existing `Constrain` floor (`scale = |before| if nonzero else 1.0`,
`src/runtime/runner.cpp`) STAYS as it is. Recorded as a deliberate difference from KARMA's own
`relative_integral_error` (which returns `+infinity` for a zero reference and a nonzero value,
karma_transfer_diagnostics.C:12-18) in `KNOWN_ISSUES.md` D97 (see that entry; this page's own
Section 2.5 already named the two as DIFFERENT, not superset, before this decision was taken).
Fixed in round 2 (the reviewer's own finding, 2026-10-29): this section previously cited "the S8
record" instead of naming D97 directly.

---

## Lead's review (2026-10-28)

Spot-checked against the tree: `fuse::Rational` has `to_double()` and no constructor or factory
from a double (MEASURED, `include/fuse/base/rational.hpp`); the field layer and the runtime component
interface carry no staging, commit or publish vocabulary outside the restart and rollback paths
(MEASURED, grep over `include/fuse/field` and `include/fuse/runtime`); `src/abi/registry.cpp` refuses
null, duplicate and not-found mounts but has no selection among several implementations. The
track's declared depth: `karma_registry.h` read only through its users; SOMA's `coupling_graph.hpp`,
`descriptor.hpp` and `yaml_io.hpp` not read; `test/model/validate_test.cpp` not read in full. The
eight verdicts stand as the track delivered them; each GAP is an owner decision recorded under the
S8 entry in `KNOWN_ISSUES.md` once taken. Nothing is retired on the strength of this page.

