# Porting an area from `upstream/`

`upstream/` is scaffolding with a scheduled end. **Every line of library code ends up under
`include/fuse/` and `src/`, in namespace `fuse::`, under FUSE's names**, and each parent's subtree is
deleted — whole, in one commit — the moment its last path is superseded. No `fuse::` code ever
includes a header from `upstream/`; the gate `tools/check-layering.py` fails the build if one does.

The parents' namespaces (`soma::`, `KARMA::`), their type prefixes (`KARMA*`), their file prefixes
(`karma_*`, `soma_*`) and their source extension (`.C`) do not survive the port. There is one
library, and it is FUSE.

## The procedure, per area

1. **Copy** the area's headers, sources and tests from `upstream/<parent>/` into their layer
   directory under `include/fuse/<layer>/`, `src/<layer>/` and `test/<layer>/`.
2. **Rename** per the map below. Nothing else changes in this step. A reviewer must be able to diff
   the renamed copy against the upstream original and see *only* renames.
3. **Port the tests first.** They must pass against the renamed copy before any behavioural change.
   These are the oracle for everything that follows.
4. **Transform** — merge with the other parent's version of the concept, adopt FUSE contracts
   (`Result`/`Diagnostics`, exact time, capability bits), replace `assert` with real invariants
   (`I6`), remove any silent fallback (`I4`). Each transformation is its own commit with its own
   test.
5. **Mutation evidence.** At least one named mutation per public kernel, killed by a ported or new
   test, recorded under `build-evidence/` (`I9`). Zero killed mutants means the port is not accepted.
6. **Record supersession.** Add the upstream paths to `superseded` in `UPSTREAMS.lock`. From that
   commit on, a backport touching those paths is classified `superseded` by
   `tools/sync-upstream.py` and ignored.
7. **Retire the subtree** when the parent's `superseded` list covers every path it ships:
   `git rm -r upstream/<parent>`, remove its `add_subdirectory`, set `retired = "<date> @ <fuse sha>"`
   in the lock. The parent's tests must by then all have FUSE equivalents; the retirement commit
   lists the count on both sides.

Until step 7 the upstream subtree keeps building and its suite keeps running in CI. That is not
inertia: the original is the **differential oracle** for the port, and removing it early would
throw away the cheapest test we have.

## Naming map

Mechanical rules, applied in step 2. Judgement calls — when the two parents each have a type for the
same concept — are made in the plan's stage that merges them, and recorded here when made.

| Upstream | FUSE | Note |
|---|---|---|
| `namespace soma` | `namespace fuse` | |
| `namespace KARMA` | `namespace fuse` | `fuse::fem` for the finite-element component (`physics-kit`) |
| `KARMAField`, `KARMAComponent`, … | `fuse::Field`, `fuse::Component`, … | strip the prefix; the namespace carries it |
| `karma_<name>.h` / `.C` | `<name>.hpp` / `.cpp` | 150 `.C` sources; the layout follows `include/fuse/<layer>/` |
| `soma/<area>/<name>.hpp` | `fuse/<layer>/<name>.hpp` | areas map to layers per [](layering.md) |
| `SOMA_*`, `KARMA_*` macros and CMake options | `FUSE_*` | |
| `soma_c.h` flat C API | the `fuse_coupler_abi()` table | not renamed: **replaced**, see [](abi-design.md) |
| `karma_*_v1` exported symbols | `fuse_plugin_abi()` table | same |
| Fortran `module soma_mod` | `module fuse_coupler` | *(proposed — see open decisions)* |
| Python `_soma` / `soma` | `fuse_coupler` | *(proposed — `fuse` is taken on PyPI by the filesystem bindings)* |
| golden-IR header `soma-ir v1` | `fuse-ir v1` | the only canonical-byte change the rename causes; the golden files are regenerated once, and the diff must be that token and nothing else |

**What the rename does not touch.** KARMA's field-identity encoding contains no `KARMA` string, so
identity hashes are unchanged by the rename — verified at import by grepping every string literal in
`include/coupling`, `src/coupling`, `include/abi` and `src/abi`. Any port that changes an identity
hash has changed semantics, not a name, and must say so.

## Merging two versions of one concept

At import the two parents share **no** simple type name once `KARMA` is stripped, so there are no
mechanical collisions. There are conceptual overlaps — both have a field handle, a component
contract, a status type, a coupling graph — and the plan decides each one:

| Concept | Base taken from | Absorbs from the other | Stage |
|---|---|---|---|
| `Result` / `Diagnostics` | coupler | the spine's full outcome vocabulary | S1 |
| exact time (`Rational`, `TimeLevel`) | coupler | — | S1 |
| identity, versioning, canonical bytes | spine | the coupler's descriptor fields | S2 |
| coupling graph, IR, schedule, plan | coupler | the spine's transform chain as a lowering rule; `CollectiveOutcome` as the `Epoch` instruction | S3 |
| field handle, transactions | spine (`FieldTransaction`) on the coupler's plan buffers | | S4 |
| component contract | coupler `Component` via the spine's registry | capability bits replace `supports_*()` | S4 |
| transport, remap, mesh | coupler | the spine's FEM transfer as a `LinearOperator` backend | S5 |

A row here is the *decision*; the port commit is the *evidence*.

## Open decisions

- **Fortran module and Python package names.** `fuse_coupler` is proposed for both because `fuse` as
  a Python module belongs to the filesystem bindings on PyPI, and a Fortran module named `fuse` would
  be an odd pairing with it. Not yet ruled.
