# Design invariants

These do not change while FUSE is built. Each one names how it is **enforced**, because a rule with
no enforcement is a preference.

`I1` **Exact rational time.**
: No floating-point time crosses an ABI, a descriptor or an instruction. *Enforced by*
  `tools/check-invariants.py` (no `double`/`float` in a time position in any ABI header) and by the
  time round-trip tests.

`I2` **Every distributed failure is reconciled.**
: A rank that fails agrees the failure with its peers before throwing or entering a later collective.
  *Enforced by* the `Epoch` instruction being the only path to a collective error, a static check that
  reconciliation has exactly one implementation, and parallel tests that assert an error rather than
  a hang.

`I3` **Every arc carries a time level.**
: A missing time level, unit or coordinate frame is a refusal, never a default. *Enforced by*
  validation tests that assert the refusal fires.

`I4` **No silent fallback.**
: No substitution, downgrade or partial publication. A capability that cannot be verified is
  unavailable and says why. *Enforced by* a refusal test per capability and per error code — a
  refusal that no test makes fire does not count as implemented.

`I5` **No allocation after setup; no per-value virtual dispatch in a hot kernel.**
: *Enforced by* an allocation-count fence around the step loop and around lazy field chains.

`I6` **Invariants throw or return a status; `assert` is never an invariant.**
: Release and sanitizer legs are mandatory, because three defects in the parent code bases escaped
  Debug-only suites. The no-`assert` rule itself is *enforced by* `tools/check-invariants.py`. Of
  the two mandatory legs, only Release exists today (`tier0`'s Debug/Release matrix,
  `.github/workflows/ci.yml`); the ASan+UBSan leg this rule also demands has no CI leg yet —
  `testing.md`'s CI-legs list carries it under Planned, for S1's exit.

`I7` **Third-party types stay out of public headers.**
: No Eigen, KokkosKernels, MOAB, PETSc, MPI or HDF5 type in a public header outside its own backend
  directory. *Enforced by* `tools/check-layering.py`.

`I8` **Import first, transform second.**
: Upstream code lands byte-identical with its suite green before it is changed, so a port can be
  checked against its source. *Enforced by* `tools/verify-import.py` and the per-import evidence
  record.

`I9` **A port is accepted only with killed mutants.**
: At least one named mutation per public kernel, killed by a test, logged under `build-evidence/`.
  Green tests that cannot detect the defect they were written for are decoration. *Enforced by*
  `tools/mutate.py` run against a configured build before a port is accepted, with the resulting log
  committed under `build-evidence/`. The one piece of this that is a CI gate today is staleness, not
  the kill itself: `tools/mutate.py --check-anchors` (static, no build) runs in the `gates` job and
  fails when a logged mutation's anchor text no longer occurs exactly once in its file — the gap that
  let three KILLED rows outlive the rename they were written against. Running the mutations and
  confirming each is KILLED still has no CI leg (see `testing.md`'s CI-legs list).

`I10` **A fact about an upstream repository is a dated observation.**
: Cite `path:line` at the SHA in `UPSTREAMS.lock`; re-verify before depending on it. *Enforced by*
  review — this rule exists because several plan-gating facts went stale within hours.

`I11` **`upstream/` has an end, and `fuse::` never depends on it.**
: All library code lives under `include/fuse/` and `src/` in namespace `fuse::` under FUSE's names;
  the parents' namespaces, type prefixes and file prefixes do not survive a port. No `fuse::` code
  includes from `upstream/`. Each parent subtree is deleted whole once its last path is superseded
  ([](porting.md)). *Enforced by* `tools/check-layering.py` (no upstream include, no legacy namespace
  or prefix under `include/fuse/` or `src/`) and by the retirement entries in `UPSTREAMS.lock`.

## API design rules

These govern the fluent interface, and exist because an ergonomic API is the easiest way to break
`I4` and `I5`. The user-facing half is documented in [](../user/fluent-api.md).

`A1`
: Authoring chains and lazy field-pipeline chains are distinct categories. No type exposes methods
  from both. *Enforced by* a static gate.

`A2`
: Authoring calls return the builder, never a status. Diagnostics accumulate; one terminal call
  reports every failure in a deterministic order. Status objects live only at `validate`, `run`,
  `initialize` and the ABI entry points.

`A3`
: Non-terminal pipeline operations are `[[nodiscard]]`, view-backed, and do no work. *Enforced by* an
  allocation-count fence: a four-stage chain allocates zero.

`A4`
: Terminals that do IO or reduction take the context and are collective in their signature. A
  rank-subset invocation is a refusal — reconciled through `Epoch` so that the *refusal itself* cannot
  hang. *Enforced by* a parallel test under a timeout, asserting refusal rather than deadlock.

`A5`
: Slice spelling follows `mdspan` (`fuse::all`, `{lo, fuse::end}`) rather than inventing a syntax.

`A6`
: Ref-qualified overloads: an rvalue chain yields an owning value, an lvalue chain a view. *Enforced
  by* a sanitizer test and a compile-fail test.

## Units carry their affine part

A `fuse::Units` is seven integer dimension exponents **and** an exact `scale` and `offset`
(`Rational`, not `double`, so a conversion applied every coupling step cannot drift and the
descriptor's bytes are reproducible). Three rules follow, each with a test:

- **Dimensional checks ignore the affine part** — `same_dimensions()` is what "the same physical
  quantity" means. Full `operator==` includes scale and offset, so kilopascal is not pascal.
- **Multiplying or dividing an affine unit is refused** (`E-BASE-UNITS-AFFINE`). Degrees Celsius
  squared is not a quantity, and dropping the offset would be a silent error no dimensional check
  can see.
- **A scale-only mismatch on a channel is its own verdict**, not an `E-UNITS` incompatibility and
  not silence. Transform inference (S2.4) has landed: under `ConnectPolicy::allow_automatic_unit_scale`
  (the default), an exact, zero-offset scale is inferred as a `UnitScale` operator and *reported*,
  not silent — `validate()` emits `I-UNIT-SCALE-INSERTED` naming the factor, because the user should
  see an operator they did not write. `E-UNITS-SCALE` is the verdict only when insertion is not
  permitted (the policy forbids it) or not possible (an offset change, which moves the zero of the
  scale and is never inserted on the caller's behalf) — applying the wrong factor, or silently
  moving a zero, every step is worse than a reported insertion or a refusal.
