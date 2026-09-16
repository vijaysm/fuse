# Roadmap

FUSE is built in ten stages. Each has a falsifiable exit condition, and the order is a dependency
order, not a preference: the canonical descriptor is frozen before anything binary depends on it, and
imported code lands byte-identical before it is transformed ([](invariants.md) `I8`).

| Stage | What it delivers | Exit |
|---|---|---|
| **S0** *(in progress)* | Repository, conventions, CI gates, both parents imported untouched | Both upstream suites green inside FUSE with zero source edits. **Coupler tree: 770/770.** Spine tree: builds clean, suite pending on an MPI host |
| **S1** | One vocabulary: exact time, units, versions, diagnostics, canonical serializer | **in progress** — `fuse::base` ported and transformed (9 targets, 78 cases, Debug and Release green, `FUSE_ASSERT` verified under `NDEBUG`); mutation gate run; ASan/UBSan leg and the canonical serializer remain |
| **S2** | One semantic model; the **canonical descriptor frozen**; multiscale and emulator contracts declared and refusing | **done** (2026-09-15) — semantic model, N-ary graph, graded compatibility with transform inference, multiscale and surrogate stubs refusing under test, YAML front-end, descriptor frozen as `waist-v1`, three-leg round-trip and cross-rank byte agreement both verified |
| **S3** | One compiler: operator IR, the checker's conversions compiled, rank agreement as instructions, acyclicity proof | **done** (2026-09-15) — `fuse::ir` and `fuse::schedule` ported with the golden dumps carried over; the connection check's conversions now compiled (exact rational `Scale`, and any authorized operation the compiler cannot emit is refused); several writers per field combined by a declared rule; an `Epoch` after every data move with the compiler proving its participants match the move's; `Map`/`Reduce`/`Closure` reserved and refused; 50,000 generated graphs agree with an independently computed loop answer |
| **S4** | One runtime: both entry modes, transactions on plan buffers, portable kernels, waveforms and acceleration | **in progress** — runtime, numerics, transport and the serial mesh are ported and green (52 targets, 723 cases); the interpreter executes the new operators; a rank that fails mid-step reconciles at every remaining agreement point; the dense least-squares solve is behind a facade with two backends; the measure-weighted residual norm is in. Field transactions, the lazy field pipeline, the device leg and the manufactured-solution runs remain |
| **S5** | Backends under one operator interface: mesh, remap, transport, persisted maps | First non-matching-mesh parallel coupled run with conservation and order gates |
| **S6** | Interfaces: both ABIs on one descriptor, adapter host, Fortran, Python, YAML, external processes | An unmodified external executable couples through the participant ABI. **`upstream/soma` deleted** — every path superseded |
| **S7** | Implicit and monolithic solvers on PETSc | A Newton-coupled problem converges at measured quadratic rate with adjoints verified |
| **S8** | `physics-kit`: the optional finite-element component | Every upstream physics baseline passes through FUSE. **`upstream/karma` deleted** — every path superseded |
| **S9** | Portability and scale: budgets, cluster rehearsal, adversarial suite | Budgets exist; scaling numbers published with their conditions. **`upstream/` does not exist**; `UPSTREAMS.lock` records both retirements |

**Critical path.** S0 → S1 → S2 → S3 → S4 → S5 → S6. S7 and S8 are independent once S5 exits. S9
begins at S4 and finishes last.

## What the end state is

One library. Everything under `include/fuse/` and `src/`, namespace `fuse::`, FUSE's names — no
`soma::`, no `KARMA::`, no `upstream/` directory. The parents survive as git history in their own
repositories and as two `retired` entries in `UPSTREAMS.lock`. The procedure and naming map are in
[](porting.md); invariant `I11` makes it enforceable from the first port.

## Where S0 stands

Both parents are tagged `fuse-import-0` and imported byte-identical (693 files, proven by
`tools/verify-import.py --parents`). The superbuild configures and runs the imported coupler suite:
770 tests, all passing, 4 correctly skipping for want of a mesh backend.

Two findings from the import, recorded in `UPSTREAMS.lock` rather than fixed, because I8 forbids
editing an import:

- The coupler tree fetches yaml-cpp 0.8.0, whose `CMakeLists.txt` predates CMake 3.5 and so cannot
  configure under CMake 4.x without a policy floor. The superbuild supplies one; an installed
  yaml-cpp avoids the fetch entirely.
- **The spine tree does not compile in Debug.** `karma_domain.C` uses two unqualified enumerators of
  a scoped `enum class`, inside an `assert`. All three of that project's verified presets define
  `NDEBUG`, so the line has never been compiled — and the assertion is a tautology, so it would check
  nothing if it did. This is invariant `I6` ([](invariants.md)) demonstrating itself on real code:
  when that area is ported, the check becomes a real invariant rather than an `assert`.

The first externally meaningful milestone is **S4**: a complete coupler that reconciles distributed
failures, running on Kokkos. The first "couple an existing code" milestone is **S6**.

:::{note}
The task-level plan — step numbering, per-step tests, interfaces produced and consumed, the risk
register and the upstream sync protocol — is kept outside this repository, like the parent projects'
plans. This page is the durable summary; it is what belongs in documentation.
:::

## Deferred, with a design already written down

**Diagrams of a coupled setup.** `dump_graphviz` draws the lowered operator graph, which is the right
picture for debugging a lowering rule and the wrong one for understanding a setup: a four-component
coupling is already a hundred nodes, most of them derived bookkeeping, and none of it shows when
anything happens. Planned instead are three views generated from the graph and the compiled plan —
a component view (who exchanges what with whom, at what rate, which exchanges are lagged or
iterated), a time view (one coupling period, one lane per component, arrows from source step to
target step, rank-agreement points as vertical bars), and a step view (the operators of one step in
execution order, with the derived ones marked).

Built as library functions emitting text (`fuse::diagram::component_dot`, `::time_svg`,
`::step_dot`), then a `fuse-diagram` command over a YAML setup, and only then a single
self-contained HTML page if one is still wanted. Text first is the point: a diagram that is a string
is golden-comparable, so a change that drops an edge fails a test instead of producing a picture
nobody checks. Scheduled at S9 because it gates no correctness result, and because the time view
wants the runtime's final agreement-point placement.
