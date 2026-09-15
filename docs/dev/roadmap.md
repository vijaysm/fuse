# Roadmap

FUSE is built in ten stages. Each has a falsifiable exit condition, and the order is a dependency
order, not a preference: the canonical descriptor is frozen before anything binary depends on it, and
imported code lands byte-identical before it is transformed ([](invariants.md) `I8`).

| Stage | What it delivers | Exit |
|---|---|---|
| **S0** *(in progress)* | Repository, conventions, CI gates, both parents imported untouched | Both upstream suites green inside FUSE with zero source edits. **Coupler tree: 770/770.** Spine tree: builds clean, suite pending on an MPI host |
| **S1** | One vocabulary: exact time, units, versions, diagnostics, canonical serializer | **in progress** — `fuse::base` ported and transformed (9 targets, 78 cases, Debug and Release green, `FUSE_ASSERT` verified under `NDEBUG`); mutation gate run; ASan/UBSan leg and the canonical serializer remain |
| **S2** | One semantic model; the **canonical descriptor frozen**; multiscale and emulator contracts declared and refusing | **done** (2026-09-15) — semantic model, N-ary graph, graded compatibility with transform inference, multiscale and surrogate stubs refusing under test, YAML front-end, descriptor frozen as `waist-v1`, three-leg round-trip and cross-rank byte agreement both verified |
| **S3** | One compiler: operator IR, transform chains as lowering rules, epochs as instructions, acyclicity proof | **in progress** — the compiler core (`fuse::ir`, `fuse::schedule`) is ported and green with the golden IR dumps carried over; the transform-chain lowering rule, `OpCode::Epoch`, `MergeSpec` and the acyclicity fuzzer remain |
| **S4** | One runtime: both entry modes, transactions on plan buffers, Kokkos kernels, waveforms and acceleration | Manufactured-solution orders and iteration counts reproduce upstream exactly |
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
