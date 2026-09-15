# Roadmap

FUSE is built in ten stages. Each has a falsifiable exit condition, and the order is a dependency
order, not a preference: the canonical descriptor is frozen before anything binary depends on it, and
imported code lands byte-identical before it is transformed ([](invariants.md) `I8`).

| Stage | What it delivers | Exit |
|---|---|---|
| **S0** | Repository, charter, CI legs, both parents imported untouched | Both upstream suites green inside FUSE, on every CI leg, with zero source edits |
| **S1** | One vocabulary: exact time, units, versions, diagnostics, canonical serializer | Tier 0 builds with either linear-algebra backend; killed mutants per public type |
| **S2** | One semantic model; the **canonical descriptor frozen**; multiscale and emulator contracts declared and refusing | C++/YAML/bytes round-trip structurally equal; descriptor tagged |
| **S3** | One compiler: operator IR, transform chains as lowering rules, epochs as instructions, acyclicity proof | A reference case lowers to a golden-compared plan; fuzzer green |
| **S4** | One runtime: both entry modes, transactions on plan buffers, Kokkos kernels, waveforms and acceleration | Manufactured-solution orders and iteration counts reproduce upstream exactly |
| **S5** | Backends under one operator interface: mesh, remap, transport, persisted maps | First non-matching-mesh parallel coupled run with conservation and order gates |
| **S6** | Interfaces: both ABIs on one descriptor, adapter host, Fortran, Python, YAML, external processes | An unmodified external executable couples through the participant ABI |
| **S7** | Implicit and monolithic solvers on PETSc | A Newton-coupled problem converges at measured quadratic rate with adjoints verified |
| **S8** | `physics-kit`: the optional finite-element component | Every upstream physics baseline passes through FUSE |
| **S9** | Portability and scale: budgets, cluster rehearsal, adversarial suite | Budgets exist; scaling numbers published with their conditions |

**Critical path.** S0 → S1 → S2 → S3 → S4 → S5 → S6. S7 and S8 are independent once S5 exits. S9
begins at S4 and finishes last.

The first externally meaningful milestone is **S4**: a complete coupler that reconciles distributed
failures, running on Kokkos. The first "couple an existing code" milestone is **S6**.

:::{note}
The task-level plan — step numbering, per-step tests, interfaces produced and consumed, the risk
register and the upstream sync protocol — is kept outside this repository, like the parent projects'
plans. This page is the durable summary; it is what belongs in documentation.
:::
