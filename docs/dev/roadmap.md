# Roadmap

FUSE is built in ten stages. Each has a falsifiable exit condition, and the order is a dependency
order, not a preference: the canonical descriptor is frozen before anything binary depends on it, and
imported code lands byte-identical before it is transformed ([](invariants.md) `I8`).

Recounted from the tree at `6a4f0c5` (560 commits), not carried forward from an earlier pass: S0–S6
are complete, S7's native path is complete and measured (seam, derivative gate, Newton-Krylov
coupler on dn_heat and the Brusselator) with the PETSc plugin deferred as D32, S8 has not started, and S9 has not started (no execution-space type beyond
`Kokkos::{OpenMP,Serial}` appears anywhere in `include/` or `src/` — grepped, zero hits).

| Stage | What it delivers | State and exit evidence |
|---|---|---|
| **S0** | Repository, conventions, CI gates, both parents imported untouched | **done**, except the CI legs needing MPI. A remote exists and the five-leg workflow has run (`KNOWN_ISSUES.md` A1, 2026-09-27); none of the five legs runs MPI, so every `parallel.*`/`layout.*`/MOAB test is still a host-only outcome, verified by hand on the owner's machine rather than by CI |
| **S1** | One vocabulary: exact time, units, versions, diagnostics, canonical serializer | **done** (2026-09-15) — `fuse::base` ported and transformed, mutation gate run (13/13 killed), `FUSE_ASSERT` verified checked under `NDEBUG` |
| **S2** | One semantic model; the **canonical descriptor frozen**; multiscale and emulator contracts declared and refusing | **done** (2026-09-15) — semantic model, N-ary graph, graded compatibility with transform inference, multiscale and surrogate stubs refusing under test, YAML front-end, descriptor frozen (now `kGraphVersion = 11` — append-only growth through the launch-mode, provider-direction and LowFive work, `include/fuse/model/descriptor.hpp`'s own version-history comments; [](abi-design.md)'s struct-growth rules are the same discipline applied one level down to the C ABI's structs), cross-rank byte agreement host-verified at 2/3/4/5 ranks |
| **S3** | One compiler: operator IR, the checker's conversions compiled, rank agreement as instructions, acyclicity proof | **done** (2026-09-15) — `fuse::ir`/`fuse::schedule` ported with golden dumps carried over; `Epoch` after every data move ([](rank-agreement.md)); 50,000 generated graphs agree with an independently computed loop answer |
| **S4** | One runtime: both entry modes, transactions on plan buffers, portable kernels, waveforms and acceleration | **done** (2026-09-17) — runtime, numerics, transport and the serial mesh ported; a rank that fails mid-step reconciles at every remaining agreement point ([](rank-agreement.md)); manufactured-solution and iteration verification matching the parent (order 1.005); the lazy field pipeline in and tested |
| **S5** | Backends under one operator interface: mesh, remap, transport, persisted maps | **done** (parallel matrix gate landed `0d920c9`, 2026-09-21; MOAB wired 2026-09-18, host-verified 12/12 rows 2026-10-08, [](testing.md)) — `LinearOperator`, the MOAB and native mesh backends, every transport, and the conservative supermesh/native remap halos ([](native-halo.md)) all pass their tier-0 and (where applicable) host-run rows |
| **S6** | Interfaces: both ABIs on one descriptor, adapter host, Fortran, YAML, external processes | **functionally done** (conformance case closed `839ffa5`, 2026-10-07) — participant and provider ABIs, the registry/adapter host, Fortran (`ffuse`/`ffuse_provider`), YAML, and an unmodified external executable coupling through the participant ABI, demonstrated by the Brusselator across three and five separately-launched ranks ([](conformance.md), `KNOWN_ISSUES.md` D19/D21/D22/D24/D25). **Not met:** the stage's own retirement exit (`upstream/soma` deleted) — 181 files still live under `upstream/soma`, `UPSTREAMS.lock`'s `retired` field for it is still empty — and the **Python** binding named in this stage's original scope was never built (only a cross-check reference model in Python exists, for validating the C++ window driver, `KNOWN_ISSUES.md` A3's naming proposal is the only trace of a real `pfuse` package) |
| **S7** | Implicit and monolithic solvers | **native path complete 2026-10-11** — the residual/linearisation seam (`fuse::ImplicitComponent`/`fuse::LinearizationToken`, [](newton-krylov.md) §1–3), the derivative gate ([](derivative-gate.md): adjoint identity over eleven operator cases, Taylor-remainder order with the round-off plateau detected, assembled-vs-matrix-free agreement), and `fuse::num::solve()` — Jacobian-free Newton-Krylov with the Walker–Pernice step, right-preconditioned `gmres`, Eisenstat–Walker forcing, cubic-backtracking/critical-point/damped line searches, named convergence codes and a lagged block-Jacobi preconditioner over the members' tokens — selectable as `AccelKind::Newton` on an ordinary or a coupling-window region. Measured: dn_heat order 2.0 ± 0.24 on a temperature-dependent conductivity ([](../user/dn-heat.md)); the Brusselator's window converges in 50/50 windows in both waveform forms where Aitken fails at the third, final means agreeing with Picard to 2e-13 ([](../user/brusselator.md)). Built on the S7 linear algebra: `DistVector`/`DistSparseMatrix` ([](linalg-distributed.md)), `pcg`/`gmres` ([](linalg-krylov.md)), `SchurSlabSolver` ([](schur-slab-solver.md)). **Open:** multi-rank window Newton (D31, reducer seam in place), the PETSc plugin as the second substrate behind the same seam (D32), Newton's plain 2-norm convergence test vs the measure-weighted norms ([](../user/convergence.md)) |
| **S8** | `physics-kit`: the optional finite-element component | **not started** — no `physics-kit/` directory exists |
| **S9** | Portability and scale: budgets, cluster rehearsal, adversarial suite | **not started** — no execution space beyond `Kokkos::{OpenMP,Serial}` is instantiated anywhere (grepped `include/`, `src/`: zero `Kokkos::Cuda`/`HIP`/`SYCL`); no measurement harness, no cluster rehearsal, no device build |

**Critical path.** S0 → S1 → S2 → S3 → S4 → S5 → S6. S7 and S8 are independent once S5 exits. S9
begins at S4 and finishes last.

## Open D-items, grouped by the stage they bear on

`KNOWN_ISSUES.md`'s `D` section is the source; this groups its still-open entries (struck-through
titles are closed) by which stage above they qualify, so a reader does not have to read the whole
file to see what is left in each.

**S0 / build infrastructure** — D26 (`FUSE_LINALG_BACKEND=builtin` fails to configure since the
examples link Eigen directly; the option's own fate, real configuration or retired, is undecided); D30 (a Release build fails three bit-for-bit cases by one ulp through
floating-point contraction; whether to set `-ffp-contract=off` everywhere or restate the assertions
per build type is the owner's decision).

**S5 / backends** — D27 (the native backend's remap halo has a forward fill,
`remap_halo_pattern`/`make_mpi_exchanger`, but no `accumulate_remap_halo` — the reverse exchange a
future native conservation-audit tool would want; nothing in the tree needs it yet); D15 (the
provider direction is conformance-tested on one process; device fields, mixed checkpoint/restore, and
a plugin on a rank subset are not); D14 (LowFive's distributed transport and Demo B are compiled, not
run — needs LowFive rebuilt against `fmt >= 9`, the host's `/opt/lowfive` links `libfmt.8`).

**S6 / interfaces** — D23 (accept/connect cannot complete on this MPICH `ch3:nemesis` build when both
programs share a node; needs MPICH ch4 or Open MPI, not a library fix); D17 (`--accel aitken`
converges through `--driver runner` but not `--driver window` on the identical config — open, not
pursued, Conformance Unit 2 needs no accelerator); D18 (`connect()`'s `Connect`-mode path supports
exactly two components; a third refuses — open, not pursued, Unit 2 routes around it); D7
(`WindowCoupler` supports at most one straddling member per window and no `Fixed` step longer than
the window — rejected, support is design work); D8 (a `Remap` into a nodal field on more than one
rank: two real correctness defects found by a host MPI run are fixed, a further host run is the
final outcome); D3 (two runtime mutations survive on the host; deferred 2026-10-02 in favour of the
core implementation).

**S7 / solvers** — what remains is tracked: D31 (Newton over a window on more than one rank), D32
(the PETSc plugin behind the same `ImplicitComponent`/`LinearizationToken` seam, for problems the native
solver should not carry), and the open design note on Newton's convergence norm in
[](../user/convergence.md).

## What the end state is

One library. Everything under `include/fuse/` and `src/`, namespace `fuse::`, FUSE's names — no
`soma::`, no `KARMA::`, no `upstream/` directory. The parents survive as git history in their own
repositories and as two `retired` entries in `UPSTREAMS.lock`. The procedure and naming map are in
[](porting.md); invariant `I11` makes it enforceable from the first port.

## Where the import stands

Both parents were tagged `fuse-import-0` and imported byte-identical (693 files, proven by
`tools/verify-import.py --parents`) at S0. Neither has been retired yet: `upstream/soma` (181 files)
and `upstream/karma` (512 files) are both still present, and `UPSTREAMS.lock`'s `retired` field for
each is still empty, even though every path S6 names as superseding `upstream/soma` has landed —
retirement is a deliberate deletion commit that has not been made, not a gap in what superseded it.
`upstream/karma`'s own retirement waits on S8, which has not started.

Two findings from the original import, recorded in `UPSTREAMS.lock` rather than fixed, because I8
forbids editing an import:

- The coupler tree fetches yaml-cpp 0.8.0, whose `CMakeLists.txt` predates CMake 3.5 and so cannot
  configure under CMake 4.x without a policy floor. The superbuild supplies one; an installed
  yaml-cpp avoids the fetch entirely.
- **The spine tree does not compile in Debug.** `karma_domain.C` uses two unqualified enumerators of
  a scoped `enum class`, inside an `assert`. All three of that project's verified presets define
  `NDEBUG`, so the line has never been compiled — and the assertion is a tautology, so it would check
  nothing if it did. This is invariant `I6` ([](invariants.md)) demonstrating itself on real code.

Both externally meaningful milestones named when this roadmap was first written have been reached:
**S4** (a complete coupler that reconciles distributed failures, running on Kokkos) and **S6** (an
unmodified external executable, in its own process, couples through the participant ABI — the
Brusselator across three and then five separately-launched programs, [](conformance.md)).

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
