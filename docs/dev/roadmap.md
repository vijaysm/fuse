# Roadmap

FUSE is built in ten stages. Each has a falsifiable exit condition, and the order is a dependency
order, not a preference: the canonical descriptor is frozen before anything binary depends on it, and
imported code lands byte-identical before it is transformed ([](invariants.md) `I8`).

Recounted from the tree at `6a4f0c5` (560 commits), not carried forward from an earlier pass: S0–S6
are complete, S7's native path is complete and measured (seam, derivative gate, Newton-Krylov
coupler on dn_heat and the Brusselator), and its PETSc substrate is built and measured against the native one behind the same seam (D32 keeps the multi-rank communicator and an agreement-bound margin); S8's inventory is done and its skeleton has landed (no physics code yet), and S9 has not started (no execution-space type beyond
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
| **S7** | Implicit and monolithic solvers | **native path complete 2026-10-11** — the residual/linearisation seam (`fuse::ImplicitComponent`/`fuse::LinearizationToken`, [](newton-krylov.md) §1–3), the derivative gate ([](derivative-gate.md): adjoint identity over eleven operator cases, Taylor-remainder order with the round-off plateau detected, assembled-vs-matrix-free agreement), and `fuse::num::solve()` — Jacobian-free Newton-Krylov with the Walker–Pernice step, right-preconditioned `gmres`, Eisenstat–Walker forcing, cubic-backtracking/critical-point/damped line searches, named convergence codes and a lagged block-Jacobi preconditioner over the members' tokens — selectable as `AccelKind::Newton` on an ordinary or a coupling-window region. Measured: dn_heat order 2.0 ± 0.24 on a temperature-dependent conductivity ([](../user/dn-heat.md)); the Brusselator's window converges in 50/50 windows in both waveform forms where Aitken fails at the third, final means agreeing with Picard to 2e-13 ([](../user/brusselator.md)). Built on the S7 linear algebra: `DistVector`/`DistSparseMatrix` ([](linalg-distributed.md)), `pcg`/`gmres` ([](linalg-krylov.md)), `SchurSlabSolver` ([](schur-slab-solver.md)). **PETSc substrate 2026-10-11:** `fuse::petsc` (`MffdToken`/`make_assembled_token`, `fuse::petsc::solve()` over `SNESNEWTONLS`, [](newton-krylov.md) §10–11) selectable per region by `NewtonConfig::substrate`, built only under `FUSE_ENABLE_PETSC` (`/opt/petsc/download-moab`) with no PETSc type in a public header; the derivative gate passes on the PETSc tokens (bit-identical MFFD action to the native one), dn_heat agrees with Picard to 2.7e-10 K and the Brusselator's 50 windows give the same final means on both substrates; SNES's missing forcing floor is applied in the KSP convergence test. **Closed 2026-10-12:** the examples' agreement bounds (built from each substrate's terminal residuals; dn_heat held with a ratio of 6.067e-04 on the owner's machine, D32), Newton's plain 2-norm against measure-weighted norms (`NewtonNorm::MeasureWeighted`, opt-in, [](../user/convergence.md)) and the direct residual path (`NewtonResidualPath::Direct`, opt-in, bit-identical to the default). **Open:** multi-rank window Newton (D31; reducer seam in place, the PETSc substrate solves on `MPI_COMM_SELF` until then), the direct path not unlocking components without rollback (`E-ITER-NOROLLBACK` is raised by the generic first-step tick, `newton-krylov.md` sec. 12; a scheduling decision), and the `FUSE_MUTATE_MPI` gate over the runtime rows. **Closed 2026-10-21 (owner decision, Option C):** the parent plan's own "first implicit participant" item (what used to be read as importing KARMA's PETSc-backed FEM component) is retitled and closed smaller: a real PETSc `Mat`/`Vec`-backed `fuse::LinearizationToken` (never `MATMFFD`), paired with a native-substrate twin of the SAME physics in one Newton-coupled region ([](newton-krylov.md) sec. 14, [](../user/petsc-reaction-diffusion.md)) -- KARMA's own FEM import (elements, quadrature, assembly, FEM transfer maps, a PETSc Vec-backed field store) stays entirely S8's own scope, untouched by this closure |
| **S8** | `physics-kit`: the optional finite-element component | **S8.0 inventory done 2026-10-27** ([](physics-kit-disposition.md), [](integrators-design.md)); owner decisions D-A to D-E recorded in the disposition page. **S8.1 skeleton landed:** `physics-kit/` exists (`fuse::physics_kit`, linking `fuse::runtime`/`fuse::backend`/`fuse::num`/`fuse::linalg`; tier 0, `FUSE_ENABLE_PHYSICS_KIT` default ON; PETSc optional through the existing `FUSE_HAVE_PETSC` define), one placeholder translation unit, the `physics.skeleton` gtest, and the two gates (`tools/check-layering.py`'s boundary check, `tools/check-physics-kit-naming.py`) that keep the dependency direction and the naming honest while the port runs — see [](physics-kit.md). No finite-element code yet. **Integrators sub-track** (time-stepping infrastructure in `fuse::num`, ahead of the finite-element component proper, [](integrators-design.md)): **Round 1 landed** (explicit Runge-Kutta family, the PI step-size controller, 21 tableaus ported from KARMA with a measured correction to each one's declared order/stability flags where the vendored metadata disagreed with a direct computation). **Round 2 landed 2026-10-31** (verified from a clean export, all 37 `num` mutation rows killed, the lead's review in [](integrators-design.md)): the DIRK/ESDIRK family (built on `fuse::num::solve()`, mass-matrix and non-autonomous support, Prothero-Robinson order-reduction measured), the Rosenbrock family (built directly on `gmres`/`MatrixFreeOperator`, the analytic-`jacobian_action()`-or-MFFD dispatch task item 2 asked for, a stage-solve degradation curve measured at two inexactness ratios), a generic rooted-tree order-condition checker (through order 5, catching a real order-4 defect in the Round 1 SDIRK44 tableau, corrected to order 3, KNOWN_ISSUES.md D98), three new tableaus (ESDIRK3(2)4L[2]SA, SSPRK(3,3), SSPRK(10,4)) with an SSP coefficient computed from each one's own Shu-Osher form rather than copied, and a cubic-Hermite dense-output interpolant (a deviation from the task's own "DOPRI5's free fourth-order interpolant" ask -- a reliable source for that specific quartic's coefficients was not found this round, logged as an open question). **Deferred to Round 3:** the Kennedy-Carpenter ESDIRK4(3)6L[2]SA/5(4)7L[2]SA and Kvaerno (2004) pairs, the Conde-Fekete-Shadid 2018 embedded SSP pair (all named candidates whose many-digit coefficients could not be cross-verified from two independent sources within this round), the dynamical-time-scale controller, and the method-specific (DOPRI5/ESDIRK) dense-output polynomials **S8.2 install/export landed 2026-10-31:** `fuse::physics_kit` ships in its own export set as the `find_package` component `physics_kit`, measured on and off by `install.consumer_find_package`; the public-header third-party rule now covers `physics-kit/include/` ([](physics-kit.md)). **S8.3 wave 1 landed 2026-10-31** ([](physics-kit.md)'s own Wave 1 section): the finite-element core (reference topology, native Gauss-Legendre/Gauss-Jacobi quadrature, the Lagrange nodal basis registry, the isoparametric geometry map, and the mesh-bound evaluator) ported into `fuse_physics::fe` for the Edge/Triangle/Quadrilateral/Tetrahedron/Hexahedron cell families at Lagrange order 0-2, plus a native (not yet MOAB/MPI-bound) `fuse_physics::spatial::Mesh`/`Domain`; prism, pyramid, and the MOAB-backed mesh binding are explicitly deferred (physics-kit.md's own open-questions list), not silently dropped. **Next:** integrators round 3 (DOPRI5's quartic dense output, the Kennedy-Carpenter 4(3)6 and 5(4)8 pairs from NASA TM-2001-211038, Kvaerno 2004 once the owner supplies it, the Conde-Fekete-Shadid SSP pairs, the dynamical-time-scale controller) and the finite-element port proper. |
| **S9** | Portability and scale: budgets, cluster rehearsal, adversarial suite | **not started** — no execution space beyond `Kokkos::{OpenMP,Serial}` is instantiated anywhere (grepped `include/`, `src/`: zero `Kokkos::Cuda`/`HIP`/`SYCL`); no measurement harness, no cluster rehearsal, no device build |

**Critical path.** S0 → S1 → S2 → S3 → S4 → S5 → S6. S7 and S8 are independent once S5 exits. S9
begins at S4 and finishes last.

## Open D-items, grouped by the stage they bear on

`KNOWN_ISSUES.md`'s `D` section is the source; this groups its still-open entries (struck-through
titles are closed) by which stage above they qualify, so a reader does not have to read the whole
file to see what is left in each.

**S0 / build infrastructure** — D26 (`FUSE_LINALG_BACKEND=builtin` fails to configure since the
examples link Eigen directly; the option's own fate, real configuration or retired, is undecided);
~~D30~~ (closed: `FUSE_FP_REPRODUCIBLE`, default ON, offers both — `-ffp-contract=off` everywhere
(bit-for-bit reproducible, the default) or the compiler's own contraction with the three affected
assertions restated as a derived forward-error bound; [](testing.md#floating-point-contraction-policy)
has the measured cost and a found-in-closing regression: two pre-existing, unrelated tests
(`num.accelerator`, `verification.fsi_energy_diag`) fail under the reproducible default because
removing contraction measurably changes an ill-conditioned iteration's own convergence).

**S5 / backends** — D27 (the native backend's remap halo has a forward fill,
`remap_halo_pattern`/`make_mpi_exchanger`, but no `accumulate_remap_halo` — the reverse exchange a
future native conservation-audit tool would want; nothing in the tree needs it yet); D15 (the
provider direction is conformance-tested on one process; mixed checkpoint/restore and a plugin on a
rank subset are tested, and device fields are built and gated at tier 0 (Option A, Stage B-2) and the
positive device-resident path ran on CUDA on 2026-10-21; open: the mount-everywhere question and one
surviving mutation row); D14 (LowFive's fmt rebuild is done and the distributed transport
now RUNS for MPMD launch — `lowfive.mpmd.3` passes, D39 — Demo B passes as the ctest row `example.demo_b` (2026-10-14), and the
accept-mode participants row passes since D14 Stage B (2026-10-13); D14 is closed).

**S6 / interfaces** — D23 (worked around 2026-10-12: the host's MPICH was reinstalled to a `ch4:ofi`
build, which does not hang on accept/connect sharing a node — `ch3:nemesis` itself was never fixed);
D17 (`--accel aitken`
converges through `--driver runner` but not `--driver window` on the identical config — open, not
pursued, Conformance Unit 2 needs no accelerator); D18 (`connect()`'s `Connect`-mode path supports
exactly two components; a third refuses — open, not pursued, Unit 2 routes around it); D7
(`WindowCoupler` supports at most one straddling member per window and no `Fixed` step longer than
the window — rejected, support is design work); D8 (a `Remap` into a nodal field on more than one
rank: two real correctness defects found by a host MPI run are fixed, a further host run is the
final outcome); D3 (two runtime mutations survive on the host; deferred 2026-10-02 in favour of the
core implementation).

**S7 / solvers** — what remains is tracked: D31 (Newton over a window on more than one rank, on either
substrate), D32's residue (the PETSc substrate's communicator follows D31; the examples' agreement
bounds carry a 20× margin until the result structs expose ‖F₀‖), and the open design note on Newton's
convergence norm in [](../user/convergence.md).

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
`upstream/karma`'s own retirement waits on S8, which is in progress (S8.0, S8.1 and S8.2 done; the finite-element port and the remaining integrator rounds are open).

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
