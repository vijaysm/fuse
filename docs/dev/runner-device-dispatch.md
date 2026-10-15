# Runner device dispatch: status and handoff

This is a working handoff for an in-progress effort on `vijaysm/kokkos-device`:
giving `Runner::Impl::execute()` a real, measured Kokkos device (Cuda) path,
plus an independent cuSPARSE/cuSOLVER sparse-solve backend for the two
examples (`StokesSolver`, `MonolithicGrayScott`) whose own cost is dominated
by repeated `Eigen::SparseLU` factorization. If you are picking this up in a
new session, read this whole page before changing anything — it records not
just the plan but the reasoning that narrowed it, so the same false starts
aren't repeated.

## Where this came from

You asked (after the three slowest `ctest` verification tests — `fsi_transient`,
`gray_scott_reference`, `brusselator_participants` — turned out to each take
minutes) whether Kokkos::Cuda + host parallelism could cut their time down.
Investigation found the real bottleneck wasn't elementwise/stencil compute at
all — it was many small, sequential `Eigen::SparseLU`/`SimplicialLDLT`
factorizations on tiny grids, a workload GPU data-parallelism doesn't help and
that Eigen has no device backend for regardless. That produced a two-phase
plan:

- **Phase 1 (done)**: cache the two factorizations that provably never change
  across calls. `39ad7a7` (Stokes saddle-point LU cache) cut `fsi_transient`
  from 386s to 57.5s. `2e9db1f` (Gray-Scott Newton-loop `analyzePattern()`
  hoisting) left `gray_scott_reference` at ~462s, unchanged — its cost is
  per-iteration numeric `factorize()`, which must run every iteration
  regardless of caching. Both verified bit-identical via the full
  `fsi_*`/`gray_scott_*` ctest suite.
- **Phase 2 (in progress, this document)**: you asked to "add Build Runner
  device-capability anyway and utilize something like cuSparse to perform
  fast factorizations on the device" — i.e. close `KNOWN_ISSUES.md`'s A5
  (Runner-level device dispatch, previously deferred) for real, and add the
  one change that could plausibly move `gray_scott_reference`'s remaining
  cost: a device-resident sparse solve.

## Confirmed environment facts (don't re-derive these — verify, don't repeat)

- This machine has a real GPU: `nvidia-smi` reports a GeForce GTX 1050 Ti
  (sm_61), driver CUDA 13.0, toolkit at `/usr/local/cuda-12.5`.
- `/opt/kokkos` (the Kokkos this repo resolves against, see
  `cmake/FuseDependencies.cmake`'s `fuse_require(Kokkos ...)`) is already
  built with `Kokkos_DEVICES = OPENMP;SERIAL;CUDA` for `sm_61`
  (`/opt/kokkos/lib/cmake/Kokkos/KokkosConfigCommon.cmake`).
- `ctest -R space_kernels -V` (the typed kernel-layer suite,
  `test/num/space_kernels_test.cpp`, driven by
  `EnabledExecutionSpaces`/`FUSE_TYPED_TEST_ON_SPACES`,
  `test/support/fuse_test_main.hpp:80-106`) prints
  `[ spaces ] enabled: Cuda OpenMP Serial (default Cuda)` and all 27 tests (9
  per space) pass on this hardware, right now, with zero code changes. This
  is, word for word, the number `KNOWN_ISSUES.md`'s A5 asked for ("configure
  against a GPU-enabled Kokkos and run the suite; the banner tells you
  whether the device space was actually picked up") — at the kernel-layer
  unit-test granularity, A5 is arguably already satisfiable with no code
  change at all. The remaining Phase 2 work below goes further: a measured
  number for `Runner::Impl` itself running a coupling workload, not just the
  isolated kernel functions.
- `CUDA_cusparse_LIBRARY`/`CUDA_cusolver_LIBRARY` are already resolvable in
  the CMake cache (transitively, via the same CUDA toolkit Kokkos was built
  against), but fuse's own CMake has never called `find_package(CUDAToolkit)`
  itself, and nothing in the repo references cuSPARSE/cuSOLVER yet (checked
  via repo-wide grep — zero hits).

## Why the Runner device-dispatch scope narrowed twice (read this before touching `runner_impl.hpp`/`runner.cpp`)

The first research pass (summarized, not read directly) suggested "template
`Runner::Impl` on `Space`, following `space_kernels.hpp`'s pattern" as a
roughly 200-400 line change. Reading the actual code directly, twice, changed
that:

1. `Runner::Impl` (`src/runtime/runner_impl.hpp`) is not a small Kokkos-
   dispatch class — it's the ~1000-line coupling-runtime core: MPI transport,
   coroutine-based suspension (`detail::Task<>`), checkpointing, window
   coupling, epoch agreement. Almost none of that logic touches Kokkos at
   all. Templating the whole class on `Space` would force every one of those
   unrelated methods (thousands of lines across `runner.cpp` and
   `iteration.cpp`) to become template definitions too, for zero benefit.
   **Decision: don't template `Impl` itself — only `execute()`'s elementwise
   dispatch needs `Space` genericity.**
2. `execute()` itself (`src/runtime/runner.cpp:2887-3385`) is one ~400-line
   switch over `OpKind`, and critically, **not every case is even
   Kokkos-dispatched today**. Some already call a plain, non-templated
   `num::copy(ptr, ptr, n)` — no `Space`, no `SpanOf` — a deliberate choice by
   whoever wrote this: `Source`/`ExternalDelivery` (no-ops), `Sink`/
   `Accelerate`/`Epoch`/`Constrain` (plain copy, some with MPI/reduction
   logic alongside — **update, runtime-exchange track (docs/dev/
   device-resident-arcs.md): `Constrain` moved out of this group — along
   with `Diagnose`, not named in this paragraph's own list but dispatched
   through the identical plain-copy shape — both now read/write through
   `span_c()`/`span_m()` and reduce on the arc's own space, joining
   `Transport` in the space-AWARE group instead**), `Transport` (MPI
   `move()`, which at the time this paragraph was written only ever
   operated on host memory regardless of `execution_space` —
   `mpi_is_gpu_aware()` was explicitly informational-only.
   **Update, transport track (docs/dev/device-resident-arcs.md Phase 3):**
   no longer true. `mpi_is_gpu_aware()` is now a run-time probe, and
   `Transport`'s own `OpKind` case in `execute_on<Space>()` packs/unpacks on
   the device and hands MPI a device pointer when the backend answers
   GPU-aware, staging through host memory otherwise — it moved from this
   paragraph's "host-only, read raw pointers directly" group into a new
   third group, space-AWARE, alongside none of the others here), `Remap`'s
   distributed-nodal branch (`r.nodal_dist >= 0`,
   `apply_nodal_dist_remap()` — an MPI exchange over host buffers, so it
   stays host-only even though the other two `Remap` branches move —
   **update, runtime-exchange track: also moved to the space-AWARE group;
   its own case stages its arcs explicitly around the still-host-buffered
   MPI exchange, the same shape Transport's own mixed-pair fallback
   uses**.
   **Update, runtime-exchange track, follow-up round (device-resident-
   arcs.md, "Runtime-exchange track, follow-up round"): "still-host-
   buffered" above is now true only when `comm_handle ==
   fuse::search::kNullComm` (a single-rank run). A REAL communicator
   drives a one-time resolved exchange plan instead
   (`nodal_remap_distributed.cpp`'s `build_distributed_remap_plan`) whose
   per-period gather (`Runner::Impl::nodal_dist_gather_on<Space>`) and
   scatter-add (`nodal_dist_scatter_on<Space>`) pack/unpack on `Space`
   and hand MPI a device pointer when `fuse::linalg::mpi_is_gpu_aware()`
   -- the same space-AWARE shape Transport's own case already has, not
   the mixed-pair host-staged fallback this paragraph described when it
   was written), `WindowAdvance`/`Map`/`Reduce`/`Closure` (refusals). Only about
   9 of ~15 `OpKind`s are templated on `HostExecSpace` today and need `Space`
   genericity: `Interpolate`'s identity-copy branch, `Remap`'s identity and
   weighted branches, `Sample`'s non-spacetime branch, `Accumulate`,
   `Extrapolate`, `Merge`, `Scale`, `Relax`, `Residual`. **Decision: only
   these ~9 cases become `Space`-templated; everything else in `execute()`
   is untouched, regardless of `execution_space`.**
   **Update, spacetime-history track (2026-10-15, device-resident-arcs.md
   step 8):** `Sample`'s and `Interpolate`'s own SPACETIME branches (`r.
   spacetime >= 0`), not named in this paragraph's list because they were
   host-only when it was written, are `Space`-templated now too --
   `Sample` writes its history row with a same-space `Kokkos::deep_copy`,
   and `Interpolate` calls `num::apply_spacetime_device`
   (space_kernels.hpp) instead of the host-only `apply_spacetime`.
   `place_arcs_on_device()` no longer pins either instruction's arcs.
   Unaffected: the host-only `apply_spacetime` (`src/backend/
   spacetime.cpp`) itself, still the Host-mode entry point and the device
   kernel's own oracle.

Both narrowings were confirmed with the user via `AskUserQuestion` before
proceeding, each time with the full size/risk picture disclosed. The user's
final call (explicitly, after both warnings) was to do the full `execute()`
port rather than stop at "update `KNOWN_ISSUES.md` with the kernel-layer
number we already have." That is the scope below.

## Design (see the plan file's "Design" section for the full version with code sketches)

1. `RunnerConfig::execution_space` (new `enum class RunnerExecutionSpace
   { Host, Device }`, default `Host`) + a `DeviceExecSpace` alias in
   `runner_impl.hpp` (`Kokkos::Cuda`/`HIP`/`SYCL` behind the matching
   `KOKKOS_ENABLE_*` macro, falling back to `HostExecSpace` with a hard
   configure-time refusal if `Device` is explicitly requested but no real
   device backend is compiled in — "configuration error, not silent
   downgrade," matching `FUSE_LINALG_BACKEND`'s existing validation style).
2. `execute()` becomes a tiny non-template dispatcher calling a new
   `template <typename Space> execute_on(...)`, which is today's exact
   `execute()` body with `HostExecSpace` → `Space` at the ~9 cases above.
   The correctness invariant that makes this a safe, zero-behavior-change
   refactor for the host path: the `span_c`/`span_m` helpers branch on
   `Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename
   Space::memory_space>::accessible` — when true (always true for
   `HostExecSpace`), they wrap the raw host pointer directly, byte-for-byte
   what today's code does. The device branch (never exercised by any
   existing test, since none sets `execution_space = Device`) stages through
   pre-allocated per-arc device mirrors instead.
3. Device-resident companions, each allocated **once** (never per-instruction
   — this file has a standing "nothing in the sweep loop may allocate"
   invariant, stated on `reduction_scratch`'s own comment and
   `iteration.cpp`'s header). **As built** (the plan's version of this item
   was revised during implementation; the deviations and why are in "What
   was actually built" below): one device staging buffer
   (`Impl::device_staging`, label `fuse::dev_staging`, `kDeviceStagingSlots`
   = 4 lanes each `max(slots[i].total())` doubles) allocated at the end of
   `setup()` only when `execution_space == Device`, through which each
   device-dispatched instruction's arcs are deep_copy-staged in before the
   kernel and written back after; and a device-space twin of
   `detail::RemapCsr` (`RemapCsrOf<DeviceMemSpace>`, `remap_csr_device[]`)
   built by `mirror_remap_csr()` via three `deep_copy`s from the already-built
   host triplets at `resolve_remap()` time — the expensive stable-sort/
   prefix-scan still only ever runs once, on host.
4. Staged rollout, **full ctest suite must stay green after every stage**
   before the next one starts (see the plan file for the exact 5-stage
   breakdown and what each commit contains). Stage 2 (templating `execute()`
   itself) is the highest-regression-risk stage — review the diff against
   the "already host-only" vs. "needs `Space`" list above line by line, don't
   just trust a green test run, since the existing 279 tests only exercise
   the host branch either way.
5. A new test (`test/runtime/runner_device_test.cpp`, ctest row
   `runtime.runner_device`) runs two representative multi-instruction
   coupling scenarios through the real `Runner` on both `Host` and `Device`,
   asserts the device run's space name really is `"Cuda"` on this machine
   and that `RunStats::device_dispatches > 0`, and compares results: bitwise
   for the accumulate/scale/copy-shaped instructions, to `1e-14` relative
   for `Relax`/weighted `Remap` (see "What was actually built" for why not
   bitwise there).
6. `KNOWN_ISSUES.md`'s A5 gets a final update recording this — now genuinely
   covering `Runner::Impl`, not just the kernel-layer unit tests.

## What was actually built (Part A, 2026-10-11) — deviations from the plan above

Part A is complete on `vijaysm/kokkos-device`; each stage is its own commit
(see `git log`). Four things differ from the design as planned, each decided
when the code contradicted the plan's assumption:

- **`previous[]` is staged, not device-native.** The plan said nothing
  outside `execute()` touches `previous[]`. Wrong: `advance_window()` reads
  it (`runner_impl.hpp`, the `WindowMemberConfig` comment: "previous[]/
  slots[] both read deliberately one sweep late"). A device-native copy
  would have desynchronised the window path. So `Relax`/`Residual` stage
  `previous[]` through the same per-instruction device lanes as the arcs;
  host memory stays authoritative. Cost: one extra deep_copy pair per
  `Relax`/`Residual` on the device path — performance only, which this
  document already concedes for the whole device path.

  NOT superseded (window-regions track, 2026-10-15, docs/dev/
  device-resident-arcs.md step 8): a device-resident companion for
  `previous[]` and an unpinned window-import arc were ATTEMPTED this
  round, on the theory that `advance_window()`'s own `slot_host_read()`
  would remove the "would have desynchronised the window path" risk this
  bullet names. The attempt was reverted -- see step 8's own account of
  the correctness regression it produced and the bisection that narrowed
  it without finding the cause. `previous[]` stays staged through the
  per-instruction device lane for `Relax`/`Residual`, unchanged from this
  bullet's own original account.
- **No `reduction_scratch` device companion.** It is used only by
  `iteration.cpp`'s deterministic norms over host arc buffers, which stay
  host-resident under this port. If `iteration.cpp` ever dispatches on
  `Space` it needs its own companion then; nothing today does.
- **Per-instruction staging buffer instead of per-arc mirrors.** One
  `device_staging` buffer of four lanes (an instruction reads at most three
  spans — `in`, `in2`/`weights`, `previous` — and writes one) sized to the
  widest slot is simpler and allocates less than one mirror per arc, and
  keeps the same "allocated once in `setup()`, never in the sweep loop"
  property. Lanes are keyed by host pointer within one instruction, so an
  arc that appears twice (e.g. in-place `Scale`) uses one lane; a lane is
  widened in place if a later span over the same pointer is longer
  (writing dirty data back first). Overflow of either bound is an
  `E-RT-DEVICE` failure, never silent truncation.
- **Relax/Remap are not bit-identical to host.** nvcc contracts `a*b+c`
  into a fused multiply-add inside `RelaxFunctor`/`RemapMultiFunctor` — the
  same gap D45 closed for the deterministic dot/norm reductions with
  `unfused_multiply_add`. Those two kernels were left untouched because
  `build-evidence/mutations.toml` anchors their exact source lines; the test
  asserts `<= 1e-14` relative for those two shapes and bitwise for the rest.
  Closing the gap for real means applying D45's idiom to the two functors
  and re-anchoring the mutation rows — a separate, small change if anyone
  wants bitwise there.

Two observables were added so the test can prove the device path ran rather
than infer it: `RunStats::device_dispatches` (an `int64` counter, following
`remap_weight_builds`' "a counter and not a boolean" precedent; 0 in `Host`
mode) and `Runner::device_execution_space_name()` (static,
`DeviceExecSpace::name()`). The refusal when `Device` is requested on a
build with no device backend is a `setup()`-time `E-RT-DEVICE` failure (the
plan said "`E-RT-NO-DEVICE` or similar"; `E-RT-<NOUN>` matches the existing
code family). That branch is unreachable on a CUDA build, so its test
`GTEST_SKIP`s here with the build's device space named in the skip message.

One build-environment fact worth knowing before running anything: every TU
that sees a Kokkos header must go through `/opt/kokkos/bin/nvcc_wrapper`,
which `mpicxx` picks up from `OMPI_CXX` (set by `CMakePresets.json`'s
`_machine` preset). A bare `cmake --build build-all` from a shell without
`OMPI_CXX=/opt/kokkos/bin/nvcc_wrapper` exported fails on every
Kokkos-touching TU with g++ rejecting `-Wext-lambda-captures-this`/
`-arch=sm_61`. Pre-existing, not introduced here.

**Honest expectation to state in the final report, not hide**: this makes
`Runner`'s device path slower than its host path, not faster (every
device-dispatched instruction adds a deep_copy staging round trip that the
host path doesn't need). That's fine — this work is about closing A5 for
real and proving the dispatch pattern end-to-end, not a performance win. The
performance lever is Part B below.

## Part B: cuSPARSE/cuSOLVER sparse-solve backend (independent of all of the above)

Touches only `examples/fsi_channel/stokes_solver.hpp` and
`examples/gray_scott/monolithic_reference.hpp`'s Eigen call sites — not
`Runner` at all, can land before/after/interleaved with the device-dispatch
work above.

- New `FUSE_ENABLE_CUSPARSE` CMake option (OFF by default), gated the same
  way `FUSE_ENABLE_MOAB`/`FUSE_ENABLE_PETSC` are
  (`cmake/FuseDependencies.cmake:119-124`, `config.hpp.in`'s
  `#cmakedefine01` pattern), requiring `Kokkos::CUDA` to already be a target.
- A new shared `examples/common/sparse_solve_backend.hpp` with a
  `SparseLuBackend` class matching `Eigen::SparseLU`'s method names exactly
  (`analyzePattern`/`factorize`/`solve`/`info`), so `StokesSolver`'s
  `CnStepLuCache::lu` and `MonolithicGrayScott`'s `jac_solver_` can hold one
  instead, with no other call-site changes. `#if FUSE_HAVE_CUSPARSE`:
  `cusolverRf` (the refactorization API, built for exactly this "same
  pattern, repeated numeric factorization" case), seeded by one `cusolverSp`
  analysis+factorization. `#else`: a one-line passthrough to
  `Eigen::SparseLU` — byte-for-byte today's behavior when the flag is off
  (the default).
- Verify twice: flag off (bit-identical to Phase 1's results, ~57s/~462s),
  flag on (compiles and links against the real libraries on this machine,
  passes the same tests against their existing numerical-tolerance bounds —
  not bit-identical, since `cusolverRf` isn't bit-identical to Eigen — and
  report the new `gray_scott_reference` wall-clock honestly, whichever way
  it comes out).

## Part B results (2026-10-11): built, measured, not a speed-up

`FUSE_ENABLE_CUSPARSE` exists, `examples/common/sparse_solve_backend.hpp`
holds both call sites, and the flag-on build was run against the same
verification rows as the flag-off one, on the same GTX 1050 Ti, Debug build
type both ways (`build-cusparse`, tier0 preset; `build-all`, all preset).

| row | flag off (quiet full run) | flag on (cusolverRf) |
|---|---|---|
| `gray_scott_reference` | 462.9 s | 471.4 s |
| `fsi_stokes_mms` | 55.1 s | 470.7 s |
| `fsi_transient` | 57.3 s | 124.0 s (passes only with the refinement step below) |
| `fsi_energy_diag` | pass | 5 of 6 cases pass; `LongRunEnergyIsBounded` SKIPPED (KNOWN_ISSUES D44) |
| `gray_scott_mms` / `multirate`, `fsi_beam` / `coupled`, `example.gray_scott` (+omp) | pass | pass |
| `example.fsi_channel` (+omp) | pass | DISABLED (KNOWN_ISSUES D43) |

**No gain, and the reason is not the backend.** `gray_scott_reference` runs
at `kN = 8`: a 128-unknown Jacobian. Its 460-odd seconds are not in the LU
at all (Phase 1's `analyzePattern()` hoisting did not move it either); they
are in the windowed Runner side of the comparison. A device factorisation
of 128 unknowns is pure launch latency and PCIe copies, so the flag-on
number is a few seconds *worse*. `fsi_stokes_mms` passes no LU cache by
design (it varies grid/viscosity/BCs per call), so every `cn_step` re-seeds
cusolverRf through a cusolverSp host LU plus a device setup: 8.5x slower.
The device path would start paying off at problem sizes two orders of
magnitude larger than anything the suite runs; that is a question about the
examples' sizes, not about the dependency.

**What the first flag-on run found, and what it turned out to be.** That
run failed `fsi_transient`, `fsi_energy_diag` (the finite-difference energy
budget) and `example.fsi_channel` — all accuracy stalls, no crashes. Three
rounds of measurement, in order:

1. A temporary residual diagnostic over 24,519 real Stokes solves showed
   the device solve is *as backward-stable as Eigen's* (5.3e-16 both), so
   "cusolverRf is inaccurate" was wrong. The device and host solutions
   differ by up to 1.3e-8 relative — κ·ε, both correct to working precision
   but not the *same* rounding. In that in-situ run ~4% of identical-input
   repeat solves also differed bitwise.
2. In isolation (unit tests on the real 196- and 900-unknown Stokes
   matrices, 300 back-to-back repeats) raw `cusolverRfSolve` **is**
   reproducible; the in-situ jitter did not reproduce and is not relied on
   either way. The fix that works regardless is one step of iterative
   refinement after every solve: the residual `Pb − B·y` is accumulated on
   the host in `long double` (64-bit mantissa) and one more device solve
   applies the correction. Measured against a `long double` Eigen reference,
   the refined backend's forward error is 3.7e-17 (n=196) / 1.4e-16 (n=900)
   against Eigen double's 2.7e-14 / 3.4e-14 — about 1000x *more* accurate
   than the host path, and pinned to the exact solution rather than to any
   one rounding of it. `fsi_transient` passes with it. Tests:
   `RepeatedSolvesOfTheRealStokesSystemAgreeToWorkingPrecision`,
   `RepeatedSolvesAgreeWhenOtherSolvesAreInterleaved`,
   `ForwardErrorOnTheRealStokesSystemIsNoWorseThanEigens`,
   `SolvesTheExampleSizedStokesSystemAccuratelyAndReproducibly`.
3. The two remaining failures were not the backend's. `fsi_energy_diag`'s
   no-energy guard scaled its floor with the *square* of the eigenvector
   quality, so the better solve (quality 7.7e-11 → 1.07e-11) dropped the
   floor under the mode's true E0 and the budget ran on noise; the floor
   now has an absolute 1e-16·|x|² term (ledger Ruling B-4, measured
   rationale in the test). `example.fsi_channel`'s IQN-ILS coupling is
   marginally stable near the beam's first natural period: a ≤1.07e-12
   change in the fluid solve diverges it, and the flag-*off* build diverges
   too under `--accel none` or `aitken` at the same periods. That is
   KNOWN_ISSUES **D43**, OPEN; the two example rows are `DISABLED` (visible
   in `ctest -N`, not hidden) when the flag is ON. The `fsi_*` verification
   rows still run under the flag.

4. **Found after the review, by the suite itself.** `fsi_energy_diag`'s
   480-period coupled run under the flag is **not reproducible run to run**:
   two runs of one binary agree bitwise through period 71, fork at 1e-12 at
   period 72, and 11 of 15 runs then fail to converge at period 140 (the
   flag-off binary is bitwise identical over all 480 periods, two runs). An
   uninitialised read and code layout were ruled out directly
   (`-ftrivial-auto-var-init=zero/pattern` and GCC's `-Wmaybe-uninitialized`
   on the TU; the same source rebuilt fails 2 of 3). What remains is the
   device triangular solve, which refinement makes accurate but not bitwise.
   That case is `GTEST_SKIP`ped under the flag and recorded as KNOWN_ISSUES
   **D44**, which also names the fix that would close it (host triangular
   solves from the device-refactored factors).

The lesson that outlives this branch: a fixed-point coupling iteration
converges through *a* rounding floor, and anything that moves that floor by
κ·ε — a different solver, a different pivot order, a different FMA
contraction, or a solve that is not bitwise reproducible from one call to
the next — can tip a marginal case. Any device sparse solver fuse adopts
has to be checked against the D45 determinism contract and against the
coupling tests first, not just against a residual; Kokkos Kernels' CUDA
paths included (the user deferred that dependency on 2026-10-11 for this
reason and the size argument above).

`FUSE_ENABLE_CUSPARSE` stays OFF by default. It is correct, more accurate
than the default, tested on the real systems, and documented as measured;
it is not a recommendation at the suite's problem sizes.

## Resuming this work

Check `git log --oneline -10` on `vijaysm/kokkos-device`: each stage above is
its own commit, in order, so the log tells you exactly how far this got.
Part B is fully independent and can be picked up regardless of Part A's
progress. The live planning document (with full code sketches for each
stage) is `/home/vijay/.claude/plans/piped-roaming-wolf.md` on the machine
this was written on — if unavailable, this page plus `git log` plus
`KNOWN_ISSUES.md`'s A5 section is enough to reconstruct the plan.
