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
   logic alongside), `Transport` (MPI `move()`, which only ever operates on
   host memory regardless of `execution_space` — `mpi_is_gpu_aware()` is
   explicitly informational-only per `include/fuse/base/config.hpp.in`'s own
   comment), `WindowAdvance`/`Map`/`Reduce`/`Closure` (refusals). Only about
   9 of ~15 `OpKind`s are templated on `HostExecSpace` today and need `Space`
   genericity: `Interpolate`'s identity-copy branch, `Remap`'s identity and
   weighted branches, `Sample`'s non-spacetime branch, `Accumulate`,
   `Extrapolate`, `Merge`, `Scale`, `Relax`, `Residual`. **Decision: only
   these ~9 cases become `Space`-templated; everything else in `execute()`
   is untouched, regardless of `execution_space`.**

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
   `iteration.cpp`'s header): per-arc field mirrors (deep_copy staged in/out
   around each device-dispatched instruction), a device-space companion to
   `detail::RemapCsr` (built via one `deep_copy` from the already-built host
   triplets — the expensive stable-sort/prefix-scan still only ever runs once,
   on host), a device-space companion to `reduction_scratch` (reusing
   `space_kernels.hpp`'s existing `finalize_deterministic_scratch`
   mirror+`deep_copy` idiom verbatim), and device-native (no staging needed)
   versions of `previous[]`/spacetime `history`, since nothing outside
   `execute()` itself ever touches those two.
4. Staged rollout, **full ctest suite must stay green after every stage**
   before the next one starts (see the plan file for the exact 5-stage
   breakdown and what each commit contains). Stage 2 (templating `execute()`
   itself) is the highest-regression-risk stage — review the diff against
   the "already host-only" vs. "needs `Space`" list above line by line, don't
   just trust a green test run, since the existing 279 tests only exercise
   the host branch either way.
5. A new test (`test/runtime/runner_device_test.cpp` or similar) runs a
   representative multi-instruction coupling scenario through the real
   `Runner::Impl::execute()` on both `Host` and `Device`, asserts
   bit-identical results (deterministic reductions make this valid) and
   asserts the device run's space name really is `"Cuda"` on this machine.
6. `KNOWN_ISSUES.md`'s A5 gets a final update recording this — now genuinely
   covering `Runner::Impl`, not just the kernel-layer unit tests.

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

## Resuming this work

Check `git log --oneline -10` on `vijaysm/kokkos-device`: each stage above is
its own commit, in order, so the log tells you exactly how far this got.
Part B is fully independent and can be picked up regardless of Part A's
progress. The live planning document (with full code sketches for each
stage) is `/home/vijay/.claude/plans/piped-roaming-wolf.md` on the machine
this was written on — if unavailable, this page plus `git log` plus
`KNOWN_ISSUES.md`'s A5 section is enough to reconstruct the plan.
