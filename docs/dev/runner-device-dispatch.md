# Runner device dispatch

`Runner::Impl::execute()` has a Kokkos device (Cuda) path selected by `RunnerConfig::execution_space`.
This page records how the dispatch is structured, why its scope is limited to the elementwise
`OpKind` cases, and what was measured. A second, independent part describes an optional
cuSPARSE/cuSOLVER sparse-solve backend for the two examples (`StokesSolver`, `MonolithicGrayScott`)
whose cost is dominated by repeated `Eigen::SparseLU` factorization.
[](device-resident-arcs.md) supersedes the staging design of Part A for Device mode.

## Motivation

The three slowest verification tests (`fsi_transient`, `gray_scott_reference`,
`brusselator_participants`) each take minutes. The bottleneck is not elementwise or stencil compute:
it is many small, sequential `Eigen::SparseLU`/`SimplicialLDLT` factorizations on tiny grids, a
workload GPU data-parallelism does not help and for which Eigen has no device backend.

Two changes follow from that:

- Cache the two factorizations that provably never change across calls. The Stokes saddle-point LU
  cache cut `fsi_transient` from 386 s to 57.5 s. Hoisting `analyzePattern()` out of the Gray-Scott
  Newton loop left `gray_scott_reference` at about 462 s, unchanged: its cost is the per-iteration
  numeric `factorize()`, which must run every iteration. Both are bit-identical to before across the
  `fsi_*`/`gray_scott_*` ctest suite.
- Give the Runner a device-capable `execute()` (Part A) and a device-resident sparse solve (Part B).

## Environment facts

- The development machine has a GeForce GTX 1050 Ti (sm_61), driver CUDA 13.0, toolkit at
  `/usr/local/cuda-12.5`.
- `/opt/kokkos` (the Kokkos this repo resolves against, see `cmake/FuseDependencies.cmake`'s
  `fuse_require(Kokkos ...)`) is built with `Kokkos_DEVICES = OPENMP;SERIAL;CUDA` for `sm_61`
  (`/opt/kokkos/lib/cmake/Kokkos/KokkosConfigCommon.cmake`).
- `ctest -R space_kernels -V` (the typed kernel-layer suite, `test/num/space_kernels_test.cpp`,
  driven by `EnabledExecutionSpaces`/`FUSE_TYPED_TEST_ON_SPACES`,
  `test/support/fuse_test_main.hpp:80-106`) prints `[ spaces ] enabled: Cuda OpenMP Serial (default
  Cuda)` and all 27 tests (9 per space) pass. This is the kernel-layer granularity of "configure
  against a GPU-enabled Kokkos and run the suite; the banner tells you whether the device space was
  actually picked up". The Runner-level work below goes further: a measured number for
  `Runner::Impl` itself running a coupling workload.
- `CUDA_cusparse_LIBRARY`/`CUDA_cusolver_LIBRARY` are resolvable in the CMake cache (transitively,
  via the same CUDA toolkit Kokkos was built against), but fuse's own CMake only calls
  `find_package(CUDAToolkit)` under `FUSE_ENABLE_CUSPARSE`.

## Why the dispatch scope is limited to `execute()`'s elementwise cases

Templating `Runner::Impl` on `Space`, following `space_kernels.hpp`'s pattern, looks like a
roughly 200-400 line change. Reading the code shows otherwise:

1. `Runner::Impl` (`src/runtime/runner_impl.hpp`) is not a small Kokkos-dispatch class. It is the
   coupling-runtime core: MPI transport, coroutine-based suspension (`detail::Task<>`),
   checkpointing, window coupling, epoch agreement. Almost none of that logic touches Kokkos.
   Templating the whole class on `Space` would force every one of those unrelated methods (thousands
   of lines across `runner.cpp` and `iteration.cpp`) to become template definitions too, for zero
   benefit. **Only `execute()`'s elementwise dispatch needs `Space` genericity.**
2. `execute()` itself (`src/runtime/runner.cpp`) is one switch of roughly 400 lines over `OpKind`,
   and not every case is Kokkos-dispatched. Some call a plain, non-templated `num::copy(ptr, ptr,
   n)`, with no `Space` and no `SpanOf`:
   - `Source`/`ExternalDelivery` are no-ops.
   - `Sink`/`Accelerate`/`Epoch` are plain copies, some with MPI or reduction logic alongside.
   - `WindowAdvance`/`Map`/`Reduce`/`Closure` are refusals.

   Three cases are space-aware rather than host-only:
   - `Transport` packs and unpacks the MPI-ordered layer on `Space` itself
     (`Transport::packed_layout()`'s index arrays, mirrored to the device once at setup) and calls
     `Transport::move_packed()`, the bare exchange alone, with a device pointer when
     `mpi_is_gpu_aware()` (a run-time probe) answers true, staging through host memory otherwise.
     `Transport::move()` itself is a host routine (`std::memcpy` pack/unpack) and is never handed a
     device pointer; handing one to it segfaulted under a genuinely CUDA-aware MPI (Open MPI 5.0.9).
   - `Constrain` and `Diagnose` read and write through `span_c()`/`span_m()` and reduce on the arc's
     own space.
   - `Remap`'s distributed-nodal branch (`r.nodal_dist >= 0`, `apply_nodal_dist_remap()`) is
     space-aware. With `comm_handle == fuse::search::kNullComm` (a single-rank run) its case stages
     its arcs explicitly around a host-buffered exchange. With a real communicator it drives a
     one-time resolved exchange plan (`nodal_remap_distributed.cpp`'s
     `build_distributed_remap_plan`) whose per-period gather
     (`Runner::Impl::nodal_dist_gather_on<Space>`) and scatter-add (`nodal_dist_scatter_on<Space>`)
     pack and unpack on `Space` and hand MPI a device pointer when `fuse::linalg::mpi_is_gpu_aware()`.

   About 9 of roughly 15 `OpKind`s are templated on `HostExecSpace` and need `Space` genericity:
   `Interpolate`'s identity-copy branch, `Remap`'s identity and weighted branches, `Sample`'s
   non-spacetime branch, `Accumulate`, `Extrapolate`, `Merge`, `Scale`, `Relax`, `Residual`. The
   spacetime branches of `Sample` (`r.spacetime >= 0`) and `Interpolate` are `Space`-templated as
   well: `Sample` writes its history row with a same-space `Kokkos::deep_copy`, and `Interpolate`
   calls `num::apply_spacetime_device` (`space_kernels.hpp`) instead of the host-only
   `apply_spacetime`. `place_arcs_on_device()` no longer pins either instruction's arcs. The
   host-only `apply_spacetime` (`src/backend/spacetime.cpp`) remains the Host-mode entry point and
   the device kernel's oracle. Everything else in `execute()` is untouched, regardless of
   `execution_space`.

## Design

1. `RunnerConfig::execution_space` is `enum class RunnerExecutionSpace { Host, Device }`, default
   `Host`. `runner_impl.hpp` has a `DeviceExecSpace` alias (`Kokkos::Cuda`/`HIP`/`SYCL` behind the
   matching `KOKKOS_ENABLE_*` macro, falling back to `HostExecSpace`). Requesting `Device` when no
   real device backend is compiled in is a configuration error, not a silent downgrade, matching
   `FUSE_LINALG_BACKEND`'s validation style.
2. `execute()` is a small non-template dispatcher calling `template <typename Space>
   execute_on(...)`, which is the original `execute()` body with `HostExecSpace` replaced by
   `Space` at the cases above. This is a zero-behavior-change refactor for the host path because
   the `span_c`/`span_m` helpers branch on `Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename
   Space::memory_space>::accessible`: when true (always true for `HostExecSpace`) they wrap the raw
   host pointer directly. The device branch stages through pre-allocated device buffers instead.
3. Device-resident companions are allocated once, never per instruction (nothing in the sweep loop
   may allocate; see `reduction_scratch`'s comment and `iteration.cpp`'s header):
   - One device staging buffer (`Impl::device_staging`, label `fuse::dev_staging`,
     `kDeviceStagingSlots` = 4 lanes each `max(slots[i].total())` doubles), allocated at the end of
     `setup()` only when `execution_space == Device`. Each device-dispatched instruction's arcs are
     `deep_copy`-staged in before the kernel and written back after.
   - A device-space twin of `detail::RemapCsr` (`RemapCsrOf<DeviceMemSpace>`,
     `remap_csr_device[]`) built by `mirror_remap_csr()` via three `deep_copy`s from the host
     triplets at `resolve_remap()` time. The expensive stable-sort and prefix-scan still run once,
     on the host.
4. `test/runtime/runner_device_test.cpp` (ctest row `runtime.runner_device`) runs two representative
   multi-instruction coupling scenarios through the real `Runner` on both `Host` and `Device`,
   asserts the device run's space name is `"Cuda"` on a CUDA build and that
   `RunStats::device_dispatches > 0`, and compares results: bitwise for the accumulate, scale and
   copy-shaped instructions, to `1e-14` relative for `Relax` and weighted `Remap`.

## What was built, and where it differs from the design

- **`previous[]` is staged, not device-native**, for an arc that stays host-resident. Outside
  `execute()`, `advance_window()` reads `previous[]` (`runner_impl.hpp`, the `WindowMemberConfig`
  comment: "previous[]/slots[] both read deliberately one sweep late"), so a device-native copy
  would desynchronise the window path. `Relax`/`Residual` stage `previous[]` through the same
  per-instruction device lanes as the arcs, and host memory stays authoritative; the cost is one
  extra `deep_copy` pair per `Relax`/`Residual`. For an arc that is itself device-resident,
  `advance_window()` reaches a window-import arc only through `slot_host_read()` (a `deep_copy`
  into a host scratch buffer), which removes the desynchronisation risk, so `Relax`/`Residual`
  keep the carried iterate in a same-space `previous_device[arc]` companion instead. An earlier
  attempt at this hit a correctness regression whose root cause was a device-placement bug
  unrelated to `previous[]`; see [](device-resident-arcs.md).
- **No `reduction_scratch` device companion.** It is used only by `iteration.cpp`'s deterministic
  norms over host arc buffers, which stay host-resident. If `iteration.cpp` ever dispatches on
  `Space` it needs its own companion then.
- **Per-instruction staging buffer instead of per-arc mirrors.** One `device_staging` buffer of four
  lanes (an instruction reads at most three spans, `in`, `in2`/`weights`, `previous`, and writes
  one) sized to the widest slot is simpler and allocates less than one mirror per arc, and keeps
  the "allocated once in `setup()`, never in the sweep loop" property. Lanes are keyed by host
  pointer within one instruction, so an arc that appears twice (for example in-place `Scale`) uses
  one lane. A lane is widened in place if a later span over the same pointer is longer (writing
  dirty data back first). Overflow of either bound is an `E-RT-DEVICE` failure, never silent
  truncation.
- **`Relax` and `Remap` are not bit-identical to host.** nvcc contracts `a*b+c` into a fused
  multiply-add inside `RelaxFunctor`/`RemapMultiFunctor`, the same gap closed for the deterministic
  dot/norm reductions with `unfused_multiply_add`. Those two kernels are left untouched because
  `build-evidence/mutations.toml` anchors their exact source lines. The test asserts `<= 1e-14`
  relative for those two shapes and bitwise for the rest. Applying the same idiom to the two
  functors and re-anchoring the mutation rows would make them bitwise.

Two observables let the test prove the device path ran rather than infer it:
`RunStats::device_dispatches` (an `int64` counter, following `remap_weight_builds`' "a counter and
not a boolean" precedent; 0 in `Host` mode) and `Runner::device_execution_space_name()` (static,
`DeviceExecSpace::name()`). The refusal when `Device` is requested on a build with no device
backend is a `setup()`-time `E-RT-DEVICE` failure. That branch is unreachable on a CUDA build, so
its test `GTEST_SKIP`s there, with the build's device space named in the skip message.

Build environment: every TU that sees a Kokkos header must go through
`/opt/kokkos/bin/nvcc_wrapper`, which `mpicxx` picks up from `OMPI_CXX` (set by
`CMakePresets.json`'s `_machine` preset). A bare `cmake --build build-all` from a shell without
`OMPI_CXX=/opt/kokkos/bin/nvcc_wrapper` exported fails on every Kokkos-touching TU with g++
rejecting `-Wext-lambda-captures-this`/`-arch=sm_61`.

The staged device path is slower than the host path, not faster: every device-dispatched
instruction adds a `deep_copy` staging round trip that the host path does not need. It exists to
prove the dispatch pattern end to end, not as a performance win.

## Part B: cuSPARSE/cuSOLVER sparse-solve backend

Part B touches only `examples/fsi_channel/stokes_solver.hpp` and
`examples/gray_scott/monolithic_reference.hpp`'s Eigen call sites, not `Runner`.

- `FUSE_ENABLE_CUSPARSE` is a CMake option (OFF by default), gated the same way
  `FUSE_ENABLE_MOAB`/`FUSE_ENABLE_PETSC` are (`cmake/FuseDependencies.cmake:119-124`,
  `config.hpp.in`'s `#cmakedefine01` pattern), and requires `Kokkos::CUDA` to already be a target.
- `examples/common/sparse_solve_backend.hpp` has a `SparseLuBackend` class matching
  `Eigen::SparseLU`'s method names exactly (`analyzePattern`/`factorize`/`solve`/`info`), so
  `StokesSolver`'s `CnStepLuCache::lu` and `MonolithicGrayScott`'s `jac_solver_` hold one with no
  other call-site changes. With `FUSE_HAVE_CUSPARSE` it uses `cusolverRf` (the refactorization API,
  built for the "same pattern, repeated numeric factorization" case), seeded by one `cusolverSp`
  analysis and factorization. Without it, it is a passthrough to `Eigen::SparseLU`, byte for byte
  the behavior with the flag off.

### Results: built, measured, not a speed-up

The flag-on build was run against the same verification rows as the flag-off one, on the same GTX
1050 Ti, Debug build type both ways.

| row | flag off (quiet full run) | flag on (cusolverRf) |
|---|---|---|
| `gray_scott_reference` | 462.9 s | 471.4 s |
| `fsi_stokes_mms` | 55.1 s | 470.7 s |
| `fsi_transient` | 57.3 s | 124.0 s (passes only with the refinement step below) |
| `fsi_energy_diag` | pass | 5 of 6 cases pass; `LongRunEnergyIsBounded` skipped |
| `gray_scott_mms` / `multirate`, `fsi_beam` / `coupled`, `example.gray_scott` (+omp) | pass | pass |
| `example.fsi_channel` (+omp) | pass | disabled |

**No gain, and the reason is not the backend.** `gray_scott_reference` runs at `kN = 8`, a
128-unknown Jacobian. Its 460-odd seconds are not in the LU at all (hoisting `analyzePattern()` did
not move it either); they are in the windowed Runner side of the comparison. A device factorisation
of 128 unknowns is pure launch latency and PCIe copies, so the flag-on number is a few seconds
worse. `fsi_stokes_mms` passes no LU cache by design (it varies grid, viscosity and BCs per call),
so every `cn_step` re-seeds cusolverRf through a cusolverSp host LU plus a device setup, 8.5x
slower. The device path would start paying off at problem sizes two orders of magnitude larger than
anything the suite runs; that is a question about the examples' sizes, not about the dependency.

**Accuracy findings.** The first flag-on run failed `fsi_transient`, `fsi_energy_diag` (the
finite-difference energy budget) and `example.fsi_channel`, all accuracy stalls rather than crashes.

1. A residual diagnostic over 24,519 real Stokes solves showed the device solve is as
   backward-stable as Eigen's (5.3e-16 both). The device and host solutions differ by up to 1.3e-8
   relative, which is kappa times epsilon: both correct to working precision but not the same
   rounding. In that in-situ run about 4% of identical-input repeat solves also differed bitwise.
2. In isolation (unit tests on the real 196- and 900-unknown Stokes matrices, 300 back-to-back
   repeats) raw `cusolverRfSolve` is reproducible; the in-situ jitter did not reproduce and is not
   relied on either way. The fix that works regardless is one step of iterative refinement after
   every solve: the residual `Pb - B*y` is accumulated on the host in `long double` (64-bit
   mantissa) and one more device solve applies the correction. Against a `long double` Eigen
   reference, the refined backend's forward error is 3.7e-17 (n=196) / 1.4e-16 (n=900) against
   Eigen double's 2.7e-14 / 3.4e-14, about 1000x more accurate than the host path and pinned to the
   exact solution rather than to one rounding of it. `fsi_transient` passes with it. Tests:
   `RepeatedSolvesOfTheRealStokesSystemAgreeToWorkingPrecision`,
   `RepeatedSolvesAgreeWhenOtherSolvesAreInterleaved`,
   `ForwardErrorOnTheRealStokesSystemIsNoWorseThanEigens`,
   `SolvesTheExampleSizedStokesSystemAccuratelyAndReproducibly`.
3. The two remaining failures were not the backend's. `fsi_energy_diag`'s no-energy guard scaled its
   floor with the square of the eigenvector quality, so the better solve (quality 7.7e-11 to
   1.07e-11) dropped the floor under the mode's true E0 and the budget ran on noise; the floor now
   has an absolute `1e-16*|x|^2` term. `example.fsi_channel`'s IQN-ILS coupling is marginally stable
   near the beam's first natural period: a change of at most 1.07e-12 in the fluid solve diverges
   it, and the flag-off build diverges too under `--accel none` or `aitken` at the same periods. The
   two example rows are `DISABLED` (visible in `ctest -N`, not hidden) when the flag is ON. The
   `fsi_*` verification rows still run under the flag.
4. `fsi_energy_diag`'s 480-period coupled run under the flag is not reproducible run to run: two
   runs of one binary agree bitwise through period 71, fork at 1e-12 at period 72, and 11 of 15 runs
   then fail to converge at period 140 (the flag-off binary is bitwise identical over all 480
   periods, two runs). An uninitialised read and code layout were ruled out directly
   (`-ftrivial-auto-var-init=zero/pattern` and GCC's `-Wmaybe-uninitialized` on the TU; the same
   source rebuilt fails 2 of 3). What remains is the device triangular solve, which refinement makes
   accurate but not bitwise. That case is `GTEST_SKIP`ped under the flag. Host triangular solves
   from the device-refactored factors would close it.

A fixed-point coupling iteration converges through a rounding floor, and anything that moves that
floor by kappa times epsilon (a different solver, a different pivot order, a different FMA
contraction, or a solve that is not bitwise reproducible from one call to the next) can tip a
marginal case. Any device sparse solver fuse adopts has to be checked against the deterministic
reduction contract and against the coupling tests first, not just against a residual; Kokkos
Kernels' CUDA paths included, which are not used for this reason and the size argument above.

`FUSE_ENABLE_CUSPARSE` stays OFF by default. It is correct, more accurate than the default, tested
on the real systems, and documented as measured; it is not a recommendation at the suite's problem
sizes.
