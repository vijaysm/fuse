# Device-resident execution: arcs, mesh, topology and transport

Status: design, approved 2026-10-12. Supersedes the staging design of
[](runner-device-dispatch.md) Part A for `RunnerConfig::execution_space == Device`;
Host mode is unchanged by anything below and must stay byte-identical throughout.

## Why a second step after A5

Part A proved the dispatch pattern: nine `OpKind` cases launch on
`Kokkos::Cuda` through a four-lane `device_staging` buffer, so every
device-dispatched instruction pays a host→device copy of its inputs and a
device→host copy of its output. The device path is therefore slower than the
host path at every size, which A5 states plainly. The owner's decision is to
run the coupler's compute on `Kokkos::DefaultExecutionSpace` (Cuda when the
build has it) with no host round trips inside the sweep, to make the mesh and
the topology operations device-capable, and to have the infrastructure for
multi-rank device runs in place before a multi-GPU machine is available.

## Buffer inventory (as of e6a92e4) and where each goes

| buffer | today | Device mode after this work |
|---|---|---|
| `Impl::storage[i]` (coupler-owned arcs), `Slot::data` | `std::vector<double>`, raw host pointer | `Kokkos::View<double*, ArcMemSpace>` per arc; `Slot` carries the space; `ArcMemSpace = DeviceMemSpace` iff `execution_space == Device` |
| component fields aliased as arcs (`owned_by_component`) | host `FieldRef` only; `MemorySpace::Device` refused `E-RT-SPACE` | a `MemorySpace::Device` `FieldRef` aliases directly; a host `FieldRef` gets one device mirror, refreshed the first time a kernel reads it after component code ran (`Impl::component_epoch`), and copied out at its `Sink` |
| `previous[]` (Relax/Residual, read one sweep late by `advance_window()`) | host vectors | `ArcMemSpace` views; `advance_window()` reads through a kernel or an explicit host mirror refreshed at the window boundary |
| `SpacetimeInfo::history` | host vector | `ArcMemSpace` view (Sample's spacetime branch is already Space-generic) |
| `reduction_scratch` | `HostSpace` view, norms in `iteration.cpp` on `HostExecSpace` | `ArcMemSpace` companion; the deterministic norms dispatch on the arc space (kernels already Space-generic, D11) |
| `RegionRuntime::x_old/x_new/x_out`, accelerator state | host vectors, `Accelerator` host-only | stay host; the gather `slots[b.feed_arc] → x_old` and scatter back become `deep_copy`s per sweep over the Newton/Picard interface vector (small by construction: interface dofs, not fields) |
| transport `send_buf/recv_buf` | host vectors, `MPI_Alltoallv` | **done (transport track, 2026-10-13):** `Impl::transport_device` (device views, built once when both of a Transport instruction's arcs land on the device) when `Transport::is_gpu_aware()` is true at run time; otherwise the pack/unpack kernels still run on the device and only the already-existing host `send`/`recv` buffers stage the `MPI_Alltoallv` itself. A mixed device/host arc pair (found to occur for real -- KNOWN_ISSUES D48 -- not a hypothetical) stages the device side through `Impl::transport_in_scratch`/`transport_out_scratch` and runs the ordinary host loop. |
| `RemapCsr` / `remap_csr_device` | host triplets + device twin built at resolve | device CSR is the primary in Device mode; builders may produce it directly (Phase 2) |
| `weight_sets` (Constrain measures) | host vectors | **done (runtime-exchange track, 2026-10-14):** host (`weight_sets` never changes after setup); `weight_sets_device` is a device-resident mirror, built once in `build_constrain_device_buffers()`, and `Constrain` integrates on the arc's own space via `fuse::num::deterministic_dot` into `constrain_reduction_scratch`, result to host |
| `NodalDistRemap` buffers | host, MPI exchange | **done (runtime-exchange track, 2026-10-14):** host; `apply_nodal_dist_remap()` stages its arcs explicitly (it is an MPI exchange like Transport) -- the arcs may now be device-resident (`place_arcs_on_device()` no longer pins them), but `apply_distributed_H`/`apply_distributed_HT`'s own `MPI_Allgatherv` calls stay on host buffers regardless of GPU-aware MPI (see the new "Runtime-exchange track" section below for why) |
| checkpoint payloads, `verify_save`, Newton seam snapshots (`fixed_point_residual.cpp`) | host copies of slots | `deep_copy` from the arc space into the existing host containers |

## The boundary rule

Host memory is touched in Device mode only at:

1. a host-registered component field: one mirror refresh per component step
   that a kernel reads it after (a `Source` is stamped `BeforeTicks`, the
   `Sample` that reads the field `AfterTicks`, so a refresh AT the `Source`
   is one tick stale -- the mirror is refreshed at the read instead), and
   one copy out at its `Sink`;
2. `Transport` when `Transport::is_gpu_aware()` is false at run time (the
   pack/gather and unpack/scatter kernels still run on the device; only the
   already-packed buffer round-trips to host around the `MPI_Alltoallv`
   itself -- two `device_boundary_copies` per exchange, not a whole-field
   copy), or when `FUSE_TRANSPORT_HOST_STAGING=1` forces that path on a
   GPU-aware MPI so the fallback has a test everywhere; one crossing in
   each direction either way. Also `Transport` when its two arcs are a
   MIXED device/host pair (KNOWN_ISSUES D48: a real case, not a
   hypothetical -- a Remap's conservation check pins one of them) -- the
   device-resident side stages through a host scratch buffer and the
   ordinary scalar host loop runs against two host pointers, one crossing
   for whichever side was device. `apply_nodal_dist_remap()`
   (runtime-exchange track, 2026-10-14) is the same shape: a device-
   resident `in`/`out` arc is staged through one `slot_to_vector()`/
   `host_to_slot()` deep_copy at the function's own entry/exit, around an
   MPI exchange (`apply_distributed_H`/`apply_distributed_HT`) that stays
   on host buffers either way -- see the dedicated section below for why
   that exchange itself is not GPU-aware;
3. `Epoch` (status reconciliation is MPI on scalars; the value copy itself
   runs on the arc space), `Diagnose` (runtime-exchange track, done:
   `has_nonfinite` is a reduction on the arc space via `any_nonfinite<Space>`;
   only the verdict is host), `Constrain` (runtime-exchange track, done: the
   conservation sum is a reduction on the arc space via `deterministic_dot`
   against `weight_sets_device`; only the scalar verdict is host);
4. the accelerator interface vector and the window-member bookkeeping
   (`advance_window`'s clocks and the straddle state, which are scalars);
5. checkpoint/resume and the Newton seam's save/restore.

Everything else — every elementwise `OpKind`, the norms, Relax's `previous[]`,
the spacetime history — runs where the arcs live. The host-only `OpKind`s
that remain are exactly the ones listed in `execute_on()`'s comment; none of
them moves data elementwise except through `num::copy`, which becomes
`num::copy<Space>` on the arc space.

## Staging after this work

`device_staging` is retained for a host-registered component field that a
kernel WRITES (a mirror is read-only; the write has to reach the host
array), for `previous[]` and history until step 3's follow-up moves them,
and for a host arc pinned by a host-only instruction. If the mirror design
covers every scenario in the verification rows, `device_staging` and its
fault path (`E-RT-DEVICE`, ruling F-1) are removed; the
`RunStats::device_dispatches` counter stays, and
`RunStats::device_boundary_copies` counts the boundary copies so the tests
can assert the no-round-trip property directly.

## Mesh and topology (Phase 2)

`MeshArrays<MemorySpace>` already defaults to the default execution space's
memory; `NativeMeshView` converts to host at construction and the NativeMesh
tests were pinned to host for the CUDA build. The change is to keep the
default-space arrays as primary, expose host mirrors on demand, and make the
consumers Space-generic in this order: search (`BoxIndex`/distributed search:
ArborX is device-native; the execution space is currently pinned to
`DefaultHostExecutionSpace`), halo and migration pack/unpack, MLS and
intersection weight assembly producing the device CSR directly, and last the
supermesh intersection. The supermesh path calls r3d, a C library with
fixed-size polytopes; whether it compiles as device code under nvcc or needs a
Kokkos-native clipping kernel is to be measured in that step and reported,
not assumed. MOAB-backed meshes stay host-resident; their weights are built
on the host and mirrored, as today.

## Phase 2 progress ("weight-builders" track, 2026-10-13)

- **Found along the way:** `to_memory_space<Target>` (native_mesh.hpp) throws converting Host -> a
  real device space for any rank >= 2 `MeshArrays` field (`vertex_coords`), a layout mismatch between
  Host's and Cuda's default `Kokkos::View` layout -- never exercised that direction before this track
  (KNOWN_ISSUES.md's D54). Not fixed here (shared infrastructure, out of this track's scope); this
  track's own device test rows skip on it with a citation rather than reporting a false `pack_to`
  failure.
- **Halo/migration pack:** done as a Space-templated Kokkos kernel
  (`fuse::native::device::pack_to`, docs/dev/native-halo.md's own "Device packing" section) -- not yet
  the symmetric merge/unpack (dedup-by-global-id stays host; see that doc and KNOWN_ISSUES.md).
- **MLS weight assembly:** done as a Kokkos BATCHED kernel
  (`fuse::device::solve_mls_on_space`, include/fuse/backend/mls_weights_device.hpp) using
  `KokkosBatched::SerialGesv` from /opt/kokkos-kernels for the per-target `P x P` normal-equations
  solve, verified against a NumPy reference and against the real `fuse::search::BoxIndex` k-nearest
  result. Not yet wired into `build_mls_weights`'s own `RemapPlan`-level dispatch, and the
  `E-MLS-RANK` per-target diagnostic (rank, condition estimate) is coarsened on device to a failure
  count -- `SerialGesv` itself reports only pass/fail.
- **Supermesh intersection:** feasibility measured, not assumed (device-resident-arcs.md's own
  instruction for this step). r3d/r2d's call path used by `build_supermesh_weights` (`r3d_init_tet`/
  `r3d_clip`/`r3d_reduce`, `r2d_init_poly`/`r2d_clip`/`r2d_reduce`) has no recursion, no heap
  allocation, and only fixed-size stack buffers on the actual code path read -- algorithmically
  device-portable -- but is blocked today by three concrete, non-algorithmic obstacles: the vendored
  TUs are plain C with no `KOKKOS_FUNCTION` annotation, `r3d_clip`'s debug path calls
  `fprintf(stderr, ...)` under a `#if !defined(NDEBUG)` guard that fires in this track's Debug build,
  and the TUs are compiled as C, not C++ (nvcc identifies device code by language). The candidate-pair
  broad phase (the O(target-simplex x source-simplex) box test, "the honest starting point" in
  supermesh.cpp's own words) is accelerated as a device kernel
  (`fuse::device::candidate_pairs`, include/fuse/backend/supermesh_candidates_device.hpp), standing
  alongside `build_supermesh_weights` rather than inside it, so Host mode is untouched; the clip itself
  stays host-side pending the patch above.

## Phase 2 progress ("weight-builders" track, round 2, 2026-10-14)

Closes the wiring gap the round above left open for two of its three kernels, and narrows the third:

- **MLS weight assembly, wired in.** `build_mls_weights` (src/backend/mls_weights.cpp) now resolves
  its execution space via `fuse::search::resolve_execution_space()` (the same `FUSE_EXECUTION_SPACE`
  convention `RunnerConfig::resolved_execution_space()` and `fuse::search` itself already use) and, on
  Device, runs the k-nearest search through `BoxIndex::build`'s own Device path and the per-target
  solve through `fuse::device::solve_mls_on_space` -- reachable now through the normal
  `fuse::build_weights` entry point, not just as a standalone kernel. The round-1 note that
  `KokkosBatched::SerialGesv` backs the solve is corrected: this host's installed Kokkos/
  KokkosKernels pairing could not build either of `SerialGesv`'s pivoting modes (exact compiler
  errors in mls_weights_device.hpp's own header comment and KNOWN_ISSUES D52); the solve is an
  in-house partial-pivot Gaussian elimination instead, with no KokkosKernels dependency. The
  `E-MLS-RANK` diagnostic is no longer merely coarsened: `MlsDeviceResult::status` (one flag per
  target, a new field) lets `build_mls_weights`'s Device path re-run the Host path's own
  rank-revealing solve on exactly the target(s) flagged, confirming the verdict and reproducing the
  Host message (same target, same condition estimate) rather than reporting only a failure count.
  Host mode is bitwise unchanged (verified: every pre-existing `backend.mls_weights` row still passes
  under the default, no-`FUSE_EXECUTION_SPACE` environment). RED/GREEN:
  `MlsWeightsDevice.DeviceMatchesHostStructureAndValuesWithinTolerance` (1e-12 relative, identical
  CSR structure) and `MlsRefusalsDevice.AllCoincidentNeighboursIsRankDeficientUnderDeviceToo`
  (test/backend/mls_weights_test.cpp), plus a `backend.mls_weights.device` ctest twin re-running the
  whole file under `FUSE_EXECUTION_SPACE=device`.
- **Supermesh broad phase, wired in; clip still blocked as measured.** `build_supermesh_weights`
  (src/backend/supermesh.cpp) now dispatches its O(target-simplex x source-simplex) broad phase to
  `fuse::device::candidate_pairs` when the resolved space is Device AND the pair is 2D or 3D (1D stays
  on the existing brute force unconditionally: `candidate_pairs`' box test has no analogue of the 1D
  path's caller-chosen single-axis test, so using it there would silently filter on the wrong axis
  whenever that axis is not axis 0). The clip itself (`intersect`, r3d/r2d) is unchanged and stays
  host-only -- round 1's three build-system blockers (no `KOKKOS_FUNCTION` annotation on the vendored
  TUs, a debug-only `fprintf` not device-callable, C rather than C++ translation units) are unchanged
  findings, not re-measured, since nothing in this round touched r3d/r2d itself. Host mode is bitwise
  unchanged. RED/GREEN: `SupermeshDevice.ThreeDimensionalDeviceBroadPhaseMatchesHostExactly` (every
  triplet bit-identical, not merely within tolerance: dim >= 2's box test is IDENTICAL to the brute
  force's own, so the candidate set -- and the accumulation order into the triplet map -- match
  exactly) and `SupermeshDevice.OneDimensionalStaysOnHostBroadPhaseUnderDeviceToo`
  (test/backend/supermesh_test.cpp), plus a `backend.supermesh.device` ctest twin.
- **Halo/migration merge: the dedup DECISION is now device-resident; the full unpack is not.**
  `fuse::native::device::dedup_by_gid` (include/fuse/backend/native/mesh_merge_device.hpp, new file)
  implements the sort-and-compact algorithm round 1's entry named as the natural approach "not
  attempted here for lack of time" -- see docs/dev/native-halo.md's own "Device merge decision"
  section for the algorithm, its complexity, and the two gaps still open (Host-convention slot
  numbering; parsing a variable-length wire record's boundaries, which stays a host-side walk). A
  reusable primitive, not yet called by `merge_halo_cells`/`unpack_migrated_mesh` themselves.
  RED/GREEN: `test/backend/native_mesh_merge_device_test.cpp`, both Host- and Device-space
  instantiations, against a from-scratch `std::unordered_map`/`std::unordered_set` reference.
  **Superseded (Migration track, 2026-10-15, round 3):** CLOSED -- the renumbering pass and a
  wire-format-v2 header close both gaps this entry named, and `migrate`/`build_native_remap_halo` now
  call the device-resident unpack/merge end to end with no host mirror materialized across the full
  round trip. See docs/dev/native-halo.md's "Device merge decision" and KNOWN_ISSUES.md D51's closing
  text for the three narrow, declared exceptions that remain.

A hit found and fixed along the way, in this round's OWN new code rather than pre-existing
infrastructure: every `Kokkos::View<double* [3], Space>` this round constructs (MLS's device point
arrays, supermesh's device box-corner arrays) hit D54's exact defect class on a plain
`Kokkos::View<double* [3], Kokkos::HostSpace>` mirror -- a rank-2 view's PER-SPACE default layout
differs between Host and Cuda, so that mirror is layout-incompatible with the same-shape device view
and `Kokkos::deep_copy` cannot bridge both a memory-space and a layout change at once. `D54` itself
(native_mesh.hpp's own fields) was already fixed; this round's NEW views needed the same fix
(`Kokkos::create_mirror_view(device_view)`, which derives the mirror's layout from the view it
mirrors) applied to their own declarations, not inherited from D54's.

## Multi-rank (Phase 3)

**Done (transport track, 2026-10-13):** a run-time GPU-aware MPI probe
(`MPIX_Query_cuda_support()` where OpenMPI's extension declares it, with
`FUSE_TRANSPORT_HOST_STAGING=1` as the configuration override that forces
the fallback) now exists in two places: `fuse::native::mpi_is_gpu_aware()`
(src/backend/native_migration.cpp, already existed as a compile-time
constant, now a real probe) and `fuse::transport::MpiTransport::
is_gpu_aware()` (src/transport/mpi_transport.cpp, new), which consults
the identical logic DUPLICATED rather than shared -- `transport`'s
library does not link `backend`'s, discovered as a link error partway
through this track (the header-include direction
tools/check-layering.py enforces is not the CMake target link graph).

**Superseded (Migration track, 2026-10-15):** the duplication above is
gone -- `fuse::linalg::mpi_is_gpu_aware()` (`include/fuse/linalg/comm.hpp`,
`src/linalg/mpi_gpu_aware.cpp`) is the one copy now, in a layer BOTH
`backend` and `transport` sit above (`linalg` < ... < `backend` <
`partition` < `transport`), and both of the functions named above are
thin wrappers over it. The halo/migration exchange itself also gained a
GPU-aware REPLACEMENT this round: `fuse::native::exchange_migration_
buffers<Space>` (`Kokkos::View<std::byte*, Space>` buffers, device
pointers handed to `MPI_Alltoallv` directly when the probe says so),
proven byte-for-byte against the existing host-staged overload at 1-5
ranks -- but not yet called by `migrate`/`build_native_remap_halo`
(KNOWN_ISSUES.md D51's own remaining gap: `unpack_migrated_mesh`/
`merge_halo_cells` still need a host-resident buffer). See
docs/dev/native-halo.md's "GPU-aware exchange" section for the detail. Rank-to-device binding (`fuse::abi::resolve_device_id()`,
header-only and unit-tested with a mocked device count) is wired into
`ensure_kokkos_initialized()` and the test main: local rank within the node
modulo the visible device count, honouring an explicit
`--kokkos-device-id`/`KOKKOS_DEVICE_ID` override. The Runner-driving MPI
rows (`runner_transport`, `nodal_remap_runner`, `window_runner_mpi`,
`native_halo_mpi`) gained `device`-labelled twins at 1/2/3 ranks via
`fuse_add_mpi_test(... DEVICE_TWIN)`; on higgs's one GPU these ranks share
the device, which exercises correctness and the GPU-aware path, not
scaling -- true multi-GPU scaling is still unmeasured anywhere in this
tree.

## Reproducibility contract on the device

Dot and norm reductions are bit-identical to the host (D45). `RelaxFunctor`
and `RemapMultiFunctor` are not: nvcc contracts `a*b + c`, and
`runtime.runner_device` asserts agreement to 1e-14 relative for those shapes.
The owner accepted this on 2026-10-12; the TODO is recorded under D45 in
`KNOWN_ISSUES.md`. Any new device kernel added by this work follows the same
rule: bitwise where the host path is a reduction the determinism contract
covers, working precision elsewhere, and the test says which.

## Rollout

Each step below is its own commit (or small series), gated by the full
flag-off suite in `build-all` and the `fsi|gray_scott|sparse_solve_backend|
runner_device|space_kernels` regex in `build-cusparse`, both on higgs, before
the next step starts. Host mode byte-identity is the standing invariant.

1. DONE (9ac976c). Arcs that only Space-generic or space-aware
   instructions touch live in `DeviceMemSpace` (`Slot::device`,
   `Impl::device_storage`); `span_c`/`span_m` hand their pointers to the
   kernels directly; `RunStats::device_resident_arcs` counts them.
2. DONE (03c1ed7). `MemorySpace::Device` `FieldRef`s aliased in Device mode
   (refused under Host, `E-RT-SPACE`); a host field a kernel reads has a
   device mirror refreshed at the first read after component code ran
   (`Impl::component_epoch`, `note_component_ran()` at every call into a
   component) -- not at its `Source`, which is stamped `BeforeTicks` while
   the `Sample` that reads it runs `AfterTicks`; Sink/Epoch/Accelerate copy
   across spaces; `RunStats::device_boundary_copies` counts the crossings.
   Deviation: the example components (brusselator, gray_scott, fsi_channel)
   keep registering host fields -- their physics is Eigen on host arrays, so
   a device registration would only move the copy into the component.
   Device registration is exercised by `runtime.runner_device`'s
   `DeviceProducer`/`DeviceConsumer`.
3. DONE for ordinary iteration regions (6510f5e). `iteration.cpp` and
   `fixed_point_residual.cpp` reach block, carried and verify arcs only
   through `slot_to_host()`/`host_to_slot()`/`slot_host_read()`
   (`runner_impl.hpp`), so `place_arcs_on_device()` pins only the arcs a
   WINDOW-scoped region names (its bookkeeping still dereferences them on
   the host) and the space-time branches. Still host, by this rule:
   `previous[]` and the space-time history (staged per instruction).
   DONE (runtime-exchange track, 2026-10-14): `reduction_scratch` itself
   stays `HostSpace` (iteration.cpp's own block reductions are
   unconditionally host, a different track's scope), but Diagnose/
   Constrain's reductions and the distributed-nodal Remap no longer pin
   their arcs at all -- see the "Runtime-exchange track" section below.
4. DONE (4610b22, 581c5d1, 7b369f4). `FUSE_EXECUTION_SPACE` resolves
   `RunnerConfig::RunnerExecutionSpace::Default`; `fuse_add_device_twin()`
   registers the 14 `.device` rows (docs/dev/testing.md, "The device
   leg"); `RunStats::device_staged_copies` counts per-instruction staging.
   The twins found two defects on their first run: the staging lanes were
   sized after `resolve_regions()` had already executed a window's import
   chain (Interpolate faulted `E-RT-DEVICE` with zero-length lanes), and
   the FSI dominant-mode budget assumed a real eigenvector for the repeated
   eigenvalue -1 (the device run's rounding gave a complex basis vector; the
   test now takes its real part, which is an eigenvector of a real matrix
   at a real eigenvalue, and still validates it against the coupled run).
5. DONE in part (2026-10-13, three tracks, integrated at 8e3c8b5).
   Search: `fuse::search::BoxIndex` and `query_points_distributed` build
   and query ArborX on `Kokkos::DefaultExecutionSpace` under
   `FUSE_EXECUTION_SPACE=device` (device-resident `Kokkos::View` entry
   points for the local index; the distributed query still takes host
   vectors). Native mesh: `NativeMeshView::device_measure()`/
   `device_centroid()` are computed by kernels on the default execution
   space; validation and the simplex decomposition stay host (`Result`/
   `fmt` are not device-callable), `MeshArrays::vertex_coords` is
   `LayoutRight` in every space (D54). Weight builders: the migration/halo
   wire pack, a batched MLS solve and the supermesh candidate-pair broad
   phase existed as space-templated kernels with their own rows but were
   not yet wired into their production entry points (wired in step 7). Transport: `MpiTransport::is_gpu_aware()` is a run-time
   probe; the `Transport` case packs/unpacks on the device and hands MPI a
   device pointer when the MPI is GPU-aware, stages through host buffers
   otherwise (`FUSE_TRANSPORT_HOST_STAGING=1` forces that path); a mixed
   device/host arc pair is handled (D48); ranks bind to a device by local
   rank (`fuse::abi::resolve_device_id()`); `fuse_add_mpi_test(...
   DEVICE_TWIN)` registers 1/2/3-rank `.device` rows. Open: D49
   (allocation count of the mixed-pair transport row under the device leg).
6. Measured on higgs (Debug, GTX 1050 Ti, OpenMPI 4.1.5 -- built without
   CUDA support, so every GPU-aware MPI path ran its host-staged fallback;
   see D66), integrated gate 2026-10-13: host suite 295/296 (`-LE
   device`; the one failure is `example.brusselator.omp_identical` at its
   1500 s timeout, a load-sensitive row), device leg 28/28, cuSPARSE
   flag-on regex 15/15, anchors 308/308. The twins are correctness rows,
   not a speed claim: every problem here is a few hundred to a few thousand
   cells, so kernel launch latency and the remaining host stops dominate.
   Timings from that gate (same binaries, one process, not a quiet-machine
   measurement -- the host suite ran at load average 8):

| row | host (s) | `.device` (s) | ratio |
|---|---|---|---|
| `backend.native_distributed_mesh` | 0.35 | 0.39 | 1.11 |
| `example.brusselator.runner` | 56.62 | 70.06 | 1.24 |
| `example.dn_heat` | 251.28 | 388.44 | 1.55 |
| `example.gray_scott` | 69.04 | 64.79 | 0.94 |
| `example.mms_pair` | 1.86 | 0.21 | 0.11 |
| `parallel.native_halo_mpi.2` | 0.51 | 0.60 | 1.18 |
| `parallel.native_halo_mpi.3` | 0.64 | 0.71 | 1.11 |
| `parallel.nodal_remap_runner.1` | 0.36 | 0.36 | 1.00 |
| `parallel.nodal_remap_runner.2` | 0.44 | 0.51 | 1.16 |
| `parallel.nodal_remap_runner.3` | 0.62 | 0.67 | 1.08 |
| `parallel.runner_transport.1` | 0.35 | 0.35 | 1.00 |
| `parallel.runner_transport.2` | 0.53 | 0.65 | 1.23 |
| `parallel.runner_transport.3` | 0.66 | 0.93 | 1.41 |
| `parallel.window_runner_mpi.2` | 0.61 | 0.91 | 1.49 |
| `parallel.window_runner_mpi.3` | 0.79 | 2.58 | 3.27 |
| `search.box_index` | 0.38 | 0.72 | 1.89 |
| `search.mesh_adapter` | 0.35 | 0.41 | 1.17 |
| `search.refusals` | 0.34 | 0.45 | 1.32 |
| `verification.brusselator_runner` | 62.32 | 1330.89 | 21.36 |
| `verification.fsi_coupled` | 5.81 | 33.35 | 5.74 |
| `verification.fsi_energy_diag` | 37.97 | 41.94 | 1.10 |
| `verification.fsi_transient` | 67.41 | 496.51 | 7.37 |
| `verification.gray_scott_multirate` | 16.63 | 21.68 | 1.30 |
| `verification.gray_scott_reference` | 428.20 | 651.30 | 1.52 |
| `verification.interface_transfer` | 0.39 | 0.40 | 1.03 |
| `verification.iteration` | 0.51 | 11.32 | 22.20 |
| `verification.mms_transfer` | 0.41 | 0.79 | 1.93 |
| `verification.nodal_remap_runner_serial` | 0.36 | 0.37 | 1.03 |

   The two large ratios are the rows whose region is window-scoped
   (`verification.brusselator_runner`) or whose sweep loop is dominated by
   tiny per-sweep copies (`verification.iteration`): there every arc is
   still staged per instruction, which is step 3's remaining item (window
   bookkeeping, `previous[]`, history). A scaled problem is the right next
   measurement; the Runner's own kernels are not the bottleneck at these
   sizes.

7. DONE (round 2, 2026-10-15, five tracks integrated at a97489aa on main
   35229b28). Weight builders: `build_mls_weights` runs its k-nearest
   search and per-target solve on the resolved execution space (D52
   closed); `build_supermesh_weights` runs the candidate-pair broad phase
   on the device for dim >= 2, the r3d/r2d clip stays host (D53). Native
   mesh: `NativeMeshView` is device-primary (`MeshArrays` in
   `DeviceMemSpace`; validation once on a transient host mirror; every
   host accessor builds and caches its mirror on first use, counted by
   `host_mirrors_materialized()`); the simplex decomposition is retained
   as a device CSR and `fuse::search::element_boxes_device_resident`
   feeds `BoxIndex::build_device_resident` from it with no host round
   trip. Halo and migration: pack (`pack_to`/`pack_fanout`), exchange
   (`exchange_migration_buffers<Space>`) and the receive-side merge
   (`dedup_by_gid`, renumbering, face-CSR gather) run on the device end
   to end with a self-describing wire format; a full halo or migration
   round materialises no host mirror (D51 closed). Runtime: Constrain's
   conservation sum and Diagnose's non-finite check are reductions on
   the arc's own space; the distributed-nodal Remap resolves its
   exchange once at setup (`build_distributed_remap_plan`) and runs its
   per-period gather and H^T scatter as Kokkos kernels around one
   `MPI_Alltoallv`, the scatter summing each target's contributions in
   contributor-global-id order so the result is bit-identical across
   rank counts; D49 closed with a per-rank capture. Exchange layer:
   `fuse::linalg` stays Kokkos-free -- `PackedExchanger` exposes the MPI
   exchanger's pack/unpack maps and a raw `MPI_Alltoallv`, and
   `fuse::backend::DeviceGhostExchanger` drives it with Kokkos kernels;
   one probe, `fuse::linalg::mpi_is_gpu_aware()`, decides device pointers
   versus host staging for Transport, the ghost exchange, the nodal
   remap and migration (`FUSE_TRANSPORT_HOST_STAGING=1` forces staging).
   Gate on higgs at a97489aa: build clean; anchors 355/355; invariants
   and layering 0 violations; `ctest -j1 -LE device` 333/335 (D55, D56
   only); `ctest -j1 -L device` 35/35. Caveat: higgs's Open MPI is not
   CUDA-aware, so the device-pointer branch of every exchange is compiled
   and reachable but ran only its host-staged fallback here (D66). Still
   host: `previous[]`, space-time history and window-region bookkeeping
   (staged per instruction); MOAB-backed meshes and the mesh file reader;
   the r3d/r2d clip; consistent/general interpolation and Tempest weights;
   `DistVector` and the component-internal linear algebra; accelerators
   and Newton; `MeshView::measure()` has no device-callable form for the
   nodal remap's H^T input scaling.

## What the placement trace shows today

`FUSE_DEVICE_DEBUG=1 ./fuse_runtime_runner_device_test` on the mean-over-four-
substeps chain prints `Sample -> Accumulate -> Transport -> Epoch -> Remap ->
Diagnose -> Sink`: every arc in the chain is now device-resident, the Sample
output and the Epoch output included, and (runtime-exchange track,
2026-10-14) `Diagnose`'s own arcs as well -- nothing in this plan pins
anything anymore. A fully device-registered chain therefore crosses the
boundary ZERO times per period and stages nothing through a `device_staging`
lane (`DeviceRegisteredFieldsCrossTheBoundaryZeroTimesAndAgreeWithHost` pins
both counts at 0). This is down from one crossing (the Sink, host ->
device) after the transport track and two before it.

## Runtime-exchange track (2026-10-14)

**Done.** `OpKind::Constrain`'s conservation sum and `OpKind::Diagnose`'s
non-finite check no longer pin their arcs to host: both read/write through
`span_c()`/`span_m()` like every other Space-generic case, and the
reduction itself runs on the arc's own space --
`fuse::num::deterministic_dot<Space>` against a device mirror of
`weight_sets` (`weight_sets_device`, built once in
`build_constrain_device_buffers()`) for Constrain, `fuse::num::
any_nonfinite<Space>` for Diagnose -- with only the scalar verdict (and,
for Constrain, the cross-rank `MPI_Allreduce`) staying on host. Constrain's
`before_arc` is not one of its own instruction's `in`/`out` arcs (it is the
Remap's own input, on the source region), so it is read through a small
local `span_any_c()` helper keyed on the `Slot` directly rather than
`span_c()`'s own ins.in/ins.out identity lookup; the same device-pointer-
or-stage-it decision either way.
`apply_nodal_dist_remap()` no longer has its arcs pinned either: its own
case now stages a device-resident `in`/`out` slot through one
`slot_to_vector()`/`host_to_slot()` deep_copy at its own entry/exit (the
same helpers `iteration.cpp`'s sweep loop already uses), around an MPI
exchange that still runs entirely on host buffers.

**SUPERSEDED by "Runtime-exchange track, follow-up round (2026-10-15)" below** (the lead's layering decision resolved the first two of these three items; kept verbatim as the historical record of why they were blocked, not because they still are).

**Why the nodal-dist and ghost-field exchanges themselves stay host
buffers, not GPU-aware MPI.** Two separate, pre-existing architectural
decisions block a literal "hand MPI a device pointer" implementation for
the other two pieces this track's instructions named, and neither is this
track's call to overturn:

- `apply_distributed_H`/`apply_distributed_HT` (`nodal_remap_distributed.cpp`)
  exchange wire records that INTERLEAVE a global id with every value
  (`allgatherv_records`, stride doubles per record), rebuilt from an
  `unordered_map` lookup every period -- unlike `Transport`'s `CommGraph`,
  whose index resolution is a ONE-TIME setup cost and whose per-step
  payload is a plain value array. There is no contiguous device buffer to
  hand `MPI_Allgatherv` a pointer to without first splitting index
  resolution out of the per-period path, the same two-phase shape
  `fuse::linalg::make_mpi_exchanger` already uses for ghost fields -- a
  larger rework than stage-and-exchange, out of this track's scope.
- `fuse::linalg::GhostExchanger` (`comm.hpp`) is deliberately isolated from
  Kokkos (`comm.hpp`'s own header comment: "linalg includes only base"),
  the same boundary `tools/check-layering.py` enforces for every other
  `fuse::linalg` header. Its `update()` already takes raw `const double*`/
  `double*` (space-agnostic at the type level), but `MpiExchanger::update`'s
  pack/unpack loops (`comm_mpi.cpp`) dereference them on the host; making
  that a device kernel needs a Kokkos include this header does not carry
  today. Adding one is a layering decision for the owner, not something
  this track changes unilaterally.
- `fuse::native::exchange_migration_buffers()` (`native_migration.hpp`)
  USED TO state its own policy as "this file's own migration exchange
  always stages through host buffers regardless, per the standing
  priority (robustness before efficiency)" -- superseded (Migration track,
  2026-10-15): `NativeMeshView` is now device-primary (as this section's
  own gating condition anticipated), `pack_for_migration`/`pack_halo_cells`
  read `mesh.device_arrays()` directly with no host-only accessor touched,
  and a GPU-aware `exchange_migration_buffers<Space>` overload exists and
  is proven standalone. The ORIGINAL `std::vector` overload this bullet
  described is still what `migrate`/`build_native_remap_halo` call,
  because `unpack_migrated_mesh`/`merge_halo_cells` had not yet been
  updated to accept a device-resident buffer (KNOWN_ISSUES.md D51) --
  **superseded (round 3, 2026-10-15):** both now delegate to the
  device-templated `unpack_migrated_mesh_device`/`merge_halo_cells_device`,
  and `migrate`/`build_native_remap_halo` call the `Space`-templated
  `exchange_migration_buffers<Space>` overload directly; see
  docs/dev/native-halo.md for the closed state.

**Lead must know.** The GhostExchanger and migration-exchange blockers
above are findings, not fixes -- report-runtime-exchange.md states them as
deviations. Overturning either standing decision is the owner's call.

## Runtime-exchange track, follow-up round (2026-10-15)

**Lead's layering decision, acted on.** `fuse::linalg` stays Kokkos-free
(comm.hpp's own header comment is unchanged); the GPU-aware probe and the
device-side packing both moved to the layers the lead named:

- `fuse::linalg::mpi_is_gpu_aware()` (comm.hpp/comm_mpi.cpp) is now the
  ONE probe `fuse::transport::MpiTransport::is_gpu_aware()` calls (it
  used to carry its own private copy, `transport_mpi_is_gpu_aware()`,
  deleted this round) and the one `fuse::backend::DeviceGhostExchanger`
  and the nodal-remap plan below also call -- `linalg` sits below both
  `transport` and `backend` (tools/check-layering.py), so this is the
  lowest layer either can include it from. `FUSE_TRANSPORT_HOST_STAGING=1`
  still forces it `false`; every new MPI row below is tested under both
  settings.
- `fuse::linalg::PackedExchanger` (comm.hpp), implemented by
  `MpiExchanger` (comm_mpi.cpp) alongside `GhostExchanger`: exposes the
  pack/unpack maps and counts/displacements `GhostExchanger::update()`
  already derives once, plus two raw `MPI_Alltoallv` primitives
  (`exchange_packed()`, and its adjoint `exchange_packed_adjoint()` for
  a scatter-add) over caller-packed buffers. `GhostExchanger::update()`'s
  own signature and contract are UNCHANGED -- reached with
  `std::dynamic_pointer_cast<const PackedExchanger>`, `nullptr` for the
  serial exchanger.
- `fuse::backend::DeviceGhostExchanger` (new,
  `include/fuse/backend/device_ghost_exchange.hpp`, header-only,
  templated on `Space`): drives `PackedExchanger` with Kokkos pack/unpack
  kernels, handing MPI a device pointer directly when `Space` is a device
  space and `mpi_is_gpu_aware()`, host-staging (deep_copy around the same
  `exchange_packed()` call) otherwise. `update()` is the forward (gather)
  direction; `update_adjoint()` is the scatter-add the nodal-remap H^T
  direction needs. NOT wired into an existing production caller --
  `fuse::linalg::DistVector` (the one existing `GhostExchanger::update()`
  caller) is plain host `std::vector` by design, and
  `NativeMeshBackend::remap_halo_pattern` only builds a `GhostPattern`,
  never calls `update()` on it itself (that is
  `examples/brusselator/species_core.hpp`'s own example code) -- this is
  new infrastructure for the next device-resident consumer, proven on
  its own terms (`test/parallel/device_ghost_exchange_test.cpp`, a ring
  pattern at 1-5 ranks, `.device` twins at 1-3, both
  `FUSE_TRANSPORT_HOST_STAGING` settings).

**The nodal-distributed-remap exchange is now a one-time resolved plan.**
`fuse::build_distributed_remap_plan` (nodal_remap_distributed.hpp/cpp,
new) resolves, ONCE at setup (alongside `build_distributed_interp_rows`,
only when `comm_handle` names a REAL communicator -- a single-rank run
keeps the unchanged `apply_distributed_H`/`apply_distributed_HT` path,
below), every row's `src_global_ids` to (owner rank, owner's own local
position) via one additional one-time `MPI_Allgatherv`, then builds a
component-expanded `fuse::linalg::GhostPattern` and its `MpiExchanger`
exactly the way `make_mpi_exchanger` already does for ghost fields. The
per-period cost drops from "replicate the whole source mesh and rebuild
an `unordered_map`" (the previous round's own blocker paragraph, deleted
below) to one `MPI_Alltoallv` of a plain value array, gather/scattered
with Kokkos kernels (`Runner::Impl::nodal_dist_gather_on`/
`nodal_dist_scatter_on`, runner.cpp) on whichever `Space` the arc already
lives in.

A SECOND exchanger (`DistributedRemapPlan::ghost_fill_exchanger`) turned
out to be required, not optional: `apply_distributed_HT`'s own contract
overwrites every LOCAL row -- owned or ghost -- with the same global
total for its global id, and the plan's primary exchanger only
scatter-adds into each vertex's OWNER. Found the hard way (a real host
MPI run, `NodalRemapRunnerMpi.DisjointRankSets` at 3 ranks: rank 0's own
assertions passed, rank 1's did not -- its one GHOST entry of a vertex
rank 0 owns stayed 0 instead of the broadcast total; every OWNED entry on
every rank was already correct). `ghost_fill_exchanger` is a second,
component-expanded `GhostPattern` over every LOCAL ghost entity of the
nodal mesh, reusing the SAME global ownership map the primary pattern's
derivation already computed -- a second cheap pattern build, not a second
Allgatherv.

**A second, independent bug found and fixed before any of this reached a
real multi-rank test: a per-rank early return cannot guard a collective.**
`DeviceGhostExchanger::update()`'s first draft returned early when THIS
rank's own `ghost_size() == 0`, reasoning (wrongly) that an exchange with
nothing to move is a no-op. `ghost_size()` is a PER-RANK count; one rank
having zero local ghosts while a PEER has some is the ordinary case for
the nodal-remap pattern specifically (a rank that owns no evaluation
entities for a channel has `n_rows == 0`, hence zero slots, while a peer
that does still needs its own `MPI_Alltoallv`), and every rank of the
communicator must still enter that collective, whatever ITS own counts
are -- `MpiExchanger::update()` itself (comm_mpi.cpp) never special-cases
a zero count for exactly this reason. The early return deadlocked
`parallel.nodal_remap_runner` at every rank count above 1 (host run,
measured: every rank stuck in `MPI_Alltoallv`, confirmed via a per-rank
`mpiexec --output-filename` trace showing no rank reaching the next
output after `[ RUN ]`). Removed; a zero-count `MPI_Alltoallv` entry is a
well-defined, cheap no-op MPI already handles, the same reason the
pack/unpack Kokkos kernels this round adds are never guarded by a
`n == 0` check either (a `RangePolicy(0, 0)` `parallel_for` is simply
zero iterations).

**Still host-only, now narrowed to one real limitation instead of three.**
Of the previous round's three blockers, two are resolved by this round
(ghost exchange, nodal-remap exchange); the third --
`fuse::native::exchange_migration_buffers()` -- is explicitly OUT of this
round's scope (owned by the separate Migration track, a different
worktree; its own header comment's "always stages through host buffers
regardless, per the standing priority" policy sentence is left UNCHANGED,
not rewritten, since this round touches neither that file nor its
caller). H^T's own INPUT gather (the eval-side value times the eval
side's own cell measure) still runs on host regardless of `Space`:
`MeshView::measure()` has no device-callable form, a pre-existing
limitation of the mesh interface this round's scope does not change.

**Hardware cannot verify the device-direct branch.** `mpi_is_gpu_aware()`
is measured `false` unconditionally on higgs (KNOWN_ISSUES.md D66):
OpenMPI 4.1.5 here was configured `--without-cuda`, so
`MPIX_Query_cuda_support()` returns 0 at run time even though the
extension macro is declared at compile time. Every new MPI row this
round adds is verified correct via the host-staged fallback (always
taken on this hardware) under both the natural and forced
`FUSE_TRANSPORT_HOST_STAGING=1` settings; the device-direct branch
(handing MPI a device pointer with no host round trip) compiles and is
reachable in the code, but no host run here can confirm it moves data
correctly with a real device pointer in MPI's own hands.

**The reproducibility trade above, recovered (2026-10-15 follow-up).**
The original nodal-remap exchange's own design (nodal_remap_distributed.hpp's
header comment) proves its H^T total "bit-identical across rank counts" by
summing every contribution in one global, rank-count-independent order
(sorted by contributor global id) after an `MPI_Allgatherv` of the WHOLE
term list. This round's FIRST submission's plan-based scatter-add used
`Kokkos::atomic_add` into each owner's own position instead -- correct,
but summing in whatever order messages/threads happened to arrive, NOT
reproducible across rank counts. The lead asked for the stronger
guarantee back rather than accepting that trade: `build_distributed_remap_plan`
now resolves, ONCE at setup, a CSR (`DistributedRemapPlan::target_row_offset`/
`sorted_answer_slot`) grouping every incoming contribution by its target
local position and sorting each group ascending by the contributor
evaluation entity's GLOBAL id -- the identical key `apply_distributed_HT`'s
own per-call sort already uses, resolved by one EXTRA one-time adjoint
exchange of a `double`-encoded gid array (reusing the exact same
`PackedExchanger::exchange_packed_adjoint()` plumbing the VALUE exchange
itself uses) rather than a fresh Allgatherv of the whole term list every
period. The per-period H^T kernel (`nodal_dist_scatter_reduce_
deterministic_kernel`, replacing `DeviceGhostExchanger::update_adjoint()`'s
own atomic-add for this one caller) sums each target's bucket
sequentially in that fixed order, one thread per target, no atomics.
`parallel.nodal_remap_runner.ConsistentTransposeIsBitIdenticalAcrossRankCounts`
(new) asserts EXACT equality against the one-rank serial reference at
every rank count this file runs at, tightened from that file's usual
1e-12 relative check specifically to catch a regression back to arrival-
order summation; GREEN on both the host and `.device` rows (host and
device multiply/sum steps here have no `a*b+c` fusion opportunity in
source, unlike the Relax/Remap kernels `docs/dev/runner-device-dispatch.md`
already notes are NOT bit-identical to host for that reason, so this
claim does not inherit that caveat). `apply_distributed_H`/
`apply_distributed_HT` themselves, used at `comm_handle ==
fuse::search::kNullComm`, were never affected (they keep their own
original guarantee throughout).
