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
| `weight_sets` (Constrain measures) | host vectors | host; `Constrain` integrates on the arc space and reduces, result to host |
| `NodalDistRemap` buffers | host, MPI exchange | host; `apply_nodal_dist_remap()` stages its arcs explicitly (it is an MPI exchange like Transport) |
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
   for whichever side was device. Still unconditionally host for
   `apply_nodal_dist_remap()` (a different track's scope);
3. `Epoch` (status reconciliation is MPI on scalars; the value copy itself runs
   on the arc space), `Diagnose` (`has_nonfinite` becomes a reduction on the
   arc space; only the verdict is host), `Constrain` (reduction result);
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
The halo/migration exchanges (`fuse::native::exchange_migration_buffers`)
do not consult their copy yet -- that stays host-only, per the standing
priority, and is a different track's scope. Rank-to-device binding (`fuse::abi::resolve_device_id()`,
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
   the host), a Constrain's `before` arc, and the arcs of host-only
   instructions (Transport, Diagnose, Constrain, nodal Remap, the
   space-time branches). Still host, by this rule: `previous[]` and the
   space-time history (staged per instruction), `reduction_scratch`,
   Diagnose/Constrain reductions.
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
   phase exist as space-templated kernels with their own rows, but none is
   wired into its production entry point yet (D51-D53 name what blocks
   each: the receive-side merge, the MLS rank diagnostic, the r3d clip's
   packaging). Transport: `MpiTransport::is_gpu_aware()` is a run-time
   probe; the `Transport` case packs/unpacks on the device and hands MPI a
   device pointer when the MPI is GPU-aware, stages through host buffers
   otherwise (`FUSE_TRANSPORT_HOST_STAGING=1` forces that path); a mixed
   device/host arc pair is handled (D48); ranks bind to a device by local
   rank (`fuse::abi::resolve_device_id()`); `fuse_add_mpi_test(...
   DEVICE_TWIN)` registers 1/2/3-rank `.device` rows. Open: D49
   (allocation count of the mixed-pair transport row under the device leg).
6. Measured on higgs (Debug, GTX 1050 Ti, OpenMPI 4.1.5 reporting
   GPU-aware), integrated gate 2026-10-13: host suite 295/296 (`-LE
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

## What the placement trace shows today

`FUSE_DEVICE_DEBUG=1 ./fuse_runtime_runner_device_test` on the mean-over-four-
substeps chain prints `Sample -> Accumulate -> Transport -> Epoch -> Remap ->
Diagnose -> Sink`: the Transport pins `%accumulate`/`%transport`, the Diagnose
pins `%remap`/`%diagnose`, the Sample output and the Epoch output are device
arcs, the producer's field is mirrored. The two host-only instructions are
what Phase 3 (Transport) and the deferred Diagnose reduction remove; until
then a fully device-registered chain crosses the boundary twice per period
(Epoch host -> device, Sink host -> device field) and stages the two kernel
outputs those instructions read (`DeviceRegisteredFieldsCrossOnlyAtHostOnly
InstructionsAndAgreeWithHost` pins both counts).
