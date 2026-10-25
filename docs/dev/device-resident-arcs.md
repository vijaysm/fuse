# Device-resident execution: arcs, mesh, topology and transport

This page describes how `RunnerConfig::execution_space == Device` keeps the coupler's compute on
`Kokkos::DefaultExecutionSpace` (Cuda when the build has it) with no host round trips inside the
sweep. It supersedes the staging design of [](runner-device-dispatch.md) Part A for Device mode.
Host mode is unchanged by anything below and stays byte-identical.

## Why staging was not enough

The staging design proved the dispatch pattern: nine `OpKind` cases launch on `Kokkos::Cuda`
through a four-lane `device_staging` buffer, so every device-dispatched instruction pays a
host-to-device copy of its inputs and a device-to-host copy of its output. That path is slower than
the host path at every size. Device mode instead runs the coupler's compute on the default
execution space with no host round trips inside the sweep, makes the mesh and topology operations
device-capable, and has the infrastructure for multi-rank device runs in place.

## Buffer inventory and where each lives

| buffer | Host mode | Device mode |
|---|---|---|
| `Impl::storage[i]` (coupler-owned arcs), `Slot::data` | `std::vector<double>`, raw host pointer | `Kokkos::View<double*, ArcMemSpace>` per arc (`Impl::device_storage`); `Slot` carries the space; `ArcMemSpace = DeviceMemSpace` iff `execution_space == Device` |
| component fields aliased as arcs (`owned_by_component`) | host `FieldRef`; `MemorySpace::Device` refused `E-RT-SPACE` | a `MemorySpace::Device` `FieldRef` aliases directly; a host `FieldRef` gets one device mirror, refreshed the first time a kernel reads it after component code ran (`Impl::component_epoch`, `note_component_ran()` at every call into a component), and copied out at its `Sink` |
| `previous[]` (Relax/Residual, read one sweep late by `advance_window()`) | host vectors | host vectors, plus a device-resident companion (`previous_device`) for a Relax/Residual output arc that is itself device-resident; `sync_previous()` gives every commit write the same device/host dispatch `host_to_slot()` gives a plain value write |
| `SpacetimeInfo::history` | host vector | `history_device` (`Kokkos::View<double*, DeviceMemSpace>`, the same zero-filled, never-resized ring), built once in `resolve_spacetime()` |
| `reduction_scratch` | `HostSpace` view, norms in `iteration.cpp` on `HostExecSpace` | stays `HostSpace`: the block reductions in `iteration.cpp` are host |
| `RegionRuntime::x_old/x_new/x_out`, accelerator state | host vectors, `Accelerator` host-only | stay host; the gather `slots[b.feed_arc] -> x_old` and scatter back are `deep_copy`s per sweep over the Newton/Picard interface vector (small by construction: interface dofs, not fields) |
| transport `send_buf/recv_buf` | host vectors, `MPI_Alltoallv` | `Impl::transport_device` (device views, built once when both of a Transport instruction's arcs land on the device) when `Transport::is_gpu_aware()` is true at run time; otherwise the pack/unpack kernels still run on the device and only the host `send`/`recv` buffers stage the `MPI_Alltoallv`. A mixed device/host arc pair occurs for real (a Remap's conservation check pins one of them): the device side stages through `Impl::transport_in_scratch`/`transport_out_scratch` and the ordinary host loop runs. See "Transport and the packed layout" |
| `RemapCsr` / `remap_csr_device` | host triplets + device twin built at resolve | device CSR is the primary; builders may produce it directly |
| `weight_sets` (Constrain measures) | host vectors | host (`weight_sets` never changes after setup); `weight_sets_device` is a device-resident mirror built once in `build_constrain_device_buffers()`; `Constrain` integrates on the arc's own space via `fuse::num::deterministic_dot` into `constrain_reduction_scratch`, result to host |
| `NodalDistRemap` buffers | host, MPI exchange | see "The distributed nodal remap exchange" |
| checkpoint payloads, `verify_save`, Newton seam snapshots (`fixed_point_residual.cpp`) | host copies of slots | `deep_copy` from the arc space into the existing host containers |

## The boundary rule

Host memory is touched in Device mode only at:

1. a host-registered component field: one mirror refresh per component step that a kernel reads it
   after (a `Source` is stamped `BeforeTicks`, the `Sample` that reads the field `AfterTicks`, so a
   refresh AT the `Source` is one tick stale -- the mirror is refreshed at the read instead), and
   one copy out at its `Sink`;
2. `Transport` when `Transport::is_gpu_aware()` is false at run time (the pack/gather and
   unpack/scatter kernels still run on the device; only the already-packed buffer round-trips to
   host around the `MPI_Alltoallv` itself -- two `device_boundary_copies` per exchange, not a
   whole-field copy), or when `FUSE_TRANSPORT_HOST_STAGING=1` forces that path on a GPU-aware MPI so
   the fallback has a test everywhere; one crossing in each direction either way. Also `Transport`
   when its two arcs are a mixed device/host pair: the device-resident side stages through a host
   scratch buffer and the ordinary scalar host loop runs against two host pointers, one crossing for
   whichever side was device. `apply_nodal_dist_remap()` for a single-rank run is the same shape: a
   device-resident `in`/`out` arc is staged through one `slot_to_vector()`/`host_to_slot()`
   deep_copy at the function's entry/exit, around an MPI exchange that stays on host buffers;
3. `Epoch` (status reconciliation is MPI on scalars; the value copy itself runs on the arc space),
   `Diagnose` (`has_nonfinite` is a reduction on the arc space via `any_nonfinite<Space>`; only the
   verdict is host), `Constrain` (the conservation sum is a reduction on the arc space via
   `deterministic_dot` against `weight_sets_device`; only the scalar verdict is host);
4. the accelerator interface vector and the window-member bookkeeping (`advance_window`'s clocks and
   the straddle state, which are scalars);
5. checkpoint/resume and the Newton seam's save/restore.

Everything else -- every elementwise `OpKind`, the norms, Relax's `previous[]`, the spacetime
history -- runs where the arcs live. The host-only `OpKind`s that remain are exactly the ones listed
in `execute_on()`'s comment; none of them moves data elementwise except through `num::copy`, which
becomes `num::copy<Space>` on the arc space.

Arcs reach host code only through `slot_to_host()`/`host_to_slot()`/`slot_to_vector()`/
`slot_host_read()`/`scalar_to_slot()` (`runner_impl.hpp`), never through a raw `Slot::data`
dereference; `scalar_to_slot()` covers the handful of writes that land at a stride other than the
arc's own start. `place_arcs_on_device()` pins an arc to the host only when a host-only instruction
names it. A window-scoped region's own arcs are not pinned: `advance_window()`,
`refresh_window_member_history()`, `resolve_pending_window_straddles()` and `run_region()`'s per-sweep
lagged-import snapshot reach them through those helpers. `OpKind::WindowAdvance` is in
`place_arcs_on_device()`'s space-aware case list.

`place_arcs_on_device()` copies `storage[i]` into `device_storage[i]` before releasing the host
vector. That copy is a no-op for an arc that still holds its cold-start zero, but required for a
window-scoped region's arc: `resolve_regions()`'s "priming through the transfer" pass writes a
member's real initial condition into its export arc once, before placement runs, and nothing
rewrites it later. A zero-filled device allocation would silently replace it.

## Staging that remains

`device_staging` is retained for a host-registered component field that a kernel WRITES (a mirror is
read-only; the write has to reach the host array) and for a host arc pinned by a host-only
instruction. `RunStats::device_dispatches` counts device launches, `RunStats::device_boundary_copies`
counts boundary crossings so tests can assert the no-round-trip property directly, and
`RunStats::device_staged_copies` counts per-instruction staging. `E-RT-DEVICE` is the staging
path's fault code.

`FUSE_DEVICE_DEBUG=1 ./fuse_runtime_runner_device_test` on the mean-over-four-substeps chain prints
`Sample -> Accumulate -> Transport -> Epoch -> Remap -> Diagnose -> Sink`: every arc in the chain is
device-resident, including the Sample output, the Epoch output and `Diagnose`'s arcs. A fully
device-registered chain therefore crosses the boundary ZERO times per period and stages nothing
through a `device_staging` lane
(`DeviceRegisteredFieldsCrossTheBoundaryZeroTimesAndAgreeWithHost` pins both counts at 0).

Example components (brusselator, gray_scott, fsi_channel) keep registering host fields -- their
physics is Eigen on host arrays, so a device registration would only move the copy into the
component. Device registration is exercised by `runtime.runner_device`'s
`DeviceProducer`/`DeviceConsumer`.

## Execution space selection and the device twins

`FUSE_EXECUTION_SPACE` resolves `RunnerConfig::RunnerExecutionSpace::Default`; a caller that names
`Host` or `Device` explicitly overrides it. `fuse_add_device_twin()` registers the `.device` rows
([](testing.md), "The device leg") and `fuse_add_mpi_test(... DEVICE_TWIN)` registers 1/2/3-rank
`.device` rows for the Runner-driving MPI tests (`runner_transport`, `nodal_remap_runner`,
`window_runner_mpi`, `native_halo_mpi`). On a machine with one GPU the ranks share the device, which
exercises correctness and the GPU-aware path, not scaling; true multi-GPU scaling is unmeasured.

Ranks bind to a device by local rank within the node modulo the visible device count
(`fuse::abi::resolve_device_id()`, header-only and unit-tested with a mocked device count), honouring
an explicit `--kokkos-device-id`/`KOKKOS_DEVICE_ID`. It is wired into `ensure_kokkos_initialized()`
and the test main.

## Mesh and topology

`MeshArrays<MemorySpace>` defaults to the default execution space's memory. `NativeMeshView` is
device-primary: `MeshArrays` live in `DeviceMemSpace`, validation runs once on a transient host
mirror, and every host accessor builds and caches its mirror on first use (counted by
`host_mirrors_materialized()`). `device_measure()`/`device_centroid()` are kernels on the default
execution space; validation and the simplex decomposition stay host (`Result`/`fmt` are not
device-callable). The simplex decomposition is retained as a device CSR, and
`fuse::search::element_boxes_device_resident` feeds `BoxIndex::build_device_resident` from it with no
host round trip. `MeshArrays::vertex_coords` is `LayoutRight` in every space: a rank-2 view's
per-space default layout differs between Host and Cuda, which made `to_memory_space<Target>` throw
converting Host to a real device space. Any new `Kokkos::View<double*[3], Space>` mirror must be
created with `Kokkos::create_mirror_view(device_view)`, which derives the mirror's layout from the
view it mirrors, because `Kokkos::deep_copy` cannot bridge a memory-space and a layout change at
once.

Search: `fuse::search::BoxIndex` and `query_points_distributed` build and query ArborX on
`Kokkos::DefaultExecutionSpace` under `FUSE_EXECUTION_SPACE=device` (device-resident `Kokkos::View`
entry points for the local index; the distributed query takes host vectors).

MOAB-backed meshes stay host-resident; their weights are built on the host and mirrored.

### Weight builders

- **MLS.** `build_mls_weights` (`src/backend/mls_weights.cpp`) resolves its execution space via
  `fuse::search::resolve_execution_space()` (the same `FUSE_EXECUTION_SPACE` convention
  `RunnerConfig::resolved_execution_space()` uses) and, on Device, runs the k-nearest search through
  `BoxIndex::build`'s Device path and the per-target solve through `fuse::device::solve_mls_on_space`
  (`include/fuse/backend/mls_weights_device.hpp`). The solve is an in-house partial-pivot Gaussian
  elimination with no KokkosKernels dependency: the installed Kokkos/KokkosKernels pairing could not
  build either of `KokkosBatched::SerialGesv`'s pivoting modes. `MlsDeviceResult::status` (one flag
  per target) lets the Device path re-run the Host path's rank-revealing solve on exactly the
  flagged targets, so `E-MLS-RANK` reports the same target and condition estimate as on the host.
  Tests: `MlsWeightsDevice.DeviceMatchesHostStructureAndValuesWithinTolerance` (1e-12 relative,
  identical CSR structure), `MlsRefusalsDevice.AllCoincidentNeighboursIsRankDeficientUnderDeviceToo`,
  and a `backend.mls_weights.device` ctest twin.
- **Supermesh.** `build_supermesh_weights` (`src/backend/supermesh.cpp`) dispatches its
  O(target-simplex x source-simplex) broad phase to `fuse::device::candidate_pairs`
  (`include/fuse/backend/supermesh_candidates_device.hpp`) when the resolved space is Device and the
  pair is 2D or 3D. 1D stays on the brute force unconditionally: `candidate_pairs`' box test has no
  analogue of the 1D path's caller-chosen single-axis test, so using it there would silently filter
  on the wrong axis whenever that axis is not axis 0. For dim >= 2 the box test is identical to the
  brute force's, so the candidate set and the accumulation order into the triplet map match exactly
  (`SupermeshDevice.ThreeDimensionalDeviceBroadPhaseMatchesHostExactly`,
  `SupermeshDevice.OneDimensionalStaysOnHostBroadPhaseUnderDeviceToo`). The clip itself (`intersect`,
  r3d/r2d) stays host-only. The call path (`r3d_init_tet`/`r3d_clip`/`r3d_reduce`,
  `r2d_init_poly`/`r2d_clip`/`r2d_reduce`) has no recursion, no heap allocation and only fixed-size
  stack buffers, so it is algorithmically device-portable, but three build-system obstacles block it:
  the vendored TUs are plain C with no `KOKKOS_FUNCTION` annotation, `r3d_clip`'s debug path calls
  `fprintf(stderr, ...)` under `#if !defined(NDEBUG)`, and the TUs are compiled as C, not C++ (nvcc
  identifies device code by language).
- **Halo and migration.** Pack (`fuse::native::device::pack_to`/`pack_fanout`), exchange
  (`exchange_migration_buffers<Space>`) and the receive-side merge (`dedup_by_gid`, the renumbering
  pass, face-CSR gather) run on the device end to end with a self-describing wire format; a full halo
  or migration round materialises no host mirror. See [](native-halo.md), "Device merge decision" and
  "GPU-aware exchange", for the algorithm and the declared exceptions.

## Transport and the packed layout

A run-time GPU-aware MPI probe (`MPIX_Query_cuda_support()` where Open MPI's extension declares it,
with `FUSE_TRANSPORT_HOST_STAGING=1` as the configuration override that forces the fallback) is
`fuse::linalg::mpi_is_gpu_aware()` (`include/fuse/linalg/comm.hpp`, `src/linalg/mpi_gpu_aware.cpp`).
It lives in `linalg` because `linalg` sits below both `backend` and `transport`
(`linalg` < ... < `backend` < `partition` < `transport`): `fuse::native::mpi_is_gpu_aware()` and
`fuse::transport::MpiTransport::is_gpu_aware()` are thin wrappers over it, and the ghost exchange, the
nodal remap and migration call it too. The header-include direction `tools/check-layering.py`
enforces is not the CMake target link graph; `transport`'s library does not link `backend`'s.

`Transport::move()` is a host routine (its pack/unpack is `std::memcpy`), so it is never handed a
device pointer. The Transport interface separates the two halves: `Transport::packed_layout()`
reports the `send_index`/`recv_index`/`local_copy` arrays `move()` itself packs and unpacks with,
mirrored to the device once at setup (`Impl::transport_device`'s `mpi_*` fields,
`mirror_transport_packed_layout()`); `Transport::move_packed()` is the bare `MPI_Alltoallv` on
already-packed buffers, host or device. Whenever `is_gpu_aware() && has_packed_layout`, the hot loop
packs/unpacks the MPI-ordered layer itself on `Space` and calls `move_packed()`, never `move()`. A
`Kokkos::fence()` is required between the device pack kernel and the `move_packed()` call that reads
its output; every other exchange in the file that hands MPI a device buffer fences first (for
example `nodal_dist_gather_reduce_kernel`'s `Kokkos::fence("nodal_dist_gather_reduce_fence")`), and
the older `Transport` code never needed one because its device-to-`move()` crossings went through a
`Kokkos::deep_copy`, which synchronizes implicitly.
`WindowRunnerMpi.DeviceDirectTransportMatchesHostAndSetsMovesCounter`
(`test/parallel/window_runner_mpi_test.cpp`) checks bitwise-identical delivery against the host twin
at 2 and 3 ranks and `RunStats::device_transport_moves > 0` on the GPU-aware path.

The Open MPI 5 launcher needs `PRTE_MCA_rmaps_default_mapping_policy=:oversubscribe` next to the
Open MPI 4 variable in every MPI row's ENVIRONMENT, or every 5-rank and MPMD row is refused with
"not enough slots".

## The distributed nodal remap exchange

`fuse::linalg` stays Kokkos-free (`comm.hpp`'s header comment: "linalg includes only base"); the
device-side packing lives in `backend`.

- `fuse::linalg::PackedExchanger` (`comm.hpp`), implemented by `MpiExchanger` (`comm_mpi.cpp`)
  alongside `GhostExchanger`, exposes the pack/unpack maps and counts/displacements
  `GhostExchanger::update()` derives once, plus two raw `MPI_Alltoallv` primitives
  (`exchange_packed()` and its adjoint `exchange_packed_adjoint()` for a scatter-add) over
  caller-packed buffers. `GhostExchanger::update()`'s signature and contract are unchanged; the
  `PackedExchanger` is reached with `std::dynamic_pointer_cast<const PackedExchanger>`, `nullptr` for
  the serial exchanger.
- `fuse::backend::DeviceGhostExchanger` (`include/fuse/backend/device_ghost_exchange.hpp`,
  header-only, templated on `Space`) drives `PackedExchanger` with Kokkos pack/unpack kernels,
  handing MPI a device pointer directly when `Space` is a device space and `mpi_is_gpu_aware()`,
  host-staging (deep_copy around the same `exchange_packed()` call) otherwise. `update()` is the
  forward (gather) direction; `update_adjoint()` is the scatter-add. No existing production caller
  uses it: `fuse::linalg::DistVector` is plain host `std::vector` by design, and
  `NativeMeshBackend::remap_halo_pattern` only builds a `GhostPattern`
  (`test/parallel/device_ghost_exchange_test.cpp` proves it on a ring pattern at 1-5 ranks).
- `fuse::build_distributed_remap_plan` (`nodal_remap_distributed.hpp/cpp`) resolves the exchange
  ONCE at setup (alongside `build_distributed_interp_rows`, only when `comm_handle` names a real
  communicator; a single-rank run keeps the `apply_distributed_H`/`apply_distributed_HT` path): every
  row's `src_global_ids` become (owner rank, owner's local position) via one extra one-time
  `MPI_Allgatherv`, and a component-expanded `fuse::linalg::GhostPattern` and its `MpiExchanger` are
  built exactly the way `make_mpi_exchanger` does for ghost fields. The per-period cost is one
  `MPI_Alltoallv` of a plain value array, gathered/scattered with Kokkos kernels
  (`Runner::Impl::nodal_dist_gather_on`/`nodal_dist_scatter_on`, `runner.cpp`) on whichever `Space`
  the arc lives in. The alternative -- exchanging wire records that interleave a global id with every
  value and rebuilding an `unordered_map` each period -- leaves no contiguous device buffer to hand
  MPI.
- A second exchanger (`DistributedRemapPlan::ghost_fill_exchanger`) is required:
  `apply_distributed_HT`'s contract overwrites every LOCAL row -- owned or ghost -- with the same
  global total for its global id, and the primary exchanger only scatter-adds into each vertex's
  owner. `ghost_fill_exchanger` is a second component-expanded `GhostPattern` over every local ghost
  entity, reusing the global ownership map the primary pattern already computed.
  `NodalRemapRunnerMpi.DisjointRankSets` at 3 ranks covers it.
- **A per-rank early return cannot guard a collective.** `DeviceGhostExchanger::update()` must not
  return early when THIS rank's `ghost_size() == 0`: `ghost_size()` is a per-rank count, and one
  rank having zero local ghosts while a peer has some is the ordinary case for the nodal-remap
  pattern (a rank that owns no evaluation entities for a channel has zero slots, while a peer still
  needs its own `MPI_Alltoallv`). Every rank must enter the collective whatever its own counts are;
  `MpiExchanger::update()` never special-cases a zero count for the same reason. A zero-count
  `MPI_Alltoallv` entry is a cheap no-op MPI handles, and the pack/unpack kernels are likewise never
  guarded by `n == 0` (a `RangePolicy(0, 0)` `parallel_for` is zero iterations).
- **Reproducibility.** The distributed H^T total is bit-identical across rank counts:
  `build_distributed_remap_plan` resolves once at setup a CSR
  (`DistributedRemapPlan::target_row_offset`/`sorted_answer_slot`) grouping every incoming
  contribution by its target local position and sorting each group ascending by the contributor
  evaluation entity's global id -- the same key `apply_distributed_HT`'s per-call sort uses. The
  gid array is resolved by one extra one-time adjoint exchange of a `double`-encoded gid array,
  reusing `PackedExchanger::exchange_packed_adjoint()`. The per-period kernel
  (`nodal_dist_scatter_reduce_deterministic_kernel`) sums each target's bucket sequentially in that
  fixed order, one thread per target, no atomics; `Kokkos::atomic_add` into each owner's position
  would be correct but would sum in arrival order.
  `parallel.nodal_remap_runner.ConsistentTransposeIsBitIdenticalAcrossRankCounts` asserts exact
  equality against the one-rank serial reference at every rank count, on the host and `.device` rows.
  The host and device multiply/sum steps have no `a*b+c` fusion opportunity in source, unlike the
  Relax/Remap kernels [](runner-device-dispatch.md) notes are not bit-identical to the host.
- H^T's own input gather (the eval-side value times the eval side's cell measure) still runs on the
  host regardless of `Space`: `MeshView::measure()` has no device-callable form.
- The device-pointer branch of every exchange needs a CUDA-aware MPI to be exercised; on a build
  without one, `mpi_is_gpu_aware()` is false (Open MPI built `--without-cuda` returns 0 from
  `MPIX_Query_cuda_support()` even though the extension macro is declared) and the host-staged
  fallback runs. Each MPI row is tested under both the natural and the forced
  `FUSE_TRANSPORT_HOST_STAGING=1` settings.

## Reproducibility contract on the device

Dot and norm reductions are bit-identical to the host. `RelaxFunctor` and `RemapMultiFunctor` are
not: nvcc contracts `a*b + c`, and `runtime.runner_device` asserts agreement to 1e-14 relative for
those shapes. Any new device kernel follows the same rule: bitwise where the host path is a
reduction the determinism contract covers, working precision elsewhere, and the test says which.

## Space-time projection

`OpKind::Sample`'s history branch writes its row into `history_device` with a same-space
`Kokkos::deep_copy` and is Space-generic unconditionally. `OpKind::Interpolate`'s history branch runs
the temporal projection and the spatial weights as one Space kernel, `num::apply_spacetime_device`
(`include/fuse/num/space_kernels.hpp`): one thread per target spatial entity, walking the same small
temporal stencil (the row_begin slice for this target step) and the entity's own row of the spatial
operator (today always the identity), in the same ascending order the host `apply_spacetime` uses, so
the device answer agrees with the host to rounding. The host-only `apply_spacetime`
(`src/backend/spacetime.cpp`) is what Host mode calls and the oracle the device kernel is checked
against. `build_temporal_projection`'s setup-time matrix assembly is exact rational arithmetic, not a
per-period cost.

## Measurements

Gate runs used a GTX 1050 Ti with Open MPI without CUDA support, so every exchange is host-staged.
The device twins are correctness rows, not a speed claim at small sizes: problems of a few hundred to
a few thousand cells are dominated by kernel launch latency. Before window-scoped regions' arcs and
`previous[]` stopped being staged per instruction, `verification.brusselator_runner` and
`verification.iteration` ran 21-22x slower on the device than on the host; afterwards they run
1.4-1.6x slower, and the remainder is launch latency.

Scaled benchmark (`test/bench/scaled_device_benchmark`, N=1000: 1e6 consumer cells, 1e6 provider
vertices, nodal consistent Remap, 5 periods, 3 repetitions, medians via `tools/bench-device.py`):

| ranks | space | build (s) | initialize (s) | period (s) |
|---|---|---|---|---|
| 1 | host | 77.4 | 25.9 | 0.409 |
| 1 | device | 75.3 | 13.2 | 0.0145 |
| 2 | host | 40.3 | 32.3 | 0.511 |
| 2 | device | 40.4 | 19.4 | 0.284 |
| 3 | host | 28.7 | 30.7 | 0.271 |
| 3 | device | 28.9 | 25.0 | 0.166 |

At one rank the per-period device time is 28x below the host's: the per-cell arithmetic dominates
and no exchange is involved. At two and three ranks the gap narrows to 1.6-1.8x because the nodal
remap's exchange and the halo fill are staged through host buffers on that MPI. `build` is mesh
generation on the host and is the same in both rows. These numbers are from the host-staged path;
they have not been re-measured with the device-direct Transport path.

## Open items

- The supermesh (r3d/r2d) clip stays host-only; the device feasibility measurement is blocked on the
  three build-system items above, not on the algorithm.
- `MeshView::measure()` has no device-callable form: the distributed nodal remap's H^T input gather
  runs on the host regardless of `Space`.
- `DistVector`, the accelerators (Aitken, IQN-ILS, IQN-IMVJ) and the Newton solves are host-only:
  arcs they touch are staged at the boundary, not kept on the device.
- Still host: MOAB-backed meshes and the mesh file reader; consistent/general interpolation and
  Tempest weights; `detail::WindowMemberState::history` (a separate structure from
  `SpacetimeInfo::history`).
- Only Kokkos CUDA (sm_61) with Open MPI has been run; HIP and SYCL builds, and MPICH with a
  CUDA-aware build, are untested.
- `verification.fsi_coupled.device` diverges under the cusolverRf backend in the
  `FUSE_ENABLE_CUSPARSE` tree (not diagnosed), and `verification.fsi_energy_diag.device` has
  failed its mass-correction gate on one host.
