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
| transport `send_buf/recv_buf` | host vectors, `MPI_Alltoallv` | device buffers when `mpi_gpu_aware()` is true at run time; otherwise pack kernel → host buffer |
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
2. `Transport` when MPI is not GPU-aware (pack kernel to a host buffer), and
   `apply_nodal_dist_remap()`;
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

## Multi-rank (Phase 3)

`mpi_is_gpu_aware()` is a compile-time constant today and informational.
It becomes a run-time probe (`MPIX_Query_cuda_support()` where the MPI
provides it, with a configuration override) consulted by the transport and
the halo/migration exchanges. Rank-to-device binding happens in
`ensure_kokkos_initialized()` and the test main (Kokkos' device-id-by-MPI-rank
mapping). Every MPI test row gains a device-mode twin labelled `device`; on a
one-GPU host such as higgs these run as several ranks sharing the device,
which exercises correctness and the GPU-aware path, not scaling.

## Reproducibility contract on the device

Dot and norm reductions are bit-identical to the host (D38). `RelaxFunctor`
and `RemapMultiFunctor` are not: nvcc contracts `a*b + c`, and
`runtime.runner_device` asserts agreement to 1e-14 relative for those shapes.
The owner accepted this on 2026-10-12; the TODO is recorded under D38 in
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
5. Phase 2 mesh/search/builders; Phase 3 transport/binding/MPI twins (in
   progress on their own worktrees; see the reports linked from the final
   review report).
6. Measurement against the quiet Host baseline recorded in the same session.

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
