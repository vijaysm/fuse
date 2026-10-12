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

1. Arcs in `ArcMemSpace`; `span_c`/`span_m` direct; staging only for
   component-owned host fields.
2. `MemorySpace::Device` `FieldRef`s accepted; read-refreshed mirrors for
   host fields; the Kokkos-based example components register device fields.
3. `reduction_scratch`, `previous[]`, history, accelerator gather/scatter,
   Epoch/Diagnose/Constrain, checkpoint and Newton seam moved to the rule
   above.
4. Device-mode test rows (the verification and example rows as `device`-
   labelled twins) and `device_boundary_copies` assertions.
5. Phase 2 mesh/search/builders; Phase 3 transport/binding/MPI twins.
6. Measurement against the quiet Host baseline recorded in the same session.
