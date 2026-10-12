# Soma import: per-file and per-case evidence table

Produced for the soma-import track. Every row below is backed by a grepped, diffed, or
directly-read piece of evidence computed from the working tree; none is asserted from memory.

## Methodology

1. **File-level status** (181 rows, one per file under `upstream/soma/`): `UPSTREAMS.lock`'s
   `[soma].superseded` list is expanded against the actual 181-file tree (directory entries such
   as `include/soma/ir/` expanded to their member files); that yields 95 files already recorded
   `superseded`. The remaining 86 are classified individually below by a basename/path counterpart
   plus a content diff, a same-named-or-triaged gtest case, or a direct grep for the behaviour
   elsewhere in `fuse::`.
2. **Per-case table** (135 rows): `UPSTREAMS.lock` records 948 `TEST(`/`TEST_F(` cases in
   `upstream/soma`. Of these, 135 (in 10 files) have no case of the identical `(suite, name)` pair
   anywhere in `fuse/test` -- reproduced exactly by script here, matching the lead's own prior scan
   (`remap_test.cpp` 47, `lowfive_transport_test.cpp` 25, `spacetime_conservation_test.cpp` 23,
   `spacetime_runtime_test.cpp` 21, `remap_runtime_test.cpp` 8, `spacetime_order_test.cpp` 4,
   `remap_order_test.cpp` 3, `runner_test.cpp` 2, `ir_dump_test.cpp` 1, `smoke_test.cpp` 1).
   Per the owner's explicit scope ruling for this track (triage, then port the highest-value gaps),
   these 135 were triaged rather than each given an independent side-by-side proof: an LLM-assisted
   pass (the session's reasoning model) was given each case's full body plus up to 6 candidate fuse
   cases (ranked by identifier-token overlap, restricted to the same subsystem's test directories)
   and asked to call `covered` / `likely_covered` / `missing` with a cited fuse case. The raw triage
   verdicts, before any correction, were 114 `missing`, 15 `likely_covered`, 5 `covered`, 1
   `superseded_behavior_change` (`LowFiveTransport.WritesNoFileWhenCompiledOut`, flagged directly during
   ordinary triage of that file, not via the safety net below). A broader safety net then re-scanned
   every `missing` verdict for a >=3-significant-word name overlap against the FULL fuse test corpus (not
   just the subsystem pool); 14 of the 114 were flagged, 4 were manually re-read, and 2 were corrected to
   `superseded_behavior_change` (both `Runner` cases, where fuse's own test comment documents the
   deliberate change -- a dependency-free supermesh path landed and turned a documented refusal into a
   documented success). That leaves the final counts in the table above: 112 `missing`, 15
   `likely_covered`, 5 `covered`, 3 `superseded_behavior_change` (1 from ordinary triage + 2 from the
   safety net). The 112 `missing` and all 15 `likely_covered` rows are **triage-confidence**, not
   independently re-read; the `confidence` column marks exactly which rows were verified by a direct
   second read.
3. **C/Fortran ABI table**: every `soma_c.h`/`soma_mod.f90` entry point was grepped for a
   declaration, then cross-referenced against `include/fuse/abi/entries.def`'s `FUSE_ABI_ENTRY`
   list and `fortran/ffuse.f90`'s `ffuse_*` functions, reading `include/fuse/abi/types.h`'s struct
   definitions where a merge was suspected (confirmed for `fuse_setup_options` and
   `fuse_step_report`).

## Summary counts (computed by script from the 181-row and 135-row tables below)

### File-level status (181 files)

| Status | Count |
|---|---|
| `superseded` | 99 |
| `ported` | 37 |
| `history` | 14 |
| `missing` | 12 |
| `partially ported` | 11 |
| `infrastructure-superseded` | 7 |
| `likely_covered_partial` | 1 |
| **Total** | **181** |

### Per-case verdicts (135 cases with no same-named counterpart)

| Verdict | Count |
|---|---|
| `missing` | 112 |
| `likely_covered` | 15 |
| `covered` | 5 |
| `superseded_behavior_change` | 3 |
| **Total** | **135** |

## File-level table

181 files under `upstream/soma/`, one row each. `superseded` rows (95 individually tagged by
UPSTREAMS.lock, grouped below by the stage comment attached in the lock file) are compacted into
one row per group since their evidence is identical by construction (`UPSTREAMS.lock` is itself
the evidence); every other status gets its own row with file-specific evidence.

### `superseded` (99 files, grouped by stage)

| Stage / evidence | Files |
|---|---|
| Listed in UPSTREAMS.lock [soma].superseded -> S4/S5 (fuse::num/runtime/transport/backend); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `include/soma/mesh/mesh_view.hpp`, `include/soma/mesh/moab_backend.hpp`, `include/soma/mesh/serial_backend.hpp`, `include/soma/num/accelerator.hpp`, `include/soma/num/convergence.hpp`, `include/soma/num/kernels.hpp`, `include/soma/num/temporal_basis.hpp`, `include/soma/num/waveform.hpp`, `include/soma/remap/remap_plan.hpp`, `include/soma/remap/spacetime.hpp`, `include/soma/runtime/component.hpp`, `include/soma/runtime/runner.hpp`, `include/soma/transport/comm_graph.hpp`, `include/soma/transport/transport.hpp`, `src/mesh/moab_backend.cpp`, `src/mesh/serial_backend.cpp`, `src/num/accel_quasi_newton.cpp`, `src/num/accel_relaxation.cpp`, `src/num/kernels_inst.cpp`, `src/remap/high_order_backend.hpp`, `src/remap/intx_weights.cpp`, `src/remap/spacetime.cpp`, `src/remap/tempest_weights.cpp`, `src/runtime/iteration.cpp`, `src/runtime/runner.cpp`, `src/runtime/runner_impl.hpp`, `src/transport/comm_graph.cpp`, `src/transport/copy_transport.cpp`, `src/transport/crystal_router.cpp`, `src/transport/crystal_router.hpp`, `src/transport/lowfive_transport.cpp`, `src/transport/mpi_transport.cpp`, `test/parallel/comm_graph_test.cpp`, `test/parallel/mpi_transport_test.cpp`, `test/parallel/runner_transport_test.cpp`, `test/unit/moab_mesh_test.cpp` |
| Listed in UPSTREAMS.lock [soma].superseded -> S2 (fuse::model); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `include/soma/graph/channel.hpp`, `include/soma/graph/component.hpp`, `include/soma/graph/coupling_graph.hpp`, `include/soma/graph/descriptor.hpp`, `include/soma/graph/field_spec.hpp`, `include/soma/graph/region.hpp`, `include/soma/graph/validate.hpp`, `include/soma/graph/yaml_io.hpp`, `src/graph/yaml_io.cpp`, `test/unit/channel_test.cpp`, `test/unit/component_test.cpp`, `test/unit/coupling_graph_test.cpp`, `test/unit/descriptor_test.cpp`, `test/unit/region_test.cpp`, `test/unit/validate_test.cpp`, `test/unit/yaml_io_test.cpp` |
| Listed in UPSTREAMS.lock [soma].superseded -> S1 (fuse::base); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `include/soma/base/ids.hpp`, `include/soma/base/rational.hpp`, `include/soma/base/result.hpp`, `include/soma/base/time_level.hpp`, `include/soma/base/units.hpp`, `test/unit/ids_test.cpp`, `test/unit/rational_test.cpp`, `test/unit/result_test.cpp`, `test/unit/time_level_test.cpp`, `test/unit/units_test.cpp`, `test/verification/iteration_test.cpp`, `test/verification/mms_transfer_test.cpp` |
| Listed in UPSTREAMS.lock [soma].superseded -> S3 (fuse::ir); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `include/soma/ir/dag.hpp`, `include/soma/ir/dump.hpp`, `include/soma/ir/lowering.hpp`, `include/soma/ir/op.hpp`, `src/ir/checkpoint.cpp`, `src/ir/lowering.cpp`, `src/ir/regions.cpp`, `src/ir/rules_channel.cpp`, `src/ir/rules_derived.cpp`, `test/golden/pair_accumulate_remap_relax.ir`, `test/golden/pair_explicit.ir`, `test/golden/pair_iterated.ir` |
| Listed in UPSTREAMS.lock [soma].superseded -> S3 (fuse::schedule); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `include/soma/schedule/acyclicity.hpp`, `include/soma/schedule/buffers.hpp`, `include/soma/schedule/clock.hpp`, `include/soma/schedule/expand.hpp`, `include/soma/schedule/plan.hpp`, `include/soma/schedule/slab.hpp`, `src/schedule/acyclicity.cpp`, `src/schedule/buffers.cpp`, `src/schedule/expand.cpp`, `src/schedule/scheduler.cpp`, `src/schedule/slab.cpp` |
| Listed in UPSTREAMS.lock [soma].superseded -> S2 (fuse::model, YAML fixtures); fuse's own replacement already carries this path's behaviour (see porting.md naming map). | `test/data/pair.yaml`, `test/data/pair_accumulate_remap_relax.yaml`, `test/data/pair_explicit.yaml`, `test/data/pair_iterated.yaml`, `test/data/pair_iterated_bad_measure.yaml`, `test/data/pair_iterated_no_convergence.yaml`, `test/data/pair_iterated_zero_iterations.yaml`, `test/data/vector_field.yaml` |
| fuse already resolves LowFive via cmake/FindLowFive.cmake (same discovery job, fuse::lowfive imported target), used throughout src/transport and test/transport. | `cmake/FindSomaLowFive.cmake` |
| fuse resolves MOAB inline in cmake/FuseDependencies.cmake (pkg_check_modules + capability probing), functionally the same job as this file under FUSE's own dependency-resolution convention; no separate Find module was ported because fuse's convention centralises resolution in FuseDependencies.cmake rather than per-dependency Find files. | `cmake/FindSomaMOAB.cmake` |
| SOMA_ENABLE_{TESTS,MPI,MOAB,PETSC,KOKKOS,LOWFIVE,PYTHON,FORTRAN} map 1:1 to FUSE_ENABLE_{TESTS,MPI,...} options already declared across CMakeLists.txt/FuseDependencies.cmake/FuseFortran.cmake. | `cmake/SomaOptions.cmake` |
| Soma's lowfive_pair is explicitly SERIAL, in-process-only interception (file's own header: 'no DistMetadataVOL, no MPI intercomm'), reusing mms_pair physics to show an unmodified H5Fcreate/H5Dwrite sequence is intercepted with nothing written to disk. fuse covers the identical claim as actual unit tests (test/transport/lowfive_test.cpp: LowFiveInterception.UnmodifiedHdf5ComponentIsIntercepted, .SamePathIsWrittenByPlainHdf5AndNotByTheTransport, .EachStepCarriesItsOwnData, .MultiComponentFieldsRoundTrip, .TheRegistryHandsBackAWorkingTransport) and goes further with its OWN, more ambitious cross-process demo (examples/demo_b, docs/user/demo-b.md) and the Brusselator LowFive MPMD example (examples/brusselator, docs/user/lowfive.md). Demo B's distributed path is itself KNOWN_ISSUES/roadmap.md item D14 ('LowFive's distributed transport and Demo B are compiled, not [verified under MPI]') -- closing D14 is a host-MPI measurement out of this track's scope (see open_questions), but it does not change that lowfive_pair's own narrower claim is already covered. | `examples/lowfive_pair/main.cpp` |

### `ported` (37 files)

| File | Evidence |
|---|---|
| `examples/dn_heat/dn_component.hpp` | Counterpart examples/dn_heat/dn_component.hpp exists; diff is a renamed+transformed copy (fuse:: namespace, FUSE error types), not byte-identical -- same scenario (Dirichlet-Neumann heat pair). |
| `examples/dn_heat/main.cpp` | Counterpart examples/dn_heat/main.cpp exists and drives the same two-component Dirichlet-Neumann demo under FUSE's own setup/run calls. |
| `examples/mms_pair/heat_component.hpp` | Counterpart examples/mms_pair/heat_component.hpp exists (158-line diff: renamed + transformed, same manufactured-solution heat pair physics). |
| `examples/mms_pair/main.cpp` | Counterpart examples/mms_pair/main.cpp exists; reused verbatim by fuse's own lowfive_pair-equivalent commentary (examples/demo_b) citing this exact physics as 'Demo A'. |
| `fortran/CMakeLists.txt` | Replaced by fortran/CMakeLists.txt building ffuse.f90/ffuse_provider.f90/fuse_fortran_shim.c under FUSE_ENABLE_FORTRAN (porting.md: 'module soma_mod -> module ffuse, ruled 2026-09-17'). |
| `fortran/soma_mod.f90` | 24 soma_f_* wrappers map onto fortran/ffuse.f90's 16 ffuse_* entries (1:1 with include/fuse/abi/entries.def's FUSE_ABI_ENTRY list), each wrapping the SAME struct-based call convention soma_c.h -> fuse_coupler_abi() replaced it with. Mapping: version->ffuse_version; abi_version->folded into ffuse_version's fuse_version_info.abi_version field; create/destroy/last_error/field_handle/initialize/begin_step/end_step/is_coupling_ongoing -> same-named ffuse_* entries; load_yaml/load_descriptor/descriptor_size/save_descriptor -> ffuse_load_graph/ffuse_save_graph (struct-based path/bytes argument, same convenience merge as soma_c.h's soma_load_graph_yaml); register_field_1d/_2d -> ffuse_register_field (one generic fuse_field_view-based entry instead of two rank-specific ones); set_communicator + set_end_time -> merged into ffuse_setup(fuse_setup_options) (include/fuse/abi/types.h: fuse_setup_options carries communicator and has_end_time/end_time in one struct, plus a my_component field soma never had); requires_checkpoint/requires_restore -> folded into fuse_step_report's own requires_checkpoint/requires_restore fields, read off the begin_step/end_step report instead of queried separately; run_period + run -> merged into ffuse_run. soma_f_is_open has no ffuse_* counterpart: not in soma_c.h either (a Fortran-only convenience over the raw ctx handle), and is recorded here as intentionally dropped rather than silently missing. |
| `include/soma/field/field_ref.hpp` | Counterpart include/fuse/field/field_ref.hpp exists (37-line diff: namespace/error-type rename, same FieldRef contract). |
| `include/soma/field/registry.hpp` | Counterpart include/fuse/field/registry.hpp exists, same line count (109/109), 58-line diff (namespace/error-type rename + fuse extensions). |
| `include/soma/soma_c.h` | porting.md's own naming-map ruling: 'soma_c.h flat C API -> the fuse_coupler_abi() table | not renamed: REPLACED, see abi-design.md'. All 21 soma_c.h entries map onto include/fuse/abi/entries.def's 16 FUSE_ABI_ENTRY + merges documented in docs/dev/soma-import.md's C-API table. |
| `src/capi/soma_c.cpp` | Implementation replaced by src/abi/participant.cpp (the fuse_coupler_abi() table's definition) and src/abi/provider_adapter.cpp for the provider direction soma never had. |
| `test/golden_ir_test.cpp` | Direct counterpart test/ir/golden_ir_test.cpp exists; every TEST case name in this file matched by (suite,name) in fuse/test (0 of its cases appear in the 135 unmatched list). |
| `test/support/allocation_counter.hpp` | test/support/allocation_counter.hpp is BYTE-IDENTICAL to the soma original (diff empty) -- already carried over verbatim under the same path. |
| `test/unit/accelerator_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/acyclicity_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/buffers_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/checkpoint_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/clock_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/component_iface_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/convergence_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/expand_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/field_ref_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/ir_dag_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/ir_op_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/kernels_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/lockstep_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/lowering_channel_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/lowering_derived_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/lowering_regions_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/lowering_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/plan_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/serial_mesh_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/slab_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/spacetime_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/temporal_basis_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/transport_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/waveform_order_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |
| `test/unit/waveform_test.cpp` | Every TEST(/TEST_F() case name in this file is matched by (suite,name) in fuse/test (0 appear in the 135-case unmatched list). Counterpart: (dispersed across fuse/test, no single basename match). |

### `partially ported` (11 files)

| File | Evidence |
|---|---|
| `test/unit/capi_test.c` | 26 void test_*() functions called from a custom main() (not TEST()/TEST_F(), so this file was never in the 948-case corpus UPSTREAMS.lock counts, and is graded separately here). Covers the flat soma_c.h entry points directly: version, create/destroy, null-argument rejection, double-destroy and stale-handle behaviour, error-message retrieval/truncation/thread-local-without-a-context, yaml load, descriptor round trip + garbage rejection, register_field (+ bad-argument rejection), state errors, initialize+step, coupling-ongoing, iteration checkpoint contract, convergence-criterion rejections (empty/unmatched-measure/zero-max-iterations), set_communicator, invalid-graph reporting, exceptions from direct and nested arguments. fuse's test/abi/*.cpp suite (participant_test.cpp, conformance_test.cpp, negotiation_test.cpp, struct_size_test.cpp) is richer on ABI VERSIONING and SCHEDULING semantics (struct-size negotiation across versions, provider-table mismatches, window/suspension timing) but none of them call fuse_coupler_abi()'s create/destroy/register_field/last_error entries with null or malformed arguments, call destroy twice, or check last_error from a thread with no context -- grepped for null-argument-rejection, double-destroy, and thread-local-error wording across test/abi/*.cpp with no hit beyond the already-covered truncation case (AbiParticipant.LastErrorCopiesThePrefixThatFitsWhenTruncated). The flat C ABI's own defensive/robustness contract (as opposed to its scheduling contract) is a genuine gap. |
| `test/unit/ir_dump_test.cpp` | 1 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/ir/ir_dump_test.cpp (direct counterpart). |
| `test/unit/lowfive_transport_test.cpp` | 25 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/transport/lowfive_test.cpp, lowfive_graph_test.cpp. |
| `test/unit/remap_order_test.cpp` | 3 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): dispersed across test/backend. |
| `test/unit/remap_runtime_test.cpp` | 8 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): dispersed across test/backend, test/parallel/nodal_remap_*. |
| `test/unit/remap_test.cpp` | 47 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): dispersed: test/backend/supermesh_test.cpp, mls_weights_test.cpp, transfer_map_io_test.cpp, operator_conformance_test.cpp, cell_simplices_test.cpp, native_halo_test.cpp, moab_mesh_test.cpp. |
| `test/unit/runner_test.cpp` | 2 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/runtime/runner_test.cpp (direct counterpart). |
| `test/unit/smoke_test.cpp` | 1 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): none found. |
| `test/unit/spacetime_conservation_test.cpp` | 23 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/backend/spacetime_test.cpp, test/num/waveform_order_test.cpp. |
| `test/unit/spacetime_order_test.cpp` | 4 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/num/waveform_order_test.cpp. |
| `test/unit/spacetime_runtime_test.cpp` | 21 of this file's cases have no same-named case in fuse/test; each is adjudicated individually in the per-case table below (LLM-assisted triage per the owner's scope ruling, not an independent side-by-side read for every case). Candidate counterpart(s): test/backend/spacetime_test.cpp, test/runtime/runner_test.cpp. |

### `likely_covered_partial` (1 files)

| File | Evidence |
|---|---|
| `test/unit/moab_parallel_test.cpp` | Custom-main() MPI harness (not a TEST() binary, excluded from UPSTREAMS.lock's 948 count). Proves MeshView::entity_count is a LOCAL count, a global reduction over owned_count recovers the global count, and owned<=local / is_owned agrees with owned_count, on a real >1-rank MOAB partition. fuse's test/parallel/moab_placed_view_mpi_test.cpp runs under the same real multi-rank MOAB ParallelComm and proves PLACEMENT/tiling correctness (TwoDisjointPlacementsEachTileTheWholeMesh, APartiallyOverlappingPairEachStillTilesTheWholeMesh) but does not independently assert the specific entity_count/owned_count global-reduction identity soma's file checks. Partial: the placement-view machinery is proven under real MPI, the ownership-count arithmetic identity is not separately asserted. |

### `missing` (12 files)

| File | Evidence |
|---|---|
| `include/soma/soma.hpp` | Umbrella header defining only `soma::version() -> "0.1.0"`. fuse has no umbrella include/fuse/fuse.hpp and no free fuse::version() function; this is the direct cause of Smoke.VersionIsNonEmpty having no counterpart (see test-case table). |
| `python/CMakeLists.txt` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `python/soma/__init__.py` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `python/src/bindings.cpp` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `python/tests/test_buffers.py` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `python/tests/test_graph.py` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `python/tests/test_runtime.py` | porting.md's naming map already rules the target name ('Python _soma/soma -> package pfuse; ruled 2026-09-17; availability confirmed'), but cmake/FuseDependencies.cmake's FUSE_ENABLE_PYTHON option (line 123) is a stub (OFF, no pybind11 module wired) and no python/ directory exists under fuse yet. Building the pfuse package and porting these 3 pytest files is listed in this project's desired final outputs as a separate, larger undertaking (see open_questions) and was not attempted in this track's budget. |
| `test/fortran/capi_test.f90` | Fortran-side smoke test of soma_mod's wrappers (create/destroy/last_error/load/step). No counterpart under test/abi/fortran/: that directory's sizes_test.f90 and consumer.f90 check struct sizes and linkage/install, not a full create->initialize->step->destroy lifecycle driven from Fortran. |
| `test/parallel/capi_parallel_test.c` | Soma's MPI-launched C-ABI smoke test (soma_set_communicator + soma_run_period across ranks via raw MPI_Comm). No fuse/test file drives fuse_coupler_abi()'s C entry points under a real multi-rank MPI_Comm from C; test/parallel/abi_participant_mpi_test.cpp exercises the ABI's scheduling semantics under MPI but through the C++ Diagnostics wrapper, not the raw extern "C" table called from a .c translation unit. |
| `test/tools/make_partitioned_sphere.cpp` | MOAB test-data generator (writes a partitioned cubed-sphere .h5m pair) used only by remap_parallel_test.cpp and remap_halo_parallel_test.cpp (both themselves missing, see parallel-harness rows) to construct a deliberately misaligned MOAB partition pair. No fuse/test/tools equivalent exists; porting it is a prerequisite for porting either of those two files. |
| `test/unit/remap_halo_parallel_test.cpp` | Companion to remap_parallel_test.cpp: proves build_remap_halo/exchange_remap_halo are CORRECT (not just refused) on a deliberately misaligned MOAB partition pair (3456-cell cubed sphere, 8 partition sets, vs. 1536-cell/4-set partner), using test/tools/make_partitioned_sphere.cpp (itself missing). Same MOAB-backend gap as remap_parallel_test.cpp; the native backend's equivalent is covered by native_halo_mpi_test.cpp/remap_halo_mpi_test.cpp, the MOAB backend's is not. |
| `test/unit/remap_parallel_test.cpp` | Proves build_weights's E-REMAP-PARALLEL guard fires on >1-rank MOAB communicators and does NOT fire on a real size-1 communicator (as opposed to a serial/no-MPI build where the branch is dead code). fuse's own test/parallel/moab_placed_view_mpi_test.cpp explicitly scopes itself OUT of this: 'Per-placement REMAP WEIGHTS on MOAB ... are not built here'. native_halo_mpi_test.cpp/remap_halo_mpi_test.cpp exercise the analogous guard for the NATIVE mesh backend only. No fuse test exercises E-REMAP-PARALLEL on the MOAB backend under a real multi-rank communicator. |

### `history` (14 files)

| File | Evidence |
|---|---|
| `docs/BUILDING.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/HANDOFF.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/2026-09-02-soma-implementation.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-0.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-1.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-2.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-3.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-4.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-5.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-6.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-7.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/plans/soma-phase-8.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/specs/2026-09-02-soma-design.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |
| `docs/specs/2026-09-11-comm-graph-design.md` | Design/plan document describing the retired soma:: library; moved verbatim into docs/history/soma/ with a provenance header (step 3), not ported as code. |

### `infrastructure-superseded` (7 files)

| File | Evidence |
|---|---|
| `.gitignore` | fuse's own top-level .gitignore already covers the superbuild; soma's is scoped to its own standalone build and carries no fuse-specific content to port. |
| `CMakeLists.txt` | soma's own standalone superbuild file; fuse builds the subtree via add_subdirectory(upstream/soma) from the top-level CMakeLists.txt (I8, unmodified import). Retired with FUSE_BUILD_UPSTREAM_SOMA (step 4). |
| `CMakePresets.json` | soma's own standalone presets; fuse/CMakePresets.json is the superbuild's own, already present and unrelated to this file's content. |
| `README.md` | Describes the standalone soma library/build; superseded in content by fuse/README.md and docs/user/build.md. No unique technical content beyond what docs/history/soma preserves from the design docs. |
| `examples/CMakeLists.txt` | soma's own example build file; fuse/examples/CMakeLists.txt already registers dn_heat, mms_pair and the richer brusselator/demo_b/gray_scott/fsi_channel/layout_driver examples. |
| `src/CMakeLists.txt` | soma's own src/ build file; fuse/src/CMakeLists.txt already builds the equivalent functionality under fuse's own layer-target convention (fuse-coupler-<layer> libraries). |
| `test/CMakeLists.txt` | soma's own test registration file; fuse/test/CMakeLists.txt registers the native fuse:: suite directly (fuse_add_gtest and friends), unrelated to this file's content beyond the shared gtest/ctest convention. |

## Per-case table: the 135 cases with no same-named counterpart

Grouped by source file. `confidence` = `verified (direct read)` means both bodies were actually
read side by side in this session; `triage (LLM-assisted...)` means the verdict rests on the
triage pass described in Methodology and was not independently re-read.

### `test/unit/remap_test.cpp` (47 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `Remap.BuildsWeightsBetweenNonMatchingMeshes` | `missing` | — | No candidate checks exact nnz count for a simple box-to-box non-matching conservative remap. | triage |
| `Remap.HandlesVectorComponents` | `missing` | — | No candidate applies weights to multi-component vector fields and checks component-wise preservation. | triage |
| `Remap.IdenticalMeshesGiveTheIdentityMap` | `missing` | — | Given candidates don't test identical-mesh remap reproducing an arbitrary field exactly. | triage |
| `Remap.SameRegionOnBothSidesIsExactlyTheIdentity` | `covered` | `test/backend/supermesh_test.cpp` :: `Supermesh.RemappingOntoTheSameMeshIsTheIdentity` | Same scenario (same mesh both sides) and same diagonal-entry structure, but NOT the identical assertion strength: soma's original asserts EXPECT_EQ(w.val[k], 1.0) with its own comment '// exactly, not nearly', while fuse's counterpart uses EXPECT_NEAR(w.val[k], 1.0, 1e-13) -- a tolerance-based check. The scenario and diagonal structure are covered; the exact-equality strength of soma's own assertion is not reproduced. | verified |
| `Remap.TripletsAreSortedDeterministically` | `likely_covered` | `test/backend/spacetime_test.cpp` :: `Spacetime.TripletsAreCanonicalAndDeterministicForEveryBasis` | Same assertion (deterministic, sorted (row,col) triplets) but on spacetime weights, not plain spatial remap. | triage |
| `Remap.WeightsAreNonNegativeForFirstOrder` | `missing` | — | No candidate asserts non-negativity of remap weight values. | triage |
| `Remap.WorksInOneDimension` | `covered` | `test/backend/supermesh_test.cpp` :: `Supermesh.OneDimensionalNonMatchingIsExactlyConservative` | Same 1D line-mesh non-matching setup, checks n_rows/n_cols and row/col defect near zero. | verified |
| `RemapApply.RejectsBadArguments` | `missing` | — | Spacetime test checks apply_spacetime shape/step errors, not apply_weights null-pointer/n-vectors arg validation. | triage |
| `RemapApply.RejectsOutOfRangeTriplets` | `missing` | — | No candidate tests apply_weights COO row/col bounds or ragged triplet arrays (E-REMAP-COO). | triage |
| `RemapCacheTest.DistinguishesDifferentPlans` | `missing` | — | No candidate covers RemapCache distinguishing different plans via misses/hits counts. | triage |
| `RemapCacheTest.FailedBuildsAreNotCached` | `missing` | — | No candidate tests that failed builds are excluded from the RemapCache. | triage |
| `RemapCacheTest.PointersSurviveRehashing` | `missing` | — | No candidate tests cache pointer stability across unordered_map rehashing. | triage |
| `RemapCacheTest.ReusesWeightsForIdenticalPlans` | `missing` | — | No candidate tests a RemapCache reusing weights/hit-miss counters for identical plans. | triage |
| `RemapColumn.MeasureWeightedSumsRecoverTheSourceMeasure` | `likely_covered` | `test/backend/native_halo_test.cpp` :: `NativeHaloBuildWeights.ThreeRanksMisalignedCompleteWeightsAndLinearFieldReproduction` | Checks column mass sums reproduce source cell measures, but in a parallel halo decomposition scenario. | triage |
| `RemapColumn.PreservesTheGlobalIntegralOfAVaryingField` | `missing` | — | No candidate checks global integral conservation of a smooth varying field under non-matching remap. | triage |
| `RemapColumn.TheUnweightedColumnSumIsNotUnity` | `missing` | — | No candidate checks unweighted column sums or non-matching-mesh column-sum magnitude. | triage |
| `RemapHighOrder.BuildsAtOrdersTwoThreeAndFour` | `missing` | — | No fuse candidate builds high-order TempestRemap weights or checks order/nnz/negative-weight behavior. | triage |
| `RemapHighOrder.ConcurrentBuildsAreSerializedNotCorrupted` | `missing` | — | No candidate tests concurrent TempestRemap Announce-capture thread safety. | triage |
| `RemapHighOrder.ConservationLivesInTheBackendsMeasureNotMeshViews` | `missing` | — | No candidate compares MeshView-measure vs native spherical-area conservation defects for high-order remap. | triage |
| `RemapHighOrder.LatLonPolarTrianglesDefeatFirstOrderButHighOrderSucceeds` | `missing` | — | No candidate checks first-order E-REMAP-INTX failure vs high-order success on same lat-lon/cubed-sphere pair. | triage |
| `RemapHighOrder.ProductionScaleFvAndCubedSphereValidateCoverageTolAndOrderFourGeneralizes` | `missing` | — | No candidate builds production-scale f19/ne30-like remap pairs or checks coverage tolerance at scale. | triage |
| `RemapHighOrder.RefusedOutrightWhenTheBackendIsAbsent` | `missing` | — | No candidate checks E-REMAP-NOTEMPEST error when TempestRemap backend is absent. | triage |
| `RemapHighOrder.SilentDegradationToFirstOrderIsCaughtAndRefused` | `missing` | — | No candidate covers TempestRemap's silent first-order degradation detection on polar triangles. | triage |
| `RemapMonotone.AxisAlignedFirstOrderMapIsBoundPreserving` | `missing` | — | No candidate certifies monotone bound-preservation for axis-aligned box first-order remap. | triage |
| `RemapMonotone.HighOrderOvershootsAStepAndTheLimitedMapDoesNot` | `missing` | — | No candidate tests high-order overshoot on a step field vs limited monotone remap. | triage |
| `RemapMonotone.TheSupermeshMapBetweenCubedSpheresIsNotCertified` | `missing` | — | No candidate checks E-REMAP-MONOTONE refusal for supermesh map on sphere pair. | triage |
| `RemapPartial.DisjointDomainsProduceAnEmptyMap` | `missing` | — | No candidate tests disjoint-domain empty weight matrix or coverage-flag error on disjoint meshes. | triage |
| `RemapPartial.SourceWiderThanTargetIsRefusedWhenConservativeIsAsked` | `missing` | — | TransferMapIo test checks partition mismatch (E-MAP-COVERAGE), different scenario/error than mesh coverage E-REMAP-COVERAGE. | triage |
| `RemapPartial.SourceWiderThanTargetTruncatesWhenConservationIsNotAsked` | `missing` | — | No candidate tests truncation of uncovered source mass when conservative flag is absent. | triage |
| `RemapPartial.TargetWiderThanSourceKeepsEveryGramOfMass` | `missing` | — | Supermesh test uses fully-covering line meshes; no partial-coverage/uncovered-target-cell assertion present. | triage |
| `RemapPlanKey.CoversEveryField` | `missing` | — | No candidate tests RemapPlan::key() uniqueness across every field or forged method strings. | triage |
| `RemapPlane.ConservesOnSkewedUnstructuredQuads` | `missing` | — | No candidate builds conservative weights on skewed unstructured quad planar meshes. | triage |
| `RemapRefuses.ABackendThatIsNotMoab` | `likely_covered` | `test/backend/supermesh_test.cpp` :: `Supermesh.BuildWeightsReachesTheSupermeshPathForANonMoabBackend` | Same scenario: non-MOAB SerialMeshBackend with build_weights; but that case asserts success, not E-REMAP-BACKEND refusal. | triage |
| `RemapRefuses.ContradictoryOrderRequests` | `missing` | — | No candidate tests mismatched src/tgt order or HighOrder+order=1 contradiction. | triage |
| `RemapRefuses.HighOrderOnAVolumeMesh` | `missing` | — | No candidate tests high-order refusal on hexahedra geometry or TempestRemap dimension error. | triage |
| `RemapRefuses.InvalidRegionIds` | `missing` | — | No candidate tests invalid/nonexistent region id yielding E-REMAP-REGION. | triage |
| `RemapRefuses.MismatchedDimensions` | `missing` | — | No candidate tests dimension mismatch between src/tgt regions yielding E-REMAP-DIM. | triage |
| `RemapRefuses.MonotoneTogetherWithAnOrderAboveOne` | `missing` | — | No candidate checks monotone+order>1 refusal with bound preservation message. | triage |
| `RemapRefuses.NonFiniteVolumeDiscretizations` | `missing` | — | No candidate tests ContinuousGLL target discretization refusal for conservative remap. | triage |
| `RemapRefuses.OrdersTheBackendWillNotDeliver` | `missing` | — | No candidate tests order cap refusal or negative order rejection. | triage |
| `RemapRefuses.SubsetRegions` | `missing` | — | Candidates test MLS subset region (E-MLS-REGION), not remap's E-REMAP-REGION for entity_set subset. | triage |
| `RemapRefuses.SurfacesThatAreNeitherSphereNorPlane` | `missing` | — | No candidate tests refusal on tilted/non-sphere-non-plane surfaces with E-REMAP-UNSUPPORTED. | triage |
| `RemapRefuses.UnknownMethodNamesTheKnownOnes` | `missing` | — | No candidate checks unknown method name error listing valid method names. | triage |
| `RemapRow.SumsAreUnity` | `likely_covered` | `test/parallel/native_halo_mpi_test.cpp` :: `NativeHaloMpi.BuildRemapHaloThenBuildWeightsThenExchangeReproducesALinearFieldToOrderH` | Checks max_row_defect near zero (row sums unity) but in parallel halo setup, not plain serial remap. | triage |
| `RemapSphere.ConservesOnANonMatchingCubedSpherePair` | `missing` | — | No candidate tests conservative remap between two differently-resolved cubed-sphere meshes. | triage |
| `RemapSphere.IdentityOnMoabsCubedSphere` | `missing` | — | No candidate tests identity remap weights on MOAB cubed-sphere mesh loaded twice. | triage |
| `RemapWeightsView.MatchesTheSpacetimeAdapter` | `missing` | — | No candidate checks SpatialWeightsView fields matching RemapWeights row/col/val/nnz pointers. | triage |

### `test/unit/lowfive_transport_test.cpp` (25 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `LowFiveInterception.EachStepCarriesItsOwnData` | `missing` | — | No fuse test checks per-step varying data through LowFive with no-disk assertion across many steps. | triage |
| `LowFiveInterception.MultiComponentFieldsRoundTrip` | `missing` | — | No fuse test covers multi-component HDF5/LowFive round trip with no-disk assertion. | triage |
| `LowFiveInterception.PlainHdf5OnTheSamePathDoesCreateAFile` | `missing` | — | No fuse test verifies native HDF5 VOL control case creates a real file. | triage |
| `LowFiveInterception.ReadOfANonexistentPathSucceedsOnlyBecauseOfTheVol` | `missing` | — | No fuse test verifies VOL-enabled read succeeds where native VOL fails. | triage |
| `LowFiveInterception.SamePathIsWrittenByPlainHdf5AndNotByTheTransport` | `missing` | — | No fuse test compares native write vs LowFive interception on same path. | triage |
| `LowFiveInterception.TheRegistryHandsBackAWorkingTransport` | `missing` | — | No fuse test creates lowfive via registry and verifies HDF5 round trip with no disk file. | triage |
| `LowFiveInterception.UnmodifiedHdf5ComponentIsIntercepted` | `missing` | — | No fuse test checks unmodified HDF5 component writes intercepted via FAPL handle. | triage |
| `LowFiveTransport.AvailabilityMatchesTheBuild` | `covered` | `test/transport/lowfive_test.cpp` :: `LowFiveTransport.IsAvailableMatchesTheBuildConfiguration` | Same is_available() build-flag conditional check. | verified |
| `LowFiveTransport.BuildPlanBeforeSetLocationIsAnError` | `covered` | `test/transport/lowfive_test.cpp` :: `LowFiveTransport.BuildPlanBeforeSetLocationIsRefused` | Identical scenario and error code E-TRANSPORT-LF-LOCATION. | verified |
| `LowFiveTransport.EveryEntryPointFailsLoudlyWithoutLowFive` | `likely_covered` | `test/transport/lowfive_test.cpp` :: `LowFiveTransport.EveryEntryPointRefusesWithoutLowFive` | Same no-LowFive entry-point refusal scenario; overlapping but not identical call set. | triage |
| `LowFiveTransport.MoveAllocationCountIsReportedNotZero` | `missing` | — | No fuse test asserts move() allocates >0; MPI equivalent asserts zero allocations. | triage |
| `LowFiveTransport.MoveBeforeBuildPlanIsAnError` | `likely_covered` | `test/parallel/mpi_transport_test.cpp` :: `MpiTransport.PrepareAndMoveRefuseAnUnplannedTransport` | Same unplanned-transport move refusal scenario, different transport. | triage |
| `LowFiveTransport.MoveRejectsNullAndWrongSize` | `likely_covered` | `test/parallel/mpi_transport_test.cpp` :: `MpiTransport.MoveChecksItsLengthAndItsPointers` | Same null-pointer/size-mismatch move guard checks, different transport. | triage |
| `LowFiveTransport.NamedAndNotLocal` | `covered` | `test/transport/lowfive_test.cpp` :: `LowFiveTransport.NameAndLocality` | Identical name/is_local() checks on LowFiveTransport. | verified |
| `LowFiveTransport.PrepareAllocatesNothingPerStep` | `likely_covered` | `test/parallel/mpi_transport_test.cpp` :: `MpiTransport.PrepareAndMoveAllocateNothingPerStep` | Same allocation-counting-per-step pattern, different transport and narrower scope (prepare only). | triage |
| `LowFiveTransport.PrepareBeforeBuildPlanIsAnError` | `likely_covered` | `test/parallel/mpi_transport_test.cpp` :: `MpiTransport.PrepareAndMoveRefuseAnUnplannedTransport` | Same unplanned-transport refusal (E-TRANSPORT-UNPLANNED) scenario, different transport. | triage |
| `LowFiveTransport.PrepareRejectsADifferentSpec` | `likely_covered` | `test/parallel/mpi_transport_test.cpp` :: `MpiTransport.PrepareRefusesADifferentSpecThanWasPlanned` | Same respec-rejection scenario (E-TRANSPORT-RESPEC), different transport. | triage |
| `LowFiveTransport.RejectsAChannelWithNothingToIntercept` | `missing` | — | Zero-element build_plan rejection for LowFive not matched; MPI cases test different size-mismatch scenarios. | triage |
| `LowFiveTransport.RejectsAnUnwritableLocation` | `missing` | — | No fuse test covers rejecting an unwritable LowFive location path. | triage |
| `LowFiveTransport.RejectsEmptyNames` | `missing` | — | No fuse test covers empty path/field name rejection for LowFive set_location. | triage |
| `LowFiveTransport.SetLocationAfterBuildPlanIsAnError` | `missing` | — | No fuse test covers rejecting set_location after build_plan already locked it. | triage |
| `LowFiveTransport.TheNameIsReleasedWhenTheTransportDies` | `missing` | — | No fuse test covers filename reclamation after transport destruction. | triage |
| `LowFiveTransport.TwoTransportsCannotClaimOneFileName` | `missing` | — | No fuse test covers two transports contending for the same filename. | triage |
| `LowFiveTransport.WritesNoFileWhenCompiledOut` | `superseded_behavior_change` | `test/transport/lowfive_test.cpp` :: `LowFiveTransport.EveryEntryPointRefusesWithoutLowFive` | Deliberate change (I4, no silent fallback): fuse refuses with E-TRANSPORT-NOLOWFIVE on every entry point instead of soma's silent no-op degrade. | verified |
| `TransportRegistry.LowFiveIsRegisteredInEveryConfiguration` | `missing` | — | No fuse test creates "lowfive" via TransportRegistry; only mpi/copy registry creation shown. | triage |

### `test/unit/spacetime_conservation_test.cpp` (23 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `PlainChainRunner.ASharedValueIsNotDeliveredAcrossTheTargetTick` | `missing` | — | Checks plain (non-spacetime) chain value visibility boundaries across target ticks; no candidate matches this timing check. | triage |
| `SpacetimeColumnSweep.DegreeChangesTheMatrixOnANonNestingPair` | `missing` | — | Checks Constant vs DG(1) produce different matrices on non-nesting but same on nesting pairs; no fuse case tests this. | triage |
| `SpacetimeColumnSweep.NonNestingSchedulesAloneCarryTheColumnIdentity` | `likely_covered` | `test/backend/spacetime_test.cpp` :: `Spacetime.ColumnSumsCarryTheSourceStepLength` | Same column-sum/conservation defect check across bases, though not isolated to non-nesting ratios specifically. | triage |
| `SpacetimeColumnSweep.TheRowFormOfTheConservationIdentityIsFalse` | `missing` | — | Asserts a specific row-form conservation formula is false for subcycling; no equivalent fuse assertion found. | triage |
| `SpacetimeColumnSweep.TheSweepCoversBothNestingClassesAndBothDirections` | `missing` | — | Meta-check on the test harness's own sweep_ratios() coverage counts; no fuse case inspects its own fixture generator. | triage |
| `SpacetimeComposition.APerturbedSpatialWeightShowsUpInTheSpaceTimeDefect` | `missing` | — | Mutation sensitivity of composed defect to a spatial weight change; no matching fuse case. | triage |
| `SpacetimeComposition.APerturbedTemporalWeightShowsUpInTheSpaceTimeDefect` | `missing` | — | Mutation sensitivity of the composed space-time defect to a temporal weight change; absent from fuse suite. | triage |
| `SpacetimeComposition.BothPropertiesSurviveWhenTheTotalMeasuresAgree` | `missing` | — | Tests tensor product T x W defects when total measures match; no fuse test composes temporal and spatial weights this way. | triage |
| `SpacetimeComposition.ConservationAndConstantPreservationAreExclusive` | `missing` | — | Checks mismatch-induced tradeoff between row/column defects in T x W composition; no fuse equivalent. | triage |
| `SpacetimeCompositionMoab.RealWeightsOnAnAxisAlignedPair` | `missing` | — | Composes temporal projection with real MOAB-built spatial weights on an axis-aligned box pair; no fuse analog. | triage |
| `SpacetimeCompositionMoab.RealWeightsOnMoabsCubedSphere` | `missing` | — | Composes temporal projection with cubed-sphere MOAB weights; not tested in candidate fuse cases. | triage |
| `SpacetimeConstrain.DoesNotFireWithoutAConservationClass` | `missing` | — | Verifies zero checks fire when Conservation::None; no candidate tests this no-op case. | triage |
| `SpacetimeConstrain.FiresOncePerTargetStepPerPeriod` | `missing` | — | Counts conservation_checks per target step per period for a spacetime channel; candidate counts one check per period, different scenario. | triage |
| `SpacetimeConstrain.IsEmittedAndSitsAfterTheRemapAndBeforeTheSink` | `missing` | — | Checks Constrain op's position relative to Interpolate/Remap/Sink; candidate chain-comparison test checks a different property. | triage |
| `SpacetimeConstrain.ReportsExactlyZeroAndIsBlindToTheTemporalOperator` | `missing` | — | Checks conservation error is exactly zero and insensitive to temporal basis degree; no matching fuse assertion. | triage |
| `SpacetimeConstrain.TheCrossMeasureChannelIsNowRefusedAtSetup` | `missing` | — | Refusal due to mismatched total measure between box and line meshes; candidate refusal cases test a different precondition (both nodal/multi-rank). | triage |
| `SpacetimeMutation.AUniformRescaleBreaksBothProperties` | `missing` | — | Mutation test for uniform weight rescale breaking both row and column defects; not present in fuse. | triage |
| `SpacetimeMutation.AWeightMovedWithinARowIsInvisibleToTheRowCheck` | `missing` | — | Tests row-check blindness vs column-check sensitivity to an intra-row weight shift; no fuse analog. | triage |
| `SpacetimeMutation.TheColumnCheckCatchesAPerturbationOfEveryWeight` | `missing` | — | Mutation-sensitivity sweep over every nonzero weight of the temporal matrix; no matching fuse test. | triage |
| `SpacetimeRunner.ConservationHoldsForTwoGenuinelyDifferentOperators` | `missing` | — | Confirms Constant vs DG(1) deliver different values through the runner on a non-nesting schedule; no fuse equivalent. | triage |
| `SpacetimeRunner.EachProjectionIsConsumedByTheNextTargetAdvance` | `missing` | — | Verifies per-target-step projection consumption timing via the runner; not matched by any candidate. | triage |
| `SpacetimeRunner.EveryTargetStepProjectionReachesTheTargetComponent` | `missing` | — | Checks exactly-once, in-order delivery of each target-step projection; no fuse case with this assertion. | triage |
| `SpacetimeRunner.TheTimeIntegralIsConservedThroughTheRunner` | `missing` | — | Checks space-time integral conservation delivered through the actual Runner for a dedicated spacetime harness; no equivalent fuse scenario. | triage |

### `test/unit/spacetime_runtime_test.cpp` (21 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `SpacetimeLowering.EmitsASampleAndAnInterpolateCarryingTheBasis` | `missing` | — | No fuse case checks DAG Sample/Interpolate node emission or Waveform/Instant typing for spacetime lowering. | triage |
| `SpacetimeLowering.EmitsNeitherNodeWithoutASpacetimeSpec` | `missing` | — | No fuse case checks absence of spacetime nodes when spec disabled. | triage |
| `SpacetimeLowering.ScheduleRunsSampleOnSourceStepsAndProjectionOnTarget` | `missing` | — | No fuse case checks Sample/Interpolate instruction placement at source/target steps for spacetime. | triage |
| `SpacetimeMatrix.DegreeFourMatchesTheHandComputation` | `missing` | — | No candidate asserts exact numeric projection matrix coefficients at degree 4. | triage |
| `SpacetimeMatrix.DegreeThreeMatchesTheHandComputation` | `missing` | — | No candidate asserts exact numeric projection matrix coefficients at degree 3. | triage |
| `SpacetimeMatrix.DegreeTwoMatchesTheHandComputation` | `missing` | — | No candidate asserts exact numeric projection matrix coefficients at degree 2. | triage |
| `SpacetimeMatrix.StencilPlacementMatchesTheDocumentedFormulaAcrossASweep` | `missing` | — | No candidate sweeps degree/step counts checking stencil start/width formula. | triage |
| `SpacetimeMatrix.TheMatrixMatchesTheHandComputation` | `missing` | — | Candidates test slab/degree rejection, not numeric hand-computed projection coefficients. | triage |
| `SpacetimeRefusal.ADegreeTheBasisDoesNotAdmitIsRejected` | `likely_covered` | `test/backend/spacetime_test.cpp` :: `Spacetime.DegreeMustMatchTheBasis` | Same basis/degree mismatches and E-ST-DEGREE code, but checked at projection builder not runtime setup. | triage |
| `SpacetimeRefusal.ANestingPartitionAdmitsAHigherDegree` | `missing` | — | No fuse case checks acceptance of higher-degree DG under nesting partition at runtime setup. | triage |
| `SpacetimeRefusal.ANonCausalStencilIsRejectedRatherThanLagged` | `missing` | — | No fuse case checks causal-stencil rejection (E-RT-SPACETIME) for non-causal step nesting at runtime setup. | triage |
| `SpacetimeRefusal.AWindowOtherThanTheCommonPeriodIsRejected` | `missing` | — | No fuse case checks window-length mismatch rejection (E-RT-SPACETIME) for spacetime channel. | triage |
| `SpacetimeRefusal.AccumulationCombinedWithSpacetimeIsRejected` | `missing` | — | No fuse case checks rejection (E-SCHED-SPACETIME) of accumulation combined with spacetime channel. | triage |
| `SpacetimeRefusal.DegreeZeroIsAcceptedInBothDirections` | `missing` | — | No fuse case checks degree-0 acceptance in both refining/coarsening directions at runtime setup. | triage |
| `SpacetimeRefusal.SplineIsRejected` | `likely_covered` | `test/backend/spacetime_test.cpp` :: `Spacetime.SplineBasisIsRejected` | Same rejection (Spline basis, E-ST-BASIS) but via direct projection builder, not full runtime setup. | triage |
| `SpacetimeRefusal.TheDefaultCoarseningScheduleIsAcceptedAndWouldTripOnAMirroredLean` | `missing` | — | No fuse case checks acceptance of default coarsening spacetime schedule at full runtime setup. | triage |
| `SpacetimeRuntime.AConstantInTimeSourceArrivesUnchanged` | `missing` | — | No fuse case feeds a frozen constant source through spacetime projection and checks exact passthrough. | triage |
| `SpacetimeRuntime.DegreeOneIsADifferentOperatorFromDegreeZero` | `missing` | — | No fuse case compares spacetime runtime projected value against both copy and degree-0 result. | triage |
| `SpacetimeRuntime.DegreeZeroProjectsTheStepMeansAndNotTheLatestValue` | `missing` | — | No fuse case runs a spacetime runtime harness checking degree-0 projected value equals step mean. | triage |
| `SpacetimeRuntime.LinearAndDgOneAgree` | `missing` | — | No fuse case runs harness comparing Linear basis vs DG degree-1 projected runtime output. | triage |
| `SpacetimeRuntime.TheProjectionReachesTheConsumersArray` | `missing` | — | No fuse case verifies spacetime-projected value lands in consumer's bound array after run_period. | triage |

### `test/unit/remap_runtime_test.cpp` (8 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `RemapRuntime.AbortsOnAConservationViolationWhenAsked` | `missing` | — | Nearest candidate disables conservation checking; none test abort_on_violation causing run_period failure with E-RT-CONSERVATION. | triage |
| `RemapRuntime.AllocatesNothingWhileRunningARealRemap` | `likely_covered` | `test/runtime/runner_test.cpp` :: `Runner.AllocatesNothingWhileRunning` | Same zero-allocation-during-run() assertion on a subcycled pipeline including a Remap step; real-remap mesh mismatch unconfirmed. | triage |
| `RemapRuntime.ConservesTheMeasureWeightedIntegralThroughTheRunner` | `missing` | — | No candidate checks Runner.run() conserves a measure-weighted integral across non-nesting meshes via conservation stats. | triage |
| `RemapRuntime.PermutedPairIsNotAnIdentityCopy` | `missing` | — | No candidate tests Runner-level remap on a permuted mesh distinguishing centroid-match delivery from index copy. | triage |
| `RemapRuntime.PreservesAConstantThroughTheRunner` | `missing` | — | Candidates test coincident-identity or per-element profile delivery, not constant preservation through a real non-nesting remap. | triage |
| `RemapRuntime.RefusesATruncatingMapWhenTheChannelAsksForConservation` | `missing` | — | No candidate reproduces E-REMAP-COVERAGE refusal when a Conservative-flagged channel maps onto a truncating target. | triage |
| `RemapRuntime.RegisteredAddressesDoNotMove` | `missing` | — | No candidate checks that a registered producer buffer's address stays stable across repeated Runner.run() periods. | triage |
| `RemapRuntime.ReportsTheMassATruncatingRemapDrops` | `missing` | — | No candidate checks Runner conservation-violation stats/worst_error for a truncating (partial-coverage) remap. | triage |

### `test/unit/spacetime_order_test.cpp` (4 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `SpacetimeOrder.DegreeIsClampedToTheAvailableStepCount` | `missing` | — | No fuse test exercises degree clamping to available source steps or checks effective_degree. | triage |
| `SpacetimeOrder.DegreePPolynomialsAreReproducedExactly` | `missing` | — | Candidate only reproduces a constant (degree 0); this case tests exact degree-p polynomial reproduction up to kMaxTemporalDegree. | triage |
| `SpacetimeOrder.DgFourIsFifthOrder` | `missing` | — | No fuse test measures empirical convergence order of DG(p) projections; candidates only check identity/averaging/row-sums. | triage |
| `SpacetimeOrder.DgOneMatchesLinear` | `missing` | — | No fuse test compares DG(1) to Linear basis for exact row/col/val equality or measures convergence order. | triage |

### `test/unit/remap_order_test.cpp` (3 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `RemapOrder.ObservedOrderMatchesTheClaimedOrder` | `missing` | — | No candidate measures convergence order across orders 1-4 with TempestRemap/supermesh builder selection. | triage |
| `RemapOrder.SkippedWithoutTheHighOrderBackend` | `missing` | — | No candidate is a build-config skip guard for missing TempestRemap/MOAB backend. | triage |
| `RemapOrder.TheAnswerDoesNotDependOnTheReferenceQuadrature` | `missing` | — | No candidate checks quadrature-rule invariance of measured remap error. | triage |

### `test/unit/runner_test.cpp` (2 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `Runner.RefusesARemapBetweenDifferentlySizedRegionsItCannotBuildWeightsFor` | `superseded_behavior_change` | `test/runtime/runner_test.cpp` :: `Runner.RemapsBetweenDifferentlySizedRegions` | fuse's test comment: 'This test asserted a REFUSAL until the dependency-free supermesh landed ... It now succeeds, because that is the capability that was added.' Same harness (src_cells=8,dst_cells=4), ASSERT_FALSE -> ASSERT_TRUE. | verified |
| `Runner.RefusesEqualCountEqualMeasureRegionsInADifferentArrangement` | `superseded_behavior_change` | `test/runtime/runner_test.cpp` :: `Runner.RemapsEqualCountEqualMeasureRegionsInADifferentArrangement` | Identical harness (src/dst mesh, cell counts) to the soma case; fuse's own comment confirms the same deliberate change: 'Also a refusal until the supermesh landed ... The supermesh does not need to see it -- it computes the actual overlaps.' | verified |

### `test/unit/ir_dump_test.cpp` (1 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `IrDump.OmegaIsFormattedLocaleIndependently` | `likely_covered` | `test/ir/ir_dump_test.cpp` :: `IrDump.TextIsIdenticalUnderAGroupingLocale` | Same scenario (Relax omega=0.7) and same assertion: dump_text decimal formatting unaffected by locale. | triage |

### `test/unit/smoke_test.cpp` (1 cases)

| Case | Verdict | Evidence (fuse case) | Reason | Confidence |
|---|---|---|---|---|
| `Smoke.VersionIsNonEmpty` | `missing` | — | No fuse::version() free function and no test calls FUSE_ABI_ENTRY(version) to check fuse_version_info.string is non-empty; only struct_size is golden-tested. | triage |

## C ABI mapping: `soma_c.h` -> `fuse_coupler_abi()`

`docs/dev/porting.md`'s own naming map rules this a replacement, not a rename ("`soma_c.h` flat C
API -> the `fuse_coupler_abi()` table | not renamed: **replaced**, see abi-design.md"). All 21
`SOMA_C_API` entries map onto `include/fuse/abi/entries.def`'s 16 `FUSE_ABI_ENTRY` names, with the
merges below confirmed by reading `include/fuse/abi/types.h`'s struct definitions directly.

| `soma_c.h` | `fuse_coupler_abi()` | Evidence |
|---|---|---|
| `soma_version` | `version` | Same-shape entry; `fuse_version_info` replaces a bare C string. |
| `soma_abi_version` | folded into `version`'s output | `fuse_version_info.abi_version` field (`include/fuse/abi/types.h`). |
| `soma_create` | `create` | Direct. |
| `soma_destroy` | `destroy` | Direct. |
| `soma_last_error` | `last_error` | Direct. |
| `soma_last_error_length` | folded into `last_error` | Same struct-based call returns length and copies the message in one call (no separate sizing call in the ABI entry list). |
| `soma_load_graph` | `load_graph` | Direct. |
| `soma_load_graph_yaml` | folded into `load_graph` | Convenience path-vs-bytes wrapper, same merge pattern as the Fortran side. |
| `soma_save_graph` | `save_graph` | Direct. |
| `soma_field_handle` | `field_handle` | Direct. |
| `soma_register_field` | `register_field` | Direct. |
| `soma_set_communicator` | folded into `setup` | `fuse_setup_options.communicator` (`types.h` line 137), plus a `my_component` field soma never had (MPMD/Connect launch). |
| `soma_set_end_time` | folded into `setup` | `fuse_setup_options.has_end_time`/`end_time` (`types.h` lines 138,140), same struct as `communicator`. |
| `soma_initialize` | `initialize` | Direct. |
| `soma_begin_step` | `begin_step` | Direct. |
| `soma_end_step` | `end_step` | Direct. |
| `soma_requires_checkpoint` | folded into `begin_step`/`end_step` report | `fuse_step_report.requires_checkpoint` (`types.h` line 198), read off the step report instead of queried separately. |
| `soma_requires_restore` | folded into `begin_step`/`end_step` report | `fuse_step_report.requires_restore` (`types.h` line 199), same merge. |
| `soma_is_coupling_ongoing` | `is_coupling_ongoing` | Direct. |
| `soma_run_period` | folded into `run` | Single `run` entry replaces the period/open-ended pair; both soma entries returned `SOMA_ERR_UNSUPPORTED` in every soma build in any case (soma_c.h's own doc comment). |
| `soma_run` | `run` | Direct (see above). |
| — (not in soma_c.h) | `capabilities` | fuse addition: capability-bit query, no soma counterpart. |
| — (not in soma_c.h) | `view_field` | fuse addition: read-back of a registered field view, no soma counterpart. |

## Fortran ABI mapping: `soma_mod` -> `ffuse`

`fortran/soma_mod.f90`'s 24 `soma_f_*` wrappers map onto `fortran/ffuse.f90`'s 16 `ffuse_*` entries,
1:1 with the C table above (both languages wrap the identical `fuse_coupler_abi()` struct calls).

| `soma_mod` routine | `ffuse` routine | Note |
|---|---|---|
| `soma_f_version` | `ffuse_version` | Direct |
| `soma_f_abi_version` | folded into `ffuse_version` | Same `fuse_version_info.abi_version` merge as the C side |
| `soma_f_create` | `ffuse_create` | Direct |
| `soma_f_destroy` | `ffuse_destroy` | Direct |
| `soma_f_is_open` | **none** | Fortran-only convenience over the raw ctx handle; not in `soma_c.h` either. Recorded here as intentionally dropped, not silently missing. |
| `soma_f_last_error`, `soma_f_last_error_length` | `ffuse_last_error` | Merged, same as C side |
| `soma_f_load_yaml`, `soma_f_load_descriptor`, `soma_f_descriptor_size` | `ffuse_load_graph` | Merged convenience wrapper |
| `soma_f_save_descriptor` | `ffuse_save_graph` | Direct |
| `soma_f_field_handle` | `ffuse_field_handle` | Direct |
| `soma_f_register_field_1d`, `soma_f_register_field_2d` | `ffuse_register_field` | One generic `fuse_field_view`-based entry instead of two rank-specific ones |
| `soma_f_set_communicator`, `soma_f_set_end_time` | `ffuse_setup` | Merged into `fuse_setup_options`, as the C side |
| `soma_f_initialize` | `ffuse_initialize` | Direct |
| `soma_f_begin_step` | `ffuse_begin_step` | Direct |
| `soma_f_end_step` | `ffuse_end_step` | Direct |
| `soma_f_requires_checkpoint`, `soma_f_requires_restore` | folded into `ffuse_begin_step`/`ffuse_end_step` report | Same `fuse_step_report` merge as the C side |
| `soma_f_is_coupling_ongoing` | `ffuse_is_coupling_ongoing` | Direct |
| `soma_f_run_period`, `soma_f_run` | `ffuse_run` | Merged |

## What remains in `upstream/soma` that FUSE still needs

**Not nothing.** Per the file-level and per-case tables above:

- 12 files are classified `missing` outright (python/ bindings x6, `include/soma/soma.hpp`'s
  `version()` free function, `test/tools/make_partitioned_sphere.cpp`, `test/parallel/capi_parallel_test.c`,
  `test/fortran/capi_test.f90`, plus the two MOAB-backend parallel-remap harnesses).
- 11 files are `partially ported` (10 of the problem test files plus `test/unit/capi_test.c`): each
  still has specific cases/behaviours with no fuse counterpart, enumerated case-by-case above.
- 1 file (`moab_parallel_test.cpp`) is `likely_covered_partial`: the placement/view machinery it
  exercises under real MPI is covered, the specific ownership-count arithmetic identity is not.
- 112 of the 135 triaged cases are `missing` at triage confidence (not individually re-proven).

None of this blocks classifying the subtree overall, but it does mean `upstream/soma` is **not** yet
reducible to nothing FUSE needs — see `design_notes`/`open_questions` in the submitted report for what
step 2 of this track actually ported (a prioritized subset, with mutation proof) versus what remains
an open backlog.

## Ported in this track, with mutation proof

Given the triage-then-port scope ruling above, this track ported a small, fully-verified set of
confirmed-missing behaviour rather than attempting breadth across all 112 case-level `missing`
rows. Each row below passed clean, was independently killed by a hand-applied mutation (FAILED/PASSED
counts from the mutation's own run, below), was restored and reconfirmed passing, and is recorded in
`build-evidence/mutations/soma-import.toml`.

| New/extended file | Ported case(s) | Source soma case(s) | Mutation id | Mutation result (mutated / clean) |
|---|---|---|---|---|
| `test/backend/remap_apply_test.cpp` (new) | `RemapApply.RejectsBadArguments`, `RemapApply.RejectsOutOfRangeTriplets` | `test/unit/remap_test.cpp` | `remap-apply-zero-components-not-rejected` | 1 FAILED / 1 PASSED  →  2 PASSED / 0 FAILED |
| `test/backend/supermesh_test.cpp` (extended) | `RemapPartial.SourceWiderThanTargetIsRefusedWhenConservativeIsAsked`, `RemapPartial.SourceWiderThanTargetTruncatesWhenConservationIsNotAsked` | `test/unit/remap_test.cpp` | `supermesh-conservative-coverage-not-enforced` | 1 FAILED / 25 PASSED  →  26 PASSED / 0 FAILED |
| `test/abi/participant_robustness_test.cpp` (new) | `AbiParticipantRobustness.VersionStringIsNonEmpty`, `.DoubleDestroyIsReportedNotCrashed`, `.ThreadLocalErrorReadableWithNoLiveContext` | `test/unit/smoke_test.cpp` (`Smoke.VersionIsNonEmpty`), `test/unit/capi_test.c` (`test_double_destroy_and_stale_handles`, `test_thread_local_error_without_a_context`) | `abi-double-destroy-not-reported` | 1 FAILED / 2 PASSED  →  3 PASSED / 0 FAILED |

**A production-code finding, fixed as part of this port (owner-approved 2026-10-12).** Porting
`RemapPartial.SourceWiderThanTargetIsRefusedWhenConservativeIsAsked` surfaced a real gap, not a test
bug: on a tier-0 (no-MOAB) build, `fuse::build_weights()`'s dependency-free supermesh path
(`src/backend/intx_weights.cpp`, the `#if !FUSE_HAVE_MOAB` half) never inspected `plan.flags` for
`RemapFlags::Conservative` at all -- only `RemapFlags::HighOrder` was checked there. The
`E-REMAP-COVERAGE` refusal existed only in the `#else // FUSE_HAVE_MOAB` half of the same file, so a
channel explicitly marked `.remap(RemapFlags::Conservative)` (set in a dozen places across
`test/{model,schedule,ir,abi}`) got no enforcement at all on a tier-0 build when coverage was
incomplete: uncovered source mass was dropped silently, exactly what the flag exists to prevent.
Fixed by moving the shared `kCoverageTol` constant out of the MOAB-only block (it is a property of
the backend-agnostic coverage audit, not of MOAB) and adding the identical check, same tolerance,
same error code, to `build_weights_via_supermesh`. See the mutation row
`supermesh-conservative-coverage-not-enforced` above for the kill proof.

**Not ported in this track (backlog).** Of the 112 case-level `missing` rows, 5 were ported above
(`RemapApply.RejectsBadArguments`, `RemapApply.RejectsOutOfRangeTriplets`,
`RemapPartial.SourceWiderThanTargetIsRefusedWhenConservativeIsAsked`,
`RemapPartial.SourceWiderThanTargetTruncatesWhenConservationIsNotAsked`, `Smoke.VersionIsNonEmpty`) and 2
were independently verified by direct re-read and confirmed still missing without being ported
(`LowFiveTransport.SetLocationAfterBuildPlanIsAnError`,
`SpacetimeRunner.EveryTargetStepProjectionReachesTheTargetComponent`, both from the safety-net re-check in
the Methodology section). The remaining 105 case-level `missing` rows are still triage-confidence, neither
independently re-read nor ported. Separately, `test/unit/capi_test.c` (a file-level `partially ported` row,
not part of the 135-case table at all, since it has no `TEST()` macros) had 2 of its 26 `test_*()` functions
ported above (`test_double_destroy_and_stale_handles`, `test_thread_local_error_without_a_context`); its
other 24 behaviours (null-argument rejection on most entries, error-message retrieval, yaml load,
descriptor round trip, register_field validation, state errors, convergence-criterion rejections, and
more) remain unported. The three MOAB-backend parallel-harness files
(`moab_parallel_test.cpp`, `remap_parallel_test.cpp`, `remap_halo_parallel_test.cpp` -- these need
`test/tools/make_partitioned_sphere.cpp`, itself unported, and a tier-1/2 MOAB+MPI build this
track's budget did not reach), and the python bindings (6 files; `pfuse` is a ruled name per
`docs/dev/porting.md` but no package exists yet under `FUSE_ENABLE_PYTHON`, which remains a stub
option) are open items for a follow-on track, not silently dropped.
