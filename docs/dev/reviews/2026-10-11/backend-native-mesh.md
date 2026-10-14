# Review 2026-10-11 — `backend`: native distributed mesh

Read-only review at HEAD `4a8a9d8`. Scope read in full: `include/fuse/backend/{native_mesh, native_backend,
native_halo, native_migration, native_mesh_io, mesh_view}.hpp`, `include/fuse/backend/native/{mesh,
geometry, cell_topology}.hpp`, `src/backend/{native_mesh, native_backend, native_geometry, native_halo,
native_migration, native_mesh_io, native_generators}.cpp`, the native tests, `intx_weights.cpp` dispatch
seams, `linalg/comm{.hpp,_mpi.cpp}`, and the layer docs. One compile-and-run experiment (default
`ostream` precision) was performed in `/tmp/opencode` to confirm NM-1; no repository file was touched.
Findings feed the ranked index in `index.md`; IDs here are referenced as `NM-*`.

## 1. Findings

| ID | Severity | file:line | Finding | KNOWN |
|---|---|---|---|---|
| NM-1 | **BLOCKER** | `src/backend/native_mesh_io.cpp:63,75,115,124` | VTK writer emits only 6 significant digits (no `precision(17)` anywhere): global ids ≥ 1e6 silently corrupted, coordinates rounded; contradicts the documented "exact" round trip | not recorded |
| NM-2 | **MAJOR** | `src/backend/native_halo.cpp:421-433,447,475,479-483` | `build_native_remap_halo` performs per-rank rejections before (and after) its collectives with no cross-rank reconciliation — the exact R-01/R-02/N-BACKEND-1 (I2) class `migrate()` guards against | not recorded (native path) |
| NM-3 | **MAJOR** | `include/fuse/backend/native_backend.hpp:169-184`; `src/backend/native_halo.cpp:303-316`; `test/parallel/native_halo_mpi_test.cpp:164-171` | `remap_halo_pattern`'s documented `make_mpi_exchanger` fill sequence is silently wrong for a second target on the same source (ghost slots assumed to start at `owned_n`), and assumes owned cells are a prefix, which `mesh_view.hpp:34-38` explicitly does not guarantee | not recorded |
| NM-4 | MINOR | `src/backend/native_backend.cpp:24,323` | halo-pattern bookkeeping keyed `(source,target,dim)` without `placement`; two placements doing the same triple clobber each other's pattern | not recorded |
| NM-5 | MINOR | `src/backend/native_halo.cpp:411-420,435` | `build_native_remap_halo` validates communicator *size* but not *rank*: `backend.rank()` is used for self-exclusion/wire owner while box owners are communicator ranks — latent for sub-communicator adoption | not recorded |
| NM-6 | MINOR | `src/backend/native_migration.cpp:334-338` | post-exchange per-rank failures in `migrate` (unpack / `create_from_host`) return unreconciled while peers enter `resolve_vertex_ownership`'s Allgatherv (line 337) | not recorded |
| NM-7 | MINOR / DEVIATION | `include/fuse/backend/native_backend.hpp:248-249` vs `src/backend/native_generators.cpp:42-68` | documented `E-MESH-SIZE` never exists in the native generator (overflow → `E-MESH-SOURCE`); a huge spec instead crashes via uncaught `std::bad_alloc`/`length_error` (`native_generators.cpp:144` `lut.local.assign(sx*sy*sz, -1)`) | not recorded |
| NM-8 | MINOR | `src/backend/native_mesh.cpp:371-382,446-459` | `create_from_host` does not validate fixed-shape corner counts or `dim∈{0..3}` despite the header's "validates" claim (`native_mesh.hpp:21-26`): a Triangle with 2 corners / Hex with 4 is `std::vector::operator[]` OOB (UB); `dim≥4` silently yields measure 0 and no simplices. All in-tree producers route through `NativeMeshBuilder::build()`, which does check — latent via the public direct call | not recorded |
| NM-9 | MINOR | `src/backend/native_geometry.cpp:120-128` | an exactly-degenerate fan triangle (collinear corner) is refused `E-MESH-POLYGON-NOT-STAR-SHAPED`: with `tri_norm == 0` the test degenerates to `0 <= 0` → refuse. The doc's own identity says only a sign *disagreement* invalidates; flat vertices on straight polygon edges (midside nodes) are valid but rejected | not recorded |
| NM-10 | MINOR | `src/backend/native_mesh.cpp:59-77` | `find_or_add_face`: same vertex *set* that is neither a forward nor reverse cyclic match (duplicate corners, or same set at different length) silently reuses the stored face with `flip=false` instead of refusing — a named rejection per I4 would be correct | not recorded |
| NM-11 | MINOR | `src/backend/native_mesh.cpp:509-551` | Point-cell meshes (dim 0; supported by `CellType::Point` and VTK type 1 in IO) are shadowed: `entity_count(0)/owned_count(0)/global_id(0)/centroid(0)` answer *vertex* data; Point cells are unreachable through `MeshView`. No generator emits them | not recorded |
| NM-12 | MINOR | `src/backend/native_migration.cpp:170-171`; `native_halo.cpp:332`; `native_migration.cpp:275`; `native_mesh.cpp:21,136-156` | int overflow on large meshes: per-rank byte counts narrowed to `int` in `exchange_migration_buffers`, box/vertex counts as `int`, builder local indices/CSR offsets as `int`, `add_vertex` returns `int` | not recorded |
| NM-13 | MINOR | `src/backend/native_mesh_io.cpp:149,154` | `TokenReader`'s `std::istringstream`s are not locale-pinned, violating conventions.md:88-93 ("pin `std::locale::classic()` on **every** stream it creates"); the writer pins (`:63`), the reader's per-line streams do not. Only `iss >> dim` extracts numerically (single digit — no practical impact today) | not recorded |
| NM-14 | IMPROVEMENT | `src/backend/native_halo.cpp:393,415-419` | `std::to_string` in diagnostics while sibling files use `fmt::format` — conventions.md:52 (fmt only) and its own `std::to_string`-reads-the-locale warning | not recorded |
| NM-15 | DEVIATION | `docs/dev/native-mesh.md:160-177` | "What is verified where" is stale: it says the Tier-1 migration MPI tests and `layout.migrate` rows are "compiled, awaiting host run", but D24's host measurement is recorded 2026-10-09 (`native_backend.cpp:97-103`, KNOWN_ISSUES D24) and `native-halo.md:110` records `parallel.native_halo_mpi.{2,3}` passing 2026-10-10 | partially (D24/D27 record the facts; the page wasn't updated) |

### Detail — NM-1 (the one that changes results)

`write_native_mesh_file` pins the locale but never sets stream precision, so `f << static_cast<double>(gid)`
/ `f << coord` emit **6 significant digits** (verified: `1234567` → `"1.23457e+06"` → read back `1234570`;
`1/24` → `"0.0416667"`). Consequences: (a) any global id ≥ 1e6 with ≥ 7 significant digits is silently
changed on round trip — and since gids are dense in the generators, `1030300` read back for `1030301`
*collides with a different real vertex*, corrupting the very matching `mesh_view.hpp:126-139` says gids
exist for (migration and halo dedup are both by gid, `native_migration.cpp:229`, `native_halo.cpp:279`);
(b) coordinates round to 6 digits, so the round trip is not exact — `native-mesh.md:98-99` ("exact for
every integer up to 2^53"; "A round trip reproduces the mesh exactly") and `native_mesh_io.hpp:24-26`
are both false as written. The round-trip tests escape because `hexbox:3x2x2`'s rounded widths
(`0.333333+0.333334+0.333333`) sum to exactly 1.0 and every gid is < 1e6, and the test compares only
counts, gid equality (small), type, and the *sum* of measures (`native_generators_test.cpp:340-351`).
Fix: one line — `f << std::setprecision(17);` after the `imbue` (and consider asserting gids < 2^53 on write).

### Detail — NM-2 (the R-01/R-02/N-BACKEND-1 class, native path)

`migrate()` computes local validity and agrees it with `MPI_Allreduce(MIN)` *before* the exchange
(`native_migration.cpp:317-331`) — the documented reconciliation pattern (`native-mesh.md:128-133`).
`build_native_remap_halo` has no equivalent: `backend.native_view(...)` rejections (`E-MESH-ID`,
`E-MESH-PLACEMENT`, `native_halo.cpp:421-424`) and the data-dependent `E-REMAP-DIM` (`:427-433`) return
per-rank **before** `allgather_boxes` (`:447`) and `exchange_migration_buffers` (`:475`); and
`merge_halo_cells`/`create_from_host` failures (`:479-483`) return per-rank after the exchange while
peers proceed to later collectives of the caller. In-tree triggers are currently benign (every rank reads
the same generated spec or file, and `register_default_placements` registers every placement on every
rank), which is why this is MAJOR rather than BLOCKER — but the project's own precedent rates this
structural class BLOCKER when a trigger is confirmed, and the migration test proves one-rank-bad-assignment
is a tested, real scenario for the sibling function. Fix: mirror `migrate()` — `Allreduce(MIN)` the local
verdict after the dimension check and before `allgather_boxes`, and reconcile the post-merge verdict the
same way before returning to the caller's next collective.

### Detail — NM-3 (halo pattern vs the exchanger's slot contract)

`merge_halo_cells` appends a second target's adopted cells after the first's, so the mesh layout is
`[owned, ghosts_V, ghosts_W]` (`native_halo.cpp:240-316`, supported explicitly by
`native_backend.hpp:127` "a second call for a different `target` ... adopts only what that pair still
needs"). But `GhostPattern`/`make_mpi_exchanger` semantics place ghosts at `data()[owned_n + g]`
(`comm.hpp:75-81,114-118`), and both consumers index `src_field[c - owned_n]`
(`native_halo_mpi_test.cpp:169`, `native_halo_test.cpp:613`). For target W this reads ghosts_V slots
with ghosts_W values and leaves the true W slots untouched — silently wrong field values in a
conservative remap. The same fill assumes owned cells are a prefix; true for
generator/migration/halo meshes, not enforced for file-read meshes (`read_native_mesh_file` preserves
file order, `native_mesh_io.cpp:337-382`; `mesh_view.hpp:34-38` says a prefix is *not* guaranteed). No
`src/` code consumes `remap_halo_pattern` today (grep: only tests), so no in-tree wrong result — but the
documented contract ("plugs directly into `make_mpi_exchanger`", `native_backend.hpp:176-177`; D27's
closure text, `KNOWN_ISSUES.md:3540-3542`) is wrong for the second-target case. Fix options: record an
`owned_offset` per pattern (or key patterns by target *and* return the append offset), and either
document + assert the owned-prefix invariant for NativeMeshView or make the pattern carry local indices
instead of exchanger slots.

## 2. Checked and found clean

- **I7 / no third-party MPI types in public headers**: `CommHandle`-as-`int` convention held everywhere; no `mpi.h` in any `include/fuse/backend/**` header (grep-verified).
- **Device-callability, current state (R-03/R-14)**: `Mesh<M,S,MemorySpace>` parameterises memory space and `finalize()` assembles through host mirrors; `measure/centroid/normal/vertex` are `KOKKOS_INLINE_FUNCTION` reading precomputed views. The **R-14 residual is fixed**: `FUSE_DEBUG_ASSERT` now splits host/device via `KOKKOS_IF_ON_HOST`/`KOKKOS_IF_ON_DEVICE` (`error.hpp:215-227`), so it is device-callable in Debug and a no-op under `NDEBUG`.
- **D27 / reverse-exchange assumption**: nothing in the tree assumes a native `accumulate_remap_halo` — the only caller is `fold_halo_contributions`, which is inside the MOAB half; the runtime's halo guard is MOAB-only. The native audit is test-level bookkeeping, exactly as `native-halo.md:117-122` records. **KNOWN: D27** (accurate).
- **Migration core correctness**: pack/dest validation, the collective Allreduce(MIN) refusal, unpack dedup by gid, min-holder-rank vertex ownership — all match `native-mesh.md`'s protocol and are tested at P=1..5 emulated plus real MPI including the only-one-rank-bad refusal and an independent min-owner re-derivation with mutation evidence.
- **Halo covering-set geometry**: box overlap is a correct superset; self-send excluded; touch-vs-overlap and the planar degenerate-axis rule match the docs and are mutation-evidenced; idempotence (same triple) tested tier-0 and MPI.
- **Shared-face triangulation consistency**: min-global-id fan, direction-symmetry argument, and the Newell/divergence cross-checks are implemented as documented, with both-sides-of-a-shared-face tests.
- **I5**: no allocation or per-value virtual dispatch in hot kernels — all native-mesh allocation is construction/setup; `NativeMeshView` queries read precomputed data.
- **I6 / conventions (errors)**: `FUSE_ASSERT`/`FUSE_THROW`/`FUSE_REFUSE` only; no bare `assert` in scope (grep-verified). (`NativeMeshView::check` throws `fuse::InvalidArgument` where `mesh_view.hpp` says `std::out_of_range`; the MOAB backend deviates the same way with a recorded rationale, so counted as ecosystem-consistent.)
- **Layout driver `--migrate`**: destination ids remapped to world ranks before `migrate`; comm, backend rank, and dest rank spaces agree.
- **`cubedsphere` seam and `--migrate` mesh-vs-field claims**: match KNOWN_ISSUES D-noted simplifications; no code contradicts them.
- **`register_default_placements`**: per-placement `(position-in-rank-set, rank-set-size)` generation matches the D24 fix; empty views for non-hosted ranks are contract-consistent.

## 3. Not reviewed

- `docs/user/geometry.md`, `docs/user/layouts.md`, `docs/dev/search.md` — listed as context but not read (time budget went to the native sources and their cross-references).
- `test/backend/native_distributed_mesh_test.cpp` and `native_mesh_test.cpp` beyond refusal/geometry spot-checks; `native_halo_test.cpp:220-280` (oracle construction detail).
- MOAB/serial backend internals (out of scope; touched only at dispatch seams), `supermesh.cpp` internals, `nodal_remap_distributed.cpp`, `consistent_interp`/`general_interp`/`mls` (out of scope), `fuse/partition` internals.
- Whether any GPU (CUDA/HIP) leg actually instantiates `Mesh<M,S>` on device — the R-03/R-14 fixes are verified by reading, not by a device compile.

## 4. Test-coverage assessment

**Strong.** Geometry contract per shape, shared-face consistency from both sides, generator conservation + rank-count-independent gids, builder/migration/halo refusals (tier-0 *and* real-MPI), the collective one-rank-bad migration refusal, vertex-ownership min-rule with mutation evidence, halo covering set against an independent brute-force oracle, idempotence, and full build_weights→fill→apply_weights conservation checks. The tier-0 "emulated ranks" technique is genuinely sharp (a dropped/misindexed ghost value fails the constant-field check O(1)).

**Gaps that map to findings:**
1. No test fires `E-MESH-HALO` (`remap_halo_pattern` before a halo is built, `native_backend.cpp:274-279`) — per I4 this refusal "does not count as implemented" until a test makes it fire.
2. No test fires `build_native_remap_halo`'s own `E-REMAP-DIM` (`native_halo.cpp:427-433`) — needs a ≥2-rank run with disagreeing dims.
3. No test covers a second halo for a **different** target end-to-end — only same-triple idempotence — which is exactly the blind spot behind NM-3.
4. The file round-trip tests pass by cancellation (NM-1): no per-cell measure or coordinate comparison, no mesh with gids ≥ 1e6 or non-cancellation coordinates (e.g. `quad:24x24`).
5. The MPI migration rows are host-unrun per `native-mesh.md` (while the halo rows are host-passed per `native-halo.md`) — the migration collectives' real-MPI evidence is the halo rows' plus the design sharing between the two.
6. No large-mesh (int-overflow) or device-space coverage of the wire format/exchange paths — both documented as deferred, consistent with NM-12's MINOR rating.
