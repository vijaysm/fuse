# SOMA Phase 5 — MOAB backend, real remap, MPI transport

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §4.1 (MOAB backend), §5.4 (remap), §6 modes 1–2, §10 level 3 (analytic operators)
**Milestone:** two components on **non-matching** meshes, on **disjoint MPI rank sets**, exchange a field through a conservative remap, and global conservation holds to machine precision.

Phase 4 proved the machinery. Phase 5 makes it a real coupler.

---

## Environment blocker, verified 2026-09-02

`MOAB_HAVE_TEMPEST` is **not** set in the installed MOAB 5.6.0, and
`nm libMOAB.dylib | grep -c Tempest` returns 0. What *is* present:

- `moab/IntxMesh/Intx2Mesh.hpp`, `Intx2MeshInPlane.hpp`, `Intx2MeshOnSphere.hpp`, `IntxUtils.hpp`
- the iMOAB C API (`iMOAB_ComputeCommGraph`, `iMOAB_CreateElements`, …)
- `MOAB_HAVE_MPI`, `MOAB_HAVE_HDF5`, `MOAB_HAVE_HDF5_PARALLEL`

So Task 5.2 builds conservative FV↔FV weights directly on `Intx2Mesh`, which
needs nothing beyond what is installed, and Task 5.3 adds high-order and monotone
methods behind the same `RemapPlan` once TempestRemap is available. Task 5.0
rebuilds MOAB; if that rebuild fails or is deferred, Tasks 5.1, 5.2, 5.4, and 5.5
still complete and the phase still meets its milestone — only Task 5.3 is blocked.

---

## File structure

| File | Responsibility |
|---|---|
| `cmake/FindSomaMOAB.cmake` | MOAB discovery and capability detection |
| `include/soma/mesh/moab_backend.hpp` | `MoabMeshBackend` — pimpl, no MOAB types |
| `src/mesh/moab_backend.cpp` | the only file that includes `moab/Core.hpp` |
| `include/soma/remap/remap_plan.hpp` | `RemapPlan`, `RemapWeights`, `RemapCache` |
| `src/remap/intx_weights.cpp` | conservative FV↔FV weights from `Intx2Mesh` |
| `src/remap/tempest_weights.cpp` | high-order/monotone, guarded by `SOMA_HAVE_TEMPEST` |
| `src/transport/mpi_transport.cpp` | n→m redistribution |
| `src/diag/conservation.cpp` | global-sum conservation check |

---

### Task 5.0: Rebuild MOAB with TempestRemap

**Files:**
- Create: `cmake/FindSomaMOAB.cmake`
- Create: `docs/BUILDING.md`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: CMake variables `SOMA_HAVE_MOAB`, `SOMA_HAVE_TEMPEST`, `SOMA_HAVE_IMOAB`, and an imported target `soma::moab`. Every later task keys off these rather than probing MOAB directly.

- [ ] **Step 1: Record the current state before changing anything**

```bash
PKG_CONFIG_PATH=/opt/petsc/download-moab/lib/pkgconfig pkg-config --modversion MOAB
grep -E "define MOAB_HAVE_(MPI|TEMPEST|NETCDF|HDF5|EIGEN|ZOLTAN)" \
     /opt/petsc/download-moab/include/moab/MOABConfig.h
nm -gU /opt/petsc/download-moab/lib/libMOAB.dylib | grep -c -i tempest
```

Expected before the rebuild: version `5.6.0`, `MOAB_HAVE_MPI` and
`MOAB_HAVE_HDF5` defined, **no** `MOAB_HAVE_TEMPEST`, Tempest symbol count `0`.
Paste this into `docs/BUILDING.md` as the "before" record.

- [ ] **Step 2: Build MOAB with TempestRemap into a separate prefix**

Do **not** overwrite `/opt/petsc/download-moab` — KARMA links against it, and
the Global Constraints forbid disturbing unrelated work.

```bash
export SOMA_MOAB_PREFIX=/opt/moab-tempest
git clone --depth 1 --branch 5.6.0 https://bitbucket.org/fathomteam/moab.git /tmp/moab-src
cd /tmp/moab-src && autoreconf -fi
./configure --prefix=$SOMA_MOAB_PREFIX \
            --with-mpi=/opt/mpich \
            --with-hdf5=/opt/hdf5 \
            --download-tempestremap \
            --enable-shared --enable-imoab \
            CC=/opt/mpich/bin/mpicc CXX=/opt/mpich/bin/mpicxx
make -j8 && make install
```

If `--download-tempestremap` fails (it needs NetCDF), build TempestRemap
separately and pass `--with-tempestremap=<prefix>`. Record whichever route
worked in `docs/BUILDING.md`.

- [ ] **Step 3: Verify the rebuild**

```bash
grep -E "define MOAB_HAVE_(MPI|TEMPEST)" $SOMA_MOAB_PREFIX/include/moab/MOABConfig.h
nm -gU $SOMA_MOAB_PREFIX/lib/libMOAB.dylib | grep -c -i tempest
ls $SOMA_MOAB_PREFIX/include/moab/Remapping/
```

Expected: `MOAB_HAVE_TEMPEST 1`, Tempest symbol count > 0, and a
`Remapping/TempestRemapper.hpp` header. **If the count is still 0, stop and
record the failure** — do not let Task 5.3 proceed against a MOAB that cannot
support it. Tasks 5.1, 5.2, 5.4, 5.5 are unaffected.

- [ ] **Step 4: Write `cmake/FindSomaMOAB.cmake`**

```cmake
# Discovers MOAB and detects which optional capabilities it was built with.
# Sets SOMA_HAVE_MOAB, SOMA_HAVE_TEMPEST, SOMA_HAVE_IMOAB and defines the
# imported target soma::moab.

find_package(PkgConfig REQUIRED)
pkg_check_modules(MOAB IMPORTED_TARGET MOAB)

if(NOT MOAB_FOUND)
  set(SOMA_HAVE_MOAB OFF PARENT_SCOPE)
  message(STATUS "MOAB: not found; SOMA_ENABLE_MOAB features are unavailable")
  return()
endif()

set(SOMA_HAVE_MOAB ON PARENT_SCOPE)

# MOABConfig.h records the capabilities the library was actually built with.
find_file(MOAB_CONFIG_HEADER moab/MOABConfig.h
          HINTS ${MOAB_INCLUDE_DIRS} REQUIRED)
file(READ "${MOAB_CONFIG_HEADER}" _moab_config)

string(FIND "${_moab_config}" "#define MOAB_HAVE_TEMPEST" _tempest_pos)
if(_tempest_pos GREATER -1)
  set(SOMA_HAVE_TEMPEST ON PARENT_SCOPE)
else()
  set(SOMA_HAVE_TEMPEST OFF PARENT_SCOPE)
  message(STATUS
    "MOAB: built WITHOUT TempestRemap. Conservative FV<->FV remap via "
    "Intx2Mesh is available; high-order and monotone methods are not. "
    "See docs/BUILDING.md to rebuild with --download-tempestremap.")
endif()

string(FIND "${_moab_config}" "#define MOAB_HAVE_MPI" _mpi_pos)
if(_mpi_pos LESS 0)
  message(WARNING "MOAB: built without MPI; parallel coupling is unavailable")
endif()

find_file(SOMA_IMOAB_HEADER moab/iMOAB.h HINTS ${MOAB_INCLUDE_DIRS})
if(SOMA_IMOAB_HEADER)
  set(SOMA_HAVE_IMOAB ON PARENT_SCOPE)
else()
  set(SOMA_HAVE_IMOAB OFF PARENT_SCOPE)
endif()

add_library(soma::moab ALIAS PkgConfig::MOAB)
```

- [ ] **Step 5: Wire it into the top-level `CMakeLists.txt`**

After the `include(SomaOptions)` line:

```cmake
if(SOMA_ENABLE_MOAB)
  include(FindSomaMOAB)
  if(NOT SOMA_HAVE_MOAB)
    message(FATAL_ERROR
      "SOMA_ENABLE_MOAB=ON but MOAB was not found. Set CMAKE_PREFIX_PATH to a "
      "MOAB prefix, e.g. -DCMAKE_PREFIX_PATH=/opt/moab-tempest")
  endif()
endif()

if(SOMA_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  set(SOMA_HAVE_MPI ON)
endif()
```

and extend `soma_print_summary` in `cmake/SomaOptions.cmake`:

```cmake
  foreach(cap MOAB TEMPEST IMOAB MPI)
    if(DEFINED SOMA_HAVE_${cap})
      message(STATUS "  SOMA_HAVE_${cap}: ${SOMA_HAVE_${cap}}")
    endif()
  endforeach()
```

- [ ] **Step 6: Verify both configurations still work**

```bash
# core: no optional dependency
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure

# full
cmake -B build-full -DCMAKE_BUILD_TYPE=Debug \
      -DSOMA_ENABLE_MPI=ON -DSOMA_ENABLE_MOAB=ON \
      -DCMAKE_PREFIX_PATH=/opt/moab-tempest -DMPI_HOME=/opt/mpich
cmake --build build-full -j
```

Expected: the configure summary prints `SOMA_HAVE_TEMPEST: ON` (or `OFF` with
the explanatory message if Step 3 failed). The core build must be unaffected.

- [ ] **Step 7: Commit**

```bash
git add cmake/FindSomaMOAB.cmake cmake/SomaOptions.cmake CMakeLists.txt docs/BUILDING.md
git commit -m "build(soma): MOAB discovery with TempestRemap capability detection"
```

---

### Task 5.1: MOAB mesh backend

Spec §4.1 and the Global Constraints dependency rule: `moab/Core.hpp` appears in
exactly one translation unit.

**Files:**
- Create: `include/soma/mesh/moab_backend.hpp`
- Create: `src/mesh/moab_backend.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/moab_mesh_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `MeshBackend`, `MeshView` (4.2).
- Produces: `class MoabMeshBackend final : public MeshBackend` implementing the
  same interface as `SerialMeshBackend`, plus:
  - `Result<void> set_communicator(int comm_handle)` — takes an `MPI_Comm` cast to `int` via `MPI_Comm_c2f`, so the public header names no MPI type. Documented as such.
  - `Result<std::vector<std::size_t>> owned_entities(const Region&) const` — local indices of owned (non-ghost) entities, needed by the conservation check to avoid double-counting.
  - `Result<int> partition_rank_of(const Region&, std::size_t) const`
  - Sources accepted: any file MOAB can read (`.h5m`, `.exo`, `.vtk`, `.g`), plus the same `"generated:..."` forms as the serial backend so tests can run without data files.
  - Entity sets resolve by MOAB tag: `MATERIAL_SET`, `DIRICHLET_SET`, `NEUMANN_SET`, or a `NAME` tag match. `"wall"` maps to a `NEUMANN_SET` lookup by name.

- [ ] **Step 1: Write the failing test**

`test/unit/moab_mesh_test.cpp`:

```cpp
#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MOAB
#include <soma/mesh/moab_backend.hpp>

using soma::MeshDecl;
using soma::MeshId;
using soma::MoabMeshBackend;
using soma::Region;

TEST(MoabMesh, LoadsAGeneratedBox) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:4x4x4"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->dimension(), 3);
  EXPECT_EQ(v.value()->entity_count(3), 64u);
}

TEST(MoabMesh, ReportsTheBackendName) {
  const MoabMeshBackend b;
  EXPECT_EQ(b.name(), "moab");
}

// Conservation depends on these summing to the domain measure.
TEST(MoabMesh, CellMeasuresSumToTheDomainVolume) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:5x4x3"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  double total = 0.0;
  for (std::size_t i = 0; i < v->entity_count(3); ++i) total += v->measure(3, i);
  EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(MoabMesh, CentroidsAreInsideTheDomain) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:3x3x3"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    const auto c = v->centroid(3, i);
    EXPECT_GT(c.x, 0.0); EXPECT_LT(c.x, 1.0);
    EXPECT_GT(c.y, 0.0); EXPECT_LT(c.y, 1.0);
    EXPECT_GT(c.z, 0.0); EXPECT_LT(c.z, 1.0);
  }
}

TEST(MoabMesh, GlobalIdsAreUnique) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:3x3x3"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  std::set<std::size_t> ids;
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    ids.insert(v->global_id(3, i));
  }
  EXPECT_EQ(ids.size(), v->entity_count(3));
}

TEST(MoabMesh, SerialRunOwnsEveryEntity) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:2x2x2"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  EXPECT_EQ(v->owned_count(3), v->entity_count(3));
}

TEST(MoabMesh, RegionSizeOfTheWholeMesh) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:3x3x3"}).ok());
  const auto n = b.region_size(Region{.name = "all", .mesh = MeshId{0},
                                      .dimension = 3});
  ASSERT_TRUE(n.ok()) << n.error().message;
  EXPECT_EQ(n.value(), 27u);
}

TEST(MoabMesh, MissingFileIsAnError) {
  MoabMeshBackend b;
  const auto r = b.load(MeshDecl{.name = "m", .source = "/nonexistent.h5m"});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-MOAB-LOAD");
}

TEST(MoabMesh, OwnedEntitiesMatchOwnedCountInSerial) {
  MoabMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:2x2x2"}).ok());
  const auto owned = b.owned_entities(Region{.name = "all", .mesh = MeshId{0},
                                             .dimension = 3});
  ASSERT_TRUE(owned.ok()) << owned.error().message;
  EXPECT_EQ(owned.value().size(), 8u);
}

// The backends must be interchangeable behind MeshView.
TEST(MoabMesh, AgreesWithTheSerialBackendOnCounts) {
  MoabMeshBackend moab;
  soma::SerialMeshBackend serial;
  const MeshDecl d{.name = "m", .source = "generated:box:4x4x4"};
  ASSERT_TRUE(moab.load(d).ok());
  ASSERT_TRUE(serial.load(d).ok());
  EXPECT_EQ(moab.view(MeshId{0}).value()->entity_count(3),
            serial.view(MeshId{0}).value()->entity_count(3));
}

#else
TEST(MoabMesh, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB";
}
#endif
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build-full -j 2>&1 | tail -5
```

Expected: FAIL — `soma/mesh/moab_backend.hpp` file not found. In the plain
`build/` directory the test compiles and skips, which is itself worth checking.

- [ ] **Step 3: Write the header — no MOAB type may appear**

`include/soma/mesh/moab_backend.hpp`:

```cpp
#pragma once

#include <memory>
#include <vector>

#include "soma/mesh/mesh_view.hpp"

namespace soma {

/// MOAB-backed topology. Per the Global Constraints dependency rule, no MOAB
/// type appears here; moab/Core.hpp is included only by moab_backend.cpp.
///
/// Sources: any file MOAB can read (.h5m, .exo, .vtk, .g), plus the same
/// "generated:line:N" / "generated:box:NxMxP" forms the serial backend
/// accepts, so tests need no data files.
class MoabMeshBackend final : public MeshBackend {
 public:
  MoabMeshBackend();
  ~MoabMeshBackend() override;

  MoabMeshBackend(MoabMeshBackend&&) noexcept;
  MoabMeshBackend& operator=(MoabMeshBackend&&) noexcept;

  Result<void> load(const MeshDecl& decl) override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "moab"; }

  /// Pass MPI_Comm_c2f(comm) so this header names no MPI type.
  /// Must be called before load() for parallel reads.
  Result<void> set_communicator(int fortran_comm_handle);

  /// Local indices of owned (non-ghost) entities. The conservation check uses
  /// this to avoid double-counting shared entities.
  [[nodiscard]] Result<std::vector<std::size_t>> owned_entities(
      const Region& r) const;

  [[nodiscard]] Result<int> partition_rank_of(const Region& r,
                                              std::size_t local_index) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soma
```

- [ ] **Step 4: Write `src/mesh/moab_backend.cpp`**

This is the only file that includes MOAB. Key implementation points, each of
which a test above pins:

1. `load` dispatches on the `"generated:"` prefix: generated sources call
   `moab::Core::create_vertices` plus `create_element` to build a structured
   hex grid; anything else goes to `load_file`, with the read options
   `"PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS"`
   appended when a communicator has been set.
2. `MoabMeshView::measure` uses `moab::CN` plus vertex coordinates to compute a
   hex volume as the sum of five tetrahedra, and a quad area by the cross
   product of its diagonals. Do **not** approximate: the conservation tests
   compare against 1.0 at 1e-12.
3. `owned_count` and `owned_entities` use `moab::ParallelComm::filter_pstatus`
   with `PSTATUS_NOT_OWNED` and `PSTATUS_NOT` to exclude ghosts; in serial the
   filter is a no-op and every entity is owned.
4. `region_size` resolves `entity_set` by, in order: empty or `"all"` → the root
   set; a `NAME` tag exact match; a `NEUMANN_SET` / `DIRICHLET_SET` /
   `MATERIAL_SET` tag whose value matches a trailing integer in the name
   (`"sideset_3"` → `NEUMANN_SET == 3`). Anything unresolved is `E-MOAB-SET`.
5. Every MOAB call's `ErrorCode` is checked and converted:
   `if (rval != moab::MB_SUCCESS) return Error{"E-MOAB-<op>", <message>};`
   No `MB_CHK_ERR` macro, which would bypass `Result`.

Write the file following that contract. The structure mirrors
`src/mesh/serial_backend.cpp`: a private `MoabMeshView : public MeshView`, a
`struct Impl` holding `moab::Core`, an optional `moab::ParallelComm*`, and a
`std::vector<std::unique_ptr<MoabMeshView>>`.

- [ ] **Step 5: Update `src/CMakeLists.txt`**

```cmake
if(SOMA_ENABLE_MOAB)
  target_sources(soma_core PRIVATE mesh/moab_backend.cpp)
  target_link_libraries(soma_core PRIVATE soma::moab)
  target_compile_definitions(soma_core PUBLIC SOMA_HAVE_MOAB)
  if(SOMA_HAVE_TEMPEST)
    target_compile_definitions(soma_core PUBLIC SOMA_HAVE_TEMPEST)
  endif()
endif()
```

`SOMA_HAVE_MOAB` is `PUBLIC` because the test above uses it for conditional
compilation; that is a capability macro, not a MOAB type, so it does not violate
the dependency rule.

- [ ] **Step 6: Register and run**

Add `soma_add_test(moab_mesh_test unit/moab_mesh_test.cpp)`.

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R MoabMesh
```

Expected: 10 tests PASS in `build-full`, all SKIP in `build`.

- [ ] **Step 7: Verify no MOAB leakage**

```bash
grep -rn "moab::\|moab/" include/ && echo "LEAK" || echo "clean"
```

Expected: `clean`.

- [ ] **Step 8: Commit**

```bash
git add include/soma/mesh/moab_backend.hpp src/mesh/moab_backend.cpp \
        test/unit/moab_mesh_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): MOAB mesh backend behind the abstract MeshView"
```

---

### Task 5.2: Conservative FV↔FV remap weights from `Intx2Mesh`

Spec §5.4. Supermesh intersection makes the transfer integral exact
(Farrell et al., CMAME 198, 2632). This task needs no TempestRemap.

**Files:**
- Create: `include/soma/remap/remap_plan.hpp`
- Create: `src/remap/intx_weights.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/remap_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `MeshBackend` (4.2), `Region`, `Discretization`, `Conservation` (1.1).
- Produces:
  - `struct RemapWeights { std::vector<std::int32_t> row; std::vector<std::int32_t> col; std::vector<double> val; std::size_t n_rows = 0; std::size_t n_cols = 0; }` — COO triplets, sorted by `(row, col)` so the layout is deterministic and comparable.
  - `struct RemapPlan { RegionId src, tgt; Discretization src_disc = Discretization::FV; Discretization tgt_disc = Discretization::FV; int src_order = 0, tgt_order = 0; RemapFlags flags = RemapFlags::None; std::string method; TemporalBasis temporal = TemporalBasis::None; int temporal_degree = 0; }` with `std::string key() const` (the cache key) and `friend bool operator==`.
    `temporal` and `temporal_degree` are present now, unused until Phase 6 Task 6.3 — the struct shape does not change later.
  - `Result<RemapWeights> build_weights(const RemapPlan&, const CouplingGraph&, const MeshBackend&)`
  - `void apply_weights(const RemapWeights&, const double* src, double* dst, int components)` — the matvec; header-only and templated on nothing (double only, matching `FieldRef`).
  - `class RemapCache` with `Result<const RemapWeights*> get_or_build(const RemapPlan&, const CouplingGraph&, const MeshBackend&)`, `std::size_t size() const`, `std::size_t hits() const`, `std::size_t misses() const`. Spec §5.4: weights are built once at setup and shared by identical channels.
  - Row-sum normalisation: for a conservative map the weights satisfy
    `sum_j w_ij * area_j = area_i`, so a constant field transfers exactly and
    the global integral is preserved. Both are tested.

- [ ] **Step 1: Write the failing test**

`test/unit/remap_test.cpp`:

```cpp
#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MOAB
#include <soma/mesh/moab_backend.hpp>
#include <soma/remap/remap_plan.hpp>

#include <numeric>
#include <vector>

using soma::CouplingGraph;
using soma::MeshDecl;
using soma::MoabMeshBackend;
using soma::Region;
using soma::RemapCache;
using soma::RemapFlags;
using soma::RemapPlan;

namespace {

struct Setup {
  CouplingGraph     graph;
  MoabMeshBackend   backend;
  soma::RegionId    coarse, fine;
};

/// Two non-matching FV meshes over the same unit cube.
Setup non_matching(const std::string& a = "generated:box:4x4x4",
                   const std::string& b = "generated:box:5x5x5") {
  Setup s;
  const auto m1 = s.graph.add_mesh(MeshDecl{.name = "coarse", .source = a});
  const auto m2 = s.graph.add_mesh(MeshDecl{.name = "fine", .source = b});
  s.coarse = s.graph.add_region(Region{.name = "c", .mesh = m1, .dimension = 3});
  s.fine = s.graph.add_region(Region{.name = "f", .mesh = m2, .dimension = 3});
  EXPECT_TRUE(s.backend.load(s.graph.mesh(m1)).ok());
  EXPECT_TRUE(s.backend.load(s.graph.mesh(m2)).ok());
  return s;
}

double integral(const soma::MeshView& v, const std::vector<double>& f) {
  double total = 0.0;
  for (std::size_t i = 0; i < f.size(); ++i) total += f[i] * v.measure(3, i);
  return total;
}

}  // namespace

TEST(Remap, BuildsWeightsBetweenNonMatchingMeshes) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(w.ok()) << w.error().message;
  EXPECT_EQ(w.value().n_rows, 125u);   // target cells
  EXPECT_EQ(w.value().n_cols, 64u);    // source cells
  EXPECT_FALSE(w.value().val.empty());
}

// The single most diagnostic test: a constant must survive exactly.
TEST(Remap, ConstantFieldTransfersExactly) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  const std::vector<double> src(w.n_cols, 3.5);
  std::vector<double> dst(w.n_rows, 0.0);
  apply_weights(w, src.data(), dst.data(), 1);

  for (const double v : dst) EXPECT_NEAR(v, 3.5, 1e-12);
}

// Spec §5.4: conservation is the property that makes the map worth having.
TEST(Remap, PreservesTheGlobalIntegral) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  const auto* src_view = s.backend.view(s.graph.region(s.coarse).mesh).value();
  const auto* tgt_view = s.backend.view(s.graph.region(s.fine).mesh).value();

  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto c = src_view->centroid(3, i);
    src[i] = 1.0 + c.x + 2.0 * c.y;   // smooth, not constant
  }
  std::vector<double> dst(w.n_rows, 0.0);
  apply_weights(w, src.data(), dst.data(), 1);

  const double before = integral(*src_view, src);
  const double after = integral(*tgt_view, dst);
  EXPECT_NEAR(after, before, 1e-11 * std::abs(before));
}

TEST(Remap, WeightsAreNonNegativeForFirstOrder) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();
  for (const double v : w.val) EXPECT_GE(v, -1e-15);
}

TEST(Remap, RowSumsAreUnity) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  std::vector<double> row_sum(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    row_sum[static_cast<std::size_t>(w.row[k])] += w.val[k];
  }
  for (const double rs : row_sum) EXPECT_NEAR(rs, 1.0, 1e-12);
}

TEST(Remap, IdenticalMeshesGiveTheIdentityMap) {
  Setup s = non_matching("generated:box:3x3x3", "generated:box:3x3x3");
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  std::vector<double> src(w.n_cols);
  std::iota(src.begin(), src.end(), 1.0);
  std::vector<double> dst(w.n_rows, 0.0);
  apply_weights(w, src.data(), dst.data(), 1);
  for (std::size_t i = 0; i < src.size(); ++i) {
    EXPECT_NEAR(dst[i], src[i], 1e-12);
  }
}

TEST(Remap, HandlesVectorComponents) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  std::vector<double> src(w.n_cols * 3);
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    src[3 * i + 0] = 1.0;
    src[3 * i + 1] = 2.0;
    src[3 * i + 2] = 3.0;
  }
  std::vector<double> dst(w.n_rows * 3, 0.0);
  apply_weights(w, src.data(), dst.data(), 3);
  for (std::size_t i = 0; i < w.n_rows; ++i) {
    EXPECT_NEAR(dst[3 * i + 0], 1.0, 1e-12);
    EXPECT_NEAR(dst[3 * i + 1], 2.0, 1e-12);
    EXPECT_NEAR(dst[3 * i + 2], 3.0, 1e-12);
  }
}

TEST(Remap, TripletsAreSortedDeterministically) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  const auto a = build_weights(plan, s.graph, s.backend).value();
  const auto b = build_weights(plan, s.graph, s.backend).value();
  EXPECT_EQ(a.row, b.row);
  EXPECT_EQ(a.col, b.col);
  EXPECT_EQ(a.val, b.val);
}

TEST(RemapCache, ReusesWeightsForIdenticalPlans) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative};
  RemapCache cache;
  const auto first = cache.get_or_build(plan, s.graph, s.backend);
  ASSERT_TRUE(first.ok()) << first.error().message;
  const auto second = cache.get_or_build(plan, s.graph, s.backend);
  ASSERT_TRUE(second.ok());
  EXPECT_EQ(first.value(), second.value());   // same object, not a rebuild
  EXPECT_EQ(cache.size(), 1u);
  EXPECT_EQ(cache.misses(), 1u);
  EXPECT_EQ(cache.hits(), 1u);
}

TEST(RemapCache, DistinguishesDifferentPlans) {
  Setup s = non_matching();
  RemapCache cache;
  ASSERT_TRUE(cache.get_or_build(
      RemapPlan{.src = s.coarse, .tgt = s.fine,
                .flags = RemapFlags::Conservative}, s.graph, s.backend).ok());
  ASSERT_TRUE(cache.get_or_build(
      RemapPlan{.src = s.fine, .tgt = s.coarse,
                .flags = RemapFlags::Conservative}, s.graph, s.backend).ok());
  EXPECT_EQ(cache.size(), 2u);
}

#else
TEST(Remap, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB";
}
#endif
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/remap/remap_plan.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/remap/remap_plan.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/mesh/mesh_view.hpp"

namespace soma {

/// COO triplets, sorted by (row, col) so the layout is deterministic.
struct RemapWeights {
  std::vector<std::int32_t> row;
  std::vector<std::int32_t> col;
  std::vector<double>       val;
  std::size_t               n_rows = 0;  // target entities
  std::size_t               n_cols = 0;  // source entities
};

/// Spec §5.4. temporal/temporal_degree are unused until Phase 6 Task 6.3;
/// they are declared now so the struct shape does not change.
struct RemapPlan {
  RegionId       src, tgt;
  Discretization src_disc = Discretization::FV;
  Discretization tgt_disc = Discretization::FV;
  int            src_order = 0;
  int            tgt_order = 0;
  RemapFlags     flags = RemapFlags::None;
  std::string    method;
  TemporalBasis  temporal = TemporalBasis::None;
  int            temporal_degree = 0;

  [[nodiscard]] std::string key() const {
    return std::to_string(src.value()) + ">" + std::to_string(tgt.value()) +
           "/" + to_cstring(src_disc) + std::to_string(src_order) + ">" +
           to_cstring(tgt_disc) + std::to_string(tgt_order) + "/" +
           std::to_string(static_cast<std::uint32_t>(flags)) + "/" + method +
           "/" + to_cstring(temporal) + std::to_string(temporal_degree);
  }
};

/// Sparse matvec: dst = W * src, applied per component.
inline void apply_weights(const RemapWeights& w, const double* src, double* dst,
                          int components) {
  const std::size_t n = w.n_rows * static_cast<std::size_t>(components);
  for (std::size_t i = 0; i < n; ++i) dst[i] = 0.0;
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    const std::size_t r = static_cast<std::size_t>(w.row[k]);
    const std::size_t c = static_cast<std::size_t>(w.col[k]);
    for (int comp = 0; comp < components; ++comp) {
      dst[r * static_cast<std::size_t>(components) + comp] +=
          w.val[k] * src[c * static_cast<std::size_t>(components) + comp];
    }
  }
}

/// Conservative FV<->FV weights via supermesh intersection (Intx2Mesh).
/// High-order and monotone methods require TempestRemap; see Task 5.3.
[[nodiscard]] Result<RemapWeights> build_weights(const RemapPlan& plan,
                                                 const CouplingGraph& graph,
                                                 const MeshBackend& backend);

/// Spec §5.4: built once at setup, shared by identical channels.
class RemapCache {
 public:
  [[nodiscard]] Result<const RemapWeights*> get_or_build(
      const RemapPlan& plan, const CouplingGraph& graph,
      const MeshBackend& backend);

  [[nodiscard]] std::size_t size() const noexcept { return cache_.size(); }
  [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
  [[nodiscard]] std::size_t misses() const noexcept { return misses_; }

 private:
  std::unordered_map<std::string, std::unique_ptr<RemapWeights>> cache_;
  std::size_t hits_ = 0;
  std::size_t misses_ = 0;
};

}  // namespace soma
```

- [ ] **Step 4: Write `src/remap/intx_weights.cpp`**

Implementation contract, each point pinned by a test:

1. Resolve both regions to MOAB entity handles through the backend. Because
   `MeshBackend` exposes no MOAB handle, add a narrow accessor to
   `MoabMeshBackend` used only here: `Result<void*> native_handle(MeshId) const`
   returning the `moab::Core*` as `void*`, documented as an internal seam. This
   keeps the public `MeshBackend` interface clean.
2. Construct `moab::Intx2MeshInPlane` for planar 2D, `Intx2MeshOnSphere` for
   spherical surfaces, and for 3D volumes compute overlaps directly by
   axis-aligned box intersection of the structured cells. **State the
   limitation plainly**: general unstructured 3D supermeshing is not attempted
   here; a non-structured 3D source or target returns
   `E-REMAP-UNSUPPORTED` with a message pointing at Task 5.3.
3. For each overlap polygon/polyhedron of source cell `j` with target cell `i`,
   accumulate `w_ij += |overlap| / |target_i|`. That normalisation is what makes
   row sums unity, constants exact, and the integral conserved — the three
   properties the tests check.
4. Sort the triplets by `(row, col)` and merge duplicates before returning.
5. Every MOAB error becomes `E-REMAP-INTX` with the MOAB message attached.

- [ ] **Step 5: Update `src/CMakeLists.txt`**

```cmake
if(SOMA_ENABLE_MOAB)
  target_sources(soma_core PRIVATE mesh/moab_backend.cpp remap/intx_weights.cpp)
  ...
endif()
```

- [ ] **Step 6: Register and run**

Add `soma_add_test(remap_test unit/remap_test.cpp)`.

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R "Remap|RemapCache"
```

Expected: 11 tests PASS. `ConstantFieldTransfersExactly` and
`PreservesTheGlobalIntegral` are the two that matter; if either fails the
normalisation in point 3 is wrong.

- [ ] **Step 7: Commit**

```bash
git add include/soma/remap/ src/remap/intx_weights.cpp test/unit/remap_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): conservative FV remap weights via supermesh intersection"
```

---

### Task 5.3: High-order and monotone remap via TempestRemap

**Blocked if Task 5.0 Step 3 reported zero Tempest symbols.** Skip and note it;
the rest of Phase 5 does not depend on this.

**Files:**
- Create: `src/remap/tempest_weights.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `test/unit/remap_test.cpp` (add a `SOMA_HAVE_TEMPEST` section)

**Interfaces:**
- Consumes: `RemapPlan`, `RemapWeights` (5.2).
- Produces: `Result<RemapWeights> build_weights_tempest(const RemapPlan&, const CouplingGraph&, const MeshBackend&)`, selected from `build_weights` when `HighOrder` or `Monotone` is set or either discretization is GLL. Method-name mapping, per the TempestRemap 2022 CLI change from `--mono*` to `--method mono*`:

| Flags | TempestRemap method |
|---|---|
| `Conservative` only | `fv` order 1 |
| `Conservative \| HighOrder` | `fv` order 2 |
| `Conservative \| Monotone` | `mono` |
| `Conservative \| HighOrder \| Monotone` | `mono3` |
| explicit `method` string | passed through verbatim |

- [ ] **Step 1: Add the failing tests**

Append to `test/unit/remap_test.cpp`, inside the `SOMA_HAVE_MOAB` block:

```cpp
#ifdef SOMA_HAVE_TEMPEST
TEST(RemapTempest, SecondOrderIsExactOnALinearField) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative | RemapFlags::HighOrder};
  const auto w = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(w.ok()) << w.error().message;

  const auto* src_view = s.backend.view(s.graph.region(s.coarse).mesh).value();
  const auto* tgt_view = s.backend.view(s.graph.region(s.fine).mesh).value();

  std::vector<double> src(w.value().n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = 2.0 * src_view->centroid(3, i).x + 1.0;
  }
  std::vector<double> dst(w.value().n_rows, 0.0);
  apply_weights(w.value(), src.data(), dst.data(), 1);

  for (std::size_t i = 0; i < dst.size(); ++i) {
    const double expected = 2.0 * tgt_view->centroid(3, i).x + 1.0;
    EXPECT_NEAR(dst[i], expected, 1e-9)
        << "second-order remap must reproduce a linear field";
  }
}

TEST(RemapTempest, MonotoneMapIntroducesNoNewExtrema) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative | RemapFlags::Monotone};
  const auto w = build_weights(plan, s.graph, s.backend);
  ASSERT_TRUE(w.ok()) << w.error().message;

  // A step function is the classic overshoot test.
  const auto* src_view = s.backend.view(s.graph.region(s.coarse).mesh).value();
  std::vector<double> src(w.value().n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = src_view->centroid(3, i).x < 0.5 ? 0.0 : 1.0;
  }
  std::vector<double> dst(w.value().n_rows, 0.0);
  apply_weights(w.value(), src.data(), dst.data(), 1);

  for (const double v : dst) {
    EXPECT_GE(v, -1e-12);
    EXPECT_LE(v, 1.0 + 1e-12);
  }
}

TEST(RemapTempest, HighOrderStillConserves) {
  Setup s = non_matching();
  const RemapPlan plan{.src = s.coarse, .tgt = s.fine,
                       .flags = RemapFlags::Conservative | RemapFlags::HighOrder};
  const auto w = build_weights(plan, s.graph, s.backend).value();

  const auto* src_view = s.backend.view(s.graph.region(s.coarse).mesh).value();
  const auto* tgt_view = s.backend.view(s.graph.region(s.fine).mesh).value();
  std::vector<double> src(w.n_cols);
  for (std::size_t i = 0; i < src.size(); ++i) {
    const auto c = src_view->centroid(3, i);
    src[i] = std::sin(3.0 * c.x) * std::cos(2.0 * c.y);
  }
  std::vector<double> dst(w.n_rows, 0.0);
  apply_weights(w, src.data(), dst.data(), 1);

  const double before = integral(*src_view, src);
  const double after = integral(*tgt_view, dst);
  EXPECT_NEAR(after, before, 1e-10 * std::abs(before));
}
#endif  // SOMA_HAVE_TEMPEST
```

- [ ] **Step 2: Run to verify it fails**

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R RemapTempest
```

Expected: FAIL with `E-REMAP-UNSUPPORTED` (the Task 5.2 fallback), or the whole
block absent if `SOMA_HAVE_TEMPEST` is off — in which case **stop here and record
the block in `docs/BUILDING.md`.**

- [ ] **Step 3: Write `src/remap/tempest_weights.cpp`**

Implementation contract:

1. `#include <moab/Remapping/TempestRemapper.hpp>` and
   `<moab/Remapping/TempestOfflineMap.hpp>`, both guarded by `SOMA_HAVE_TEMPEST`.
2. Construct `moab::TempestRemapper`, set source and target meshes from the
   backend's native handles, call `ConstructCoveringSet` then
   `ComputeOverlapMesh`.
3. Construct `moab::TempestOnlineMap`, call `GenerateRemappingWeights` with the
   method name from the table above, the discretization names (`"fv"`, `"cgll"`,
   `"dgll"`), and the orders from the plan.
4. Extract the triplets via the map's sparse-matrix accessor, sort, and return.
5. Any TempestRemap exception becomes `E-REMAP-TEMPEST` — TempestRemap throws
   rather than returning codes, so wrap the whole body in `try`/`catch`.

- [ ] **Step 4: Dispatch from `build_weights`**

In `src/remap/intx_weights.cpp`, at the top of `build_weights`:

```cpp
#ifdef SOMA_HAVE_TEMPEST
  const bool needs_tempest =
      has(plan.flags, RemapFlags::HighOrder) ||
      has(plan.flags, RemapFlags::Monotone) ||
      plan.src_disc != Discretization::FV ||
      plan.tgt_disc != Discretization::FV ||
      !plan.method.empty();
  if (needs_tempest) return build_weights_tempest(plan, graph, backend);
#else
  if (has(plan.flags, RemapFlags::HighOrder) ||
      has(plan.flags, RemapFlags::Monotone) ||
      plan.src_disc != Discretization::FV ||
      plan.tgt_disc != Discretization::FV) {
    return Error{"E-REMAP-UNSUPPORTED",
                 "high-order, monotone, and GLL remap need MOAB built with "
                 "TempestRemap; see docs/BUILDING.md"};
  }
#endif
```

- [ ] **Step 5: Run**

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R Remap
```

Expected: 14 tests PASS (11 from Task 5.2 plus 3 Tempest).
`SecondOrderIsExactOnALinearField` is the one that proves the order claim.

- [ ] **Step 6: Commit**

```bash
git add src/remap/tempest_weights.cpp src/remap/intx_weights.cpp \
        test/unit/remap_test.cpp src/CMakeLists.txt
git commit -m "feat(soma): high-order and monotone remap via TempestRemap"
```

---

### Task 5.4: MPI transport with n→m redistribution

Spec §4.1: component ranks and coupler ranks may be disjoint.

**Files:**
- Create: `src/transport/mpi_transport.cpp`
- Modify: `include/soma/transport/transport.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/parallel/mpi_transport_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Transport`, `TransportSpec` (4.3).
- Produces:
  - `TransportSpec` gains `std::vector<int> src_ranks; std::vector<int> dst_ranks; std::vector<std::size_t> global_ids;` — the redistribution pattern. Adding fields to the existing struct rather than subclassing keeps `Transport::prepare` monomorphic.
  - `class MpiTransport final : public Transport` — registered as `"mpi"`, `is_local()` false. `prepare` builds the send/receive schedule once (an `MPI_Alltoallv` displacement plan derived from `global_ids`); `move` executes it. Compiled only when `SOMA_ENABLE_MPI`.
  - `Result<void> set_communicator(int fortran_comm_handle)` on `MpiTransport`, same convention as the MOAB backend so no MPI type reaches a public header.
  - A CTest entry `mpi_transport_4rank` running under `mpiexec -n 4`.

- [ ] **Step 1: Write the failing test**

`test/parallel/mpi_transport_test.cpp`:

```cpp
#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MPI
#include <mpi.h>
#include <soma/transport/transport.hpp>

#include <numeric>
#include <vector>

using soma::MpiTransport;
using soma::TransportSpec;

namespace {

int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

int world_size() {
  int s = 1;
  MPI_Comm_size(MPI_COMM_WORLD, &s);
  return s;
}

}  // namespace

// Ranks 0,1 produce; ranks 2,3 consume. This is the disjoint-rank-set case
// spec §4.1 calls the normal one.
TEST(MpiTransport, RedistributesAcrossDisjointRankSets) {
  ASSERT_EQ(world_size(), 4) << "run this test with mpiexec -n 4";
  const int rank = world_rank();
  const bool is_source = rank < 2;

  constexpr std::size_t kGlobal = 8;
  constexpr std::size_t kLocal = kGlobal / 2;

  std::vector<std::size_t> gids(kLocal);
  const std::size_t base = static_cast<std::size_t>(rank % 2) * kLocal;
  std::iota(gids.begin(), gids.end(), base);

  TransportSpec spec;
  spec.src_rank_set = 0;
  spec.dst_rank_set = 1;
  spec.src_ranks = {0, 1};
  spec.dst_ranks = {2, 3};
  spec.elements = kLocal;
  spec.components = 1;
  spec.global_ids = gids;

  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  ASSERT_TRUE(t.prepare(spec).ok());

  std::vector<double> send(kLocal, 0.0);
  std::vector<double> recv(kLocal, -1.0);
  if (is_source) {
    for (std::size_t i = 0; i < kLocal; ++i) {
      send[i] = static_cast<double>(gids[i]) * 10.0;
    }
  }

  ASSERT_TRUE(t.move(send.data(), recv.data(), kLocal).ok());

  if (!is_source) {
    for (std::size_t i = 0; i < kLocal; ++i) {
      EXPECT_DOUBLE_EQ(recv[i], static_cast<double>(gids[i]) * 10.0)
          << "rank " << rank << " index " << i;
    }
  }
}

TEST(MpiTransport, IsNotLocal) {
  const MpiTransport t;
  EXPECT_FALSE(t.is_local());
  EXPECT_EQ(t.name(), "mpi");
}

TEST(MpiTransport, RegisteredInTheTransportRegistry) {
  const auto t = soma::TransportRegistry::instance().create("mpi");
  ASSERT_TRUE(t.ok()) << t.error().message;
  EXPECT_EQ(t.value()->name(), "mpi");
}

TEST(MpiTransport, PrepareRejectsMismatchedRankSets) {
  TransportSpec spec;
  spec.src_ranks = {};     // empty
  spec.dst_ranks = {2, 3};
  spec.elements = 4;
  MpiTransport t;
  ASSERT_TRUE(t.set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD)).ok());
  const auto r = t.prepare(spec);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-RANKS");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  MPI_Finalize();
  return result;
}

#else
int main() { return 0; }   // built without MPI
#endif
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `MpiTransport` is not declared.

- [ ] **Step 3: Extend `TransportSpec` and declare `MpiTransport`**

In `include/soma/transport/transport.hpp`, extend the struct:

```cpp
struct TransportSpec {
  int         src_rank_set = -1;
  int         dst_rank_set = -1;
  std::size_t elements = 0;
  int         components = 1;

  /// Redistribution pattern for non-local transports (spec §4.1: component
  /// and coupler rank sets may be disjoint). Empty for local transports.
  std::vector<int>         src_ranks;
  std::vector<int>         dst_ranks;
  std::vector<std::size_t> global_ids;  // this rank's entity ids
};
```

and add, guarded:

```cpp
#ifdef SOMA_HAVE_MPI
/// n->m redistribution. The communicator is passed as MPI_Comm_c2f(comm) so
/// this header names no MPI type (Global Constraints).
class MpiTransport final : public Transport {
 public:
  MpiTransport();
  ~MpiTransport() override;
  MpiTransport(MpiTransport&&) noexcept;
  MpiTransport& operator=(MpiTransport&&) noexcept;

  Result<void> set_communicator(int fortran_comm_handle);
  Result<void> prepare(const TransportSpec& spec) override;
  Result<void> move(const double* src, double* dst, std::size_t n) override;
  [[nodiscard]] std::string name() const override { return "mpi"; }
  [[nodiscard]] bool is_local() const override { return false; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
#endif
```

- [ ] **Step 4: Write `src/transport/mpi_transport.cpp`**

Implementation contract:

1. `prepare` validates that both rank vectors are non-empty
   (`E-TRANSPORT-RANKS`) and that this rank appears in one of them.
2. Build the schedule by exchanging `global_ids`: an `MPI_Allgatherv` over the
   union of source and destination ranks, then for each destination rank match
   its ids against locally held ids to build `sendcounts`/`sdispls` and
   `recvcounts`/`rdispls`. Done once in `prepare`; `move` only calls
   `MPI_Alltoallv`.
3. A rank in neither set participates in the collective with zero counts, so the
   call is still collective over the communicator — otherwise a four-rank run
   with two idle ranks deadlocks.
4. `move` packs into a contiguous send buffer in schedule order, calls
   `MPI_Alltoallv`, and unpacks by global id.
5. Register `"mpi"` in `TransportRegistry::instance()` alongside `"copy"`,
   inside `#ifdef SOMA_HAVE_MPI`.

- [ ] **Step 5: Update the build**

`src/CMakeLists.txt`:

```cmake
if(SOMA_ENABLE_MPI)
  target_sources(soma_core PRIVATE transport/mpi_transport.cpp)
  target_link_libraries(soma_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(soma_core PUBLIC SOMA_HAVE_MPI)
endif()
```

`test/CMakeLists.txt`:

```cmake
if(SOMA_ENABLE_MPI)
  add_executable(mpi_transport_test parallel/mpi_transport_test.cpp)
  target_link_libraries(mpi_transport_test PRIVATE soma soma_core
                        GTest::gtest MPI::MPI_CXX)
  add_test(NAME mpi_transport_4rank
           COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 4
                   $<TARGET_FILE:mpi_transport_test>)
endif()
```

Note this test links `GTest::gtest`, not `gtest_main` — it provides its own
`main` for `MPI_Init`.

- [ ] **Step 6: Run**

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R mpi_transport_4rank
```

Expected: PASS. If it hangs, point 3 is the cause — a rank is skipping the
collective.

- [ ] **Step 7: Commit**

```bash
git add src/transport/mpi_transport.cpp include/soma/transport/transport.hpp \
        test/parallel/mpi_transport_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): MPI transport with n-to-m redistribution"
```

---

### Task 5.5: Wire remap into the runner and check conservation

Replaces Phase 4's identity `Remap` with real weights and turns `Constrain` into
a working conservation check over owned entities.

**Files:**
- Modify: `src/runtime/runner.cpp`
- Modify: `include/soma/runtime/runner.hpp`
- Create: `src/diag/conservation.cpp`
- Create: `include/soma/diag/conservation.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/verification/conservation_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `RemapCache` (5.2), `Runner` (4.6), `MeshView` (4.2).
- Produces:
  - `struct ConservationReport { double source_integral = 0.0; double target_integral = 0.0; double absolute_error = 0.0; double relative_error = 0.0; bool within_tolerance = true; }`
  - `ConservationReport check_conservation(const MeshView& src_view, int src_dim, const double* src, std::size_t n_src, const MeshView& tgt_view, int tgt_dim, const double* tgt, std::size_t n_tgt, double tolerance)` — uses `num::weighted_sum` with `measure` as weights, so the comparison is of true integrals.
  - `RunnerConfig` gains `RemapCache* remap_cache = nullptr;` — when null, `Runner::initialize` creates and owns one.
  - `Runner::initialize` builds a `RemapPlan` for every `Remap` instruction from the arc types plus the channel's flags, calls `RemapCache::get_or_build`, and stores the resulting `const RemapWeights*` in a vector indexed by instruction. `Remap` execution becomes `apply_weights`, with the identity fallback retained only when no MOAB backend is present.
  - `RunStats` gains `std::int64_t remap_applications = 0;`.

- [ ] **Step 1: Write the failing test**

`test/verification/conservation_test.cpp`:

```cpp
#include <gtest/gtest.h>

#ifdef SOMA_HAVE_MOAB
#include <soma/ir/lowering.hpp>
#include <soma/mesh/moab_backend.hpp>
#include <soma/runtime/runner.hpp>

#include <cmath>
#include <vector>

using soma::ComponentSet;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::Runner;
using soma::RunnerConfig;
namespace u = soma::units;

namespace {

class FieldHolder final : public soma::Component {
 public:
  FieldHolder(std::string name, std::size_t n)
      : name_(std::move(name)), data_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext&) override {
    return soma::Result<void>::success();
  }

  std::vector<double>& data() { return data_; }
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::string         name_;
  std::vector<double> data_;
};

struct Fixture {
  CouplingGraph        graph;
  soma::MoabMeshBackend backend;
  soma::OperatorDAG    dag;
  soma::ExecutionPlan  plan;
  ComponentSet         components;
  FieldRegistry        registry;
  FieldHolder*         src = nullptr;
  FieldHolder*         dst = nullptr;
  soma::RegionId       r_src, r_dst;
};

/// 4^3 source, 5^3 target: genuinely non-matching.
std::unique_ptr<Fixture> build_non_matching() {
  auto f = std::make_unique<Fixture>();
  auto& g = f->graph;
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "coarse",
                                            .source = "generated:box:4x4x4"});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "fine",
                                            .source = "generated:box:5x5x5"});
  f->r_src = g.add_region(soma::Region{.name = "rs", .mesh = m1,
                                       .dimension = 3});
  f->r_dst = g.add_region(soma::Region{.name = "rt", .mesh = m2,
                                       .dimension = 3});
  const auto ca = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto cb = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});
  const auto fs = g.add_field(ca, soma::FieldSpec{
      .name = "q", .region = f->r_src, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive,
      .intent = soma::Intent::Provides});
  const auto ft = g.add_field(cb, soma::FieldSpec{
      .name = "q_in", .region = f->r_dst, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("flux", fs, ft);
  g.channel(ch).remap(soma::RemapFlags::Conservative);

  EXPECT_TRUE(f->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(f->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  f->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched = schedule(g, f->dag, clocks,
                        soma::make_region_sizes(g, f->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  f->plan = sched.plan;

  auto a = std::make_unique<FieldHolder>("a", 64);
  auto b = std::make_unique<FieldHolder>("b", 125);
  f->src = a.get();
  f->dst = b.get();
  f->components.add(std::move(a));
  f->components.add(std::move(b));
  return f;
}

double integral(const soma::MeshView& v, const std::vector<double>& d) {
  double total = 0.0;
  for (std::size_t i = 0; i < d.size(); ++i) total += d[i] * v.measure(3, i);
  return total;
}

}  // namespace

// The Phase 5 milestone: non-matching meshes, real weights, conservation holds.
TEST(Conservation, HoldsAcrossNonMatchingMeshes) {
  auto f = build_non_matching();
  const auto* sv = f->backend.view(f->graph.region(f->r_src).mesh).value();
  const auto* tv = f->backend.view(f->graph.region(f->r_dst).mesh).value();

  for (std::size_t i = 0; i < f->src->data().size(); ++i) {
    const auto c = sv->centroid(3, i);
    f->src->data()[i] = 1.0 + std::sin(2.0 * c.x) * std::cos(3.0 * c.y);
  }
  const double before = integral(*sv, f->src->data());

  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
  ASSERT_TRUE(runner.run_period(f->components).ok());

  const double after = integral(*tv, f->dst->data());
  EXPECT_NEAR(after, before, 1e-11 * std::abs(before));
  EXPECT_GT(runner.stats().remap_applications, 0);
}

TEST(Conservation, ConstantFieldSurvivesTheFullPipeline) {
  auto f = build_non_matching();
  for (double& d : f->src->data()) d = 2.5;

  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
  ASSERT_TRUE(runner.run_period(f->components).ok());

  for (const double v : f->dst->data()) EXPECT_NEAR(v, 2.5, 1e-12);
}

TEST(Conservation, ReportsNoViolationsWithinTolerance) {
  auto f = build_non_matching();
  for (double& d : f->src->data()) d = 1.0;

  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  RunnerConfig cfg;
  cfg.check_conservation = true;
  cfg.conservation_tolerance = 1e-11;
  ASSERT_TRUE(runner.initialize(f->components, f->registry, cfg).ok());
  ASSERT_TRUE(runner.run(f->components, 5).ok());

  EXPECT_GT(runner.stats().conservation_checks, 0);
  EXPECT_EQ(runner.stats().conservation_violations, 0);
  EXPECT_LT(runner.stats().worst_conservation_error, 1e-11);
}

TEST(Conservation, WeightsAreBuiltOncePerDistinctPlan) {
  auto f = build_non_matching();
  soma::RemapCache cache;
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  RunnerConfig cfg;
  cfg.remap_cache = &cache;
  ASSERT_TRUE(runner.initialize(f->components, f->registry, cfg).ok());
  ASSERT_TRUE(runner.run(f->components, 10).ok());

  EXPECT_EQ(cache.misses(), 1u) << "weights must be built once at setup";
  EXPECT_EQ(cache.size(), 1u);
}

#else
TEST(Conservation, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB";
}
#endif
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `E-RT-REMAP-SIZE`, because Phase 4's identity remap cannot
map 64 cells onto 125.

- [ ] **Step 3: Write the conservation diagnostic**

`include/soma/diag/conservation.hpp`:

```cpp
#pragma once

#include <cstddef>

#include "soma/mesh/mesh_view.hpp"

namespace soma {

struct ConservationReport {
  double source_integral = 0.0;
  double target_integral = 0.0;
  double absolute_error = 0.0;
  double relative_error = 0.0;
  bool   within_tolerance = true;
};

/// Spec §10 level 3. Integrates with MeshView::measure as weights, so this
/// compares true integrals rather than raw sums.
[[nodiscard]] ConservationReport check_conservation(
    const MeshView& src_view, int src_dim, const double* src, std::size_t n_src,
    const MeshView& tgt_view, int tgt_dim, const double* tgt, std::size_t n_tgt,
    double tolerance);

}  // namespace soma
```

`src/diag/conservation.cpp` implements it with `num::weighted_sum` over a
locally built measure vector; owned-entity filtering is applied by the caller in
parallel runs.

- [ ] **Step 4: Wire remap into the runner**

In `include/soma/runtime/runner.hpp`, add to `RunnerConfig`:

```cpp
  /// When null, initialize() creates and owns a cache. Supply one to share
  /// weights across several Runners or to inspect hit/miss counts.
  RemapCache* remap_cache = nullptr;
```

and to `RunStats`:

```cpp
  std::int64_t remap_applications = 0;
```

In `src/runtime/runner.cpp`:

1. Add `std::vector<const RemapWeights*> remap_for_instruction;` to `Impl`,
   sized to the instruction count and default `nullptr`.
2. In `initialize`, for each `Remap` instruction, build a `RemapPlan` from the
   input and output arc types (`region`, `disc` via the owning `FieldSpec`) plus
   `ins.attrs.remap_flags` and `ins.attrs.method`, then
   `cache->get_or_build(...)` and store the pointer. Report `E-RT-REMAP-BUILD`
   with the underlying message on failure.
3. In `execute`, replace the `OpKind::Remap` case:

```cpp
    case OpKind::Remap: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const Slot& in = slots[ins.in[0].value()];
      const RemapWeights* w = remap_for_instruction[instruction_index];
      if (w != nullptr) {
        apply_weights(*w, in.data, out.data, out.components);
        ++stats.remap_applications;
        break;
      }
      // No weights: identity, which requires matching sizes.
      if (total_elements(in) != total_elements(out)) {
        return Error{"E-RT-REMAP-SIZE",
                     "identity remap needs matching sizes (" +
                         std::to_string(total_elements(in)) + " vs " +
                         std::to_string(total_elements(out)) +
                         "); build with SOMA_ENABLE_MOAB for real weights"};
      }
      num::copy(in.data, out.data, total_elements(out));
      ++stats.remap_applications;
      break;
    }
```

`execute` needs the instruction index, so change its signature to
`Result<void> execute(const Instruction& ins, std::size_t instruction_index)`
and pass the loop index from `run_period`.

- [ ] **Step 5: Run**

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R Conservation
```

Expected: 4 tests PASS. `HoldsAcrossNonMatchingMeshes` is the Phase 5
milestone.

- [ ] **Step 6: Run everything, both configurations**

```bash
ctest --test-dir build --output-on-failure         # core: no optional deps
ctest --test-dir build-full --output-on-failure    # MOAB + MPI
```

Expected: both green. The core build must still pass with MOAB tests skipping.

- [ ] **Step 7: Commit**

```bash
git add include/soma/diag/ src/diag/ include/soma/runtime/runner.hpp \
        src/runtime/runner.cpp test/verification/conservation_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): real remap in the runner with conservation verification"
```

---

## Phase 5 exit criteria

- [ ] MOAB is discovered with capability detection; `SOMA_HAVE_TEMPEST` is reported honestly either way.
- [ ] `grep -rn "moab::\|moab/" include/` finds nothing.
- [ ] A constant field transfers exactly across 4³ → 5³ non-matching meshes.
- [ ] The global integral is conserved to 1e-11 relative on a smooth field.
- [ ] Remap weights are built **once** at setup, verified by `RemapCache::misses() == 1` after ten periods.
- [ ] MPI transport redistributes correctly across disjoint rank sets under `mpiexec -n 4`.
- [ ] The core build still configures, builds, and passes with no optional dependency.
- [ ] Both `ctest` runs fully green.

**Deferred:** high-order and monotone remap if Task 5.0 could not produce a
TempestRemap-enabled MOAB — recorded in `docs/BUILDING.md`, with Task 5.3's tests
compiled out rather than silently passing.
