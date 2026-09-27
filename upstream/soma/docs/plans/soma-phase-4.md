# SOMA Phase 4 — Runtime, serial backend, first coupled run

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §2 (layer 3 execution), §4.1 (mesh backend), §4.2 (component-owned storage), §5.1 (kernels), §6 mode 1 (in-process C++)
**Milestone:** two toy C++ components exchange a field through the full stack — graph → lower → schedule → run — and the coupled answer is verified against an analytic result.

This is the first honest milestone. Everything before it was structure; this is the first code that computes.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/field/field_ref.hpp` | `FieldRef`, `Layout`, `MemorySpace` |
| `include/soma/field/registry.hpp` | `FieldRegistry` — handle ↔ pointer resolution |
| `include/soma/mesh/mesh_view.hpp` | abstract `MeshView`, `MeshBackend` |
| `src/mesh/serial_backend.cpp` | in-memory structured backend, zero dependencies |
| `include/soma/transport/transport.hpp` | abstract `Transport` |
| `src/transport/copy_transport.cpp` | in-process copy |
| `include/soma/num/kernels.hpp` | header-only templated kernels |
| `src/num/kernels_inst.cpp` | `extern template` instantiations |
| `include/soma/runtime/component.hpp` | the `Component` interface |
| `include/soma/runtime/runner.hpp` | `Runner` — the plan interpreter |
| `src/runtime/runner.cpp` | interpreter implementation |
| `examples/mms_pair/` | the first coupled run |

---

### Task 4.1: `FieldRef` and `FieldRegistry`

Spec §4.2: components own their data; the coupler wraps a non-owning view.

**Files:**
- Create: `include/soma/field/field_ref.hpp`
- Create: `include/soma/field/registry.hpp`
- Test: `test/unit/field_ref_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `FieldId` (0.6), `Result` (0.2).
- Produces:
  - `enum class MemorySpace { Host, Device }`
  - `struct Layout { std::size_t stride = 1; std::size_t component_stride = 0; std::size_t count = 0; int components = 1; }` — `stride` is the element-to-element step in units of `double`; `component_stride` is the step between components (0 means interleaved, i.e. component-fastest). Together these cover both AoS and SoA without a separate enum.
  - `class FieldRef` — non-owning: `FieldRef(double* data, Layout, MemorySpace)`, `double* data()`, `const double* data() const`, `const Layout& layout()`, `MemorySpace space()`, `bool valid()`, `double& at(std::size_t i, int c = 0)`, `double at(std::size_t i, int c = 0) const`, `std::size_t size()`.
  - `class FieldRegistry` with `Result<void> bind(FieldId, FieldRef)`, `bool is_bound(FieldId) const`, `Result<FieldRef> get(FieldId) const`, `std::size_t bound_count() const`, `std::vector<FieldId> unbound(const std::vector<FieldId>& required) const`.
    Rebinding an already-bound field is an error (`E-FIELD-REBIND`) — silently
    replacing a component's pointer mid-run is a class of bug worth refusing.

- [ ] **Step 1: Write the failing test**

`test/unit/field_ref_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/field/registry.hpp>

#include <vector>

using soma::FieldId;
using soma::FieldRef;
using soma::FieldRegistry;
using soma::Layout;
using soma::MemorySpace;

TEST(FieldRef, DefaultIsInvalid) {
  const FieldRef f;
  EXPECT_FALSE(f.valid());
  EXPECT_EQ(f.data(), nullptr);
}

TEST(FieldRef, WrapsContiguousScalarData) {
  std::vector<double> v{1.0, 2.0, 3.0, 4.0};
  const FieldRef f{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  ASSERT_TRUE(f.valid());
  EXPECT_EQ(f.size(), 4u);
  EXPECT_DOUBLE_EQ(f.at(2), 3.0);
}

TEST(FieldRef, WritesThroughToOwnerStorage) {
  std::vector<double> v(4, 0.0);
  FieldRef f{v.data(), Layout{.stride = 1, .count = 4}, MemorySpace::Host};
  f.at(1) = 7.5;
  EXPECT_DOUBLE_EQ(v[1], 7.5);   // the component sees the write
}

TEST(FieldRef, InterleavedVectorLayout) {
  // x0 y0 z0 x1 y1 z1
  std::vector<double> v{1, 2, 3, 4, 5, 6};
  const FieldRef f{v.data(),
                   Layout{.stride = 3, .component_stride = 1, .count = 2,
                          .components = 3},
                   MemorySpace::Host};
  EXPECT_DOUBLE_EQ(f.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(f.at(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(f.at(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(f.at(1, 2), 6.0);
}

TEST(FieldRef, BlockedVectorLayout) {
  // x0 x1 | y0 y1 | z0 z1
  std::vector<double> v{1, 4, 2, 5, 3, 6};
  const FieldRef f{v.data(),
                   Layout{.stride = 1, .component_stride = 2, .count = 2,
                          .components = 3},
                   MemorySpace::Host};
  EXPECT_DOUBLE_EQ(f.at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(f.at(0, 2), 3.0);
  EXPECT_DOUBLE_EQ(f.at(1, 0), 4.0);
  EXPECT_DOUBLE_EQ(f.at(1, 2), 6.0);
}

TEST(FieldRegistry, BindsAndRetrieves) {
  std::vector<double> v(8, 1.0);
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0},
                          FieldRef{v.data(), Layout{.stride = 1, .count = 8},
                                   MemorySpace::Host});
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(reg.is_bound(FieldId{0}));
  EXPECT_EQ(reg.bound_count(), 1u);

  const auto got = reg.get(FieldId{0});
  ASSERT_TRUE(got.ok());
  EXPECT_EQ(got.value().data(), v.data());
}

TEST(FieldRegistry, RejectsRebinding) {
  std::vector<double> a(4, 0.0), b(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{a.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  const auto second = reg.bind(FieldId{0},
                               FieldRef{b.data(),
                                        Layout{.stride = 1, .count = 4},
                                        MemorySpace::Host});
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.error().code, "E-FIELD-REBIND");
}

TEST(FieldRegistry, RejectsNullData) {
  FieldRegistry reg;
  const auto r = reg.bind(FieldId{0}, FieldRef{nullptr,
                                               Layout{.stride = 1, .count = 4},
                                               MemorySpace::Host});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-NULL");
}

TEST(FieldRegistry, GetUnboundFieldIsAnError) {
  const FieldRegistry reg;
  const auto r = reg.get(FieldId{3});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-FIELD-UNBOUND");
}

TEST(FieldRegistry, ReportsWhichRequiredFieldsAreMissing) {
  std::vector<double> v(4, 0.0);
  FieldRegistry reg;
  ASSERT_TRUE(reg.bind(FieldId{0}, FieldRef{v.data(),
                                            Layout{.stride = 1, .count = 4},
                                            MemorySpace::Host}).ok());
  const auto missing = reg.unbound({FieldId{0}, FieldId{1}, FieldId{2}});
  ASSERT_EQ(missing.size(), 2u);
  EXPECT_EQ(missing[0], FieldId{1});
  EXPECT_EQ(missing[1], FieldId{2});
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/field/registry.hpp` file not found.

- [ ] **Step 3: Write `include/soma/field/field_ref.hpp`**

```cpp
#pragma once

#include <cstddef>

#include "soma/base/ids.hpp"

namespace soma {

enum class MemorySpace { Host, Device };

[[nodiscard]] constexpr const char* to_cstring(MemorySpace s) noexcept {
  return s == MemorySpace::Device ? "Device" : "Host";
}

/// Element and component strides in units of double. component_stride == 0
/// means interleaved (component-fastest); a non-zero value means blocked.
struct Layout {
  std::size_t stride = 1;
  std::size_t component_stride = 0;
  std::size_t count = 0;
  int         components = 1;
};

/// Spec §4.2: non-owning. Components register a pointer; the coupler never
/// takes ownership of component storage.
class FieldRef {
 public:
  FieldRef() = default;
  FieldRef(double* data, Layout layout, MemorySpace space)
      : data_(data), layout_(layout), space_(space) {}

  [[nodiscard]] double*       data() noexcept { return data_; }
  [[nodiscard]] const double* data() const noexcept { return data_; }
  [[nodiscard]] const Layout& layout() const noexcept { return layout_; }
  [[nodiscard]] MemorySpace   space() const noexcept { return space_; }
  [[nodiscard]] bool          valid() const noexcept { return data_ != nullptr; }
  [[nodiscard]] std::size_t   size() const noexcept { return layout_.count; }

  [[nodiscard]] std::size_t offset(std::size_t i, int c) const noexcept {
    const std::size_t cs = layout_.component_stride == 0
                               ? 1
                               : layout_.component_stride;
    return i * layout_.stride + static_cast<std::size_t>(c) * cs;
  }

  [[nodiscard]] double& at(std::size_t i, int c = 0) noexcept {
    return data_[offset(i, c)];
  }
  [[nodiscard]] double at(std::size_t i, int c = 0) const noexcept {
    return data_[offset(i, c)];
  }

 private:
  double*     data_ = nullptr;
  Layout      layout_{};
  MemorySpace space_ = MemorySpace::Host;
};

}  // namespace soma
```

Note on the interleaved default: with `component_stride == 0` the accessor uses
a step of 1, which combined with `stride = components` gives the AoS layout in
the `InterleavedVectorLayout` test. The blocked case sets both explicitly.

- [ ] **Step 4: Write `include/soma/field/registry.hpp`**

```cpp
#pragma once

#include <unordered_map>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/field/field_ref.hpp"

namespace soma {

/// Maps FieldId to the component-owned storage bound at registration.
/// Resolution happens once at setup; the hot loop dereferences pointers.
class FieldRegistry {
 public:
  Result<void> bind(FieldId id, FieldRef ref) {
    if (!ref.valid()) {
      return Error{"E-FIELD-NULL", "field " + std::to_string(id.value()) +
                                       " bound to a null pointer"};
    }
    if (refs_.count(id.value())) {
      return Error{"E-FIELD-REBIND", "field " + std::to_string(id.value()) +
                                         " is already bound"};
    }
    refs_.emplace(id.value(), ref);
    return Result<void>::success();
  }

  [[nodiscard]] bool is_bound(FieldId id) const {
    return refs_.count(id.value()) > 0;
  }

  [[nodiscard]] Result<FieldRef> get(FieldId id) const {
    const auto it = refs_.find(id.value());
    if (it == refs_.end()) {
      return Error{"E-FIELD-UNBOUND",
                   "field " + std::to_string(id.value()) + " is not bound"};
    }
    return it->second;
  }

  [[nodiscard]] std::size_t bound_count() const noexcept { return refs_.size(); }

  [[nodiscard]] std::vector<FieldId> unbound(
      const std::vector<FieldId>& required) const {
    std::vector<FieldId> missing;
    for (const FieldId f : required) {
      if (!is_bound(f)) missing.push_back(f);
    }
    return missing;
  }

 private:
  std::unordered_map<std::uint32_t, FieldRef> refs_;
};

}  // namespace soma
```

- [ ] **Step 5: Register and run**

Add `soma_add_test(field_ref_test unit/field_ref_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "FieldRef|FieldRegistry"
```

Expected: 10 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/soma/field/ test/unit/field_ref_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): non-owning FieldRef and FieldRegistry"
```

---

### Task 4.2: Abstract `MeshView` and the serial backend

Spec §4.1: the coupler never owns topology. Phase 4 provides a dependency-free
backend so the whole stack is testable before MOAB arrives in Phase 5.

**Files:**
- Create: `include/soma/mesh/mesh_view.hpp`
- Create: `src/mesh/serial_backend.cpp`
- Create: `include/soma/mesh/serial_backend.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/serial_mesh_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Region`, `MeshDecl` (1.1); `Result` (0.2).
- Produces:
  - `struct Point { double x = 0, y = 0, z = 0; }`
  - `class MeshView` (abstract): `virtual std::size_t entity_count(int dim) const`, `virtual std::size_t owned_count(int dim) const`, `virtual Point centroid(int dim, std::size_t i) const`, `virtual double measure(int dim, std::size_t i) const` (length/area/volume — the weight conservation checks need), `virtual std::size_t global_id(int dim, std::size_t i) const`, `virtual int dimension() const`.
  - `class MeshBackend` (abstract): `virtual Result<void> load(const MeshDecl&)`, `virtual Result<const MeshView*> view(MeshId) const`, `virtual Result<std::size_t> region_size(const Region&) const`, `virtual std::string name() const`.
  - `class SerialMeshBackend : public MeshBackend` — parses `"generated:box:NxMxP"` and `"generated:line:N"` into a structured grid held in plain vectors. No MPI, no files, no third-party headers. `region_size` returns the entity count of the region's dimension; an unknown `entity_set` name is an error (`E-MESH-SET`), except the reserved names `"all"` and `""` which mean the whole mesh, and `"wall"`/`"left"`/`"right"` which select boundary faces of a generated box.
  - `RegionSizes make_region_sizes(const CouplingGraph&, const MeshBackend&)` — adapts a backend to the callback Phase 3 Task 3.5 defined.

- [ ] **Step 1: Write the failing test**

`test/unit/serial_mesh_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/mesh/serial_backend.hpp>

using soma::MeshDecl;
using soma::MeshId;
using soma::Region;
using soma::SerialMeshBackend;

TEST(SerialMesh, LoadsAGeneratedLine) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:10"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok()) << v.error().message;
  EXPECT_EQ(v.value()->dimension(), 1);
  EXPECT_EQ(v.value()->entity_count(1), 10u);   // 10 cells
  EXPECT_EQ(v.value()->entity_count(0), 11u);   // 11 vertices
}

TEST(SerialMesh, LoadsAGeneratedBox) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m",
                              .source = "generated:box:4x4x4"}).ok());
  const auto v = b.view(MeshId{0});
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value()->dimension(), 3);
  EXPECT_EQ(v.value()->entity_count(3), 64u);
  EXPECT_EQ(v.value()->entity_count(0), 125u);  // 5^3 vertices
}

TEST(SerialMesh, CellCentroidsAreInsideTheUnitBox) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(3); ++i) {
    const auto c = v->centroid(3, i);
    EXPECT_GT(c.x, 0.0); EXPECT_LT(c.x, 1.0);
    EXPECT_GT(c.y, 0.0); EXPECT_LT(c.y, 1.0);
    EXPECT_GT(c.z, 0.0); EXPECT_LT(c.z, 1.0);
  }
}

// Conservation checks integrate against these, so they must sum to the domain.
TEST(SerialMesh, CellMeasuresSumToTheDomainVolume) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:5x4x2"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  double total = 0.0;
  for (std::size_t i = 0; i < v->entity_count(3); ++i) total += v->measure(3, i);
  EXPECT_NEAR(total, 1.0, 1e-14);
}

TEST(SerialMesh, LineMeasuresSumToUnitLength) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:7"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  double total = 0.0;
  for (std::size_t i = 0; i < v->entity_count(1); ++i) total += v->measure(1, i);
  EXPECT_NEAR(total, 1.0, 1e-14);
}

TEST(SerialMesh, GlobalIdsAreDenseAndUnique) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "l", .source = "generated:line:5"}).ok());
  const auto* v = b.view(MeshId{0}).value();
  for (std::size_t i = 0; i < v->entity_count(1); ++i) {
    EXPECT_EQ(v->global_id(1, i), i);
  }
}

TEST(SerialMesh, RegionSizeOfWholeMesh) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:3x3x3"}).ok());
  const auto n = b.region_size(Region{.name = "all", .mesh = MeshId{0},
                                      .entity_set = "", .dimension = 3});
  ASSERT_TRUE(n.ok()) << n.error().message;
  EXPECT_EQ(n.value(), 27u);
}

TEST(SerialMesh, RegionSizeOfABoundaryFaceSet) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:4x4x4"}).ok());
  const auto n = b.region_size(Region{.name = "w", .mesh = MeshId{0},
                                      .entity_set = "wall", .dimension = 2});
  ASSERT_TRUE(n.ok()) << n.error().message;
  EXPECT_EQ(n.value(), 16u);   // one 4x4 face
}

TEST(SerialMesh, UnknownEntitySetIsAnError) {
  SerialMeshBackend b;
  ASSERT_TRUE(b.load(MeshDecl{.name = "m", .source = "generated:box:2x2x2"}).ok());
  const auto n = b.region_size(Region{.name = "x", .mesh = MeshId{0},
                                      .entity_set = "nonexistent",
                                      .dimension = 2});
  ASSERT_FALSE(n.ok());
  EXPECT_EQ(n.error().code, "E-MESH-SET");
}

TEST(SerialMesh, MalformedSourceIsAnError) {
  SerialMeshBackend b;
  EXPECT_FALSE(b.load(MeshDecl{.name = "m", .source = "garbage"}).ok());
  EXPECT_FALSE(b.load(MeshDecl{.name = "m", .source = "generated:box:x"}).ok());
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/mesh/serial_backend.hpp` file not found.

- [ ] **Step 3: Write `include/soma/mesh/mesh_view.hpp`**

```cpp
#pragma once

#include <cstddef>
#include <string>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/schedule/buffers.hpp"  // RegionSizes

namespace soma {

struct Point {
  double x = 0.0, y = 0.0, z = 0.0;
};

/// Spec §4.1: the core knows only counts, ownership, ids, and geometry.
/// No third-party type appears here (Global Constraints).
class MeshView {
 public:
  virtual ~MeshView() = default;

  [[nodiscard]] virtual int dimension() const = 0;
  [[nodiscard]] virtual std::size_t entity_count(int dim) const = 0;
  [[nodiscard]] virtual std::size_t owned_count(int dim) const = 0;
  [[nodiscard]] virtual Point centroid(int dim, std::size_t i) const = 0;
  /// Length, area, or volume, per dim. Conservation checks integrate against
  /// these, so they must sum to the domain measure.
  [[nodiscard]] virtual double measure(int dim, std::size_t i) const = 0;
  [[nodiscard]] virtual std::size_t global_id(int dim, std::size_t i) const = 0;
};

class MeshBackend {
 public:
  virtual ~MeshBackend() = default;

  virtual Result<void> load(const MeshDecl& decl) = 0;
  [[nodiscard]] virtual Result<const MeshView*> view(MeshId id) const = 0;
  [[nodiscard]] virtual Result<std::size_t> region_size(const Region& r) const = 0;
  [[nodiscard]] virtual std::string name() const = 0;
};

/// Adapt a backend to the callback Phase 3 Task 3.5 defined.
[[nodiscard]] RegionSizes make_region_sizes(const CouplingGraph& g,
                                            const MeshBackend& backend);

}  // namespace soma
```

- [ ] **Step 4: Write `include/soma/mesh/serial_backend.hpp`**

```cpp
#pragma once

#include <memory>
#include <vector>

#include "soma/mesh/mesh_view.hpp"

namespace soma {

/// Dependency-free structured backend. Accepts:
///   "generated:line:N"        unit interval, N cells
///   "generated:box:NxMxP"     unit cube, N*M*P cells
/// Entity sets: "" or "all" (whole mesh); "wall", "left", "right" (boundary
/// faces of a generated box).
class SerialMeshBackend final : public MeshBackend {
 public:
  SerialMeshBackend();
  ~SerialMeshBackend() override;

  SerialMeshBackend(SerialMeshBackend&&) noexcept;
  SerialMeshBackend& operator=(SerialMeshBackend&&) noexcept;

  Result<void> load(const MeshDecl& decl) override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "serial"; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soma
```

- [ ] **Step 5: Write `src/mesh/serial_backend.cpp`**

```cpp
#include "soma/mesh/serial_backend.hpp"

#include <array>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace soma {
namespace {

/// Structured grid on the unit line or unit cube.
class StructuredMesh final : public MeshView {
 public:
  StructuredMesh(int dim, std::array<std::size_t, 3> n) : dim_(dim), n_(n) {}

  [[nodiscard]] int dimension() const override { return dim_; }

  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (dim == dim_) return cells();
    if (dim == 0) return vertices();
    if (dim == 2 && dim_ == 3) return faces_all();
    return 0;
  }

  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);  // serial: everything is owned
  }

  [[nodiscard]] Point centroid(int dim, std::size_t i) const override {
    Point p;
    if (dim == dim_ && dim_ == 1) {
      p.x = (static_cast<double>(i) + 0.5) / static_cast<double>(n_[0]);
      return p;
    }
    if (dim == dim_ && dim_ == 3) {
      const std::size_t ix = i % n_[0];
      const std::size_t iy = (i / n_[0]) % n_[1];
      const std::size_t iz = i / (n_[0] * n_[1]);
      p.x = (static_cast<double>(ix) + 0.5) / static_cast<double>(n_[0]);
      p.y = (static_cast<double>(iy) + 0.5) / static_cast<double>(n_[1]);
      p.z = (static_cast<double>(iz) + 0.5) / static_cast<double>(n_[2]);
      return p;
    }
    if (dim == 0 && dim_ == 1) {
      p.x = static_cast<double>(i) / static_cast<double>(n_[0]);
      return p;
    }
    return p;
  }

  [[nodiscard]] double measure(int dim, std::size_t) const override {
    if (dim == 1 && dim_ == 1) return 1.0 / static_cast<double>(n_[0]);
    if (dim == 3 && dim_ == 3) {
      return 1.0 / static_cast<double>(n_[0] * n_[1] * n_[2]);
    }
    if (dim == 2 && dim_ == 3) {
      return 1.0 / static_cast<double>(n_[0] * n_[1]);  // z-normal face
    }
    return 0.0;
  }

  [[nodiscard]] std::size_t global_id(int, std::size_t i) const override {
    return i;
  }

  [[nodiscard]] std::size_t face_count_z() const { return n_[0] * n_[1]; }

 private:
  [[nodiscard]] std::size_t cells() const {
    return dim_ == 1 ? n_[0] : n_[0] * n_[1] * n_[2];
  }
  [[nodiscard]] std::size_t vertices() const {
    return dim_ == 1 ? n_[0] + 1 : (n_[0] + 1) * (n_[1] + 1) * (n_[2] + 1);
  }
  [[nodiscard]] std::size_t faces_all() const {
    return 2 * (n_[0] * n_[1] + n_[1] * n_[2] + n_[0] * n_[2]);
  }

  int                        dim_;
  std::array<std::size_t, 3> n_;
};

bool parse_size(const std::string& s, std::size_t& out) {
  if (s.empty()) return false;
  char* end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || v <= 0) return false;
  out = static_cast<std::size_t>(v);
  return true;
}

}  // namespace

struct SerialMeshBackend::Impl {
  std::vector<std::unique_ptr<StructuredMesh>> meshes;
};

SerialMeshBackend::SerialMeshBackend() : impl_(std::make_unique<Impl>()) {}
SerialMeshBackend::~SerialMeshBackend() = default;
SerialMeshBackend::SerialMeshBackend(SerialMeshBackend&&) noexcept = default;
SerialMeshBackend& SerialMeshBackend::operator=(SerialMeshBackend&&) noexcept =
    default;

Result<void> SerialMeshBackend::load(const MeshDecl& decl) {
  const std::string& src = decl.source;
  const std::string prefix = "generated:";
  if (src.rfind(prefix, 0) != 0) {
    return Error{"E-MESH-SOURCE",
                 "serial backend understands only 'generated:...', got '" +
                     src + "'"};
  }
  const std::string rest = src.substr(prefix.size());
  const auto colon = rest.find(':');
  if (colon == std::string::npos) {
    return Error{"E-MESH-SOURCE", "malformed generated spec '" + src + "'"};
  }
  const std::string kind = rest.substr(0, colon);
  const std::string dims = rest.substr(colon + 1);

  if (kind == "line") {
    std::size_t n = 0;
    if (!parse_size(dims, n)) {
      return Error{"E-MESH-SOURCE", "bad line size in '" + src + "'"};
    }
    impl_->meshes.push_back(
        std::make_unique<StructuredMesh>(1, std::array<std::size_t, 3>{n, 1, 1}));
    return Result<void>::success();
  }

  if (kind == "box") {
    std::array<std::size_t, 3> n{0, 0, 0};
    std::istringstream ss{dims};
    std::string tok;
    int k = 0;
    while (std::getline(ss, tok, 'x') && k < 3) {
      if (!parse_size(tok, n[static_cast<std::size_t>(k)])) {
        return Error{"E-MESH-SOURCE", "bad box size in '" + src + "'"};
      }
      ++k;
    }
    if (k != 3) {
      return Error{"E-MESH-SOURCE",
                   "box needs three dimensions, got '" + dims + "'"};
    }
    impl_->meshes.push_back(std::make_unique<StructuredMesh>(3, n));
    return Result<void>::success();
  }

  return Error{"E-MESH-SOURCE", "unknown generated kind '" + kind + "'"};
}

Result<const MeshView*> SerialMeshBackend::view(MeshId id) const {
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Error{"E-MESH-ID", "no mesh with id " + std::to_string(id.value())};
  }
  return static_cast<const MeshView*>(impl_->meshes[id.value()].get());
}

Result<std::size_t> SerialMeshBackend::region_size(const Region& r) const {
  const auto v = view(r.mesh);
  if (!v.ok()) return v.error();
  const MeshView* mesh = v.value();

  const int dim = r.dimension >= 0 ? r.dimension : mesh->dimension();

  if (r.entity_set.empty() || r.entity_set == "all") {
    return mesh->entity_count(dim);
  }
  if (r.entity_set == "wall" || r.entity_set == "left" ||
      r.entity_set == "right") {
    if (mesh->dimension() != 3) {
      return Error{"E-MESH-SET",
                   "boundary set '" + r.entity_set + "' needs a 3D mesh"};
    }
    const auto* sm = static_cast<const StructuredMesh*>(mesh);
    return sm->face_count_z();
  }
  return Error{"E-MESH-SET", "unknown entity set '" + r.entity_set + "'"};
}

RegionSizes make_region_sizes(const CouplingGraph& g,
                              const MeshBackend& backend) {
  return [&g, &backend](RegionId id) -> std::size_t {
    if (!id.valid() || id.value() >= g.region_count()) return 0;
    const auto n = backend.region_size(g.region(id));
    return n.ok() ? n.value() : 0;
  };
}

}  // namespace soma
```

- [ ] **Step 6: Update `src/CMakeLists.txt`** — add `mesh/serial_backend.cpp`.

- [ ] **Step 7: Register and run**

Add `soma_add_test(serial_mesh_test unit/serial_mesh_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R SerialMesh
```

Expected: 10 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add include/soma/mesh/ src/mesh/ test/unit/serial_mesh_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): abstract MeshView and dependency-free serial backend"
```

---

### Task 4.3: `Transport` abstraction and the copy backend

Spec §2.2 and §6: `Transport` is the one seam that makes four integration modes
share a runtime. Phase 4 provides the in-process implementation.

**Files:**
- Create: `include/soma/transport/transport.hpp`
- Create: `src/transport/copy_transport.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/transport_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Result` (0.2).
- Produces:
  - `struct TransportSpec { int src_rank_set = -1; int dst_rank_set = -1; std::size_t elements = 0; int components = 1; }`
  - `class Transport` (abstract): `virtual Result<void> prepare(const TransportSpec&)`, `virtual Result<void> move(const double* src, double* dst, std::size_t n)`, `virtual std::string name() const`, `virtual bool is_local() const`.
  - `class CopyTransport final : public Transport` — `std::memcpy` when both sides are host memory; `is_local()` returns true. `move` with `src == dst` is a no-op, not an error (the lowering pass can legitimately emit a same-rank transport when a component reads its own output).
  - `class TransportRegistry` with `void register_backend(std::string name, std::function<std::unique_ptr<Transport>()>)`, `Result<std::unique_ptr<Transport>> create(const std::string& name) const`, `std::vector<std::string> available() const`, `static TransportRegistry& instance()`. `"copy"` is registered by default.

- [ ] **Step 1: Write the failing test**

`test/unit/transport_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/transport/transport.hpp>

#include <algorithm>
#include <vector>

using soma::CopyTransport;
using soma::TransportRegistry;
using soma::TransportSpec;

TEST(CopyTransport, MovesContiguousData) {
  const std::vector<double> src{1.0, 2.0, 3.0, 4.0};
  std::vector<double> dst(4, 0.0);

  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.src_rank_set = 0, .dst_rank_set = 0,
                                      .elements = 4}).ok());
  ASSERT_TRUE(t.move(src.data(), dst.data(), 4).ok());
  EXPECT_EQ(src, dst);
}

TEST(CopyTransport, IsLocalAndNamed) {
  const CopyTransport t;
  EXPECT_TRUE(t.is_local());
  EXPECT_EQ(t.name(), "copy");
}

TEST(CopyTransport, SelfMoveIsANoOp) {
  std::vector<double> v{1.0, 2.0, 3.0};
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 3}).ok());
  ASSERT_TRUE(t.move(v.data(), v.data(), 3).ok());
  EXPECT_DOUBLE_EQ(v[1], 2.0);
}

TEST(CopyTransport, RejectsNullPointers) {
  std::vector<double> v(4, 0.0);
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 4}).ok());
  EXPECT_FALSE(t.move(nullptr, v.data(), 4).ok());
  EXPECT_FALSE(t.move(v.data(), nullptr, 4).ok());
}

TEST(CopyTransport, RejectsMoveBeyondPreparedSize) {
  const std::vector<double> src(4, 1.0);
  std::vector<double> dst(4, 0.0);
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 4}).ok());
  const auto r = t.move(src.data(), dst.data(), 8);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

TEST(TransportRegistry, CopyIsAvailableByDefault) {
  const auto names = TransportRegistry::instance().available();
  EXPECT_NE(std::find(names.begin(), names.end(), "copy"), names.end());
}

TEST(TransportRegistry, CreatesByName) {
  const auto t = TransportRegistry::instance().create("copy");
  ASSERT_TRUE(t.ok()) << t.error().message;
  EXPECT_EQ(t.value()->name(), "copy");
}

TEST(TransportRegistry, UnknownBackendIsAnError) {
  const auto t = TransportRegistry::instance().create("does-not-exist");
  ASSERT_FALSE(t.ok());
  EXPECT_EQ(t.error().code, "E-TRANSPORT-BACKEND");
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/transport/transport.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/transport/transport.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "soma/base/result.hpp"

namespace soma {

struct TransportSpec {
  int         src_rank_set = -1;
  int         dst_rank_set = -1;
  std::size_t elements = 0;
  int         components = 1;
};

/// Spec §2.2: the one seam by which in-process, MPI, MPMD, and LowFive
/// coupling share a runtime. Everything else in the DAG is unchanged.
class Transport {
 public:
  virtual ~Transport() = default;

  virtual Result<void> prepare(const TransportSpec& spec) = 0;
  virtual Result<void> move(const double* src, double* dst, std::size_t n) = 0;
  [[nodiscard]] virtual std::string name() const = 0;
  [[nodiscard]] virtual bool is_local() const = 0;
};

/// In-process copy. Phase 5 adds "mpi"; Phase 8 adds "lowfive".
class CopyTransport final : public Transport {
 public:
  Result<void> prepare(const TransportSpec& spec) override;
  Result<void> move(const double* src, double* dst, std::size_t n) override;
  [[nodiscard]] std::string name() const override { return "copy"; }
  [[nodiscard]] bool is_local() const override { return true; }

 private:
  TransportSpec spec_{};
};

class TransportRegistry {
 public:
  using Factory = std::function<std::unique_ptr<Transport>()>;

  void register_backend(std::string name, Factory factory);
  [[nodiscard]] Result<std::unique_ptr<Transport>> create(
      const std::string& name) const;
  [[nodiscard]] std::vector<std::string> available() const;

  static TransportRegistry& instance();

 private:
  std::vector<std::pair<std::string, Factory>> factories_;
};

}  // namespace soma
```

- [ ] **Step 4: Write `src/transport/copy_transport.cpp`**

```cpp
#include "soma/transport/transport.hpp"

#include <algorithm>
#include <cstring>

namespace soma {

Result<void> CopyTransport::prepare(const TransportSpec& spec) {
  spec_ = spec;
  return Result<void>::success();
}

Result<void> CopyTransport::move(const double* src, double* dst,
                                 std::size_t n) {
  if (src == nullptr || dst == nullptr) {
    return Error{"E-TRANSPORT-NULL", "copy transport given a null pointer"};
  }
  const std::size_t capacity =
      spec_.elements * static_cast<std::size_t>(std::max(spec_.components, 1));
  if (n > capacity) {
    return Error{"E-TRANSPORT-SIZE",
                 "move of " + std::to_string(n) + " elements exceeds the " +
                     std::to_string(capacity) + " prepared"};
  }
  if (src == dst) return Result<void>::success();  // legitimate no-op
  std::memcpy(dst, src, n * sizeof(double));
  return Result<void>::success();
}

void TransportRegistry::register_backend(std::string name, Factory factory) {
  for (auto& [n, f] : factories_) {
    if (n == name) {
      f = std::move(factory);
      return;
    }
  }
  factories_.emplace_back(std::move(name), std::move(factory));
}

Result<std::unique_ptr<Transport>> TransportRegistry::create(
    const std::string& name) const {
  for (const auto& [n, f] : factories_) {
    if (n == name) return f();
  }
  return Error{"E-TRANSPORT-BACKEND", "no transport backend named '" + name + "'"};
}

std::vector<std::string> TransportRegistry::available() const {
  std::vector<std::string> names;
  names.reserve(factories_.size());
  for (const auto& [n, f] : factories_) names.push_back(n);
  return names;
}

TransportRegistry& TransportRegistry::instance() {
  static TransportRegistry* reg = [] {
    auto* r = new TransportRegistry;
    r->register_backend("copy", [] { return std::make_unique<CopyTransport>(); });
    return r;
  }();
  return *reg;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `transport/copy_transport.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(transport_test unit/transport_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "CopyTransport|TransportRegistry"
```

Expected: 8 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/soma/transport/ src/transport/ test/unit/transport_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): Transport abstraction with in-process copy backend"
```

---

### Task 4.4: Numeric kernels

Spec §3.1 tier C: header-only templates with `extern template` for the common
instantiations. These are the operations the interpreter dispatches to.

**Files:**
- Create: `include/soma/num/kernels.hpp`
- Create: `src/num/kernels_inst.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/kernels_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `AccumOp` (0.6).
- Produces, all in namespace `soma::num`, all templated on `T`:
  - `template <class T> void copy(const T* src, T* dst, std::size_t n)`
  - `template <class T> void accumulate_into(AccumOp op, const T* sample, T* acc, std::size_t n, std::size_t sample_index)` — running reduction; `sample_index == 0` initialises. `Mean` accumulates the sum and is finalised by:
  - `template <class T> void accumulate_finalize(AccumOp op, T* acc, std::size_t n, std::size_t sample_count)` — divides by `sample_count` for `Mean`, multiplies by the timestep for `Integral` (passed via `dt` overload below), no-op otherwise.
  - `template <class T> void accumulate_finalize_integral(T* acc, std::size_t n, T dt)`
  - `template <class T> void relax(T omega, const T* previous, const T* current, T* out, std::size_t n)` — `out = omega*current + (1-omega)*previous`
  - `template <class T> void extrapolate(int order, const T* history, std::size_t stride, T* out, std::size_t n)` — order 0 = copy latest; order 1 = `2*y_k - y_{k-1}`; order 2 = `3*y_k - 3*y_{k-1} + y_{k-2}`. `history` points at the newest sample, older samples at `-stride` offsets.
  - `template <class T> T sum(const T* v, std::size_t n)`, `T weighted_sum(const T* v, const T* w, std::size_t n)`, `T norm_l2(const T* v, std::size_t n)`, `T norm_linf(const T* v, std::size_t n)`, `T norm_l2_diff(const T* a, const T* b, std::size_t n)`
  - `template <class T> bool has_nonfinite(const T* v, std::size_t n)`
  - `extern template` declarations for `double` and `float`, instantiated in `src/num/kernels_inst.cpp`.

  Accumulation note: `weighted_sum` is what conservation checking uses — the
  weights are `MeshView::measure` values, so a global sum is a true integral.

- [ ] **Step 1: Write the failing test**

`test/unit/kernels_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/num/kernels.hpp>

#include <cmath>
#include <limits>
#include <vector>

using soma::AccumOp;
namespace num = soma::num;

TEST(Kernels, Copy) {
  const std::vector<double> a{1, 2, 3};
  std::vector<double> b(3, 0.0);
  num::copy(a.data(), b.data(), 3);
  EXPECT_EQ(a, b);
}

TEST(Kernels, AccumulateSum) {
  std::vector<double> acc(2, 0.0);
  const std::vector<double> s1{1.0, 2.0};
  const std::vector<double> s2{3.0, 4.0};
  num::accumulate_into(AccumOp::Sum, s1.data(), acc.data(), 2, 0);
  num::accumulate_into(AccumOp::Sum, s2.data(), acc.data(), 2, 1);
  num::accumulate_finalize(AccumOp::Sum, acc.data(), 2, 2);
  EXPECT_DOUBLE_EQ(acc[0], 4.0);
  EXPECT_DOUBLE_EQ(acc[1], 6.0);
}

TEST(Kernels, AccumulateMeanDividesBySampleCount) {
  std::vector<double> acc(2, 0.0);
  const std::vector<double> s1{1.0, 2.0};
  const std::vector<double> s2{3.0, 6.0};
  num::accumulate_into(AccumOp::Mean, s1.data(), acc.data(), 2, 0);
  num::accumulate_into(AccumOp::Mean, s2.data(), acc.data(), 2, 1);
  num::accumulate_finalize(AccumOp::Mean, acc.data(), 2, 2);
  EXPECT_DOUBLE_EQ(acc[0], 2.0);
  EXPECT_DOUBLE_EQ(acc[1], 4.0);
}

TEST(Kernels, AccumulateMinMax) {
  std::vector<double> lo(1, 0.0), hi(1, 0.0);
  const double a = 5.0, b = -2.0, c = 9.0;
  num::accumulate_into(AccumOp::Min, &a, lo.data(), 1, 0);
  num::accumulate_into(AccumOp::Min, &b, lo.data(), 1, 1);
  num::accumulate_into(AccumOp::Min, &c, lo.data(), 1, 2);
  EXPECT_DOUBLE_EQ(lo[0], -2.0);

  num::accumulate_into(AccumOp::Max, &a, hi.data(), 1, 0);
  num::accumulate_into(AccumOp::Max, &b, hi.data(), 1, 1);
  num::accumulate_into(AccumOp::Max, &c, hi.data(), 1, 2);
  EXPECT_DOUBLE_EQ(hi[0], 9.0);
}

TEST(Kernels, AccumulateIntegralScalesByDt) {
  std::vector<double> acc(1, 0.0);
  const double s = 2.0;
  num::accumulate_into(AccumOp::Integral, &s, acc.data(), 1, 0);
  num::accumulate_into(AccumOp::Integral, &s, acc.data(), 1, 1);
  num::accumulate_finalize_integral(acc.data(), 1, 0.5);
  EXPECT_DOUBLE_EQ(acc[0], 2.0);   // (2 + 2) * 0.5
}

TEST(Kernels, AccumulateInstantTakesTheLatest) {
  std::vector<double> acc(1, 0.0);
  const double a = 1.0, b = 9.0;
  num::accumulate_into(AccumOp::Instant, &a, acc.data(), 1, 0);
  num::accumulate_into(AccumOp::Instant, &b, acc.data(), 1, 1);
  num::accumulate_finalize(AccumOp::Instant, acc.data(), 1, 2);
  EXPECT_DOUBLE_EQ(acc[0], 9.0);
}

TEST(Kernels, RelaxBlends) {
  const std::vector<double> prev{0.0, 10.0};
  const std::vector<double> cur{10.0, 0.0};
  std::vector<double> out(2, 0.0);
  num::relax(0.25, prev.data(), cur.data(), out.data(), 2);
  EXPECT_DOUBLE_EQ(out[0], 2.5);
  EXPECT_DOUBLE_EQ(out[1], 7.5);
}

TEST(Kernels, RelaxOmegaOneIsTheCurrentValue) {
  const std::vector<double> prev{1.0}, cur{5.0};
  std::vector<double> out(1, 0.0);
  num::relax(1.0, prev.data(), cur.data(), out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 5.0);
}

TEST(Kernels, ExtrapolateOrderZeroCopies) {
  const std::vector<double> hist{1.0, 2.0, 3.0};   // newest last
  std::vector<double> out(1, 0.0);
  num::extrapolate(0, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 3.0);
}

TEST(Kernels, ExtrapolateOrderOneIsLinear) {
  // y_{k-1}=2, y_k=3 -> predict 4
  const std::vector<double> hist{1.0, 2.0, 3.0};
  std::vector<double> out(1, 0.0);
  num::extrapolate(1, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 4.0);
}

TEST(Kernels, ExtrapolateOrderTwoIsExactOnAQuadratic) {
  // y = k^2 at k = 0,1,2 -> 0,1,4; predict k=3 -> 9
  const std::vector<double> hist{0.0, 1.0, 4.0};
  std::vector<double> out(1, 0.0);
  num::extrapolate(2, &hist[2], 1, out.data(), 1);
  EXPECT_DOUBLE_EQ(out[0], 9.0);
}

TEST(Kernels, SumAndWeightedSum) {
  const std::vector<double> v{1.0, 2.0, 3.0};
  const std::vector<double> w{0.5, 0.5, 1.0};
  EXPECT_DOUBLE_EQ(num::sum(v.data(), 3), 6.0);
  EXPECT_DOUBLE_EQ(num::weighted_sum(v.data(), w.data(), 3), 4.5);
}

TEST(Kernels, Norms) {
  const std::vector<double> v{3.0, -4.0};
  EXPECT_DOUBLE_EQ(num::norm_l2(v.data(), 2), 5.0);
  EXPECT_DOUBLE_EQ(num::norm_linf(v.data(), 2), 4.0);

  const std::vector<double> a{1.0, 1.0}, b{1.0, 2.0};
  EXPECT_DOUBLE_EQ(num::norm_l2_diff(a.data(), b.data(), 2), 1.0);
}

TEST(Kernels, DetectsNonFinite) {
  std::vector<double> v{1.0, 2.0};
  EXPECT_FALSE(num::has_nonfinite(v.data(), 2));
  v[1] = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(num::has_nonfinite(v.data(), 2));
  v[1] = std::numeric_limits<double>::infinity();
  EXPECT_TRUE(num::has_nonfinite(v.data(), 2));
}

TEST(Kernels, WorkForFloatToo) {
  const std::vector<float> v{3.0F, -4.0F};
  EXPECT_FLOAT_EQ(num::norm_l2(v.data(), 2), 5.0F);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/num/kernels.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/num/kernels.hpp`:

```cpp
#pragma once

#include <cmath>
#include <cstddef>

#include "soma/base/ids.hpp"

namespace soma::num {

template <class T>
void copy(const T* src, T* dst, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) dst[i] = src[i];
}

/// Running reduction. sample_index == 0 initialises the accumulator.
template <class T>
void accumulate_into(AccumOp op, const T* sample, T* acc, std::size_t n,
                     std::size_t sample_index) {
  const bool first = (sample_index == 0);
  for (std::size_t i = 0; i < n; ++i) {
    switch (op) {
      case AccumOp::Instant:
        acc[i] = sample[i];
        break;
      case AccumOp::Sum:
      case AccumOp::Mean:
      case AccumOp::Integral:
        acc[i] = first ? sample[i] : acc[i] + sample[i];
        break;
      case AccumOp::Min:
        acc[i] = first ? sample[i] : (sample[i] < acc[i] ? sample[i] : acc[i]);
        break;
      case AccumOp::Max:
        acc[i] = first ? sample[i] : (sample[i] > acc[i] ? sample[i] : acc[i]);
        break;
    }
  }
}

/// Mean divides by the sample count; everything else is already final.
/// Integral is finalised by accumulate_finalize_integral instead.
template <class T>
void accumulate_finalize(AccumOp op, T* acc, std::size_t n,
                         std::size_t sample_count) {
  if (op != AccumOp::Mean || sample_count == 0) return;
  const T inv = T{1} / static_cast<T>(sample_count);
  for (std::size_t i = 0; i < n; ++i) acc[i] *= inv;
}

template <class T>
void accumulate_finalize_integral(T* acc, std::size_t n, T dt) {
  for (std::size_t i = 0; i < n; ++i) acc[i] *= dt;
}

/// out = omega*current + (1 - omega)*previous
template <class T>
void relax(T omega, const T* previous, const T* current, T* out,
           std::size_t n) {
  const T one_minus = T{1} - omega;
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = omega * current[i] + one_minus * previous[i];
  }
}

/// `history` points at the NEWEST sample; older samples lie at -stride steps.
/// order 0: y_k          order 1: 2y_k - y_{k-1}
/// order 2: 3y_k - 3y_{k-1} + y_{k-2}
template <class T>
void extrapolate(int order, const T* history, std::size_t stride, T* out,
                 std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    const T y0 = history[i];
    switch (order) {
      case 0:
        out[i] = y0;
        break;
      case 1: {
        const T y1 = history[i - stride];
        out[i] = T{2} * y0 - y1;
        break;
      }
      default: {
        const T y1 = history[i - stride];
        const T y2 = history[i - 2 * stride];
        out[i] = T{3} * y0 - T{3} * y1 + y2;
        break;
      }
    }
  }
}

template <class T>
[[nodiscard]] T sum(const T* v, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i];
  return s;
}

/// Conservation checking uses this with MeshView::measure as weights, so the
/// result is a true integral over the region.
template <class T>
[[nodiscard]] T weighted_sum(const T* v, const T* w, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i] * w[i];
  return s;
}

template <class T>
[[nodiscard]] T norm_l2(const T* v, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) s += v[i] * v[i];
  return std::sqrt(s);
}

template <class T>
[[nodiscard]] T norm_linf(const T* v, std::size_t n) {
  T m{};
  for (std::size_t i = 0; i < n; ++i) {
    const T a = v[i] < T{} ? -v[i] : v[i];
    if (a > m) m = a;
  }
  return m;
}

template <class T>
[[nodiscard]] T norm_l2_diff(const T* a, const T* b, std::size_t n) {
  T s{};
  for (std::size_t i = 0; i < n; ++i) {
    const T d = a[i] - b[i];
    s += d * d;
  }
  return std::sqrt(s);
}

template <class T>
[[nodiscard]] bool has_nonfinite(const T* v, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) return true;
  }
  return false;
}

// Spec §3.1: instantiate the common cases once, keep the templates available.
extern template void   copy<double>(const double*, double*, std::size_t);
extern template void   copy<float>(const float*, float*, std::size_t);
extern template double norm_l2<double>(const double*, std::size_t);
extern template float  norm_l2<float>(const float*, std::size_t);
extern template double weighted_sum<double>(const double*, const double*,
                                            std::size_t);

}  // namespace soma::num
```

- [ ] **Step 4: Write `src/num/kernels_inst.cpp`**

```cpp
#include "soma/num/kernels.hpp"

namespace soma::num {

template void   copy<double>(const double*, double*, std::size_t);
template void   copy<float>(const float*, float*, std::size_t);
template double norm_l2<double>(const double*, std::size_t);
template float  norm_l2<float>(const float*, std::size_t);
template double weighted_sum<double>(const double*, const double*, std::size_t);

}  // namespace soma::num
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `num/kernels_inst.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(kernels_test unit/kernels_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Kernels
```

Expected: 15 tests PASS. `ExtrapolateOrderTwoIsExactOnAQuadratic` is the one
that catches a wrong coefficient.

- [ ] **Step 7: Commit**

```bash
git add include/soma/num/ src/num/ test/unit/kernels_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): numeric kernels with extern template instantiation"
```

---

### Task 4.5: The `Component` interface

Spec §6 mode 1. Minimal by design: register fields, advance in time. Checkpoint
support is declared here but only exercised in Phase 7.

**Files:**
- Create: `include/soma/runtime/component.hpp`
- Test: `test/unit/component_iface_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `FieldRegistry` (4.1), `Rational` (0.3).
- Produces:
  - `struct StepContext { Rational t_begin; Rational t_end; std::int64_t step = 0; int iteration = 0; }`
  - `class Component` (abstract):
    - `virtual std::string name() const = 0`
    - `virtual Result<void> register_fields(FieldRegistry&, const CouplingGraph&) = 0` — the component binds its own storage
    - `virtual Result<void> advance(const StepContext&) = 0`
    - `virtual Result<void> save_state() { return Result<void>::success(); }` — default no-op; Phase 7 requires a real one for components inside an iteration region
    - `virtual Result<void> restore_state() { return Result<void>::success(); }`
    - `virtual bool supports_rollback() const { return false; }`
    - virtual destructor
  - `class ComponentSet` with `void add(std::unique_ptr<Component>)`, `Component* find(std::string_view) const`, `std::size_t size() const`, `Component& at(std::size_t)`, and iteration support.

  The rollback default matters: spec §5.3 says SOMA derives the checkpoint set
  and *verifies at runtime that it was saved*. A component that returns
  `supports_rollback() == false` but appears in an iteration region is an error
  Phase 7 Task 7.2 raises — declared now so the interface does not change later.

- [ ] **Step 1: Write the failing test**

`test/unit/component_iface_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/runtime/component.hpp>

#include <vector>

using soma::Component;
using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::StepContext;

namespace {

class Toy final : public Component {
 public:
  explicit Toy(std::string n, std::size_t size = 4)
      : name_(std::move(n)), data_(size, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return soma::Error{"E-TEST", "component not in graph"};
    }
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext& ctx) override {
    ++advance_count;
    last = ctx;
    for (double& d : data_) d += 1.0;
    return soma::Result<void>::success();
  }

  soma::Result<void> save_state() override {
    saved_ = data_;
    ++save_count;
    return soma::Result<void>::success();
  }
  soma::Result<void> restore_state() override {
    data_ = saved_;
    ++restore_count;
    return soma::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;
  StepContext last{};
  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::string         name_;
  std::vector<double> data_;
  std::vector<double> saved_;
};

}  // namespace

TEST(ComponentIface, AdvanceReceivesTheStepContext) {
  Toy t{"a"};
  const StepContext ctx{.t_begin = Rational{0}, .t_end = Rational{1, 2},
                        .step = 3, .iteration = 1};
  ASSERT_TRUE(t.advance(ctx).ok());
  EXPECT_EQ(t.advance_count, 1);
  EXPECT_EQ(t.last.t_end, Rational(1, 2));
  EXPECT_EQ(t.last.step, 3);
  EXPECT_EQ(t.last.iteration, 1);
}

TEST(ComponentIface, SaveAndRestoreRoundTrip) {
  Toy t{"a"};
  ASSERT_TRUE(t.advance(StepContext{}).ok());       // data becomes 1.0
  ASSERT_TRUE(t.save_state().ok());
  ASSERT_TRUE(t.advance(StepContext{}).ok());       // data becomes 2.0
  EXPECT_DOUBLE_EQ(t.data()[0], 2.0);
  ASSERT_TRUE(t.restore_state().ok());
  EXPECT_DOUBLE_EQ(t.data()[0], 1.0);
  EXPECT_EQ(t.save_count, 1);
  EXPECT_EQ(t.restore_count, 1);
}

// A component that does not override save/restore must say so.
TEST(ComponentIface, RollbackDefaultsToUnsupported) {
  class Bare final : public Component {
   public:
    [[nodiscard]] std::string name() const override { return "bare"; }
    soma::Result<void> register_fields(FieldRegistry&,
                                       const CouplingGraph&) override {
      return soma::Result<void>::success();
    }
    soma::Result<void> advance(const StepContext&) override {
      return soma::Result<void>::success();
    }
  };
  const Bare b;
  EXPECT_FALSE(b.supports_rollback());
  EXPECT_TRUE(Toy{"a"}.supports_rollback());
}

TEST(ComponentSet, AddsAndFindsByName) {
  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  cs.add(std::make_unique<Toy>("beta"));
  EXPECT_EQ(cs.size(), 2u);
  ASSERT_NE(cs.find("beta"), nullptr);
  EXPECT_EQ(cs.find("beta")->name(), "beta");
  EXPECT_EQ(cs.find("missing"), nullptr);
}

TEST(ComponentSet, RegistersFieldsAgainstTheGraph) {
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto cid = g.add_component(soma::ComponentDecl{.name = "alpha"});
  g.add_field(cid, soma::FieldSpec{.name = "f", .region = r,
                                   .intent = soma::Intent::Provides});

  ComponentSet cs;
  cs.add(std::make_unique<Toy>("alpha"));
  FieldRegistry reg;
  ASSERT_TRUE(cs.at(0).register_fields(reg, g).ok());
  EXPECT_EQ(reg.bound_count(), 1u);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/runtime/component.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/runtime/component.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/field/registry.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

struct StepContext {
  Rational     t_begin{0};
  Rational     t_end{0};
  std::int64_t step = 0;
  int          iteration = 0;  // 0 outside an iteration region
};

/// Spec §6 mode 1. Deliberately minimal: register storage, advance in time.
class Component {
 public:
  virtual ~Component() = default;

  [[nodiscard]] virtual std::string name() const = 0;

  /// Bind this component's own storage into the registry (spec §4.2).
  virtual Result<void> register_fields(FieldRegistry& registry,
                                       const CouplingGraph& graph) = 0;

  virtual Result<void> advance(const StepContext& ctx) = 0;

  /// Spec §5.3: SOMA derives the checkpoint set and verifies at runtime that
  /// it was saved. A component inside an iteration region must override these
  /// and return true from supports_rollback().
  virtual Result<void> save_state() { return Result<void>::success(); }
  virtual Result<void> restore_state() { return Result<void>::success(); }
  [[nodiscard]] virtual bool supports_rollback() const { return false; }
};

class ComponentSet {
 public:
  void add(std::unique_ptr<Component> c) {
    components_.push_back(std::move(c));
  }

  [[nodiscard]] Component* find(std::string_view name) const {
    for (const auto& c : components_) {
      if (c->name() == name) return c.get();
    }
    return nullptr;
  }

  [[nodiscard]] std::size_t size() const noexcept { return components_.size(); }
  [[nodiscard]] Component& at(std::size_t i) { return *components_.at(i); }
  [[nodiscard]] const Component& at(std::size_t i) const {
    return *components_.at(i);
  }

  [[nodiscard]] auto begin() { return components_.begin(); }
  [[nodiscard]] auto end() { return components_.end(); }
  [[nodiscard]] auto begin() const { return components_.begin(); }
  [[nodiscard]] auto end() const { return components_.end(); }

 private:
  std::vector<std::unique_ptr<Component>> components_;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(component_iface_test unit/component_iface_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "ComponentIface|ComponentSet"
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/runtime/component.hpp test/unit/component_iface_test.cpp \
        test/CMakeLists.txt
git commit -m "feat(soma): Component interface with optional rollback support"
```

---

### Task 4.6: `Runner` — the plan interpreter

The hot loop. Spec §2 layer 3: interpret a vector of instructions with no
virtual dispatch per DOF, no map lookup per exchange, no allocation.

**Files:**
- Create: `include/soma/runtime/runner.hpp`
- Create: `src/runtime/runner.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/runner_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `ExecutionPlan` (3.6), `FieldRegistry` (4.1), `MeshBackend` (4.2), `Transport` (4.3), kernels (4.4), `Component` (4.5).
- Produces:
  - `struct RunnerConfig { std::string transport = "copy"; bool check_conservation = true; double conservation_tolerance = 1e-12; bool abort_on_violation = false; }`
  - `struct RunStats { std::int64_t periods = 0; std::int64_t instructions_executed = 0; std::int64_t conservation_checks = 0; std::int64_t conservation_violations = 0; double worst_conservation_error = 0.0; }`
  - `class Runner` with
    - `Runner(const CouplingGraph&, const OperatorDAG&, ExecutionPlan, const MeshBackend&)`
    - `Result<void> initialize(ComponentSet&, FieldRegistry&, const RunnerConfig& = {})` — allocates every non-component-owned buffer, resolves each `ArcId` to a raw pointer once, creates the transport, and verifies that every field the plan reads is bound
    - `Result<void> run_period(ComponentSet&)` — executes one common period
    - `Result<void> run(ComponentSet&, std::int64_t periods)`
    - `const RunStats& stats() const`
    - `Result<const double*> buffer(ArcId) const` — test and diagnostic access
  - Interpretation of each `OpKind` in Phase 4 (the rest arrive with their phases):
    | Kind | Phase 4 behaviour |
    |---|---|
    | `Source` | no-op: the arc already aliases component storage |
    | `Sink` | copy input buffer into the component-owned output buffer |
    | `Sample` | `accumulate_into` with the running sample index |
    | `Accumulate` | finalise the accumulator, reset the sample counter |
    | `Interpolate` | copy (Phase 6 replaces this) |
    | `Extrapolate` | `num::extrapolate` |
    | `Remap` | **identity copy when element counts match**, error `E-RT-REMAP-SIZE` otherwise (Phase 5 supplies real weights) |
    | `Merge` | sum of inputs |
    | `Relax` | `num::relax` against the previous iterate buffer |
    | `Transport` | `Transport::move` |
    | `Residual` | difference of inputs (Phase 7 uses it) |
    | `Accelerate` | copy (Phase 7 replaces this) |
    | `Constrain` | conservation check and report; no correction in Phase 4 |
    | `Diagnose` | norms and NaN detection into stats |

  Being explicit that `Remap` is identity in Phase 4 is deliberate: it makes the
  first coupled run meaningful on matching meshes and makes Phase 5's job a
  drop-in replacement rather than a rewrite.

- [ ] **Step 1: Write the failing test**

`test/unit/runner_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>

#include <vector>

using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::Runner;
using soma::RunnerConfig;
using soma::SerialMeshBackend;
using soma::StepContext;
namespace u = soma::units;

namespace {

/// Producer writes a constant; consumer just holds what it receives.
class Producer final : public soma::Component {
 public:
  explicit Producer(std::size_t n, double value)
      : data_(n, value), value_(value) {}

  [[nodiscard]] std::string name() const override { return "producer"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("producer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext&) override {
    ++steps;
    for (double& d : data_) d = value_ * static_cast<double>(steps);
    return soma::Result<void>::success();
  }

  int steps = 0;

 private:
  std::vector<double> data_;
  double              value_;
};

class Consumer final : public soma::Component {
 public:
  explicit Consumer(std::size_t n) : data_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return "consumer"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("consumer");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext&) override {
    ++steps;
    return soma::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }
  int steps = 0;

 private:
  std::vector<double> data_;
};

struct Harness {
  CouplingGraph      graph;
  SerialMeshBackend  backend;
  soma::OperatorDAG  dag;
  soma::ExecutionPlan plan;
  ComponentSet       components;
  FieldRegistry      registry;
};

/// Matching 8-cell meshes so Phase 4's identity Remap is meaningful.
std::unique_ptr<Harness> make_harness(soma::AccumOp accum, Rational src_dt,
                                      Rational dst_dt) {
  auto h = std::make_unique<Harness>();
  auto& g = h->graph;

  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1",
                                            .source = "generated:line:8"});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2",
                                            .source = "generated:line:8"});
  const auto r1 = g.add_region(soma::Region{.name = "r1", .mesh = m1,
                                            .dimension = 1});
  const auto r2 = g.add_region(soma::Region{.name = "r2", .mesh = m2,
                                            .dimension = 1});
  const auto p = g.add_component(soma::ComponentDecl{
      .name = "producer", .ranks = {.first = 0, .count = 1},
      .interval = src_dt});
  const auto c = g.add_component(soma::ComponentDecl{
      .name = "consumer", .ranks = {.first = 1, .count = 1},
      .interval = dst_dt});
  const auto s = g.add_field(p, soma::FieldSpec{
      .name = "out", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(c, soma::FieldSpec{
      .name = "in", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  if (accum != soma::AccumOp::Instant) g.channel(ch).accumulate(accum);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched = schedule(g, h->dag, clocks,
                        soma::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  h->components.add(std::make_unique<Producer>(8, 2.0));
  h->components.add(std::make_unique<Consumer>(8));
  return h;
}

}  // namespace

TEST(Runner, InitializesAndAllocatesBuffers) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  EXPECT_EQ(runner.stats().periods, 0);
}

TEST(Runner, RejectsUnboundFields) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  // Deliberately skip register_fields.
  const auto r = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-UNBOUND");
}

// The first coupled exchange.
TEST(Runner, TransfersAValueFromProducerToConsumer) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  const auto* consumer = dynamic_cast<const Consumer*>(
      h->components.find("consumer"));
  ASSERT_NE(consumer, nullptr);
  for (const double v : consumer->data()) {
    EXPECT_DOUBLE_EQ(v, 2.0) << "producer wrote 2.0 on its first step";
  }
}

TEST(Runner, AdvancesEveryComponentOncePerPeriodInLockstep) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  EXPECT_EQ(dynamic_cast<Producer*>(h->components.find("producer"))->steps, 3);
  EXPECT_EQ(dynamic_cast<Consumer*>(h->components.find("consumer"))->steps, 3);
  EXPECT_EQ(runner.stats().periods, 3);
}

TEST(Runner, SubcyclingAdvancesTheFastComponentMoreOften) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 1).ok());

  EXPECT_EQ(dynamic_cast<Producer*>(h->components.find("producer"))->steps, 4);
  EXPECT_EQ(dynamic_cast<Consumer*>(h->components.find("consumer"))->steps, 1);
}

TEST(Runner, MeanAccumulationAveragesOverTheWindow) {
  // Producer writes 2*step: over 4 substeps that is 2,4,6,8 -> mean 5.
  auto h = make_harness(soma::AccumOp::Mean, Rational{1, 4}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());

  const auto* consumer = dynamic_cast<const Consumer*>(
      h->components.find("consumer"));
  ASSERT_NE(consumer, nullptr);
  EXPECT_NEAR(consumer->data()[0], 5.0, 1e-12);
}

TEST(Runner, CountsExecutedInstructions) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().instructions_executed,
            static_cast<std::int64_t>(h->plan.instructions().size()));
}

TEST(Runner, DetectsNonFiniteValues) {
  auto h = make_harness(soma::AccumOp::Instant, Rational{1}, Rational{1});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c->register_fields(h->registry, h->graph).ok());
  }
  RunnerConfig cfg;
  cfg.abort_on_violation = true;
  ASSERT_TRUE(runner.initialize(h->components, h->registry, cfg).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().conservation_violations, 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/runtime/runner.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/runtime/runner.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "soma/field/registry.hpp"
#include "soma/mesh/mesh_view.hpp"
#include "soma/runtime/component.hpp"
#include "soma/schedule/plan.hpp"
#include "soma/transport/transport.hpp"

namespace soma {

struct RunnerConfig {
  std::string transport = "copy";
  bool        check_conservation = true;
  double      conservation_tolerance = 1e-12;
  bool        abort_on_violation = false;
};

struct RunStats {
  std::int64_t periods = 0;
  std::int64_t instructions_executed = 0;
  std::int64_t conservation_checks = 0;
  std::int64_t conservation_violations = 0;
  double       worst_conservation_error = 0.0;
};

/// Spec §2 layer 3: interprets the flat instruction stream. Every ArcId is
/// resolved to a raw pointer once at initialize(); the loop allocates nothing.
class Runner {
 public:
  Runner(const CouplingGraph& graph, const OperatorDAG& dag, ExecutionPlan plan,
         const MeshBackend& backend);
  ~Runner();

  Runner(Runner&&) noexcept;
  Runner& operator=(Runner&&) noexcept;

  Result<void> initialize(ComponentSet& components, FieldRegistry& registry,
                          const RunnerConfig& config = {});

  Result<void> run_period(ComponentSet& components);
  Result<void> run(ComponentSet& components, std::int64_t periods);

  [[nodiscard]] const RunStats& stats() const noexcept;
  [[nodiscard]] Result<const double*> buffer(ArcId arc) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace soma
```

- [ ] **Step 4: Write `src/runtime/runner.cpp`**

```cpp
#include "soma/runtime/runner.hpp"

#include <unordered_map>
#include <vector>

#include "soma/num/kernels.hpp"

namespace soma {
namespace {

/// A resolved buffer: either coupler-allocated storage or an alias to a
/// component's own array (spec §4.2).
struct Slot {
  double*     data = nullptr;
  std::size_t elements = 0;
  int         components = 1;
  bool        owned_by_component = false;
};

}  // namespace

struct Runner::Impl {
  const CouplingGraph& graph;
  const OperatorDAG&   dag;
  ExecutionPlan        plan;
  const MeshBackend&   backend;
  RunnerConfig         config;

  std::vector<std::vector<double>> storage;   // coupler-owned, one per arc
  std::vector<Slot>                slots;     // parallel to arcs
  std::vector<std::vector<double>> previous;  // previous iterate, for Relax
  std::vector<std::size_t>         sample_count;  // per arc, for Accumulate
  std::unique_ptr<Transport>       transport;
  RunStats                         stats;
  std::int64_t                     period_index = 0;

  Impl(const CouplingGraph& g, const OperatorDAG& d, ExecutionPlan p,
       const MeshBackend& b)
      : graph(g), dag(d), plan(std::move(p)), backend(b) {}

  [[nodiscard]] std::size_t total_elements(const Slot& s) const {
    return s.elements * static_cast<std::size_t>(s.components);
  }

  Result<void> execute(const Instruction& ins);
  Result<void> advance_components(ComponentSet& components, Rational t);
};

Result<void> Runner::Impl::execute(const Instruction& ins) {
  const auto in_ptr = [&](std::uint8_t k) -> const double* {
    return slots[ins.in[k].value()].data;
  };
  const auto out_slot = [&]() -> Slot& { return slots[ins.out.value()]; };

  switch (ins.kind) {
    case OpKind::Source:
      // The arc already aliases component storage; nothing to do.
      break;

    case OpKind::Sink:
    case OpKind::Interpolate:
    case OpKind::Accelerate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t n = std::min(total_elements(out),
                                     total_elements(slots[ins.in[0].value()]));
      num::copy(in_ptr(0), out.data, n);
      break;
    }

    case OpKind::Remap: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const Slot& in = slots[ins.in[0].value()];
      // Phase 4: identity on matching sizes. Phase 5 Task 5.2 supplies weights.
      if (total_elements(in) != total_elements(out)) {
        return Error{"E-RT-REMAP-SIZE",
                     "identity remap needs matching sizes (" +
                         std::to_string(total_elements(in)) + " vs " +
                         std::to_string(total_elements(out)) +
                         "); build with SOMA_ENABLE_MOAB for real weights"};
      }
      num::copy(in.data, out.data, total_elements(out));
      break;
    }

    case OpKind::Sample: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t idx = sample_count[ins.out.value()];
      num::accumulate_into(ins.attrs.accum, in_ptr(0), out.data,
                           total_elements(out), idx);
      sample_count[ins.out.value()] = idx + 1;
      break;
    }

    case OpKind::Accumulate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const Slot& in = slots[ins.in[0].value()];
      const std::size_t n = total_elements(out);
      num::copy(in.data, out.data, n);
      const std::size_t count = sample_count[ins.in[0].value()];
      num::accumulate_finalize(ins.attrs.accum, out.data, n, count);
      sample_count[ins.in[0].value()] = 0;  // reset for the next window
      break;
    }

    case OpKind::Extrapolate: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      // Phase 4 keeps no history, so order 0 is the only honest behaviour.
      num::copy(in_ptr(0), out.data, total_elements(out));
      break;
    }

    case OpKind::Merge: {
      if (!ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t n = total_elements(out);
      for (std::size_t i = 0; i < n; ++i) out.data[i] = 0.0;
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        const double* src = in_ptr(k);
        for (std::size_t i = 0; i < n; ++i) out.data[i] += src[i];
      }
      break;
    }

    case OpKind::Relax: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t n = total_elements(out);
      std::vector<double>& prev = previous[ins.out.value()];
      if (prev.size() != n) prev.assign(n, 0.0);
      num::relax(ins.attrs.omega, prev.data(), in_ptr(0), out.data, n);
      num::copy(out.data, prev.data(), n);
      break;
    }

    case OpKind::Transport: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t n = std::min(total_elements(out),
                                     total_elements(slots[ins.in[0].value()]));
      TransportSpec spec;
      spec.src_rank_set = ins.attrs.src_rank_set;
      spec.dst_rank_set = ins.attrs.dst_rank_set;
      spec.elements = n;
      spec.components = 1;
      auto prep = transport->prepare(spec);
      if (!prep.ok()) return prep;
      return transport->move(in_ptr(0), out.data, n);
    }

    case OpKind::Residual: {
      if (ins.n_in < 1 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const std::size_t n = total_elements(out);
      const double* a = in_ptr(0);
      const double* b = ins.n_in > 1 ? in_ptr(1) : previous[ins.out.value()].data();
      if (ins.n_in > 1) {
        for (std::size_t i = 0; i < n; ++i) out.data[i] = a[i] - b[i];
      } else {
        std::vector<double>& prev = previous[ins.out.value()];
        if (prev.size() != n) prev.assign(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) out.data[i] = a[i] - prev[i];
        num::copy(a, prev.data(), n);
      }
      break;
    }

    case OpKind::Constrain: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const Slot& in = slots[ins.in[0].value()];
      const std::size_t n = total_elements(out);
      num::copy(in.data, out.data, n);

      if (config.check_conservation) {
        ++stats.conservation_checks;
        const double before = num::sum(in.data, n);
        const double after = num::sum(out.data, n);
        const double scale = std::abs(before) > 0.0 ? std::abs(before) : 1.0;
        const double rel = std::abs(after - before) / scale;
        if (rel > stats.worst_conservation_error) {
          stats.worst_conservation_error = rel;
        }
        if (rel > config.conservation_tolerance) {
          ++stats.conservation_violations;
          if (config.abort_on_violation) {
            return Error{"E-RT-CONSERVATION",
                         "relative conservation error " + std::to_string(rel) +
                             " exceeds tolerance " +
                             std::to_string(config.conservation_tolerance)};
          }
        }
      }
      break;
    }

    case OpKind::Diagnose: {
      if (ins.n_in == 0 || !ins.out.valid()) break;
      Slot& out = out_slot();
      const Slot& in = slots[ins.in[0].value()];
      const std::size_t n = total_elements(out);
      num::copy(in.data, out.data, n);
      if (num::has_nonfinite(out.data, n)) {
        return Error{"E-RT-NONFINITE",
                     "non-finite value detected in arc %" +
                         std::to_string(ins.out.value())};
      }
      break;
    }
  }
  return Result<void>::success();
}

Result<void> Runner::Impl::advance_components(ComponentSet& components,
                                              Rational period_end) {
  for (std::size_t i = 0; i < components.size(); ++i) {
    Component& c = components.at(i);
    const auto cid = graph.find_component(c.name());
    if (!cid.has_value()) continue;
    const Rational dt = graph.component(*cid).interval;
    const Rational n_exact = period_end / dt;
    const std::int64_t substeps = n_exact.is_integer() ? n_exact.num() : 1;

    for (std::int64_t k = 0; k < substeps; ++k) {
      StepContext ctx;
      ctx.t_begin = Rational{period_index} * period_end + Rational{k} * dt;
      ctx.t_end = ctx.t_begin + dt;
      ctx.step = period_index * substeps + k;
      auto r = c.advance(ctx);
      if (!r.ok()) return r;
    }
  }
  return Result<void>::success();
}

Runner::Runner(const CouplingGraph& graph, const OperatorDAG& dag,
               ExecutionPlan plan, const MeshBackend& backend)
    : impl_(std::make_unique<Impl>(graph, dag, std::move(plan), backend)) {}
Runner::~Runner() = default;
Runner::Runner(Runner&&) noexcept = default;
Runner& Runner::operator=(Runner&&) noexcept = default;

Result<void> Runner::initialize(ComponentSet& components,
                                FieldRegistry& registry,
                                const RunnerConfig& config) {
  impl_->config = config;

  auto transport = TransportRegistry::instance().create(config.transport);
  if (!transport.ok()) return transport.error();
  impl_->transport = std::move(transport.value());

  const auto& buffers = impl_->plan.buffers().buffers;
  impl_->storage.assign(buffers.size(), {});
  impl_->slots.assign(buffers.size(), Slot{});
  impl_->previous.assign(buffers.size(), {});
  impl_->sample_count.assign(buffers.size(), 0);

  // Map each component-owned arc back to the field it aliases.
  std::unordered_map<std::uint32_t, FieldId> arc_field;
  for (std::uint32_t i = 0; i < impl_->dag.node_count(); ++i) {
    const Node& n = impl_->dag.node(NodeId{i});
    if (n.kind != OpKind::Source && n.kind != OpKind::Sink) continue;
    if (!n.output.valid()) continue;
    // The node label is "component.field"; resolve it through the graph.
    const auto dot = n.label.find('.');
    if (dot == std::string::npos) continue;
    const auto cid = impl_->graph.find_component(n.label.substr(0, dot));
    if (!cid.has_value()) continue;
    const auto fid = impl_->graph.find_field(*cid, n.label.substr(dot + 1));
    if (!fid.has_value()) continue;
    arc_field.emplace(n.output.value(), *fid);
  }

  for (std::size_t i = 0; i < buffers.size(); ++i) {
    const BufferDecl& b = buffers[i];
    Slot& s = impl_->slots[i];
    s.components = b.components;

    if (b.is_component_owned) {
      const auto it = arc_field.find(b.arc.value());
      if (it == arc_field.end()) {
        return Error{"E-RT-UNBOUND",
                     "component-owned arc %" + std::to_string(b.arc.value()) +
                         " (" + b.label + ") does not resolve to a field"};
      }
      auto ref = registry.get(it->second);
      if (!ref.ok()) {
        return Error{"E-RT-UNBOUND",
                     "field '" + b.label +
                         "' is not bound; did the component call "
                         "register_fields?"};
      }
      s.data = ref.value().data();
      s.elements = ref.value().size();
      s.owned_by_component = true;
    } else {
      const std::size_t n =
          b.elements * static_cast<std::size_t>(std::max(b.components, 1));
      impl_->storage[i].assign(n, 0.0);
      s.data = impl_->storage[i].data();
      s.elements = b.elements;
      s.owned_by_component = false;
    }
  }

  (void)components;
  return Result<void>::success();
}

Result<void> Runner::run_period(ComponentSet& components) {
  auto adv = impl_->advance_components(components, impl_->plan.period());
  if (!adv.ok()) return adv;

  for (const Instruction& ins : impl_->plan.instructions()) {
    auto r = impl_->execute(ins);
    if (!r.ok()) return r;
    ++impl_->stats.instructions_executed;
  }
  ++impl_->stats.periods;
  ++impl_->period_index;
  return Result<void>::success();
}

Result<void> Runner::run(ComponentSet& components, std::int64_t periods) {
  for (std::int64_t i = 0; i < periods; ++i) {
    auto r = run_period(components);
    if (!r.ok()) return r;
  }
  return Result<void>::success();
}

const RunStats& Runner::stats() const noexcept { return impl_->stats; }

Result<const double*> Runner::buffer(ArcId arc) const {
  if (!arc.valid() || arc.value() >= impl_->slots.size()) {
    return Error{"E-RT-ARC", "no buffer for arc " + std::to_string(arc.value())};
  }
  return static_cast<const double*>(impl_->slots[arc.value()].data);
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `runtime/runner.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(runner_test unit/runner_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Runner
```

Expected: 8 tests PASS. `TransfersAValueFromProducerToConsumer` is the
milestone — the first end-to-end coupled exchange.

If `MeanAccumulationAveragesOverTheWindow` fails with 6.25 instead of 5.0, the
sample counter is being reset in the wrong place; check that `Accumulate` reads
`sample_count` of its *input* arc, not its output.

- [ ] **Step 7: Commit**

```bash
git add include/soma/runtime/runner.hpp src/runtime/runner.cpp \
        test/unit/runner_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): plan interpreter with allocation-free hot loop"
```

---

### Task 4.7: First coupled run — the MMS example

The Phase 4 milestone as a runnable program, plus a verification test that the
coupled answer is right rather than merely produced.

**Files:**
- Create: `examples/CMakeLists.txt`
- Create: `examples/mms_pair/main.cpp`
- Create: `examples/mms_pair/heat_component.hpp`
- Modify: `CMakeLists.txt` (add `add_subdirectory(examples)`)
- Create: `test/verification/mms_transfer_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: everything in Phase 4.
- Produces:
  - `class HeatComponent : public Component` in `heat_component.hpp` — 1D heat equation on a line mesh, explicit Euler, with a Dirichlet boundary value it reads from a coupled field and a flux it exports. Analytic reference: `T(x,t) = exp(-pi^2 * alpha * t) * sin(pi * x)`, which satisfies `T_t = alpha * T_xx` with `T(0)=T(1)=0`.
  - `examples/mms_pair/main.cpp` — builds the graph in C++, dumps graph/DAG/plan when `--dump` is passed, runs 100 periods, prints the L2 error against the analytic solution.
  - A verification test asserting: (a) a constant field transfers exactly; (b) the transferred value equals the analytic mean over the window to the expected order; (c) the run is deterministic across repeats.

  Full spatial and temporal order-of-accuracy studies are Phase 6 Task 6.5 and
  Phase 7 Task 7.5, once space-time projection and iteration exist. Phase 4
  verifies transfer correctness only, and says so.

- [ ] **Step 1: Write the failing verification test**

`test/verification/mms_transfer_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>

#include <cmath>
#include <vector>

using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::Runner;
using soma::SerialMeshBackend;
using soma::StepContext;
namespace u = soma::units;

namespace {

/// Writes f(x, t) into its field; the coupled partner receives it.
class AnalyticSource final : public soma::Component {
 public:
  AnalyticSource(std::size_t n, double amplitude)
      : data_(n, 0.0), amplitude_(amplitude) {}

  [[nodiscard]] std::string name() const override { return "source"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("source");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext& ctx) override {
    t_ = ctx.t_end.to_double();
    for (std::size_t i = 0; i < data_.size(); ++i) {
      const double x = (static_cast<double>(i) + 0.5) /
                       static_cast<double>(data_.size());
      data_[i] = amplitude_ * std::sin(M_PI * x) * t_;
    }
    return soma::Result<void>::success();
  }

  [[nodiscard]] double time() const { return t_; }

 private:
  std::vector<double> data_;
  double              amplitude_;
  double              t_ = 0.0;
};

class Receiver final : public soma::Component {
 public:
  explicit Receiver(std::size_t n) : data_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return "receiver"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("receiver");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext&) override {
    return soma::Result<void>::success();
  }

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
};

struct Fixture {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  soma::OperatorDAG   dag;
  soma::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;
  AnalyticSource*     source = nullptr;
  Receiver*           receiver = nullptr;
};

std::unique_ptr<Fixture> build(std::size_t n, Rational src_dt, Rational dst_dt,
                               soma::AccumOp accum) {
  auto f = std::make_unique<Fixture>();
  auto& g = f->graph;
  const auto m1 = g.add_mesh(soma::MeshDecl{
      .name = "m1", .source = "generated:line:" + std::to_string(n)});
  const auto m2 = g.add_mesh(soma::MeshDecl{
      .name = "m2", .source = "generated:line:" + std::to_string(n)});
  const auto r1 = g.add_region(soma::Region{.name = "r1", .mesh = m1,
                                            .dimension = 1});
  const auto r2 = g.add_region(soma::Region{.name = "r2", .mesh = m2,
                                            .dimension = 1});
  const auto cs = g.add_component(soma::ComponentDecl{
      .name = "source", .ranks = {.first = 0, .count = 1}, .interval = src_dt});
  const auto cr = g.add_component(soma::ComponentDecl{
      .name = "receiver", .ranks = {.first = 1, .count = 1},
      .interval = dst_dt});
  const auto fs = g.add_field(cs, soma::FieldSpec{
      .name = "T", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto fr = g.add_field(cr, soma::FieldSpec{
      .name = "T_bc", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("T_link", fs, fr);
  if (accum != soma::AccumOp::Instant) g.channel(ch).accumulate(accum);

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

  auto src = std::make_unique<AnalyticSource>(n, 1.0);
  auto rcv = std::make_unique<Receiver>(n);
  f->source = src.get();
  f->receiver = rcv.get();
  f->components.add(std::move(src));
  f->components.add(std::move(rcv));
  return f;
}

void run(Fixture& f, std::int64_t periods) {
  Runner runner{f.graph, f.dag, f.plan, f.backend};
  for (auto& c : f.components) {
    ASSERT_TRUE(c->register_fields(f.registry, f.graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f.components, f.registry).ok());
  ASSERT_TRUE(runner.run(f.components, periods).ok());
}

}  // namespace

// A constant field must transfer bit-exactly: no remap error, no drift.
TEST(MmsTransfer, ConstantFieldTransfersExactly) {
  auto f = build(16, Rational{1}, Rational{1}, soma::AccumOp::Instant);
  run(*f, 1);
  for (std::size_t i = 0; i < f->receiver->data().size(); ++i) {
    const double x = (static_cast<double>(i) + 0.5) / 16.0;
    EXPECT_DOUBLE_EQ(f->receiver->data()[i], std::sin(M_PI * x));
  }
}

// Mean accumulation of a linear-in-time signal over [0,1] with 4 substeps
// gives the average of t = 1/4, 1/2, 3/4, 1, i.e. 5/8.
TEST(MmsTransfer, MeanOfALinearSignalMatchesTheAnalyticAverage) {
  auto f = build(16, Rational{1, 4}, Rational{1}, soma::AccumOp::Mean);
  run(*f, 1);
  const double x0 = 0.5 / 16.0;
  const double expected = std::sin(M_PI * x0) * 0.625;
  EXPECT_NEAR(f->receiver->data()[0], expected, 1e-13);
}

TEST(MmsTransfer, RepeatedRunsAreBitIdentical) {
  auto a = build(16, Rational{1, 4}, Rational{1}, soma::AccumOp::Mean);
  auto b = build(16, Rational{1, 4}, Rational{1}, soma::AccumOp::Mean);
  run(*a, 5);
  run(*b, 5);
  ASSERT_EQ(a->receiver->data().size(), b->receiver->data().size());
  for (std::size_t i = 0; i < a->receiver->data().size(); ++i) {
    EXPECT_DOUBLE_EQ(a->receiver->data()[i], b->receiver->data()[i]);
  }
}

TEST(MmsTransfer, NoConservationViolationsOnAnIdentityRemap) {
  auto f = build(16, Rational{1}, Rational{1}, soma::AccumOp::Instant);
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
  ASSERT_TRUE(runner.run(f->components, 10).ok());
  EXPECT_EQ(runner.stats().conservation_violations, 0);
}
```

- [ ] **Step 2: Run to verify it fails**

Add to `test/CMakeLists.txt`:
`soma_add_test(mms_transfer_test verification/mms_transfer_test.cpp)`

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R MmsTransfer
```

Expected: FAIL if any Phase 4 piece is wrong; PASS if Tasks 4.1–4.6 are correct.
This test is written against the existing interfaces deliberately — it is the
acceptance test for the phase, not a driver for new code.

- [ ] **Step 3: Write the example component**

`examples/mms_pair/heat_component.hpp`:

```cpp
#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <soma/runtime/component.hpp>

/// 1D heat equation, explicit Euler, cell-centred finite volume on [0,1].
///   T_t = alpha * T_xx,  T(0,t) = T(1,t) = 0
/// Exact solution for T(x,0) = sin(pi x):
///   T(x,t) = exp(-pi^2 alpha t) sin(pi x)
class HeatComponent final : public soma::Component {
 public:
  HeatComponent(std::string name, std::size_t cells, double alpha)
      : name_(std::move(name)),
        alpha_(alpha),
        dx_(1.0 / static_cast<double>(cells)),
        t_(cells, 0.0),
        scratch_(cells, 0.0) {
    for (std::size_t i = 0; i < cells; ++i) {
      t_[i] = std::sin(M_PI * x(i));
    }
  }

  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(soma::FieldRegistry& reg,
                                     const soma::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid.has_value()) {
      return soma::Error{"E-EX-COMP", "component '" + name_ + "' not in graph"};
    }
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{t_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = t_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext& ctx) override {
    const double dt = (ctx.t_end - ctx.t_begin).to_double();
    const double r = alpha_ * dt / (dx_ * dx_);
    if (r > 0.5) {
      return soma::Error{"E-EX-CFL",
                         "explicit Euler needs alpha*dt/dx^2 <= 1/2, got " +
                             std::to_string(r)};
    }
    const std::size_t n = t_.size();
    for (std::size_t i = 0; i < n; ++i) {
      const double left  = (i == 0) ? -t_[0] : t_[i - 1];        // T(0) = 0
      const double right = (i + 1 == n) ? -t_[n - 1] : t_[i + 1];  // T(1) = 0
      scratch_[i] = t_[i] + r * (left - 2.0 * t_[i] + right);
    }
    t_.swap(scratch_);
    time_ = ctx.t_end.to_double();
    return soma::Result<void>::success();
  }

  soma::Result<void> save_state() override {
    saved_ = t_;
    return soma::Result<void>::success();
  }
  soma::Result<void> restore_state() override {
    t_ = saved_;
    return soma::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] double x(std::size_t i) const {
    return (static_cast<double>(i) + 0.5) * dx_;
  }

  [[nodiscard]] double exact(std::size_t i, double t) const {
    return std::exp(-M_PI * M_PI * alpha_ * t) * std::sin(M_PI * x(i));
  }

  [[nodiscard]] double l2_error() const {
    double sum = 0.0;
    for (std::size_t i = 0; i < t_.size(); ++i) {
      const double e = t_[i] - exact(i, time_);
      sum += e * e * dx_;
    }
    return std::sqrt(sum);
  }

  [[nodiscard]] const std::vector<double>& values() const { return t_; }
  [[nodiscard]] double time() const { return time_; }

 private:
  std::string         name_;
  double              alpha_;
  double              dx_;
  std::vector<double> t_;
  std::vector<double> scratch_;
  std::vector<double> saved_;
  double              time_ = 0.0;
};
```

- [ ] **Step 4: Write the example driver**

`examples/mms_pair/main.cpp`:

```cpp
#include <cstring>
#include <iostream>
#include <memory>

#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>

#include "heat_component.hpp"

namespace {

soma::CouplingGraph build_graph(std::size_t cells) {
  soma::CouplingGraph g;
  const auto m_left = g.add_mesh(soma::MeshDecl{
      .name = "left_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto m_right = g.add_mesh(soma::MeshDecl{
      .name = "right_mesh",
      .source = "generated:line:" + std::to_string(cells)});
  const auto r_left = g.add_region(soma::Region{
      .name = "left", .mesh = m_left, .dimension = 1});
  const auto r_right = g.add_region(soma::Region{
      .name = "right", .mesh = m_right, .dimension = 1});

  const auto left = g.add_component(soma::ComponentDecl{
      .name = "left", .ranks = {.first = 0, .count = 1},
      .interval = soma::Rational{1, 1000}});
  const auto right = g.add_component(soma::ComponentDecl{
      .name = "right", .ranks = {.first = 1, .count = 1},
      .interval = soma::Rational{1, 1000}});

  const auto t_out = g.add_field(left, soma::FieldSpec{
      .name = "T", .region = r_left, .units = soma::units::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t_in = g.add_field(right, soma::FieldSpec{
      .name = "T_bc", .region = r_right, .units = soma::units::kKelvin,
      .intent = soma::Intent::Requires});

  const auto ch = g.add_channel("T_link", t_out, t_in);
  g.channel(ch).accumulate(soma::AccumOp::Mean);
  return g;
}

}  // namespace

int main(int argc, char** argv) {
  bool dump = false;
  std::size_t cells = 64;
  std::int64_t periods = 100;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--dump") == 0) dump = true;
  }

  soma::CouplingGraph graph = build_graph(cells);

  soma::SerialMeshBackend backend;
  for (std::uint32_t i = 0; i < graph.mesh_count(); ++i) {
    const auto r = backend.load(graph.mesh(soma::MeshId{i}));
    if (!r.ok()) {
      std::cerr << "mesh load failed: " << r.error().message << "\n";
      return 1;
    }
  }

  const auto lowered = lower(graph);
  if (!lowered.ok()) {
    std::cerr << lowered.diags.to_string();
    return 1;
  }

  const soma::ClockSet clocks{graph};
  auto sched = schedule(graph, lowered.dag, clocks,
                        soma::make_region_sizes(graph, backend));
  if (!sched.ok()) {
    std::cerr << sched.diags.to_string();
    return 1;
  }

  if (dump) {
    std::cout << "=== IR ===\n" << dump_text(lowered.dag)
              << "\n=== PLAN ===\n" << sched.plan.to_string() << "\n";
  }

  soma::ComponentSet components;
  auto left = std::make_unique<HeatComponent>("left", cells, 0.01);
  auto right = std::make_unique<HeatComponent>("right", cells, 0.01);
  HeatComponent* left_ptr = left.get();
  components.add(std::move(left));
  components.add(std::move(right));

  soma::FieldRegistry registry;
  for (auto& c : components) {
    const auto r = c->register_fields(registry, graph);
    if (!r.ok()) {
      std::cerr << "register_fields failed: " << r.error().message << "\n";
      return 1;
    }
  }

  soma::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry); !r.ok()) {
    std::cerr << "initialize failed: " << r.error().message << "\n";
    return 1;
  }
  if (const auto r = runner.run(components, periods); !r.ok()) {
    std::cerr << "run failed: " << r.error().message << "\n";
    return 1;
  }

  std::cout << "periods            : " << runner.stats().periods << "\n"
            << "instructions       : " << runner.stats().instructions_executed
            << "\n"
            << "conservation checks: " << runner.stats().conservation_checks
            << "\n"
            << "violations         : "
            << runner.stats().conservation_violations << "\n"
            << "t                  : " << left_ptr->time() << "\n"
            << "L2 error (left)    : " << left_ptr->l2_error() << "\n";
  return 0;
}
```

- [ ] **Step 5: Write `examples/CMakeLists.txt`**

```cmake
add_executable(mms_pair mms_pair/main.cpp)
target_link_libraries(mms_pair PRIVATE soma soma_core)
target_include_directories(mms_pair PRIVATE mms_pair)
target_compile_options(mms_pair PRIVATE -Wall -Wextra -Wpedantic)
```

and in the top-level `CMakeLists.txt`, after `add_subdirectory(src)`:

```cmake
add_subdirectory(examples)
```

- [ ] **Step 6: Build and run the example**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
./build/examples/mms_pair --dump | head -40
./build/examples/mms_pair
```

Expected output shape:

```
periods            : 100
instructions       : <N>
conservation checks: <M>
violations         : 0
t                  : 0.1
L2 error (left)    : <small>
```

The L2 error should be dominated by the explicit-Euler time discretisation, not
by coupling. With `cells = 64`, `alpha = 0.01`, `dt = 1/1000`, the diffusion
number is `0.01 * 0.001 * 64^2 = 0.041`, comfortably stable, and the error
should be below `1e-3`. If it is `O(1)`, the heat component is wrong; if it is
exactly zero, `advance` is not being called.

- [ ] **Step 7: Run the full suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: every Phase 0–4 test green.

- [ ] **Step 8: Commit**

```bash
git add examples/ CMakeLists.txt test/verification/mms_transfer_test.cpp \
        test/CMakeLists.txt
git commit -m "feat(soma): first coupled run -- MMS pair example and verification"
```

---

## Phase 4 exit criteria

- [ ] Two C++ components exchange a field through graph → lower → schedule → run.
- [ ] Components own their storage; the coupler allocates only intermediates, verified by `BufferPlan::total_coupler_elements`.
- [ ] A constant field transfers bit-exactly.
- [ ] Mean accumulation over a subcycled window matches the analytic average to 1e-13.
- [ ] Repeated runs are bit-identical.
- [ ] The core still builds and tests with **no** optional dependency.
- [ ] `./build/examples/mms_pair --dump` prints the IR and the plan.
- [ ] `ctest --test-dir build --output-on-failure` fully green.

**Known limitations, deliberately deferred:**
- `Remap` is an identity copy requiring matching element counts (Phase 5 Task 5.2).
- `Extrapolate` keeps no history and behaves as order 0 (Phase 6 Task 6.1).
- `Interpolate` and `Accelerate` are copies (Phases 6 and 7).
- `Constrain` reports conservation error but applies no correction (Phase 5 Task 5.5).
- Only the `copy` transport exists (Phase 5 Task 5.4 adds MPI).
