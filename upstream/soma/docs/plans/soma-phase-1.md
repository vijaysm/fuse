# SOMA Phase 1 — CouplingGraph, validation, YAML round-trip

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §2 (layer 1), §3.2 (narrow waist), §4.1 (regions), §4.2 (fields)
**Milestone:** a coupled setup can be authored in C++, validated, serialised to a flat descriptor, written as YAML, read back, and compared equal.

Phase 1 builds tier B: header-only authoring with STL only. Nothing here executes; everything here is data.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/graph/region.hpp` | `Region`, `MeshDecl` |
| `include/soma/graph/field_spec.hpp` | `FieldSpec` |
| `include/soma/graph/component.hpp` | `ComponentDecl` |
| `include/soma/graph/channel.hpp` | `Channel`, `ChannelBuilder`, `Window` |
| `include/soma/graph/coupling_graph.hpp` | `CouplingGraph` — owns all four index spaces |
| `include/soma/graph/validate.hpp` | validation rules over a graph |
| `include/soma/graph/descriptor.hpp` | flat POD `soma_graph_v1` + convert both ways |
| `src/graph/yaml_io.cpp` | YAML reader/writer (compiled; yaml-cpp is not in a public header) |
| `include/soma/graph/yaml_io.hpp` | YAML entry points, no yaml-cpp types |

---

### Task 1.1: `Region`, `MeshDecl`, and `FieldSpec`

Spec §4.1: the connectable unit is a `Region` = (mesh, entity set, dimension), not a mesh.

**Files:**
- Create: `include/soma/graph/region.hpp`
- Create: `include/soma/graph/field_spec.hpp`
- Test: `test/unit/region_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `MeshId`, `RegionId`, `Discretization`, `Conservation`, `Intent` (0.6); `Units` (0.4).
- Produces:
  - `struct MeshDecl { std::string name; std::string source; }` — `source` is a file path or `"generated:<spec>"`; the mesh backend interprets it, the graph does not.
  - `struct Region { std::string name; MeshId mesh; std::string entity_set; int dimension = -1; }` — `entity_set` empty means the whole mesh; `dimension` is the topological dimension of the coupled entities (2 for a surface, 3 for a volume).
  - `struct FieldSpec { std::string name; RegionId region; Discretization disc = Discretization::FV; int order = 0; int components = 1; Units units{}; Conservation conservation = Conservation::None; Intent intent = Intent::Provides; }` — `order` is the polynomial order for GLL discretizations, ignored for FV.
  - `bool same_shape(const FieldSpec&, const FieldSpec&)` — components and region dimension agree.

- [ ] **Step 1: Write the failing test**

`test/unit/region_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/field_spec.hpp>
#include <soma/graph/region.hpp>

namespace u = soma::units;
using soma::Conservation;
using soma::Discretization;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshId;
using soma::Region;
using soma::RegionId;

TEST(Region, DefaultsToWholeMesh) {
  const Region r{.name = "fluid_all", .mesh = MeshId{0}};
  EXPECT_TRUE(r.entity_set.empty());
  EXPECT_EQ(r.dimension, -1);
}

TEST(Region, NamedEntitySetIsASurface) {
  const Region r{.name = "wall", .mesh = MeshId{0},
                 .entity_set = "sideset_3", .dimension = 2};
  EXPECT_EQ(r.entity_set, "sideset_3");
  EXPECT_EQ(r.dimension, 2);
}

TEST(FieldSpec, DefaultsAreFvScalarProvidesNone) {
  const FieldSpec f{.name = "T", .region = RegionId{0}};
  EXPECT_EQ(f.disc, Discretization::FV);
  EXPECT_EQ(f.components, 1);
  EXPECT_EQ(f.conservation, Conservation::None);
  EXPECT_EQ(f.intent, Intent::Provides);
  EXPECT_TRUE(f.units.is_dimensionless());
}

TEST(FieldSpec, CarriesUnitsAndConservation) {
  const FieldSpec q{.name = "heat_flux",
                    .region = RegionId{1},
                    .units = u::kHeatFluxDensity,
                    .conservation = Conservation::Extensive,
                    .intent = Intent::Provides};
  EXPECT_EQ(q.units, u::kHeatFluxDensity);
  EXPECT_EQ(q.conservation, Conservation::Extensive);
}

TEST(FieldSpec, SameShapeComparesComponents) {
  const FieldSpec a{.name = "v", .region = RegionId{0}, .components = 3};
  const FieldSpec b{.name = "w", .region = RegionId{1}, .components = 3};
  const FieldSpec c{.name = "s", .region = RegionId{1}, .components = 1};
  EXPECT_TRUE(same_shape(a, b));
  EXPECT_FALSE(same_shape(a, c));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/region.hpp` file not found.

- [ ] **Step 3: Write `region.hpp`**

```cpp
#pragma once

#include <string>

#include "soma/base/ids.hpp"

namespace soma {

/// A mesh the coupler is told about. `source` is opaque to the graph layer;
/// the mesh backend (Phase 4 serial, Phase 5 MOAB) interprets it.
struct MeshDecl {
  std::string name;
  std::string source;  // file path, or "generated:<spec>"
};

/// Spec §4.1: the connectable unit. Coupling happens on interfaces, not whole
/// meshes, so channels connect Regions.
struct Region {
  std::string name;
  MeshId      mesh;
  std::string entity_set;   // empty = whole mesh
  int         dimension = -1;  // topological dim of coupled entities; -1 = mesh dim
};

}  // namespace soma
```

- [ ] **Step 4: Write `field_spec.hpp`**

```cpp
#pragma once

#include <string>

#include "soma/base/ids.hpp"
#include "soma/base/units.hpp"

namespace soma {

/// Spec §4.2. disc+order select the remap operator; units are checked at
/// lowering; conservation drives which verification runs.
struct FieldSpec {
  std::string    name;
  RegionId       region;
  Discretization disc = Discretization::FV;
  int            order = 0;       // polynomial order for GLL; ignored for FV
  int            components = 1;  // 1 scalar, 3 vector, 9 tensor
  Units          units{};
  Conservation   conservation = Conservation::None;
  Intent         intent = Intent::Provides;
};

[[nodiscard]] inline bool same_shape(const FieldSpec& a,
                                     const FieldSpec& b) noexcept {
  return a.components == b.components;
}

}  // namespace soma
```

- [ ] **Step 5: Register and run**

Add `soma_add_test(region_test unit/region_test.cpp)`.

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R "Region|FieldSpec"
```

Expected: 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add include/soma/graph/region.hpp include/soma/graph/field_spec.hpp \
        test/unit/region_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): Region, MeshDecl, FieldSpec"
```

---

### Task 1.2: `ComponentDecl` and `Window`

**Files:**
- Create: `include/soma/graph/component.hpp`
- Test: `test/unit/component_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `ComponentId`, `FieldId`, `RegionId` (0.6); `Rational` (0.3).
- Produces:
  - `struct RankSet { int first = 0; int count = 1; }` with `bool overlaps(const RankSet&) const`, `bool contains(int rank) const`. A component occupies a contiguous rank range of the world communicator; disjoint sets are the normal case (spec §4.1).
  - `struct ComponentDecl { std::string name; RankSet ranks; Rational interval{1}; std::vector<FieldId> fields; std::vector<RegionId> regions; }` — `interval` is the component's own coupling interval; spec §8.1 makes independent intervals the default.
  - `struct Window { enum class Kind { CouplingPeriod, FromComponent, Explicit } kind = Kind::CouplingPeriod; ComponentId component; Rational length{0}; }` with `static Window coupling_period()`, `static Window from(ComponentId)`, `static Window explicit_length(Rational)`.

- [ ] **Step 1: Write the failing test**

`test/unit/component_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/component.hpp>

using soma::ComponentDecl;
using soma::ComponentId;
using soma::RankSet;
using soma::Rational;
using soma::Window;

TEST(RankSet, ContainsAndOverlaps) {
  const RankSet a{.first = 0, .count = 4};
  const RankSet b{.first = 4, .count = 4};
  const RankSet c{.first = 2, .count = 4};
  EXPECT_TRUE(a.contains(3));
  EXPECT_FALSE(a.contains(4));
  EXPECT_FALSE(a.overlaps(b));   // disjoint: the normal case
  EXPECT_TRUE(a.overlaps(c));
}

TEST(ComponentDecl, DefaultsToUnitIntervalOnRankZero) {
  const ComponentDecl c{.name = "atm"};
  EXPECT_EQ(c.interval, Rational{1});
  EXPECT_EQ(c.ranks.first, 0);
  EXPECT_EQ(c.ranks.count, 1);
  EXPECT_TRUE(c.fields.empty());
}

TEST(ComponentDecl, CarriesItsOwnInterval) {
  const ComponentDecl c{.name = "ocn", .ranks = {.first = 4, .count = 8},
                        .interval = Rational{1, 3}};
  EXPECT_EQ(c.interval, Rational(1, 3));
  EXPECT_EQ(c.ranks.count, 8);
}

TEST(Window, ThreeKinds) {
  EXPECT_EQ(Window::coupling_period().kind, Window::Kind::CouplingPeriod);
  const Window w = Window::from(ComponentId{2});
  EXPECT_EQ(w.kind, Window::Kind::FromComponent);
  EXPECT_EQ(w.component, ComponentId{2});
  const Window e = Window::explicit_length(Rational{1, 4});
  EXPECT_EQ(e.kind, Window::Kind::Explicit);
  EXPECT_EQ(e.length, Rational(1, 4));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/component.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/graph/component.hpp`:

```cpp
#pragma once

#include <string>
#include <vector>

#include "soma/base/ids.hpp"
#include "soma/base/rational.hpp"

namespace soma {

/// Contiguous range of world-communicator ranks owned by a component.
/// Disjoint rank sets are the normal case (spec §4.1); overlap is legal.
struct RankSet {
  int first = 0;
  int count = 1;

  [[nodiscard]] bool contains(int rank) const noexcept {
    return rank >= first && rank < first + count;
  }
  [[nodiscard]] bool overlaps(const RankSet& o) const noexcept {
    return first < o.first + o.count && o.first < first + count;
  }
  [[nodiscard]] int last() const noexcept { return first + count - 1; }
};

/// Spec §8.1: `interval` is the component's own coupling interval.
/// Independent intervals are the default; lockstep is all intervals equal.
struct ComponentDecl {
  std::string           name;
  RankSet               ranks;
  Rational              interval{1};
  std::vector<FieldId>  fields;
  std::vector<RegionId> regions;
};

/// The time span an accumulation or space-time projection covers.
struct Window {
  enum class Kind { CouplingPeriod, FromComponent, Explicit };

  Kind        kind = Kind::CouplingPeriod;
  ComponentId component;   // valid only when kind == FromComponent
  Rational    length{0};   // valid only when kind == Explicit

  [[nodiscard]] static Window coupling_period() { return Window{}; }
  [[nodiscard]] static Window from(ComponentId c) {
    return Window{Kind::FromComponent, c, Rational{0}};
  }
  [[nodiscard]] static Window explicit_length(Rational len) {
    return Window{Kind::Explicit, ComponentId{}, len};
  }
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(component_test unit/component_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "RankSet|ComponentDecl|Window"
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/graph/component.hpp test/unit/component_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): ComponentDecl, RankSet, Window"
```

---

### Task 1.3: `Channel` and the fluent builder

Spec §5.2 shows the target syntax. The builder mutates a `Channel` in place and returns `*this`; it never validates (that is Task 1.5's job, so that all problems are reported together).

**Files:**
- Create: `include/soma/graph/channel.hpp`
- Test: `test/unit/channel_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `FieldId`, `ChannelId`, `AccumOp`, `TemporalBasis` (0.6); `Window` (1.2); `Rational` (0.3).
- Produces:
  - `enum class RemapFlags : std::uint32_t { None=0, Conservative=1, Monotone=2, HighOrder=4 }` with `|`, `&`, and `bool has(RemapFlags set, RemapFlags bit)`
  - `struct AccumulateSpec { AccumOp op = AccumOp::Instant; Window window{}; }`
  - `struct RemapSpec { RemapFlags flags = RemapFlags::None; std::string method_override; }`
  - `struct SpacetimeSpec { TemporalBasis basis = TemporalBasis::None; int degree = 0; Window window{}; }`
  - `struct RelaxSpec { bool enabled = false; double omega = 1.0; }`
  - `struct ExtrapolateSpec { bool enabled = false; int order = 0; }`
  - `struct Channel { std::string name; FieldId source; FieldId target; AccumulateSpec accumulate; RemapSpec remap; SpacetimeSpec spacetime; RelaxSpec relax; ExtrapolateSpec extrapolate; int lag = 0; std::string time_level_mismatch_reason; }`
  - `class ChannelBuilder` wrapping a `Channel&` with `accumulate(AccumOp, Window)`, `remap(RemapFlags)`, `remap_method(std::string)`, `spacetime(TemporalBasis, int degree, Window)`, `relax(double)`, `extrapolate(int order)`, `lag(int)`, `conservative()`, `allow_time_level_mismatch(std::string reason)` — each returning `ChannelBuilder&`

- [ ] **Step 1: Write the failing test**

`test/unit/channel_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/channel.hpp>

using soma::AccumOp;
using soma::Channel;
using soma::ChannelBuilder;
using soma::ComponentId;
using soma::FieldId;
using soma::RemapFlags;
using soma::TemporalBasis;
using soma::Window;

TEST(Channel, DefaultsToInstantNoRemapNoRelax) {
  const Channel c;
  EXPECT_EQ(c.accumulate.op, AccumOp::Instant);
  EXPECT_EQ(c.remap.flags, RemapFlags::None);
  EXPECT_FALSE(c.relax.enabled);
  EXPECT_FALSE(c.extrapolate.enabled);
  EXPECT_EQ(c.lag, 0);
  EXPECT_EQ(c.spacetime.basis, TemporalBasis::None);
}

TEST(RemapFlags, CombineAndTest) {
  const RemapFlags f = RemapFlags::Conservative | RemapFlags::Monotone;
  EXPECT_TRUE(has(f, RemapFlags::Conservative));
  EXPECT_TRUE(has(f, RemapFlags::Monotone));
  EXPECT_FALSE(has(f, RemapFlags::HighOrder));
}

// This is the exact syntax from spec §5.2.
TEST(ChannelBuilder, BuildsTheSpecExample) {
  Channel c{.name = "T_wall->T_bc", .source = FieldId{0}, .target = FieldId{1}};
  ChannelBuilder{c}
      .accumulate(AccumOp::Mean, Window::coupling_period())
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .relax(0.7);

  EXPECT_EQ(c.accumulate.op, AccumOp::Mean);
  EXPECT_EQ(c.accumulate.window.kind, Window::Kind::CouplingPeriod);
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Conservative));
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Monotone));
  EXPECT_TRUE(c.relax.enabled);
  EXPECT_DOUBLE_EQ(c.relax.omega, 0.7);
}

// Spec §7.3 example.
TEST(ChannelBuilder, BuildsSpacetimeChannel) {
  Channel c{.name = "sst->T_surf", .source = FieldId{2}, .target = FieldId{3}};
  ChannelBuilder{c}
      .spacetime(TemporalBasis::DG, 1, Window::from(ComponentId{0}))
      .conservative();

  EXPECT_EQ(c.spacetime.basis, TemporalBasis::DG);
  EXPECT_EQ(c.spacetime.degree, 1);
  EXPECT_EQ(c.spacetime.window.kind, Window::Kind::FromComponent);
  EXPECT_TRUE(has(c.remap.flags, RemapFlags::Conservative));
}

TEST(ChannelBuilder, LagIsRecorded) {
  Channel c;
  ChannelBuilder{c}.lag(1);
  EXPECT_EQ(c.lag, 1);
}

// Spec §4.3: the escape hatch must record a reason, never be silent.
TEST(ChannelBuilder, TimeLevelMismatchRequiresAReason) {
  Channel c;
  EXPECT_TRUE(c.time_level_mismatch_reason.empty());
  ChannelBuilder{c}.allow_time_level_mismatch("deliberate leapfrog stagger");
  EXPECT_EQ(c.time_level_mismatch_reason, "deliberate leapfrog stagger");
}

TEST(ChannelBuilder, ChainsReturnSelf) {
  Channel c;
  ChannelBuilder{c}.lag(2).relax(0.5).extrapolate(1);
  EXPECT_EQ(c.lag, 2);
  EXPECT_DOUBLE_EQ(c.relax.omega, 0.5);
  EXPECT_TRUE(c.extrapolate.enabled);
  EXPECT_EQ(c.extrapolate.order, 1);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/channel.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/graph/channel.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <string>
#include <utility>

#include "soma/base/ids.hpp"
#include "soma/graph/component.hpp"

namespace soma {

enum class RemapFlags : std::uint32_t {
  None         = 0,
  Conservative = 1u << 0,
  Monotone     = 1u << 1,
  HighOrder    = 1u << 2,
};

[[nodiscard]] constexpr RemapFlags operator|(RemapFlags a, RemapFlags b) noexcept {
  return static_cast<RemapFlags>(static_cast<std::uint32_t>(a) |
                                 static_cast<std::uint32_t>(b));
}
[[nodiscard]] constexpr RemapFlags operator&(RemapFlags a, RemapFlags b) noexcept {
  return static_cast<RemapFlags>(static_cast<std::uint32_t>(a) &
                                 static_cast<std::uint32_t>(b));
}
[[nodiscard]] constexpr bool has(RemapFlags set, RemapFlags bit) noexcept {
  return static_cast<std::uint32_t>(set & bit) != 0u;
}

struct AccumulateSpec {
  AccumOp op = AccumOp::Instant;
  Window  window{};
};

struct RemapSpec {
  RemapFlags  flags = RemapFlags::None;
  std::string method_override;  // empty = derive from disc pair + conservation
};

struct SpacetimeSpec {
  TemporalBasis basis = TemporalBasis::None;
  int           degree = 0;
  Window        window{};
};

struct RelaxSpec {
  bool   enabled = false;
  double omega = 1.0;
};

struct ExtrapolateSpec {
  bool enabled = false;
  int  order = 0;
};

/// One directed coupling edge with its temporal and spatial semantics.
/// Spec §5.2 lowers this into primitive operators.
struct Channel {
  std::string     name;
  FieldId         source;
  FieldId         target;
  AccumulateSpec  accumulate;
  RemapSpec       remap;
  SpacetimeSpec   spacetime;
  RelaxSpec       relax;
  ExtrapolateSpec extrapolate;
  int             lag = 0;  // spec §4.3: lag(k) is a TimeLevel offset of -k
  std::string     time_level_mismatch_reason;  // empty = strict typing enforced
};

/// Fluent mutator. Deliberately does not validate: spec Global Constraints
/// require all problems to be reported together by validate() (Task 1.5).
class ChannelBuilder {
 public:
  explicit ChannelBuilder(Channel& c) : c_(c) {}

  ChannelBuilder& accumulate(AccumOp op, Window w = Window::coupling_period()) {
    c_.accumulate.op = op;
    c_.accumulate.window = w;
    return *this;
  }
  ChannelBuilder& remap(RemapFlags flags) {
    c_.remap.flags = c_.remap.flags | flags;
    return *this;
  }
  ChannelBuilder& remap_method(std::string method) {
    c_.remap.method_override = std::move(method);
    return *this;
  }
  ChannelBuilder& conservative() {
    return remap(RemapFlags::Conservative);
  }
  ChannelBuilder& spacetime(TemporalBasis basis, int degree,
                            Window w = Window::coupling_period()) {
    c_.spacetime.basis = basis;
    c_.spacetime.degree = degree;
    c_.spacetime.window = w;
    return *this;
  }
  ChannelBuilder& relax(double omega) {
    c_.relax.enabled = true;
    c_.relax.omega = omega;
    return *this;
  }
  ChannelBuilder& extrapolate(int order) {
    c_.extrapolate.enabled = true;
    c_.extrapolate.order = order;
    return *this;
  }
  ChannelBuilder& lag(int steps) {
    c_.lag = steps;
    return *this;
  }
  ChannelBuilder& allow_time_level_mismatch(std::string reason) {
    c_.time_level_mismatch_reason = std::move(reason);
    return *this;
  }

 private:
  Channel& c_;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(channel_test unit/channel_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "Channel|RemapFlags"
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/graph/channel.hpp test/unit/channel_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): Channel semantics and fluent ChannelBuilder"
```

---

### Task 1.4: `CouplingGraph`

The container owning all index spaces. Every `add_*` returns a strong-typed id; every accessor is by id.

**Files:**
- Create: `include/soma/graph/coupling_graph.hpp`
- Test: `test/unit/coupling_graph_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: everything from 1.1–1.3.
- Produces: `class CouplingGraph` with
  - `MeshId add_mesh(MeshDecl)`, `RegionId add_region(Region)`, `ComponentId add_component(ComponentDecl)`, `FieldId add_field(ComponentId, FieldSpec)`, `ChannelId add_channel(std::string name, FieldId source, FieldId target)`
  - `ChannelBuilder channel(ChannelId)` — returns a builder over the stored channel
  - accessors `const MeshDecl& mesh(MeshId) const`, `const Region& region(RegionId) const`, `const ComponentDecl& component(ComponentId) const`, `const FieldSpec& field(FieldId) const`, `const Channel& channel_at(ChannelId) const`
  - counts `mesh_count()`, `region_count()`, `component_count()`, `field_count()`, `channel_count()`
  - `std::optional<ComponentId> find_component(std::string_view) const`, `std::optional<FieldId> find_field(ComponentId, std::string_view) const`
  - `ComponentId owner_of(FieldId) const`
  - `IterationRegionIdx add_iteration_region(IterationRegionDecl)` — declared here, populated in Phase 7. `struct IterationRegionDecl { std::string name; std::vector<ChannelId> channels; int max_iterations = 1; }` is enough for now; Phase 7 Task 7.1 extends it with convergence and acceleration.

  Note: `IterationRegionIdx` is the alias `RegionIdx` from Task 0.6. Renamed here for readability at call sites; both names refer to `Id<IterRegionTag>`.

- [ ] **Step 1: Write the failing test**

`test/unit/coupling_graph_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldSpec;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
namespace u = soma::units;

namespace {

// Two components, one channel: the smallest useful graph.
CouplingGraph make_pair_graph() {
  CouplingGraph g;
  const auto m_f = g.add_mesh(MeshDecl{.name = "fluid_mesh",
                                       .source = "generated:box:8x8x8"});
  const auto m_s = g.add_mesh(MeshDecl{.name = "solid_mesh",
                                       .source = "generated:box:5x5x5"});
  const auto r_f = g.add_region(Region{.name = "fluid_wall", .mesh = m_f,
                                       .entity_set = "wall", .dimension = 2});
  const auto r_s = g.add_region(Region{.name = "solid_wall", .mesh = m_s,
                                       .entity_set = "wall", .dimension = 2});

  const auto fluid = g.add_component(
      ComponentDecl{.name = "fluid", .ranks = {.first = 0, .count = 2},
                    .interval = Rational{1, 2}});
  const auto solid = g.add_component(
      ComponentDecl{.name = "solid", .ranks = {.first = 2, .count = 2},
                    .interval = Rational{1}});

  const auto q_out = g.add_field(
      fluid, FieldSpec{.name = "heat_flux", .region = r_f,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = soma::Intent::Provides});
  const auto q_in = g.add_field(
      solid, FieldSpec{.name = "q_wall", .region = r_s,
                       .units = u::kHeatFluxDensity,
                       .conservation = Conservation::Extensive,
                       .intent = soma::Intent::Requires});

  const auto ch = g.add_channel("flux", q_out, q_in);
  g.channel(ch).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);
  return g;
}

}  // namespace

TEST(CouplingGraph, EmptyGraphHasNoEntities) {
  const CouplingGraph g;
  EXPECT_EQ(g.component_count(), 0u);
  EXPECT_EQ(g.channel_count(), 0u);
}

TEST(CouplingGraph, IdsAreSequentialAndValid) {
  CouplingGraph g;
  const auto a = g.add_mesh(MeshDecl{.name = "a"});
  const auto b = g.add_mesh(MeshDecl{.name = "b"});
  EXPECT_TRUE(a.valid());
  EXPECT_EQ(a.value(), 0u);
  EXPECT_EQ(b.value(), 1u);
  EXPECT_EQ(g.mesh_count(), 2u);
}

TEST(CouplingGraph, BuildsThePairGraph) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.mesh_count(), 2u);
  EXPECT_EQ(g.region_count(), 2u);
  EXPECT_EQ(g.component_count(), 2u);
  EXPECT_EQ(g.field_count(), 2u);
  EXPECT_EQ(g.channel_count(), 1u);
}

TEST(CouplingGraph, FieldsAreOwnedByTheirComponent) {
  const CouplingGraph g = make_pair_graph();
  const auto fluid = g.find_component("fluid").value();
  const auto q = g.find_field(fluid, "heat_flux").value();
  EXPECT_EQ(g.owner_of(q), fluid);
  EXPECT_EQ(g.component(fluid).fields.size(), 1u);
}

TEST(CouplingGraph, FindReturnsNulloptWhenAbsent) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_FALSE(g.find_component("nonexistent").has_value());
  const auto fluid = g.find_component("fluid").value();
  EXPECT_FALSE(g.find_field(fluid, "nope").has_value());
}

TEST(CouplingGraph, ChannelBuilderMutatesStoredChannel) {
  const CouplingGraph g = make_pair_graph();
  const auto& ch = g.channel_at(soma::ChannelId{0});
  EXPECT_EQ(ch.accumulate.op, AccumOp::Mean);
  EXPECT_TRUE(has(ch.remap.flags, RemapFlags::Conservative));
}

TEST(CouplingGraph, ComponentsKeepIndependentIntervals) {
  const CouplingGraph g = make_pair_graph();
  EXPECT_EQ(g.component(g.find_component("fluid").value()).interval,
            Rational(1, 2));
  EXPECT_EQ(g.component(g.find_component("solid").value()).interval,
            Rational(1));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/coupling_graph.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/graph/coupling_graph.hpp`:

```cpp
#pragma once

#include <cassert>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "soma/graph/channel.hpp"
#include "soma/graph/component.hpp"
#include "soma/graph/field_spec.hpp"
#include "soma/graph/region.hpp"

namespace soma {

using IterationRegionIdx = RegionIdx;

/// Phase 7 Task 7.1 extends this with convergence measures and acceleration.
struct IterationRegionDecl {
  std::string            name;
  std::vector<ChannelId> channels;
  int                    max_iterations = 1;
};

/// Spec §2 layer 1: the authored representation. Pure data — no execution,
/// no MPI, no MOAB. This is the only thing users write and the only thing
/// serialised.
class CouplingGraph {
 public:
  MeshId add_mesh(MeshDecl d) {
    meshes_.push_back(std::move(d));
    return MeshId{static_cast<std::uint32_t>(meshes_.size() - 1)};
  }

  RegionId add_region(Region r) {
    regions_.push_back(std::move(r));
    return RegionId{static_cast<std::uint32_t>(regions_.size() - 1)};
  }

  ComponentId add_component(ComponentDecl c) {
    components_.push_back(std::move(c));
    return ComponentId{static_cast<std::uint32_t>(components_.size() - 1)};
  }

  FieldId add_field(ComponentId owner, FieldSpec f) {
    assert(owner.valid() && owner.value() < components_.size());
    fields_.push_back(std::move(f));
    const FieldId id{static_cast<std::uint32_t>(fields_.size() - 1)};
    field_owner_.push_back(owner);
    components_[owner.value()].fields.push_back(id);
    return id;
  }

  ChannelId add_channel(std::string name, FieldId source, FieldId target) {
    channels_.push_back(Channel{.name = std::move(name),
                                .source = source,
                                .target = target});
    return ChannelId{static_cast<std::uint32_t>(channels_.size() - 1)};
  }

  IterationRegionIdx add_iteration_region(IterationRegionDecl d) {
    iteration_regions_.push_back(std::move(d));
    return IterationRegionIdx{
        static_cast<std::uint32_t>(iteration_regions_.size() - 1)};
  }

  [[nodiscard]] ChannelBuilder channel(ChannelId id) {
    assert(id.valid() && id.value() < channels_.size());
    return ChannelBuilder{channels_[id.value()]};
  }

  [[nodiscard]] const MeshDecl& mesh(MeshId id) const {
    return meshes_.at(id.value());
  }
  [[nodiscard]] const Region& region(RegionId id) const {
    return regions_.at(id.value());
  }
  [[nodiscard]] const ComponentDecl& component(ComponentId id) const {
    return components_.at(id.value());
  }
  [[nodiscard]] const FieldSpec& field(FieldId id) const {
    return fields_.at(id.value());
  }
  [[nodiscard]] const Channel& channel_at(ChannelId id) const {
    return channels_.at(id.value());
  }
  [[nodiscard]] const IterationRegionDecl& iteration_region(
      IterationRegionIdx id) const {
    return iteration_regions_.at(id.value());
  }

  [[nodiscard]] std::size_t mesh_count() const noexcept { return meshes_.size(); }
  [[nodiscard]] std::size_t region_count() const noexcept { return regions_.size(); }
  [[nodiscard]] std::size_t component_count() const noexcept {
    return components_.size();
  }
  [[nodiscard]] std::size_t field_count() const noexcept { return fields_.size(); }
  [[nodiscard]] std::size_t channel_count() const noexcept {
    return channels_.size();
  }
  [[nodiscard]] std::size_t iteration_region_count() const noexcept {
    return iteration_regions_.size();
  }

  [[nodiscard]] ComponentId owner_of(FieldId f) const {
    return field_owner_.at(f.value());
  }

  [[nodiscard]] std::optional<ComponentId> find_component(
      std::string_view name) const {
    for (std::size_t i = 0; i < components_.size(); ++i) {
      if (components_[i].name == name) {
        return ComponentId{static_cast<std::uint32_t>(i)};
      }
    }
    return std::nullopt;
  }

  [[nodiscard]] std::optional<FieldId> find_field(ComponentId owner,
                                                  std::string_view name) const {
    for (const FieldId f : components_.at(owner.value()).fields) {
      if (fields_.at(f.value()).name == name) return f;
    }
    return std::nullopt;
  }

 private:
  std::vector<MeshDecl>            meshes_;
  std::vector<Region>              regions_;
  std::vector<ComponentDecl>       components_;
  std::vector<FieldSpec>           fields_;
  std::vector<ComponentId>         field_owner_;  // parallel to fields_
  std::vector<Channel>             channels_;
  std::vector<IterationRegionDecl> iteration_regions_;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(coupling_graph_test unit/coupling_graph_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R CouplingGraph
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/graph/coupling_graph.hpp test/unit/coupling_graph_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): CouplingGraph container with strong-typed index spaces"
```

---

### Task 1.5: Graph validation

Every rule accumulates into `Diagnostics`. Nothing aborts early.

**Files:**
- Create: `include/soma/graph/validate.hpp`
- Test: `test/unit/validate_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `CouplingGraph` (1.4), `Diagnostics` (0.2).
- Produces: `Diagnostics validate(const CouplingGraph&)` implementing these rules, each with a stable code:

| Code | Severity | Rule |
|---|---|---|
| `E-DUP-COMP` | Error | duplicate component name |
| `E-DUP-FIELD` | Error | duplicate field name within one component |
| `E-BAD-REF` | Error | channel references an invalid `FieldId`, or a field an invalid `RegionId`, or a region an invalid `MeshId` |
| `E-UNITS` | Error | channel source and target units differ |
| `E-SHAPE` | Error | channel source and target component counts differ |
| `E-CONSERV` | Error | channel source and target conservation classes differ |
| `E-INTENT` | Error | channel source is not `Provides`/`Bidirectional`, or target is not `Requires`/`Bidirectional` |
| `E-SELF` | Error | channel source and target belong to the same component |
| `E-INTERVAL` | Error | a component interval is zero or negative |
| `E-LAG` | Error | `abs(lag) * source interval` exceeds the coupling window (OASIS's documented deadlock hazard, spec §4.3) |
| `W-NO-CONSUMER` | Warning | a `Provides` field is not the source of any channel |
| `W-MULTI-WRITER` | Warning | two channels target the same field without a `Merge` (spec §5.1 `Merge` is required to combine) |
| `W-MISMATCH-OK` | Warning | `time_level_mismatch_reason` is set — recorded in provenance, never silent |

- [ ] **Step 1: Write the failing test**

`test/unit/validate_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/graph/validate.hpp>

#include <algorithm>
#include <string_view>

using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::Diagnostics;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
namespace u = soma::units;

namespace {

bool has_code(const Diagnostics& d, std::string_view code) {
  return std::any_of(d.items().begin(), d.items().end(),
                     [code](const auto& i) { return i.code == code; });
}

// A graph that validates clean; individual tests perturb one thing.
struct Fixture {
  CouplingGraph g;
  soma::ComponentId fluid, solid;
  soma::FieldId src, dst;
  soma::ChannelId ch;

  Fixture() {
    const auto m = g.add_mesh(MeshDecl{.name = "m", .source = "generated:box"});
    const auto r = g.add_region(Region{.name = "wall", .mesh = m,
                                       .entity_set = "w", .dimension = 2});
    fluid = g.add_component(ComponentDecl{.name = "fluid",
                                          .ranks = {.first = 0, .count = 1},
                                          .interval = Rational{1}});
    solid = g.add_component(ComponentDecl{.name = "solid",
                                          .ranks = {.first = 1, .count = 1},
                                          .interval = Rational{1}});
    src = g.add_field(fluid, FieldSpec{.name = "q", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
                                       .intent = Intent::Provides});
    dst = g.add_field(solid, FieldSpec{.name = "q_in", .region = r,
                                       .units = u::kHeatFluxDensity,
                                       .conservation = Conservation::Extensive,
                                       .intent = Intent::Requires});
    ch = g.add_channel("flux", src, dst);
  }
};

}  // namespace

TEST(Validate, CleanGraphHasNoErrors) {
  Fixture f;
  const Diagnostics d = validate(f.g);
  EXPECT_FALSE(d.has_errors()) << d.to_string();
}

TEST(Validate, RejectsDuplicateComponentNames) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "fluid"});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-DUP-COMP")) << d.to_string();
}

TEST(Validate, RejectsUnitMismatch) {
  Fixture f;
  const auto r = f.g.field(f.dst).region;
  const auto bad = f.g.add_field(
      f.solid, FieldSpec{.name = "T_in", .region = r, .units = u::kKelvin,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad", f.src, bad);   // W/m^2 -> K
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-UNITS")) << d.to_string();
}

TEST(Validate, RejectsComponentCountMismatch) {
  Fixture f;
  const auto r = f.g.field(f.dst).region;
  const auto vec = f.g.add_field(
      f.solid, FieldSpec{.name = "v_in", .region = r, .components = 3,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad_shape", f.src, vec);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-SHAPE")) << d.to_string();
}

TEST(Validate, RejectsIntentViolation) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  // A Requires field used as a channel source.
  const auto wrong = f.g.add_field(
      f.fluid, FieldSpec{.name = "wrong", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Requires});
  f.g.add_channel("bad_intent", wrong, f.dst);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-INTENT")) << d.to_string();
}

TEST(Validate, RejectsSelfCoupling) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto other = f.g.add_field(
      f.fluid, FieldSpec{.name = "q2", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Requires});
  f.g.add_channel("self", f.src, other);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-SELF")) << d.to_string();
}

TEST(Validate, RejectsZeroInterval) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "broken", .interval = Rational{0}});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-INTERVAL")) << d.to_string();
}

// Spec §4.3: OASIS documents lag > coupling period as a deadlock hazard.
TEST(Validate, RejectsLagExceedingWindow) {
  Fixture f;
  f.g.channel(f.ch).lag(5);   // source interval 1, window 1 => 5 > 1
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-LAG")) << d.to_string();
}

TEST(Validate, WarnsOnUnconsumedProvidesField) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  f.g.add_field(f.fluid, FieldSpec{.name = "orphan", .region = r,
                                   .intent = Intent::Provides});
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-NO-CONSUMER")) << d.to_string();
  EXPECT_FALSE(d.has_errors());  // warning only
}

TEST(Validate, WarnsOnTwoWritersToOneTarget) {
  Fixture f;
  const auto r = f.g.field(f.src).region;
  const auto src2 = f.g.add_field(
      f.fluid, FieldSpec{.name = "q_alt", .region = r,
                         .units = u::kHeatFluxDensity,
                         .conservation = Conservation::Extensive,
                         .intent = Intent::Provides});
  f.g.add_channel("second_writer", src2, f.dst);
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-MULTI-WRITER")) << d.to_string();
}

TEST(Validate, RecordsTimeLevelMismatchAsWarning) {
  Fixture f;
  f.g.channel(f.ch).allow_time_level_mismatch("leapfrog by design");
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "W-MISMATCH-OK")) << d.to_string();
  EXPECT_FALSE(d.has_errors());
}

TEST(Validate, ReportsEveryProblemNotJustTheFirst) {
  Fixture f;
  f.g.add_component(ComponentDecl{.name = "fluid"});           // E-DUP-COMP
  f.g.add_component(ComponentDecl{.name = "z", .interval = Rational{0}});  // E-INTERVAL
  const Diagnostics d = validate(f.g);
  EXPECT_TRUE(has_code(d, "E-DUP-COMP"));
  EXPECT_TRUE(has_code(d, "E-INTERVAL"));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/validate.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/graph/validate.hpp`:

```cpp
#pragma once

#include <cstdlib>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {
namespace detail {

inline std::string channel_ctx(const CouplingGraph& g, ChannelId c) {
  return "channel '" + g.channel_at(c).name + "'";
}

/// The window length a channel's lag is measured against.
inline Rational channel_window(const CouplingGraph& g, const Channel& ch) {
  const Window& w = ch.accumulate.window;
  switch (w.kind) {
    case Window::Kind::Explicit:
      return w.length;
    case Window::Kind::FromComponent:
      return w.component.valid() ? g.component(w.component).interval
                                 : Rational{1};
    case Window::Kind::CouplingPeriod:
    default: {
      // The target's interval defines the period the value must survive.
      const ComponentId tgt = g.owner_of(ch.target);
      return g.component(tgt).interval;
    }
  }
}

}  // namespace detail

/// Spec Global Constraints: accumulate every problem, report them together.
[[nodiscard]] inline Diagnostics validate(const CouplingGraph& g) {
  Diagnostics d;

  // --- name uniqueness -----------------------------------------------------
  std::unordered_set<std::string> comp_names;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    if (!comp_names.insert(c.name).second) {
      d.add(Severity::Error, "E-DUP-COMP",
            "duplicate component name '" + c.name + "'");
    }
    if (!(c.interval > Rational{0})) {
      d.add(Severity::Error, "E-INTERVAL",
            "component '" + c.name + "' has non-positive interval " +
                c.interval.to_string());
    }
    std::unordered_set<std::string> field_names;
    for (const FieldId f : c.fields) {
      if (!field_names.insert(g.field(f).name).second) {
        d.add(Severity::Error, "E-DUP-FIELD",
              "duplicate field '" + g.field(f).name + "' in component '" +
                  c.name + "'");
      }
    }
  }

  // --- reference integrity -------------------------------------------------
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    if (!f.region.valid() || f.region.value() >= g.region_count()) {
      d.add(Severity::Error, "E-BAD-REF",
            "field '" + f.name + "' references an invalid region");
    }
  }
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    if (!r.mesh.valid() || r.mesh.value() >= g.mesh_count()) {
      d.add(Severity::Error, "E-BAD-REF",
            "region '" + r.name + "' references an invalid mesh");
    }
  }

  // --- per-channel rules ---------------------------------------------------
  std::unordered_map<std::uint32_t, int> writers;  // target field -> count

  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    const auto& ch = g.channel_at(cid);
    const std::string ctx = detail::channel_ctx(g, cid);

    const bool src_ok = ch.source.valid() && ch.source.value() < g.field_count();
    const bool tgt_ok = ch.target.valid() && ch.target.value() < g.field_count();
    if (!src_ok || !tgt_ok) {
      d.add(Severity::Error, "E-BAD-REF",
            "channel references an invalid field", ctx);
      continue;  // every later rule would dereference garbage
    }

    const auto& sf = g.field(ch.source);
    const auto& tf = g.field(ch.target);
    ++writers[ch.target.value()];

    if (!(sf.units == tf.units)) {
      d.add(Severity::Error, "E-UNITS",
            "unit mismatch: source '" + sf.name + "' is " + sf.units.to_string() +
                ", target '" + tf.name + "' is " + tf.units.to_string(), ctx);
    }
    if (!same_shape(sf, tf)) {
      d.add(Severity::Error, "E-SHAPE",
            "component count mismatch: " + std::to_string(sf.components) +
                " vs " + std::to_string(tf.components), ctx);
    }
    if (sf.conservation != tf.conservation) {
      d.add(Severity::Error, "E-CONSERV",
            std::string{"conservation class mismatch: "} +
                to_cstring(sf.conservation) + " vs " +
                to_cstring(tf.conservation), ctx);
    }
    if (sf.intent == Intent::Requires) {
      d.add(Severity::Error, "E-INTENT",
            "source field '" + sf.name + "' is Requires; must Provide", ctx);
    }
    if (tf.intent == Intent::Provides) {
      d.add(Severity::Error, "E-INTENT",
            "target field '" + tf.name + "' is Provides; must Require", ctx);
    }
    if (g.owner_of(ch.source) == g.owner_of(ch.target)) {
      d.add(Severity::Error, "E-SELF",
            "source and target belong to the same component", ctx);
    }

    // Spec §4.3: OASIS documents lag beyond the coupling period as a
    // restartability and deadlock hazard.
    if (ch.lag != 0) {
      const Rational span =
          Rational{std::abs(ch.lag)} * g.component(g.owner_of(ch.source)).interval;
      const Rational window = detail::channel_window(g, ch);
      if (span > window) {
        d.add(Severity::Error, "E-LAG",
              "lag spans " + span.to_string() + " but the coupling window is " +
                  window.to_string(), ctx);
      }
    }

    if (!ch.time_level_mismatch_reason.empty()) {
      d.add(Severity::Warning, "W-MISMATCH-OK",
            "time-level mismatch permitted: " + ch.time_level_mismatch_reason,
            ctx);
    }
  }

  // --- cross-channel rules -------------------------------------------------
  for (const auto& [field_value, count] : writers) {
    if (count > 1) {
      d.add(Severity::Warning, "W-MULTI-WRITER",
            "field '" + g.field(FieldId{field_value}).name + "' is written by " +
                std::to_string(count) + " channels; add an explicit Merge");
    }
  }

  std::unordered_set<std::uint32_t> sources;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& ch = g.channel_at(ChannelId{i});
    if (ch.source.valid()) sources.insert(ch.source.value());
  }
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    if (f.intent == Intent::Provides && !sources.count(i)) {
      d.add(Severity::Warning, "W-NO-CONSUMER",
            "field '" + f.name + "' is Provides but no channel consumes it");
    }
  }

  return d;
}

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(validate_test unit/validate_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Validate
```

Expected: 12 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/graph/validate.hpp test/unit/validate_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): graph validation with accumulated diagnostics"
```

---

### Task 1.6: Flat descriptor — the narrow waist

Spec §3.2. The `CouplingGraph` serialises to a POD byte buffer that the compiled core, the C ABI, Fortran, Python, and YAML all consume identically.

**Files:**
- Create: `include/soma/graph/descriptor.hpp`
- Test: `test/unit/descriptor_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `CouplingGraph` (1.4).
- Produces:
  - `inline constexpr std::uint32_t kGraphMagic = 0x534F4D41;` (`"SOMA"`) and `inline constexpr std::uint32_t kGraphVersion = 1;`
  - `std::vector<std::byte> serialize(const CouplingGraph&)`
  - `Result<CouplingGraph> deserialize(std::span<const std::byte>)`
  - `bool structurally_equal(const CouplingGraph&, const CouplingGraph&)` — field-by-field comparison used by round-trip tests

  Format: a fixed header (`magic`, `version`, then a count for each of the seven index spaces), followed by each table's records. Strings are stored as `uint32` length + bytes. `Rational` as two `int64`. `Units` as seven `int8`. All integers little-endian; the format is not portable across endianness in v1 and the header records a byte-order marker so a future reader can detect it.

- [ ] **Step 1: Write the failing test**

`test/unit/descriptor_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/graph/descriptor.hpp>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
using soma::TemporalBasis;
namespace u = soma::units;

namespace {

// Exercises every field the descriptor must carry, including the awkward ones:
// rational intervals, spacetime specs, a lag, and a mismatch reason.
CouplingGraph rich_graph() {
  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "mesh_a", .source = "a.h5m"});
  const auto r = g.add_region(Region{.name = "iface", .mesh = m,
                                     .entity_set = "ss1", .dimension = 2});
  const auto a = g.add_component(ComponentDecl{.name = "A",
                                               .ranks = {.first = 0, .count = 3},
                                               .interval = Rational{1, 3}});
  const auto b = g.add_component(ComponentDecl{.name = "B",
                                               .ranks = {.first = 3, .count = 5},
                                               .interval = Rational{1, 7}});
  const auto s = g.add_field(a, FieldSpec{.name = "flux", .region = r,
                                          .disc = soma::Discretization::DiscontinuousGLL,
                                          .order = 3, .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Provides});
  const auto t = g.add_field(b, FieldSpec{.name = "flux_in", .region = r,
                                          .disc = soma::Discretization::FV,
                                          .components = 3,
                                          .units = u::kHeatFluxDensity,
                                          .conservation = Conservation::Extensive,
                                          .intent = Intent::Requires});
  const auto ch = g.add_channel("A->B", s, t);
  g.channel(ch)
      .accumulate(AccumOp::Integral, soma::Window::from(a))
      .remap(RemapFlags::Conservative | RemapFlags::HighOrder)
      .remap_method("mono3")
      .spacetime(TemporalBasis::DG, 2, soma::Window::explicit_length(Rational{2, 5}))
      .relax(0.35)
      .extrapolate(2)
      .lag(0)
      .allow_time_level_mismatch("verified by hand");
  g.add_iteration_region(soma::IterationRegionDecl{
      .name = "outer", .channels = {ch}, .max_iterations = 25});
  return g;
}

}  // namespace

TEST(Descriptor, HeaderHasMagicAndVersion) {
  const auto bytes = serialize(rich_graph());
  ASSERT_GE(bytes.size(), 8u);
  std::uint32_t magic = 0, version = 0;
  std::memcpy(&magic, bytes.data(), 4);
  std::memcpy(&version, bytes.data() + 4, 4);
  EXPECT_EQ(magic, soma::kGraphMagic);
  EXPECT_EQ(version, soma::kGraphVersion);
}

TEST(Descriptor, RoundTripsRichGraph) {
  const CouplingGraph original = rich_graph();
  const auto bytes = serialize(original);
  auto restored = deserialize(bytes);
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  EXPECT_TRUE(structurally_equal(original, restored.value()));
}

TEST(Descriptor, PreservesRationalIntervalsExactly) {
  const auto bytes = serialize(rich_graph());
  const auto g = deserialize(bytes).value();
  EXPECT_EQ(g.component(g.find_component("A").value()).interval, Rational(1, 3));
  EXPECT_EQ(g.component(g.find_component("B").value()).interval, Rational(1, 7));
}

TEST(Descriptor, PreservesChannelSemantics) {
  const auto g = deserialize(serialize(rich_graph())).value();
  const auto& ch = g.channel_at(soma::ChannelId{0});
  EXPECT_EQ(ch.accumulate.op, AccumOp::Integral);
  EXPECT_EQ(ch.remap.method_override, "mono3");
  EXPECT_EQ(ch.spacetime.basis, TemporalBasis::DG);
  EXPECT_EQ(ch.spacetime.degree, 2);
  EXPECT_EQ(ch.spacetime.window.length, Rational(2, 5));
  EXPECT_DOUBLE_EQ(ch.relax.omega, 0.35);
  EXPECT_EQ(ch.extrapolate.order, 2);
  EXPECT_EQ(ch.time_level_mismatch_reason, "verified by hand");
}

TEST(Descriptor, PreservesIterationRegions) {
  const auto g = deserialize(serialize(rich_graph())).value();
  ASSERT_EQ(g.iteration_region_count(), 1u);
  EXPECT_EQ(g.iteration_region(soma::IterationRegionIdx{0}).max_iterations, 25);
}

TEST(Descriptor, RejectsTruncatedBuffer) {
  auto bytes = serialize(rich_graph());
  bytes.resize(bytes.size() / 2);
  const auto r = deserialize(bytes);
  EXPECT_FALSE(r.ok());
}

TEST(Descriptor, RejectsWrongMagic) {
  auto bytes = serialize(rich_graph());
  bytes[0] = std::byte{0xFF};
  const auto r = deserialize(bytes);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-DESC-MAGIC");
}

TEST(Descriptor, SerializationIsDeterministic) {
  EXPECT_EQ(serialize(rich_graph()), serialize(rich_graph()));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/graph/descriptor.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/graph/descriptor.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

inline constexpr std::uint32_t kGraphMagic   = 0x534F4D41u;  // "SOMA"
inline constexpr std::uint32_t kGraphVersion = 1u;

namespace detail {

class Writer {
 public:
  template <class T>
  void pod(const T& v) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* p = reinterpret_cast<const std::byte*>(&v);
    buf_.insert(buf_.end(), p, p + sizeof(T));
  }
  void str(const std::string& s) {
    pod(static_cast<std::uint32_t>(s.size()));
    const auto* p = reinterpret_cast<const std::byte*>(s.data());
    buf_.insert(buf_.end(), p, p + s.size());
  }
  void rational(const Rational& r) {
    pod(r.num());
    pod(r.den());
  }
  void units(const Units& u) {
    pod(u.length); pod(u.mass); pod(u.time); pod(u.current);
    pod(u.temperature); pod(u.amount); pod(u.luminosity);
  }
  void window(const Window& w) {
    pod(static_cast<std::uint8_t>(w.kind));
    pod(w.component.value());
    rational(w.length);
  }
  [[nodiscard]] std::vector<std::byte> take() { return std::move(buf_); }

 private:
  std::vector<std::byte> buf_;
};

class Reader {
 public:
  explicit Reader(std::span<const std::byte> b) : buf_(b) {}

  [[nodiscard]] bool ok() const noexcept { return ok_; }

  template <class T>
  T pod() {
    static_assert(std::is_trivially_copyable_v<T>);
    T v{};
    if (pos_ + sizeof(T) > buf_.size()) { ok_ = false; return v; }
    std::memcpy(&v, buf_.data() + pos_, sizeof(T));
    pos_ += sizeof(T);
    return v;
  }
  std::string str() {
    const auto n = pod<std::uint32_t>();
    if (!ok_ || pos_ + n > buf_.size()) { ok_ = false; return {}; }
    std::string s(reinterpret_cast<const char*>(buf_.data() + pos_), n);
    pos_ += n;
    return s;
  }
  Rational rational() {
    const auto n = pod<std::int64_t>();
    const auto d = pod<std::int64_t>();
    if (!ok_ || d == 0) { ok_ = false; return Rational{}; }
    return Rational{n, d};
  }
  Units units() {
    Units u;
    u.length = pod<std::int8_t>();      u.mass = pod<std::int8_t>();
    u.time = pod<std::int8_t>();        u.current = pod<std::int8_t>();
    u.temperature = pod<std::int8_t>(); u.amount = pod<std::int8_t>();
    u.luminosity = pod<std::int8_t>();
    return u;
  }
  Window window() {
    Window w;
    w.kind = static_cast<Window::Kind>(pod<std::uint8_t>());
    w.component = ComponentId{pod<std::uint32_t>()};
    w.length = rational();
    return w;
  }
  [[nodiscard]] bool exhausted() const noexcept { return pos_ == buf_.size(); }

 private:
  std::span<const std::byte> buf_;
  std::size_t pos_ = 0;
  bool ok_ = true;
};

}  // namespace detail

/// Spec §3.2: the ABI narrow waist. C++, Fortran, Python, and YAML all
/// produce and consume this buffer.
[[nodiscard]] inline std::vector<std::byte> serialize(const CouplingGraph& g) {
  detail::Writer w;
  w.pod(kGraphMagic);
  w.pod(kGraphVersion);
  w.pod(static_cast<std::uint32_t>(g.mesh_count()));
  w.pod(static_cast<std::uint32_t>(g.region_count()));
  w.pod(static_cast<std::uint32_t>(g.component_count()));
  w.pod(static_cast<std::uint32_t>(g.field_count()));
  w.pod(static_cast<std::uint32_t>(g.channel_count()));
  w.pod(static_cast<std::uint32_t>(g.iteration_region_count()));

  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto& m = g.mesh(MeshId{i});
    w.str(m.name);
    w.str(m.source);
  }
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    w.str(r.name);
    w.pod(r.mesh.value());
    w.str(r.entity_set);
    w.pod(static_cast<std::int32_t>(r.dimension));
  }
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    w.str(c.name);
    w.pod(static_cast<std::int32_t>(c.ranks.first));
    w.pod(static_cast<std::int32_t>(c.ranks.count));
    w.rational(c.interval);
  }
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const auto& f = g.field(FieldId{i});
    w.str(f.name);
    w.pod(g.owner_of(FieldId{i}).value());
    w.pod(f.region.value());
    w.pod(static_cast<std::uint8_t>(f.disc));
    w.pod(static_cast<std::int32_t>(f.order));
    w.pod(static_cast<std::int32_t>(f.components));
    w.units(f.units);
    w.pod(static_cast<std::uint8_t>(f.conservation));
    w.pod(static_cast<std::uint8_t>(f.intent));
  }
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& c = g.channel_at(ChannelId{i});
    w.str(c.name);
    w.pod(c.source.value());
    w.pod(c.target.value());
    w.pod(static_cast<std::uint8_t>(c.accumulate.op));
    w.window(c.accumulate.window);
    w.pod(static_cast<std::uint32_t>(c.remap.flags));
    w.str(c.remap.method_override);
    w.pod(static_cast<std::uint8_t>(c.spacetime.basis));
    w.pod(static_cast<std::int32_t>(c.spacetime.degree));
    w.window(c.spacetime.window);
    w.pod(static_cast<std::uint8_t>(c.relax.enabled));
    w.pod(c.relax.omega);
    w.pod(static_cast<std::uint8_t>(c.extrapolate.enabled));
    w.pod(static_cast<std::int32_t>(c.extrapolate.order));
    w.pod(static_cast<std::int32_t>(c.lag));
    w.str(c.time_level_mismatch_reason);
  }
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const auto& r = g.iteration_region(IterationRegionIdx{i});
    w.str(r.name);
    w.pod(static_cast<std::uint32_t>(r.channels.size()));
    for (const ChannelId c : r.channels) w.pod(c.value());
    w.pod(static_cast<std::int32_t>(r.max_iterations));
  }
  return w.take();
}

[[nodiscard]] inline Result<CouplingGraph> deserialize(
    std::span<const std::byte> bytes) {
  detail::Reader r{bytes};
  if (r.pod<std::uint32_t>() != kGraphMagic || !r.ok()) {
    return Error{"E-DESC-MAGIC", "not a SOMA graph descriptor"};
  }
  const auto version = r.pod<std::uint32_t>();
  if (version != kGraphVersion) {
    return Error{"E-DESC-VERSION",
                 "descriptor version " + std::to_string(version) +
                     " but this build reads " + std::to_string(kGraphVersion)};
  }

  const auto n_mesh = r.pod<std::uint32_t>();
  const auto n_region = r.pod<std::uint32_t>();
  const auto n_comp = r.pod<std::uint32_t>();
  const auto n_field = r.pod<std::uint32_t>();
  const auto n_chan = r.pod<std::uint32_t>();
  const auto n_iter = r.pod<std::uint32_t>();
  if (!r.ok()) return Error{"E-DESC-TRUNC", "truncated descriptor header"};

  CouplingGraph g;
  for (std::uint32_t i = 0; i < n_mesh; ++i) {
    MeshDecl m;
    m.name = r.str();
    m.source = r.str();
    g.add_mesh(std::move(m));
  }
  for (std::uint32_t i = 0; i < n_region; ++i) {
    Region reg;
    reg.name = r.str();
    reg.mesh = MeshId{r.pod<std::uint32_t>()};
    reg.entity_set = r.str();
    reg.dimension = r.pod<std::int32_t>();
    g.add_region(std::move(reg));
  }
  for (std::uint32_t i = 0; i < n_comp; ++i) {
    ComponentDecl c;
    c.name = r.str();
    c.ranks.first = r.pod<std::int32_t>();
    c.ranks.count = r.pod<std::int32_t>();
    c.interval = r.rational();
    g.add_component(std::move(c));
  }
  for (std::uint32_t i = 0; i < n_field; ++i) {
    FieldSpec f;
    f.name = r.str();
    const ComponentId owner{r.pod<std::uint32_t>()};
    f.region = RegionId{r.pod<std::uint32_t>()};
    f.disc = static_cast<Discretization>(r.pod<std::uint8_t>());
    f.order = r.pod<std::int32_t>();
    f.components = r.pod<std::int32_t>();
    f.units = r.units();
    f.conservation = static_cast<Conservation>(r.pod<std::uint8_t>());
    f.intent = static_cast<Intent>(r.pod<std::uint8_t>());
    if (!r.ok() || owner.value() >= n_comp) {
      return Error{"E-DESC-TRUNC", "truncated or invalid field table"};
    }
    g.add_field(owner, std::move(f));
  }
  for (std::uint32_t i = 0; i < n_chan; ++i) {
    const std::string name = r.str();
    const FieldId src{r.pod<std::uint32_t>()};
    const FieldId tgt{r.pod<std::uint32_t>()};
    if (!r.ok()) return Error{"E-DESC-TRUNC", "truncated channel table"};
    const ChannelId cid = g.add_channel(name, src, tgt);
    ChannelBuilder b{const_cast<Channel&>(g.channel_at(cid))};
    const auto op = static_cast<AccumOp>(r.pod<std::uint8_t>());
    const Window aw = r.window();
    b.accumulate(op, aw);
    b.remap(static_cast<RemapFlags>(r.pod<std::uint32_t>()));
    b.remap_method(r.str());
    const auto basis = static_cast<TemporalBasis>(r.pod<std::uint8_t>());
    const auto degree = r.pod<std::int32_t>();
    const Window sw = r.window();
    b.spacetime(basis, degree, sw);
    const bool relax_on = r.pod<std::uint8_t>() != 0;
    const double omega = r.pod<double>();
    if (relax_on) b.relax(omega);
    const bool extrap_on = r.pod<std::uint8_t>() != 0;
    const auto extrap_order = r.pod<std::int32_t>();
    if (extrap_on) b.extrapolate(extrap_order);
    b.lag(r.pod<std::int32_t>());
    const std::string reason = r.str();
    if (!reason.empty()) b.allow_time_level_mismatch(reason);
  }
  for (std::uint32_t i = 0; i < n_iter; ++i) {
    IterationRegionDecl d;
    d.name = r.str();
    const auto n = r.pod<std::uint32_t>();
    if (!r.ok()) return Error{"E-DESC-TRUNC", "truncated iteration region"};
    d.channels.reserve(n);
    for (std::uint32_t k = 0; k < n; ++k) {
      d.channels.push_back(ChannelId{r.pod<std::uint32_t>()});
    }
    d.max_iterations = r.pod<std::int32_t>();
    g.add_iteration_region(std::move(d));
  }

  if (!r.ok()) return Error{"E-DESC-TRUNC", "truncated descriptor body"};
  if (!r.exhausted()) {
    return Error{"E-DESC-TRAIL", "trailing bytes after descriptor"};
  }
  return g;
}

/// Compares every field the descriptor carries. Used by round-trip tests and
/// by the YAML round-trip in Task 1.7.
[[nodiscard]] inline bool structurally_equal(const CouplingGraph& a,
                                             const CouplingGraph& b) {
  return serialize(a) == serialize(b);
}

}  // namespace soma
```

Note on `const_cast`: `channel_at` returns const, but the descriptor reader must
populate a channel it just created. Rather than widen `CouplingGraph`'s public
surface for one caller, the reader casts. If a later task needs mutable channel
access elsewhere, add a proper `Channel& mutable_channel(ChannelId)` and drop
the cast.

- [ ] **Step 4: Register and run**

Add `soma_add_test(descriptor_test unit/descriptor_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Descriptor
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/graph/descriptor.hpp test/unit/descriptor_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): flat graph descriptor as the ABI narrow waist"
```

---

### Task 1.7: YAML round-trip

yaml-cpp is a compiled dependency and must not appear in a public header, so the
implementation lives in `src/graph/yaml_io.cpp` and this task introduces the
first compiled target, `soma_core`.

**Files:**
- Create: `include/soma/graph/yaml_io.hpp`
- Create: `src/graph/yaml_io.cpp`
- Create: `src/CMakeLists.txt`
- Modify: `CMakeLists.txt` (add `add_subdirectory(src)`)
- Modify: `test/CMakeLists.txt` (link `soma_core`)
- Create: `test/unit/yaml_io_test.cpp`
- Create: `test/data/pair.yaml`

**Interfaces:**
- Consumes: `CouplingGraph` (1.4), `Result` (0.2), `structurally_equal` (1.6).
- Produces:
  - `Result<std::string> to_yaml(const CouplingGraph&)`
  - `Result<CouplingGraph> from_yaml(std::string_view)`
  - `Result<CouplingGraph> load_yaml_file(const std::string& path)`
  - New CMake target `soma_core` (static or shared per `BUILD_SHARED_LIBS`), linking yaml-cpp `PRIVATE` so it never leaks to consumers.

  YAML schema (this exact shape; the reader rejects unknown keys):

```yaml
version: 1
meshes:
  - {name: fluid_mesh, source: "generated:box:8x8x8"}
regions:
  - {name: fluid_wall, mesh: fluid_mesh, entity_set: wall, dimension: 2}
components:
  - name: fluid
    ranks: {first: 0, count: 2}
    interval: 1/2
    fields:
      - {name: heat_flux, region: fluid_wall, disc: FV, components: 1,
         units: W/m^2, conservation: Extensive, intent: Provides}
channels:
  - name: flux
    source: fluid.heat_flux
    target: solid.q_wall
    accumulate: {op: Mean, window: coupling_period}
    remap: {flags: [Conservative, Monotone]}
    relax: 0.7
    lag: 0
iteration_regions:
  - {name: outer, channels: [flux], max_iterations: 30}
```

  Names, not indices, are the YAML-level references (`fluid.heat_flux`); the
  reader resolves them to ids and reports `E-YAML-REF` when resolution fails.
  `interval` accepts `"1/2"` or a plain integer. `units` accepts the symbolic
  names in `soma::units` (`W/m^2`, `K`, `1`, …) — the reader maps a fixed table
  and reports `E-YAML-UNITS` for anything else.

- [ ] **Step 1: Write the failing test**

`test/unit/yaml_io_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/graph/descriptor.hpp>
#include <soma/graph/yaml_io.hpp>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshDecl;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
namespace u = soma::units;

namespace {

CouplingGraph pair_graph() {
  CouplingGraph g;
  const auto mf = g.add_mesh(MeshDecl{.name = "fluid_mesh",
                                      .source = "generated:box:8x8x8"});
  const auto ms = g.add_mesh(MeshDecl{.name = "solid_mesh",
                                      .source = "generated:box:5x5x5"});
  const auto rf = g.add_region(Region{.name = "fluid_wall", .mesh = mf,
                                      .entity_set = "wall", .dimension = 2});
  const auto rs = g.add_region(Region{.name = "solid_wall", .mesh = ms,
                                      .entity_set = "wall", .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{
      .name = "fluid", .ranks = {.first = 0, .count = 2},
      .interval = Rational{1, 2}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = {.first = 2, .count = 2},
      .interval = Rational{1}});
  const auto s = g.add_field(fluid, FieldSpec{
      .name = "heat_flux", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto t = g.add_field(solid, FieldSpec{
      .name = "q_wall", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  const auto ch = g.add_channel("flux", s, t);
  g.channel(ch)
      .accumulate(AccumOp::Mean)
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .relax(0.7);
  return g;
}

}  // namespace

TEST(YamlIo, RoundTripsThroughText) {
  const CouplingGraph original = pair_graph();
  const auto text = to_yaml(original);
  ASSERT_TRUE(text.ok()) << text.error().message;
  const auto restored = from_yaml(text.value());
  ASSERT_TRUE(restored.ok()) << restored.error().message;
  EXPECT_TRUE(structurally_equal(original, restored.value()));
}

TEST(YamlIo, EmittedTextContainsReadableNames) {
  const auto text = to_yaml(pair_graph()).value();
  EXPECT_NE(text.find("fluid_mesh"), std::string::npos);
  EXPECT_NE(text.find("fluid.heat_flux"), std::string::npos);
  EXPECT_NE(text.find("1/2"), std::string::npos);
}

TEST(YamlIo, ReadsTheCheckedInFixture) {
  const auto g = soma::load_yaml_file(SOMA_TEST_DATA_DIR "/pair.yaml");
  ASSERT_TRUE(g.ok()) << g.error().message;
  EXPECT_EQ(g.value().component_count(), 2u);
  EXPECT_EQ(g.value().channel_count(), 1u);
  EXPECT_TRUE(structurally_equal(pair_graph(), g.value()));
}

TEST(YamlIo, RejectsUnknownFieldReference) {
  const std::string bad = R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, intent: Provides}]
channels:
  - {name: c, source: a.f, target: b.nope}
)";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-REF");
}

TEST(YamlIo, RejectsUnknownUnitSymbol) {
  const std::string bad = R"(
version: 1
meshes: [{name: m, source: "generated:box"}]
regions: [{name: r, mesh: m, entity_set: w, dimension: 2}]
components:
  - name: a
    ranks: {first: 0, count: 1}
    interval: 1
    fields: [{name: f, region: r, units: furlongs, intent: Provides}]
channels: []
)";
  const auto g = from_yaml(bad);
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-UNITS");
}

TEST(YamlIo, RejectsWrongSchemaVersion) {
  const auto g = from_yaml("version: 99\nmeshes: []\n");
  ASSERT_FALSE(g.ok());
  EXPECT_EQ(g.error().code, "E-YAML-VERSION");
}

TEST(YamlIo, ParsesRationalIntervalExactly) {
  const auto g = soma::load_yaml_file(SOMA_TEST_DATA_DIR "/pair.yaml").value();
  EXPECT_EQ(g.component(g.find_component("fluid").value()).interval,
            Rational(1, 2));
}
```

- [ ] **Step 2: Write the fixture**

`test/data/pair.yaml` — must deserialise structurally equal to `pair_graph()`:

```yaml
version: 1
meshes:
  - {name: fluid_mesh, source: "generated:box:8x8x8"}
  - {name: solid_mesh, source: "generated:box:5x5x5"}
regions:
  - {name: fluid_wall, mesh: fluid_mesh, entity_set: wall, dimension: 2}
  - {name: solid_wall, mesh: solid_mesh, entity_set: wall, dimension: 2}
components:
  - name: fluid
    ranks: {first: 0, count: 2}
    interval: 1/2
    fields:
      - {name: heat_flux, region: fluid_wall, disc: FV, components: 1,
         units: W/m^2, conservation: Extensive, intent: Provides}
  - name: solid
    ranks: {first: 2, count: 2}
    interval: 1
    fields:
      - {name: q_wall, region: solid_wall, disc: FV, components: 1,
         units: W/m^2, conservation: Extensive, intent: Requires}
channels:
  - name: flux
    source: fluid.heat_flux
    target: solid.q_wall
    accumulate: {op: Mean, window: coupling_period}
    remap: {flags: [Conservative, Monotone]}
    relax: 0.7
iteration_regions: []
```

- [ ] **Step 3: Run to verify it fails**

Expected: FAIL — `soma/graph/yaml_io.hpp` file not found.

- [ ] **Step 4: Write the public header**

`include/soma/graph/yaml_io.hpp` — note it names no yaml-cpp type:

```cpp
#pragma once

#include <string>
#include <string_view>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

/// Emit the graph as YAML. Names, not indices, are used for references.
[[nodiscard]] Result<std::string> to_yaml(const CouplingGraph& g);

/// Parse YAML into a graph. Reference and unit resolution failures are
/// reported as E-YAML-REF / E-YAML-UNITS.
[[nodiscard]] Result<CouplingGraph> from_yaml(std::string_view text);

[[nodiscard]] Result<CouplingGraph> load_yaml_file(const std::string& path);

}  // namespace soma
```

- [ ] **Step 5: Write `src/CMakeLists.txt`**

```cmake
include(FetchContent)
FetchContent_Declare(yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG        0.8.0
  FIND_PACKAGE_ARGS NAMES yaml-cpp)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

add_library(soma_core
  graph/yaml_io.cpp)

target_link_libraries(soma_core
  PUBLIC  soma
  PRIVATE yaml-cpp::yaml-cpp)   # PRIVATE: never leaks into consumers

target_compile_options(soma_core PRIVATE -Wall -Wextra -Wpedantic)
```

Add to the top-level `CMakeLists.txt`, immediately before the `if(SOMA_ENABLE_TESTS)` block:

```cmake
add_subdirectory(src)
```

- [ ] **Step 6: Write `src/graph/yaml_io.cpp`**

```cpp
#include "soma/graph/yaml_io.hpp"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <sstream>
#include <unordered_map>

#include "soma/graph/descriptor.hpp"

namespace soma {
namespace {

constexpr int kSchemaVersion = 1;

// --- symbol tables ---------------------------------------------------------

struct UnitEntry { const char* symbol; Units value; };

const UnitEntry kUnitTable[] = {
    {"1", units::kDimensionless}, {"m", units::kMetre},
    {"kg", units::kKilogram},     {"s", units::kSecond},
    {"K", units::kKelvin},        {"N", units::kNewton},
    {"J", units::kJoule},         {"W", units::kWatt},
    {"Pa", units::kPascal},       {"W/m^2", units::kHeatFluxDensity},
};

bool lookup_units(const std::string& sym, Units& out) {
  for (const auto& e : kUnitTable) {
    if (sym == e.symbol) { out = e.value; return true; }
  }
  return false;
}

std::string units_symbol(const Units& u) {
  for (const auto& e : kUnitTable) {
    if (u == e.value) return e.symbol;
  }
  return u.to_string();  // falls back to the exponent form
}

template <class E, std::size_t N>
bool lookup_enum(const std::string& s, const char* const (&names)[N], E& out) {
  for (std::size_t i = 0; i < N; ++i) {
    if (s == names[i]) { out = static_cast<E>(i); return true; }
  }
  return false;
}

const char* const kDiscNames[]    = {"FV", "cGLL", "dGLL", "PointCloud"};
const char* const kConservNames[] = {"None", "Extensive", "Intensive"};
const char* const kIntentNames[]  = {"Provides", "Requires", "Bidirectional"};
const char* const kAccumNames[]   = {"Instant", "Sum", "Mean", "Min", "Max",
                                     "Integral"};
const char* const kBasisNames[]   = {"None", "Constant", "Linear", "DG",
                                     "Spline"};

// --- rational parsing ------------------------------------------------------

bool parse_rational(const std::string& s, Rational& out) {
  const auto slash = s.find('/');
  try {
    if (slash == std::string::npos) {
      out = Rational{std::stoll(s)};
    } else {
      const auto n = std::stoll(s.substr(0, slash));
      const auto d = std::stoll(s.substr(slash + 1));
      if (d == 0) return false;
      out = Rational{n, d};
    }
  } catch (...) {
    return false;
  }
  return true;
}

}  // namespace

Result<std::string> to_yaml(const CouplingGraph& g) {
  YAML::Emitter e;
  e << YAML::BeginMap;
  e << YAML::Key << "version" << YAML::Value << kSchemaVersion;

  e << YAML::Key << "meshes" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
    const auto& m = g.mesh(MeshId{i});
    e << YAML::BeginMap << YAML::Key << "name" << YAML::Value << m.name
      << YAML::Key << "source" << YAML::Value << m.source << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "regions" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.region_count(); ++i) {
    const auto& r = g.region(RegionId{i});
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << r.name
      << YAML::Key << "mesh" << YAML::Value << g.mesh(r.mesh).name
      << YAML::Key << "entity_set" << YAML::Value << r.entity_set
      << YAML::Key << "dimension" << YAML::Value << r.dimension
      << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "components" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const ComponentId cid{i};
    const auto& c = g.component(cid);
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "ranks" << YAML::Value << YAML::Flow << YAML::BeginMap
      << YAML::Key << "first" << YAML::Value << c.ranks.first
      << YAML::Key << "count" << YAML::Value << c.ranks.count << YAML::EndMap
      << YAML::Key << "interval" << YAML::Value << c.interval.to_string()
      << YAML::Key << "fields" << YAML::Value << YAML::BeginSeq;
    for (const FieldId fid : c.fields) {
      const auto& f = g.field(fid);
      e << YAML::BeginMap
        << YAML::Key << "name" << YAML::Value << f.name
        << YAML::Key << "region" << YAML::Value << g.region(f.region).name
        << YAML::Key << "disc" << YAML::Value << to_cstring(f.disc)
        << YAML::Key << "order" << YAML::Value << f.order
        << YAML::Key << "components" << YAML::Value << f.components
        << YAML::Key << "units" << YAML::Value << units_symbol(f.units)
        << YAML::Key << "conservation" << YAML::Value << to_cstring(f.conservation)
        << YAML::Key << "intent" << YAML::Value << to_cstring(f.intent)
        << YAML::EndMap;
    }
    e << YAML::EndSeq << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "channels" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const auto& c = g.channel_at(ChannelId{i});
    const auto qualified = [&](FieldId f) {
      return g.component(g.owner_of(f)).name + "." + g.field(f).name;
    };
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << c.name
      << YAML::Key << "source" << YAML::Value << qualified(c.source)
      << YAML::Key << "target" << YAML::Value << qualified(c.target)
      << YAML::Key << "accumulate" << YAML::Value << YAML::Flow << YAML::BeginMap
      << YAML::Key << "op" << YAML::Value << to_cstring(c.accumulate.op)
      << YAML::EndMap;

    if (c.remap.flags != RemapFlags::None) {
      e << YAML::Key << "remap" << YAML::Value << YAML::BeginMap
        << YAML::Key << "flags" << YAML::Value << YAML::Flow << YAML::BeginSeq;
      if (has(c.remap.flags, RemapFlags::Conservative)) e << "Conservative";
      if (has(c.remap.flags, RemapFlags::Monotone))     e << "Monotone";
      if (has(c.remap.flags, RemapFlags::HighOrder))    e << "HighOrder";
      e << YAML::EndSeq << YAML::EndMap;
    }
    if (c.spacetime.basis != TemporalBasis::None) {
      e << YAML::Key << "spacetime" << YAML::Value << YAML::Flow << YAML::BeginMap
        << YAML::Key << "basis" << YAML::Value << to_cstring(c.spacetime.basis)
        << YAML::Key << "degree" << YAML::Value << c.spacetime.degree
        << YAML::EndMap;
    }
    if (c.relax.enabled) e << YAML::Key << "relax" << YAML::Value << c.relax.omega;
    if (c.extrapolate.enabled) {
      e << YAML::Key << "extrapolate" << YAML::Value << c.extrapolate.order;
    }
    if (c.lag != 0) e << YAML::Key << "lag" << YAML::Value << c.lag;
    if (!c.time_level_mismatch_reason.empty()) {
      e << YAML::Key << "allow_time_level_mismatch" << YAML::Value
        << c.time_level_mismatch_reason;
    }
    e << YAML::EndMap;
  }
  e << YAML::EndSeq;

  e << YAML::Key << "iteration_regions" << YAML::Value << YAML::BeginSeq;
  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const auto& r = g.iteration_region(IterationRegionIdx{i});
    e << YAML::BeginMap
      << YAML::Key << "name" << YAML::Value << r.name
      << YAML::Key << "channels" << YAML::Value << YAML::Flow << YAML::BeginSeq;
    for (const ChannelId c : r.channels) e << g.channel_at(c).name;
    e << YAML::EndSeq
      << YAML::Key << "max_iterations" << YAML::Value << r.max_iterations
      << YAML::EndMap;
  }
  e << YAML::EndSeq << YAML::EndMap;

  if (!e.good()) return Error{"E-YAML-EMIT", e.GetLastError()};
  return std::string{e.c_str()};
}

Result<CouplingGraph> from_yaml(std::string_view text) {
  YAML::Node root;
  try {
    root = YAML::Load(std::string{text});
  } catch (const YAML::Exception& ex) {
    return Error{"E-YAML-PARSE", ex.what()};
  }

  if (!root["version"] || root["version"].as<int>() != kSchemaVersion) {
    return Error{"E-YAML-VERSION",
                 "expected schema version " + std::to_string(kSchemaVersion)};
  }

  CouplingGraph g;
  std::unordered_map<std::string, MeshId>     mesh_by_name;
  std::unordered_map<std::string, RegionId>   region_by_name;
  std::unordered_map<std::string, FieldId>    field_by_qualified;
  std::unordered_map<std::string, ChannelId>  channel_by_name;

  for (const auto& n : root["meshes"]) {
    const auto name = n["name"].as<std::string>();
    mesh_by_name[name] = g.add_mesh(
        MeshDecl{.name = name,
                 .source = n["source"] ? n["source"].as<std::string>() : ""});
  }

  for (const auto& n : root["regions"]) {
    const auto mesh_name = n["mesh"].as<std::string>();
    const auto it = mesh_by_name.find(mesh_name);
    if (it == mesh_by_name.end()) {
      return Error{"E-YAML-REF", "unknown mesh '" + mesh_name + "'"};
    }
    const auto name = n["name"].as<std::string>();
    region_by_name[name] = g.add_region(Region{
        .name = name,
        .mesh = it->second,
        .entity_set = n["entity_set"] ? n["entity_set"].as<std::string>() : "",
        .dimension = n["dimension"] ? n["dimension"].as<int>() : -1});
  }

  for (const auto& n : root["components"]) {
    ComponentDecl c;
    c.name = n["name"].as<std::string>();
    if (n["ranks"]) {
      c.ranks.first = n["ranks"]["first"].as<int>();
      c.ranks.count = n["ranks"]["count"].as<int>();
    }
    if (n["interval"]) {
      if (!parse_rational(n["interval"].as<std::string>(), c.interval)) {
        return Error{"E-YAML-PARSE",
                     "bad interval for component '" + c.name + "'"};
      }
    }
    const ComponentId cid = g.add_component(c);

    for (const auto& fn : n["fields"]) {
      FieldSpec f;
      f.name = fn["name"].as<std::string>();
      const auto rname = fn["region"].as<std::string>();
      const auto rit = region_by_name.find(rname);
      if (rit == region_by_name.end()) {
        return Error{"E-YAML-REF", "unknown region '" + rname + "'"};
      }
      f.region = rit->second;
      if (fn["disc"] &&
          !lookup_enum(fn["disc"].as<std::string>(), kDiscNames, f.disc)) {
        return Error{"E-YAML-PARSE",
                     "unknown discretization '" + fn["disc"].as<std::string>() + "'"};
      }
      if (fn["order"])      f.order = fn["order"].as<int>();
      if (fn["components"]) f.components = fn["components"].as<int>();
      if (fn["units"] && !lookup_units(fn["units"].as<std::string>(), f.units)) {
        return Error{"E-YAML-UNITS",
                     "unknown unit symbol '" + fn["units"].as<std::string>() + "'"};
      }
      if (fn["conservation"] &&
          !lookup_enum(fn["conservation"].as<std::string>(), kConservNames,
                       f.conservation)) {
        return Error{"E-YAML-PARSE", "unknown conservation class"};
      }
      if (fn["intent"] &&
          !lookup_enum(fn["intent"].as<std::string>(), kIntentNames, f.intent)) {
        return Error{"E-YAML-PARSE", "unknown intent"};
      }
      field_by_qualified[c.name + "." + f.name] = g.add_field(cid, f);
    }
  }

  for (const auto& n : root["channels"]) {
    const auto name = n["name"].as<std::string>();
    const auto src_name = n["source"].as<std::string>();
    const auto tgt_name = n["target"].as<std::string>();
    const auto sit = field_by_qualified.find(src_name);
    const auto tit = field_by_qualified.find(tgt_name);
    if (sit == field_by_qualified.end()) {
      return Error{"E-YAML-REF", "unknown source field '" + src_name + "'"};
    }
    if (tit == field_by_qualified.end()) {
      return Error{"E-YAML-REF", "unknown target field '" + tgt_name + "'"};
    }
    const ChannelId cid = g.add_channel(name, sit->second, tit->second);
    channel_by_name[name] = cid;
    auto b = g.channel(cid);

    if (n["accumulate"]) {
      AccumOp op = AccumOp::Instant;
      if (n["accumulate"]["op"] &&
          !lookup_enum(n["accumulate"]["op"].as<std::string>(), kAccumNames, op)) {
        return Error{"E-YAML-PARSE", "unknown accumulate op"};
      }
      b.accumulate(op, Window::coupling_period());
    }
    if (n["remap"] && n["remap"]["flags"]) {
      RemapFlags flags = RemapFlags::None;
      for (const auto& f : n["remap"]["flags"]) {
        const auto s = f.as<std::string>();
        if (s == "Conservative")   flags = flags | RemapFlags::Conservative;
        else if (s == "Monotone")  flags = flags | RemapFlags::Monotone;
        else if (s == "HighOrder") flags = flags | RemapFlags::HighOrder;
        else return Error{"E-YAML-PARSE", "unknown remap flag '" + s + "'"};
      }
      b.remap(flags);
    }
    if (n["remap"] && n["remap"]["method"]) {
      b.remap_method(n["remap"]["method"].as<std::string>());
    }
    if (n["spacetime"]) {
      TemporalBasis basis = TemporalBasis::None;
      if (!lookup_enum(n["spacetime"]["basis"].as<std::string>(), kBasisNames,
                       basis)) {
        return Error{"E-YAML-PARSE", "unknown temporal basis"};
      }
      const int degree =
          n["spacetime"]["degree"] ? n["spacetime"]["degree"].as<int>() : 0;
      b.spacetime(basis, degree, Window::coupling_period());
    }
    if (n["relax"])       b.relax(n["relax"].as<double>());
    if (n["extrapolate"]) b.extrapolate(n["extrapolate"].as<int>());
    if (n["lag"])         b.lag(n["lag"].as<int>());
    if (n["allow_time_level_mismatch"]) {
      b.allow_time_level_mismatch(
          n["allow_time_level_mismatch"].as<std::string>());
    }
  }

  for (const auto& n : root["iteration_regions"]) {
    IterationRegionDecl d;
    d.name = n["name"].as<std::string>();
    for (const auto& c : n["channels"]) {
      const auto cname = c.as<std::string>();
      const auto it = channel_by_name.find(cname);
      if (it == channel_by_name.end()) {
        return Error{"E-YAML-REF", "unknown channel '" + cname + "'"};
      }
      d.channels.push_back(it->second);
    }
    if (n["max_iterations"]) d.max_iterations = n["max_iterations"].as<int>();
    g.add_iteration_region(std::move(d));
  }

  return g;
}

Result<CouplingGraph> load_yaml_file(const std::string& path) {
  std::ifstream in{path};
  if (!in) return Error{"E-YAML-IO", "cannot open '" + path + "'"};
  std::ostringstream ss;
  ss << in.rdbuf();
  return from_yaml(ss.str());
}

}  // namespace soma
```

- [ ] **Step 7: Update the test CMakeLists**

Replace the `soma_add_test` function body so tests link `soma_core` and learn
where the data directory is:

```cmake
function(soma_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE soma soma_core GTest::gtest_main)
  target_compile_options(${name} PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_definitions(${name} PRIVATE
    SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
  gtest_discover_tests(${name})
endfunction()
```

Add `soma_add_test(yaml_io_test unit/yaml_io_test.cpp)`.

- [ ] **Step 8: Build and run**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build -j
ctest --test-dir build --output-on-failure -R YamlIo
```

Expected: 7 tests PASS. If `RoundTripsThroughText` fails, print both YAML texts
and diff them — the usual cause is an emitter/reader disagreement on a field
that `structurally_equal` compares but one side drops.

- [ ] **Step 9: Verify no yaml-cpp leakage**

```bash
grep -rn "yaml-cpp\|yaml_cpp\|YAML::" include/ && echo "LEAK" || echo "clean"
```

Expected: `clean`. A hit here violates the Global Constraints dependency rule.

- [ ] **Step 10: Commit**

```bash
git add include/soma/graph/yaml_io.hpp src/ test/unit/yaml_io_test.cpp \
        test/data/pair.yaml test/CMakeLists.txt CMakeLists.txt
git commit -m "feat(soma): YAML front-end round-tripping through the graph"
```

---

## Phase 1 exit criteria

- [ ] A two-component coupled setup can be authored in C++, validated, serialised, written to YAML, read back, and compared structurally equal.
- [ ] `validate()` reports every problem in one pass, never aborting on the first.
- [ ] `grep -rn "YAML::" include/` finds nothing — yaml-cpp is confined to `src/`.
- [ ] The core still configures and builds with **no** optional dependency (yaml-cpp is fetched, not optional, and is the only compiled dependency so far).
- [ ] `ctest --test-dir build --output-on-failure` fully green.
