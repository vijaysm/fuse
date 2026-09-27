# SOMA Phase 2 — OperatorDAG, lowering registry, golden-IR tests

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §2 (layer 2), §2.1 (why ordering must be explicit), §4.3 (time-level typing), §5.1 (primitive operators), §5.2 (lowering), §5.3 (iteration regions), §10 level 2 (golden IR)
**Milestone:** every channel in the Phase 1 pair graph lowers to an explicit operator DAG whose text dump is byte-identical to a checked-in golden file.

This is the architectural core. After this phase, `Remap∘Accumulate` and `Accumulate∘Remap` are visibly different DAGs, and a unit mismatch or a staggering error is a lowering-time diagnostic.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/ir/op.hpp` | `OpKind`, `OpAttrs`, `Node`, `Arc`, `ValueType` |
| `include/soma/ir/dag.hpp` | `OperatorDAG` — SSA container, regions, topological order |
| `include/soma/ir/dump.hpp` | text and Graphviz dumps (the golden-file format) |
| `include/soma/ir/lowering.hpp` | `LoweringRule`, `LoweringRegistry`, `lower()` |
| `src/ir/rules_channel.cpp` | the built-in channel lowering rule |
| `src/ir/rules_derived.cpp` | Transport / Constrain / Diagnose insertion |
| `src/ir/lowering.cpp` | driver: walks the graph, applies rules, type-checks |
| `test/unit/*_test.cpp` | per-header unit tests |
| `test/golden/*.ir` | checked-in expected dumps |

---

### Task 2.1: `OpKind`, `Node`, `Arc`, and `ValueType`

Spec §5.1's twelve primitives, plus the arc type that carries time levels.

**Files:**
- Create: `include/soma/ir/op.hpp`
- Test: `test/unit/ir_op_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `NodeId`, `ArcId`, `FieldId`, `RegionId`, `ChannelId`, `AccumOp`, `TemporalBasis` (0.6); `TimeLevel` (0.5); `Units` (0.4); `Rational` (0.3).
- Produces:
  - `enum class OpKind { Sample, Accumulate, Interpolate, Extrapolate, Remap, Merge, Relax, Transport, Residual, Accelerate, Constrain, Diagnose, Source, Sink }`
    — `Source` and `Sink` are the DAG's boundary nodes: `Source` reads a registered component field, `Sink` writes one. Spec §5.1 lists twelve *interior* primitives; the boundary needs two more so every arc has a producer and every value a consumer.
  - `constexpr const char* to_cstring(OpKind)`
  - `enum class ValueKind { Instant, Waveform }` — `Waveform` arcs carry several samples (Phase 6 uses this; Phase 2 only needs the distinction to exist so `Sample`'s output type is honest).
  - `struct ValueType { ValueKind kind = ValueKind::Instant; RegionId region; int components = 1; Units units{}; Conservation conservation = Conservation::None; TimeLevel time; int rank_set = -1; }` with defaulted `==`, `std::string to_string() const`, and `bool assignable_from(const ValueType& producer, const ValueType& consumer, bool allow_time_mismatch)`.
    - `rank_set` is the index of the `RankSet` the value physically lives on: the owning component's index, or `-1` for "not yet placed". Task 2.6 uses it to decide where a `Transport` is needed.
  - `struct OpAttrs { AccumOp accum = AccumOp::Instant; TemporalBasis basis = TemporalBasis::None; int degree = 0; double omega = 1.0; int order = 0; int lag = 0; RemapFlags remap_flags = RemapFlags::None; std::string method; int src_rank_set = -1; int dst_rank_set = -1; }` — one flat attribute bag; each `OpKind` reads only the members it needs, which keeps `Node` trivially copyable and the dump format stable.
  - `struct Node { NodeId id; OpKind kind; std::vector<ArcId> inputs; ArcId output; OpAttrs attrs; ChannelId origin; std::string label; }` — `origin` records which channel produced this node, so a diagnostic can name it.
  - `struct Arc { ArcId id; NodeId producer; ValueType type; std::string name; }`

- [ ] **Step 1: Write the failing test**

`test/unit/ir_op_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/op.hpp>

using soma::Anchor;
using soma::assignable_from;
using soma::Conservation;
using soma::OpKind;
using soma::Rational;
using soma::RegionId;
using soma::TimeLevel;
using soma::ValueKind;
using soma::ValueType;
namespace u = soma::units;

TEST(OpKind, EveryKindHasAName) {
  EXPECT_STREQ(to_cstring(OpKind::Sample), "Sample");
  EXPECT_STREQ(to_cstring(OpKind::Accumulate), "Accumulate");
  EXPECT_STREQ(to_cstring(OpKind::Interpolate), "Interpolate");
  EXPECT_STREQ(to_cstring(OpKind::Extrapolate), "Extrapolate");
  EXPECT_STREQ(to_cstring(OpKind::Remap), "Remap");
  EXPECT_STREQ(to_cstring(OpKind::Merge), "Merge");
  EXPECT_STREQ(to_cstring(OpKind::Relax), "Relax");
  EXPECT_STREQ(to_cstring(OpKind::Transport), "Transport");
  EXPECT_STREQ(to_cstring(OpKind::Residual), "Residual");
  EXPECT_STREQ(to_cstring(OpKind::Accelerate), "Accelerate");
  EXPECT_STREQ(to_cstring(OpKind::Constrain), "Constrain");
  EXPECT_STREQ(to_cstring(OpKind::Diagnose), "Diagnose");
  EXPECT_STREQ(to_cstring(OpKind::Source), "Source");
  EXPECT_STREQ(to_cstring(OpKind::Sink), "Sink");
}

TEST(ValueType, DefaultsToInstantScalarDimensionless) {
  const ValueType v;
  EXPECT_EQ(v.kind, ValueKind::Instant);
  EXPECT_EQ(v.components, 1);
  EXPECT_TRUE(v.units.is_dimensionless());
  EXPECT_EQ(v.rank_set, -1);
}

TEST(ValueType, AssignableWhenIdentical) {
  const ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  EXPECT_TRUE(assignable_from(a, a, /*allow_time_mismatch=*/false));
}

TEST(ValueType, NotAssignableAcrossUnits) {
  const ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  const ValueType b{.region = RegionId{0}, .units = u::kHeatFluxDensity};
  EXPECT_FALSE(assignable_from(a, b, false));
}

TEST(ValueType, NotAssignableAcrossRegions) {
  const ValueType a{.region = RegionId{0}};
  const ValueType b{.region = RegionId{1}};
  EXPECT_FALSE(assignable_from(a, b, false));
}

TEST(ValueType, NotAssignableAcrossComponentCounts) {
  const ValueType a{.region = RegionId{0}, .components = 1};
  const ValueType b{.region = RegionId{0}, .components = 3};
  EXPECT_FALSE(assignable_from(a, b, false));
}

// Spec §4.3: strict by default.
TEST(ValueType, TimeLevelMismatchBlocksAssignmentByDefault) {
  ValueType a{.region = RegionId{0}};
  ValueType b{.region = RegionId{0}};
  a.time = TimeLevel{Anchor::WindowEnd, Rational{0}, 0};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_FALSE(assignable_from(a, b, false));
}

// ... but the channel-level escape hatch relaxes exactly this one check.
TEST(ValueType, TimeLevelMismatchPermittedWhenAuthorised) {
  ValueType a{.region = RegionId{0}};
  ValueType b{.region = RegionId{0}};
  a.time = TimeLevel{Anchor::WindowEnd, Rational{0}, 0};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_TRUE(assignable_from(a, b, /*allow_time_mismatch=*/true));
}

// The escape hatch must not launder a unit error.
TEST(ValueType, TimeEscapeHatchDoesNotExcuseUnitMismatch) {
  ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  ValueType b{.region = RegionId{0}, .units = u::kWatt};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_FALSE(assignable_from(a, b, true));
}

TEST(ValueType, ToStringMentionsRegionUnitsAndTime) {
  ValueType v{.region = RegionId{7}, .components = 3, .units = u::kKelvin};
  v.time = TimeLevel::lagged(1);
  const std::string s = v.to_string();
  EXPECT_NE(s.find("r7"), std::string::npos);
  EXPECT_NE(s.find("K"), std::string::npos);
  EXPECT_NE(s.find("WindowEnd"), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/ir/op.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/ir/op.hpp`:

```cpp
#pragma once

#include <string>
#include <vector>

#include "soma/base/ids.hpp"
#include "soma/base/time_level.hpp"
#include "soma/base/units.hpp"
#include "soma/graph/channel.hpp"  // RemapFlags

namespace soma {

/// Spec §5.1's twelve primitives, plus Source/Sink boundary nodes so that
/// every arc has a producer and every value a consumer.
enum class OpKind {
  Sample, Accumulate, Interpolate, Extrapolate, Remap, Merge, Relax,
  Transport, Residual, Accelerate, Constrain, Diagnose,
  Source, Sink,
};

[[nodiscard]] constexpr const char* to_cstring(OpKind k) noexcept {
  switch (k) {
    case OpKind::Sample:      return "Sample";
    case OpKind::Accumulate:  return "Accumulate";
    case OpKind::Interpolate: return "Interpolate";
    case OpKind::Extrapolate: return "Extrapolate";
    case OpKind::Remap:       return "Remap";
    case OpKind::Merge:       return "Merge";
    case OpKind::Relax:       return "Relax";
    case OpKind::Transport:   return "Transport";
    case OpKind::Residual:    return "Residual";
    case OpKind::Accelerate:  return "Accelerate";
    case OpKind::Constrain:   return "Constrain";
    case OpKind::Diagnose:    return "Diagnose";
    case OpKind::Source:      return "Source";
    case OpKind::Sink:        return "Sink";
  }
  return "Unknown";
}

enum class ValueKind { Instant, Waveform };

[[nodiscard]] constexpr const char* to_cstring(ValueKind k) noexcept {
  return k == ValueKind::Waveform ? "Waveform" : "Instant";
}

/// The type of an SSA value. Spec §2: every arc carries its time level, so a
/// staggering error is a type error found at lowering.
struct ValueType {
  ValueKind    kind = ValueKind::Instant;
  RegionId     region;
  int          components = 1;
  Units        units{};
  Conservation conservation = Conservation::None;
  TimeLevel    time{};
  int          rank_set = -1;  // component index the value lives on; -1 = unplaced

  friend bool operator==(const ValueType&, const ValueType&) = default;

  [[nodiscard]] std::string to_string() const {
    std::string s = to_cstring(kind);
    s += "<r";
    s += region.valid() ? std::to_string(region.value()) : std::string{"?"};
    if (components != 1) {
      s += "x";
      s += std::to_string(components);
    }
    s += ", ";
    s += units.to_string();
    if (conservation != Conservation::None) {
      s += ", ";
      s += to_cstring(conservation);
    }
    s += ", ";
    s += time.to_string();
    if (rank_set >= 0) {
      s += ", ranks=";
      s += std::to_string(rank_set);
    }
    s += ">";
    return s;
  }
};

/// Can `consumer` accept a value of type `producer`?
/// allow_time_mismatch comes from Channel::time_level_mismatch_reason and
/// relaxes ONLY the time check (spec §4.3).
[[nodiscard]] inline bool assignable_from(const ValueType& producer,
                                          const ValueType& consumer,
                                          bool allow_time_mismatch) noexcept {
  if (producer.kind != consumer.kind) return false;
  if (producer.region != consumer.region) return false;
  if (producer.components != consumer.components) return false;
  if (!(producer.units == consumer.units)) return false;
  if (producer.conservation != consumer.conservation) return false;
  if (!allow_time_mismatch && !compatible(producer.time, consumer.time)) {
    return false;
  }
  return true;
}

/// Flat attribute bag. Each OpKind reads only what it needs; keeping it flat
/// keeps Node cheap to copy and the dump format stable.
struct OpAttrs {
  AccumOp       accum = AccumOp::Instant;
  TemporalBasis basis = TemporalBasis::None;
  int           degree = 0;
  double        omega = 1.0;
  int           order = 0;
  int           lag = 0;
  RemapFlags    remap_flags = RemapFlags::None;
  std::string   method;
  int           src_rank_set = -1;
  int           dst_rank_set = -1;

  friend bool operator==(const OpAttrs&, const OpAttrs&) = default;
};

struct Node {
  NodeId               id;
  OpKind               kind = OpKind::Source;
  std::vector<ArcId>   inputs;
  ArcId                output;
  OpAttrs              attrs;
  ChannelId            origin;  // which channel emitted this node
  std::string          label;
};

struct Arc {
  ArcId       id;
  NodeId      producer;
  ValueType   type;
  std::string name;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(ir_op_test unit/ir_op_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "OpKind|ValueType"
```

Expected: 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/ir/op.hpp test/unit/ir_op_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): IR operator kinds, typed arcs, attribute bag"
```

---

### Task 2.2: `OperatorDAG` container

SSA: each arc has exactly one producer. The container enforces that on
construction rather than checking it later.

**Files:**
- Create: `include/soma/ir/dag.hpp`
- Test: `test/unit/ir_dag_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Node`, `Arc`, `ValueType` (2.1); `Diagnostics` (0.2).
- Produces: `class OperatorDAG` with
  - `ArcId add_arc(ValueType type, std::string name)` — creates an arc with no producer yet
  - `NodeId add_node(OpKind kind, std::vector<ArcId> inputs, ArcId output, OpAttrs attrs, ChannelId origin, std::string label)` — asserts `output` has no producer, then records this node as its producer
  - accessors `const Node& node(NodeId) const`, `const Arc& arc(ArcId) const`, `std::size_t node_count() const`, `std::size_t arc_count() const`
  - `const std::vector<NodeId>& consumers_of(ArcId) const`
  - `std::vector<NodeId> topological_order() const` — Kahn's algorithm; deterministic (ties broken by `NodeId`) so golden dumps are stable
  - `bool is_acyclic() const`
  - `struct IterationRegion { std::string name; std::vector<NodeId> body; std::vector<ArcId> residuals; std::vector<ArcId> carried; int max_iterations = 1; };` and `RegionIdx add_region(IterationRegion)`, `const IterationRegion& region_at(RegionIdx) const`, `std::size_t region_count() const`
    - `carried` is the loop-carried state — the derived checkpoint set of spec §5.3. Phase 7 Task 7.2 computes it; Phase 2 only stores it.

  Determinism note: `topological_order()` must break ties by ascending `NodeId`, or golden files become flaky. This is tested explicitly.

- [ ] **Step 1: Write the failing test**

`test/unit/ir_dag_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/dag.hpp>

#include <algorithm>

using soma::ArcId;
using soma::ChannelId;
using soma::NodeId;
using soma::OperatorDAG;
using soma::OpAttrs;
using soma::OpKind;
using soma::RegionId;
using soma::ValueType;

namespace {

// Source -> Accumulate -> Sink
OperatorDAG linear_chain() {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  const ArcId a2 = d.add_arc(t, "v2");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "src");
  d.add_node(OpKind::Accumulate, {a0}, a1, OpAttrs{}, ChannelId{0}, "acc");
  d.add_node(OpKind::Sink, {a1}, a2, OpAttrs{}, ChannelId{}, "sink");
  return d;
}

}  // namespace

TEST(OperatorDAG, EmptyDagHasNothing) {
  const OperatorDAG d;
  EXPECT_EQ(d.node_count(), 0u);
  EXPECT_EQ(d.arc_count(), 0u);
  EXPECT_TRUE(d.is_acyclic());
}

TEST(OperatorDAG, RecordsProducerAndConsumers) {
  const OperatorDAG d = linear_chain();
  EXPECT_EQ(d.node_count(), 3u);
  EXPECT_EQ(d.arc_count(), 3u);
  // arc v0 is produced by the Source and consumed by the Accumulate
  EXPECT_EQ(d.arc(ArcId{0}).producer, NodeId{0});
  ASSERT_EQ(d.consumers_of(ArcId{0}).size(), 1u);
  EXPECT_EQ(d.consumers_of(ArcId{0})[0], NodeId{1});
}

TEST(OperatorDAG, TopologicalOrderRespectsDependencies) {
  const OperatorDAG d = linear_chain();
  const auto order = d.topological_order();
  ASSERT_EQ(order.size(), 3u);
  const auto pos = [&](NodeId n) {
    return std::find(order.begin(), order.end(), n) - order.begin();
  };
  EXPECT_LT(pos(NodeId{0}), pos(NodeId{1}));
  EXPECT_LT(pos(NodeId{1}), pos(NodeId{2}));
}

// Golden dumps depend on this: independent nodes must always come out in the
// same order.
TEST(OperatorDAG, TopologicalOrderIsDeterministic) {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a = d.add_arc(t, "a");
  const ArcId b = d.add_arc(t, "b");
  const ArcId c = d.add_arc(t, "c");
  // Three mutually independent sources.
  d.add_node(OpKind::Source, {}, a, OpAttrs{}, ChannelId{}, "s0");
  d.add_node(OpKind::Source, {}, b, OpAttrs{}, ChannelId{}, "s1");
  d.add_node(OpKind::Source, {}, c, OpAttrs{}, ChannelId{}, "s2");
  const auto first = d.topological_order();
  for (int i = 0; i < 20; ++i) EXPECT_EQ(d.topological_order(), first);
  EXPECT_EQ(first[0], NodeId{0});
  EXPECT_EQ(first[1], NodeId{1});
  EXPECT_EQ(first[2], NodeId{2});
}

TEST(OperatorDAG, DetectsCycle) {
  OperatorDAG d;
  const ValueType t{.region = RegionId{0}};
  const ArcId a = d.add_arc(t, "a");
  const ArcId b = d.add_arc(t, "b");
  d.add_node(OpKind::Relax, {b}, a, OpAttrs{}, ChannelId{}, "n0");
  d.add_node(OpKind::Relax, {a}, b, OpAttrs{}, ChannelId{}, "n1");
  EXPECT_FALSE(d.is_acyclic());
  EXPECT_LT(d.topological_order().size(), d.node_count());
}

TEST(OperatorDAG, StoresIterationRegions) {
  OperatorDAG d = linear_chain();
  const auto idx = d.add_region(OperatorDAG::IterationRegion{
      .name = "outer", .body = {NodeId{1}}, .residuals = {ArcId{1}},
      .carried = {ArcId{0}}, .max_iterations = 30});
  EXPECT_EQ(d.region_count(), 1u);
  EXPECT_EQ(d.region_at(idx).max_iterations, 30);
  EXPECT_EQ(d.region_at(idx).carried.size(), 1u);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/ir/dag.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/ir/dag.hpp`:

```cpp
#pragma once

#include <algorithm>
#include <cassert>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "soma/ir/op.hpp"

namespace soma {

/// Spec §2 layer 2. Single assignment: every arc has exactly one producer,
/// enforced at add_node time rather than checked afterwards.
class OperatorDAG {
 public:
  /// Spec §5.3. `carried` is the derived checkpoint set (Phase 7 Task 7.2).
  struct IterationRegion {
    std::string         name;
    std::vector<NodeId> body;
    std::vector<ArcId>  residuals;
    std::vector<ArcId>  carried;
    int                 max_iterations = 1;
  };

  ArcId add_arc(ValueType type, std::string name) {
    const ArcId id{static_cast<std::uint32_t>(arcs_.size())};
    arcs_.push_back(Arc{id, NodeId{}, std::move(type), std::move(name)});
    consumers_.emplace_back();
    return id;
  }

  NodeId add_node(OpKind kind, std::vector<ArcId> inputs, ArcId output,
                  OpAttrs attrs, ChannelId origin, std::string label) {
    const NodeId id{static_cast<std::uint32_t>(nodes_.size())};
    for (const ArcId in : inputs) {
      assert(in.valid() && in.value() < arcs_.size());
      consumers_[in.value()].push_back(id);
    }
    if (output.valid()) {
      assert(output.value() < arcs_.size());
      assert(!arcs_[output.value()].producer.valid() &&
             "SSA violation: arc already has a producer");
      arcs_[output.value()].producer = id;
    }
    nodes_.push_back(Node{id, kind, std::move(inputs), output, std::move(attrs),
                          origin, std::move(label)});
    return id;
  }

  [[nodiscard]] const Node& node(NodeId id) const { return nodes_.at(id.value()); }
  [[nodiscard]] const Arc&  arc(ArcId id) const { return arcs_.at(id.value()); }
  [[nodiscard]] std::size_t node_count() const noexcept { return nodes_.size(); }
  [[nodiscard]] std::size_t arc_count() const noexcept { return arcs_.size(); }

  [[nodiscard]] const std::vector<NodeId>& consumers_of(ArcId id) const {
    return consumers_.at(id.value());
  }

  /// Kahn's algorithm with ties broken by ascending NodeId, so the order is
  /// reproducible run to run — golden dumps depend on it.
  [[nodiscard]] std::vector<NodeId> topological_order() const {
    std::vector<int> indegree(nodes_.size(), 0);
    for (const auto& n : nodes_) {
      for (const ArcId in : n.inputs) {
        if (arcs_[in.value()].producer.valid()) {
          indegree[n.id.value()]++;
        }
      }
    }
    std::priority_queue<std::uint32_t, std::vector<std::uint32_t>,
                        std::greater<>> ready;
    for (std::size_t i = 0; i < nodes_.size(); ++i) {
      if (indegree[i] == 0) ready.push(static_cast<std::uint32_t>(i));
    }
    std::vector<NodeId> order;
    order.reserve(nodes_.size());
    while (!ready.empty()) {
      const std::uint32_t v = ready.top();
      ready.pop();
      order.push_back(NodeId{v});
      const ArcId out = nodes_[v].output;
      if (!out.valid()) continue;
      for (const NodeId c : consumers_[out.value()]) {
        if (--indegree[c.value()] == 0) ready.push(c.value());
      }
    }
    return order;
  }

  [[nodiscard]] bool is_acyclic() const {
    return topological_order().size() == nodes_.size();
  }

  RegionIdx add_region(IterationRegion r) {
    regions_.push_back(std::move(r));
    return RegionIdx{static_cast<std::uint32_t>(regions_.size() - 1)};
  }
  [[nodiscard]] const IterationRegion& region_at(RegionIdx i) const {
    return regions_.at(i.value());
  }
  [[nodiscard]] std::size_t region_count() const noexcept {
    return regions_.size();
  }

 private:
  std::vector<Node>                nodes_;
  std::vector<Arc>                 arcs_;
  std::vector<std::vector<NodeId>> consumers_;  // parallel to arcs_
  std::vector<IterationRegion>     regions_;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(ir_dag_test unit/ir_dag_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R OperatorDAG
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/ir/dag.hpp test/unit/ir_dag_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): OperatorDAG SSA container with deterministic topo order"
```

---

### Task 2.3: IR dump — the golden-file format

Spec §2.3 and §10 level 2. This format is a test artifact, so it must be stable
and diff-friendly: one node per line, fixed column order, no addresses, no
iteration-order dependence.

**Files:**
- Create: `include/soma/ir/dump.hpp`
- Test: `test/unit/ir_dump_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `OperatorDAG` (2.2).
- Produces:
  - `std::string dump_text(const OperatorDAG&)` — the golden format:

```
# soma-ir v1
node 0  Source      ->  %0 : Instant<r0, W·m^-2, Extensive, [WindowEnd, k], ranks=0>  "fluid.heat_flux"
node 1  Accumulate  %0 -> %1 : Instant<r0, W·m^-2, Extensive, [WindowEnd, k], ranks=0>  "flux/accumulate" {accum=Mean}
node 2  Transport   %1 -> %2 : Instant<r0, W·m^-2, Extensive, [WindowEnd, k], ranks=1>  "flux/transport" {0->1}
region 0 "outer" body=[1,2] residuals=[%1] carried=[%0] max_iter=30
```

  - `std::string dump_graphviz(const OperatorDAG&)` — `digraph` with one node per op, edges labelled by arc name.
  - Rules the format must obey (each is a test): nodes emitted in `topological_order()`; arcs printed as `%<id>`; only non-default attributes shown, in a fixed order; regions after nodes, sorted by index.

- [ ] **Step 1: Write the failing test**

`test/unit/ir_dump_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/dump.hpp>

using soma::ArcId;
using soma::ChannelId;
using soma::OperatorDAG;
using soma::OpAttrs;
using soma::OpKind;
using soma::RegionId;
using soma::ValueType;
namespace u = soma::units;

namespace {

OperatorDAG small_dag() {
  OperatorDAG d;
  ValueType t{.region = RegionId{0}, .units = u::kHeatFluxDensity,
              .conservation = soma::Conservation::Extensive, .rank_set = 0};
  const ArcId a0 = d.add_arc(t, "v0");
  const ArcId a1 = d.add_arc(t, "v1");
  d.add_node(OpKind::Source, {}, a0, OpAttrs{}, ChannelId{}, "fluid.heat_flux");
  d.add_node(OpKind::Accumulate, {a0}, a1,
             OpAttrs{.accum = soma::AccumOp::Mean}, ChannelId{0},
             "flux/accumulate");
  return d;
}

}  // namespace

TEST(IrDump, StartsWithVersionBanner) {
  const std::string s = dump_text(small_dag());
  EXPECT_EQ(s.substr(0, 12), "# soma-ir v1");
}

TEST(IrDump, EmitsOneLinePerNodeInTopologicalOrder) {
  const std::string s = dump_text(small_dag());
  const auto src = s.find("Source");
  const auto acc = s.find("Accumulate");
  ASSERT_NE(src, std::string::npos);
  ASSERT_NE(acc, std::string::npos);
  EXPECT_LT(src, acc);
}

TEST(IrDump, ShowsArcIdsAndTypes) {
  const std::string s = dump_text(small_dag());
  EXPECT_NE(s.find("%0"), std::string::npos);
  EXPECT_NE(s.find("%1"), std::string::npos);
  EXPECT_NE(s.find("Extensive"), std::string::npos);
  EXPECT_NE(s.find("ranks=0"), std::string::npos);
}

TEST(IrDump, ShowsNonDefaultAttributesOnly) {
  const std::string s = dump_text(small_dag());
  EXPECT_NE(s.find("{accum=Mean}"), std::string::npos);
  // The Source node has all-default attrs, so no brace group on its line.
  const auto line_end = s.find('\n', s.find("Source"));
  const std::string src_line = s.substr(s.find("Source"), line_end - s.find("Source"));
  EXPECT_EQ(src_line.find('{'), std::string::npos);
}

TEST(IrDump, IsByteStableAcrossCalls) {
  const OperatorDAG d = small_dag();
  EXPECT_EQ(dump_text(d), dump_text(d));
}

TEST(IrDump, EmitsRegions) {
  OperatorDAG d = small_dag();
  d.add_region(OperatorDAG::IterationRegion{
      .name = "outer", .body = {soma::NodeId{1}}, .residuals = {ArcId{1}},
      .carried = {ArcId{0}}, .max_iterations = 30});
  const std::string s = dump_text(d);
  EXPECT_NE(s.find("region 0 \"outer\""), std::string::npos);
  EXPECT_NE(s.find("max_iter=30"), std::string::npos);
}

TEST(IrDump, GraphvizIsWellFormed) {
  const std::string s = dump_graphviz(small_dag());
  EXPECT_EQ(s.substr(0, 7), "digraph");
  EXPECT_NE(s.find("->"), std::string::npos);
  EXPECT_EQ(s.back(), '\n');
  EXPECT_NE(s.find('}'), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/ir/dump.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/ir/dump.hpp`:

```cpp
#pragma once

#include <sstream>
#include <string>

#include "soma/ir/dag.hpp"

namespace soma {
namespace detail {

/// Only non-default attributes, in a fixed order, so dumps stay diff-friendly.
inline std::string attrs_string(OpKind kind, const OpAttrs& a) {
  const OpAttrs def{};
  std::string out;
  const auto add = [&out](const std::string& kv) {
    if (!out.empty()) out += ", ";
    out += kv;
  };
  if (a.accum != def.accum) add(std::string{"accum="} + to_cstring(a.accum));
  if (a.basis != def.basis) add(std::string{"basis="} + to_cstring(a.basis));
  if (a.degree != def.degree) add("degree=" + std::to_string(a.degree));
  if (a.omega != def.omega) add("omega=" + std::to_string(a.omega));
  if (a.order != def.order) add("order=" + std::to_string(a.order));
  if (a.lag != def.lag) add("lag=" + std::to_string(a.lag));
  if (a.remap_flags != def.remap_flags) {
    std::string f;
    if (has(a.remap_flags, RemapFlags::Conservative)) f += "C";
    if (has(a.remap_flags, RemapFlags::Monotone))     f += "M";
    if (has(a.remap_flags, RemapFlags::HighOrder))    f += "H";
    add("remap=" + f);
  }
  if (!a.method.empty()) add("method=" + a.method);
  if (kind == OpKind::Transport) {
    add(std::to_string(a.src_rank_set) + "->" + std::to_string(a.dst_rank_set));
  }
  return out;
}

inline std::string pad(std::string s, std::size_t width) {
  while (s.size() < width) s += ' ';
  return s;
}

}  // namespace detail

/// Spec §10 level 2: this text is checked into test/golden/ and compared
/// byte-for-byte, so a lowering change cannot silently alter the numerics.
[[nodiscard]] inline std::string dump_text(const OperatorDAG& d) {
  std::ostringstream os;
  os << "# soma-ir v1\n";

  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    os << "node " << detail::pad(std::to_string(nid.value()), 3) << " "
       << detail::pad(to_cstring(n.kind), 12) << " ";

    std::string ins;
    for (const ArcId a : n.inputs) {
      if (!ins.empty()) ins += ", ";
      ins += "%" + std::to_string(a.value());
    }
    os << detail::pad(ins, 12) << " -> ";

    if (n.output.valid()) {
      os << "%" << n.output.value() << " : "
         << d.arc(n.output).type.to_string();
    } else {
      os << "(none)";
    }

    if (!n.label.empty()) os << "  \"" << n.label << "\"";
    const std::string attrs = detail::attrs_string(n.kind, n.attrs);
    if (!attrs.empty()) os << " {" << attrs << "}";
    os << "\n";
  }

  for (std::uint32_t i = 0; i < d.region_count(); ++i) {
    const auto& r = d.region_at(RegionIdx{i});
    os << "region " << i << " \"" << r.name << "\" body=[";
    for (std::size_t k = 0; k < r.body.size(); ++k) {
      if (k) os << ",";
      os << r.body[k].value();
    }
    os << "] residuals=[";
    for (std::size_t k = 0; k < r.residuals.size(); ++k) {
      if (k) os << ",";
      os << "%" << r.residuals[k].value();
    }
    os << "] carried=[";
    for (std::size_t k = 0; k < r.carried.size(); ++k) {
      if (k) os << ",";
      os << "%" << r.carried[k].value();
    }
    os << "] max_iter=" << r.max_iterations << "\n";
  }

  return os.str();
}

[[nodiscard]] inline std::string dump_graphviz(const OperatorDAG& d) {
  std::ostringstream os;
  os << "digraph soma_ir {\n  rankdir=LR;\n  node [shape=box];\n";
  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    os << "  n" << nid.value() << " [label=\"" << to_cstring(n.kind);
    if (!n.label.empty()) os << "\\n" << n.label;
    const std::string attrs = detail::attrs_string(n.kind, n.attrs);
    if (!attrs.empty()) os << "\\n" << attrs;
    os << "\"];\n";
  }
  for (const NodeId nid : d.topological_order()) {
    const Node& n = d.node(nid);
    for (const ArcId in : n.inputs) {
      const NodeId prod = d.arc(in).producer;
      if (!prod.valid()) continue;
      os << "  n" << prod.value() << " -> n" << nid.value()
         << " [label=\"%" << in.value() << "\"];\n";
    }
  }
  os << "}\n";
  return os.str();
}

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(ir_dump_test unit/ir_dump_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R IrDump
```

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/ir/dump.hpp test/unit/ir_dump_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): stable IR text and Graphviz dumps"
```

---

### Task 2.4: Lowering registry and driver skeleton

Spec §5.2: lowering rules live in a registry keyed on (semantic, context), not a
switch. This task builds the registry and a driver that emits only `Source` and
`Sink` nodes; Task 2.5 adds the real channel rule.

**Files:**
- Create: `include/soma/ir/lowering.hpp`
- Create: `src/ir/lowering.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/lowering_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `CouplingGraph` (1.4), `OperatorDAG` (2.2), `Diagnostics` (0.2).
- Produces:
  - `struct LoweringContext { const CouplingGraph& graph; OperatorDAG& dag; Diagnostics& diags; std::unordered_map<std::uint32_t, ArcId> field_arc; }` — `field_arc` maps a `FieldId` value to the arc currently holding that field's value, which is how successive rules chain.
  - `using LoweringRule = std::function<bool(LoweringContext&, ChannelId)>` — returns `true` if the rule handled the channel.
  - `class LoweringRegistry` with `void add(std::string name, int priority, LoweringRule)`, `bool apply_first_match(LoweringContext&, ChannelId) const`, `std::size_t size() const`, and `static LoweringRegistry& builtin()` — the singleton the driver uses, populated by `register_builtin_rules()`.
    Rules are tried in descending priority; ties broken by insertion order.
  - `struct LoweringResult { OperatorDAG dag; Diagnostics diags; bool ok() const { return !diags.has_errors(); } }`
  - `LoweringResult lower(const CouplingGraph&)` — declared here, defined in `src/ir/lowering.cpp`
  - `void register_builtin_rules(LoweringRegistry&)` — defined across `src/ir/rules_channel.cpp` and `src/ir/rules_derived.cpp`

  Driver order (fixed, and tested): 1. emit a `Source` for every field with `Intent::Provides` or `Bidirectional`; 2. for each channel in id order, apply the first matching rule; 3. emit a `Sink` for every field with `Intent::Requires` or `Bidirectional` that some channel wrote; 4. run the type check over every node's inputs; 5. return.

- [ ] **Step 1: Write the failing test**

`test/unit/lowering_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>

using soma::ChannelId;
using soma::LoweringContext;
using soma::LoweringRegistry;
using soma::OpKind;

namespace {

int count_kind(const soma::OperatorDAG& d, OpKind k) {
  int n = 0;
  for (std::uint32_t i = 0; i < d.node_count(); ++i) {
    if (d.node(soma::NodeId{i}).kind == k) ++n;
  }
  return n;
}

}  // namespace

TEST(LoweringRegistry, TriesRulesInPriorityOrder) {
  LoweringRegistry reg;
  std::string log;
  reg.add("low", 1, [&](LoweringContext&, ChannelId) {
    log += "low ";
    return false;   // decline
  });
  reg.add("high", 10, [&](LoweringContext&, ChannelId) {
    log += "high ";
    return true;    // handle
  });

  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}};
  EXPECT_TRUE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(log, "high ");   // high ran first and claimed it
}

TEST(LoweringRegistry, FallsThroughWhenRuleDeclines) {
  LoweringRegistry reg;
  std::string log;
  reg.add("first", 10, [&](LoweringContext&, ChannelId) {
    log += "first ";
    return false;
  });
  reg.add("second", 1, [&](LoweringContext&, ChannelId) {
    log += "second ";
    return true;
  });
  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}};
  EXPECT_TRUE(reg.apply_first_match(ctx, ChannelId{0}));
  EXPECT_EQ(log, "first second ");
}

TEST(LoweringRegistry, ReportsNoMatch) {
  LoweringRegistry reg;
  soma::CouplingGraph g;
  soma::OperatorDAG d;
  soma::Diagnostics diags;
  LoweringContext ctx{g, d, diags, {}};
  EXPECT_FALSE(reg.apply_first_match(ctx, ChannelId{0}));
}

TEST(LoweringRegistry, BuiltinRegistryIsPopulated) {
  EXPECT_GT(LoweringRegistry::builtin().size(), 0u);
}

// The driver must emit boundary nodes even with no channels at all.
TEST(Lowering, EmitsSourceForEveryProvidedField) {
  soma::CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto c = g.add_component(soma::ComponentDecl{.name = "a"});
  g.add_field(c, soma::FieldSpec{.name = "f", .region = r,
                                 .intent = soma::Intent::Provides});
  const auto result = lower(g);
  EXPECT_TRUE(result.ok()) << result.diags.to_string();
  EXPECT_EQ(count_kind(result.dag, OpKind::Source), 1);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/ir/lowering.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/ir/lowering.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/ir/dag.hpp"

namespace soma {

/// Threaded through every lowering rule. `field_arc` maps FieldId::value() to
/// the arc currently holding that field, which is how rules chain.
struct LoweringContext {
  const CouplingGraph&                      graph;
  OperatorDAG&                              dag;
  Diagnostics&                              diags;
  std::unordered_map<std::uint32_t, ArcId>  field_arc;
};

/// Returns true if the rule claimed the channel.
using LoweringRule = std::function<bool(LoweringContext&, ChannelId)>;

/// Spec §5.2: a registry, not a switch. New numerics register a rule; the
/// core is untouched.
class LoweringRegistry {
 public:
  void add(std::string name, int priority, LoweringRule rule) {
    rules_.push_back(Entry{std::move(name), priority, std::move(rule),
                           next_seq_++});
    std::stable_sort(rules_.begin(), rules_.end(),
                     [](const Entry& a, const Entry& b) {
                       if (a.priority != b.priority) return a.priority > b.priority;
                       return a.seq < b.seq;
                     });
  }

  [[nodiscard]] bool apply_first_match(LoweringContext& ctx, ChannelId c) const {
    for (const Entry& e : rules_) {
      if (e.rule(ctx, c)) return true;
    }
    return false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return rules_.size(); }

  /// The registry the driver uses. Populated on first access.
  static LoweringRegistry& builtin();

 private:
  struct Entry {
    std::string  name;
    int          priority;
    LoweringRule rule;
    int          seq;
  };
  std::vector<Entry> rules_;
  int                next_seq_ = 0;
};

struct LoweringResult {
  OperatorDAG dag;
  Diagnostics diags;
  [[nodiscard]] bool ok() const { return !diags.has_errors(); }
};

/// Spec §2: CouplingGraph -> OperatorDAG.
[[nodiscard]] LoweringResult lower(const CouplingGraph& g);

/// Defined across src/ir/rules_channel.cpp and src/ir/rules_derived.cpp.
void register_builtin_rules(LoweringRegistry& reg);

}  // namespace soma
```

- [ ] **Step 4: Write the driver**

`src/ir/lowering.cpp`:

```cpp
#include "soma/ir/lowering.hpp"

#include "soma/ir/op.hpp"

namespace soma {

LoweringRegistry& LoweringRegistry::builtin() {
  static LoweringRegistry* reg = [] {
    auto* r = new LoweringRegistry;
    register_builtin_rules(*r);
    return r;
  }();
  return *reg;
}

namespace {

/// The type a field has as it leaves its component.
ValueType source_type(const CouplingGraph& g, FieldId f) {
  const FieldSpec& spec = g.field(f);
  const ComponentId owner = g.owner_of(f);
  return ValueType{.kind = ValueKind::Instant,
                   .region = spec.region,
                   .components = spec.components,
                   .units = spec.units,
                   .conservation = spec.conservation,
                   .time = TimeLevel::current(),
                   .rank_set = static_cast<int>(owner.value())};
}

std::string qualified(const CouplingGraph& g, FieldId f) {
  return g.component(g.owner_of(f)).name + "." + g.field(f).name;
}

/// Step 4 of the driver: every node input must be assignable from its arc.
void type_check(LoweringContext& ctx) {
  const OperatorDAG& d = ctx.dag;
  for (std::uint32_t i = 0; i < d.node_count(); ++i) {
    const Node& n = d.node(NodeId{i});
    if (n.kind != OpKind::Sink) continue;  // interior types are set by rules

    if (n.inputs.empty() || !n.output.valid()) continue;
    const ValueType& produced = d.arc(n.inputs.front()).type;
    const ValueType& required = d.arc(n.output).type;

    bool allow_time = false;
    if (n.origin.valid() && n.origin.value() < ctx.graph.channel_count()) {
      allow_time =
          !ctx.graph.channel_at(n.origin).time_level_mismatch_reason.empty();
    }
    if (!assignable_from(produced, required, allow_time)) {
      ctx.diags.add(Severity::Error, "E-IR-TYPE",
                    "cannot assign " + produced.to_string() + " to " +
                        required.to_string(),
                    "node " + std::to_string(i) + " \"" + n.label + "\"");
    }
  }
}

}  // namespace

LoweringResult lower(const CouplingGraph& g) {
  LoweringResult result;
  LoweringContext ctx{g, result.dag, result.diags, {}};

  // 1. Source node per provided field.
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Provides && intent != Intent::Bidirectional) continue;
    const ArcId a = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
    ctx.dag.add_node(OpKind::Source, {}, a, OpAttrs{}, ChannelId{},
                     qualified(g, f));
    ctx.field_arc[i] = a;
  }

  // 2. One rule application per channel, in id order.
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    if (!LoweringRegistry::builtin().apply_first_match(ctx, cid)) {
      ctx.diags.add(Severity::Error, "E-IR-NORULE",
                    "no lowering rule matched",
                    "channel '" + g.channel_at(cid).name + "'");
    }
  }

  // 3. Sink node per required field that a channel actually wrote.
  for (std::uint32_t i = 0; i < g.field_count(); ++i) {
    const FieldId f{i};
    const Intent intent = g.field(f).intent;
    if (intent != Intent::Requires && intent != Intent::Bidirectional) continue;
    const auto it = ctx.field_arc.find(i);
    if (it == ctx.field_arc.end()) continue;  // W-NO-WRITER is validate()'s job

    // Find the channel that wrote it, so the Sink can carry its origin.
    ChannelId origin;
    for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
      if (g.channel_at(ChannelId{c}).target == f) { origin = ChannelId{c}; break; }
    }

    const ArcId sink_arc = ctx.dag.add_arc(source_type(g, f), qualified(g, f));
    ctx.dag.add_node(OpKind::Sink, {it->second}, sink_arc, OpAttrs{}, origin,
                     qualified(g, f));
  }

  // 4. Type check.
  type_check(ctx);
  return result;
}

}  // namespace soma
```

- [ ] **Step 5: Create a placeholder rules file so the build links**

Task 2.5 replaces this body. It exists now only so `register_builtin_rules` has
a definition.

`src/ir/rules_channel.cpp`:

```cpp
#include "soma/ir/lowering.hpp"

namespace soma {

void register_builtin_rules(LoweringRegistry& reg) {
  // Task 2.5 installs the real channel rule at priority 100.
  // A placeholder that declines everything keeps the link valid and makes the
  // "no rule matched" path testable.
  reg.add("placeholder", -1000, [](LoweringContext&, ChannelId) {
    return false;
  });
}

}  // namespace soma
```

- [ ] **Step 6: Update `src/CMakeLists.txt`**

```cmake
add_library(soma_core
  graph/yaml_io.cpp
  ir/lowering.cpp
  ir/rules_channel.cpp)
```

- [ ] **Step 7: Register and run**

Add `soma_add_test(lowering_test unit/lowering_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "LoweringRegistry|Lowering"
```

Expected: 5 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add include/soma/ir/lowering.hpp src/ir/ src/CMakeLists.txt \
        test/unit/lowering_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): lowering registry and driver with Source/Sink boundary"
```

---

### Task 2.5: The channel lowering rule

Spec §5.2's worked example. This rule turns one `Channel` into the operator
chain, in the order the channel declares.

**Files:**
- Modify: `src/ir/rules_channel.cpp` (replace the placeholder)
- Test: `test/unit/lowering_channel_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `LoweringContext`, `LoweringRegistry` (2.4).
- Produces: a rule named `"channel.default"` at priority 100 that emits, in this
  exact order, skipping any stage the channel did not request:

| Order | Emitted when | Node |
|---|---|---|
| 1 | always | reads `ctx.field_arc[source]`; error `E-IR-NOSRC` if absent |
| 2 | `accumulate.op != Instant` | `Sample` then `Accumulate` |
| 3 | `extrapolate.enabled` | `Extrapolate` |
| 4 | `lag != 0` | no node; shifts the arc type's `TimeLevel.offset` by `-lag` |
| 5 | `remap.flags != None` or source/target regions differ | `Remap` (region changes here) |
| 6 | `relax.enabled` | `Relax` |
| 7 | always | binds the final arc into `ctx.field_arc[target]` |

  Task 2.6 inserts `Transport`, `Constrain`, and `Diagnose` — the operators the
  user did not write.

  Every emitted node's output arc gets a type derived from the previous arc:
  `Remap` changes `region` (and nothing else); `Accumulate` preserves everything
  but may change `kind` from `Waveform` back to `Instant`; `Relax` and
  `Extrapolate` preserve the type entirely.

- [ ] **Step 1: Write the failing test**

`test/unit/lowering_channel_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>

#include <vector>

using soma::AccumOp;
using soma::ComponentDecl;
using soma::Conservation;
using soma::CouplingGraph;
using soma::FieldSpec;
using soma::Intent;
using soma::MeshDecl;
using soma::OpKind;
using soma::Rational;
using soma::Region;
using soma::RemapFlags;
namespace u = soma::units;

namespace {

struct Pair {
  CouplingGraph g;
  soma::ChannelId ch;
};

// fluid.heat_flux -> solid.q_wall on distinct regions and rank sets.
Pair make_pair() {
  Pair p;
  auto& g = p.g;
  const auto mf = g.add_mesh(MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(MeshDecl{.name = "ms"});
  const auto rf = g.add_region(Region{.name = "rf", .mesh = mf, .dimension = 2});
  const auto rs = g.add_region(Region{.name = "rs", .mesh = ms, .dimension = 2});
  const auto fluid = g.add_component(ComponentDecl{
      .name = "fluid", .ranks = {.first = 0, .count = 2}, .interval = Rational{1}});
  const auto solid = g.add_component(ComponentDecl{
      .name = "solid", .ranks = {.first = 2, .count = 2}, .interval = Rational{1}});
  const auto s = g.add_field(fluid, FieldSpec{
      .name = "heat_flux", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Provides});
  const auto t = g.add_field(solid, FieldSpec{
      .name = "q_wall", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive, .intent = Intent::Requires});
  p.ch = g.add_channel("flux", s, t);
  return p;
}

std::vector<OpKind> kinds_in_order(const soma::OperatorDAG& d) {
  std::vector<OpKind> out;
  for (const soma::NodeId n : d.topological_order()) out.push_back(d.node(n).kind);
  return out;
}

bool contains(const std::vector<OpKind>& v, OpKind k) {
  return std::find(v.begin(), v.end(), k) != v.end();
}

int index_of(const std::vector<OpKind>& v, OpKind k) {
  return static_cast<int>(std::find(v.begin(), v.end(), k) - v.begin());
}

}  // namespace

TEST(ChannelLowering, MinimalChannelIsSourceRemapSink) {
  Pair p = make_pair();
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);
  EXPECT_TRUE(contains(k, OpKind::Source));
  EXPECT_TRUE(contains(k, OpKind::Sink));
  // Regions differ, so a Remap is emitted even though no flags were set.
  EXPECT_TRUE(contains(k, OpKind::Remap));
  EXPECT_FALSE(contains(k, OpKind::Accumulate));
  EXPECT_FALSE(contains(k, OpKind::Relax));
}

// This is the exact chain from spec §5.2.
TEST(ChannelLowering, SpecExampleEmitsSampleAccumulateRemapRelax) {
  Pair p = make_pair();
  p.g.channel(p.ch)
      .accumulate(AccumOp::Mean)
      .remap(RemapFlags::Conservative | RemapFlags::Monotone)
      .relax(0.7);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);

  ASSERT_TRUE(contains(k, OpKind::Sample));
  ASSERT_TRUE(contains(k, OpKind::Accumulate));
  ASSERT_TRUE(contains(k, OpKind::Remap));
  ASSERT_TRUE(contains(k, OpKind::Relax));
  // Order is the point of the whole design (spec §2.1).
  EXPECT_LT(index_of(k, OpKind::Sample), index_of(k, OpKind::Accumulate));
  EXPECT_LT(index_of(k, OpKind::Accumulate), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Relax));
}

TEST(ChannelLowering, RemapChangesRegionAndNothingElse) {
  Pair p = make_pair();
  p.g.channel(p.ch).remap(RemapFlags::Conservative);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Remap) continue;
    const auto& in = r.dag.arc(n.inputs.front()).type;
    const auto& out = r.dag.arc(n.output).type;
    EXPECT_NE(in.region, out.region);
    EXPECT_EQ(in.units, out.units);
    EXPECT_EQ(in.components, out.components);
    EXPECT_EQ(in.conservation, out.conservation);
    return;
  }
  FAIL() << "no Remap node emitted";
}

TEST(ChannelLowering, ExtrapolatePrecedesRemap) {
  Pair p = make_pair();
  p.g.channel(p.ch).extrapolate(1).remap(RemapFlags::Conservative);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds_in_order(r.dag);
  ASSERT_TRUE(contains(k, OpKind::Extrapolate));
  EXPECT_LT(index_of(k, OpKind::Extrapolate), index_of(k, OpKind::Remap));
}

// Spec §4.3: lag(k) is a TimeLevel offset of -k, not a node.
TEST(ChannelLowering, LagShiftsTimeLevelWithoutEmittingANode) {
  Pair p = make_pair();
  p.g.channel(p.ch).lag(1).allow_time_level_mismatch("lag by design");
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();

  bool found_shifted = false;
  for (std::uint32_t i = 0; i < r.dag.arc_count(); ++i) {
    if (r.dag.arc(soma::ArcId{i}).type.time.offset == Rational{-1}) {
      found_shifted = true;
    }
  }
  EXPECT_TRUE(found_shifted);
}

TEST(ChannelLowering, AccumulateRecordsItsOpInAttrs) {
  Pair p = make_pair();
  p.g.channel(p.ch).accumulate(AccumOp::Integral);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind == OpKind::Accumulate) {
      EXPECT_EQ(n.attrs.accum, AccumOp::Integral);
      return;
    }
  }
  FAIL() << "no Accumulate node emitted";
}

TEST(ChannelLowering, RelaxRecordsOmega) {
  Pair p = make_pair();
  p.g.channel(p.ch).relax(0.35);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind == OpKind::Relax) {
      EXPECT_DOUBLE_EQ(n.attrs.omega, 0.35);
      return;
    }
  }
  FAIL() << "no Relax node emitted";
}

TEST(ChannelLowering, EveryNodeCarriesItsOriginChannel) {
  Pair p = make_pair();
  p.g.channel(p.ch).accumulate(AccumOp::Mean).relax(0.5);
  const auto r = lower(p.g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind == OpKind::Source || n.kind == OpKind::Sink) continue;
    EXPECT_TRUE(n.origin.valid()) << to_cstring(n.kind);
  }
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — the placeholder rule declines, so `lower()` reports
`E-IR-NORULE` and `ok()` is false.

- [ ] **Step 3: Replace `src/ir/rules_channel.cpp`**

```cpp
#include <string>

#include "soma/ir/lowering.hpp"
#include "soma/ir/op.hpp"

namespace soma {
namespace {

/// Emit one node and return its output arc, whose type is `out_type`.
ArcId emit(LoweringContext& ctx, OpKind kind, ArcId input, ValueType out_type,
           OpAttrs attrs, ChannelId origin, const std::string& label) {
  const ArcId out = ctx.dag.add_arc(std::move(out_type), label);
  ctx.dag.add_node(kind, {input}, out, std::move(attrs), origin, label);
  return out;
}

bool lower_channel(LoweringContext& ctx, ChannelId cid) {
  const CouplingGraph& g = ctx.graph;
  if (!cid.valid() || cid.value() >= g.channel_count()) return false;

  const Channel& ch = g.channel_at(cid);
  const std::string base = ch.name;

  const auto it = ctx.field_arc.find(ch.source.value());
  if (it == ctx.field_arc.end()) {
    ctx.diags.add(Severity::Error, "E-IR-NOSRC",
                  "source field has no producing arc; is its intent Provides?",
                  "channel '" + base + "'");
    return true;  // claimed, but failed
  }

  ArcId cur = it->second;
  ValueType type = ctx.dag.arc(cur).type;

  // 2. Sample + Accumulate.
  if (ch.accumulate.op != AccumOp::Instant) {
    ValueType wave = type;
    wave.kind = ValueKind::Waveform;
    cur = emit(ctx, OpKind::Sample, cur, wave, OpAttrs{}, cid, base + "/sample");

    ValueType acc = type;  // back to Instant
    cur = emit(ctx, OpKind::Accumulate, cur, acc,
               OpAttrs{.accum = ch.accumulate.op}, cid, base + "/accumulate");
    type = acc;
  }

  // 3. Extrapolate.
  if (ch.extrapolate.enabled) {
    cur = emit(ctx, OpKind::Extrapolate, cur, type,
               OpAttrs{.order = ch.extrapolate.order}, cid,
               base + "/extrapolate");
  }

  // 4. Lag: a TimeLevel shift, not a node (spec §4.3).
  if (ch.lag != 0) {
    ValueType shifted = type;
    shifted.time.offset = shifted.time.offset - Rational{ch.lag};
    // Rebind by emitting nothing: retype the value by threading a fresh arc
    // through a no-op Interpolate would be wrong, so instead record the shift
    // on the next emitted arc. If no further op follows, the Sink sees it.
    type = shifted;
  }

  // 5. Remap: emitted when flags ask for it or the regions differ.
  const RegionId tgt_region = g.field(ch.target).region;
  const bool region_changes = (type.region != tgt_region);
  if (ch.remap.flags != RemapFlags::None || region_changes) {
    ValueType remapped = type;
    remapped.region = tgt_region;
    cur = emit(ctx, OpKind::Remap, cur, remapped,
               OpAttrs{.remap_flags = ch.remap.flags,
                       .method = ch.remap.method_override},
               cid, base + "/remap");
    type = remapped;
  } else if (ch.lag != 0) {
    // The lag shift needs somewhere to live when no Remap follows.
    cur = emit(ctx, OpKind::Interpolate, cur, type, OpAttrs{.lag = ch.lag}, cid,
               base + "/lag");
  }

  // 6. Relax.
  if (ch.relax.enabled) {
    cur = emit(ctx, OpKind::Relax, cur, type, OpAttrs{.omega = ch.relax.omega},
               cid, base + "/relax");
  }

  // 7. Bind the result to the target field.
  ctx.field_arc[ch.target.value()] = cur;
  return true;
}

}  // namespace

void register_builtin_rules(LoweringRegistry& reg) {
  reg.add("channel.default", 100, lower_channel);
}

}  // namespace soma
```

- [ ] **Step 4: Run**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R ChannelLowering
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ir/rules_channel.cpp test/unit/lowering_channel_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): channel lowering rule with explicit operator ordering"
```

---

### Task 2.6: Derived operators — Transport, Constrain, Diagnose

Spec §5.2: three operators appear that the user did not write. This task adds
them as a post-pass over the DAG the channel rule produced, which keeps the
channel rule readable and makes the derivation independently testable.

**Files:**
- Create: `src/ir/rules_derived.cpp`
- Modify: `src/ir/lowering.cpp` (call the post-pass between steps 2 and 3)
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/lowering_derived_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `LoweringContext` (2.4).
- Produces: `void insert_derived_ops(LoweringContext&)`, declared in
  `soma/ir/lowering.hpp`, applying three rules:

  1. **Transport** — walk each channel's chain; wherever consecutive arcs have
     different `rank_set`, insert a `Transport` node whose `attrs.src_rank_set`
     and `attrs.dst_rank_set` record the move. Placement policy for v1: remap
     runs on the **target's** ranks, so the `Transport` goes immediately *before*
     the `Remap`. (Spec §5.1 notes placement is a real performance knob; Phase 5
     Task 5.4 makes it configurable. v1 hardcodes target-side and says so.)
  2. **Constrain** — after every `Remap` whose arc type has
     `conservation != None`, insert `Constrain`.
  3. **Diagnose** — after every `Constrain`, and after every `Remap` that did not
     get one, insert `Diagnose`. `Diagnose` is a pass-through: its output type
     equals its input type.

  Because the DAG is append-only, "inserting" means: create the new node
  consuming the old arc, then rebind `ctx.field_arc[target]` (and any later
  consumer) to the new arc. The pass therefore runs per channel, immediately
  after that channel's rule, so no rebinding of existing consumers is required.

- [ ] **Step 1: Write the failing test**

`test/unit/lowering_derived_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>
#include <vector>

using soma::Conservation;
using soma::CouplingGraph;
using soma::OpKind;
using soma::RemapFlags;
namespace u = soma::units;

namespace {

// Same-rank or cross-rank pair, with conservation switchable.
CouplingGraph pair(bool disjoint_ranks, Conservation cons) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 2}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = disjoint_ranks ? 2 : 0, .count = 2}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = cons, .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).remap(RemapFlags::Conservative);
  return g;
}

std::vector<OpKind> kinds(const soma::OperatorDAG& d) {
  std::vector<OpKind> out;
  for (const soma::NodeId n : d.topological_order()) out.push_back(d.node(n).kind);
  return out;
}

int index_of(const std::vector<OpKind>& v, OpKind k) {
  return static_cast<int>(std::find(v.begin(), v.end(), k) - v.begin());
}

bool contains(const std::vector<OpKind>& v, OpKind k) {
  return std::find(v.begin(), v.end(), k) != v.end();
}

}  // namespace

TEST(DerivedOps, InsertsTransportWhenRankSetsDiffer) {
  const auto r = lower(pair(/*disjoint_ranks=*/true, Conservation::Extensive));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_TRUE(contains(kinds(r.dag), OpKind::Transport));
}

TEST(DerivedOps, NoTransportWhenRankSetsMatch) {
  const auto r = lower(pair(/*disjoint_ranks=*/false, Conservation::Extensive));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_FALSE(contains(kinds(r.dag), OpKind::Transport));
}

TEST(DerivedOps, TransportRecordsSourceAndDestinationRankSets) {
  const auto r = lower(pair(true, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Transport) continue;
    EXPECT_EQ(n.attrs.src_rank_set, 0);
    EXPECT_EQ(n.attrs.dst_rank_set, 1);
    return;
  }
  FAIL() << "no Transport emitted";
}

// v1 policy: remap runs on the target's ranks, so Transport precedes Remap.
TEST(DerivedOps, TransportPrecedesRemap) {
  const auto r = lower(pair(true, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  const auto k = kinds(r.dag);
  EXPECT_LT(index_of(k, OpKind::Transport), index_of(k, OpKind::Remap));
}

TEST(DerivedOps, InsertsConstrainForExtensiveFields) {
  const auto r = lower(pair(false, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  const auto k = kinds(r.dag);
  ASSERT_TRUE(contains(k, OpKind::Constrain));
  EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Constrain));
}

TEST(DerivedOps, NoConstrainWhenConservationIsNone) {
  const auto r = lower(pair(false, Conservation::None));
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(contains(kinds(r.dag), OpKind::Constrain));
}

TEST(DerivedOps, AlwaysInsertsDiagnoseAfterRemap) {
  for (const Conservation c : {Conservation::None, Conservation::Extensive}) {
    const auto r = lower(pair(false, c));
    ASSERT_TRUE(r.ok()) << r.diags.to_string();
    const auto k = kinds(r.dag);
    ASSERT_TRUE(contains(k, OpKind::Diagnose)) << to_cstring(c);
    EXPECT_LT(index_of(k, OpKind::Remap), index_of(k, OpKind::Diagnose));
  }
}

TEST(DerivedOps, DiagnoseIsATypePreservingPassThrough) {
  const auto r = lower(pair(false, Conservation::Extensive));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Diagnose) continue;
    EXPECT_EQ(r.dag.arc(n.inputs.front()).type, r.dag.arc(n.output).type);
    return;
  }
  FAIL() << "no Diagnose emitted";
}

// The full spec §5.2 chain, end to end.
TEST(DerivedOps, ReproducesTheSpecChainOrder) {
  CouplingGraph g = pair(true, Conservation::Extensive);
  g.channel(soma::ChannelId{0}).accumulate(soma::AccumOp::Mean).relax(0.7);
  const auto r = lower(g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds(r.dag);
  // Sample, Accumulate, Transport, Remap, Constrain, Diagnose, Relax, Transport
  EXPECT_LT(index_of(k, OpKind::Sample),     index_of(k, OpKind::Accumulate));
  EXPECT_LT(index_of(k, OpKind::Accumulate), index_of(k, OpKind::Remap));
  EXPECT_LT(index_of(k, OpKind::Remap),      index_of(k, OpKind::Constrain));
  EXPECT_LT(index_of(k, OpKind::Constrain),  index_of(k, OpKind::Diagnose));
  EXPECT_LT(index_of(k, OpKind::Diagnose),   index_of(k, OpKind::Relax));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — no `Transport`, `Constrain`, or `Diagnose` nodes exist yet.

- [ ] **Step 3: Declare the pass in `include/soma/ir/lowering.hpp`**

Add immediately before `register_builtin_rules`:

```cpp
/// Spec §5.2: inserts the operators the user did not write — Transport (from
/// rank layouts), Constrain (from conservation), Diagnose (always).
/// Runs per channel, immediately after that channel's rule.
void insert_derived_ops(LoweringContext& ctx, ChannelId cid);
```

- [ ] **Step 4: Write `src/ir/rules_derived.cpp`**

```cpp
#include <string>
#include <vector>

#include "soma/ir/lowering.hpp"
#include "soma/ir/op.hpp"

namespace soma {
namespace {

ArcId append(LoweringContext& ctx, OpKind kind, ArcId input, ValueType type,
             OpAttrs attrs, ChannelId origin, const std::string& label) {
  const ArcId out = ctx.dag.add_arc(std::move(type), label);
  ctx.dag.add_node(kind, {input}, out, std::move(attrs), origin, label);
  return out;
}

/// The chain of nodes this channel emitted, in dependency order, starting from
/// the source field's arc and following single-consumer links.
std::vector<NodeId> channel_chain(const LoweringContext& ctx, ChannelId cid,
                                  ArcId start) {
  std::vector<NodeId> chain;
  ArcId cur = start;
  while (true) {
    const auto& consumers = ctx.dag.consumers_of(cur);
    NodeId next;
    for (const NodeId n : consumers) {
      if (ctx.dag.node(n).origin == cid) { next = n; break; }
    }
    if (!next.valid()) break;
    chain.push_back(next);
    const ArcId out = ctx.dag.node(next).output;
    if (!out.valid()) break;
    cur = out;
  }
  return chain;
}

}  // namespace

void insert_derived_ops(LoweringContext& ctx, ChannelId cid) {
  const Channel& ch = ctx.graph.channel_at(cid);
  const std::string base = ch.name;

  const auto tgt_it = ctx.field_arc.find(ch.target.value());
  if (tgt_it == ctx.field_arc.end()) return;  // the rule failed; nothing to do

  const int dst_rank_set =
      static_cast<int>(ctx.graph.owner_of(ch.target).value());

  // --- Transport, placed immediately before the Remap (v1 policy: remap runs
  //     on the target's ranks; Phase 5 Task 5.4 makes this configurable).
  //
  // The DAG is append-only, so rather than splice a node in front of the
  // existing Remap, the channel chain is rebuilt: find the Remap, and if its
  // input arc lives on a different rank set, emit Transport + a fresh Remap
  // and rebind the target.
  const auto src_it = ctx.field_arc.find(ch.source.value());
  if (src_it == ctx.field_arc.end()) return;

  const std::vector<NodeId> chain = channel_chain(ctx, cid, src_it->second);

  ArcId cur = tgt_it->second;
  ValueType type = ctx.dag.arc(cur).type;

  // Post-remap derived ops. Find whether this channel emitted a Remap.
  bool has_remap = false;
  for (const NodeId n : chain) {
    if (ctx.dag.node(n).kind == OpKind::Remap) { has_remap = true; break; }
  }

  if (type.rank_set != dst_rank_set) {
    ValueType moved = type;
    moved.rank_set = dst_rank_set;
    OpAttrs a;
    a.src_rank_set = type.rank_set;
    a.dst_rank_set = dst_rank_set;
    cur = append(ctx, OpKind::Transport, cur, moved, a, cid, base + "/transport");
    type = moved;
  }

  if (has_remap && type.conservation != Conservation::None) {
    cur = append(ctx, OpKind::Constrain, cur, type, OpAttrs{}, cid,
                 base + "/constrain");
  }
  if (has_remap) {
    cur = append(ctx, OpKind::Diagnose, cur, type, OpAttrs{}, cid,
                 base + "/diagnose");
  }

  ctx.field_arc[ch.target.value()] = cur;
}

}  // namespace soma
```

Ordering note: this emits `Transport` after the `Remap` rather than before it,
which contradicts the `TransportPrecedesRemap` test. Fix by moving the transport
decision into the channel rule where the pre-remap arc is still current — see
Step 5.

- [ ] **Step 5: Move the Transport decision into the channel rule**

In `src/ir/rules_channel.cpp`, inside `lower_channel`, immediately **before** the
Remap block (Step 5 of that function), insert:

```cpp
  // Transport before Remap: v1 places remap on the target's ranks.
  const int dst_rank_set = static_cast<int>(g.owner_of(ch.target).value());
  if (type.rank_set != dst_rank_set) {
    ValueType moved = type;
    moved.rank_set = dst_rank_set;
    OpAttrs ta;
    ta.src_rank_set = type.rank_set;
    ta.dst_rank_set = dst_rank_set;
    cur = emit(ctx, OpKind::Transport, cur, moved, ta, cid, base + "/transport");
    type = moved;
  }
```

and in `src/ir/rules_derived.cpp`, delete the Transport block (the
`if (type.rank_set != dst_rank_set)` clause and the now-unused `moved`/`a`
locals), leaving `insert_derived_ops` responsible only for `Constrain` and
`Diagnose`.

- [ ] **Step 6: Call the pass from the driver**

In `src/ir/lowering.cpp`, inside the channel loop, after a successful rule
application:

```cpp
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    const ChannelId cid{i};
    if (!LoweringRegistry::builtin().apply_first_match(ctx, cid)) {
      ctx.diags.add(Severity::Error, "E-IR-NORULE",
                    "no lowering rule matched",
                    "channel '" + g.channel_at(cid).name + "'");
      continue;
    }
    insert_derived_ops(ctx, cid);
  }
```

- [ ] **Step 7: Update `src/CMakeLists.txt`**

```cmake
add_library(soma_core
  graph/yaml_io.cpp
  ir/lowering.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp)
```

- [ ] **Step 8: Run**

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R "DerivedOps|ChannelLowering"
```

Expected: all 9 `DerivedOps` and all 8 `ChannelLowering` tests PASS. If
`TransportPrecedesRemap` still fails, the Step 5 edit did not land in the right
place — the transport block must be inside `lower_channel`, above the
`region_changes` computation.

- [ ] **Step 9: Commit**

```bash
git add src/ir/ include/soma/ir/lowering.hpp test/unit/lowering_derived_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): derived Transport, Constrain, and Diagnose operators"
```

---

### Task 2.7: Iteration regions in the DAG

Spec §5.3. The graph-level `IterationRegionDecl` from Task 1.4 becomes a DAG
region covering the nodes its channels emitted. The convergence measure and
accelerator are Phase 7; this task establishes the region boundary and the
residual arcs.

**Files:**
- Create: `src/ir/regions.cpp`
- Modify: `src/ir/lowering.cpp` (call after all channels are lowered)
- Modify: `include/soma/ir/lowering.hpp` (declare `build_iteration_regions`)
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/lowering_regions_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `LoweringContext` (2.4), `IterationRegionDecl` (1.4).
- Produces: `void build_iteration_regions(LoweringContext&)`, which for each
  `IterationRegionDecl`:
  1. collects every node whose `origin` is one of the region's channels → `body`;
  2. for each channel in the region, emits a `Residual` node consuming the
     channel's final arc, whose output arc is the residual → `residuals`;
  3. leaves `carried` **empty** — Phase 7 Task 7.2 derives it by dataflow. The
     field exists now so the dump format does not change later.
  4. adds the region with `max_iterations` copied from the declaration.

  Error `E-IR-REGION-EMPTY` when a declared region's channels produced no nodes.

- [ ] **Step 1: Write the failing test**

`test/unit/lowering_regions_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>

using soma::CouplingGraph;
using soma::OpKind;
namespace u = soma::units;

namespace {

CouplingGraph iterated_pair(int max_iters) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).relax(0.8);
  g.add_iteration_region(soma::IterationRegionDecl{
      .name = "outer", .channels = {ch}, .max_iterations = max_iters});
  return g;
}

}  // namespace

TEST(IterationRegions, RegionAppearsInTheDag) {
  const auto r = lower(iterated_pair(30));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  ASSERT_EQ(r.dag.region_count(), 1u);
  EXPECT_EQ(r.dag.region_at(soma::RegionIdx{0}).name, "outer");
  EXPECT_EQ(r.dag.region_at(soma::RegionIdx{0}).max_iterations, 30);
}

TEST(IterationRegions, BodyContainsTheChannelsNodes) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  const auto& region = r.dag.region_at(soma::RegionIdx{0});
  EXPECT_FALSE(region.body.empty());
  for (const soma::NodeId n : region.body) {
    EXPECT_TRUE(r.dag.node(n).origin.valid());
  }
}

TEST(IterationRegions, EmitsAResidualPerChannel) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  const auto& region = r.dag.region_at(soma::RegionIdx{0});
  EXPECT_EQ(region.residuals.size(), 1u);

  int residual_nodes = 0;
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    if (r.dag.node(soma::NodeId{i}).kind == OpKind::Residual) ++residual_nodes;
  }
  EXPECT_EQ(residual_nodes, 1);
}

// Phase 7 Task 7.2 fills this in; it must be present and empty until then.
TEST(IterationRegions, CarriedSetIsEmptyUntilPhaseSeven) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(r.dag.region_at(soma::RegionIdx{0}).carried.empty());
}

TEST(IterationRegions, DagRemainsAcyclic) {
  const auto r = lower(iterated_pair(10));
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(r.dag.is_acyclic());
}

TEST(IterationRegions, NoRegionsWhenNoneDeclared) {
  CouplingGraph g = iterated_pair(10);
  const CouplingGraph without = [&] {
    CouplingGraph h;
    const auto m = h.add_mesh(soma::MeshDecl{.name = "m"});
    const auto rr = h.add_region(soma::Region{.name = "r", .mesh = m});
    const auto c = h.add_component(soma::ComponentDecl{.name = "solo"});
    h.add_field(c, soma::FieldSpec{.name = "f", .region = rr,
                                   .intent = soma::Intent::Provides});
    return h;
  }();
  const auto r = lower(without);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.dag.region_count(), 0u);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `region_count()` is 0 because nothing builds regions yet.

- [ ] **Step 3: Declare the pass**

In `include/soma/ir/lowering.hpp`, after `insert_derived_ops`:

```cpp
/// Spec §5.3: turns graph-level IterationRegionDecls into DAG regions with
/// residual arcs. The carried (checkpoint) set is derived in Phase 7.
void build_iteration_regions(LoweringContext& ctx);
```

- [ ] **Step 4: Write `src/ir/regions.cpp`**

```cpp
#include <string>
#include <unordered_set>
#include <vector>

#include "soma/ir/lowering.hpp"
#include "soma/ir/op.hpp"

namespace soma {

void build_iteration_regions(LoweringContext& ctx) {
  const CouplingGraph& g = ctx.graph;

  for (std::uint32_t i = 0; i < g.iteration_region_count(); ++i) {
    const IterationRegionDecl& decl = g.iteration_region(IterationRegionIdx{i});

    std::unordered_set<std::uint32_t> member_channels;
    for (const ChannelId c : decl.channels) member_channels.insert(c.value());

    OperatorDAG::IterationRegion region;
    region.name = decl.name;
    region.max_iterations = decl.max_iterations;

    for (std::uint32_t n = 0; n < ctx.dag.node_count(); ++n) {
      const Node& node = ctx.dag.node(NodeId{n});
      if (node.origin.valid() && member_channels.count(node.origin.value())) {
        region.body.push_back(NodeId{n});
      }
    }

    if (region.body.empty()) {
      ctx.diags.add(Severity::Error, "E-IR-REGION-EMPTY",
                    "iteration region '" + decl.name +
                        "' covers no lowered nodes");
      continue;
    }

    // One Residual per channel, consuming that channel's final value.
    for (const ChannelId c : decl.channels) {
      const Channel& ch = g.channel_at(c);
      const auto it = ctx.field_arc.find(ch.target.value());
      if (it == ctx.field_arc.end()) continue;

      ValueType rtype = ctx.dag.arc(it->second).type;
      const ArcId res = ctx.dag.add_arc(rtype, ch.name + "/residual");
      ctx.dag.add_node(OpKind::Residual, {it->second}, res, OpAttrs{}, c,
                       ch.name + "/residual");
      region.residuals.push_back(res);
    }

    // region.carried stays empty: Phase 7 Task 7.2 derives it by dataflow.
    ctx.dag.add_region(std::move(region));
  }
}

}  // namespace soma
```

- [ ] **Step 5: Call it from the driver**

In `src/ir/lowering.cpp`, between the channel loop and the Sink loop:

```cpp
  build_iteration_regions(ctx);
```

- [ ] **Step 6: Update `src/CMakeLists.txt`**

```cmake
add_library(soma_core
  graph/yaml_io.cpp
  ir/lowering.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp
  ir/regions.cpp)
```

- [ ] **Step 7: Run**

Add `soma_add_test(lowering_regions_test unit/lowering_regions_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R IterationRegions
```

Expected: 6 tests PASS.

- [ ] **Step 8: Commit**

```bash
git add src/ir/regions.cpp include/soma/ir/lowering.hpp \
        test/unit/lowering_regions_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): iteration regions with residual arcs in the DAG"
```

---

### Task 2.8: Golden-IR test harness

Spec §10 level 2. A lowering refactor must not silently change the numerics; the
diff is the review.

**Files:**
- Create: `test/golden/CMakeLists.txt` (or extend `test/CMakeLists.txt`)
- Create: `test/golden_ir_test.cpp`
- Create: `test/golden/pair_explicit.ir`
- Create: `test/golden/pair_accumulate_remap_relax.ir`
- Create: `test/golden/pair_iterated.ir`
- Create: `test/golden/order_swap.ir`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `lower()` (2.4), `dump_text()` (2.3), `load_yaml_file()` (1.7).
- Produces:
  - A test that, for each `test/data/<case>.yaml`, lowers it and compares
    `dump_text()` byte-for-byte with `test/golden/<case>.ir`.
  - `SOMA_UPDATE_GOLDEN=1` in the environment rewrites the golden files instead
    of failing, so an intentional change is one command plus a reviewed diff.
  - A dedicated test proving `Accumulate∘Remap` and `Remap∘Accumulate` produce
    **different** dumps — the claim spec §2.1 makes for the whole architecture.

- [ ] **Step 1: Write the failing test**

`test/golden_ir_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/yaml_io.hpp>
#include <soma/ir/dump.hpp>
#include <soma/ir/lowering.hpp>

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

namespace {

std::string read_file(const std::string& path) {
  std::ifstream in{path};
  if (!in) return {};
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

void write_file(const std::string& path, const std::string& text) {
  std::ofstream out{path};
  out << text;
}

bool updating() {
  const char* v = std::getenv("SOMA_UPDATE_GOLDEN");
  return v != nullptr && std::string{v} == "1";
}

/// Lower test/data/<name>.yaml and compare with test/golden/<name>.ir.
void check_golden(const std::string& name) {
  const std::string yaml_path =
      std::string{SOMA_TEST_DATA_DIR} + "/" + name + ".yaml";
  const std::string golden_path =
      std::string{SOMA_TEST_GOLDEN_DIR} + "/" + name + ".ir";

  const auto graph = soma::load_yaml_file(yaml_path);
  ASSERT_TRUE(graph.ok()) << graph.error().message;

  const auto result = lower(graph.value());
  ASSERT_TRUE(result.ok()) << result.diags.to_string();

  const std::string actual = dump_text(result.dag);

  if (updating()) {
    write_file(golden_path, actual);
    GTEST_SKIP() << "golden file updated: " << golden_path;
  }

  const std::string expected = read_file(golden_path);
  ASSERT_FALSE(expected.empty())
      << "missing golden file " << golden_path
      << "\nrun with SOMA_UPDATE_GOLDEN=1 to create it";
  EXPECT_EQ(actual, expected)
      << "IR changed. If intentional, run:\n"
      << "  SOMA_UPDATE_GOLDEN=1 ctest --test-dir build -R GoldenIr\n"
      << "and review the diff.";
}

}  // namespace

TEST(GoldenIr, PairExplicit)             { check_golden("pair_explicit"); }
TEST(GoldenIr, PairAccumulateRemapRelax) { check_golden("pair_accumulate_remap_relax"); }
TEST(GoldenIr, PairIterated)             { check_golden("pair_iterated"); }

// Spec §2.1: the architecture's central claim is that operator order is
// explicit and therefore checkable. If these two dumps were ever equal, the
// IR would not be recording what it promises to record.
TEST(GoldenIr, OperatorOrderIsObservable) {
  const auto a = soma::load_yaml_file(
      std::string{SOMA_TEST_DATA_DIR} + "/pair_accumulate_remap_relax.yaml");
  ASSERT_TRUE(a.ok()) << a.error().message;
  const auto b = soma::load_yaml_file(
      std::string{SOMA_TEST_DATA_DIR} + "/pair_explicit.yaml");
  ASSERT_TRUE(b.ok()) << b.error().message;

  const std::string da = dump_text(lower(a.value()).dag);
  const std::string db = dump_text(lower(b.value()).dag);
  EXPECT_NE(da, db);
}
```

- [ ] **Step 2: Write the YAML cases**

`test/data/pair_explicit.yaml` — no accumulation, no relaxation, disjoint ranks:

```yaml
version: 1
meshes:
  - {name: mf, source: "generated:box:4x4x4"}
  - {name: ms, source: "generated:box:3x3x3"}
regions:
  - {name: rf, mesh: mf, entity_set: wall, dimension: 2}
  - {name: rs, mesh: ms, entity_set: wall, dimension: 2}
components:
  - name: fluid
    ranks: {first: 0, count: 1}
    interval: 1
    fields:
      - {name: q, region: rf, disc: FV, components: 1, units: W/m^2,
         conservation: Extensive, intent: Provides}
  - name: solid
    ranks: {first: 1, count: 1}
    interval: 1
    fields:
      - {name: q_in, region: rs, disc: FV, components: 1, units: W/m^2,
         conservation: Extensive, intent: Requires}
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    remap: {flags: [Conservative]}
iteration_regions: []
```

`test/data/pair_accumulate_remap_relax.yaml` — the spec §5.2 chain: copy the
file above and replace the `channels:` block with

```yaml
channels:
  - name: flux
    source: fluid.q
    target: solid.q_in
    accumulate: {op: Mean}
    remap: {flags: [Conservative, Monotone]}
    relax: 0.7
```

`test/data/pair_iterated.yaml` — copy `pair_accumulate_remap_relax.yaml` and
replace the `iteration_regions:` line with

```yaml
iteration_regions:
  - {name: outer, channels: [flux], max_iterations: 30}
```

- [ ] **Step 3: Run to verify it fails**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R GoldenIr
```

Expected: FAIL — the golden files do not exist; the message tells you to run
with `SOMA_UPDATE_GOLDEN=1`.

- [ ] **Step 4: Update the test CMakeLists**

Add the golden directory define to `soma_add_test`:

```cmake
  target_compile_definitions(${name} PRIVATE
    SOMA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
    SOMA_TEST_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/golden")
```

and register the test:

```cmake
soma_add_test(golden_ir_test golden_ir_test.cpp)
```

- [ ] **Step 5: Generate the golden files and read them**

```bash
mkdir -p test/golden
cmake --build build -j
SOMA_UPDATE_GOLDEN=1 ctest --test-dir build -R GoldenIr
cat test/golden/pair_accumulate_remap_relax.ir
```

**Do not skip reading them.** This is the review step that gives the golden
files their value. Check by eye that:
- the operator order is Sample → Accumulate → Transport → Remap → Constrain → Diagnose → Relax;
- `Constrain` is present (the fields are `Extensive`);
- the `Transport` line shows `{0->1}`;
- every arc type shows `W·m^-2` and `Extensive`;
- `pair_explicit.ir` has **no** `Sample`, `Accumulate`, or `Relax`.

If any of those is wrong, the bug is in Task 2.5 or 2.6, not in the harness —
fix it there and regenerate.

- [ ] **Step 6: Re-run without the update flag**

```bash
ctest --test-dir build --output-on-failure -R GoldenIr
```

Expected: 4 tests PASS.

- [ ] **Step 7: Run the whole suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: every Phase 0, 1, and 2 test green.

- [ ] **Step 8: Commit**

```bash
git add test/golden_ir_test.cpp test/golden/ test/data/pair_explicit.yaml \
        test/data/pair_accumulate_remap_relax.yaml test/data/pair_iterated.yaml \
        test/CMakeLists.txt
git commit -m "test(soma): golden-IR harness with checked-in expected dumps"
```

---

## Phase 2 exit criteria

- [ ] Every channel in the pair graph lowers to an explicit operator DAG.
- [ ] `Transport`, `Constrain`, and `Diagnose` are derived, never authored.
- [ ] Operator order is visible in the dump and differs between
      `Accumulate→Remap` and the explicit case — spec §2.1's claim is a passing test.
- [ ] A unit or region mismatch that survives Phase 1 validation is caught as
      `E-IR-TYPE` at lowering.
- [ ] Golden files exist for four cases and are byte-compared in CI.
- [ ] `SOMA_UPDATE_GOLDEN=1` regenerates them; the diff is reviewable.
- [ ] `ctest --test-dir build --output-on-failure` fully green.
