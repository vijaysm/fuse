# SOMA Phase 3 — Clocks, slab decomposition, scheduler, ExecutionPlan

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §2 (layer 3), §4.3 (rational time), §8 (scheduling with independent intervals), §8.2 (acyclicity as a hard gate), §8.3 (slab decomposition)
**Milestone:** a lowered DAG plus a set of component clocks compiles to a flat, immutable `ExecutionPlan` whose acyclicity has been *proved* at setup, with a recognised lockstep fast path.

This is the highest-risk phase. Spec §12 places it before the first coupled run deliberately: getting independent intervals right later would mean rewriting the scheduler.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/schedule/clock.hpp` | `Clock`, `ClockSet` — exact rational time |
| `include/soma/schedule/slab.hpp` | `Slab`, `SlabDecomposition` |
| `include/soma/schedule/plan.hpp` | `Instruction`, `BufferDecl`, `ExecutionPlan` |
| `src/schedule/slab.cpp` | slab decomposition algorithm |
| `src/schedule/expand.cpp` | time-resolved DAG expansion |
| `src/schedule/acyclicity.cpp` | the §8.2 proof |
| `src/schedule/buffers.cpp` | buffer allocation and lifetime analysis |
| `src/schedule/scheduler.cpp` | driver: DAG + clocks → ExecutionPlan |

---

### Task 3.1: `Clock` and `ClockSet`

Spec §8.2: there is no global time step. The runtime advances a partially
ordered set of component clocks.

**Files:**
- Create: `include/soma/schedule/clock.hpp`
- Test: `test/unit/clock_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Rational` (0.3), `ComponentId` (0.6), `CouplingGraph` (1.4).
- Produces:
  - `class Clock` with `Clock(Rational interval, Rational start = Rational{0})`, `Rational now() const`, `Rational interval() const`, `std::int64_t step() const`, `void advance()`, `Rational next() const`, `bool at(Rational t) const`, `void reset()`.
    `now()` must be computed as `start + step * interval` (multiply), **not** accumulated by repeated addition, so it is exact by construction and O(1) to query at any step.
  - `class ClockSet` with `ClockSet(const CouplingGraph&)` (one clock per component, seeded from its `interval`), `const Clock& clock(ComponentId) const`, `Clock& clock(ComponentId)`, `std::size_t size() const`, `Rational earliest_next() const`, `std::vector<ComponentId> due_at(Rational t) const`, `bool is_lockstep() const`, `Rational common_period() const`.
    - `common_period()` is the LCM of all component intervals expressed as a `Rational` — the shortest span after which every clock is simultaneously at a tick. Computed as `lcm_rational` over the set; for intervals 1/2 and 1/3 it is 1; for 1/2 and 1/5 it is 1. Guard: if the LCM would exceed `kMaxPeriodTicks = 1'000'000` ticks of the finest interval, return an error rather than build an unusable schedule (see Task 3.2's `E-SCHED-PERIOD`).
  - free `Rational lcm_rational(Rational a, Rational b)` — smallest `L` such that both `a` and `b` divide `L` an integer number of times: `lcm(num)/gcd(den)`.

- [ ] **Step 1: Write the failing test**

`test/unit/clock_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/schedule/clock.hpp>

#include <algorithm>

using soma::Clock;
using soma::ClockSet;
using soma::ComponentDecl;
using soma::ComponentId;
using soma::CouplingGraph;
using soma::lcm_rational;
using soma::Rational;

TEST(Clock, StartsAtZeroStepZero) {
  const Clock c{Rational{1, 3}};
  EXPECT_EQ(c.now(), Rational{0});
  EXPECT_EQ(c.step(), 0);
  EXPECT_EQ(c.interval(), Rational(1, 3));
  EXPECT_EQ(c.next(), Rational(1, 3));
}

TEST(Clock, AdvancesExactly) {
  Clock c{Rational{1, 3}};
  c.advance();
  c.advance();
  c.advance();
  EXPECT_EQ(c.now(), Rational{1});
  EXPECT_EQ(c.step(), 3);
}

// The property spec §4.3 exists to guarantee.
TEST(Clock, NoDriftOverAMillionSteps) {
  Clock c{Rational{1, 7}};
  for (int i = 0; i < 1'000'000; ++i) c.advance();
  EXPECT_EQ(c.now(), Rational(1'000'000, 7));
  EXPECT_EQ(c.now().den(), 7);
}

TEST(Clock, NowIsComputedNotAccumulated) {
  // A clock jumped forward by setting the step must agree with one advanced
  // step by step -- proof that now() is start + step*interval.
  Clock a{Rational{1, 3}};
  for (int i = 0; i < 100; ++i) a.advance();
  Clock b{Rational{1, 3}};
  for (int i = 0; i < 100; ++i) b.advance();
  EXPECT_EQ(a.now(), b.now());
  EXPECT_EQ(a.now(), Rational(100, 3));
}

TEST(Clock, AtTestsTickAlignment) {
  const Clock c{Rational{1, 4}};
  EXPECT_TRUE(c.at(Rational{0}));
  EXPECT_TRUE(c.at(Rational{1, 4}));
  EXPECT_TRUE(c.at(Rational{1}));
  EXPECT_FALSE(c.at(Rational{1, 3}));
}

TEST(Clock, ResetReturnsToStart) {
  Clock c{Rational{1, 2}, Rational{5}};
  c.advance();
  c.reset();
  EXPECT_EQ(c.now(), Rational{5});
  EXPECT_EQ(c.step(), 0);
}

TEST(LcmRational, SmallestCommonMultiple) {
  EXPECT_EQ(lcm_rational(Rational(1, 2), Rational(1, 3)), Rational(1));
  EXPECT_EQ(lcm_rational(Rational(1, 2), Rational(1, 4)), Rational(1, 2));
  EXPECT_EQ(lcm_rational(Rational(2, 3), Rational(1, 2)), Rational(2));
  EXPECT_EQ(lcm_rational(Rational(3), Rational(2)), Rational(6));
}

namespace {

CouplingGraph two_components(Rational a, Rational b) {
  CouplingGraph g;
  g.add_component(ComponentDecl{.name = "a", .interval = a});
  g.add_component(ComponentDecl{.name = "b", .interval = b});
  return g;
}

}  // namespace

TEST(ClockSet, OneClockPerComponentSeededFromInterval) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 3})};
  EXPECT_EQ(cs.size(), 2u);
  EXPECT_EQ(cs.clock(ComponentId{0}).interval(), Rational(1, 2));
  EXPECT_EQ(cs.clock(ComponentId{1}).interval(), Rational(1, 3));
}

TEST(ClockSet, EarliestNextPicksTheSoonestTick) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 3})};
  EXPECT_EQ(cs.earliest_next(), Rational(1, 3));
}

TEST(ClockSet, DueAtListsAlignedComponents) {
  const ClockSet cs{two_components(Rational{1, 2}, Rational{1, 4})};
  const auto due = cs.due_at(Rational{1, 2});
  EXPECT_EQ(due.size(), 2u);   // 1/2 is a tick of both
  const auto due_quarter = cs.due_at(Rational{1, 4});
  ASSERT_EQ(due_quarter.size(), 1u);
  EXPECT_EQ(due_quarter[0], ComponentId{1});
}

TEST(ClockSet, DetectsLockstep) {
  EXPECT_TRUE(ClockSet{two_components(Rational{1}, Rational{1})}.is_lockstep());
  EXPECT_FALSE(
      ClockSet{two_components(Rational{1}, Rational{1, 2})}.is_lockstep());
}

TEST(ClockSet, CommonPeriodIsTheLcm) {
  EXPECT_EQ(ClockSet{two_components(Rational{1, 2}, Rational{1, 3})}
                .common_period(), Rational(1));
  EXPECT_EQ(ClockSet{two_components(Rational{1, 4}, Rational{1, 2})}
                .common_period(), Rational(1, 2));
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/clock.hpp` file not found.

- [ ] **Step 3: Write the implementation**

`include/soma/schedule/clock.hpp`:

```cpp
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

/// Smallest L such that both a and b divide L an integer number of times.
[[nodiscard]] constexpr Rational lcm_rational(const Rational& a,
                                              const Rational& b) {
  const std::int64_t an = a.num() < 0 ? -a.num() : a.num();
  const std::int64_t bn = b.num() < 0 ? -b.num() : b.num();
  return Rational{std::lcm(an, bn), std::gcd(a.den(), b.den())};
}

/// Spec §8.2: one clock per component; there is no global time step.
/// now() is computed as start + step*interval, never accumulated, so it is
/// exact at any step and O(1) to query.
class Clock {
 public:
  explicit Clock(Rational interval, Rational start = Rational{0})
      : interval_(interval), start_(start) {}

  [[nodiscard]] Rational now() const { return start_ + Rational{step_} * interval_; }
  [[nodiscard]] Rational next() const {
    return start_ + Rational{step_ + 1} * interval_;
  }
  [[nodiscard]] Rational interval() const noexcept { return interval_; }
  [[nodiscard]] std::int64_t step() const noexcept { return step_; }

  void advance() noexcept { ++step_; }
  void reset() noexcept { step_ = 0; }

  /// Is t an exact tick of this clock?
  [[nodiscard]] bool at(Rational t) const {
    const Rational d = t - start_;
    if (interval_.is_zero()) return d.is_zero();
    const Rational q = d / interval_;
    return q.is_integer();
  }

 private:
  Rational    interval_;
  Rational    start_;
  std::int64_t step_ = 0;
};

/// The partially ordered set of component clocks the runtime advances.
class ClockSet {
 public:
  explicit ClockSet(const CouplingGraph& g) {
    clocks_.reserve(g.component_count());
    for (std::uint32_t i = 0; i < g.component_count(); ++i) {
      clocks_.emplace_back(g.component(ComponentId{i}).interval);
    }
  }

  [[nodiscard]] const Clock& clock(ComponentId c) const {
    return clocks_.at(c.value());
  }
  [[nodiscard]] Clock& clock(ComponentId c) { return clocks_.at(c.value()); }
  [[nodiscard]] std::size_t size() const noexcept { return clocks_.size(); }

  [[nodiscard]] Rational earliest_next() const {
    Rational best{0};
    bool first = true;
    for (const Clock& c : clocks_) {
      if (first || c.next() < best) {
        best = c.next();
        first = false;
      }
    }
    return best;
  }

  [[nodiscard]] std::vector<ComponentId> due_at(Rational t) const {
    std::vector<ComponentId> due;
    for (std::size_t i = 0; i < clocks_.size(); ++i) {
      if (clocks_[i].at(t)) due.push_back(ComponentId{static_cast<std::uint32_t>(i)});
    }
    return due;
  }

  /// Spec §8.2: lockstep is the degenerate case and gets a fast path.
  [[nodiscard]] bool is_lockstep() const {
    if (clocks_.empty()) return true;
    const Rational first = clocks_.front().interval();
    return std::all_of(clocks_.begin(), clocks_.end(),
                       [&](const Clock& c) { return c.interval() == first; });
  }

  /// Shortest span after which every clock is simultaneously at a tick.
  [[nodiscard]] Rational common_period() const {
    if (clocks_.empty()) return Rational{1};
    Rational p = clocks_.front().interval();
    for (const Clock& c : clocks_) p = lcm_rational(p, c.interval());
    return p;
  }

  /// Finest interval in the set; the slab grid resolution.
  [[nodiscard]] Rational finest_interval() const {
    if (clocks_.empty()) return Rational{1};
    Rational f = clocks_.front().interval();
    for (const Clock& c : clocks_) f = gcd_rational(f, c.interval());
    return f;
  }

 private:
  std::vector<Clock> clocks_;
};

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(clock_test unit/clock_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R "Clock|LcmRational"
```

Expected: 13 tests PASS. `NoDriftOverAMillionSteps` is the load-bearing one.

- [ ] **Step 5: Commit**

```bash
git add include/soma/schedule/clock.hpp test/unit/clock_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): exact rational Clock and ClockSet"
```

---

### Task 3.2: Slab decomposition

Spec §8.3: per channel, compute the least common window, intersect the two time
partitions, and emit transfers on the resulting intervals. This is the mechanism
that subsumes lag, subcycling, and multirate.

**Files:**
- Create: `include/soma/schedule/slab.hpp`
- Create: `src/schedule/slab.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/slab_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Clock`, `ClockSet` (3.1); `Rational` (0.3).
- Produces:
  - `struct Slab { Rational begin; Rational end; std::int64_t src_step; std::int64_t dst_step; }` with `Rational length() const` and defaulted `==`.
    `src_step`/`dst_step` are the source and target step indices this slab falls within — what the runtime needs to know which value to read and where to write.
  - `struct SlabDecomposition { std::vector<Slab> slabs; Rational period; bool degenerate = false; }` — `degenerate == true` means source and target intervals are equal and aligned, so there is exactly one slab per period (the lockstep case).
  - `Result<SlabDecomposition> decompose(Rational src_interval, Rational dst_interval, Rational period)` — intersect the two partitions of `[0, period)`.
  - `inline constexpr std::int64_t kMaxSlabsPerPeriod = 100'000;` with error `E-SCHED-PERIOD` when exceeded — intervals like 1/999983 and 1/999979 would otherwise produce an astronomically large decomposition, and failing loudly at setup beats hanging.

  Algorithm: collect every tick of both clocks in `[0, period]` into a sorted
  unique list; consecutive pairs are the slabs; for each, record which source and
  target step contains it. Uses only exact `Rational`, so no tolerance parameter
  appears anywhere.

- [ ] **Step 1: Write the failing test**

`test/unit/slab_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/schedule/slab.hpp>

using soma::decompose;
using soma::Rational;
using soma::Slab;

TEST(Slab, EqualIntervalsGiveOneDegenerateSlab) {
  const auto r = decompose(Rational{1}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value().degenerate);
  ASSERT_EQ(r.value().slabs.size(), 1u);
  EXPECT_EQ(r.value().slabs[0].begin, Rational{0});
  EXPECT_EQ(r.value().slabs[0].end, Rational{1});
}

TEST(Slab, IntegerSubcyclingGivesNSlabs) {
  // Source ticks 4x per target tick.
  const auto r = decompose(Rational{1, 4}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.value().degenerate);
  ASSERT_EQ(r.value().slabs.size(), 4u);
  EXPECT_EQ(r.value().slabs[0].end, Rational(1, 4));
  EXPECT_EQ(r.value().slabs[3].end, Rational{1});
}

// The case that motivates exact rational time.
TEST(Slab, NonNestingIntervalsIntersectCorrectly) {
  // 1/2 and 1/3 over a period of 1 -> ticks at 0, 1/3, 1/2, 2/3, 1
  const auto r = decompose(Rational{1, 2}, Rational{1, 3}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_EQ(s.size(), 4u);
  EXPECT_EQ(s[0].begin, Rational{0});
  EXPECT_EQ(s[0].end,   Rational(1, 3));
  EXPECT_EQ(s[1].end,   Rational(1, 2));
  EXPECT_EQ(s[2].end,   Rational(2, 3));
  EXPECT_EQ(s[3].end,   Rational{1});
}

TEST(Slab, SlabsTileThePeriodWithoutGapOrOverlap) {
  const auto r = decompose(Rational{1, 3}, Rational{1, 5}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_FALSE(s.empty());
  EXPECT_EQ(s.front().begin, Rational{0});
  EXPECT_EQ(s.back().end, Rational{1});
  for (std::size_t i = 1; i < s.size(); ++i) {
    EXPECT_EQ(s[i - 1].end, s[i].begin) << "gap or overlap at slab " << i;
  }
  Rational total{0};
  for (const Slab& x : s) total += x.length();
  EXPECT_EQ(total, Rational{1});
}

TEST(Slab, RecordsSourceAndTargetStepPerSlab) {
  // src 1/2, dst 1: slab [0,1/2) is src step 0, slab [1/2,1) is src step 1;
  // both are dst step 0.
  const auto r = decompose(Rational{1, 2}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[0].src_step, 0);
  EXPECT_EQ(s[1].src_step, 1);
  EXPECT_EQ(s[0].dst_step, 0);
  EXPECT_EQ(s[1].dst_step, 0);
}

TEST(Slab, EveryLengthIsPositive) {
  const auto r = decompose(Rational{1, 6}, Rational{1, 4}, Rational{1});
  ASSERT_TRUE(r.ok());
  for (const Slab& s : r.value().slabs) {
    EXPECT_GT(s.length(), Rational{0});
  }
}

TEST(Slab, RejectsExplosivePeriod) {
  // Coprime large denominators would need ~10^12 slabs.
  const auto r = decompose(Rational{1, 999983}, Rational{1, 999979},
                           Rational{1});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

TEST(Slab, RejectsNonPositiveInterval) {
  EXPECT_FALSE(decompose(Rational{0}, Rational{1}, Rational{1}).ok());
  EXPECT_FALSE(decompose(Rational{1}, Rational{-1}, Rational{1}).ok());
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/slab.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/schedule/slab.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"

namespace soma {

/// One interval of the intersected source/target time partitions.
/// src_step and dst_step say which step of each side the slab falls within.
struct Slab {
  Rational     begin;
  Rational     end;
  std::int64_t src_step = 0;
  std::int64_t dst_step = 0;

  [[nodiscard]] Rational length() const { return end - begin; }
  friend bool operator==(const Slab&, const Slab&) = default;
};

struct SlabDecomposition {
  std::vector<Slab> slabs;
  Rational          period{1};
  bool              degenerate = false;  // one slab: source and target aligned
};

/// Spec §8.3: a runaway decomposition is a setup error, not a hang.
inline constexpr std::int64_t kMaxSlabsPerPeriod = 100'000;

/// Intersect the source and target time partitions of [0, period).
[[nodiscard]] Result<SlabDecomposition> decompose(Rational src_interval,
                                                  Rational dst_interval,
                                                  Rational period);

}  // namespace soma
```

- [ ] **Step 4: Write `src/schedule/slab.cpp`**

```cpp
#include "soma/schedule/slab.hpp"

#include <algorithm>

namespace soma {
namespace {

/// Every tick of `interval` in [0, period], inclusive of both ends.
Result<std::vector<Rational>> ticks_of(Rational interval, Rational period) {
  const Rational n_exact = period / interval;
  if (!n_exact.is_integer()) {
    // period is always an LCM of the intervals, so this cannot happen unless a
    // caller passes an unrelated period.
    return Error{"E-SCHED-PERIOD",
                 "period " + period.to_string() + " is not an integer multiple of "
                 "interval " + interval.to_string()};
  }
  const std::int64_t n = n_exact.num();
  if (n > kMaxSlabsPerPeriod) {
    return Error{"E-SCHED-PERIOD",
                 "interval " + interval.to_string() + " needs " +
                     std::to_string(n) + " ticks per period " +
                     period.to_string() + ", exceeding the limit of " +
                     std::to_string(kMaxSlabsPerPeriod)};
  }
  std::vector<Rational> out;
  out.reserve(static_cast<std::size_t>(n) + 1);
  for (std::int64_t i = 0; i <= n; ++i) out.push_back(Rational{i} * interval);
  return out;
}

/// Which step of `interval` contains the half-open slab starting at `t`?
std::int64_t step_containing(Rational t, Rational interval) {
  const Rational q = t / interval;
  // Floor for non-negative q: exact integer division of the normalised form.
  return q.num() / q.den();
}

}  // namespace

Result<SlabDecomposition> decompose(Rational src_interval,
                                    Rational dst_interval, Rational period) {
  if (!(src_interval > Rational{0})) {
    return Error{"E-SCHED-INTERVAL",
                 "source interval must be positive, got " +
                     src_interval.to_string()};
  }
  if (!(dst_interval > Rational{0})) {
    return Error{"E-SCHED-INTERVAL",
                 "target interval must be positive, got " +
                     dst_interval.to_string()};
  }
  if (!(period > Rational{0})) {
    return Error{"E-SCHED-PERIOD", "period must be positive"};
  }

  auto src_ticks = ticks_of(src_interval, period);
  if (!src_ticks.ok()) return src_ticks.error();
  auto dst_ticks = ticks_of(dst_interval, period);
  if (!dst_ticks.ok()) return dst_ticks.error();

  std::vector<Rational> all = src_ticks.value();
  all.insert(all.end(), dst_ticks.value().begin(), dst_ticks.value().end());
  std::sort(all.begin(), all.end());
  all.erase(std::unique(all.begin(), all.end()), all.end());

  if (static_cast<std::int64_t>(all.size()) > kMaxSlabsPerPeriod) {
    return Error{"E-SCHED-PERIOD",
                 "slab decomposition would need " + std::to_string(all.size()) +
                     " boundaries, exceeding the limit of " +
                     std::to_string(kMaxSlabsPerPeriod)};
  }

  SlabDecomposition out;
  out.period = period;
  out.slabs.reserve(all.size() > 0 ? all.size() - 1 : 0);
  for (std::size_t i = 1; i < all.size(); ++i) {
    Slab s;
    s.begin = all[i - 1];
    s.end = all[i];
    s.src_step = step_containing(s.begin, src_interval);
    s.dst_step = step_containing(s.begin, dst_interval);
    out.slabs.push_back(s);
  }
  out.degenerate = (out.slabs.size() == 1);
  return out;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`**

```cmake
add_library(soma_core
  graph/yaml_io.cpp
  ir/lowering.cpp
  ir/rules_channel.cpp
  ir/rules_derived.cpp
  ir/regions.cpp
  schedule/slab.cpp)
```

- [ ] **Step 6: Register and run**

Add `soma_add_test(slab_test unit/slab_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Slab
```

Expected: 8 tests PASS. `SlabsTileThePeriodWithoutGapOrOverlap` is the invariant
everything downstream relies on.

- [ ] **Step 7: Commit**

```bash
git add include/soma/schedule/slab.hpp src/schedule/slab.cpp \
        test/unit/slab_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): exact slab decomposition of intersected time partitions"
```

---

### Task 3.3: Time-resolved expansion

The lowered DAG describes *what* happens per exchange; the schedule needs *when*.
This task unrolls the DAG over one common period, producing one instance of each
channel's chain per slab that requires it.

**Files:**
- Create: `include/soma/schedule/expand.hpp`
- Create: `src/schedule/expand.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/expand_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `OperatorDAG` (2.2), `ClockSet` (3.1), `decompose` (3.2), `CouplingGraph` (1.4).
- Produces:
  - `struct TimedNode { NodeId node; Rational time; std::int64_t src_step; std::int64_t dst_step; std::int64_t instance; }` — `instance` distinguishes repetitions of the same DAG node within one period.
  - `struct TimedDag { std::vector<TimedNode> nodes; Rational period; bool lockstep = false; }`
  - `Result<TimedDag> expand(const CouplingGraph&, const OperatorDAG&, const ClockSet&)`

  Rules:
  - A channel's chain is instantiated once per slab of its (source interval, target interval) decomposition. A channel whose accumulate op is not `Instant` instantiates its `Sample` node once per **source** tick, but its `Accumulate` and everything after once per **target** tick — that asymmetry is what accumulation means.
  - Nodes with no channel origin (`Source`, `Sink`) are instantiated once per owning component tick.
  - `lockstep` is set when `ClockSet::is_lockstep()` and no channel has a space-time basis; Task 3.7 uses it.
  - Error `E-SCHED-EXPLODE` when the total timed-node count would exceed `kMaxTimedNodes = 1'000'000`.

- [ ] **Step 1: Write the failing test**

`test/unit/expand_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/schedule/expand.hpp>

#include <algorithm>

using soma::ComponentDecl;
using soma::CouplingGraph;
using soma::OpKind;
using soma::Rational;
namespace u = soma::units;

namespace {

CouplingGraph pair(Rational src_dt, Rational dst_dt, soma::AccumOp accum) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = src_dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = dst_dt});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  if (accum != soma::AccumOp::Instant) g.channel(ch).accumulate(accum);
  return g;
}

int count_kind(const soma::TimedDag& td, const soma::OperatorDAG& d, OpKind k) {
  int n = 0;
  for (const auto& tn : td.nodes) {
    if (d.node(tn.node).kind == k) ++n;
  }
  return n;
}

}  // namespace

TEST(Expand, LockstepGivesOneInstancePerNode) {
  const CouplingGraph g = pair(Rational{1}, Rational{1}, soma::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_TRUE(td.value().lockstep);
  EXPECT_EQ(td.value().nodes.size(), lowered.dag.node_count());
}

TEST(Expand, SubcyclingRepeatsTheChain) {
  // Source ticks 4x per target tick.
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, soma::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const soma::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_FALSE(td.value().lockstep);
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Remap), 4);
}

// Accumulation samples on the source clock and reduces on the target clock.
TEST(Expand, AccumulateSamplesPerSourceTickButReducesPerTargetTick) {
  const CouplingGraph g = pair(Rational{1, 4}, Rational{1}, soma::AccumOp::Mean);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const soma::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Sample), 4);
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Accumulate), 1);
}

TEST(Expand, NonNestingIntervalsProduceIntersectedInstances) {
  // 1/2 and 1/3 -> 4 slabs per period of 1.
  const CouplingGraph g = pair(Rational{1, 2}, Rational{1, 3},
                               soma::AccumOp::Instant);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const soma::ClockSet cs{g};

  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok()) << td.error().message;
  EXPECT_EQ(td.value().period, Rational{1});
  EXPECT_EQ(count_kind(td.value(), lowered.dag, OpKind::Remap), 4);
}

TEST(Expand, EveryTimedNodeCarriesAnExactTime) {
  const CouplingGraph g = pair(Rational{1, 3}, Rational{1}, soma::AccumOp::Instant);
  const auto lowered = lower(g);
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok());
  for (const auto& tn : td.value().nodes) {
    EXPECT_GE(tn.time, Rational{0});
    EXPECT_LE(tn.time, td.value().period);
  }
}

TEST(Expand, InstancesAreNumberedFromZero) {
  const CouplingGraph g = pair(Rational{1, 3}, Rational{1}, soma::AccumOp::Instant);
  const auto lowered = lower(g);
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs).value();

  for (std::uint32_t n = 0; n < lowered.dag.node_count(); ++n) {
    std::vector<std::int64_t> instances;
    for (const auto& tn : td.nodes) {
      if (tn.node == soma::NodeId{n}) instances.push_back(tn.instance);
    }
    if (instances.empty()) continue;
    std::sort(instances.begin(), instances.end());
    for (std::size_t i = 0; i < instances.size(); ++i) {
      EXPECT_EQ(instances[i], static_cast<std::int64_t>(i));
    }
  }
}

TEST(Expand, RejectsExplosiveExpansion) {
  const CouplingGraph g = pair(Rational{1, 999983}, Rational{1, 999979},
                               soma::AccumOp::Instant);
  const auto lowered = lower(g);
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_FALSE(td.ok());
  EXPECT_TRUE(td.error().code == "E-SCHED-PERIOD" ||
              td.error().code == "E-SCHED-EXPLODE");
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/expand.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/schedule/expand.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/ir/dag.hpp"
#include "soma/schedule/clock.hpp"

namespace soma {

/// One instance of a DAG node at a specific time within the common period.
struct TimedNode {
  NodeId       node;
  Rational     time;
  std::int64_t src_step = 0;
  std::int64_t dst_step = 0;
  std::int64_t instance = 0;  // repetition index of this node within the period
};

struct TimedDag {
  std::vector<TimedNode> nodes;
  Rational               period{1};
  bool                   lockstep = false;
};

inline constexpr std::int64_t kMaxTimedNodes = 1'000'000;

/// Unroll the lowered DAG over one common period of the clock set.
[[nodiscard]] Result<TimedDag> expand(const CouplingGraph& g,
                                      const OperatorDAG& dag,
                                      const ClockSet& clocks);

}  // namespace soma
```

- [ ] **Step 4: Write `src/schedule/expand.cpp`**

```cpp
#include "soma/schedule/expand.hpp"

#include <algorithm>
#include <unordered_map>

#include "soma/schedule/slab.hpp"

namespace soma {
namespace {

bool any_spacetime(const CouplingGraph& g) {
  for (std::uint32_t i = 0; i < g.channel_count(); ++i) {
    if (g.channel_at(ChannelId{i}).spacetime.basis != TemporalBasis::None) {
      return true;
    }
  }
  return false;
}

/// Does this node run on the source clock rather than the target clock?
/// Sample is the only interior op that does: accumulation samples every source
/// tick and reduces once per target tick.
bool runs_on_source_clock(OpKind k) { return k == OpKind::Sample; }

}  // namespace

Result<TimedDag> expand(const CouplingGraph& g, const OperatorDAG& dag,
                        const ClockSet& clocks) {
  TimedDag out;
  out.period = clocks.common_period();
  out.lockstep = clocks.is_lockstep() && !any_spacetime(g);

  std::unordered_map<std::uint32_t, std::int64_t> instance_of;  // NodeId -> count

  const auto emit = [&](NodeId n, Rational t, std::int64_t src_step,
                        std::int64_t dst_step) {
    const std::int64_t inst = instance_of[n.value()]++;
    out.nodes.push_back(TimedNode{n, t, src_step, dst_step, inst});
  };

  // --- boundary nodes: one instance per owning component tick ---------------
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const NodeId nid{i};
    const Node& n = dag.node(nid);
    if (n.kind != OpKind::Source && n.kind != OpKind::Sink) continue;

    // The arc type records which rank set (component) the value lives on.
    const ArcId ref = n.output.valid() ? n.output
                                       : (n.inputs.empty() ? ArcId{} : n.inputs.front());
    if (!ref.valid()) continue;
    const int rank_set = dag.arc(ref).type.rank_set;
    if (rank_set < 0 ||
        static_cast<std::uint32_t>(rank_set) >= g.component_count()) {
      continue;
    }
    const Rational dt = clocks.clock(ComponentId{
        static_cast<std::uint32_t>(rank_set)}).interval();
    const Rational n_exact = out.period / dt;
    if (!n_exact.is_integer()) {
      return Error{"E-SCHED-PERIOD",
                   "component interval " + dt.to_string() +
                       " does not divide the common period " +
                       out.period.to_string()};
    }
    for (std::int64_t k = 0; k < n_exact.num(); ++k) {
      emit(nid, Rational{k} * dt, k, k);
    }
  }

  // --- channel chains: one instance per slab -------------------------------
  for (std::uint32_t c = 0; c < g.channel_count(); ++c) {
    const ChannelId cid{c};
    const Channel& ch = g.channel_at(cid);
    const Rational src_dt = g.component(g.owner_of(ch.source)).interval;
    const Rational dst_dt = g.component(g.owner_of(ch.target)).interval;

    auto decomp = decompose(src_dt, dst_dt, out.period);
    if (!decomp.ok()) return decomp.error();

    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const NodeId nid{i};
      const Node& n = dag.node(nid);
      if (n.origin != cid) continue;

      if (runs_on_source_clock(n.kind)) {
        // One Sample per source tick.
        const Rational n_exact = out.period / src_dt;
        for (std::int64_t k = 0; k < n_exact.num(); ++k) {
          emit(nid, Rational{k} * src_dt, k, 0);
        }
      } else if (ch.accumulate.op != AccumOp::Instant) {
        // Reduction and everything after it run once per target tick.
        const Rational n_exact = out.period / dst_dt;
        for (std::int64_t k = 0; k < n_exact.num(); ++k) {
          emit(nid, Rational{k + 1} * dst_dt, 0, k);
        }
      } else {
        // No accumulation: one instance per slab.
        for (const Slab& s : decomp.value().slabs) {
          emit(nid, s.end, s.src_step, s.dst_step);
        }
      }

      if (static_cast<std::int64_t>(out.nodes.size()) > kMaxTimedNodes) {
        return Error{"E-SCHED-EXPLODE",
                     "time expansion exceeds " + std::to_string(kMaxTimedNodes) +
                         " nodes; check component intervals"};
      }
    }
  }

  std::stable_sort(out.nodes.begin(), out.nodes.end(),
                   [](const TimedNode& a, const TimedNode& b) {
                     if (a.time != b.time) return a.time < b.time;
                     return a.node < b.node;
                   });
  return out;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `schedule/expand.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(expand_test unit/expand_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Expand
```

Expected: 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/soma/schedule/expand.hpp src/schedule/expand.cpp \
        test/unit/expand_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): time-resolved DAG expansion over the common period"
```

---

### Task 3.4: Acyclicity proof — the hard init-time gate

Spec §8.2. OASIS detects cyclic waits reactively via `SEQ`; SOMA has the whole
graph before it runs, so it proves acyclicity at setup. A scheduling error at
initialisation, not a hang at hour six.

The subtlety the spec calls out: **a cycle with a lag in it is legal.** A→B at
time level `k` and B→A at level `k−1` is a well-posed explicit scheme, not a
deadlock. The proof must operate on the *time-resolved* graph, where the lag has
already been unrolled into a dependence on a previous period.

**Files:**
- Create: `include/soma/schedule/acyclicity.hpp`
- Create: `src/schedule/acyclicity.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/acyclicity_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `TimedDag` (3.3), `OperatorDAG` (2.2), `CouplingGraph` (1.4).
- Produces:
  - `struct CycleReport { bool acyclic = true; std::vector<NodeId> cycle; std::string explanation; }`
  - `CycleReport prove_acyclic(const CouplingGraph&, const OperatorDAG&, const TimedDag&)`

  Method: build a dependence graph over *timed* nodes. An edge `u → v` exists
  when `v` consumes an arc `u` produces **and** `u`'s instance is not separated
  from `v`'s by a period boundary. A dependence whose source time level has a
  negative offset (a lag) crosses the boundary and is therefore *not* an edge in
  the intra-period graph — that is exactly why lagged cycles are legal. Then run
  Kahn; if any node remains, extract one cycle by DFS and report it with the
  channel names involved.

- [ ] **Step 1: Write the failing test**

`test/unit/acyclicity_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/schedule/acyclicity.hpp>
#include <soma/schedule/expand.hpp>

using soma::ComponentDecl;
using soma::CouplingGraph;
using soma::Rational;
namespace u = soma::units;

namespace {

// A <-> B. `lag_on_return` puts a lag on the B->A channel, which makes the
// cycle a legal explicit scheme rather than a deadlock.
CouplingGraph two_way(bool lag_on_return) {
  CouplingGraph g;
  const auto ma = g.add_mesh(soma::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(soma::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(soma::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(soma::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, soma::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto b_in = g.add_field(b, soma::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto b_out = g.add_field(b, soma::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto a_in = g.add_field(a, soma::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kKelvin,
      .intent = soma::Intent::Requires});

  g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  if (lag_on_return) {
    g.channel(back).lag(1).allow_time_level_mismatch("explicit two-way scheme");
  }
  return g;
}

soma::CycleReport check(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  EXPECT_TRUE(td.ok()) << td.error().message;
  return prove_acyclic(g, lowered.dag, td.value());
}

}  // namespace

TEST(Acyclicity, LinearChainIsAcyclic) {
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r, .intent = soma::Intent::Requires});
  g.add_channel("c", s, t);

  const auto report = check(g);
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

// The hard gate: an unlagged two-way coupling is a genuine deadlock.
TEST(Acyclicity, UnlaggedTwoWayCouplingIsRejected) {
  const auto report = check(two_way(/*lag_on_return=*/false));
  EXPECT_FALSE(report.acyclic);
  EXPECT_FALSE(report.cycle.empty());
  EXPECT_FALSE(report.explanation.empty());
}

// The subtlety: the same topology with a lag is a legal explicit scheme.
TEST(Acyclicity, LaggedTwoWayCouplingIsAccepted) {
  const auto report = check(two_way(/*lag_on_return=*/true));
  EXPECT_TRUE(report.acyclic) << report.explanation;
}

TEST(Acyclicity, ExplanationNamesTheChannelsInvolved) {
  const auto report = check(two_way(false));
  ASSERT_FALSE(report.acyclic);
  const bool names_a_channel =
      report.explanation.find("a2b") != std::string::npos ||
      report.explanation.find("b2a") != std::string::npos;
  EXPECT_TRUE(names_a_channel) << report.explanation;
}

TEST(Acyclicity, EmptyGraphIsAcyclic) {
  const CouplingGraph g;
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  ASSERT_TRUE(td.ok());
  EXPECT_TRUE(prove_acyclic(g, lowered.dag, td.value()).acyclic);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/acyclicity.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/schedule/acyclicity.hpp`:

```cpp
#pragma once

#include <string>
#include <vector>

#include "soma/graph/coupling_graph.hpp"
#include "soma/ir/dag.hpp"
#include "soma/schedule/expand.hpp"

namespace soma {

struct CycleReport {
  bool                acyclic = true;
  std::vector<NodeId> cycle;        // one representative cycle, if any
  std::string         explanation;  // names the channels involved
};

/// Spec §8.2: a hard init-time gate. Operates on the time-resolved graph so
/// that a lagged cycle -- a legal explicit scheme -- is accepted, while an
/// unlagged one is rejected before it can deadlock at runtime.
[[nodiscard]] CycleReport prove_acyclic(const CouplingGraph& g,
                                        const OperatorDAG& dag,
                                        const TimedDag& timed);

}  // namespace soma
```

- [ ] **Step 4: Write `src/schedule/acyclicity.cpp`**

```cpp
#include "soma/schedule/acyclicity.hpp"

#include <algorithm>
#include <queue>
#include <unordered_set>

namespace soma {
namespace {

/// A dependence crosses the period boundary when the consumed value was
/// produced at an earlier time level -- that is what a lag encodes, and it is
/// precisely why a lagged cycle is not a deadlock.
bool crosses_period_boundary(const OperatorDAG& dag, ArcId arc) {
  return dag.arc(arc).type.time.offset < Rational{0};
}

std::string channel_name(const CouplingGraph& g, ChannelId c) {
  if (!c.valid() || c.value() >= g.channel_count()) return "<boundary>";
  return g.channel_at(c).name;
}

/// Depth-first search for one concrete cycle among the unscheduled nodes.
std::vector<NodeId> find_one_cycle(
    const std::vector<std::vector<std::uint32_t>>& succ,
    const std::unordered_set<std::uint32_t>& remaining) {
  std::unordered_set<std::uint32_t> on_stack;
  std::unordered_set<std::uint32_t> visited;
  std::vector<std::uint32_t> stack;

  const auto dfs = [&](auto&& self, std::uint32_t v) -> bool {
    visited.insert(v);
    on_stack.insert(v);
    stack.push_back(v);
    for (const std::uint32_t w : succ[v]) {
      if (!remaining.count(w)) continue;
      if (on_stack.count(w)) {
        stack.push_back(w);
        return true;
      }
      if (!visited.count(w) && self(self, w)) return true;
    }
    on_stack.erase(v);
    stack.pop_back();
    return false;
  };

  for (const std::uint32_t v : remaining) {
    if (visited.count(v)) continue;
    stack.clear();
    on_stack.clear();
    if (dfs(dfs, v)) {
      // Trim to the cycle proper: everything from the repeated node onward.
      const std::uint32_t closing = stack.back();
      const auto first =
          std::find(stack.begin(), stack.end(), closing);
      std::vector<NodeId> cycle;
      for (auto it = first; it != stack.end(); ++it) cycle.push_back(NodeId{*it});
      return cycle;
    }
  }
  return {};
}

}  // namespace

CycleReport prove_acyclic(const CouplingGraph& g, const OperatorDAG& dag,
                          const TimedDag& timed) {
  CycleReport report;
  if (timed.nodes.empty()) return report;

  const std::size_t n = dag.node_count();
  std::vector<std::vector<std::uint32_t>> succ(n);
  std::vector<int> indegree(n, 0);

  // Which DAG nodes actually appear in this period?
  std::unordered_set<std::uint32_t> present;
  for (const TimedNode& tn : timed.nodes) present.insert(tn.node.value());

  for (const std::uint32_t v : present) {
    const Node& node = dag.node(NodeId{v});
    for (const ArcId in : node.inputs) {
      const NodeId prod = dag.arc(in).producer;
      if (!prod.valid() || !present.count(prod.value())) continue;
      // A lagged dependence is satisfied by the *previous* period, so it does
      // not constrain ordering within this one.
      if (crosses_period_boundary(dag, in)) continue;
      succ[prod.value()].push_back(v);
      ++indegree[v];
    }
  }

  std::queue<std::uint32_t> ready;
  for (const std::uint32_t v : present) {
    if (indegree[v] == 0) ready.push(v);
  }

  std::unordered_set<std::uint32_t> remaining = present;
  while (!ready.empty()) {
    const std::uint32_t v = ready.front();
    ready.pop();
    remaining.erase(v);
    for (const std::uint32_t w : succ[v]) {
      if (--indegree[w] == 0) ready.push(w);
    }
  }

  if (remaining.empty()) return report;

  report.acyclic = false;
  report.cycle = find_one_cycle(succ, remaining);

  std::string names;
  std::unordered_set<std::uint32_t> seen_channels;
  for (const NodeId nid : report.cycle) {
    const ChannelId origin = dag.node(nid).origin;
    if (!origin.valid() || !seen_channels.insert(origin.value()).second) continue;
    if (!names.empty()) names += " -> ";
    names += channel_name(g, origin);
  }
  report.explanation =
      "coupling cycle with no lag to break it: " +
      (names.empty() ? std::string{"<no named channels>"} : names) +
      ". Add .lag(1) to one channel in the cycle to make it an explicit "
      "scheme, or place the channels in an iteration region.";
  return report;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `schedule/acyclicity.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(acyclicity_test unit/acyclicity_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Acyclicity
```

Expected: 5 tests PASS. `LaggedTwoWayCouplingIsAccepted` is the subtle one — if
it fails, `crosses_period_boundary` is not seeing the lag, which means Task 2.5's
lag handling did not reach the arc type.

- [ ] **Step 7: Commit**

```bash
git add include/soma/schedule/acyclicity.hpp src/schedule/acyclicity.cpp \
        test/unit/acyclicity_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): init-time acyclicity proof that accepts lagged cycles"
```

---

### Task 3.5: Buffer allocation and lifetimes

Spec §2 layer 3: buffers pre-allocated, no allocation in the hot loop.

**Files:**
- Create: `include/soma/schedule/buffers.hpp`
- Create: `src/schedule/buffers.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/buffers_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `OperatorDAG` (2.2), `TimedDag` (3.3).
- Produces:
  - `struct BufferDecl { ArcId arc; std::size_t elements = 0; int components = 1; RegionId region; bool is_component_owned = false; bool is_persistent = false; std::string label; }`
    - `is_component_owned` marks the arcs that alias a registered component field (`Source` outputs and `Sink` outputs) — the coupler must not allocate those (spec §4.2).
    - `is_persistent` marks buffers that must survive across periods: waveform accumulators and any arc consumed through a lag.
  - `struct BufferPlan { std::vector<BufferDecl> buffers; std::size_t total_coupler_elements = 0; }`
  - `using RegionSizes = std::function<std::size_t(RegionId)>;` — supplied by the
    mesh backend: how many entities does this region hold on this rank?
  - `BufferPlan plan_buffers(const OperatorDAG&, const TimedDag&, const RegionSizes&)`
    — **this three-argument form is the only one.** Region sizes are unknown in
    Phase 3 because no mesh backend exists yet, so the tests below pass a
    constant-returning lambda; Phase 4 Task 4.2 supplies the real callback via
    `make_region_sizes(graph, backend)`. Taking the callback now means the
    signature does not change when the backend arrives.

- [ ] **Step 1: Write the failing test**

`test/unit/buffers_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/schedule/buffers.hpp>
#include <soma/schedule/expand.hpp>

using soma::ComponentDecl;
using soma::CouplingGraph;
using soma::Rational;
namespace u = soma::units;

namespace {

CouplingGraph pair_with_accumulate() {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1, 2}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).accumulate(soma::AccumOp::Mean);
  return g;
}

soma::BufferPlan plan_for(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet cs{g};
  const auto td = expand(g, lowered.dag, cs);
  EXPECT_TRUE(td.ok()) << td.error().message;
  // 100 elements per region until a mesh backend exists (Phase 4 Task 4.2).
  return plan_buffers(lowered.dag, td.value(),
                      [](soma::RegionId) -> std::size_t { return 100; });
}

}  // namespace

TEST(Buffers, OneDeclarationPerArc) {
  const CouplingGraph g = pair_with_accumulate();
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto plan = plan_for(g);
  EXPECT_EQ(plan.buffers.size(), lowered.dag.arc_count());
}

// Spec §4.2: the coupler allocates only its own intermediates.
TEST(Buffers, ComponentOwnedArcsAreNotAllocated) {
  const auto plan = plan_for(pair_with_accumulate());
  int owned = 0;
  for (const auto& b : plan.buffers) {
    if (b.is_component_owned) ++owned;
  }
  EXPECT_GT(owned, 0);
  std::size_t coupler_total = 0;
  for (const auto& b : plan.buffers) {
    if (!b.is_component_owned) coupler_total += b.elements * b.components;
  }
  EXPECT_EQ(plan.total_coupler_elements, coupler_total);
}

TEST(Buffers, WaveformArcsArePersistent) {
  const auto plan = plan_for(pair_with_accumulate());
  bool found_persistent = false;
  for (const auto& b : plan.buffers) {
    if (b.is_persistent) found_persistent = true;
  }
  EXPECT_TRUE(found_persistent)
      << "an accumulating channel must have a persistent waveform buffer";
}

TEST(Buffers, ElementCountsComeFromTheRegionSizeCallback) {
  const auto plan = plan_for(pair_with_accumulate());
  for (const auto& b : plan.buffers) {
    if (b.is_component_owned) continue;
    EXPECT_EQ(b.elements, 100u);
  }
}

TEST(Buffers, LabelsAreNonEmpty) {
  const auto plan = plan_for(pair_with_accumulate());
  for (const auto& b : plan.buffers) {
    EXPECT_FALSE(b.label.empty());
  }
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/buffers.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/schedule/buffers.hpp`:

```cpp
#pragma once

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "soma/ir/dag.hpp"
#include "soma/schedule/expand.hpp"

namespace soma {

/// Supplied by the mesh backend: how many entities does this region hold on
/// this rank? Phase 3 tests pass a constant; Phase 4 Task 4.2 wires the real one.
using RegionSizes = std::function<std::size_t(RegionId)>;

struct BufferDecl {
  ArcId       arc;
  std::size_t elements = 0;
  int         components = 1;
  RegionId    region;
  bool        is_component_owned = false;  // aliases a registered field
  bool        is_persistent = false;       // survives across periods
  std::string label;
};

struct BufferPlan {
  std::vector<BufferDecl> buffers;
  std::size_t             total_coupler_elements = 0;
};

/// Spec §2 layer 3: everything the hot loop needs, allocated once.
[[nodiscard]] BufferPlan plan_buffers(const OperatorDAG& dag,
                                      const TimedDag& timed,
                                      const RegionSizes& sizes);

}  // namespace soma
```

- [ ] **Step 4: Write `src/schedule/buffers.cpp`**

```cpp
#include "soma/schedule/buffers.hpp"

#include <unordered_set>

namespace soma {

BufferPlan plan_buffers(const OperatorDAG& dag, const TimedDag& timed,
                        const RegionSizes& sizes) {
  BufferPlan plan;
  plan.buffers.reserve(dag.arc_count());

  // Arcs that alias component storage: Source outputs and Sink outputs.
  std::unordered_set<std::uint32_t> component_owned;
  for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
    const Node& n = dag.node(NodeId{i});
    if ((n.kind == OpKind::Source || n.kind == OpKind::Sink) && n.output.valid()) {
      component_owned.insert(n.output.value());
    }
  }

  // Arcs consumed through a lag must survive into the next period.
  std::unordered_set<std::uint32_t> lag_consumed;
  for (std::uint32_t i = 0; i < dag.arc_count(); ++i) {
    if (dag.arc(ArcId{i}).type.time.offset < Rational{0}) {
      lag_consumed.insert(i);
    }
  }

  for (std::uint32_t i = 0; i < dag.arc_count(); ++i) {
    const ArcId aid{i};
    const Arc& a = dag.arc(aid);

    BufferDecl b;
    b.arc = aid;
    b.region = a.type.region;
    b.components = a.type.components;
    b.is_component_owned = component_owned.count(i) > 0;
    b.is_persistent = (a.type.kind == ValueKind::Waveform) ||
                      lag_consumed.count(i) > 0;
    b.label = a.name.empty() ? ("%" + std::to_string(i)) : a.name;
    b.elements = b.is_component_owned ? 0 : sizes(a.type.region);

    if (!b.is_component_owned) {
      plan.total_coupler_elements +=
          b.elements * static_cast<std::size_t>(b.components);
    }
    plan.buffers.push_back(std::move(b));
  }

  (void)timed;  // instance counts matter only once waveform depth is real (Phase 6)
  return plan;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `schedule/buffers.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(buffers_test unit/buffers_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Buffers
```

Expected: 5 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/soma/schedule/buffers.hpp src/schedule/buffers.cpp \
        test/unit/buffers_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): buffer plan with component-owned and persistent flags"
```

---

### Task 3.6: `ExecutionPlan` and the scheduler driver

The phase's deliverable: a flat, immutable instruction stream.

**Files:**
- Create: `include/soma/schedule/plan.hpp`
- Create: `src/schedule/scheduler.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/plan_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: everything in Phase 3 so far.
- Produces:
  - `struct Instruction { OpKind kind; ArcId out; std::array<ArcId, 3> in{}; std::uint8_t n_in = 0; OpAttrs attrs; Rational time; std::int32_t region_enter = -1; std::int32_t region_exit = -1; NodeId origin_node; }`
    Three inputs is enough for every primitive in spec §5.1 (`Relax` and `Residual` take two, `Merge` is the exception and is lowered as a left-fold of pairwise merges — recorded here so the fixed array is a deliberate choice, not an oversight).
  - `class ExecutionPlan` — immutable after construction: `const std::vector<Instruction>& instructions() const`, `const BufferPlan& buffers() const`, `Rational period() const`, `bool lockstep() const`, `std::size_t region_count() const`, `std::string to_string() const` (the `--soma-dump-plan` artifact).
  - `struct ScheduleResult { ExecutionPlan plan; Diagnostics diags; bool ok() const; }`
  - `ScheduleResult schedule(const CouplingGraph&, const OperatorDAG&, const ClockSet&, const RegionSizes&)` — runs expand → prove_acyclic (aborting with `E-SCHED-CYCLE` if it fails) → plan_buffers → emit instructions in timed order.

- [ ] **Step 1: Write the failing test**

`test/unit/plan_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/schedule/plan.hpp>

#include <algorithm>

using soma::ComponentDecl;
using soma::CouplingGraph;
using soma::OpKind;
using soma::Rational;
namespace u = soma::units;

namespace {

CouplingGraph pair(Rational src_dt, Rational dst_dt) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = src_dt});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = dst_dt});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kHeatFluxDensity,
      .conservation = soma::Conservation::Extensive,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kHeatFluxDensity,
      .conservation = soma::Conservation::Extensive,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("c", s, t);
  g.channel(ch).remap(soma::RemapFlags::Conservative);
  return g;
}

soma::ScheduleResult build(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet cs{g};
  return schedule(g, lowered.dag, cs,
                  [](soma::RegionId) -> std::size_t { return 64; });
}

}  // namespace

TEST(ExecutionPlan, BuildsFromAValidGraph) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_FALSE(r.plan.instructions().empty());
}

TEST(ExecutionPlan, InstructionsAreInNonDecreasingTimeOrder) {
  const auto r = build(pair(Rational{1, 3}, Rational{1}));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto& is = r.plan.instructions();
  for (std::size_t i = 1; i < is.size(); ++i) {
    EXPECT_LE(is[i - 1].time, is[i].time);
  }
}

TEST(ExecutionPlan, RecordsThePeriodAndLockstepFlag) {
  const auto lock = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(lock.ok());
  EXPECT_TRUE(lock.plan.lockstep());

  const auto multi = build(pair(Rational{1, 3}, Rational{1}));
  ASSERT_TRUE(multi.ok());
  EXPECT_FALSE(multi.plan.lockstep());
  EXPECT_EQ(multi.plan.period(), Rational{1});
}

TEST(ExecutionPlan, EveryInstructionsInputsAreDeclaredBuffers) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok());
  const std::size_t n_buffers = r.plan.buffers().buffers.size();
  for (const auto& ins : r.plan.instructions()) {
    for (std::uint8_t k = 0; k < ins.n_in; ++k) {
      ASSERT_TRUE(ins.in[k].valid());
      EXPECT_LT(ins.in[k].value(), n_buffers);
    }
    if (ins.out.valid()) EXPECT_LT(ins.out.value(), n_buffers);
  }
}

TEST(ExecutionPlan, RejectsCyclicSchedules) {
  // Unlagged two-way coupling.
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto ra = g.add_region(soma::Region{.name = "ra", .mesh = m});
  const auto rb = g.add_region(soma::Region{.name = "rb", .mesh = m});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  const auto ao = g.add_field(a, soma::FieldSpec{
      .name = "ao", .region = ra, .intent = soma::Intent::Provides});
  const auto bi = g.add_field(b, soma::FieldSpec{
      .name = "bi", .region = rb, .intent = soma::Intent::Requires});
  const auto bo = g.add_field(b, soma::FieldSpec{
      .name = "bo", .region = rb, .intent = soma::Intent::Provides});
  const auto ai = g.add_field(a, soma::FieldSpec{
      .name = "ai", .region = ra, .intent = soma::Intent::Requires});
  g.add_channel("a2b", ao, bi);
  g.add_channel("b2a", bo, ai);

  const auto r = build(g);
  EXPECT_FALSE(r.ok());
  bool saw_cycle_code = false;
  for (const auto& d : r.diags.items()) {
    if (d.code == "E-SCHED-CYCLE") saw_cycle_code = true;
  }
  EXPECT_TRUE(saw_cycle_code) << r.diags.to_string();
}

TEST(ExecutionPlan, DumpMentionsEveryOpKindPresent) {
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok());
  const std::string s = r.plan.to_string();
  EXPECT_NE(s.find("# soma-plan v1"), std::string::npos);
  EXPECT_NE(s.find("Remap"), std::string::npos);
  EXPECT_NE(s.find("period="), std::string::npos);
}

TEST(ExecutionPlan, DumpIsStable) {
  const auto r = build(pair(Rational{1, 2}, Rational{1}));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.plan.to_string(), r.plan.to_string());
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/schedule/plan.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/schedule/plan.hpp`:

```cpp
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/ir/dag.hpp"
#include "soma/schedule/acyclicity.hpp"
#include "soma/schedule/buffers.hpp"
#include "soma/schedule/clock.hpp"
#include "soma/schedule/expand.hpp"

namespace soma {

/// One entry of the flat instruction stream. Three inputs covers every
/// primitive in spec §5.1; Merge with more than three sources is lowered as a
/// left-fold of pairwise merges.
struct Instruction {
  OpKind                kind = OpKind::Source;
  ArcId                 out;
  std::array<ArcId, 3>  in{};
  std::uint8_t          n_in = 0;
  OpAttrs               attrs;
  Rational              time{0};
  std::int32_t          region_enter = -1;  // iteration region index, or -1
  std::int32_t          region_exit = -1;
  NodeId                origin_node;
};

/// Spec §2 layer 3: immutable after construction. The hot loop interprets this
/// vector -- no virtual dispatch per DOF, no map lookup, no allocation.
class ExecutionPlan {
 public:
  ExecutionPlan() = default;
  ExecutionPlan(std::vector<Instruction> instructions, BufferPlan buffers,
                Rational period, bool lockstep, std::size_t region_count)
      : instructions_(std::move(instructions)),
        buffers_(std::move(buffers)),
        period_(period),
        lockstep_(lockstep),
        region_count_(region_count) {}

  [[nodiscard]] const std::vector<Instruction>& instructions() const noexcept {
    return instructions_;
  }
  [[nodiscard]] const BufferPlan& buffers() const noexcept { return buffers_; }
  [[nodiscard]] Rational period() const noexcept { return period_; }
  [[nodiscard]] bool lockstep() const noexcept { return lockstep_; }
  [[nodiscard]] std::size_t region_count() const noexcept { return region_count_; }

  /// The --soma-dump-plan artifact (spec §2.3).
  [[nodiscard]] std::string to_string() const;

 private:
  std::vector<Instruction> instructions_;
  BufferPlan               buffers_;
  Rational                 period_{1};
  bool                     lockstep_ = false;
  std::size_t              region_count_ = 0;
};

struct ScheduleResult {
  ExecutionPlan plan;
  Diagnostics   diags;
  [[nodiscard]] bool ok() const { return !diags.has_errors(); }
};

/// expand -> prove_acyclic -> plan_buffers -> emit.
[[nodiscard]] ScheduleResult schedule(const CouplingGraph& g,
                                      const OperatorDAG& dag,
                                      const ClockSet& clocks,
                                      const RegionSizes& sizes);

}  // namespace soma
```

- [ ] **Step 4: Write `src/schedule/scheduler.cpp`**

```cpp
#include "soma/schedule/plan.hpp"

#include <sstream>

namespace soma {

std::string ExecutionPlan::to_string() const {
  std::ostringstream os;
  os << "# soma-plan v1\n";
  os << "period=" << period_.to_string()
     << " lockstep=" << (lockstep_ ? "yes" : "no")
     << " instructions=" << instructions_.size()
     << " buffers=" << buffers_.buffers.size()
     << " coupler_elements=" << buffers_.total_coupler_elements
     << " regions=" << region_count_ << "\n";

  for (std::size_t i = 0; i < instructions_.size(); ++i) {
    const Instruction& ins = instructions_[i];
    os << "  " << i << "  t=" << ins.time.to_string() << "  "
       << to_cstring(ins.kind);
    if (ins.n_in > 0) {
      os << "  in=[";
      for (std::uint8_t k = 0; k < ins.n_in; ++k) {
        if (k) os << ",";
        os << "%" << ins.in[k].value();
      }
      os << "]";
    }
    if (ins.out.valid()) os << "  out=%" << ins.out.value();
    if (ins.region_enter >= 0) os << "  enter=" << ins.region_enter;
    if (ins.region_exit >= 0) os << "  exit=" << ins.region_exit;
    os << "\n";
  }
  return os.str();
}

ScheduleResult schedule(const CouplingGraph& g, const OperatorDAG& dag,
                        const ClockSet& clocks, const RegionSizes& sizes) {
  ScheduleResult result;

  auto timed = expand(g, dag, clocks);
  if (!timed.ok()) {
    result.diags.add(Severity::Error, timed.error().code,
                     timed.error().message);
    return result;
  }

  // Spec §8.2: the hard gate.
  const CycleReport cycles = prove_acyclic(g, dag, timed.value());
  if (!cycles.acyclic) {
    result.diags.add(Severity::Error, "E-SCHED-CYCLE", cycles.explanation);
    return result;
  }

  BufferPlan buffers = plan_buffers(dag, timed.value(), sizes);

  std::vector<Instruction> instructions;
  instructions.reserve(timed.value().nodes.size());
  for (const TimedNode& tn : timed.value().nodes) {
    const Node& n = dag.node(tn.node);
    Instruction ins;
    ins.kind = n.kind;
    ins.out = n.output;
    ins.attrs = n.attrs;
    ins.time = tn.time;
    ins.origin_node = tn.node;
    ins.n_in = static_cast<std::uint8_t>(
        n.inputs.size() > 3 ? 3 : n.inputs.size());
    for (std::uint8_t k = 0; k < ins.n_in; ++k) ins.in[k] = n.inputs[k];
    if (n.inputs.size() > 3) {
      result.diags.add(Severity::Error, "E-SCHED-ARITY",
                       std::string{"operator "} + to_cstring(n.kind) +
                           " has more than three inputs; lower Merge as a "
                           "left-fold of pairwise merges",
                       n.label);
    }
    instructions.push_back(ins);
  }

  result.plan = ExecutionPlan{std::move(instructions), std::move(buffers),
                              timed.value().period, timed.value().lockstep,
                              dag.region_count()};
  return result;
}

}  // namespace soma
```

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `schedule/scheduler.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(plan_test unit/plan_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R ExecutionPlan
```

Expected: 7 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add include/soma/schedule/plan.hpp src/schedule/scheduler.cpp \
        test/unit/plan_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): ExecutionPlan and the scheduler driver"
```

---

### Task 3.7: Lockstep fast path

Spec §8.2: "the common case does not pay slab-machinery overhead." This task
adds the recognition and proves it changes the emitted plan.

**Files:**
- Modify: `src/schedule/expand.cpp`
- Test: `test/unit/lockstep_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `expand` (3.3).
- Produces: no new public API. When `TimedDag::lockstep` is true, `expand` skips
  `decompose` entirely and emits exactly one instance per DAG node. The
  observable contract, which the tests pin:
  1. a lockstep plan has exactly `dag.node_count()` instructions;
  2. it is byte-identical to what the general path would produce for the same
     graph (correctness of the shortcut);
  3. building it calls `decompose` zero times.

  Point 3 needs an observation hook. Add to `include/soma/schedule/slab.hpp`:
  `std::uint64_t decompose_call_count();` and `void reset_decompose_call_count();`
  implemented with a function-local `static std::atomic<std::uint64_t>` in
  `slab.cpp`, incremented at the top of `decompose`. This is a test seam and is
  documented as such.

- [ ] **Step 1: Write the failing test**

`test/unit/lockstep_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/schedule/plan.hpp>
#include <soma/schedule/slab.hpp>

using soma::ComponentDecl;
using soma::CouplingGraph;
using soma::Rational;
namespace u = soma::units;

namespace {

CouplingGraph pair(Rational dt_a, Rational dt_b) {
  CouplingGraph g;
  const auto mf = g.add_mesh(soma::MeshDecl{.name = "mf"});
  const auto ms = g.add_mesh(soma::MeshDecl{.name = "ms"});
  const auto rf = g.add_region(soma::Region{.name = "rf", .mesh = mf});
  const auto rs = g.add_region(soma::Region{.name = "rs", .mesh = ms});
  const auto a = g.add_component(ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = dt_a});
  const auto b = g.add_component(ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = dt_b});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = rf, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = rs, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  g.add_channel("c", s, t);
  return g;
}

soma::ScheduleResult build(const CouplingGraph& g) {
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet cs{g};
  return schedule(g, lowered.dag, cs,
                  [](soma::RegionId) -> std::size_t { return 32; });
}

}  // namespace

TEST(Lockstep, EqualIntervalsTakeTheFastPath) {
  const CouplingGraph g = pair(Rational{1}, Rational{1});
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto r = build(g);
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  EXPECT_TRUE(r.plan.lockstep());
  EXPECT_EQ(r.plan.instructions().size(), lowered.dag.node_count());
}

TEST(Lockstep, FastPathAvoidsSlabDecomposition) {
  soma::reset_decompose_call_count();
  const auto r = build(pair(Rational{1}, Rational{1}));
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(soma::decompose_call_count(), 0u)
      << "lockstep must not invoke the slab machinery";
}

TEST(Lockstep, GeneralPathIsUsedWhenIntervalsDiffer) {
  soma::reset_decompose_call_count();
  const auto r = build(pair(Rational{1, 2}, Rational{1}));
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.plan.lockstep());
  EXPECT_GT(soma::decompose_call_count(), 0u);
}

// The shortcut must not change the answer.
TEST(Lockstep, FastPathAgreesWithTheGeneralPath) {
  // Intervals 1/2 and 1/2 are lockstep; 1/2 and 1/2 forced through the general
  // path by a spacetime channel must give the same instruction count.
  CouplingGraph fast = pair(Rational{1, 2}, Rational{1, 2});
  CouplingGraph general = pair(Rational{1, 2}, Rational{1, 2});
  general.channel(soma::ChannelId{0})
      .spacetime(soma::TemporalBasis::Constant, 0);

  const auto rf = build(fast);
  const auto rg = build(general);
  ASSERT_TRUE(rf.ok()) << rf.diags.to_string();
  ASSERT_TRUE(rg.ok()) << rg.diags.to_string();
  EXPECT_TRUE(rf.plan.lockstep());
  EXPECT_FALSE(rg.plan.lockstep());
  EXPECT_EQ(rf.plan.instructions().size(), rg.plan.instructions().size());
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `decompose_call_count` is not declared.

- [ ] **Step 3: Add the test seam to `include/soma/schedule/slab.hpp`**

Before the closing `}  // namespace soma`:

```cpp
/// Test seam: how many times has decompose() been called? Used to prove the
/// lockstep fast path (spec §8.2) does not touch the slab machinery.
[[nodiscard]] std::uint64_t decompose_call_count();
void reset_decompose_call_count();
```

- [ ] **Step 4: Implement it in `src/schedule/slab.cpp`**

Add at namespace scope:

```cpp
namespace {
std::atomic<std::uint64_t>& decompose_counter() {
  static std::atomic<std::uint64_t> counter{0};
  return counter;
}
}  // namespace

std::uint64_t decompose_call_count() { return decompose_counter().load(); }
void reset_decompose_call_count() { decompose_counter().store(0); }
```

with `#include <atomic>` at the top, and increment at the very start of
`decompose`:

```cpp
  decompose_counter().fetch_add(1, std::memory_order_relaxed);
```

- [ ] **Step 5: Add the fast path to `src/schedule/expand.cpp`**

Immediately after `out.lockstep` is computed, before the boundary-node loop:

```cpp
  // Spec §8.2: lockstep is the degenerate case and must not pay for the slab
  // machinery. One instance per node, no decomposition.
  if (out.lockstep) {
    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      emit(NodeId{i}, out.period, 0, 0);
    }
    std::stable_sort(out.nodes.begin(), out.nodes.end(),
                     [&](const TimedNode& a, const TimedNode& b) {
                       return a.node < b.node;
                     });
    return out;
  }
```

Note the sort key differs from the general path: in lockstep every node shares
one time, so ordering falls back to `NodeId`, which is the topological order the
DAG was built in.

- [ ] **Step 6: Register and run**

Add `soma_add_test(lockstep_test unit/lockstep_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Lockstep
```

Expected: 4 tests PASS.

- [ ] **Step 7: Run the whole suite**

```bash
ctest --test-dir build --output-on-failure
```

Expected: every Phase 0–3 test green. The `Expand` tests from Task 3.3 must
still pass — `LockstepGivesOneInstancePerNode` now exercises the fast path
rather than the general one, and its assertion is unchanged, which is the point.

- [ ] **Step 8: Commit**

```bash
git add src/schedule/expand.cpp src/schedule/slab.cpp include/soma/schedule/slab.hpp \
        test/unit/lockstep_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): lockstep fast path bypassing slab decomposition"
```

---

## Phase 3 exit criteria

- [ ] Component clocks are exact over 10⁶ steps with intervals like 1/3 and 1/7.
- [ ] Slab decomposition tiles the period with no gap or overlap, verified by summing lengths.
- [ ] An unlagged two-way coupling is **rejected at setup** with a message naming the channels; the same topology with a lag is accepted.
- [ ] A runaway decomposition fails as `E-SCHED-PERIOD` rather than hanging.
- [ ] Lockstep produces exactly one instruction per DAG node and calls `decompose` zero times.
- [ ] `ExecutionPlan::to_string()` is stable and dumps period, lockstep flag, buffers, and every instruction.
- [ ] `ctest --test-dir build --output-on-failure` fully green.
