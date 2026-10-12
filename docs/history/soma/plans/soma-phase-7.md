:::{note}
**Historical document.** Preserved verbatim from `upstream/soma` (parent repository `coupe`,
commit `e52b981`, imported 2026-09-14 per `UPSTREAMS.lock`). Describes the design, build and
handoff state of the retired `soma::` library as it stood at that commit -- not FUSE's current
architecture. See [](../../../dev/soma-import.md) for what from this design was ported, superseded,
or remains open.
:::

# SOMA Phase 7 — Iteration regions, accelerators, Demo A

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §5.3 (iteration regions), §5.1 (`Residual`, `Accelerate`), §10 level 4, §11 Demo A
**Milestone:** the heterogeneous coupled heat equations (DNWR benchmark) converge on non-matching meshes with independent intervals, through the explicit → Picard → Aitken → IQN-ILS ladder, with measured spatial and temporal order.

The distinguishing claim of this phase, from spec §5.3: **SOMA derives the checkpoint set rather than asking the user for it.** preCICE pushes that onto the adapter via `requiresWritingCheckpoint()`; here dataflow analysis over the SSA form computes it and the runtime verifies the component actually saved.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/num/convergence.hpp` | `ConvergenceMeasure`, composition with `&` |
| `include/soma/num/accelerator.hpp` | abstract `Accelerator` + factory |
| `src/num/accel_relaxation.cpp` | constant relaxation, Aitken |
| `src/num/accel_quasi_newton.cpp` | IQN-ILS, IQN-IMVJ |
| `src/num/qr_filter.cpp` | QR1/QR2/QR3 column filtering |
| `src/ir/checkpoint.cpp` | derived checkpoint-set analysis |
| `src/runtime/iteration.cpp` | region execution: iterate, measure, accelerate, roll back |
| `examples/dnwr_pair/` | Demo A |

---

### Task 7.1: Convergence measures

**Files:**
- Create: `include/soma/num/convergence.hpp`
- Test: `test/unit/convergence_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: kernels (4.4).
- Produces:
  - `enum class MeasureKind { Relative, Absolute, AbsoluteOrRelative, ResidualRelative }`
  - `struct ConvergenceMeasure { MeasureKind kind = MeasureKind::Relative; double tolerance = 1e-6; std::string field; }` with `bool satisfied(double residual_norm, double reference_norm) const`
  - `class ConvergenceCriterion` — a conjunction of measures, matching spec §5.3's `Converged::Relative(...) & Converged::Absolute(...)`: `void add(ConvergenceMeasure)`, `bool all_satisfied(const std::vector<double>& residual_norms, const std::vector<double>& reference_norms) const`, `std::size_t size() const`, `std::string report(...) const`
  - free helpers in namespace `soma::Converged`: `ConvergenceMeasure Relative(double tol, std::string field)`, `Absolute(...)`, `AbsoluteOrRelative(...)`, and `ConvergenceCriterion operator&(ConvergenceMeasure, ConvergenceMeasure)` plus `operator&(ConvergenceCriterion, ConvergenceMeasure)`.
  - Per preCICE's documented behaviour, when several measures are combined **all** must be satisfied before the window advances.
  - Division guard: a reference norm of exactly zero makes `Relative` fall back to absolute comparison rather than producing NaN. Tested.

- [ ] **Step 1: Write the failing test**

`test/unit/convergence_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/num/convergence.hpp>

using soma::ConvergenceCriterion;
using soma::ConvergenceMeasure;
using soma::MeasureKind;
namespace C = soma::Converged;

TEST(ConvergenceMeasure, RelativeComparesAgainstTheReference) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-7, 1.0));
  EXPECT_FALSE(m.satisfied(1e-5, 1.0));
  EXPECT_TRUE(m.satisfied(1e-5, 1e3));   // relative to a large reference
}

TEST(ConvergenceMeasure, AbsoluteIgnoresTheReference) {
  const auto m = C::Absolute(1e-8, "flux");
  EXPECT_TRUE(m.satisfied(1e-9, 1e6));
  EXPECT_FALSE(m.satisfied(1e-7, 1e-6));
}

TEST(ConvergenceMeasure, AbsoluteOrRelativeAcceptsEither) {
  const auto m = C::AbsoluteOrRelative(1e-8, "T");
  EXPECT_TRUE(m.satisfied(1e-9, 1.0));      // absolute passes
  EXPECT_TRUE(m.satisfied(1e-6, 1e3));      // relative passes
  EXPECT_FALSE(m.satisfied(1e-6, 1.0));     // neither
}

// A zero reference must not produce NaN.
TEST(ConvergenceMeasure, ZeroReferenceFallsBackToAbsolute) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-9, 0.0));
  EXPECT_FALSE(m.satisfied(1.0, 0.0));
}

TEST(ConvergenceMeasure, ExactlyAtToleranceCounts) {
  const auto m = C::Absolute(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-6, 1.0));
}

// Spec §5.3: `Relative(...) & Absolute(...)`.
TEST(ConvergenceCriterion, ConjunctionRequiresEveryMeasure) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  EXPECT_EQ(c.size(), 2u);
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9}, {1.0, 1.0}));
  EXPECT_FALSE(c.all_satisfied({1e-7, 1e-6}, {1.0, 1.0}));  // second fails
  EXPECT_FALSE(c.all_satisfied({1e-5, 1e-9}, {1.0, 1.0}));  // first fails
}

TEST(ConvergenceCriterion, ChainsMoreThanTwoMeasures) {
  const ConvergenceCriterion c = C::Relative(1e-6, "a") &
                                 C::Absolute(1e-8, "b") &
                                 C::Relative(1e-4, "c");
  EXPECT_EQ(c.size(), 3u);
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9, 1e-5}, {1.0, 1.0, 1.0}));
}

TEST(ConvergenceCriterion, EmptyCriterionIsSatisfied) {
  const ConvergenceCriterion c;
  EXPECT_TRUE(c.all_satisfied({}, {}));
}

TEST(ConvergenceCriterion, ReportNamesEveryFieldAndItsState) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  const std::string s = c.report({1e-7, 1e-6}, {1.0, 1.0});
  EXPECT_NE(s.find("T"), std::string::npos);
  EXPECT_NE(s.find("flux"), std::string::npos);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/num/convergence.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/num/convergence.hpp`:

```cpp
#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace soma {

enum class MeasureKind { Relative, Absolute, AbsoluteOrRelative, ResidualRelative };

struct ConvergenceMeasure {
  MeasureKind kind = MeasureKind::Relative;
  double      tolerance = 1e-6;
  std::string field;

  /// A zero reference falls back to absolute comparison rather than NaN.
  [[nodiscard]] bool satisfied(double residual_norm,
                               double reference_norm) const {
    const bool absolute_ok = residual_norm <= tolerance;
    const bool have_ref = reference_norm > 0.0;
    const bool relative_ok =
        have_ref ? (residual_norm / reference_norm <= tolerance) : absolute_ok;

    switch (kind) {
      case MeasureKind::Absolute:           return absolute_ok;
      case MeasureKind::Relative:
      case MeasureKind::ResidualRelative:   return relative_ok;
      case MeasureKind::AbsoluteOrRelative: return absolute_ok || relative_ok;
    }
    return false;
  }
};

/// Conjunction: when several measures are combined, all must hold before the
/// window advances (matching preCICE's documented behaviour).
class ConvergenceCriterion {
 public:
  void add(ConvergenceMeasure m) { measures_.push_back(std::move(m)); }

  [[nodiscard]] std::size_t size() const noexcept { return measures_.size(); }
  [[nodiscard]] const std::vector<ConvergenceMeasure>& measures() const noexcept {
    return measures_;
  }

  [[nodiscard]] bool all_satisfied(
      const std::vector<double>& residual_norms,
      const std::vector<double>& reference_norms) const {
    for (std::size_t i = 0; i < measures_.size(); ++i) {
      const double r = i < residual_norms.size() ? residual_norms[i] : 0.0;
      const double ref = i < reference_norms.size() ? reference_norms[i] : 0.0;
      if (!measures_[i].satisfied(r, ref)) return false;
    }
    return true;
  }

  [[nodiscard]] std::string report(
      const std::vector<double>& residual_norms,
      const std::vector<double>& reference_norms) const {
    std::ostringstream os;
    for (std::size_t i = 0; i < measures_.size(); ++i) {
      const double r = i < residual_norms.size() ? residual_norms[i] : 0.0;
      const double ref = i < reference_norms.size() ? reference_norms[i] : 0.0;
      os << measures_[i].field << ": " << r << " / tol "
         << measures_[i].tolerance
         << (measures_[i].satisfied(r, ref) ? " [ok]" : " [pending]");
      if (i + 1 < measures_.size()) os << ", ";
    }
    return os.str();
  }

 private:
  std::vector<ConvergenceMeasure> measures_;
};

namespace Converged {

[[nodiscard]] inline ConvergenceMeasure Relative(double tol, std::string field) {
  return ConvergenceMeasure{MeasureKind::Relative, tol, std::move(field)};
}
[[nodiscard]] inline ConvergenceMeasure Absolute(double tol, std::string field) {
  return ConvergenceMeasure{MeasureKind::Absolute, tol, std::move(field)};
}
[[nodiscard]] inline ConvergenceMeasure AbsoluteOrRelative(double tol,
                                                           std::string field) {
  return ConvergenceMeasure{MeasureKind::AbsoluteOrRelative, tol,
                            std::move(field)};
}

}  // namespace Converged

[[nodiscard]] inline ConvergenceCriterion operator&(ConvergenceMeasure a,
                                                    ConvergenceMeasure b) {
  ConvergenceCriterion c;
  c.add(std::move(a));
  c.add(std::move(b));
  return c;
}

[[nodiscard]] inline ConvergenceCriterion operator&(ConvergenceCriterion c,
                                                    ConvergenceMeasure m) {
  c.add(std::move(m));
  return c;
}

}  // namespace soma
```

- [ ] **Step 4: Register and run**

Add `soma_add_test(convergence_test unit/convergence_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Convergence
```

Expected: 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/num/convergence.hpp test/unit/convergence_test.cpp \
        test/CMakeLists.txt
git commit -m "feat(soma): convergence measures with conjunction semantics"
```

---

### Task 7.2: Derived checkpoint sets

Spec §5.3. Dataflow analysis over the SSA form determines exactly which buffers
are mutable state crossing the region boundary.

**Files:**
- Create: `src/ir/checkpoint.cpp`
- Modify: `include/soma/ir/lowering.hpp`
- Modify: `src/ir/regions.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/checkpoint_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `OperatorDAG` (2.2), `IterationRegion` (2.7).
- Produces:
  - `struct CheckpointSet { std::vector<ArcId> coupler_arcs; std::vector<ComponentId> components; std::string explanation; }`
  - `CheckpointSet derive_checkpoint_set(const CouplingGraph&, const OperatorDAG&, const OperatorDAG::IterationRegion&)`
  - `Diagnostics verify_rollback_support(const CouplingGraph&, const CheckpointSet&, const ComponentSet&)` — `E-ITER-NOROLLBACK` when a component in the set returns `supports_rollback() == false`.
  - `build_iteration_regions` (2.7) now fills `carried` from `coupler_arcs`.

  The analysis, stated precisely so it can be checked:
  1. An arc is **loop-carried** if it is produced inside the region and consumed
     inside the region *at a previous iterate* (`time.iterate < 0`) — the
     `Relax` previous-value input is the canonical case.
  2. An arc is **live-in** if produced outside the region and consumed inside;
     it needs no checkpoint because re-executing the region recomputes from it
     unchanged.
  3. An arc is **live-out** if produced inside and consumed outside; it needs no
     checkpoint either, because a rejected iterate never escapes.
  4. A **component** needs a checkpoint if any arc it owns (a `Source` output)
     is consumed inside the region *and* the component advances inside the
     region — its internal state moves and must be restorable.
  5. Pure functions of live-in values are recomputed, never saved.

- [ ] **Step 1: Write the failing test**

`test/unit/checkpoint_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>

#include <algorithm>

using soma::CouplingGraph;
using soma::derive_checkpoint_set;
using soma::Rational;
namespace u = soma::units;

namespace {

/// Two-way coupling inside an iteration region, with relaxation on the return
/// channel so there is a genuine loop-carried value.
CouplingGraph iterated_two_way(bool relax_return) {
  CouplingGraph g;
  const auto ma = g.add_mesh(soma::MeshDecl{.name = "ma"});
  const auto mb = g.add_mesh(soma::MeshDecl{.name = "mb"});
  const auto ra = g.add_region(soma::Region{.name = "ra", .mesh = ma});
  const auto rb = g.add_region(soma::Region{.name = "rb", .mesh = mb});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(a, soma::FieldSpec{
      .name = "a_out", .region = ra, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto b_in = g.add_field(b, soma::FieldSpec{
      .name = "b_in", .region = rb, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto b_out = g.add_field(b, soma::FieldSpec{
      .name = "b_out", .region = rb, .units = u::kHeatFluxDensity,
      .intent = soma::Intent::Provides});
  const auto a_in = g.add_field(a, soma::FieldSpec{
      .name = "a_in", .region = ra, .units = u::kHeatFluxDensity,
      .intent = soma::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated scheme");
  if (relax_return) g.channel(back).relax(0.7);

  g.add_iteration_region(soma::IterationRegionDecl{
      .name = "fsi", .channels = {fwd, back}, .max_iterations = 30});
  return g;
}

}  // namespace

TEST(Checkpoint, RegionCarriesTheRelaxedValue) {
  const CouplingGraph g = iterated_two_way(/*relax_return=*/true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  ASSERT_EQ(lowered.dag.region_count(), 1u);
  const auto& region = lowered.dag.region_at(soma::RegionIdx{0});
  EXPECT_FALSE(region.carried.empty())
      << "a relaxed channel has a previous-iterate value to carry";
}

TEST(Checkpoint, ComponentsInsideTheRegionNeedCheckpoints) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_EQ(set.components.size(), 2u)
      << "both components advance inside the region";
}

TEST(Checkpoint, ExplanationIsNonEmpty) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_FALSE(set.explanation.empty())
      << "the derived set must be explainable to the user";
}

// The point of deriving rather than asking: pure recomputable values are not
// saved.
TEST(Checkpoint, PureValuesAreNotCheckpointed) {
  const CouplingGraph g = iterated_two_way(/*relax_return=*/false);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const auto set = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  // With no relaxation there is no previous-iterate coupler buffer to save;
  // only component state needs restoring.
  EXPECT_TRUE(set.coupler_arcs.empty())
      << "without relaxation nothing coupler-side is loop-carried";
  EXPECT_FALSE(set.components.empty());
}

TEST(Checkpoint, DerivedSetIsStable) {
  const CouplingGraph g = iterated_two_way(true);
  const auto lowered = lower(g);
  const auto a = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  const auto b = derive_checkpoint_set(
      g, lowered.dag, lowered.dag.region_at(soma::RegionIdx{0}));
  EXPECT_EQ(a.coupler_arcs, b.coupler_arcs);
  EXPECT_EQ(a.components, b.components);
}

TEST(Checkpoint, NoRegionMeansNoCheckpoints) {
  CouplingGraph g;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m"});
  const auto r = g.add_region(soma::Region{.name = "r", .mesh = m});
  const auto a = g.add_component(soma::ComponentDecl{.name = "a"});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r, .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r, .intent = soma::Intent::Requires});
  g.add_channel("c", s, t);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  EXPECT_EQ(lowered.dag.region_count(), 0u);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `derive_checkpoint_set` is not declared.

- [ ] **Step 3: Declare and implement**

Add to `include/soma/ir/lowering.hpp`:

```cpp
/// Spec §5.3: SOMA derives the checkpoint set rather than asking the user.
/// preCICE pushes this onto the adapter via requiresWritingCheckpoint();
/// dataflow over the SSA form computes it here.
struct CheckpointSet {
  std::vector<ArcId>       coupler_arcs;  // loop-carried coupler buffers
  std::vector<ComponentId> components;    // components whose state must persist
  std::string              explanation;
};

[[nodiscard]] CheckpointSet derive_checkpoint_set(
    const CouplingGraph& g, const OperatorDAG& dag,
    const OperatorDAG::IterationRegion& region);
```

`src/ir/checkpoint.cpp` implements rules 1–5 above. Then in
`src/ir/regions.cpp`, after building `region.residuals`:

```cpp
    const CheckpointSet cp = derive_checkpoint_set(g, ctx.dag, region);
    region.carried = cp.coupler_arcs;
```

- [ ] **Step 4: Run**

Add `soma_add_test(checkpoint_test unit/checkpoint_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Checkpoint
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ir/checkpoint.cpp include/soma/ir/lowering.hpp src/ir/regions.cpp \
        test/unit/checkpoint_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): derived checkpoint sets from SSA dataflow"
```

---

### Task 7.3: Accelerators

Spec §5.3: accelerators operate on residual history and know no physics. One
implementation family serves Picard, Aitken, secant, Steffensen, IQN-ILS, and
IQN-IMVJ.

**Files:**
- Create: `include/soma/num/accelerator.hpp`
- Create: `src/num/accel_relaxation.cpp`
- Create: `src/num/accel_quasi_newton.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/accelerator_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: kernels (4.4).
- Produces:
  - `enum class AccelKind { None, Constant, Aitken, Secant, Steffensen, IQN_ILS, IQN_IMVJ }`
  - `enum class FilterKind { None, QR1, QR2, QR3 }`
  - `struct AccelConfig { AccelKind kind = AccelKind::Constant; double initial_relaxation = 0.1; double omega = 0.5; int reuse = 8; FilterKind filter = FilterKind::QR2; double limit = 1e-2; bool scale_fields = true; }`
  - `class Accelerator` (abstract): `virtual void reset()`, `virtual Result<void> update(const double* x_old, const double* x_new, double* x_out, std::size_t n, int iteration)`, `virtual std::string name() const`, `virtual std::size_t columns() const` (retained secant columns; 0 for relaxation methods)
  - `Result<std::unique_ptr<Accelerator>> make_accelerator(const AccelConfig&)`
  - Multi-field support: `update` takes the concatenated vector; `AccelConfig::scale_fields` plus `void set_field_partition(std::vector<std::size_t> offsets)` enables per-field scaling, which the literature says is required for parallel multi-field quasi-Newton.

  Semantics, each tested:
  - **Constant**: `x_out = x_old + omega * (x_new - x_old)`.
  - **Aitken**: adaptive `omega` from successive residuals; first iteration uses `initial_relaxation`.
  - **IQN-ILS**: least-squares over difference matrices `V` (residual differences) and `W` (value differences), solved by QR; first iteration is under-relaxed by `initial_relaxation` because no Jacobian estimate exists yet.
  - **IQN-IMVJ**: retains information across time windows in the inverse-Jacobian approximation.
  - Filtering is applied before the solve for all quasi-Newton variants.

- [ ] **Step 1: Write the failing test**

`test/unit/accelerator_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/num/accelerator.hpp>

#include <cmath>
#include <vector>

using soma::AccelConfig;
using soma::AccelKind;
using soma::FilterKind;
using soma::make_accelerator;

namespace {

/// Fixed-point map x -> G(x) with a known contraction factor. Every
/// accelerator must converge on this; the interesting question is how fast.
struct Contraction {
  double factor;
  std::vector<double> fixed_point;

  std::vector<double> apply(const std::vector<double>& x) const {
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
      out[i] = fixed_point[i] + factor * (x[i] - fixed_point[i]);
    }
    return out;
  }
};

/// Iterate to convergence, returning the iteration count (-1 if it did not).
int iterate_to_convergence(soma::Accelerator& accel, const Contraction& g,
                           double tol, int max_iters) {
  std::vector<double> x(g.fixed_point.size(), 0.0);
  for (int k = 0; k < max_iters; ++k) {
    const std::vector<double> gx = g.apply(x);
    double residual = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
      const double d = gx[i] - x[i];
      residual += d * d;
    }
    if (std::sqrt(residual) < tol) return k;

    std::vector<double> next(x.size());
    EXPECT_TRUE(accel.update(x.data(), gx.data(), next.data(), x.size(), k).ok());
    x = next;
  }
  return -1;
}

}  // namespace

TEST(Accelerator, ConstantRelaxationBlends) {
  AccelConfig cfg{.kind = AccelKind::Constant, .omega = 0.25};
  auto a = make_accelerator(cfg);
  ASSERT_TRUE(a.ok()) << a.error().message;

  const std::vector<double> old_x{0.0, 10.0};
  const std::vector<double> new_x{10.0, 0.0};
  std::vector<double> out(2, 0.0);
  ASSERT_TRUE(a.value()->update(old_x.data(), new_x.data(), out.data(), 2, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 2.5);
  EXPECT_DOUBLE_EQ(out[1], 7.5);
}

TEST(Accelerator, ConstantRelaxationConvergesOnAContraction) {
  AccelConfig cfg{.kind = AccelKind::Constant, .omega = 1.0};
  auto a = make_accelerator(cfg).value();
  const Contraction g{.factor = 0.5, .fixed_point = {1.0, 2.0, 3.0}};
  const int iters = iterate_to_convergence(*a, g, 1e-10, 200);
  EXPECT_GT(iters, 0);
  EXPECT_LT(iters, 200);
}

TEST(Accelerator, AitkenBeatsConstantRelaxation) {
  const Contraction g{.factor = 0.9, .fixed_point = {1.0, 2.0}};  // slow

  auto constant = make_accelerator(
      AccelConfig{.kind = AccelKind::Constant, .omega = 1.0}).value();
  auto aitken = make_accelerator(
      AccelConfig{.kind = AccelKind::Aitken, .initial_relaxation = 0.5}).value();

  const int n_const = iterate_to_convergence(*constant, g, 1e-10, 500);
  const int n_aitken = iterate_to_convergence(*aitken, g, 1e-10, 500);
  ASSERT_GT(n_const, 0);
  ASSERT_GT(n_aitken, 0);
  EXPECT_LT(n_aitken, n_const)
      << "aitken " << n_aitken << " vs constant " << n_const;
}

TEST(Accelerator, IqnIlsConvergesFastOnALinearProblem) {
  const Contraction g{.factor = 0.95, .fixed_point = {1.0, 2.0, 3.0, 4.0}};
  auto a = make_accelerator(AccelConfig{.kind = AccelKind::IQN_ILS,
                                        .initial_relaxation = 0.1,
                                        .reuse = 8,
                                        .filter = FilterKind::QR2,
                                        .limit = 1e-2}).value();
  const int iters = iterate_to_convergence(*a, g, 1e-10, 100);
  ASSERT_GT(iters, 0);
  // A quasi-Newton method should solve a linear fixed point in about as many
  // iterations as there are degrees of freedom, far fewer than 1/(1-0.95)=20.
  EXPECT_LT(iters, 15) << "IQN-ILS took " << iters << " iterations";
}

TEST(Accelerator, FirstIterationIsUnderRelaxed) {
  // No Jacobian estimate exists yet, so iteration 0 must use
  // initial_relaxation rather than attempting a quasi-Newton step.
  AccelConfig cfg{.kind = AccelKind::IQN_ILS, .initial_relaxation = 0.25};
  auto a = make_accelerator(cfg).value();
  const std::vector<double> old_x{0.0};
  const std::vector<double> new_x{4.0};
  std::vector<double> out(1, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 1, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 1.0);   // 0 + 0.25 * (4 - 0)
}

TEST(Accelerator, ResetClearsHistory) {
  auto a = make_accelerator(AccelConfig{.kind = AccelKind::IQN_ILS}).value();
  const std::vector<double> old_x{0.0, 0.0};
  const std::vector<double> new_x{1.0, 1.0};
  std::vector<double> out(2, 0.0);
  for (int k = 0; k < 3; ++k) {
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 2, k).ok());
  }
  EXPECT_GT(a->columns(), 0u);
  a->reset();
  EXPECT_EQ(a->columns(), 0u);
}

TEST(Accelerator, RespectsTheReuseLimit) {
  AccelConfig cfg{.kind = AccelKind::IQN_ILS, .reuse = 3,
                  .filter = FilterKind::None};
  auto a = make_accelerator(cfg).value();
  std::vector<double> old_x(4, 0.0), out(4, 0.0);
  for (int k = 0; k < 10; ++k) {
    std::vector<double> new_x(4);
    for (std::size_t i = 0; i < 4; ++i) {
      new_x[i] = static_cast<double>(k) * 0.1 + static_cast<double>(i);
    }
    ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 4, k).ok());
    old_x = out;
  }
  EXPECT_LE(a->columns(), 3u) << "reuse limit must bound the column count";
}

TEST(Accelerator, EveryKindIsConstructible) {
  for (const AccelKind k : {AccelKind::Constant, AccelKind::Aitken,
                            AccelKind::Secant, AccelKind::Steffensen,
                            AccelKind::IQN_ILS, AccelKind::IQN_IMVJ}) {
    const auto a = make_accelerator(AccelConfig{.kind = k});
    ASSERT_TRUE(a.ok()) << "kind " << static_cast<int>(k);
    EXPECT_FALSE(a.value()->name().empty());
  }
}

TEST(Accelerator, NoneKindIsAPassThrough) {
  auto a = make_accelerator(AccelConfig{.kind = AccelKind::None}).value();
  const std::vector<double> old_x{1.0};
  const std::vector<double> new_x{5.0};
  std::vector<double> out(1, 0.0);
  ASSERT_TRUE(a->update(old_x.data(), new_x.data(), out.data(), 1, 0).ok());
  EXPECT_DOUBLE_EQ(out[0], 5.0) << "plain Picard takes the new value";
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/num/accelerator.hpp` file not found.

- [ ] **Step 3: Write the header and the two implementation files**

`include/soma/num/accelerator.hpp` declares the enums, `AccelConfig`, the
abstract `Accelerator`, and `make_accelerator`.

`src/num/accel_relaxation.cpp` implements `None`, `Constant`, `Aitken`,
`Secant`, and `Steffensen` — all of which need only the previous residual.

`src/num/accel_quasi_newton.cpp` implements `IQN_ILS` and `IQN_IMVJ`:
1. Maintain `V` (columns of residual differences) and `W` (value differences),
   newest column first, truncated to `reuse`.
2. Apply the column filter (Task 7.4) before solving.
3. Solve `min ||V*alpha - r||` by Householder QR, then
   `x_out = x_new + W*alpha - V*alpha`.
4. Iteration 0 falls back to `x_old + initial_relaxation * (x_new - x_old)`.
5. `IQN_IMVJ` additionally retains an explicit inverse-Jacobian update across
   `reset()` calls at window boundaries.

The QR is a dense Householder factorisation written inline — the matrices are
`n × reuse` with `reuse ≤ 8`, so LAPACK would be a dependency without a benefit,
and the core must build without one.

- [ ] **Step 4: Update `src/CMakeLists.txt`** — add both files.

- [ ] **Step 5: Run**

Add `soma_add_test(accelerator_test unit/accelerator_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Accelerator
```

Expected: 9 tests PASS. `IqnIlsConvergesFastOnALinearProblem` is the one that
proves the quasi-Newton step is actually being taken rather than silently
degrading to relaxation.

- [ ] **Step 6: Commit**

```bash
git add include/soma/num/accelerator.hpp src/num/accel_relaxation.cpp \
        src/num/accel_quasi_newton.cpp test/unit/accelerator_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): accelerator family from relaxation to IQN-IMVJ"
```

---

### Task 7.4: QR column filtering

Required for parallel multi-field quasi-Newton, not optional: without it the
multi-secant system loses column independence and the solve degrades.

**Files:**
- Create: `src/num/qr_filter.cpp`
- Modify: `include/soma/num/accelerator.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/qr_filter_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: nothing beyond the standard library.
- Produces:
  - `struct FilterResult { std::vector<std::size_t> kept; std::size_t dropped = 0; }`
  - `FilterResult filter_columns(FilterKind kind, const std::vector<double>& v_matrix, std::size_t rows, std::size_t cols, double limit)`
  - Semantics, following the documented preCICE behaviour: **QR1** drops a column whose R-diagonal magnitude falls below `limit * ||column||` (typical `limit` 1e-6…1e-5); **QR2** drops based on the ratio to the largest diagonal (typical 1e-3…1e-2); **QR3** is QR2 with pre-scaling weight monitoring, which is cheaper at scale. Newest columns are preferred when dropping.

- [ ] **Step 1: Write the failing test**

`test/unit/qr_filter_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/num/accelerator.hpp>

#include <vector>

using soma::FilterKind;
using soma::filter_columns;

namespace {

/// Column-major rows x cols.
std::vector<double> columns_of(const std::vector<std::vector<double>>& cols) {
  std::vector<double> out;
  for (const auto& c : cols) out.insert(out.end(), c.begin(), c.end());
  return out;
}

}  // namespace

TEST(QrFilter, KeepsIndependentColumns) {
  const auto v = columns_of({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}});
  const auto r = filter_columns(FilterKind::QR2, v, 3, 3, 1e-2);
  EXPECT_EQ(r.kept.size(), 3u);
  EXPECT_EQ(r.dropped, 0u);
}

TEST(QrFilter, DropsAnExactlyDependentColumn) {
  // Third column is the first one again.
  const auto v = columns_of({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}});
  const auto r = filter_columns(FilterKind::QR2, v, 3, 3, 1e-2);
  EXPECT_EQ(r.kept.size(), 2u);
  EXPECT_EQ(r.dropped, 1u);
}

TEST(QrFilter, DropsANearlyDependentColumn) {
  const auto v = columns_of({{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {1.0, 1e-9, 0.0}});
  const auto r = filter_columns(FilterKind::QR2, v, 3, 3, 1e-2);
  EXPECT_EQ(r.dropped, 1u);
}

TEST(QrFilter, NoneKeepsEverything) {
  const auto v = columns_of({{1.0, 0.0}, {1.0, 0.0}});   // dependent
  const auto r = filter_columns(FilterKind::None, v, 2, 2, 1e-2);
  EXPECT_EQ(r.kept.size(), 2u);
  EXPECT_EQ(r.dropped, 0u);
}

TEST(QrFilter, Qr1IsLooserThanQr2AtTheSameLimit) {
  const auto v = columns_of({{1.0, 0.0, 0.0},
                             {0.0, 1.0, 0.0},
                             {1.0, 1e-4, 0.0}});
  const auto q1 = filter_columns(FilterKind::QR1, v, 3, 3, 1e-6);
  const auto q2 = filter_columns(FilterKind::QR2, v, 3, 3, 1e-2);
  EXPECT_GE(q1.kept.size(), q2.kept.size());
}

TEST(QrFilter, KeptIndicesAreSortedAndInRange) {
  const auto v = columns_of({{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}});
  const auto r = filter_columns(FilterKind::QR2, v, 2, 3, 1e-2);
  for (std::size_t i = 1; i < r.kept.size(); ++i) {
    EXPECT_LT(r.kept[i - 1], r.kept[i]);
  }
  for (const std::size_t k : r.kept) EXPECT_LT(k, 3u);
}

TEST(QrFilter, HandlesTheEmptyMatrix) {
  const auto r = filter_columns(FilterKind::QR2, {}, 0, 0, 1e-2);
  EXPECT_TRUE(r.kept.empty());
  EXPECT_EQ(r.dropped, 0u);
}

TEST(QrFilter, DropsAZeroColumn) {
  const auto v = columns_of({{1.0, 0.0}, {0.0, 0.0}});
  const auto r = filter_columns(FilterKind::QR2, v, 2, 2, 1e-2);
  EXPECT_EQ(r.kept.size(), 1u);
  EXPECT_EQ(r.kept[0], 0u);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `filter_columns` is not declared.

- [ ] **Step 3: Declare and implement**

Add to `include/soma/num/accelerator.hpp`:

```cpp
struct FilterResult {
  std::vector<std::size_t> kept;
  std::size_t              dropped = 0;
};

/// Column filtering for quasi-Newton. Required, not optional: without it the
/// multi-secant system loses column independence for parallel multi-field
/// coupling. Typical limits: QR1 1e-6..1e-5, QR2/QR3 1e-3..1e-2.
[[nodiscard]] FilterResult filter_columns(FilterKind kind,
                                          const std::vector<double>& v_matrix,
                                          std::size_t rows, std::size_t cols,
                                          double limit);
```

`src/num/qr_filter.cpp` performs modified Gram–Schmidt, recording each column's
R-diagonal, and applies the per-kind threshold.

- [ ] **Step 4: Run**

Add `soma_add_test(qr_filter_test unit/qr_filter_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R QrFilter
```

Expected: 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/num/qr_filter.cpp include/soma/num/accelerator.hpp \
        test/unit/qr_filter_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): QR column filtering for quasi-Newton acceleration"
```

---

### Task 7.5: Iteration region execution

Wires convergence, checkpointing, and acceleration into the runner.

**Files:**
- Create: `src/runtime/iteration.cpp`
- Modify: `src/runtime/runner.cpp`
- Modify: `include/soma/runtime/runner.hpp`
- Modify: `include/soma/graph/coupling_graph.hpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/verification/iteration_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: 7.1–7.4, `Runner` (4.6).
- Produces:
  - `IterationRegionDecl` (1.4) extended with `ConvergenceCriterion convergence; AccelConfig acceleration; double initial_relaxation = 0.1;` and builder methods matching spec §5.3's syntax: `IterationBuilder g.iterate(std::vector<ChannelId>)` returning a builder with `.until(ConvergenceCriterion)`, `.max_iterations(int)`, `.accelerate(AccelConfig)`, `.initial_relaxation(double)`.
  - `struct IterationStats { int iterations = 0; bool converged = false; std::vector<double> residual_history; }`
  - `RunStats` gains `std::int64_t total_iterations = 0; std::int64_t windows_not_converged = 0;`
  - `Runner::run_period` executes an iteration region as: save checkpoints → loop { run body instructions, compute residual norms, test convergence, if converged break, else accelerate and restore checkpoints } → on exhaustion, report `E-RT-NOCONVERGE` if configured to abort, otherwise count and continue.
  - `RunnerConfig` gains `bool abort_on_nonconvergence = false;`.
  - `verify_rollback_support` runs in `initialize`: a component in a derived checkpoint set that returns `supports_rollback() == false` is `E-ITER-NOROLLBACK`.

- [ ] **Step 1: Write the failing test**

`test/verification/iteration_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>

#include <cmath>
#include <vector>

using soma::AccelConfig;
using soma::AccelKind;
using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::Rational;
using soma::Runner;
using soma::RunnerConfig;
namespace C = soma::Converged;
namespace u = soma::units;

namespace {

/// x <- a + k*y, a linear coupled pair whose fixed point is known exactly.
/// With k_a * k_b < 1 the Picard iteration converges; the fixed point is
/// x* = (a + k_a*b) / (1 - k_a*k_b).
class LinearBlock final : public soma::Component {
 public:
  LinearBlock(std::string name, double a, double k, std::size_t n)
      : name_(std::move(name)), a_(a), k_(k), out_(n, 0.0), in_(n, 0.0) {}

  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    for (const soma::FieldId f : g.component(*cid).fields) {
      const bool is_out = g.field(f).intent == soma::Intent::Provides;
      auto& buf = is_out ? out_ : in_;
      auto r = reg.bind(f, soma::FieldRef{buf.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = buf.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext&) override {
    ++advance_count;
    for (std::size_t i = 0; i < out_.size(); ++i) {
      out_[i] = a_ + k_ * in_[i];
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> save_state() override {
    saved_out_ = out_;
    saved_in_ = in_;
    ++save_count;
    return soma::Result<void>::success();
  }
  soma::Result<void> restore_state() override {
    out_ = saved_out_;
    in_ = saved_in_;
    ++restore_count;
    return soma::Result<void>::success();
  }
  [[nodiscard]] bool supports_rollback() const override { return true; }

  [[nodiscard]] const std::vector<double>& out() const { return out_; }
  int advance_count = 0;
  int save_count = 0;
  int restore_count = 0;

 private:
  std::string         name_;
  double              a_, k_;
  std::vector<double> out_, in_, saved_out_, saved_in_;
};

/// A component that cannot roll back, for the negative test.
class NoRollback final : public soma::Component {
 public:
  explicit NoRollback(std::size_t n) : data_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "norollback"; }
  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("norollback");
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

 private:
  std::vector<double> data_;
};

struct Fixture {
  CouplingGraph        graph;
  soma::SerialMeshBackend backend;
  soma::OperatorDAG    dag;
  soma::ExecutionPlan  plan;
  ComponentSet         components;
  FieldRegistry        registry;
  LinearBlock*         a = nullptr;
  LinearBlock*         b = nullptr;
};

std::unique_ptr<Fixture> build(AccelConfig accel, int max_iters,
                               bool with_rollback = true) {
  auto f = std::make_unique<Fixture>();
  auto& g = f->graph;
  const auto m = g.add_mesh(soma::MeshDecl{.name = "m",
                                           .source = "generated:line:4"});
  const auto r1 = g.add_region(soma::Region{.name = "r1", .mesh = m,
                                            .dimension = 1});
  const auto r2 = g.add_region(soma::Region{.name = "r2", .mesh = m,
                                            .dimension = 1});
  const auto ca = g.add_component(soma::ComponentDecl{
      .name = "A", .ranks = {.first = 0, .count = 1}, .interval = Rational{1}});
  const auto cb = g.add_component(soma::ComponentDecl{
      .name = with_rollback ? "B" : "norollback",
      .ranks = {.first = 1, .count = 1}, .interval = Rational{1}});

  const auto a_out = g.add_field(ca, soma::FieldSpec{
      .name = "a_out", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto b_in = g.add_field(cb, soma::FieldSpec{
      .name = "b_in", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto b_out = g.add_field(cb, soma::FieldSpec{
      .name = "b_out", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto a_in = g.add_field(ca, soma::FieldSpec{
      .name = "a_in", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Requires});

  const auto fwd = g.add_channel("a2b", a_out, b_in);
  const auto back = g.add_channel("b2a", b_out, a_in);
  g.channel(back).lag(1).allow_time_level_mismatch("iterated");

  g.iterate({fwd, back})
      .until(C::Relative(1e-10, "a_out"))
      .max_iterations(max_iters)
      .accelerate(accel);

  EXPECT_TRUE(f->backend.load(g.mesh(m)).ok());
  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  f->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto sched = schedule(g, f->dag, clocks,
                        soma::make_region_sizes(g, f->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  f->plan = sched.plan;

  auto a = std::make_unique<LinearBlock>("A", 1.0, 0.5, 4);
  f->a = a.get();
  f->components.add(std::move(a));
  if (with_rollback) {
    auto b = std::make_unique<LinearBlock>("B", 2.0, 0.4, 4);
    f->b = b.get();
    f->components.add(std::move(b));
  } else {
    f->components.add(std::make_unique<NoRollback>(4));
  }
  return f;
}

soma::Result<void> run(Fixture& f, RunnerConfig cfg, Runner** out = nullptr) {
  static Runner* held = nullptr;
  auto* runner = new Runner{f.graph, f.dag, f.plan, f.backend};
  held = runner;
  if (out) *out = runner;
  for (auto& c : f.components) {
    const auto r = c->register_fields(f.registry, f.graph);
    if (!r.ok()) return r;
  }
  const auto init = runner->initialize(f.components, f.registry, cfg);
  if (!init.ok()) return init;
  return runner->run_period(f.components);
}

}  // namespace

TEST(Iteration, PicardConvergesOnALinearPair) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  Runner* runner = nullptr;
  ASSERT_TRUE(run(*f, RunnerConfig{}, &runner).ok());
  EXPECT_GT(runner->stats().total_iterations, 1);
  EXPECT_EQ(runner->stats().windows_not_converged, 0);

  // Fixed point of x = 1 + 0.5*(2 + 0.4*x) is x = 2 / (1 - 0.2) = 2.5
  for (const double v : f->a->out()) EXPECT_NEAR(v, 2.5, 1e-8);
}

TEST(Iteration, CheckpointsAreSavedAndRestored) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  Runner* runner = nullptr;
  ASSERT_TRUE(run(*f, RunnerConfig{}, &runner).ok());
  EXPECT_GT(f->a->save_count, 0) << "the derived set must include component A";
  EXPECT_GT(f->a->restore_count, 0) << "a rejected iterate must roll back";
}

TEST(Iteration, ComponentsAdvanceOncePerIteration) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  Runner* runner = nullptr;
  ASSERT_TRUE(run(*f, RunnerConfig{}, &runner).ok());
  EXPECT_EQ(f->a->advance_count, runner->stats().total_iterations);
}

TEST(Iteration, AitkenNeedsFewerIterationsThanPicard) {
  auto picard = build(AccelConfig{.kind = AccelKind::None}, 200);
  auto aitken = build(AccelConfig{.kind = AccelKind::Aitken,
                                  .initial_relaxation = 0.5}, 200);
  Runner* rp = nullptr;
  Runner* ra = nullptr;
  ASSERT_TRUE(run(*picard, RunnerConfig{}, &rp).ok());
  ASSERT_TRUE(run(*aitken, RunnerConfig{}, &ra).ok());
  EXPECT_LE(ra->stats().total_iterations, rp->stats().total_iterations);
}

TEST(Iteration, IqnIlsConvergesInFewIterations) {
  auto f = build(AccelConfig{.kind = AccelKind::IQN_ILS,
                             .initial_relaxation = 0.5,
                             .reuse = 8,
                             .filter = soma::FilterKind::QR2,
                             .limit = 1e-2}, 100);
  Runner* runner = nullptr;
  ASSERT_TRUE(run(*f, RunnerConfig{}, &runner).ok());
  EXPECT_LT(runner->stats().total_iterations, 12);
  for (const double v : f->a->out()) EXPECT_NEAR(v, 2.5, 1e-8);
}

TEST(Iteration, ReportsNonConvergenceRatherThanLooping) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 2);  // too few
  Runner* runner = nullptr;
  RunnerConfig cfg;
  cfg.abort_on_nonconvergence = false;
  ASSERT_TRUE(run(*f, cfg, &runner).ok());
  EXPECT_EQ(runner->stats().windows_not_converged, 1);
  EXPECT_EQ(runner->stats().total_iterations, 2);
}

TEST(Iteration, AbortsOnNonConvergenceWhenConfigured) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 2);
  RunnerConfig cfg;
  cfg.abort_on_nonconvergence = true;
  const auto r = run(*f, cfg);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-RT-NOCONVERGE");
}

// Spec §5.3: the runtime verifies the component can do what the derived set
// requires of it.
TEST(Iteration, RejectsAComponentThatCannotRollBack) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 10,
                 /*with_rollback=*/false);
  const auto r = run(*f, RunnerConfig{});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-ITER-NOROLLBACK");
}

TEST(Iteration, ResidualHistoryDecreasesMonotonicallyForPicard) {
  auto f = build(AccelConfig{.kind = AccelKind::None}, 100);
  Runner* runner = nullptr;
  ASSERT_TRUE(run(*f, RunnerConfig{}, &runner).ok());
  // A contraction must reduce the residual every iteration.
  EXPECT_GT(runner->stats().total_iterations, 2);
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `g.iterate(...)` is not a member of `CouplingGraph`.

- [ ] **Step 3: Add the builder to `CouplingGraph`**

In `include/soma/graph/coupling_graph.hpp`, extend `IterationRegionDecl` and add
an `IterationBuilder` mirroring `ChannelBuilder`, plus
`IterationBuilder iterate(std::vector<ChannelId>)` which appends a declaration
and returns a builder over it.

- [ ] **Step 4: Implement region execution**

`src/runtime/iteration.cpp` holds the loop; `src/runtime/runner.cpp` calls it
when the plan has regions. The order within one iteration is fixed and testable:

1. iteration 0: `save_state()` on every component in the derived set, snapshot
   every carried coupler buffer;
2. execute the region's body instructions;
3. compute each residual arc's norm and its reference norm;
4. test the convergence criterion — if satisfied, exit the loop;
5. if `iteration + 1 == max_iterations`, record non-convergence and exit;
6. otherwise call the accelerator, `restore_state()` on the checkpoint set,
   write the accelerated values into the carried buffers, and repeat.

- [ ] **Step 5: Run**

Add `soma_add_test(iteration_test verification/iteration_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Iteration
```

Expected: 9 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/runtime/iteration.cpp src/runtime/runner.cpp \
        include/soma/runtime/runner.hpp include/soma/graph/coupling_graph.hpp \
        test/verification/iteration_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): iteration region execution with derived checkpointing"
```

---

### Task 7.6: Demo A — the DNWR heat-transfer pair

Spec §11 Demo A. Heterogeneous coupled heat equations on non-matching meshes,
the Dirichlet–Neumann waveform-relaxation benchmark.

**Files:**
- Create: `examples/dnwr_pair/main.cpp`
- Create: `examples/dnwr_pair/heat_domain.hpp`
- Create: `examples/dnwr_pair/dnwr.yaml`
- Modify: `examples/CMakeLists.txt`
- Create: `test/verification/demo_a_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: everything through Phase 7.
- Produces:
  - `class HeatDomain : public Component` — 1D heat equation on `[0, 1]` or `[1, 2]` with conductivity `lambda` and capacity `rho_c`, SDIRK2 in time so the component itself is second order and does not mask the coupling order. Exports interface temperature (Dirichlet side) or interface flux (Neumann side); imports the other.
  - `examples/dnwr_pair/main.cpp` — CLI: `--accel {picard,aitken,iqn}`, `--dt-left`, `--dt-right`, `--cells-left`, `--cells-right`, `--basis {constant,linear,dg2}`, `--dump`. Prints iteration counts per window, final L2 error against the analytic solution, and conservation statistics.
  - `dnwr.yaml` — the same setup expressed in YAML, proving the front-ends agree.
  - `test/verification/demo_a_test.cpp` — the acceptance test:

| Check | Assertion |
|---|---|
| Convergence | every window converges within `max_iterations` |
| Accelerator ladder | `iters(IQN-ILS) ≤ iters(Aitken) ≤ iters(Picard)` |
| Spatial order | ≈2 with `dt` fixed small and `dx` refined |
| Temporal order | ≈2 with `dx` fixed fine, `dt` refined, `Linear` basis |
| Independent intervals | left `dt = 1/30`, right `dt = 1/50` converges and conserves |
| Non-matching meshes | 40 vs 60 cells; conservation to 1e-10 |
| YAML agreement | the YAML-built graph is `structurally_equal` to the C++-built one |
| Determinism | two runs are bit-identical |

- [ ] **Step 1: Write the failing acceptance test**

`test/verification/demo_a_test.cpp` — written against the interfaces above,
following the same fixture pattern as Task 7.5, with these test bodies:
`EveryWindowConverges`, `AcceleratorLadderIsMonotone`, `SpatialOrderIsTwo`,
`TemporalOrderIsTwo`, `IndependentIntervalsConverge`,
`NonMatchingMeshesConserve`, `YamlMatchesTheCppGraph`, `RunsAreDeterministic`.
Each uses `fit_order` from Phase 6 Task 6.5 for the order checks, with the same
loose-but-diagnostic ranges (`[1.8, 2.3]`).

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `heat_domain.hpp` does not exist.

- [ ] **Step 3: Write `examples/dnwr_pair/heat_domain.hpp`**

SDIRK2 (two-stage, L-stable, `gamma = 1 - sqrt(2)/2`) on a cell-centred FV
discretisation, with a Dirichlet or Neumann interface condition selected at
construction. The analytic reference for the heterogeneous two-domain problem
with matched interface temperature and flux is documented in a comment block at
the top of the file, with its derivation, so the error computation can be
checked independently.

- [ ] **Step 4: Write `examples/dnwr_pair/main.cpp` and `dnwr.yaml`**

Following the Phase 4 `mms_pair/main.cpp` structure, extended with the CLI flags
and the iteration-region setup.

- [ ] **Step 5: Build and run**

```bash
cmake --build build -j
./build/examples/dnwr_pair --accel picard
./build/examples/dnwr_pair --accel iqn
./build/examples/dnwr_pair --accel iqn --dt-left 1/30 --dt-right 1/50 --basis linear
./build/examples/dnwr_pair --dump | head -60
```

Expected: IQN-ILS converges in noticeably fewer iterations than Picard; the
independent-interval run converges and reports conservation error below 1e-10.

- [ ] **Step 6: Run the acceptance test and the full suite**

```bash
ctest --test-dir build --output-on-failure -R DemoA
ctest --test-dir build --output-on-failure
ctest --test-dir build-full --output-on-failure
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add examples/dnwr_pair/ examples/CMakeLists.txt \
        test/verification/demo_a_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): Demo A -- DNWR heat pair with accelerator ladder"
```

---

## Phase 7 exit criteria

- [ ] Convergence measures compose with `&`; all must hold before the window advances.
- [ ] The checkpoint set is **derived** by dataflow, not declared; a component that cannot roll back is rejected at `initialize` with `E-ITER-NOROLLBACK`.
- [ ] Picard, Aitken, and IQN-ILS all converge on a linear pair; IQN-ILS in under 12 iterations.
- [ ] QR filtering drops dependent columns and is on by default for quasi-Newton.
- [ ] Non-convergence is reported, never an infinite loop.
- [ ] **Demo A**: DNWR pair converges on non-matching meshes with independent intervals, measured spatial and temporal order ≈2, conservation to 1e-10.
- [ ] The YAML and C++ front-ends produce structurally equal graphs.
- [ ] Both `ctest` runs fully green.

**Still deferred:** JFNK/monolithic residual assembly (spec §5.5) — it needs a
different component contract (`Residual` evaluation rather than `advance`) and
PETSc `SNES`. The `Residual` operator and region machinery built here are its
attachment points.
