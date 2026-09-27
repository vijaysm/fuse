# SOMA Phase 6 — Waveforms, space-time projection, independent intervals

**Parent plan:** `2026-09-02-soma-implementation.md`
**Spec sections:** §4.4 (waveforms), §7 (space-time remap), §7.2 (two tiers), §7.4 (conservation check), §8.1 (independent by default), §10 level 4 (order of accuracy)
**Milestone:** two components with **unrelated** coupling intervals (1/3 and 1/5) exchange a field whose **space-time** flux integral is conserved, and the coupled scheme's measured temporal order matches the temporal basis used.

This is the phase that delivers "each component talks at whatever interval it wants." Phase 3 built the slab machinery; Phase 6 puts real numerics on it.

---

## Scope: Tier 1 only

Per spec §7.2, this phase implements **tensor-product space-time**: spatial
supermesh weights `W` (Phase 5) ⊗ temporal projection `T`, factorising as
`M = T ⊗ W` because the spatial mesh is static over the window. That gives exact
space-time flux conservation for static meshes with arbitrary independent time
steps.

**Tier 2 — true 4D pentatope intersection for moving meshes — is not built
here.** The `RemapPlan` already carries `temporal`/`temporal_degree` (Phase 5
Task 5.2) so the seam exists, and Task 6.3 documents where a Tier 2
implementation would attach.

---

## File structure

| File | Responsibility |
|---|---|
| `include/soma/num/waveform.hpp` | `Waveform` — samples plus interpolation policy |
| `include/soma/num/temporal_basis.hpp` | evaluation and projection for each basis |
| `src/num/temporal_basis.cpp` | Legendre/B-spline coefficient assembly |
| `include/soma/remap/spacetime.hpp` | `TemporalProjection`, slab-integral weights |
| `src/remap/spacetime.cpp` | tensor-product application |
| `src/schedule/expand.cpp` | (modify) independent-interval instancing |
| `src/runtime/runner.cpp` | (modify) real `Sample`/`Interpolate`/`Extrapolate` |
| `test/verification/temporal_order_test.cpp` | the order-of-accuracy study |

---

### Task 6.1: `Waveform`

Spec §4.4: a field sampled at several points across a window, plus an
interpolation policy. Explicit coupling is the degenerate case — one sample,
constant interpolation — not a separate path.

**Files:**
- Create: `include/soma/num/waveform.hpp`
- Test: `test/unit/waveform_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Rational` (0.3), `TemporalBasis` (0.6).
- Produces: `class Waveform` with
  - `Waveform(std::size_t elements, std::size_t capacity, TemporalBasis basis, int degree)`
  - `void reset(Rational window_begin, Rational window_end)` — clears samples, sets the window
  - `Result<void> push(Rational t, const double* values)` — records a sample; `E-WF-FULL` past capacity, `E-WF-ORDER` if `t` is not strictly increasing, `E-WF-WINDOW` if `t` is outside `[begin, end]`
  - `Result<void> evaluate(Rational t, double* out) const` — interpolate **inside** the window; `E-WF-EMPTY` if no samples, `E-WF-OUTSIDE` if `t` is beyond `[t_first, t_last]` (extrapolation is a different operator by design — spec §5.1)
  - `Result<void> integrate(Rational a, Rational b, double* out) const` — `∫ₐᵇ f dt`, the primitive the space-time conservation check needs
  - `Result<void> mean(double* out) const` — `integrate(window) / (end - begin)`
  - accessors `std::size_t sample_count()`, `std::size_t elements()`, `TemporalBasis basis()`, `int degree()`, `Rational sample_time(std::size_t)`, `const double* sample_values(std::size_t)`, `Rational window_begin()`, `Rational window_end()`
  - `bool is_degenerate() const` — one sample with `Constant` basis: the explicit-coupling case

  Interpolation per basis: `Constant` → nearest preceding sample (piecewise
  constant, exact for degree 0); `Linear` → linear between bracketing samples;
  `DG(p)` → least-squares polynomial of degree `min(p, n-1)` over the window,
  evaluated at `t`; `Spline(p)` → for `p == 3`, a natural cubic through the
  samples. `integrate` uses the same representation: exact for `Constant` and
  `Linear`, Gauss–Legendre with `⌈(p+1)/2⌉` points for `DG`, and per-interval
  exact integration of the cubic pieces for `Spline`.

- [ ] **Step 1: Write the failing test**

`test/unit/waveform_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/num/waveform.hpp>

#include <cmath>
#include <vector>

using soma::Rational;
using soma::TemporalBasis;
using soma::Waveform;

namespace {

/// Fill a 1-element waveform with f(t) at the given times.
template <class F>
Waveform sampled(TemporalBasis basis, int degree,
                 const std::vector<Rational>& times, F f) {
  Waveform w{1, times.size(), basis, degree};
  w.reset(times.front(), times.back());
  for (const Rational t : times) {
    const double v = f(t.to_double());
    EXPECT_TRUE(w.push(t, &v).ok());
  }
  return w;
}

double eval(const Waveform& w, Rational t) {
  double out = 0.0;
  EXPECT_TRUE(w.evaluate(t, &out).ok());
  return out;
}

}  // namespace

TEST(Waveform, StartsEmpty) {
  Waveform w{4, 8, TemporalBasis::Linear, 1};
  EXPECT_EQ(w.sample_count(), 0u);
  EXPECT_EQ(w.elements(), 4u);
  EXPECT_FALSE(w.evaluate(Rational{0}, nullptr).ok());
}

TEST(Waveform, RecordsSamplesInOrder) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double a = 1.0, b = 2.0;
  ASSERT_TRUE(w.push(Rational{0}, &a).ok());
  ASSERT_TRUE(w.push(Rational{1, 2}, &b).ok());
  EXPECT_EQ(w.sample_count(), 2u);
  EXPECT_EQ(w.sample_time(1), Rational(1, 2));
  EXPECT_DOUBLE_EQ(w.sample_values(1)[0], 2.0);
}

TEST(Waveform, RejectsOutOfOrderSamples) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{1, 2}, &v).ok());
  const auto r = w.push(Rational{1, 4}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-ORDER");
}

TEST(Waveform, RejectsSamplesOutsideTheWindow) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  const auto r = w.push(Rational{2}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-WINDOW");
}

TEST(Waveform, RejectsOverfill) {
  Waveform w{1, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  ASSERT_TRUE(w.push(Rational{1, 2}, &v).ok());
  const auto r = w.push(Rational{1}, &v);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-WF-FULL");
}

// Spec §4.4: explicit coupling is the degenerate waveform, not a separate path.
TEST(Waveform, SingleSampleConstantIsTheExplicitCase) {
  Waveform w{1, 1, TemporalBasis::Constant, 0};
  w.reset(Rational{0}, Rational{1});
  const double v = 7.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  EXPECT_TRUE(w.is_degenerate());
  EXPECT_DOUBLE_EQ(eval(w, Rational{0}), 7.0);
}

TEST(Waveform, ConstantBasisHoldsThePrecedingSample) {
  const auto w = sampled(TemporalBasis::Constant, 0,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return t; });
  EXPECT_DOUBLE_EQ(eval(w, Rational{1, 4}), 0.0);   // still the t=0 sample
  EXPECT_DOUBLE_EQ(eval(w, Rational{3, 4}), 0.5);   // the t=1/2 sample
}

TEST(Waveform, LinearBasisIsExactOnALinearFunction) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 3.0 * t + 1.0; });
  EXPECT_NEAR(eval(w, Rational{1, 3}), 2.0, 1e-14);
  EXPECT_NEAR(eval(w, Rational{2, 3}), 3.0, 1e-14);
}

TEST(Waveform, DgDegreeTwoIsExactOnAQuadratic) {
  const auto w = sampled(TemporalBasis::DG, 2,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return 2.0 * t * t - t + 3.0; });
  for (const Rational t : {Rational{1, 4}, Rational{1, 3}, Rational{3, 4}}) {
    const double x = t.to_double();
    EXPECT_NEAR(eval(w, t), 2.0 * x * x - x + 3.0, 1e-12) << "at t=" << x;
  }
}

TEST(Waveform, EvaluateRefusesToExtrapolate) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1, 2}},
                         [](double t) { return t; });
  double out = 0.0;
  const auto r = w.evaluate(Rational{3, 4}, &out);
  ASSERT_FALSE(r.ok()) << "extrapolation is Extrapolate's job (spec §5.1)";
  EXPECT_EQ(r.error().code, "E-WF-OUTSIDE");
}

TEST(Waveform, IntegratesAConstantExactly) {
  const auto w = sampled(TemporalBasis::Constant, 0,
                         {Rational{0}, Rational{1, 2}},
                         [](double) { return 4.0; });
  double out = 0.0;
  ASSERT_TRUE(w.integrate(Rational{0}, Rational{1}, &out).ok());
  EXPECT_NEAR(out, 4.0, 1e-14);
}

TEST(Waveform, IntegratesALinearFunctionExactly) {
  // int_0^1 (2t) dt = 1
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  double out = 0.0;
  ASSERT_TRUE(w.integrate(Rational{0}, Rational{1}, &out).ok());
  EXPECT_NEAR(out, 1.0, 1e-14);
}

// The primitive the space-time conservation check depends on.
TEST(Waveform, IntegratesAQuadraticExactlyWithDgTwo) {
  // int_0^1 (3t^2) dt = 1
  const auto w = sampled(TemporalBasis::DG, 2,
                         {Rational{0}, Rational{1, 2}, Rational{1}},
                         [](double t) { return 3.0 * t * t; });
  double out = 0.0;
  ASSERT_TRUE(w.integrate(Rational{0}, Rational{1}, &out).ok());
  EXPECT_NEAR(out, 1.0, 1e-12);
}

TEST(Waveform, IntegratesASubIntervalCorrectly) {
  // int_{1/4}^{3/4} (2t) dt = 1/2
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  double out = 0.0;
  ASSERT_TRUE(w.integrate(Rational{1, 4}, Rational{3, 4}, &out).ok());
  EXPECT_NEAR(out, 0.5, 1e-14);
}

TEST(Waveform, MeanIsTheWindowIntegralOverItsLength) {
  const auto w = sampled(TemporalBasis::Linear, 1, {Rational{0}, Rational{1}},
                         [](double t) { return 2.0 * t; });
  double out = 0.0;
  ASSERT_TRUE(w.mean(&out).ok());
  EXPECT_NEAR(out, 1.0, 1e-14);
}

TEST(Waveform, HandlesMultipleElements) {
  Waveform w{3, 2, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double a[3] = {0.0, 10.0, 20.0};
  const double b[3] = {2.0, 12.0, 22.0};
  ASSERT_TRUE(w.push(Rational{0}, a).ok());
  ASSERT_TRUE(w.push(Rational{1}, b).ok());

  double out[3] = {0, 0, 0};
  ASSERT_TRUE(w.evaluate(Rational{1, 2}, out).ok());
  EXPECT_NEAR(out[0], 1.0, 1e-14);
  EXPECT_NEAR(out[1], 11.0, 1e-14);
  EXPECT_NEAR(out[2], 21.0, 1e-14);
}

TEST(Waveform, ResetClearsSamplesButKeepsCapacity) {
  Waveform w{1, 4, TemporalBasis::Linear, 1};
  w.reset(Rational{0}, Rational{1});
  const double v = 1.0;
  ASSERT_TRUE(w.push(Rational{0}, &v).ok());
  w.reset(Rational{1}, Rational{2});
  EXPECT_EQ(w.sample_count(), 0u);
  EXPECT_EQ(w.window_begin(), Rational{1});
  ASSERT_TRUE(w.push(Rational{1}, &v).ok());
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/num/waveform.hpp` file not found.

- [ ] **Step 3: Write `include/soma/num/temporal_basis.hpp`**

The basis evaluation and integration, separated so Task 6.3 can reuse it:

```cpp
#pragma once

#include <cstddef>
#include <vector>

#include "soma/base/ids.hpp"

namespace soma::num {

/// Gauss-Legendre nodes and weights on [-1, 1] for n = 1..4 points.
/// Exact for polynomials of degree <= 2n-1.
struct GaussRule {
  const double* nodes;
  const double* weights;
  int           count;
};

[[nodiscard]] GaussRule gauss_rule(int points);

/// Least-squares fit of a degree-`degree` polynomial to (t_i, f_i), returning
/// coefficients in the monomial basis about `t0`. Used by TemporalBasis::DG.
[[nodiscard]] std::vector<double> fit_polynomial(const std::vector<double>& t,
                                                 const std::vector<double>& f,
                                                 int degree, double t0);

[[nodiscard]] double eval_polynomial(const std::vector<double>& coeffs,
                                     double t, double t0);

/// Exact integral of the fitted polynomial over [a, b].
[[nodiscard]] double integrate_polynomial(const std::vector<double>& coeffs,
                                          double a, double b, double t0);

/// Natural cubic spline second derivatives, for TemporalBasis::Spline.
[[nodiscard]] std::vector<double> cubic_spline_moments(
    const std::vector<double>& t, const std::vector<double>& f);

[[nodiscard]] double eval_cubic_spline(const std::vector<double>& t,
                                       const std::vector<double>& f,
                                       const std::vector<double>& m, double x);

[[nodiscard]] double integrate_cubic_spline(const std::vector<double>& t,
                                            const std::vector<double>& f,
                                            const std::vector<double>& m,
                                            double a, double b);

}  // namespace soma::num
```

`src/num/temporal_basis.cpp` implements these. `fit_polynomial` builds the
normal equations for the Vandermonde system and solves by Gaussian elimination
with partial pivoting — the systems are at most 4×4, so LAPACK is unnecessary
and would add a dependency to the core.

- [ ] **Step 4: Write `include/soma/num/waveform.hpp`**

Structure:

```cpp
#pragma once

#include <cstddef>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/num/temporal_basis.hpp"

namespace soma {

/// Spec §4.4: the base abstraction for coupled data in time. Explicit coupling
/// is the degenerate case (one sample, Constant basis), not a separate path.
class Waveform {
 public:
  Waveform(std::size_t elements, std::size_t capacity, TemporalBasis basis,
           int degree);

  void reset(Rational window_begin, Rational window_end);

  Result<void> push(Rational t, const double* values);

  /// Interpolate INSIDE the sampled range. Beyond it is Extrapolate's job.
  [[nodiscard]] Result<void> evaluate(Rational t, double* out) const;

  /// The primitive the space-time conservation check needs.
  [[nodiscard]] Result<void> integrate(Rational a, Rational b,
                                       double* out) const;
  [[nodiscard]] Result<void> mean(double* out) const;

  [[nodiscard]] std::size_t sample_count() const noexcept;
  [[nodiscard]] std::size_t elements() const noexcept;
  [[nodiscard]] TemporalBasis basis() const noexcept;
  [[nodiscard]] int degree() const noexcept;
  [[nodiscard]] Rational sample_time(std::size_t i) const;
  [[nodiscard]] const double* sample_values(std::size_t i) const;
  [[nodiscard]] Rational window_begin() const noexcept;
  [[nodiscard]] Rational window_end() const noexcept;
  [[nodiscard]] bool is_degenerate() const noexcept;

 private:
  std::size_t           elements_;
  std::size_t           capacity_;
  TemporalBasis         basis_;
  int                   degree_;
  Rational              begin_{0}, end_{0};
  std::vector<Rational> times_;
  std::vector<double>   values_;  // capacity_ * elements_, row per sample
};

}  // namespace soma
```

The implementation is header-only (tier C per spec §3.1) except for the basis
helpers in `temporal_basis.cpp`. Key points, each pinned by a test:

- `evaluate` with `Constant`: binary search for the last sample with
  `t_i <= t`, copy it.
- `evaluate` with `Linear`: bracket and lerp using exact `Rational` → `double`
  conversion only at the final arithmetic.
- `evaluate` with `DG`: per element, gather the sample values, call
  `fit_polynomial` with `degree = min(degree_, n-1)` and `t0 = begin_`, then
  `eval_polynomial`. Fitting per element per call is acceptable at this stage
  and is the reason Task 6.3 caches projection weights instead.
- `integrate` with `Constant`: sum of `value_i * overlap(interval_i, [a,b])`,
  where `interval_i` runs to the next sample time and the last runs to `end_`.
- `integrate` with `Linear`: trapezoid per bracketing sub-interval, clipped
  to `[a, b]` — exact for a piecewise-linear representation.
- `integrate` with `DG`: `integrate_polynomial`, exact.
- Out-of-window `push` and non-increasing `t` are errors, not clamped.

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `num/temporal_basis.cpp`.

- [ ] **Step 6: Register and run**

Add `soma_add_test(waveform_test unit/waveform_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Waveform
```

Expected: 16 tests PASS. `IntegratesAQuadraticExactlyWithDgTwo` is the one that
matters for Task 6.3 — if it is off by more than 1e-12, the polynomial fit or
its integral is wrong and space-time conservation will fail later for reasons
that look like a remap bug.

- [ ] **Step 7: Commit**

```bash
git add include/soma/num/waveform.hpp include/soma/num/temporal_basis.hpp \
        src/num/temporal_basis.cpp test/unit/waveform_test.cpp \
        test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): Waveform with per-basis interpolation and exact integration"
```

---

### Task 6.2: Temporal projection weights

Spec §7.3: `M = T ⊗ W`. This task builds `T` — the matrix projecting a source
time partition onto a target time partition over one window.

**Files:**
- Create: `include/soma/remap/spacetime.hpp`
- Create: `src/remap/spacetime.cpp`
- Modify: `src/CMakeLists.txt`
- Test: `test/unit/spacetime_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `Slab`, `SlabDecomposition` (3.2); `Waveform`, basis helpers (6.1).
- Produces:
  - `struct TemporalProjection { std::vector<std::int32_t> row; std::vector<std::int32_t> col; std::vector<double> val; std::size_t n_src_steps = 0; std::size_t n_tgt_steps = 0; TemporalBasis basis = TemporalBasis::None; int degree = 0; }` — the same COO shape as `RemapWeights`, deliberately: the tensor product is then two sparse applications rather than a special case.
  - `Result<TemporalProjection> build_temporal_projection(const SlabDecomposition&, TemporalBasis, int degree, Rational src_interval, Rational dst_interval)`
  - `void apply_spacetime(const RemapWeights& spatial, const TemporalProjection& temporal, const double* src, double* dst, std::size_t n_src_space, int components, std::size_t src_step, std::size_t tgt_step)` — applies `T` in time and `W` in space.
  - Conservation property, tested: for each target step `i`,
    `Σⱼ T_ij · Δt_j = Δt_i`, so a signal constant in time projects exactly and
    the time integral is preserved. This is the temporal analogue of Phase 5's
    row-sum-unity property.

- [ ] **Step 1: Write the failing test**

`test/unit/spacetime_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/remap/spacetime.hpp>
#include <soma/schedule/slab.hpp>

#include <vector>

using soma::build_temporal_projection;
using soma::Rational;
using soma::TemporalBasis;

namespace {

soma::TemporalProjection projection(Rational src_dt, Rational dst_dt,
                                    TemporalBasis basis, int degree) {
  const auto decomp = soma::decompose(src_dt, dst_dt, Rational{1});
  EXPECT_TRUE(decomp.ok()) << decomp.error().message;
  auto p = build_temporal_projection(decomp.value(), basis, degree, src_dt,
                                     dst_dt);
  EXPECT_TRUE(p.ok()) << p.error().message;
  return p.value();
}

}  // namespace

TEST(Spacetime, LockstepProjectionIsTheIdentity) {
  const auto t = projection(Rational{1}, Rational{1}, TemporalBasis::Constant, 0);
  EXPECT_EQ(t.n_src_steps, 1u);
  EXPECT_EQ(t.n_tgt_steps, 1u);
  ASSERT_EQ(t.val.size(), 1u);
  EXPECT_NEAR(t.val[0], 1.0, 1e-14);
}

TEST(Spacetime, SubcyclingAveragesTheSourceSteps) {
  // Source ticks 4x per target tick; a constant-in-time projection weights
  // each source step by its share of the target interval.
  const auto t = projection(Rational{1, 4}, Rational{1}, TemporalBasis::Constant,
                            0);
  EXPECT_EQ(t.n_src_steps, 4u);
  EXPECT_EQ(t.n_tgt_steps, 1u);
  ASSERT_EQ(t.val.size(), 4u);
  for (const double v : t.val) EXPECT_NEAR(v, 0.25, 1e-14);
}

// The temporal analogue of Phase 5's row-sum-unity property.
TEST(Spacetime, RowSumsAreUnity) {
  for (const auto [a, b] : {std::pair{Rational{1, 3}, Rational{1, 5}},
                            std::pair{Rational{1, 2}, Rational{1, 3}},
                            std::pair{Rational{1, 7}, Rational{1, 4}}}) {
    const auto t = projection(a, b, TemporalBasis::Constant, 0);
    std::vector<double> row_sum(t.n_tgt_steps, 0.0);
    for (std::size_t k = 0; k < t.val.size(); ++k) {
      row_sum[static_cast<std::size_t>(t.row[k])] += t.val[k];
    }
    for (const double rs : row_sum) {
      EXPECT_NEAR(rs, 1.0, 1e-13)
          << "src=" << a.to_string() << " dst=" << b.to_string();
    }
  }
}

TEST(Spacetime, NonNestingIntervalsProjectWithoutGaps) {
  // 1/3 and 1/5 share no common tick except 0 and 1.
  const auto t = projection(Rational{1, 3}, Rational{1, 5},
                            TemporalBasis::Constant, 0);
  EXPECT_EQ(t.n_src_steps, 3u);
  EXPECT_EQ(t.n_tgt_steps, 5u);
  EXPECT_FALSE(t.val.empty());

  std::vector<double> row_sum(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.val.size(); ++k) {
    row_sum[static_cast<std::size_t>(t.row[k])] += t.val[k];
  }
  for (const double rs : row_sum) EXPECT_NEAR(rs, 1.0, 1e-13);
}

TEST(Spacetime, WeightsAreNonNegativeForConstantBasis) {
  const auto t = projection(Rational{1, 3}, Rational{1, 5},
                            TemporalBasis::Constant, 0);
  for (const double v : t.val) EXPECT_GE(v, -1e-15);
}

TEST(Spacetime, LinearBasisReproducesALinearSignal) {
  const auto t = projection(Rational{1, 4}, Rational{1, 2},
                            TemporalBasis::Linear, 1);
  // Source samples of f(t) = t at the midpoint of each source step.
  std::vector<double> src(t.n_src_steps);
  for (std::size_t i = 0; i < src.size(); ++i) {
    src[i] = (static_cast<double>(i) + 0.5) / static_cast<double>(t.n_src_steps);
  }
  std::vector<double> dst(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.val.size(); ++k) {
    dst[static_cast<std::size_t>(t.row[k])] +=
        t.val[k] * src[static_cast<std::size_t>(t.col[k])];
  }
  for (std::size_t i = 0; i < dst.size(); ++i) {
    const double expected =
        (static_cast<double>(i) + 0.5) / static_cast<double>(t.n_tgt_steps);
    EXPECT_NEAR(dst[i], expected, 1e-12);
  }
}

TEST(Spacetime, PreservesTheTimeIntegralOfAConstant) {
  const auto t = projection(Rational{1, 3}, Rational{1, 5},
                            TemporalBasis::Constant, 0);
  const std::vector<double> src(t.n_src_steps, 4.0);
  std::vector<double> dst(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.val.size(); ++k) {
    dst[static_cast<std::size_t>(t.row[k])] +=
        t.val[k] * src[static_cast<std::size_t>(t.col[k])];
  }
  for (const double v : dst) EXPECT_NEAR(v, 4.0, 1e-13);
}

TEST(Spacetime, TripletsAreDeterministic) {
  const auto a = projection(Rational{1, 3}, Rational{1, 5},
                            TemporalBasis::Constant, 0);
  const auto b = projection(Rational{1, 3}, Rational{1, 5},
                            TemporalBasis::Constant, 0);
  EXPECT_EQ(a.row, b.row);
  EXPECT_EQ(a.col, b.col);
  EXPECT_EQ(a.val, b.val);
}

TEST(Spacetime, NoneBasisIsRejected) {
  const auto decomp = soma::decompose(Rational{1}, Rational{1}, Rational{1});
  ASSERT_TRUE(decomp.ok());
  const auto p = build_temporal_projection(decomp.value(), TemporalBasis::None,
                                           0, Rational{1}, Rational{1});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-BASIS");
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/remap/spacetime.hpp` file not found.

- [ ] **Step 3: Write the header**

`include/soma/remap/spacetime.hpp`:

```cpp
#pragma once

#include <cstdint>
#include <vector>

#include "soma/base/result.hpp"
#include "soma/remap/remap_plan.hpp"
#include "soma/schedule/slab.hpp"

namespace soma {

/// Spec §7.3: the T of M = T (x) W. Same COO shape as RemapWeights so the
/// tensor product is two sparse applications rather than a special case.
struct TemporalProjection {
  std::vector<std::int32_t> row;   // target step
  std::vector<std::int32_t> col;   // source step
  std::vector<double>       val;
  std::size_t               n_src_steps = 0;
  std::size_t               n_tgt_steps = 0;
  TemporalBasis             basis = TemporalBasis::None;
  int                       degree = 0;
};

/// Project the source time partition onto the target's over one window.
/// Row sums are unity, so a signal constant in time projects exactly and the
/// time integral is preserved (spec §7.4).
[[nodiscard]] Result<TemporalProjection> build_temporal_projection(
    const SlabDecomposition& slabs, TemporalBasis basis, int degree,
    Rational src_interval, Rational dst_interval);

/// Tier 1 tensor-product application: T in time, W in space.
/// Tier 2 (true 4D intersection for moving meshes, spec §7.2) would replace
/// this with a non-factorising operator; the seam is here.
void apply_spacetime(const RemapWeights& spatial,
                     const TemporalProjection& temporal, const double* src,
                     double* dst, std::size_t n_src_space, int components,
                     std::size_t src_step, std::size_t tgt_step);

}  // namespace soma
```

- [ ] **Step 4: Write `src/remap/spacetime.cpp`**

Implementation contract:

1. Reject `TemporalBasis::None` with `E-ST-BASIS` — a space-time projection
   needs a basis.
2. `n_src_steps = period / src_interval`, `n_tgt_steps = period / dst_interval`,
   both exact integers because Phase 3's `decompose` already guaranteed it.
3. For `Constant`: for each slab, `T[dst_step][src_step] += slab.length() /
   dst_interval`. Summing over the slabs covering one target step gives exactly
   1, which is the row-sum property.
4. For `Linear` and `DG(p)`: for each target step, integrate the basis
   functions of the source representation over the target interval using
   `gauss_rule` with enough points to be exact for degree `p`, then normalise by
   the target interval length. The result still has unit row sums because the
   source basis is a partition of unity.
5. Sort triplets by `(row, col)` and merge duplicates — the same determinism
   requirement as Phase 5.
6. `apply_spacetime` composes: gather the temporal contribution for
   `(src_step, tgt_step)`, scale, and apply the spatial matvec.

- [ ] **Step 5: Update `src/CMakeLists.txt`** — add `remap/spacetime.cpp` (unconditionally: it depends only on `slab.cpp` and the basis helpers, not on MOAB).

- [ ] **Step 6: Register and run**

Add `soma_add_test(spacetime_test unit/spacetime_test.cpp)`.

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure -R Spacetime
```

Expected: 9 tests PASS. `RowSumsAreUnity` across three interval pairs is the
one that guarantees conservation later.

- [ ] **Step 7: Commit**

```bash
git add include/soma/remap/spacetime.hpp src/remap/spacetime.cpp \
        test/unit/spacetime_test.cpp test/CMakeLists.txt src/CMakeLists.txt
git commit -m "feat(soma): temporal projection weights for space-time remap"
```

---

### Task 6.3: Space-time remap in the lowering pass and runner

Wires Tasks 6.1 and 6.2 into the pipeline so `.spacetime(...)` on a channel
actually does something.

**Files:**
- Modify: `src/ir/rules_channel.cpp`
- Modify: `src/runtime/runner.cpp`
- Modify: `include/soma/runtime/runner.hpp`
- Test: `test/unit/lowering_spacetime_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `TemporalProjection` (6.2), `Waveform` (6.1), the channel rule (2.5).
- Produces:
  - Lowering: when `ch.spacetime.basis != None`, the rule emits
    `Sample → Interpolate → [Transport] → Remap → …` instead of
    `Sample → Accumulate → …`. The `Interpolate` node carries `attrs.basis` and
    `attrs.degree`, and its output arc keeps `ValueKind::Instant` while its
    input is `ValueKind::Waveform`.
  - A channel may not set both `accumulate` (non-`Instant`) and `spacetime` —
    they are two different temporal treatments of the same data. Validation
    gains `E-CH-TEMPORAL` for that combination (added to Phase 1's
    `validate()`).
  - Runner: `Sample` pushes into a `Waveform` keyed by arc; `Interpolate`
    evaluates it at the target time; `Extrapolate` uses the waveform's history
    rather than the Phase 4 order-0 stub.
  - `RunStats` gains `std::int64_t waveform_samples = 0;` and
    `std::int64_t temporal_projections = 0;`.

- [ ] **Step 1: Write the failing test**

`test/unit/lowering_spacetime_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/graph/coupling_graph.hpp>
#include <soma/graph/validate.hpp>
#include <soma/ir/lowering.hpp>

#include <algorithm>
#include <vector>

using soma::CouplingGraph;
using soma::OpKind;
using soma::Rational;
using soma::TemporalBasis;
namespace u = soma::units;

namespace {

CouplingGraph spacetime_pair(TemporalBasis basis, int degree,
                             bool also_accumulate = false) {
  CouplingGraph g;
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1"});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2"});
  const auto r1 = g.add_region(soma::Region{.name = "r1", .mesh = m1});
  const auto r2 = g.add_region(soma::Region{.name = "r2", .mesh = m2});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "a", .ranks = {.first = 0, .count = 1},
      .interval = Rational{1, 3}});
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "b", .ranks = {.first = 1, .count = 1},
      .interval = Rational{1, 5}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("st", s, t);
  auto builder = g.channel(ch);
  builder.spacetime(basis, degree, soma::Window::from(a));
  if (also_accumulate) builder.accumulate(soma::AccumOp::Mean);
  return g;
}

std::vector<OpKind> kinds(const soma::OperatorDAG& d) {
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

TEST(SpacetimeLowering, EmitsSampleThenInterpolate) {
  const auto r = lower(spacetime_pair(TemporalBasis::DG, 1));
  ASSERT_TRUE(r.ok()) << r.diags.to_string();
  const auto k = kinds(r.dag);
  ASSERT_TRUE(contains(k, OpKind::Sample));
  ASSERT_TRUE(contains(k, OpKind::Interpolate));
  EXPECT_LT(index_of(k, OpKind::Sample), index_of(k, OpKind::Interpolate));
}

TEST(SpacetimeLowering, DoesNotEmitAccumulate) {
  const auto r = lower(spacetime_pair(TemporalBasis::DG, 1));
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(contains(kinds(r.dag), OpKind::Accumulate))
      << "spacetime replaces accumulation, it does not add to it";
}

TEST(SpacetimeLowering, InterpolateCarriesBasisAndDegree) {
  const auto r = lower(spacetime_pair(TemporalBasis::DG, 2));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Interpolate) continue;
    EXPECT_EQ(n.attrs.basis, TemporalBasis::DG);
    EXPECT_EQ(n.attrs.degree, 2);
    return;
  }
  FAIL() << "no Interpolate node emitted";
}

TEST(SpacetimeLowering, SampleProducesAWaveformArc) {
  const auto r = lower(spacetime_pair(TemporalBasis::Linear, 1));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Sample) continue;
    EXPECT_EQ(r.dag.arc(n.output).type.kind, soma::ValueKind::Waveform);
    return;
  }
  FAIL() << "no Sample node emitted";
}

TEST(SpacetimeLowering, InterpolateReturnsToAnInstantArc) {
  const auto r = lower(spacetime_pair(TemporalBasis::Linear, 1));
  ASSERT_TRUE(r.ok());
  for (std::uint32_t i = 0; i < r.dag.node_count(); ++i) {
    const auto& n = r.dag.node(soma::NodeId{i});
    if (n.kind != OpKind::Interpolate) continue;
    EXPECT_EQ(r.dag.arc(n.inputs.front()).type.kind, soma::ValueKind::Waveform);
    EXPECT_EQ(r.dag.arc(n.output).type.kind, soma::ValueKind::Instant);
    return;
  }
  FAIL() << "no Interpolate node emitted";
}

// Two different temporal treatments of the same data is a configuration error.
TEST(SpacetimeLowering, AccumulateAndSpacetimeTogetherAreRejected) {
  const auto d = validate(spacetime_pair(TemporalBasis::DG, 1,
                                         /*also_accumulate=*/true));
  bool found = false;
  for (const auto& item : d.items()) {
    if (item.code == "E-CH-TEMPORAL") found = true;
  }
  EXPECT_TRUE(found) << d.to_string();
}

TEST(SpacetimeLowering, PlanSchedulesIndependentIntervals) {
  const CouplingGraph g = spacetime_pair(TemporalBasis::DG, 1);
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok());
  const soma::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks,
                        [](soma::RegionId) -> std::size_t { return 8; });
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  EXPECT_FALSE(sched.plan.lockstep());
  EXPECT_EQ(sched.plan.period(), Rational{1});
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — no `Interpolate` node is emitted; the channel rule ignores
`spacetime`.

- [ ] **Step 3: Add the validation rule**

In `include/soma/graph/validate.hpp`, inside the per-channel loop:

```cpp
    if (ch.spacetime.basis != TemporalBasis::None &&
        ch.accumulate.op != AccumOp::Instant) {
      d.add(Severity::Error, "E-CH-TEMPORAL",
            "channel sets both accumulate and spacetime; these are two "
            "different temporal treatments of the same data -- choose one",
            ctx);
    }
```

- [ ] **Step 4: Extend the channel lowering rule**

In `src/ir/rules_channel.cpp`, replace the accumulate block with a branch:

```cpp
  // 2. Temporal treatment: spacetime projection or accumulation, never both
  //    (validate() enforces E-CH-TEMPORAL).
  if (ch.spacetime.basis != TemporalBasis::None) {
    ValueType wave = type;
    wave.kind = ValueKind::Waveform;
    cur = emit(ctx, OpKind::Sample, cur, wave,
               OpAttrs{.basis = ch.spacetime.basis,
                       .degree = ch.spacetime.degree},
               cid, base + "/sample");

    ValueType inst = type;  // back to Instant at the target's time
    cur = emit(ctx, OpKind::Interpolate, cur, inst,
               OpAttrs{.basis = ch.spacetime.basis,
                       .degree = ch.spacetime.degree},
               cid, base + "/interpolate");
    type = inst;
  } else if (ch.accumulate.op != AccumOp::Instant) {
    // ... the existing Sample + Accumulate block, unchanged
  }
```

- [ ] **Step 5: Implement real `Sample`/`Interpolate`/`Extrapolate` in the runner**

In `src/runtime/runner.cpp`:

1. Add `std::unordered_map<std::uint32_t, Waveform> waveforms;` to `Impl`,
   created in `initialize` for every arc whose `ValueType::kind == Waveform`.
   Capacity is `period / source_interval + 1`, known from the plan.
2. `OpKind::Sample` when the output arc is a waveform: `push(ins.time, input)`
   and increment `stats.waveform_samples`. The existing accumulator path stays
   for the non-waveform case.
3. `OpKind::Interpolate`: `waveform.evaluate(ins.time, out.data)`, incrementing
   `stats.temporal_projections`. An `E-WF-OUTSIDE` here is a real scheduling
   bug, so surface it rather than clamping.
4. `OpKind::Extrapolate`: if the input arc is a waveform, gather the last
   `order + 1` samples and call `num::extrapolate`; otherwise keep the Phase 4
   copy.
5. Reset each waveform at the start of every period in `run_period`.

- [ ] **Step 6: Run**

Add `soma_add_test(lowering_spacetime_test unit/lowering_spacetime_test.cpp)`.

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R "SpacetimeLowering|Waveform|Spacetime"
```

Expected: 7 + 16 + 9 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ir/rules_channel.cpp src/runtime/runner.cpp \
        include/soma/runtime/runner.hpp include/soma/graph/validate.hpp \
        test/unit/lowering_spacetime_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): space-time channels lower to Sample plus Interpolate"
```

---

### Task 6.4: Space-time conservation

Spec §7.4: the invariant becomes ∬ q dA dt over the window, matched between
sides — strictly stronger than a per-exchange spatial sum.

**Files:**
- Modify: `include/soma/diag/conservation.hpp`
- Modify: `src/diag/conservation.cpp`
- Modify: `src/runtime/runner.cpp`
- Test: `test/verification/spacetime_conservation_test.cpp`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: `ConservationReport` (5.5), `Waveform` (6.1).
- Produces:
  - `struct SpacetimeConservationReport { double source_integral = 0.0; double target_integral = 0.0; double absolute_error = 0.0; double relative_error = 0.0; Rational window_begin, window_end; bool within_tolerance = true; }`
  - `SpacetimeConservationReport check_spacetime_conservation(const MeshView& src_view, int src_dim, const Waveform& src_wave, const MeshView& tgt_view, int tgt_dim, const Waveform& tgt_wave, double tolerance)` — integrates each waveform in time (per element) and in space (with `measure` weights), then compares.
  - `RunnerConfig` gains `bool check_spacetime_conservation = true;` and `RunStats` gains `double worst_spacetime_error = 0.0;`.
  - The runner accumulates a target-side waveform for every channel with a
    space-time basis, so the check has both sides to compare at window end.

- [ ] **Step 1: Write the failing test**

`test/verification/spacetime_conservation_test.cpp`:

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
using soma::TemporalBasis;
namespace u = soma::units;

namespace {

/// Writes a smooth space-time signal q(x, t) = (1 + sin(2x)) * (1 + t).
class SpaceTimeSource final : public soma::Component {
 public:
  SpaceTimeSource(std::size_t n, const soma::MeshView* view)
      : data_(n, 0.0), view_(view) {}

  [[nodiscard]] std::string name() const override { return "src"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("src");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext& ctx) override {
    const double t = ctx.t_end.to_double();
    for (std::size_t i = 0; i < data_.size(); ++i) {
      const auto c = view_->centroid(3, i);
      data_[i] = (1.0 + std::sin(2.0 * c.x)) * (1.0 + t);
    }
    return soma::Result<void>::success();
  }

 private:
  std::vector<double>   data_;
  const soma::MeshView* view_;
};

class Receiver final : public soma::Component {
 public:
  explicit Receiver(std::size_t n) : data_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "dst"; }

  soma::Result<void> register_fields(FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("dst");
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

  [[nodiscard]] const std::vector<double>& data() const { return data_; }

 private:
  std::vector<double> data_;
};

struct Fixture {
  CouplingGraph         graph;
  soma::MoabMeshBackend backend;
  soma::OperatorDAG     dag;
  soma::ExecutionPlan   plan;
  ComponentSet          components;
  FieldRegistry         registry;
};

/// Non-matching meshes AND unrelated intervals: the Phase 6 milestone case.
std::unique_ptr<Fixture> build(Rational src_dt, Rational dst_dt,
                               TemporalBasis basis, int degree) {
  auto f = std::make_unique<Fixture>();
  auto& g = f->graph;
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "coarse",
                                            .source = "generated:box:4x4x4"});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "fine",
                                            .source = "generated:box:5x5x5"});
  const auto r1 = g.add_region(soma::Region{.name = "rs", .mesh = m1,
                                            .dimension = 3});
  const auto r2 = g.add_region(soma::Region{.name = "rt", .mesh = m2,
                                            .dimension = 3});
  const auto ca = g.add_component(soma::ComponentDecl{
      .name = "src", .ranks = {.first = 0, .count = 1}, .interval = src_dt});
  const auto cb = g.add_component(soma::ComponentDecl{
      .name = "dst", .ranks = {.first = 1, .count = 1}, .interval = dst_dt});
  const auto fs = g.add_field(ca, soma::FieldSpec{
      .name = "q", .region = r1, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive,
      .intent = soma::Intent::Provides});
  const auto ft = g.add_field(cb, soma::FieldSpec{
      .name = "q_in", .region = r2, .units = u::kHeatFluxDensity,
      .conservation = Conservation::Extensive,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("flux", fs, ft);
  g.channel(ch)
      .spacetime(basis, degree, soma::Window::from(ca))
      .conservative();

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

  const auto* sv = f->backend.view(m1).value();
  f->components.add(std::make_unique<SpaceTimeSource>(64, sv));
  f->components.add(std::make_unique<Receiver>(125));
  return f;
}

}  // namespace

// The Phase 6 milestone: unrelated intervals 1/3 and 1/5, non-matching meshes,
// space-time flux integral conserved.
TEST(SpacetimeConservation, HoldsForUnrelatedIntervals) {
  auto f = build(Rational{1, 3}, Rational{1, 5}, TemporalBasis::Linear, 1);
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  RunnerConfig cfg;
  cfg.check_spacetime_conservation = true;
  ASSERT_TRUE(runner.initialize(f->components, f->registry, cfg).ok());
  ASSERT_TRUE(runner.run(f->components, 3).ok());

  EXPECT_LT(runner.stats().worst_spacetime_error, 1e-10)
      << "space-time flux integral must be conserved across the window";
}

TEST(SpacetimeConservation, ConstantInSpaceAndTimeIsExact) {
  auto f = build(Rational{1, 3}, Rational{1, 5}, TemporalBasis::Constant, 0);
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
  ASSERT_TRUE(runner.run(f->components, 2).ok());
  EXPECT_LT(runner.stats().worst_spacetime_error, 1e-12);
}

TEST(SpacetimeConservation, WorksForSeveralIntervalPairs) {
  for (const auto [a, b] : {std::pair{Rational{1, 2}, Rational{1, 3}},
                            std::pair{Rational{1, 7}, Rational{1, 4}},
                            std::pair{Rational{1, 5}, Rational{1, 5}}}) {
    auto f = build(a, b, TemporalBasis::Linear, 1);
    Runner runner{f->graph, f->dag, f->plan, f->backend};
    for (auto& c : f->components) {
      ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
    }
    ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
    ASSERT_TRUE(runner.run(f->components, 2).ok());
    EXPECT_LT(runner.stats().worst_spacetime_error, 1e-10)
        << "src=" << a.to_string() << " dst=" << b.to_string();
  }
}

TEST(SpacetimeConservation, SamplesAndProjectionsAreCounted) {
  auto f = build(Rational{1, 3}, Rational{1, 5}, TemporalBasis::Linear, 1);
  Runner runner{f->graph, f->dag, f->plan, f->backend};
  for (auto& c : f->components) {
    ASSERT_TRUE(c->register_fields(f->registry, f->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(f->components, f->registry).ok());
  ASSERT_TRUE(runner.run_period(f->components).ok());
  EXPECT_EQ(runner.stats().waveform_samples, 3);        // src ticks 3x
  EXPECT_EQ(runner.stats().temporal_projections, 5);    // dst ticks 5x
}

#else
TEST(SpacetimeConservation, SkippedWithoutMoab) {
  GTEST_SKIP() << "built without SOMA_ENABLE_MOAB";
}
#endif
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `check_spacetime_conservation` is not a member of
`RunnerConfig`.

- [ ] **Step 3: Implement**

Extend `include/soma/diag/conservation.hpp` with the report struct and the
function; implement in `src/diag/conservation.cpp` by, per element, calling
`Waveform::integrate` over the window and then a spatial `weighted_sum` with
`measure`. Add the target-side waveform accumulation and the window-end check to
`src/runtime/runner.cpp`.

- [ ] **Step 4: Run**

```bash
cmake --build build-full -j
ctest --test-dir build-full --output-on-failure -R SpacetimeConservation
```

Expected: 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add include/soma/diag/conservation.hpp src/diag/conservation.cpp \
        src/runtime/runner.cpp include/soma/runtime/runner.hpp \
        test/verification/spacetime_conservation_test.cpp test/CMakeLists.txt
git commit -m "feat(soma): space-time conservation check over the coupling window"
```

---

### Task 6.5: Temporal order-of-accuracy study

Spec §10 level 4: the test that catches order degradation and distinguishes a
coupler that *claims* second order from one that *has* it.

**Files:**
- Create: `test/verification/temporal_order_test.cpp`
- Create: `include/soma/diag/order_study.hpp`
- Create: `src/diag/order_study.cpp`
- Modify: `src/CMakeLists.txt`
- Modify: `test/CMakeLists.txt`

**Interfaces:**
- Consumes: everything in Phase 6.
- Produces:
  - `struct OrderStudyPoint { double h = 0.0; double error = 0.0; }`
  - `struct OrderStudyResult { std::vector<OrderStudyPoint> points; std::vector<double> pairwise_orders; double fitted_order = 0.0; double r_squared = 0.0; }`
  - `OrderStudyResult fit_order(const std::vector<OrderStudyPoint>&)` — least-squares fit of `log(error)` against `log(h)`; `pairwise_orders[i] = log(e_i/e_{i+1}) / log(h_i/h_{i+1})`.
  - A test asserting the measured order for each temporal basis:

| Basis | Expected order | Asserted range |
|---|---|---|
| `Constant` | 1 | `[0.85, 1.30]` |
| `Linear` | 2 | `[1.80, 2.30]` |
| `DG(2)` | 3 | `[2.70, 3.40]` |

  The ranges are deliberately loose: a measured order is a fit over finite
  refinements, and a tight window makes the test flaky without making it more
  informative. What it must catch is *degradation* — a scheme claiming 2 that
  measures 1.

- [ ] **Step 1: Write the failing test**

`test/verification/temporal_order_test.cpp`:

```cpp
#include <gtest/gtest.h>
#include <soma/diag/order_study.hpp>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/runtime/runner.hpp>

#include <cmath>
#include <vector>

using soma::CouplingGraph;
using soma::fit_order;
using soma::OrderStudyPoint;
using soma::Rational;
using soma::TemporalBasis;
namespace u = soma::units;

namespace {

/// Source of a known smooth signal; the receiver's error against the exact
/// value at its own tick times is what the study measures.
class Signal final : public soma::Component {
 public:
  explicit Signal(std::size_t n) : data_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "signal"; }

  soma::Result<void> register_fields(soma::FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("signal");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext& ctx) override {
    const double t = ctx.t_end.to_double();
    for (double& d : data_) d = std::exp(std::sin(3.0 * t));
    return soma::Result<void>::success();
  }

 private:
  std::vector<double> data_;
};

class Sampler final : public soma::Component {
 public:
  explicit Sampler(std::size_t n) : data_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "sampler"; }

  soma::Result<void> register_fields(soma::FieldRegistry& reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("sampler");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, soma::FieldRef{data_.data(),
                                          soma::Layout{.stride = 1,
                                                       .count = data_.size()},
                                          soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const soma::StepContext& ctx) override {
    const double t = ctx.t_end.to_double();
    const double exact = std::exp(std::sin(3.0 * t));
    const double err = std::abs(data_[0] - exact);
    if (err > max_error) max_error = err;
    return soma::Result<void>::success();
  }

  double max_error = 0.0;

 private:
  std::vector<double> data_;
};

/// One refinement level: run with source interval 1/n and measure the error.
double run_at_resolution(int n, TemporalBasis basis, int degree) {
  CouplingGraph g;
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1",
                                            .source = "generated:line:4"});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2",
                                            .source = "generated:line:4"});
  const auto r1 = g.add_region(soma::Region{.name = "r1", .mesh = m1,
                                            .dimension = 1});
  const auto r2 = g.add_region(soma::Region{.name = "r2", .mesh = m2,
                                            .dimension = 1});
  const auto a = g.add_component(soma::ComponentDecl{
      .name = "signal", .ranks = {.first = 0, .count = 1},
      .interval = Rational{1, n}});
  // Target ticks at an unrelated rate so interpolation genuinely happens.
  const auto b = g.add_component(soma::ComponentDecl{
      .name = "sampler", .ranks = {.first = 1, .count = 1},
      .interval = Rational{1, 2 * n + 1}});
  const auto s = g.add_field(a, soma::FieldSpec{
      .name = "f", .region = r1, .units = u::kKelvin,
      .intent = soma::Intent::Provides});
  const auto t = g.add_field(b, soma::FieldSpec{
      .name = "f_in", .region = r2, .units = u::kKelvin,
      .intent = soma::Intent::Requires});
  const auto ch = g.add_channel("link", s, t);
  g.channel(ch).spacetime(basis, degree, soma::Window::from(a));

  soma::SerialMeshBackend backend;
  EXPECT_TRUE(backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  const soma::ClockSet clocks{g};
  auto sched = schedule(g, lowered.dag, clocks,
                        soma::make_region_sizes(g, backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();

  soma::ComponentSet components;
  components.add(std::make_unique<Signal>(4));
  auto sampler = std::make_unique<Sampler>(4);
  Sampler* sampler_ptr = sampler.get();
  components.add(std::move(sampler));

  soma::FieldRegistry registry;
  for (auto& c : components) {
    EXPECT_TRUE(c->register_fields(registry, g).ok());
  }
  soma::Runner runner{g, lowered.dag, sched.plan, backend};
  EXPECT_TRUE(runner.initialize(components, registry).ok());
  EXPECT_TRUE(runner.run(components, 2).ok());
  return sampler_ptr->max_error;
}

soma::OrderStudyResult study(TemporalBasis basis, int degree) {
  std::vector<OrderStudyPoint> points;
  for (const int n : {4, 8, 16, 32}) {
    points.push_back(OrderStudyPoint{.h = 1.0 / static_cast<double>(n),
                                     .error = run_at_resolution(n, basis, degree)});
  }
  return fit_order(points);
}

}  // namespace

TEST(OrderStudy, FitRecoversAKnownSlope) {
  // Synthetic data with error = 3 * h^2.
  std::vector<OrderStudyPoint> pts;
  for (const double h : {0.1, 0.05, 0.025, 0.0125}) {
    pts.push_back(OrderStudyPoint{.h = h, .error = 3.0 * h * h});
  }
  const auto r = fit_order(pts);
  EXPECT_NEAR(r.fitted_order, 2.0, 1e-10);
  EXPECT_NEAR(r.r_squared, 1.0, 1e-10);
  for (const double p : r.pairwise_orders) EXPECT_NEAR(p, 2.0, 1e-10);
}

TEST(TemporalOrder, ConstantBasisIsFirstOrder) {
  const auto r = study(TemporalBasis::Constant, 0);
  EXPECT_GT(r.fitted_order, 0.85) << "measured " << r.fitted_order;
  EXPECT_LT(r.fitted_order, 1.30) << "measured " << r.fitted_order;
}

TEST(TemporalOrder, LinearBasisIsSecondOrder) {
  const auto r = study(TemporalBasis::Linear, 1);
  EXPECT_GT(r.fitted_order, 1.80)
      << "measured " << r.fitted_order
      << " -- order degradation is the defect this test exists to catch";
  EXPECT_LT(r.fitted_order, 2.30) << "measured " << r.fitted_order;
}

TEST(TemporalOrder, DgTwoIsThirdOrder) {
  const auto r = study(TemporalBasis::DG, 2);
  EXPECT_GT(r.fitted_order, 2.70) << "measured " << r.fitted_order;
  EXPECT_LT(r.fitted_order, 3.40) << "measured " << r.fitted_order;
}

TEST(TemporalOrder, HigherBasisIsMoreAccurateAtTheSameResolution) {
  const double e_const = run_at_resolution(16, TemporalBasis::Constant, 0);
  const double e_linear = run_at_resolution(16, TemporalBasis::Linear, 1);
  EXPECT_LT(e_linear, e_const);
}

TEST(TemporalOrder, ErrorsDecreaseMonotonically) {
  const auto r = study(TemporalBasis::Linear, 1);
  for (std::size_t i = 1; i < r.points.size(); ++i) {
    EXPECT_LT(r.points[i].error, r.points[i - 1].error)
        << "refinement must reduce the error at level " << i;
  }
}
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `soma/diag/order_study.hpp` file not found.

- [ ] **Step 3: Write the order-study helper**

`include/soma/diag/order_study.hpp`:

```cpp
#pragma once

#include <vector>

namespace soma {

struct OrderStudyPoint {
  double h = 0.0;      // refinement parameter (dt or dx)
  double error = 0.0;
};

struct OrderStudyResult {
  std::vector<OrderStudyPoint> points;
  std::vector<double>          pairwise_orders;
  double                       fitted_order = 0.0;
  double                       r_squared = 0.0;
};

/// Least-squares fit of log(error) against log(h). Spec §10 level 4: the
/// observed order is fit and asserted, not assumed.
[[nodiscard]] OrderStudyResult fit_order(const std::vector<OrderStudyPoint>& pts);

}  // namespace soma
```

`src/diag/order_study.cpp` implements the fit with plain accumulation — no
external dependency.

- [ ] **Step 4: Run**

```bash
cmake --build build -j
ctest --test-dir build --output-on-failure -R "OrderStudy|TemporalOrder"
```

Expected: 6 tests PASS. **If `LinearBasisIsSecondOrder` measures ≈1**, the
waveform is being evaluated at the wrong time or the projection is falling back
to `Constant` — that is exactly the order degradation the spec calls out, and it
must be fixed rather than accommodated by widening the range.

- [ ] **Step 5: Run everything**

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build-full --output-on-failure
```

Expected: both fully green.

- [ ] **Step 6: Commit**

```bash
git add include/soma/diag/order_study.hpp src/diag/order_study.cpp \
        test/verification/temporal_order_test.cpp test/CMakeLists.txt \
        src/CMakeLists.txt
git commit -m "test(soma): temporal order-of-accuracy study per basis"
```

---

## Phase 6 exit criteria

- [ ] `Waveform` interpolates exactly for each basis's design degree and integrates a quadratic exactly with DG(2).
- [ ] Temporal projection row sums are unity for unrelated interval pairs (1/3 ↔ 1/5, 1/7 ↔ 1/4).
- [ ] A channel may not set both `accumulate` and `spacetime` — `E-CH-TEMPORAL`.
- [ ] Space-time flux conservation holds to 1e-10 across non-matching meshes **and** unrelated intervals.
- [ ] Measured temporal order matches the basis: Constant ≈1, Linear ≈2, DG(2) ≈3.
- [ ] Explicit coupling still works as the degenerate one-sample waveform — Phase 4's tests are unchanged and still pass.
- [ ] Both `ctest` runs fully green.

**Deferred to Tier 2 (spec §7.2):** true 4D pentatope intersection for meshes
that move within the window. `RemapPlan::temporal` and the `apply_spacetime`
signature are the attachment points.
