// Task 6.4 -- space-time conservation verification (spec §7.4).
//
// ===========================================================================
// WHICH IDENTITY THIS FILE CHECKS. Read this before editing anything below.
// ===========================================================================
//
// `include/soma/remap/spacetime.hpp` states the two properties of `T`:
//
//   1. CONSTANT REPRODUCTION -- a plain ROW sum:
//          sum_j T_ij == 1                for every target step i
//   2. CONSERVATION OF THE TIME INTEGRAL -- a COLUMN sum:
//          sum_i T_ij * dt_tgt == dt_src  for every source step j
//
// Property 2 is the one spec §7.4's conservation check rides on. An earlier
// brief on this project stated it as `sum_j T_ij * dt_src == dt_tgt`, a ROW
// sum. That identity is FALSE for these weights: under 4x subcycling every
// T_0j is 1/4 and dt_src is 1/4, so the left side is 1/4 while dt_tgt is 1.
// `TheRowFormOfTheConservationIdentityIsFalse` below pins the counter-example
// so the wrong form cannot be reintroduced without a named test failure.
//
// The spatial operator `W` uses the same orientation
// (`include/soma/remap/remap_plan.hpp`): row = target, col = source, ROW sums
// unity, MEASURE-WEIGHTED COLUMN sums carrying the measure.
//
// ===========================================================================
// WHAT IS AND IS NOT VERIFIED END TO END
// ===========================================================================
//
// `W` IS NOT WIRED INTO THE RUNTIME. `OpKind::Remap` in src/runtime/runner.cpp
// is still `num::copy`, an identity; every caller of `build_weights` is a
// test. So the runner sections below exercise `T` only, and say so. The
// composition `T (x) W` is verified at the LIBRARY level, through the real
// `apply_spacetime` entry point, and never claimed to have run inside the
// runner. Wiring `W` is Task 5.5's job.
//
// `build_weights` additionally needs MOAB, which is OFF in the three gating
// presets. The MOAB-backed composition checks are therefore compiled only
// under SOMA_HAVE_MOAB; the MOAB-free ones build an exact 1-D supermesh
// operator by hand and always run.

#include <gtest/gtest.h>

#include <soma/graph/coupling_graph.hpp>
#include <soma/ir/lowering.hpp>
#include <soma/mesh/serial_backend.hpp>
#include <soma/remap/remap_plan.hpp>
#include <soma/remap/spacetime.hpp>
#include <soma/runtime/runner.hpp>
#include <soma/schedule/plan.hpp>
#include <soma/schedule/slab.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <string_view>
#include <vector>

using soma::ComponentSet;
using soma::CouplingGraph;
using soma::FieldRegistry;
using soma::OpKind;
using soma::Rational;
using soma::RemapWeights;
using soma::Runner;
using soma::SerialMeshBackend;
using soma::SpatialWeightsView;
using soma::StepContext;
using soma::TemporalBasis;
using soma::TemporalProjection;
namespace u = soma::units;

// ===========================================================================
// 0. Helpers -- the two defects, spelled out once
// ===========================================================================

namespace {

/// Tolerance on both identities, everywhere in this file.
///
/// NOT a round number picked to make something pass. The worst residual
/// actually measured over the 3066-matrix sweep below is 1.264e-15, and it is
/// BIT-IDENTICAL in the Debug, Release and ASan+UBSan legs (AppleClang 21,
/// arm64), so there is no observed compiler-dependent spread to accommodate.
/// 1e-13 is ~80x that noise floor: loose enough not to be flaky, tight enough
/// that the mutation study below detects a 1e-12 relative perturbation applied
/// to every weight, and a 1e-12 absolute transfer of weight between two
/// columns of one row. Anyone raising this number should say what they
/// measured first.
constexpr double kIdentityTol = 1e-13;

/// Tolerance on the runner-level space-time integral. Worst clean value
/// measured: 0 (degree 0) and 1.95e-16 (degree 1), again identical across all
/// three legs.
constexpr double kRunnerTol = 1e-14;

struct Basis {
  TemporalBasis basis;
  int           degree;
  const char*   label;
};

constexpr Basis kBases[] = {
    {TemporalBasis::Constant, 0, "Constant"},
    {TemporalBasis::Linear, 1, "Linear"},
    {TemporalBasis::DG, 0, "DG(0)"},
    {TemporalBasis::DG, 1, "DG(1)"},
    {TemporalBasis::DG, 2, "DG(2)"},
    {TemporalBasis::DG, 3, "DG(3)"},
    {TemporalBasis::DG, 4, "DG(4)"},
};

/// A schedule expressed as "the period cut into `n_src` source steps and
/// `n_tgt` target steps". Building the intervals this way rather than picking
/// two rationals guarantees both divide the period exactly, which is what
/// `build_temporal_projection` requires, and makes nesting a plain statement
/// about the two integers.
struct Ratio {
  std::int64_t n_src;
  std::int64_t n_tgt;
  Rational     period;

  [[nodiscard]] Rational src() const { return period / Rational{n_src}; }
  [[nodiscard]] Rational dst() const { return period / Rational{n_tgt}; }
  [[nodiscard]] bool nesting() const {
    return n_src % n_tgt == 0 || n_tgt % n_src == 0;
  }
  [[nodiscard]] std::string label() const {
    return std::to_string(n_src) + "src->" + std::to_string(n_tgt) +
           "tgt over period " + period.to_string();
  }
};

soma::Result<TemporalProjection> try_build(const Ratio& r, const Basis& b) {
  const auto d = soma::decompose(r.src(), r.dst(), r.period);
  if (!d.ok()) return d.error();
  return soma::build_temporal_projection(d.value(), b.basis, b.degree, r.src(),
                                         r.dst());
}

TemporalProjection build(const Ratio& r, const Basis& b) {
  auto p = try_build(r, b);
  EXPECT_TRUE(p.ok()) << (p.ok() ? std::string{} : p.error().message) << " for "
                      << b.label << " " << r.label();
  return p.ok() ? p.value() : TemporalProjection{};
}

/// PROPERTY 2, the conservation identity, exactly as spacetime.hpp states it:
///     max_j | (sum_i T_ij) * dt_tgt - dt_src | / dt_src
/// A COLUMN sum. Relative to `dt_src` so the number is comparable across
/// schedules whose intervals differ by orders of magnitude.
[[nodiscard]] double column_defect(const TemporalProjection& t) {
  const double dt_tgt = t.dst_interval.to_double();
  const double dt_src = t.src_interval.to_double();
  std::vector<double> col(t.n_src_steps, 0.0);
  for (std::size_t k = 0; k < t.nnz(); ++k) {
    col[static_cast<std::size_t>(t.col[k])] += t.val[k];
  }
  double worst = 0.0;
  for (const double c : col) {
    worst = std::max(worst, std::fabs(c * dt_tgt - dt_src) / dt_src);
  }
  return worst;
}

/// PROPERTY 1, constant reproduction: a plain, unweighted ROW sum.
[[nodiscard]] double row_defect(const TemporalProjection& t) {
  std::vector<double> row(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.nnz(); ++k) {
    row[static_cast<std::size_t>(t.row[k])] += t.val[k];
  }
  double worst = 0.0;
  for (const double r : row) worst = std::max(worst, std::fabs(r - 1.0));
  return worst;
}

/// The FALSE identity from the earlier brief: `sum_j T_ij * dt_src == dt_tgt`,
/// a ROW sum. Computed only so a test can assert it is large.
[[nodiscard]] double row_form_of_conservation_defect(
    const TemporalProjection& t) {
  const double dt_tgt = t.dst_interval.to_double();
  const double dt_src = t.src_interval.to_double();
  std::vector<double> row(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.nnz(); ++k) {
    row[static_cast<std::size_t>(t.row[k])] += t.val[k];
  }
  double worst = 0.0;
  for (const double r : row) {
    worst = std::max(worst, std::fabs(r * dt_src - dt_tgt) / dt_tgt);
  }
  return worst;
}

/// Every schedule the column sweep runs over. Built once so the coverage
/// assertions below describe exactly what the sweep saw.
const std::vector<Ratio>& sweep_ratios() {
  static const std::vector<Ratio> v = [] {
    std::vector<Ratio> r;
    // The dense block: every (n_src, n_tgt) in [1, 12]^2, over three periods
    // whose lengths are unrelated (integer, even, and a fraction with a prime
    // denominator, so the exact rational arithmetic is genuinely exercised
    // rather than always landing on halves).
    for (const Rational period : {Rational{1}, Rational{2}, Rational{3, 7}}) {
      for (std::int64_t a = 1; a <= 12; ++a) {
        for (std::int64_t b = 1; b <= 12; ++b) {
          r.push_back(Ratio{a, b, period});
        }
      }
    }
    // Large, coprime, and deliberately at the edge of what a naive
    // `t/dt - j` in double would survive.
    r.push_back(Ratio{97, 89, Rational{1}});
    r.push_back(Ratio{89, 97, Rational{1}});
    r.push_back(Ratio{1000, 999, Rational{1}});
    r.push_back(Ratio{999, 1000, Rational{1}});
    r.push_back(Ratio{1024, 1, Rational{1}});
    r.push_back(Ratio{1, 1024, Rational{1}});
    return r;
  }();
  return v;
}

}  // namespace

// ===========================================================================
// 1. The column identity, swept
// ===========================================================================

/// A sweep that only covers nesting ratios is nearly worthless: when the two
/// partitions nest, every slab is a whole source step, the degree-p
/// reconstruction integrates to the same cell mean whatever p is, and
/// `Constant` and `DG(1)` build the SAME matrix. Whole classes of error are
/// invisible there. This test asserts the sweep's shape so a later narrowing
/// is a named failure rather than a silent loss of coverage.
TEST(SpacetimeColumnSweep, TheSweepCoversBothNestingClassesAndBothDirections) {
  std::size_t nesting = 0, non_nesting = 0, refining = 0, coarsening = 0,
              lockstep = 0;
  for (const Ratio& r : sweep_ratios()) {
    (r.nesting() ? nesting : non_nesting)++;
    if (r.n_tgt > r.n_src) {
      ++refining;
    } else if (r.n_tgt < r.n_src) {
      ++coarsening;
    } else {
      ++lockstep;
    }
  }
  EXPECT_GT(non_nesting, 200u) << "the sweep lost its non-nesting cases";
  EXPECT_GT(nesting, 100u);
  EXPECT_GT(refining, 150u);
  EXPECT_GT(coarsening, 150u);
  EXPECT_GT(lockstep, 30u);
  std::cout << "[ COVERAGE ] " << sweep_ratios().size() << " schedules x "
            << std::size(kBases) << " bases = "
            << sweep_ratios().size() * std::size(kBases) << " matrices; "
            << nesting << " nesting, " << non_nesting << " non-nesting, "
            << refining << " refining, " << coarsening << " coarsening, "
            << lockstep << " lockstep\n";
}

/// THE CHECK spec §7.4 rides on. `sum_i T_ij * dt_tgt == dt_src`, per source
/// step, over every schedule and every basis.
TEST(SpacetimeColumnSweep, ColumnSumsCarryTheSourceStepLength) {
  double      worst = 0.0;
  std::string worst_at;
  std::size_t built = 0;
  for (const Ratio& r : sweep_ratios()) {
    for (const Basis& b : kBases) {
      const auto p = try_build(r, b);
      ASSERT_TRUE(p.ok()) << p.error().message << " for " << b.label << " "
                          << r.label();
      ++built;
      const double d = column_defect(p.value());
      if (d > worst) {
        worst    = d;
        worst_at = std::string(b.label) + " " + r.label();
      }
      ASSERT_LT(d, kIdentityTol) << b.label << " " << r.label();
    }
  }
  std::cout << "[ RESIDUAL ] column identity over " << built
            << " matrices: worst relative defect " << worst << " at "
            << worst_at << "\n";
  testing::Test::RecordProperty("worst_column_defect", std::to_string(worst));
  testing::Test::RecordProperty("worst_column_defect_at", worst_at);
}

/// Property 1 over the same sweep, so the two are measured on identical
/// inputs and the report can state both.
TEST(SpacetimeColumnSweep, RowSumsAreUnity) {
  double      worst = 0.0;
  std::string worst_at;
  for (const Ratio& r : sweep_ratios()) {
    for (const Basis& b : kBases) {
      const auto p = try_build(r, b);
      ASSERT_TRUE(p.ok()) << p.error().message;
      const double d = row_defect(p.value());
      if (d > worst) {
        worst    = d;
        worst_at = std::string(b.label) + " " + r.label();
      }
      ASSERT_LT(d, kIdentityTol) << b.label << " " << r.label();
    }
  }
  std::cout << "[ RESIDUAL ] row identity: worst |sum - 1| " << worst << " at "
            << worst_at << "\n";
  testing::Test::RecordProperty("worst_row_defect", std::to_string(worst));
}

/// The non-nesting half of the sweep on its own. Stated separately because it
/// is the half that can actually distinguish the degrees: on a nesting pair
/// `Constant` and `DG(1)` are the same matrix, so a defect that only afflicts
/// the higher-degree reconstruction would hide in the aggregate.
TEST(SpacetimeColumnSweep, NonNestingSchedulesAloneCarryTheColumnIdentity) {
  double      worst = 0.0;
  std::string worst_at;
  std::size_t n = 0;
  for (const Ratio& r : sweep_ratios()) {
    if (r.nesting()) continue;
    for (const Basis& b : kBases) {
      const auto p = try_build(r, b);
      ASSERT_TRUE(p.ok()) << p.error().message;
      ++n;
      const double d = column_defect(p.value());
      if (d > worst) {
        worst    = d;
        worst_at = std::string(b.label) + " " + r.label();
      }
      ASSERT_LT(d, kIdentityTol) << b.label << " " << r.label();
    }
  }
  std::cout << "[ RESIDUAL ] column identity, non-nesting only (" << n
            << " matrices): worst " << worst << " at " << worst_at << "\n";
}

/// On a non-nesting pair the degrees must genuinely differ, otherwise the
/// sweep above is testing one matrix seven times. This is the guard on that.
TEST(SpacetimeColumnSweep, DegreeChangesTheMatrixOnANonNestingPair) {
  const Ratio r{3, 2, Rational{1}};  // 1/3 against 1/2: two partial slabs
  ASSERT_FALSE(r.nesting());
  const auto c  = build(r, kBases[0]);  // Constant
  const auto d1 = build(r, kBases[3]);  // DG(1)
  bool       differs = false;
  for (std::size_t i = 0; i < c.n_tgt_steps; ++i) {
    for (std::size_t j = 0; j < c.n_src_steps; ++j) {
      if (std::fabs(c.coefficient(i, j) - d1.coefficient(i, j)) > 1e-9) {
        differs = true;
      }
    }
  }
  EXPECT_TRUE(differs)
      << "Constant and DG(1) built the same matrix on a non-nesting pair; the "
         "sweep can no longer tell a silent degree downgrade from the real "
         "operator";

  // ... and on a nesting pair they must NOT, which is why a nesting-only
  // sweep proves so little.
  const Ratio nest{4, 2, Rational{1}};
  ASSERT_TRUE(nest.nesting());
  const auto nc  = build(nest, kBases[0]);
  const auto nd1 = build(nest, kBases[3]);
  for (std::size_t i = 0; i < nc.n_tgt_steps; ++i) {
    for (std::size_t j = 0; j < nc.n_src_steps; ++j) {
      EXPECT_NEAR(nc.coefficient(i, j), nd1.coefficient(i, j), 1e-14);
    }
  }
}

/// The trap this task exists to avoid. `sum_j T_ij * dt_src == dt_tgt` is the
/// ROW form of the conservation statement and it is FALSE. Pinned with the
/// concrete counter-example from spacetime.hpp so that writing a conservation
/// check against the row form fails here, loudly, instead of producing a
/// spurious imbalance somewhere downstream.
TEST(SpacetimeColumnSweep, TheRowFormOfTheConservationIdentityIsFalse) {
  const Ratio r{4, 1, Rational{1}};  // 4x subcycling: dt_src 1/4, dt_tgt 1
  const auto  t = build(r, kBases[0]);
  ASSERT_EQ(t.n_src_steps, 4u);
  ASSERT_EQ(t.n_tgt_steps, 1u);
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_NEAR(t.coefficient(0, j), 0.25, 1e-14);
  }
  // TRUE: the row sums are unity, and the column sums carry dt_src.
  EXPECT_LT(row_defect(t), 1e-14);
  EXPECT_LT(column_defect(t), 1e-14);
  // FALSE: sum_j T_0j * dt_src is 4 * (1/4) * (1/4) = 1/4, not dt_tgt = 1.
  EXPECT_NEAR(row_form_of_conservation_defect(t), 0.75, 1e-14)
      << "the row form of the conservation identity is no longer false; "
         "either the weights changed meaning or this test drifted";
}

// ===========================================================================
// 2. Mutation sensitivity -- how small a perturbation the check catches
// ===========================================================================
//
// A green test is not a correct test. Every claim above is only worth what its
// sensitivity is, so the sensitivity is measured here rather than asserted.
// The tolerance used by the sweep is `kIdentityTol` on a relative defect;
// these tests establish what relative perturbation of a single weight that
// corresponds to.
//
// ---------------------------------------------------------------------------
// SOURCE-LEVEL MUTATION STUDY (Task 6.4). Not automated -- these were applied
// by hand to src/remap/spacetime.cpp, run, and reverted. Recorded so the
// sensitivity claims can be re-derived rather than taken on trust, and so a
// later change that quietly weakens them has something to be checked against.
// Debug leg, AppleClang 21, arm64.
//
//   MUT-A  every weight scaled by (1 + eps) in the canonicalisation loop.
//          eps = 1e-13  ColumnSums FAIL (1.12e-13), RowSums FAIL,
//                       runner integral FAIL (9.97e-14)
//          eps = 1e-14  all PASS -- 1e-14 is below the noise floor
//          => smallest relative rescale detected: 1e-13.
//
//   MUT-B  `delta` moved from the last to the first entry of every row, which
//          preserves the row sums EXACTLY and breaks only the column sums.
//          delta = 1e-12  ColumnSums FAIL, RowSums PASS (4.4e-16),
//                         runner integral FAIL (1.27e-13)
//          delta = 1e-13  ColumnSums FAIL, RowSums PASS, runner FAIL (1.27e-14)
//          delta = 1e-14  ColumnSums FAIL, RowSums PASS, runner PASS (1.6e-15)
//          delta = 1e-16  ColumnSums FAIL, RowSums PASS
//          => the ROW property is blind to this entire class of error at every
//             magnitude tested. This is why the conservation check has to be
//             the column form. The matrix-level column check reaches 1e-16
//             absolute because the 1024src->1tgt schedule in the sweep
//             amplifies an absolute weight error by dt_tgt/dt_src = 1024.
//
//   MUT-C  `w_scale` inverted: (src_interval/dst_interval) -> the reciprocal.
//          A plausible transcription slip, and it only touches degree >= 1.
//          => runner integral defect 1.78 (178%) at DG(1), 0 at Constant(0),
//             plus 10 matrix-level failures.
//
//   MUT-D  degree-0 slab weight normalised by src_interval instead of
//          dst_interval.
//          => runner integral defect 0.667 at Constant(0), 1.9e-16 at DG(1),
//             plus 6 matrix-level failures.
//
//   Under BOTH MUT-C and MUT-D the runtime's own `Constrain` operator still
//   reported `conservation_checks == 6`, `conservation_violations == 0` and
//   `worst_conservation_error == 0` exactly. That is the empirical form of the
//   finding in section 5: `Constrain` does not see the temporal operator.

namespace {

/// The schedule the perturbation tests run on: non-nesting, coarsening, with
/// negative weights in the DG(1) operator, so nothing about it is degenerate.
const Ratio kProbe{5, 3, Rational{1}};

/// Smallest relative perturbation of a single weight, over a geometric ladder,
/// that pushes `column_defect` above `tol`.
[[nodiscard]] double smallest_detected(const TemporalProjection& base,
                                       double tol, std::size_t which) {
  double detected = 1.0;
  for (double eps = 1e-6; eps > 1e-17; eps *= 0.1) {
    TemporalProjection t = base;
    t.val[which] *= (1.0 + eps);
    if (column_defect(t) > tol) detected = eps;
  }
  return detected;
}

}  // namespace

TEST(SpacetimeMutation, TheColumnCheckCatchesAPerturbationOfEveryWeight) {
  const auto base = build(kProbe, kBases[3]);  // DG(1)
  ASSERT_LT(column_defect(base), kIdentityTol);
  ASSERT_GT(base.nnz(), 6u);

  constexpr double kTol = kIdentityTol;
  // Task 5.2's spatial tolerances catch a 1e-9 (one part per billion) weight
  // perturbation. This asserts three orders better, because it is what was
  // measured: at `kIdentityTol` every weight of the probe matrix is caught at
  // 1e-12 relative, and the loop below prints the worst case so the claim can
  // be re-derived rather than taken on trust.
  constexpr double kEps = 1e-12;
  double           worst_smallest = 0.0;
  for (std::size_t k = 0; k < base.nnz(); ++k) {
    TemporalProjection t = base;
    t.val[k] *= (1.0 + kEps);
    ASSERT_GT(column_defect(t), kTol)
        << "a " << kEps << " relative perturbation of weight " << k
        << " (value " << base.val[k] << ") slipped past the column check";
    worst_smallest = std::max(worst_smallest, smallest_detected(base, kTol, k));
  }
  std::cout << "[ SENSITIVITY ] column check, tolerance " << kTol
            << ": every one of " << base.nnz()
            << " weights is caught at a relative perturbation of " << kEps
            << "; the largest 'smallest detected' over all weights is "
            << worst_smallest << "\n";
  testing::Test::RecordProperty("smallest_detected_perturbation",
                                std::to_string(worst_smallest));
}

/// A perturbation that leaves the ROW sums intact but breaks the COLUMN sums.
/// This is the case a row-only check cannot see, and it is not exotic: moving
/// weight between two columns of the same row is exactly what a wrong stencil
/// offset does.
TEST(SpacetimeMutation, AWeightMovedWithinARowIsInvisibleToTheRowCheck) {
  const auto  base = build(kProbe, kBases[3]);
  std::size_t a = base.nnz(), b = base.nnz();
  for (std::size_t k = 1; k < base.nnz(); ++k) {
    if (base.row[k] == base.row[k - 1]) {
      a = k - 1;
      b = k;
      break;
    }
  }
  ASSERT_LT(a, base.nnz()) << "the probe matrix has no row with two entries";

  TemporalProjection t     = base;
  const double       delta = 1e-12;
  t.val[a] += delta;
  t.val[b] -= delta;

  EXPECT_LT(row_defect(t), 1e-14) << "the row check should be blind to this";
  EXPECT_GT(column_defect(t), kIdentityTol)
      << "the column check missed weight moved between two columns of one row "
         "-- exactly the error a wrong stencil offset produces";
}

/// The converse: a uniform rescale breaks both. Recorded so the report can
/// state which defect a given class of bug shows up in.
TEST(SpacetimeMutation, AUniformRescaleBreaksBothProperties) {
  const auto         base = build(kProbe, kBases[3]);
  TemporalProjection t    = base;
  for (double& v : t.val) v *= (1.0 + 1e-12);
  EXPECT_GT(row_defect(t), kIdentityTol);
  EXPECT_GT(column_defect(t), kIdentityTol);
}

// ===========================================================================
// 3. The composition `M = T (x) W`, at the library level
// ===========================================================================
//
// `(T (x) W) = (T (x) I)(I (x) W)`, which is how the runtime factors it (a
// Sample/Interpolate pair for the temporal half and a separate Remap node for
// the spatial half). The two properties factorise the same way:
//
//   ROW      sum_{j,c} M_{(i,r),(j,c)} = (sum_j T_ij)(sum_c W_rc)
//   COLUMN   sum_{i,r} M_{(i,r),(j,c)} dt_tgt m^t_r
//                = (sum_i T_ij dt_tgt)(sum_r W_rc m^t_r)
//
// so the space-time map conserves exactly when both factors do, and reproduces
// constants exactly when both factors do. Task 5.2's result -- that the two
// cannot both hold unless the two meshes report the same total measure -- is
// therefore inherited unchanged by the space-time operator. It is not a new
// defect and it is not a defect of `T`.

namespace {

/// An exact 1-D conservative supermesh operator between two uniform partitions
/// of [0, 1]. Built here rather than taken from `build_weights` because
/// `build_weights` needs MOAB, which is off in the gating presets; the
/// geometry is simple enough that the weights are exact by construction, which
/// is what a composition test needs.
///
/// `W_rc = |tgt cell r ∩ src cell c| / m^t_r`, so the row sums are one and
/// `sum_r W_rc m^t_r = m^s_c` exactly -- both properties hold, because both
/// partitions cover the same interval with the same total measure.
struct LineOverlap {
  RemapWeights        w;
  std::vector<double> src_measure, tgt_measure;
};

LineOverlap line_overlap(std::size_t n_src, std::size_t n_tgt,
                         double tgt_measure_scale = 1.0) {
  LineOverlap  o;
  o.w.n_rows = n_tgt;
  o.w.n_cols = n_src;
  const double hs = 1.0 / static_cast<double>(n_src);
  const double ht = 1.0 / static_cast<double>(n_tgt);
  o.src_measure.assign(n_src, hs);
  o.tgt_measure.assign(n_tgt, ht * tgt_measure_scale);
  for (std::size_t r = 0; r < n_tgt; ++r) {
    for (std::size_t c = 0; c < n_src; ++c) {
      const double lo =
          std::max(static_cast<double>(r) * ht, static_cast<double>(c) * hs);
      const double hi = std::min(static_cast<double>(r + 1) * ht,
                                 static_cast<double>(c + 1) * hs);
      if (hi <= lo) continue;
      o.w.row.push_back(static_cast<std::int32_t>(r));
      o.w.col.push_back(static_cast<std::int32_t>(c));
      o.w.val.push_back((hi - lo) / ht);  // normalised by the TARGET measure
    }
  }
  o.w.src_measure_total =
      std::accumulate(o.src_measure.begin(), o.src_measure.end(), 0.0);
  o.w.tgt_measure_total =
      std::accumulate(o.tgt_measure.begin(), o.tgt_measure.end(), 0.0);
  return o;
}

[[nodiscard]] double w_row_defect(const RemapWeights& w) {
  std::vector<double> row(w.n_rows, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    row[static_cast<std::size_t>(w.row[k])] += w.val[k];
  }
  double worst = 0.0;
  for (const double r : row) worst = std::max(worst, std::fabs(r - 1.0));
  return worst;
}

[[nodiscard]] double w_col_defect(const RemapWeights&        w,
                                  const std::vector<double>& src_measure,
                                  const std::vector<double>& tgt_measure) {
  std::vector<double> col(w.n_cols, 0.0);
  for (std::size_t k = 0; k < w.val.size(); ++k) {
    col[static_cast<std::size_t>(w.col[k])] +=
        w.val[k] * tgt_measure[static_cast<std::size_t>(w.row[k])];
  }
  double worst = 0.0;
  for (std::size_t c = 0; c < w.n_cols; ++c) {
    worst =
        std::max(worst, std::fabs(col[c] - src_measure[c]) / src_measure[c]);
  }
  return worst;
}

/// The space-time column defect, measured THROUGH `apply_spacetime` rather
/// than from the triplets: for each (source step j, source cell c) push a unit
/// impulse through the real entry point and integrate the result over the
/// target space-time grid. That integral must be `dt_src * m^s_c`.
///
/// This is the check spec §7.4 describes, and it is deliberately built on the
/// same function the runtime calls -- a defect in `apply_spacetime` itself
/// (a dropped component, a mis-strided buffer) shows up here and would not
/// show up in a triplet-only check.
///
/// `col_stride` samples the source columns rather than visiting all of them.
/// The cost is O(n_src_steps * n_cols/stride * n_tgt_steps * nnz), which on
/// the 3456-cell cubed sphere is ~7e8 dense operations at stride 1 -- too slow
/// for a unit test under ASan. Every MOAB-free caller uses stride 1.
[[nodiscard]] double spacetime_column_defect(
    const TemporalProjection& t, const RemapWeights& w,
    const std::vector<double>& src_measure,
    const std::vector<double>& tgt_measure, std::size_t col_stride = 1) {
  const SpatialWeightsView view   = soma::spatial_view(w);
  const double             dt_tgt = t.dst_interval.to_double();
  const double             dt_src = t.src_interval.to_double();

  double              worst = 0.0;
  std::vector<double> src(w.n_cols, 0.0);
  std::vector<double> dst(w.n_rows, 0.0);
  for (std::size_t j = 0; j < t.n_src_steps; ++j) {
    for (std::size_t c = 0; c < w.n_cols; c += col_stride) {
      std::fill(src.begin(), src.end(), 0.0);
      src[c]          = 1.0;
      double integral = 0.0;
      for (std::size_t i = 0; i < t.n_tgt_steps; ++i) {
        std::fill(dst.begin(), dst.end(), 0.0);
        const auto r = soma::apply_spacetime(view, t, src.data(), dst.data(),
                                             w.n_cols, 1, j, i);
        EXPECT_TRUE(r.ok()) << (r.ok() ? std::string{} : r.error().message);
        for (std::size_t e = 0; e < w.n_rows; ++e) {
          integral += dst[e] * tgt_measure[e] * dt_tgt;
        }
      }
      const double want = dt_src * src_measure[c];
      worst = std::max(worst, std::fabs(integral - want) / want);
    }
  }
  return worst;
}

/// The space-time ROW defect: a field constant in space and time must arrive
/// as that constant on every target step and every target cell.
[[nodiscard]] double spacetime_row_defect(const TemporalProjection& t,
                                          const RemapWeights&       w) {
  const SpatialWeightsView  view = soma::spatial_view(w);
  const std::vector<double> src(w.n_cols, 1.0);
  double                    worst = 0.0;
  std::vector<double>       dst(w.n_rows, 0.0);
  for (std::size_t i = 0; i < t.n_tgt_steps; ++i) {
    std::fill(dst.begin(), dst.end(), 0.0);
    for (std::size_t j = 0; j < t.n_src_steps; ++j) {
      const auto r = soma::apply_spacetime(view, t, src.data(), dst.data(),
                                           w.n_cols, 1, j, i);
      EXPECT_TRUE(r.ok()) << (r.ok() ? std::string{} : r.error().message);
    }
    for (const double v : dst) worst = std::max(worst, std::fabs(v - 1.0));
  }
  return worst;
}

}  // namespace

/// Both meshes cover [0, 1] and report the same total measure, so both
/// properties survive the tensor product to rounding. This is the case Task
/// 5.2 calls "measures additive over the supermesh".
TEST(SpacetimeComposition, BothPropertiesSurviveWhenTheTotalMeasuresAgree) {
  const LineOverlap o = line_overlap(7, 4);
  ASSERT_LT(w_row_defect(o.w), 1e-15);
  ASSERT_LT(w_col_defect(o.w, o.src_measure, o.tgt_measure), 1e-15);
  ASSERT_NEAR(o.w.src_measure_total, o.w.tgt_measure_total, 1e-15);

  double      worst_col = 0.0, worst_row = 0.0;
  std::string worst_at;
  for (const Ratio& r : {Ratio{5, 3, Rational{1}}, Ratio{3, 5, Rational{1}},
                         Ratio{4, 4, Rational{1}}, Ratio{7, 2, Rational{2}}}) {
    for (const Basis& b : kBases) {
      const auto   t = build(r, b);
      const double dc =
          spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure);
      const double dr = spacetime_row_defect(t, o.w);
      if (dc > worst_col) {
        worst_col = dc;
        worst_at  = std::string(b.label) + " " + r.label();
      }
      worst_row = std::max(worst_row, dr);
      ASSERT_LT(dc, kIdentityTol) << b.label << " " << r.label();
      ASSERT_LT(dr, kIdentityTol) << b.label << " " << r.label();
    }
  }
  std::cout << "[ RESIDUAL ] T (x) W, equal total measures: worst space-time "
               "column defect "
            << worst_col << " (at " << worst_at << "), worst row defect "
            << worst_row << "\n";
}

/// TASK 5.2'S MUTUAL EXCLUSION, inherited by the space-time operator.
///
/// Take `f == c`. The row property gives `g == c`; the column property then
/// forces `c * sum m^t == c * sum m^s`. So both can hold only if the two
/// meshes report the same total measure. Here the target measures are inflated
/// by a known factor to stand in for the real effect Task 5.2 measured on the
/// cubed-sphere pair (`MeshView::measure` integrates the bilinear patch, which
/// is inscribed and short of the spherical area, giving a 9.87e-4 relative
/// difference in the totals). The arithmetic is the same and the point that
/// SURVIVES is the same: the residual is a measure mismatch, NOT a defect of
/// `T` or `W`.
///
/// WHAT DOES NOT GENERALISE, found by an independent review of this task: the
/// residual below comes out numerically EQUAL to `kMismatch` only because
/// `line_overlap`'s `tgt_measure_scale` inflates every target cell's measure
/// by the SAME uniform factor, which is why `W`'s own `w_col_defect` (a
/// single relative number, asserted above) also comes out at exactly
/// `kMismatch`. Repeating the construction with a non-uniform per-cell
/// mismatch that totals the same 9.87e-4 gives a space-time column defect of
/// 2.4675e-3 -- 2.5x the total mismatch, not equal to it. The factorisation
/// that IS general -- `(T (x) W)`'s column defect equals `W`'s own column
/// defect, unaffected by `T` -- is what `st_col == w_col_defect(o.w, ...)`
/// (both computed above) is really evidence of; "the space-time defect equals
/// the scalar `kMismatch`" is a fact about THIS uniform construction's `W`,
/// not a general identity relating a measure mismatch's total size to the
/// resulting defect.
TEST(SpacetimeComposition, ConservationAndConstantPreservationAreExclusive) {
  constexpr double  kMismatch = 9.87e-4;  // Task 5.2's cubed-sphere number
  const LineOverlap o         = line_overlap(7, 4, 1.0 + kMismatch);
  ASSERT_NEAR(o.w.tgt_measure_total / o.w.src_measure_total - 1.0, kMismatch,
              1e-15);

  // The hand-built `W` is normalised by the TRUE overlap geometry, so it is
  // the constant-preserving one; its measure-weighted column sums are then off
  // by exactly the mismatch.
  EXPECT_LT(w_row_defect(o.w), 1e-15);
  EXPECT_NEAR(w_col_defect(o.w, o.src_measure, o.tgt_measure), kMismatch,
              kIdentityTol);

  const auto t = build(Ratio{5, 3, Rational{1}}, kBases[3]);
  ASSERT_LT(column_defect(t), kIdentityTol);
  ASSERT_LT(row_defect(t), kIdentityTol);

  // `T` contributes nothing to the discrepancy: the space-time defects are the
  // spatial ones, unchanged, because the properties factorise.
  const double st_row = spacetime_row_defect(t, o.w);
  const double st_col =
      spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure);
  EXPECT_LT(st_row, kIdentityTol)
      << "constant preservation survived, as the row factorisation predicts";
  EXPECT_NEAR(st_col, kMismatch, 1e-12)
      << "in THIS uniform-measure-scaling construction the space-time column "
         "defect equals kMismatch because W's own column defect does (see "
         "the assertion above); that is a property of this construction, not "
         "a general 'the residual always equals the total mismatch' identity "
         "-- a non-uniform mismatch totalling the same 9.87e-4 measures "
         "2.4675e-3 here, 2.5x the total, per an independent review of this "
         "task";

  std::cout << "[ RESIDUAL ] T (x) W with a " << kMismatch
            << " total-measure mismatch: space-time row defect " << st_row
            << ", space-time column defect " << st_col
            << " (= W's own column defect in this UNIFORM construction; not "
               "a general equality -- see the comment above the test)\n";
}

/// The composition inherits the sensitivity too: perturbing one temporal
/// weight shows up in the space-time column defect at the same order.
TEST(SpacetimeComposition, APerturbedTemporalWeightShowsUpInTheSpaceTimeDefect) {
  const LineOverlap o = line_overlap(7, 4);
  auto              t = build(Ratio{5, 3, Rational{1}}, kBases[3]);
  ASSERT_LT(spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure),
            kIdentityTol);
  t.val[t.nnz() / 2] *= (1.0 + 1e-12);
  EXPECT_GT(spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure),
            kIdentityTol);
}

/// ... and perturbing one spatial weight likewise. Both halves are load
/// bearing; neither can be deleted without a failure here.
TEST(SpacetimeComposition, APerturbedSpatialWeightShowsUpInTheSpaceTimeDefect) {
  LineOverlap o = line_overlap(7, 4);
  const auto  t = build(Ratio{5, 3, Rational{1}}, kBases[3]);
  ASSERT_LT(spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure),
            kIdentityTol);
  o.w.val[o.w.val.size() / 2] *= (1.0 + 1e-12);
  EXPECT_GT(spacetime_column_defect(t, o.w, o.src_measure, o.tgt_measure),
            kIdentityTol);
}

#ifdef SOMA_HAVE_MOAB
// The same composition against weights `build_weights` actually produced.
// Compiled only when MOAB is available; the three gating presets have
// SOMA_ENABLE_MOAB=OFF, so these do not run there. See the report.
#include <soma/mesh/moab_backend.hpp>

namespace {

struct MoabPair {
  CouplingGraph         graph;
  soma::MoabMeshBackend backend;
  soma::RegionId        a, b;
};

std::unique_ptr<MoabPair> moab_pair(const std::string& sa,
                                    const std::string& sb, int dim) {
  auto       p  = std::make_unique<MoabPair>();
  const auto ma = p->graph.add_mesh(soma::MeshDecl{.name = "a", .source = sa});
  const auto mb = p->graph.add_mesh(soma::MeshDecl{.name = "b", .source = sb});
  p->a          = p->graph.add_region(
      soma::Region{.name = "ra", .mesh = ma, .dimension = dim});
  p->b = p->graph.add_region(
      soma::Region{.name = "rb", .mesh = mb, .dimension = dim});
  EXPECT_TRUE(p->backend.load(p->graph.mesh(ma)).ok());
  EXPECT_TRUE(p->backend.load(p->graph.mesh(mb)).ok());
  return p;
}

std::vector<double> measures_of(const MoabPair& p, soma::RegionId r) {
  const auto id = p.backend.mesh_id(p.graph.mesh(p.graph.region(r).mesh).name);
  EXPECT_TRUE(id.ok());
  const auto v = p.backend.view(id.value());
  EXPECT_TRUE(v.ok());
  const int           dim = p.graph.region(r).dimension;
  std::vector<double> m(v.value()->entity_count(dim));
  for (std::size_t i = 0; i < m.size(); ++i) m[i] = v.value()->measure(dim, i);
  return m;
}

}  // namespace

TEST(SpacetimeCompositionMoab, RealWeightsOnAnAxisAlignedPair) {
  auto                  p = moab_pair("generated:box:4x4x4",
                                      "generated:box:5x5x5", 3);
  const soma::RemapPlan plan{
      .src = p->a, .tgt = p->b, .flags = soma::RemapFlags::Conservative};
  const auto w = soma::build_weights(plan, p->graph, p->backend);
  ASSERT_TRUE(w.ok()) << w.error().message;

  const std::vector<double> ms = measures_of(*p, p->a);
  const std::vector<double> mt = measures_of(*p, p->b);
  std::cout << "[ MOAB ] axis-aligned pair: W row defect "
            << w.value().max_row_defect << ", W col defect "
            << w.value().max_col_defect << ", totals "
            << w.value().src_measure_total << " vs "
            << w.value().tgt_measure_total << "\n";

  const auto   t      = build(Ratio{5, 3, Rational{1}}, kBases[3]);
  const double st_col = spacetime_column_defect(t, w.value(), ms, mt);
  const double st_row = spacetime_row_defect(t, w.value());
  std::cout << "[ MOAB ] T (x) W space-time row defect " << st_row
            << ", column defect " << st_col << "\n";
  EXPECT_LT(st_col, 1e-11);
  EXPECT_LT(st_row, 1e-11);
}

#ifdef SOMA_MOAB_TEST_MESH
/// Real, unstructured, non-planar geometry: MOAB's shipped 6x24^2 cubed
/// sphere, 3456 warped quadrilaterals, loaded twice. The supermesh machinery
/// runs for real and the composition with `T` is exercised through
/// `apply_spacetime`.
///
/// SCOPE, stated because the name of an earlier draft of this test overclaimed
/// it: this is a SELF pair, so the two total measures agree and Task 5.2's
/// mutual exclusion does not bite here. The genuinely mismatched pair -- the
/// same 6x24^2 sphere against an equiangular 6x16^2 partner, whose totals
/// differ by 9.87e-4 -- is measured by
/// `RemapSphere.ConservesOnANonMatchingCubedSpherePair` in remap_test.cpp, and
/// what this file adds is the argument (verified above on the hand-built
/// operator) that the space-time defect is exactly that spatial defect,
/// because the two properties factorise across the tensor product.
TEST(SpacetimeCompositionMoab, RealWeightsOnMoabsCubedSphere) {
  auto p = moab_pair(std::string{SOMA_MOAB_TEST_MESH},
                     std::string{SOMA_MOAB_TEST_MESH}, 2);
  const soma::RemapPlan plan{
      .src = p->a, .tgt = p->b, .flags = soma::RemapFlags::Conservative};
  const auto w = soma::build_weights(plan, p->graph, p->backend);
  ASSERT_TRUE(w.ok()) << w.error().code << ": " << w.error().message;

  const std::vector<double> ms = measures_of(*p, p->a);
  const std::vector<double> mt = measures_of(*p, p->b);
  ASSERT_EQ(ms.size(), 3456u);

  const auto   t = build(Ratio{5, 3, Rational{1}}, kBases[3]);
  // Stride 37 is coprime with 3456, so the sampled columns walk the whole
  // mesh rather than one panel of it.
  const double st_col = spacetime_column_defect(t, w.value(), ms, mt, 37);
  const double st_row = spacetime_row_defect(t, w.value());
  std::cout << "[ MOAB ] cubed sphere (self pair, 3456 cells): W row defect "
            << w.value().max_row_defect << ", W col defect "
            << w.value().max_col_defect << ", totals "
            << w.value().src_measure_total << " vs "
            << w.value().tgt_measure_total << "; space-time row defect "
            << st_row << ", column defect " << st_col << "\n";
  EXPECT_LT(st_col, 1e-11);
  EXPECT_LT(st_row, 1e-11);
}
#endif  // SOMA_MOAB_TEST_MESH
#endif  // SOMA_HAVE_MOAB

// ===========================================================================
// 4. Through the runner
// ===========================================================================
//
// A matrix identity is not a runtime property. This section integrates a
// genuinely time-varying, spatially varying source over one coupling period
// and compares the space-time integral the source produced with the one the
// target received, taking the target values out of the running coupler.
//
// A time-CONSTANT source cannot discriminate: it passes against a plain copy,
// as Task 6.3 documented. The source below changes on every step and every
// cell.
//
// WHAT IS AND IS NOT EXERCISED. `OpKind::Remap` is `num::copy` (see
// src/runtime/runner.cpp) and `W` is not wired in, so the spatial half of the
// operator running here is the identity. Both regions are the same 4-cell
// line, so the spatial measures cancel and what is being verified is `T`'s
// column identity carried through Sample -> Interpolate -> Transport ->
// Remap -> Constrain -> Diagnose -> Sink.
//
// OBSERVING EVERY TARGET STEP. `Runner` exposes buffers only between periods,
// and only the LAST target step's value survives to the end of a period. The
// intermediate ones are read here from inside the SOURCE component's
// `advance()`, which the merged tick/instruction walk interleaves between the
// target-step writes. The address of the receiver's registered array is fixed
// for the whole run (`Component::register_fields`'s contract), so a pointer
// captured before the first period stays valid. Which source ticks fall
// between which target writes is derived from the schedule, not assumed -- see
// `recover_target_steps` -- and every target step must be covered or the test
// fails.

namespace {

constexpr std::size_t kCells = 4;

/// Source: value depends on the step index AND the cell, with no symmetry and
/// a monotone component within a period, so nothing cancels by accident and a
/// copy of the last step is a visibly different integral from a projection.
class Varying final : public soma::Component {
 public:
  explicit Varying(std::size_t cells = kCells)
      : data_(cells, 0.0), width_(cells) {}

  [[nodiscard]] std::string name() const override { return "src"; }

  soma::Result<void> register_fields(FieldRegistry&       reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("src");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(
          f, soma::FieldRef{data_.data(),
                            soma::Layout{.stride = 1, .count = data_.size()},
                            soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext& ctx) override {
    // Read whatever the coupler has most recently delivered to the target, as
    // of the START of this advance. See the observability note above.
    if (watch != nullptr) {
      observed_at.push_back(ctx.t_begin);
      observed.emplace_back(watch, watch + width_);
    }
    for (std::size_t c = 0; c < data_.size(); ++c) {
      const double s = static_cast<double>(step_);
      data_[c]       = 1.0 + 0.4 * s + std::sin(1.7 * s + 0.3) +
                 0.25 * std::cos(0.11 * s + static_cast<double>(c)) +
                 0.5 * static_cast<double>(c);
    }
    ++step_;
    emitted.push_back(data_);
    return soma::Result<void>::success();
  }

  const double* watch = nullptr;
  /// The field this component wrote, one entry per advance.
  std::vector<std::vector<double>> emitted;
  /// The target-side array as it stood at the START of each advance.
  std::vector<std::vector<double>> observed;
  std::vector<Rational>            observed_at;

 private:
  std::vector<double> data_;
  std::size_t         width_ = kCells;
  std::int64_t        step_  = 0;
};

class Receiver final : public soma::Component {
 public:
  explicit Receiver(std::size_t cells = kCells) : data_(cells, 0.0) {}

  [[nodiscard]] std::string name() const override { return "dst"; }

  soma::Result<void> register_fields(FieldRegistry&       reg,
                                     const CouplingGraph& g) override {
    const auto cid = g.find_component("dst");
    for (const soma::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(
          f, soma::FieldRef{data_.data(),
                            soma::Layout{.stride = 1, .count = data_.size()},
                            soma::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return soma::Result<void>::success();
  }

  soma::Result<void> advance(const StepContext& ctx) override {
    // What this component actually consumes: its own input array, as of the
    // start of its own step. Recorded so the exchange's visibility can be
    // measured rather than assumed.
    consumed.push_back(data_[0]);
    consumed_at.push_back(ctx.t_begin);
    return soma::Result<void>::success();
  }

  [[nodiscard]] const double* data() const { return data_.data(); }

  std::vector<double>   consumed;
  std::vector<Rational> consumed_at;

 private:
  std::vector<double> data_;
};

struct RunOpts {
  Rational           src_dt{1, 5};
  Rational           dst_dt{1, 3};
  TemporalBasis      basis  = TemporalBasis::DG;
  int                degree = 1;
  soma::Conservation conservation = soma::Conservation::None;
  bool               enable_spacetime = true;
};

struct RunHarness {
  CouplingGraph       graph;
  SerialMeshBackend   backend;
  soma::OperatorDAG   dag;
  soma::ExecutionPlan plan;
  ComponentSet        components;
  FieldRegistry       registry;

  [[nodiscard]] soma::ArcId arc_of(std::string_view label) const {
    for (std::uint32_t i = 0; i < dag.node_count(); ++i) {
      const soma::Node& n = dag.node(soma::NodeId{i});
      if (n.label == label) return n.output;
    }
    return soma::ArcId{};
  }
  [[nodiscard]] Varying* source() {
    return dynamic_cast<Varying*>(components.find("src"));
  }
  [[nodiscard]] Receiver* receiver() {
    return dynamic_cast<Receiver*>(components.find("dst"));
  }
};

std::unique_ptr<RunHarness> make_run(RunOpts o) {
  auto           h = std::make_unique<RunHarness>();
  CouplingGraph& g = h->graph;

  const std::string src = "generated:line:" + std::to_string(kCells);
  const auto m1 = g.add_mesh(soma::MeshDecl{.name = "m1", .source = src});
  const auto m2 = g.add_mesh(soma::MeshDecl{.name = "m2", .source = src});
  const auto r1 =
      g.add_region(soma::Region{.name = "r1", .mesh = m1, .dimension = 1});
  const auto r2 =
      g.add_region(soma::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto pc = g.add_component(soma::ComponentDecl{
      .name = "src", .ranks = {.first = 0, .count = 1}, .interval = o.src_dt});
  const auto cc = g.add_component(soma::ComponentDecl{
      .name = "dst", .ranks = {.first = 1, .count = 1}, .interval = o.dst_dt});
  const auto sf = g.add_field(
      pc, soma::FieldSpec{.name         = "out",
                          .region       = r1,
                          .units        = u::kKelvin,
                          .conservation = o.conservation,
                          .intent       = soma::Intent::Provides});
  const auto tf = g.add_field(
      cc, soma::FieldSpec{.name         = "in",
                          .region       = r2,
                          .units        = u::kKelvin,
                          .conservation = o.conservation,
                          .intent       = soma::Intent::Requires});
  const auto ch = g.add_channel("link", sf, tf);
  if (o.enable_spacetime) g.channel(ch).spacetime(o.basis, o.degree);

  EXPECT_TRUE(h->backend.load(g.mesh(m1)).ok());
  EXPECT_TRUE(h->backend.load(g.mesh(m2)).ok());

  const auto lowered = lower(g);
  EXPECT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;

  const soma::ClockSet clocks{g};
  auto                 sched =
      schedule(g, h->dag, clocks, soma::make_region_sizes(g, h->backend));
  EXPECT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;

  h->components.add(std::make_unique<Varying>());
  h->components.add(std::make_unique<Receiver>());
  return h;
}

/// The cell measures the runner itself would use for a conservation sum.
std::vector<double> region_measures(RunHarness& h, const char* mesh) {
  const auto id = h.backend.mesh_id(mesh);
  EXPECT_TRUE(id.ok());
  const auto v = h.backend.view(id.value());
  EXPECT_TRUE(v.ok());
  std::vector<double> m(kCells);
  for (std::size_t i = 0; i < kCells; ++i) m[i] = v.value()->measure(1, i);
  return m;
}

struct TargetSteps {
  std::vector<std::vector<double>> value;
  std::vector<bool>                seen;
};

/// Recovers every target step's delivered value for period `period_index`.
///
/// Target step `i` is written by the chain instruction stamped `(i+1)*dt_tgt`,
/// so an observation taken at the start of a source advance at period-local
/// time `t` shows the value written by the LAST target-step boundary at or
/// before `t` -- and on the 1/5-against-1/3 schedule these tests use, the two
/// clocks share no instant inside a period, so that is exactly the single
/// largest `i` with `(i+1)*dt_tgt < t`. Not every such `i`: assigning to all of
/// them would silently overwrite the earlier steps with a later step's value,
/// and the resulting integral would be wrong in a way that looks like a
/// conservation defect. The last target step is written at the period end,
/// after every source tick, and is read from the receiver's array afterwards.
///
/// A step left unseen is reported through `seen` rather than defaulted, so a
/// schedule on which the observation trick does not work fails loudly.
TargetSteps recover_target_steps(const Varying& src, Rational dst_dt,
                                 Rational period, std::int64_t period_index,
                                 const double* final_array,
                                 std::size_t   width = kCells) {
  const auto n_tgt =
      static_cast<std::size_t>((period / dst_dt).to_double() + 0.5);
  TargetSteps out;
  out.value.assign(n_tgt, std::vector<double>{});
  out.seen.assign(n_tgt, false);

  const Rational lo = Rational{period_index} * period;
  const Rational hi = lo + period;
  for (std::size_t k = 0; k < src.observed.size(); ++k) {
    const Rational abs = src.observed_at[k];
    if (abs < lo || !(abs < hi)) continue;
    const Rational t = abs - lo;
    // The largest i with (i+1)*dt_tgt < t, if any.
    std::size_t best  = 0;
    bool        found = false;
    for (std::size_t i = 0; i < n_tgt; ++i) {
      if (Rational{static_cast<std::int64_t>(i) + 1} * dst_dt < t) {
        best  = i;
        found = true;
      }
    }
    if (!found) continue;
    out.value[best] = src.observed[k];
    out.seen[best]  = true;
  }
  out.value[n_tgt - 1].assign(final_array, final_array + width);
  out.seen[n_tgt - 1] = true;
  return out;
}

/// First cell of the value the coupler WROTE for each target step of period
/// `p`, recovered from the source component's snapshots of the receiver array.
///
/// Deliberately independent of the exchange's tick convention, because the
/// convention is what the tests below are about. Target step `i` is written by
/// the chain instruction stamped `(i+1)*dst_dt`, and the source snapshots the
/// array at the start of each of its own advances; the first snapshot at an
/// absolute time STRICTLY GREATER than that stamp therefore shows step `i`'s
/// value under any ordering of the two, because the next write is later still.
/// Nothing here assumes on which side of a tick a write lands.
///
/// A step whose snapshot falls outside the run is simply absent, so the result
/// is short and the callers assert on its length.
std::vector<double> writes_in_period(const Varying& src, std::int64_t p,
                                     Rational dst_dt, Rational period) {
  const auto n_tgt =
      static_cast<std::size_t>((period / dst_dt).to_double() + 0.5);
  std::vector<double> out;
  for (std::size_t i = 0; i < n_tgt; ++i) {
    const Rational stamp = Rational{p} * period +
                           Rational{static_cast<std::int64_t>(i) + 1} * dst_dt;
    for (std::size_t k = 0; k < src.observed.size(); ++k) {
      if (!(stamp < src.observed_at[k])) continue;
      if (src.observed[k].empty()) break;
      out.push_back(src.observed[k][0]);
      break;
    }
  }
  return out;
}

}  // namespace

/// THE RUNNER-LEVEL CONSERVATION RESULT.
///
/// `sum_i sum_c y_{i,c} m_c dt_tgt == sum_j sum_c x_{j,c} m_c dt_src` over one
/// coupling period, with `y` read out of the running coupler's receiver array
/// and `x` recorded by the source component as it produced it.
///
/// 1/5 against 1/3 is non-nesting and coarsening, so the projection is not a
/// copy, DG(1) is not the same matrix as Constant, and the reconstruction
/// stencil is causal.
TEST(SpacetimeRunner, TheTimeIntegralIsConservedThroughTheRunner) {
  for (const int degree : {0, 1}) {
    auto h = make_run(RunOpts{.basis  = degree == 0 ? TemporalBasis::Constant
                                                    : TemporalBasis::DG,
                              .degree = degree});
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    for (auto& c : h->components) {
      ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
    }
    ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
    // Watch the RECEIVER's own array -- the far end of the whole chain, not a
    // coupler intermediate.
    h->source()->watch = h->receiver()->data();

    ASSERT_TRUE(runner.run(h->components, 2).ok());

    const std::vector<double> ms = region_measures(*h, "m1");
    const std::vector<double> mt = region_measures(*h, "m2");
    ASSERT_NEAR(std::accumulate(ms.begin(), ms.end(), 0.0),
                std::accumulate(mt.begin(), mt.end(), 0.0), 1e-15);

    const TargetSteps obs = recover_target_steps(
        *h->source(), Rational{1, 3}, Rational{1}, 1, h->receiver()->data());
    ASSERT_EQ(obs.seen.size(), 3u);
    for (std::size_t i = 0; i < obs.seen.size(); ++i) {
      ASSERT_TRUE(obs.seen[i])
          << "target step " << i
          << " was never observable through the runner's public surface";
    }

    // The source values of the SECOND period, the one being audited.
    const std::size_t n_src = h->source()->emitted.size() / 2;
    ASSERT_EQ(n_src, 5u);
    double src_integral = 0.0;
    for (std::size_t j = 0; j < n_src; ++j) {
      const std::vector<double>& x = h->source()->emitted[n_src + j];
      for (std::size_t c = 0; c < kCells; ++c) src_integral += x[c] * ms[c];
    }
    src_integral *= 1.0 / 5.0;

    double tgt_integral = 0.0;
    for (const std::vector<double>& y : obs.value) {
      for (std::size_t c = 0; c < kCells; ++c) tgt_integral += y[c] * mt[c];
    }
    tgt_integral *= 1.0 / 3.0;

    const double rel =
        std::fabs(tgt_integral - src_integral) / std::fabs(src_integral);
    std::cout << "[ RUNNER ] degree " << degree
              << ": source space-time integral " << src_integral
              << ", target " << tgt_integral << ", relative defect " << rel
              << "\n";
    EXPECT_LT(rel, kRunnerTol) << "degree " << degree;

    // ... and the run must not have been a copy in disguise. The last source
    // step's field, held for the whole period on the target grid, is a
    // visibly different integral.
    double                     copy_integral = 0.0;
    const std::vector<double>& last          = h->source()->emitted[2 * n_src - 1];
    for (std::size_t c = 0; c < kCells; ++c) copy_integral += last[c] * mt[c];
    EXPECT_GT(std::fabs(copy_integral - src_integral) / std::fabs(src_integral),
              1e-2)
        << "the source signal stopped varying enough for this run to tell a "
           "projection from a copy";
  }
}

/// WHEN each projection arrives, not merely that it does. The lag has to be
/// exactly one target step: the projection that closes target step `i` is
/// stamped `(i+1)*dt_tgt`, which is where step `i+1` begins, and step `i+1`
/// must be the advance that consumes it. Anything later is a stale exchange,
/// and anything earlier is not causal -- the projection is not complete before
/// its own step ends.
///
/// This test replaces one that pinned the opposite. Until R2 was fixed the
/// target's tick at `(i+1)*dt_tgt` ran BEFORE the chain stamped there, so what
/// step `i+1` consumed was the write for step `i-1`, two steps back, and the
/// write for step `n_tgt - 2` was overwritten unread every period. That was
/// never a conservation defect -- the value was computed correctly and the
/// space-time integral over the written values was conserved -- which is
/// exactly why it needed a test of its own. See `TickPhase` in
/// include/soma/schedule/expand.hpp for the mechanism.
TEST(SpacetimeRunner, EachProjectionIsConsumedByTheNextTargetAdvance) {
  auto h = make_run(RunOpts{.basis = TemporalBasis::Constant, .degree = 0});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  h->source()->watch = h->receiver()->data();
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  const std::vector<double> wrote =
      writes_in_period(*h->source(), 1, Rational{1, 3}, Rational{1});
  ASSERT_EQ(wrote.size(), 3u);

  const Receiver& rx = *h->receiver();
  ASSERT_EQ(rx.consumed.size(), 9u);  // 3 target steps x 3 periods
  ASSERT_EQ(rx.consumed_at.size(), rx.consumed.size());
  std::cout << "[ VISIBILITY ] target steps written in period 1:";
  for (const double v : wrote) std::cout << " " << v;
  std::cout << "; values the target component consumed at its advances:";
  for (const double v : rx.consumed) std::cout << " " << v;
  std::cout << "\n";

  for (std::size_t i = 0; i < wrote.size(); ++i) {
    // Absolute time of the advance that has to see it: the target step that
    // begins where the projection for step i was stamped.
    const Rational when =
        Rational{1} + Rational{static_cast<std::int64_t>(i) + 1} * Rational{1, 3};
    bool checked = false;
    for (std::size_t k = 0; k < rx.consumed.size(); ++k) {
      if (rx.consumed_at[k] != when) continue;
      EXPECT_NEAR(rx.consumed[k], wrote[i], 1e-12)
          << "the target's advance at t=" << when.to_string()
          << " did not consume the projection for target step " << i
          << " of period 1, which was stamped at exactly that instant";
      checked = true;
    }
    EXPECT_TRUE(checked) << "no target advance begins at t=" << when.to_string();
  }
}

/// R2, THE POINT OF THE FEATURE. A space-time projection exists to give a
/// multi-step target a DIFFERENT value on each of its own steps. Every one of
/// the `n_tgt` values the coupler computes in a period must therefore be
/// observed by the target component exactly once, on the step that follows the
/// one it closes -- the shortest lag causality allows, since the projection for
/// target step `i` is only complete at `(i+1)*dst_dt`.
///
/// Nothing weaker will do. Conservation of the time integral is satisfied by a
/// schedule that drops a sub-step (the value is folded into a later step's
/// sum), so a conservation test cannot see this; and "the last value arrives"
/// is satisfied by a plain copy.
TEST(SpacetimeRunner, EveryTargetStepProjectionReachesTheTargetComponent) {
  auto   h = make_run(RunOpts{.basis = TemporalBasis::Constant, .degree = 0});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  h->source()->watch = h->receiver()->data();
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  const std::vector<double> wrote =
      writes_in_period(*h->source(), 1, Rational{1, 3}, Rational{1});
  ASSERT_EQ(wrote.size(), 3u)
      << "the three target-step writes of period 1 are not all observable";

  // The three must be three different numbers, or the check below would pass
  // on a schedule that delivered one value three times.
  EXPECT_GT(std::fabs(wrote[0] - wrote[1]), 1e-9);
  EXPECT_GT(std::fabs(wrote[1] - wrote[2]), 1e-9);
  EXPECT_GT(std::fabs(wrote[0] - wrote[2]), 1e-9);

  const Receiver& rx = *h->receiver();
  std::vector<std::size_t> at(wrote.size(), 0);
  for (std::size_t i = 0; i < wrote.size(); ++i) {
    std::size_t seen = 0;
    for (std::size_t k = 0; k < rx.consumed.size(); ++k) {
      if (std::fabs(rx.consumed[k] - wrote[i]) > 1e-12) continue;
      ++seen;
      at[i] = k;
    }
    EXPECT_EQ(seen, 1u)
        << "the projection for target step " << i << " of period 1 ("
        << wrote[i] << ") was consumed by the target component " << seen
        << " times; it must be consumed exactly once";
  }
  // ... and in the order they were produced.
  EXPECT_LT(at[0], at[1]);
  EXPECT_LT(at[1], at[2]);
}

/// WHAT SEPARATES A SPACE-TIME CHANNEL FROM A PLAIN ONE, stated as a test
/// because an earlier version of R2's fix got it wrong in exactly this gap.
///
/// Above, three target steps per period receive three DIFFERENT projections,
/// each one target step after the step it summarises. Here, with no temporal
/// operator and one source step per period, the two target steps of a period
/// necessarily receive the SAME number: both chain instances carry
/// `src_step = 0`. So there is nothing for an early delivery to buy, and the
/// scheduler must not buy it -- handing the target the source's state over
/// `[0, 1)` at `t = 1/2` would have it integrate against its own future for no
/// gain in resolution. The plain chain stays explicitly one-period lagged.
///
/// The causality contract this rests on lives on `TickPhase` in
/// include/soma/schedule/expand.hpp; `Runner.NoTargetStepIntegratesAgainst
/// UnfinishedSourceState` in runner_test.cpp checks it over a matrix of
/// schedules with exact provenance.
TEST(PlainChainRunner, ASharedValueIsNotDeliveredAcrossTheTargetTick) {
  auto h = make_run(RunOpts{.src_dt           = Rational{1},
                            .dst_dt           = Rational{1, 2},
                            .enable_spacetime = false});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 3).ok());

  const Varying&  tx = *h->source();
  const Receiver& rx = *h->receiver();
  ASSERT_EQ(tx.emitted.size(), 3u);   // one source step per period
  ASSERT_EQ(rx.consumed.size(), 6u);  // two target steps per period

  // Both chain instances of period p carry source step p, produced over
  // [p, p+1). Neither of the target's steps inside period p may see it -- both
  // begin before p+1. Both of period p+1's steps do.
  const std::vector<double> want = {0.0,
                                    0.0,
                                    tx.emitted[0][0],
                                    tx.emitted[0][0],
                                    tx.emitted[1][0],
                                    tx.emitted[1][0]};
  for (std::size_t k = 0; k < want.size(); ++k) {
    EXPECT_NEAR(rx.consumed[k], want[k], 1e-12)
        << "target advance " << k << " at t=" << rx.consumed_at[k].to_string()
        << " consumed a value from the wrong exchange";
  }
}

/// The two degrees must deliver DIFFERENT per-target-step values on this
/// non-nesting schedule while both conserving. Without this, the conservation
/// result above could be passing against a silently downgraded operator.
TEST(SpacetimeRunner, ConservationHoldsForTwoGenuinelyDifferentOperators) {
  std::vector<std::vector<std::vector<double>>> per_degree;
  for (const int degree : {0, 1}) {
    auto h = make_run(RunOpts{.basis  = degree == 0 ? TemporalBasis::Constant
                                                    : TemporalBasis::DG,
                              .degree = degree});
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    for (auto& c : h->components) {
      ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
    }
    ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
    h->source()->watch = h->receiver()->data();
    ASSERT_TRUE(runner.run(h->components, 2).ok());
    per_degree.push_back(
        recover_target_steps(*h->source(), Rational{1, 3}, Rational{1}, 1,
                             h->receiver()->data())
            .value);
  }
  ASSERT_EQ(per_degree.size(), 2u);
  ASSERT_EQ(per_degree[0].size(), per_degree[1].size());
  bool differs = false;
  for (std::size_t i = 0; i < per_degree[0].size(); ++i) {
    for (std::size_t c = 0; c < kCells; ++c) {
      if (std::fabs(per_degree[0][i][c] - per_degree[1][i][c]) > 1e-6) {
        differs = true;
      }
    }
  }
  EXPECT_TRUE(differs)
      << "Constant and DG(1) delivered the same values through the runner on a "
         "non-nesting schedule; the conservation test can no longer tell a "
         "downgraded operator from the real one";
}

// ===========================================================================
// 5. What `Constrain` actually does on a space-time channel
// ===========================================================================
//
// FINDING, stated up front because it is the substantive one in this section.
//
// Lowering emits the channel as
//
//     Sample -> Interpolate(T) -> Transport -> Remap -> Constrain -> Diagnose
//               -> Sink
//
// (src/ir/rules_channel.cpp, then src/ir/rules_derived.cpp). The runtime
// resolves the Constrain by walking back up the SSA arcs to the nearest
// `Remap` and taking THAT node's input as the "before" value
// (`Runner::Impl::resolve_constrain`, src/runtime/runner.cpp). On a space-time
// channel the Remap's input is the output of the temporal projection, so
//
//     Constrain compares the POST-PROJECTION field with the POST-REMAP field.
//     The temporal operator lies entirely UPSTREAM of both. `Constrain` cannot
//     see it and does not check it.
//
// Two further consequences, both measured below rather than argued:
//
//   * The check is per TARGET STEP and instantaneous. It compares two spatial
//     integrals at one instant. It is not, and structurally cannot be, the
//     time-integral identity of spec §7.4 -- that identity relates a sum over
//     ALL target steps to a sum over ALL source steps.
//   * With `W` not wired in, `Remap` is `num::copy`, so on coincident regions
//     the two integrals are bitwise equal and `worst_conservation_error` reads
//     exactly 0. A reading of 0 therefore does not distinguish "checked and
//     perfect" from "the operator under test was never applied". The count is
//     the only load-bearing statistic, so the tests below assert COUNTS.

namespace {

std::vector<OpKind> kinds_in_stream(const soma::ExecutionPlan& p) {
  std::vector<OpKind> k;
  for (const auto& ins : p.instructions()) k.push_back(ins.kind);
  return k;
}

}  // namespace

TEST(SpacetimeConstrain, IsEmittedAndSitsAfterTheRemapAndBeforeTheSink) {
  auto h = make_run(RunOpts{.conservation = soma::Conservation::Extensive});
  ASSERT_TRUE(h->arc_of("link/constrain").valid())
      << "a space-time channel with an extensive field emitted no Constrain";

  const std::vector<OpKind> k     = kinds_in_stream(h->plan);
  const auto                first = [&](OpKind want) {
    return std::find(k.begin(), k.end(), want) - k.begin();
  };
  EXPECT_LT(first(OpKind::Interpolate), first(OpKind::Remap));
  EXPECT_LT(first(OpKind::Remap), first(OpKind::Constrain));
  EXPECT_LT(first(OpKind::Constrain), first(OpKind::Diagnose));
  // Upstream of the Sink: a conservation statistic is therefore not evidence
  // that the exchange completed.
  EXPECT_LT(first(OpKind::Constrain), first(OpKind::Sink));
}

/// The count, which is the statistic that carries information. One check per
/// target step per period -- Constrain rides the chain, and the chain runs
/// once per target step.
TEST(SpacetimeConstrain, FiresOncePerTargetStepPerPeriod) {
  auto   h = make_run(RunOpts{.conservation = soma::Conservation::Extensive});
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run(h->components, 2).ok());

  // 3 target steps per period, 2 periods.
  EXPECT_EQ(runner.stats().conservation_checks, 6);
  EXPECT_EQ(runner.stats().conservation_violations, 0);
}

/// No Constrain at all when the field declares no conservation class -- so a
/// zero check count is ambiguous in the other direction too, and a test that
/// only asserted `violations == 0` would pass with the operator removed.
TEST(SpacetimeConstrain, DoesNotFireWithoutAConservationClass) {
  auto h = make_run(RunOpts{});  // Conservation::None
  EXPECT_FALSE(h->arc_of("link/constrain").valid());
  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }
  ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
  ASSERT_TRUE(runner.run_period(h->components).ok());
  EXPECT_EQ(runner.stats().conservation_checks, 0);
}

/// THE BLINDNESS, measured.
///
/// The projection genuinely changes the per-target-step spatial integral --
/// the three target steps differ from each other, and the two degrees disagree
/// about all of them. If `Constrain` were checking the space-time identity it
/// would have something to say. It reports exactly zero, for every degree,
/// because both sides of its comparison sit downstream of the projection and
/// `Remap` is an identity copy.
TEST(SpacetimeConstrain, ReportsExactlyZeroAndIsBlindToTheTemporalOperator) {
  double reported[2] = {-1.0, -1.0};
  std::vector<std::vector<std::vector<double>>> delivered;

  for (const int degree : {0, 1}) {
    auto h = make_run(RunOpts{.basis  = degree == 0 ? TemporalBasis::Constant
                                                    : TemporalBasis::DG,
                              .degree = degree,
                              .conservation = soma::Conservation::Extensive});
    Runner runner{h->graph, h->dag, h->plan, h->backend};
    for (auto& c : h->components) {
      ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
    }
    ASSERT_TRUE(runner.initialize(h->components, h->registry).ok());
    h->source()->watch = h->receiver()->data();
    ASSERT_TRUE(runner.run(h->components, 2).ok());

    EXPECT_EQ(runner.stats().conservation_checks, 6) << "degree " << degree;
    reported[degree] = runner.stats().worst_conservation_error;
    delivered.push_back(
        recover_target_steps(*h->source(), Rational{1, 3}, Rational{1}, 1,
                             h->receiver()->data())
            .value);
  }

  // Exactly zero, bitwise, for both operators.
  EXPECT_EQ(reported[0], 0.0);
  EXPECT_EQ(reported[1], 0.0);

  // ... while the thing it did not check is genuinely non-trivial.
  const auto integral = [](const std::vector<double>& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / 4.0;
  };
  ASSERT_EQ(delivered[0].size(), 3u);
  EXPECT_GT(std::fabs(integral(delivered[0][0]) - integral(delivered[0][1])),
            1e-3);
  EXPECT_GT(std::fabs(integral(delivered[0][1]) - integral(delivered[0][2])),
            1e-3);
  bool degrees_disagree = false;
  for (std::size_t i = 0; i < 3; ++i) {
    if (std::fabs(integral(delivered[0][i]) - integral(delivered[1][i])) >
        1e-3) {
      degrees_disagree = true;
    }
  }
  EXPECT_TRUE(degrees_disagree);

  std::cout << "[ CONSTRAIN ] 6 checks per 2-period run, "
               "worst_conservation_error exactly 0 for Constant(0) and DG(1), "
               "which deliver different values -- the check is downstream of "
               "the temporal operator and does not see it\n";
}

/// This test used to give the two regions different total measures, watch
/// `Constrain` fire, and conclude that the operator is not inert -- that what
/// the finding above shows is specifically a blindness to the TEMPORAL half.
///
/// TASK 5.5 REMOVED THE VEHICLE, CORRECTLY. Reaching `Constrain` with a
/// non-conserving spatial transfer required the runner to identity-copy
/// between two regions that are not the same cells in the same order, and 5.5
/// now refuses exactly that at setup: `regions_coincide()` cannot prove
/// correspondence, so the runner tries to build real weights instead, and in a
/// build without MOAB there is no supermesh implementation to build them with.
/// The refusal names both halves --
///
///   E-REMAP-NOMOAB: 'link/remap' must transfer between region 'r1' and region
///   'r2', which are not provably the same cells in the same order (they hold
///   entities of different dimension (2 and 1)), so an element-wise copy would
///   deliver wrong values with no diagnostic. Building conservative weights
///   instead failed: ...
///
/// -- and that is a strictly better outcome than the silent wrong copy this
/// test was built on. So the test now pins the refusal, which is worth pinning
/// in its own right: it is the guard against equal-count, non-corresponding
/// regions, and it must not regress.
///
/// THE POSITIVE EVIDENCE MOVED, it did not disappear. That `Constrain` really
/// does police the spatial integral is shown by
/// `RemapRuntime.ReportsTheMassATruncatingRemapDrops` (Task 5.5,
/// test/unit/remap_runtime_test.cpp), which needs MOAB because it needs real
/// weights: a truncating map over a partial target reports `checks == 1,
/// violations == 1` with a relative error of exactly 1.34375/1.5. Without MOAB
/// there is now NO way to produce a spatial conservation violation at all,
/// because the only route to one was the bug 5.5 closed.
TEST(SpacetimeConstrain, TheCrossMeasureChannelIsNowRefusedAtSetup) {
  // 24 entities on both sides, so the identity `Remap` is legal, but the box's
  // 24 boundary faces have total area 6 while the line's 24 cells have total
  // length 1. (`generated:line:4` against `generated:box:4x1x1` does NOT work
  // for this: both are the unit cube cut up, so both total exactly 1 and there
  // is nothing to violate. Measured, not assumed -- the first draft of this
  // test used that pair and reported zero violations.)
  constexpr std::size_t kWide = 24;
  auto                  h     = std::make_unique<RunHarness>();
  CouplingGraph&        g     = h->graph;
  const auto            m1 =
      g.add_mesh(soma::MeshDecl{.name = "m1", .source = "generated:box:2x2x2"});
  const auto m2 =
      g.add_mesh(soma::MeshDecl{.name = "m2", .source = "generated:line:24"});
  const auto r1 =
      g.add_region(soma::Region{.name = "r1", .mesh = m1, .dimension = 2});
  const auto r2 =
      g.add_region(soma::Region{.name = "r2", .mesh = m2, .dimension = 1});
  const auto pc =
      g.add_component(soma::ComponentDecl{.name     = "src",
                                          .ranks    = {.first = 0, .count = 1},
                                          .interval = Rational{1, 5}});
  const auto cc =
      g.add_component(soma::ComponentDecl{.name     = "dst",
                                          .ranks    = {.first = 1, .count = 1},
                                          .interval = Rational{1, 3}});
  const auto sf = g.add_field(
      pc, soma::FieldSpec{.name         = "out",
                          .region       = r1,
                          .units        = u::kKelvin,
                          .conservation = soma::Conservation::Extensive,
                          .intent       = soma::Intent::Provides});
  const auto tf = g.add_field(
      cc, soma::FieldSpec{.name         = "in",
                          .region       = r2,
                          .units        = u::kKelvin,
                          .conservation = soma::Conservation::Extensive,
                          .intent       = soma::Intent::Requires});
  const auto ch = g.add_channel("link", sf, tf);
  g.channel(ch).spacetime(TemporalBasis::Constant, 0);

  ASSERT_TRUE(h->backend.load(g.mesh(m1)).ok());
  ASSERT_TRUE(h->backend.load(g.mesh(m2)).ok());
  const auto lowered = lower(g);
  ASSERT_TRUE(lowered.ok()) << lowered.diags.to_string();
  h->dag = lowered.dag;
  const soma::ClockSet clocks{g};
  auto                 sched =
      schedule(g, h->dag, clocks, soma::make_region_sizes(g, h->backend));
  ASSERT_TRUE(sched.ok()) << sched.diags.to_string();
  h->plan = sched.plan;
  h->components.add(std::make_unique<Varying>(kWide));
  h->components.add(std::make_unique<Receiver>(kWide));

  Runner runner{h->graph, h->dag, h->plan, h->backend};
  for (auto& c : h->components) {
    ASSERT_TRUE(c.register_fields(h->registry, h->graph).ok());
  }

  const auto init = runner.initialize(h->components, h->registry);
  ASSERT_FALSE(init.ok())
      << "two regions of different dimension holding equal cell counts were "
         "accepted; before Task 5.5 this was silently identity-copied, which "
         "is the defect that guard exists to prevent";

  // The code depends on the build, because the two builds get different
  // distances into the fallback. Without MOAB there is no supermesh
  // implementation at all, so the attempt stops immediately. With MOAB it gets
  // as far as the backend, which cannot intersect a dimension-2 region against
  // a dimension-1 one. Both are refusals; neither is a copy. Both codes are
  // measured, not predicted -- E-REMAP-REGION would have been the natural
  // guess for the MOAB branch and it is wrong.
#ifdef SOMA_HAVE_MOAB
  EXPECT_EQ(init.error().code, "E-REMAP-BACKEND") << init.error().message;
#else
  EXPECT_EQ(init.error().code, "E-REMAP-NOMOAB") << init.error().message;
#endif

  // The diagnostic has to say WHY the copy was refused, not merely that it
  // was: the failure mode being prevented is a wrong answer with no symptom,
  // so a bare error code would leave the user no way to tell this apart from
  // a missing dependency.
  EXPECT_NE(init.error().message.find("not provably the same cells"),
            std::string::npos)
      << init.error().message;

  std::cout << "[ CONSTRAIN ] cross-measure channel is refused at setup: "
            << init.error().code << "\n";
}
