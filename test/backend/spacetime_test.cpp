#include <gtest/gtest.h>
#include <fuse/backend/spacetime.hpp>
#include <fuse/schedule/slab.hpp>

#include <vector>

using fuse::build_temporal_projection;
using fuse::Rational;
using fuse::TemporalBasis;

namespace {

fuse::TemporalProjection projection(Rational src_dt, Rational dst_dt,
                                    TemporalBasis basis, int degree) {
  const auto decomp = fuse::decompose(src_dt, dst_dt, Rational{1});
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
  const auto decomp = fuse::decompose(Rational{1}, Rational{1}, Rational{1});
  ASSERT_TRUE(decomp.ok());
  const auto p = build_temporal_projection(decomp.value(), TemporalBasis::None,
                                           0, Rational{1}, Rational{1});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-BASIS");
}

// ===========================================================================
// Everything above is the task brief's test file, transcribed verbatim.
// Everything below was added while implementing it. Three of the additions
// exist because the brief was wrong about something; those say so in place.
// ===========================================================================

#include <fuse/base/ids.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <utility>

namespace {

struct Case {
  Rational src;
  Rational dst;
  Rational period;
};

/// Interval pairs exercised by every property test below: lockstep, pure
/// subcycling in both directions, two non-nesting pairs whose only common
/// ticks are the ends, a non-unit period with non-unit intervals, and a
/// thousand-step case that puts the reconstruction's local coordinate at a
/// large step index (where a naive `t/dt - j` in double would lose most of
/// its digits).
const Case kCases[] = {
    {Rational{1}, Rational{1}, Rational{1}},
    {Rational{1, 4}, Rational{1}, Rational{1}},
    {Rational{1}, Rational{1, 4}, Rational{1}},
    {Rational{1, 3}, Rational{1, 5}, Rational{1}},
    {Rational{1, 7}, Rational{1, 4}, Rational{1}},
    {Rational{2, 3}, Rational{2, 5}, Rational{2}},
    {Rational{1, 1000}, Rational{1, 999}, Rational{1}},
};

struct Basis {
  TemporalBasis basis;
  int           degree;
  const char*   label;
};

const Basis kBases[] = {
    {TemporalBasis::Constant, 0, "Constant"},
    {TemporalBasis::Linear, 1, "Linear"},
    {TemporalBasis::DG, 0, "DG(0)"},
    {TemporalBasis::DG, 1, "DG(1)"},
    {TemporalBasis::DG, 2, "DG(2)"},
    {TemporalBasis::DG, 3, "DG(3)"},
    {TemporalBasis::DG, 4, "DG(4)"},
};

fuse::TemporalProjection build(const Case& c, const Basis& b) {
  const auto decomp = fuse::decompose(c.src, c.dst, c.period);
  EXPECT_TRUE(decomp.ok()) << (decomp.ok() ? "" : decomp.error().message);
  auto p = build_temporal_projection(decomp.value(), b.basis, b.degree, c.src,
                                     c.dst);
  EXPECT_TRUE(p.ok()) << (p.ok() ? "" : p.error().message);
  return p.value();
}

std::vector<double> project(const fuse::TemporalProjection& t,
                            const std::vector<double>& x) {
  std::vector<double> y(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.val.size(); ++k) {
    y[static_cast<std::size_t>(t.row[k])] +=
        t.val[k] * x[static_cast<std::size_t>(t.col[k])];
  }
  return y;
}

std::string name(const Case& c, const Basis& b) {
  return std::string(b.label) + " src=" + c.src.to_string() +
         " dst=" + c.dst.to_string() + " period=" + c.period.to_string();
}

}  // namespace

// ---------------------------------------------------------------------------
// PROPERTY 1: constant reproduction (partition of unity)
// ---------------------------------------------------------------------------
// If the weights for a target step do not sum to one, every downstream answer
// is biased by a factor nobody will notice. Note the convention, which the
// header states and this test enforces: the row sums are ONE, not the step
// length, because a coefficient is a step MEAN and not a step integral.

TEST(Spacetime, ReproducesAConstantExactlyForEveryBasis) {
  constexpr double kValue = 3.5;
  double           worst  = 0.0;
  for (const Case& c : kCases) {
    for (const Basis& b : kBases) {
      const auto                t = build(c, b);
      const std::vector<double> x(t.n_src_steps, kValue);
      const std::vector<double> y = project(t, x);
      ASSERT_EQ(y.size(), t.n_tgt_steps) << name(c, b);
      for (std::size_t i = 0; i < y.size(); ++i) {
        const double residual = std::fabs(y[i] - kValue);
        worst                 = std::max(worst, residual);
        ASSERT_LT(residual, 1e-13)
            << name(c, b) << " target step " << i << " got " << y[i];
      }
    }
  }
  std::cout << "[ RESIDUAL ] constant reproduction, worst over all cases: "
            << worst << "\n";
  testing::Test::RecordProperty("constant_reproduction_residual",
                                std::to_string(worst));
}

TEST(Spacetime, RowSumsAreUnityForEveryBasis) {
  double worst = 0.0;
  for (const Case& c : kCases) {
    for (const Basis& b : kBases) {
      const auto          t = build(c, b);
      std::vector<double> row_sum(t.n_tgt_steps, 0.0);
      for (std::size_t k = 0; k < t.val.size(); ++k) {
        row_sum[static_cast<std::size_t>(t.row[k])] += t.val[k];
      }
      for (std::size_t i = 0; i < row_sum.size(); ++i) {
        worst = std::max(worst, std::fabs(row_sum[i] - 1.0));
        ASSERT_NEAR(row_sum[i], 1.0, 1e-13) << name(c, b) << " row " << i;
      }
    }
  }
  std::cout << "[ RESIDUAL ] row-sum-unity, worst |sum - 1|: " << worst << "\n";
  testing::Test::RecordProperty("row_sum_residual", std::to_string(worst));
}

// ---------------------------------------------------------------------------
// PROPERTY 2: conservation of the time integral
// ---------------------------------------------------------------------------
// The projected field integrated over the target grid must equal the source
// integral over the same span. Task 6.4 checks space-time conservation
// globally; a defect here would surface there as an unexplained imbalance.
//
// CORRECTION TO THE BRIEF. The brief states the conservation property as
//
//     for each target step i,  sum_j T_ij * dt_src == dt_tgt
//
// which is a ROW sum and is false for the weights the same brief prescribes.
// Counter-example, from the brief's own SubcyclingAveragesTheSourceSteps: with
// src = 1/4 and dst = 1, every T_0j is 1/4 and dt_src is 1/4, so the left side
// is 4 * (1/4) * (1/4) = 1/4 while dt_tgt is 1. The row sums are unity (tested
// above); the identity that actually carries the time integral is the COLUMN
// sum
//
//     for each source step j,  sum_i T_ij * dt_tgt == dt_src
//
// which is exactly `sum_i (Tx)_i dt_tgt == sum_j x_j dt_src` for every x.
// `ColumnSumsCarryTheSourceStepLength` below is the direct check;
// `ConservesTheTimeIntegral` is the same statement applied to a signal.

TEST(Spacetime, ColumnSumsCarryTheSourceStepLength) {
  double worst = 0.0;
  for (const Case& c : kCases) {
    for (const Basis& b : kBases) {
      const auto   t      = build(c, b);
      const double dt_tgt = c.dst.to_double();
      const double dt_src = c.src.to_double();
      std::vector<double> col_sum(t.n_src_steps, 0.0);
      for (std::size_t k = 0; k < t.val.size(); ++k) {
        col_sum[static_cast<std::size_t>(t.col[k])] += t.val[k] * dt_tgt;
      }
      for (std::size_t j = 0; j < col_sum.size(); ++j) {
        const double rel = std::fabs(col_sum[j] - dt_src) / dt_src;
        worst            = std::max(worst, rel);
        ASSERT_LT(rel, 1e-12) << name(c, b) << " column " << j << " got "
                              << col_sum[j] << ", want " << dt_src;
      }
    }
  }
  std::cout << "[ RESIDUAL ] column sums, worst relative |sum - dt_src|: "
            << worst << "\n";
  testing::Test::RecordProperty("column_sum_relative_residual",
                                std::to_string(worst));
}

TEST(Spacetime, ConservesTheTimeIntegral) {
  double worst = 0.0;
  for (const Case& c : kCases) {
    for (const Basis& b : kBases) {
      const auto t = build(c, b);
      // A signal with no symmetry, so nothing cancels by accident.
      std::vector<double> x(t.n_src_steps);
      for (std::size_t j = 0; j < x.size(); ++j) {
        const double s = static_cast<double>(j);
        x[j] = 1.0 + std::sin(1.7 * s + 0.3) + 0.25 * std::cos(0.11 * s);
      }
      const std::vector<double> y = project(t, x);

      double src_integral = 0.0;
      for (const double v : x) src_integral += v;
      src_integral *= c.src.to_double();
      double tgt_integral = 0.0;
      for (const double v : y) tgt_integral += v;
      tgt_integral *= c.dst.to_double();

      const double rel = std::fabs(tgt_integral - src_integral) /
                         std::max(1.0, std::fabs(src_integral));
      worst = std::max(worst, rel);
      ASSERT_LT(rel, 1e-12) << name(c, b) << " src integral " << src_integral
                            << " tgt integral " << tgt_integral;
    }
  }
  std::cout << "[ RESIDUAL ] time-integral conservation, worst relative: "
            << worst << "\n";
  testing::Test::RecordProperty("conservation_relative_residual",
                                std::to_string(worst));
}

// ---------------------------------------------------------------------------
// Convention and structure
// ---------------------------------------------------------------------------

TEST(Spacetime, LockstepIsTheIdentityForEveryBasis) {
  // Equal intervals: the conservative projection between identical partitions
  // is the identity, for every degree. There is no fast path for this in the
  // implementation -- it falls out of the general reconstruction -- so this is
  // a real check on the general code, not on a special case.
  for (const Basis& b : kBases) {
    const Case c{Rational{1, 4}, Rational{1, 4}, Rational{1}};
    const auto t = build(c, b);
    ASSERT_EQ(t.n_src_steps, 4u);
    ASSERT_EQ(t.n_tgt_steps, 4u);
    for (std::size_t i = 0; i < 4; ++i) {
      for (std::size_t j = 0; j < 4; ++j) {
        EXPECT_NEAR(t.coefficient(i, j), i == j ? 1.0 : 0.0, 1e-14)
            << b.label << " entry (" << i << ", " << j << ")";
      }
    }
  }
}

// Non-negativity holds for the piecewise-constant reconstruction only. A
// higher-degree conservative reconstruction is not monotone and genuinely
// produces negative weights; that is a property of the method, not a bug, and
// pinning it here stops someone "tidying up" by asserting non-negativity
// across the board later.
TEST(Spacetime, HigherDegreeWeightsAreNotAllNonNegative) {
  const Case c{Rational{1, 3}, Rational{1, 5}, Rational{1}};
  const auto t = build(c, Basis{TemporalBasis::DG, 2, "DG(2)"});
  bool       any_negative = false;
  for (const double v : t.val) {
    if (v < -1e-12) any_negative = true;
  }
  EXPECT_TRUE(any_negative)
      << "DG(2) produced no negative weight; either the reconstruction "
         "degenerated to piecewise constant or the test case stopped "
         "exercising a non-monotone stencil";
}

TEST(Spacetime, TripletsAreCanonicalAndDeterministicForEveryBasis) {
  for (const Case& c : kCases) {
    for (const Basis& b : kBases) {
      const auto a = build(c, b);
      const auto d = build(c, b);
      ASSERT_EQ(a.row, d.row) << name(c, b);
      ASSERT_EQ(a.col, d.col) << name(c, b);
      ASSERT_EQ(a.val, d.val) << name(c, b);  // bitwise, not near
      // Strictly ascending in (row, col): sorted, deduplicated, no zeros.
      for (std::size_t k = 1; k < a.val.size(); ++k) {
        const bool ordered = a.row[k - 1] < a.row[k] ||
                             (a.row[k - 1] == a.row[k] && a.col[k - 1] < a.col[k]);
        ASSERT_TRUE(ordered) << name(c, b) << " at k=" << k;
      }
      for (const double v : a.val) ASSERT_NE(v, 0.0) << name(c, b);
      ASSERT_EQ(a.row.size(), a.val.size());
      ASSERT_EQ(a.col.size(), a.val.size());
      ASSERT_EQ(a.nnz(), a.val.size());
    }
  }
}

TEST(Spacetime, CoefficientLookupAgreesWithTheTriplets) {
  const Case c{Rational{1, 3}, Rational{1, 5}, Rational{1}};
  const auto t = build(c, Basis{TemporalBasis::DG, 2, "DG(2)"});
  for (std::size_t k = 0; k < t.nnz(); ++k) {
    EXPECT_EQ(t.coefficient(static_cast<std::size_t>(t.row[k]),
                            static_cast<std::size_t>(t.col[k])),
              t.val[k]);
  }
  // Structurally absent and out of range both read as zero.
  EXPECT_EQ(t.coefficient(t.n_tgt_steps + 5, 0), 0.0);
  EXPECT_EQ(t.coefficient(0, t.n_src_steps + 5), 0.0);
}

TEST(Spacetime, CarriesTheIntervalsItWasBuiltFor) {
  const Case c{Rational{2, 3}, Rational{2, 5}, Rational{2}};
  const auto t = build(c, Basis{TemporalBasis::Linear, 1, "Linear"});
  EXPECT_EQ(t.src_interval, c.src);
  EXPECT_EQ(t.dst_interval, c.dst);
  EXPECT_EQ(t.basis, TemporalBasis::Linear);
  EXPECT_EQ(t.degree, 1);
  EXPECT_EQ(t.effective_degree, 1);
  EXPECT_EQ(t.n_src_steps, 3u);
  EXPECT_EQ(t.n_tgt_steps, 5u);
}

// ---------------------------------------------------------------------------
// Rejections
// ---------------------------------------------------------------------------

namespace {

fuse::SlabDecomposition decomp_of(Rational src, Rational dst, Rational period) {
  const auto d = fuse::decompose(src, dst, period);
  EXPECT_TRUE(d.ok()) << (d.ok() ? "" : d.error().message);
  return d.value();
}

}  // namespace

TEST(Spacetime, SplineBasisIsRejected) {
  // Rejected rather than silently downgraded to a local degree-3
  // reconstruction: a conservative spline projection is globally coupled, and
  // handing back something else under the name Spline is the same silent
  // substitution Waveform's constructor refuses for Spline(5).
  const auto p = build_temporal_projection(
      decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1}),
      TemporalBasis::Spline, 3, Rational{1, 3}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-BASIS");
}

TEST(Spacetime, DegreeMustMatchTheBasis) {
  const auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  const auto src = Rational{1, 3};
  const auto dst = Rational{1, 5};
  for (const auto& [basis, degree] :
       {std::pair{TemporalBasis::Constant, 1}, std::pair{TemporalBasis::Linear, 0},
        std::pair{TemporalBasis::Linear, 2}, std::pair{TemporalBasis::DG, -1},
        std::pair{TemporalBasis::DG, fuse::kMaxTemporalDegree + 1}}) {
    const auto p = build_temporal_projection(d, basis, degree, src, dst);
    ASSERT_FALSE(p.ok()) << to_cstring(basis) << " degree " << degree;
    EXPECT_EQ(p.error().code, "E-ST-DEGREE")
        << to_cstring(basis) << " degree " << degree;
  }
}

// The decomposition is the intersection of two SPECIFIC partitions. Handing in
// one built from other intervals must not produce a plausible-looking matrix:
// the row sums would still be unity (they are normalised by dst_interval), so
// the error would be entirely silent.
TEST(Spacetime, DecompositionBuiltFromOtherIntervalsIsRejected) {
  const auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  const auto p = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 4}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-SLABS") << p.error().message;

  const auto q = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 3}, Rational{1, 6});
  ASSERT_FALSE(q.ok());
  EXPECT_EQ(q.error().code, "E-ST-SLABS") << q.error().message;
}

TEST(Spacetime, AGapInTheSlabsIsRejected) {
  auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  ASSERT_GE(d.slabs.size(), 3u);
  d.slabs.erase(d.slabs.begin() + 1);  // punch a hole
  const auto p = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 3}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-SLABS") << p.error().message;
}

TEST(Spacetime, TruncatedSlabsAreRejected) {
  auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  d.slabs.pop_back();  // stops short of the period
  const auto p = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 3}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-SLABS") << p.error().message;
}

TEST(Spacetime, AWrongStepIndexIsRejected) {
  auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  d.slabs.front().src_step = 2;  // slab [0, 1/5) is in source step 0, not 2
  const auto p = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 3}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-SLABS") << p.error().message;
}

TEST(Spacetime, AnEmptyDecompositionIsRejected) {
  fuse::SlabDecomposition d;
  d.period = Rational{1};
  const auto p = build_temporal_projection(d, TemporalBasis::Constant, 0,
                                           Rational{1, 3}, Rational{1, 5});
  ASSERT_FALSE(p.ok());
  EXPECT_EQ(p.error().code, "E-ST-SLABS");
}

TEST(Spacetime, NonPositiveOrIndivisibleIntervalsAreRejected) {
  const auto d = decomp_of(Rational{1, 3}, Rational{1, 5}, Rational{1});
  for (const auto& [src, dst] :
       {std::pair{Rational{0}, Rational{1, 5}},
        std::pair{Rational{-1, 3}, Rational{1, 5}},
        std::pair{Rational{1, 3}, Rational{0}},
        std::pair{Rational{2, 7}, Rational{1, 5}}}) {  // 1 is not a multiple
    const auto p =
        build_temporal_projection(d, TemporalBasis::Constant, 0, src, dst);
    ASSERT_FALSE(p.ok()) << src.to_string() << " " << dst.to_string();
    EXPECT_EQ(p.error().code, "E-ST-INTERVAL")
        << src.to_string() << " " << dst.to_string() << ": "
        << p.error().message;
  }
}

// ---------------------------------------------------------------------------
// apply_spacetime
// ---------------------------------------------------------------------------

namespace {

/// A 2-row, 3-column spatial operator with unit row sums, so the composition
/// with T also has unit row sums and a constant field survives both halves.
struct TinySpatial {
  std::vector<std::int32_t> row{0, 0, 1, 1};
  std::vector<std::int32_t> col{0, 1, 1, 2};
  std::vector<double>       val{0.25, 0.75, 0.5, 0.5};

  [[nodiscard]] fuse::SpatialWeightsView view() const {
    return fuse::SpatialWeightsView{row.data(), col.data(), val.data(),
                                    val.size(), 2, 3};
  }
};

}  // namespace

TEST(Spacetime, ApplySpacetimeIsTheTensorProduct) {
  const Case  c{Rational{1, 4}, Rational{1, 2}, Rational{1}};
  const auto  t = build(c, Basis{TemporalBasis::Linear, 1, "Linear"});
  TinySpatial w;
  constexpr int kComponents = 2;

  // Source field: 3 spatial points x 2 components, one value per source step.
  const auto value = [](std::size_t step, std::size_t space, int comp) {
    return 1.0 + 0.5 * static_cast<double>(step) +
           0.25 * static_cast<double>(space) + 0.125 * static_cast<double>(comp);
  };

  for (std::size_t tgt = 0; tgt < t.n_tgt_steps; ++tgt) {
    std::vector<double> dst(2 * kComponents, 0.0);
    for (std::size_t s = 0; s < t.n_src_steps; ++s) {
      std::vector<double> src(3 * kComponents);
      for (std::size_t k = 0; k < 3; ++k) {
        for (int e = 0; e < kComponents; ++e) {
          src[k * kComponents + static_cast<std::size_t>(e)] = value(s, k, e);
        }
      }
      const auto r = fuse::apply_spacetime(w.view(), t, src.data(), dst.data(),
                                           3, kComponents, s, tgt);
      ASSERT_TRUE(r.ok()) << r.error().message;
    }
    // Reference: apply W to the time-projected coefficients instead.
    for (std::size_t r = 0; r < 2; ++r) {
      for (int e = 0; e < kComponents; ++e) {
        double want = 0.0;
        for (std::size_t k = 0; k < w.val.size(); ++k) {
          if (static_cast<std::size_t>(w.row[k]) != r) continue;
          double projected = 0.0;
          for (std::size_t s = 0; s < t.n_src_steps; ++s) {
            projected += t.coefficient(tgt, s) *
                         value(s, static_cast<std::size_t>(w.col[k]), e);
          }
          want += w.val[k] * projected;
        }
        EXPECT_NEAR(dst[r * kComponents + static_cast<std::size_t>(e)], want,
                    1e-13)
            << "tgt=" << tgt << " row=" << r << " comp=" << e;
      }
    }
  }
}

TEST(Spacetime, ApplySpacetimePreservesAConstantThroughBothHalves) {
  const Case  c{Rational{1, 3}, Rational{1, 5}, Rational{1}};
  const auto  t = build(c, Basis{TemporalBasis::DG, 2, "DG(2)"});
  TinySpatial w;
  constexpr double kValue = 7.25;

  for (std::size_t tgt = 0; tgt < t.n_tgt_steps; ++tgt) {
    std::vector<double>       dst(2, 0.0);
    const std::vector<double> src(3, kValue);
    for (std::size_t s = 0; s < t.n_src_steps; ++s) {
      const auto r = fuse::apply_spacetime(w.view(), t, src.data(), dst.data(),
                                           3, 1, s, tgt);
      ASSERT_TRUE(r.ok()) << r.error().message;
    }
    EXPECT_NEAR(dst[0], kValue, 1e-13) << "tgt=" << tgt;
    EXPECT_NEAR(dst[1], kValue, 1e-13) << "tgt=" << tgt;
  }
}

TEST(Spacetime, ApplySpacetimeChecksItsPreconditions) {
  const Case  c{Rational{1, 4}, Rational{1, 2}, Rational{1}};
  const auto  t = build(c, Basis{TemporalBasis::Constant, 0, "Constant"});
  TinySpatial w;
  std::vector<double> src(3, 1.0);
  std::vector<double> dst(2, 0.0);

  const auto bad_components = fuse::apply_spacetime(
      w.view(), t, src.data(), dst.data(), 3, 0, 0, 0);
  ASSERT_FALSE(bad_components.ok());
  EXPECT_EQ(bad_components.error().code, "E-ST-SHAPE");

  const auto bad_space = fuse::apply_spacetime(w.view(), t, src.data(),
                                               dst.data(), 4, 1, 0, 0);
  ASSERT_FALSE(bad_space.ok());
  EXPECT_EQ(bad_space.error().code, "E-ST-SHAPE");

  const auto bad_src_step = fuse::apply_spacetime(
      w.view(), t, src.data(), dst.data(), 3, 1, t.n_src_steps, 0);
  ASSERT_FALSE(bad_src_step.ok());
  EXPECT_EQ(bad_src_step.error().code, "E-ST-STEP");

  const auto bad_tgt_step = fuse::apply_spacetime(
      w.view(), t, src.data(), dst.data(), 3, 1, 0, t.n_tgt_steps);
  ASSERT_FALSE(bad_tgt_step.ok());
  EXPECT_EQ(bad_tgt_step.error().code, "E-ST-STEP");

  const auto null_src =
      fuse::apply_spacetime(w.view(), t, nullptr, dst.data(), 3, 1, 0, 0);
  ASSERT_FALSE(null_src.ok());
  EXPECT_EQ(null_src.error().code, "E-ST-NULL");

  const auto null_dst =
      fuse::apply_spacetime(w.view(), t, src.data(), nullptr, 3, 1, 0, 0);
  ASSERT_FALSE(null_dst.ok());
  EXPECT_EQ(null_dst.error().code, "E-ST-NULL");

  fuse::SpatialWeightsView null_trips = w.view();
  null_trips.val                      = nullptr;
  const auto bad_trips = fuse::apply_spacetime(null_trips, t, src.data(),
                                               dst.data(), 3, 1, 0, 0);
  ASSERT_FALSE(bad_trips.ok());
  EXPECT_EQ(bad_trips.error().code, "E-ST-NULL");

  // An out-of-range spatial index is caught before it is dereferenced. This is
  // the check that has to survive -O3 -DNDEBUG, which is why apply_spacetime
  // returns a Result instead of the brief's void.
  TinySpatial oob;
  oob.row[2]              = 5;  // n_rows is 2
  const auto bad_row = fuse::apply_spacetime(oob.view(), t, src.data(),
                                             dst.data(), 3, 1, 0, 0);
  ASSERT_FALSE(bad_row.ok());
  EXPECT_EQ(bad_row.error().code, "E-ST-SHAPE");
}
