// Numerical order-of-accuracy study for the temporal projection weights.
//
// The failure mode this exists to catch is quiet: a projection that is one
// order low still reproduces constants, still conserves, and still passes every
// algebraic test in spacetime_test.cpp -- it just makes the coupled answer
// converge at the wrong rate, which surfaces (if at all) as an unexplained
// discrepancy in Phase 6.5's convergence study. So every order claimed in
// spacetime.hpp's header comment is measured here, in the style of
// waveform_order_test.cpp.
//
// WHAT IS BEING REFINED, since that is the easy thing to get wrong. The
// operator maps source step MEANS to target step MEANS, so the exact answer is
// known in closed form for any f: the target step means of f itself. The error
// is therefore purely the reconstruction error of the piecewise degree-p fit,
// and the refinement is of BOTH time grids together, at a fixed non-nesting
// ratio (3n source steps against 5n target steps, which share no tick but 0
// and 1). Over a source step of length h the reconstruction error is
// O(h^(p+1)) and averaging over target steps does not cancel it, because the
// target steps cut across the source steps.
//
// The test function is exp(4t), for exactly the reasons waveform_order_test.cpp
// documents at length: every derivative is a positive multiple of the function,
// so no leading error term accidentally vanishes and every degree sits the same
// distance from its asymptotic range on one shared h ladder.

#include <gtest/gtest.h>
#include <soma/remap/spacetime.hpp>
#include <soma/schedule/slab.hpp>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using soma::Rational;
using soma::TemporalBasis;
using soma::TemporalProjection;

namespace {

/// Exact mean of exp(4t) over [a, b]. The antiderivative is closed form, so
/// the reference carries no quadrature error of its own.
double cell_mean(double a, double b) {
  return (std::exp(4.0 * b) - std::exp(4.0 * a)) / (4.0 * (b - a));
}

TemporalProjection build(std::int64_t n_src, std::int64_t n_tgt,
                         TemporalBasis basis, int degree) {
  const Rational src{1, n_src};
  const Rational dst{1, n_tgt};
  const auto     decomp = soma::decompose(src, dst, Rational{1});
  EXPECT_TRUE(decomp.ok()) << (decomp.ok() ? "" : decomp.error().message);
  auto p = soma::build_temporal_projection(decomp.value(), basis, degree, src,
                                           dst);
  EXPECT_TRUE(p.ok()) << (p.ok() ? "" : p.error().message);
  return p.value();
}

std::vector<double> project(const TemporalProjection& t,
                            const std::vector<double>& x) {
  std::vector<double> y(t.n_tgt_steps, 0.0);
  for (std::size_t k = 0; k < t.val.size(); ++k) {
    y[static_cast<std::size_t>(t.row[k])] +=
        t.val[k] * x[static_cast<std::size_t>(t.col[k])];
  }
  return y;
}

/// Max error of the projected target step means against the exact ones.
double projection_error(std::int64_t n, TemporalBasis basis, int degree) {
  const std::int64_t n_src = 3 * n;
  const std::int64_t n_tgt = 5 * n;
  const auto         t     = build(n_src, n_tgt, basis, degree);
  EXPECT_EQ(t.n_src_steps, static_cast<std::size_t>(n_src));
  EXPECT_EQ(t.n_tgt_steps, static_cast<std::size_t>(n_tgt));

  std::vector<double> x(t.n_src_steps);
  for (std::size_t j = 0; j < x.size(); ++j) {
    const double a = static_cast<double>(j) / static_cast<double>(n_src);
    const double b = static_cast<double>(j + 1) / static_cast<double>(n_src);
    x[j]           = cell_mean(a, b);
  }
  const std::vector<double> y = project(t, x);

  double worst = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) {
    const double a = static_cast<double>(i) / static_cast<double>(n_tgt);
    const double b = static_cast<double>(i + 1) / static_cast<double>(n_tgt);
    worst = std::max(worst, std::fabs(y[i] - cell_mean(a, b)));
  }
  return worst;
}

double order_of(double coarse, double fine) { return std::log2(coarse / fine); }

/// Tolerance on every measured order. Same reasoning as
/// waveform_order_test.cpp: the observed deviations are well under 0.15 and a
/// genuine loss of one order would be 1.0.
constexpr double kTol = 0.15;

void expect_order(const char* label, TemporalBasis basis, int degree,
                  double expected) {
  const std::int64_t ns[] = {8, 16, 32, 64};
  double             err[4];
  for (int i = 0; i < 4; ++i) {
    err[i] = projection_error(ns[i], basis, degree);
    ASSERT_GT(err[i], 0.0)
        << label << " n=" << ns[i]
        << ": exactly zero error means the study is not measuring anything";
  }
  std::string trace;
  for (int i = 1; i < 4; ++i) {
    trace += "  h_src=1/" + std::to_string(3 * ns[i - 1]) + "->1/" +
             std::to_string(3 * ns[i]) + "  err " + std::to_string(err[i - 1]) +
             " -> " + std::to_string(err[i]) + "  order " +
             std::to_string(order_of(err[i - 1], err[i])) + "\n";
  }
  const double observed = order_of(err[2], err[3]);
  testing::Test::RecordProperty(std::string(label) + "_order",
                                std::to_string(observed));
  EXPECT_NEAR(observed, expected, kTol)
      << label << " claims order " << expected << "\n"
      << trace;
  std::cout << "[ ORDER    ] " << label << " expected " << expected
            << ", observed " << observed << "\n"
            << trace;
}

}  // namespace

TEST(SpacetimeOrder, ConstantIsFirstOrder) {
  expect_order("Constant", TemporalBasis::Constant, 0, 1.0);
}

TEST(SpacetimeOrder, LinearIsSecondOrder) {
  expect_order("Linear", TemporalBasis::Linear, 1, 2.0);
}

TEST(SpacetimeOrder, DgOneMatchesLinear) {
  // Linear and DG(1) are the same conservative reconstruction; the header says
  // so, and this measures it rather than leaving it as a claim.
  for (const std::int64_t n : {2, 4, 8}) {
    const auto a = build(3 * n, 5 * n, TemporalBasis::Linear, 1);
    const auto b = build(3 * n, 5 * n, TemporalBasis::DG, 1);
    EXPECT_EQ(a.row, b.row) << "n=" << n;
    EXPECT_EQ(a.col, b.col) << "n=" << n;
    EXPECT_EQ(a.val, b.val) << "n=" << n;
  }
  expect_order("DG(1)", TemporalBasis::DG, 1, 2.0);
}

TEST(SpacetimeOrder, DgTwoIsThirdOrder) {
  expect_order("DG(2)", TemporalBasis::DG, 2, 3.0);
}

TEST(SpacetimeOrder, DgThreeIsFourthOrder) {
  expect_order("DG(3)", TemporalBasis::DG, 3, 4.0);
}

TEST(SpacetimeOrder, DgFourIsFifthOrder) {
  expect_order("DG(4)", TemporalBasis::DG, 4, 5.0);
}

// A degree-p reconstruction must reproduce the step means of a degree-p
// polynomial EXACTLY, not merely to O(h^(p+1)). This is the algebraic
// statement behind the orders above, and it fails loudly if a stencil is
// mis-centred or a moment is off by one.
TEST(SpacetimeOrder, DegreePPolynomialsAreReproducedExactly) {
  // Mean of t^q over [a, b] = (b^(q+1) - a^(q+1)) / ((q+1)(b - a)).
  const auto poly_mean = [](const std::vector<double>& c, double a, double b) {
    double sum = 0.0;
    double pa  = a;
    double pb  = b;
    for (std::size_t q = 0; q < c.size(); ++q) {
      sum += c[q] * (pb - pa) / static_cast<double>(q + 1);
      pa *= a;
      pb *= b;
    }
    return sum / (b - a);
  };

  for (int p = 0; p <= soma::kMaxTemporalDegree; ++p) {
    // Coefficients chosen so no term is negligible against the others.
    std::vector<double> c(static_cast<std::size_t>(p) + 1);
    for (std::size_t q = 0; q < c.size(); ++q) {
      c[q] = 1.0 + 0.5 * static_cast<double>(q);
    }
    const std::int64_t n_src = 7;
    const std::int64_t n_tgt = 11;
    const auto         t     = build(n_src, n_tgt, TemporalBasis::DG, p);
    ASSERT_EQ(t.effective_degree, p);

    std::vector<double> x(t.n_src_steps);
    for (std::size_t j = 0; j < x.size(); ++j) {
      x[j] = poly_mean(c, static_cast<double>(j) / static_cast<double>(n_src),
                       static_cast<double>(j + 1) / static_cast<double>(n_src));
    }
    const std::vector<double> y = project(t, x);

    double worst = 0.0;
    for (std::size_t i = 0; i < y.size(); ++i) {
      const double want =
          poly_mean(c, static_cast<double>(i) / static_cast<double>(n_tgt),
                    static_cast<double>(i + 1) / static_cast<double>(n_tgt));
      worst = std::max(worst, std::fabs(y[i] - want));
    }
    std::cout << "[ ORDER    ] DG(" << p
              << ") degree-p polynomial reproduction residual " << worst
              << "\n";
    testing::Test::RecordProperty(
        "dg" + std::to_string(p) + "_polynomial_residual",
        std::to_string(worst));
    EXPECT_LT(worst, 1e-12) << "DG(" << p << ") does not reproduce a degree-"
                            << p << " polynomial exactly";
  }
}

// The degree the caller asks for is not always the degree available: a source
// partition with fewer than p+1 steps cannot support a degree-p reconstruction
// from step means. It is clamped, reported, and still exact at the clamped
// degree -- not silently wrong, and not silently full order.
TEST(SpacetimeOrder, DegreeIsClampedToTheAvailableStepCount) {
  const auto t = build(2, 5, TemporalBasis::DG, 4);
  EXPECT_EQ(t.degree, 4);
  EXPECT_EQ(t.effective_degree, 1) << "2 source steps support at most degree 1";

  // Still reproduces a linear signal exactly at the clamped degree.
  std::vector<double> x(t.n_src_steps);
  for (std::size_t j = 0; j < x.size(); ++j) {
    x[j] = (static_cast<double>(j) + 0.5) / static_cast<double>(t.n_src_steps);
  }
  const std::vector<double> y = project(t, x);
  for (std::size_t i = 0; i < y.size(); ++i) {
    EXPECT_NEAR(y[i],
                (static_cast<double>(i) + 0.5) /
                    static_cast<double>(t.n_tgt_steps),
                1e-14);
  }
}
