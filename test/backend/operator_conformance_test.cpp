// The fixture every LinearOperator instantiates.
//
// WHAT THIS IS FOR. The properties below were previously checked for the remap
// path and assumed for everything else. They are the properties a coupled run
// silently depends on: an adjoint that is not the adjoint gives the monolithic
// solver a wrong Jacobian block, a composition applied in the wrong order gives
// a plausible field, and a claim of conservation that the matrix does not
// satisfy is how mass disappears without any test failing.
//
// HOW IT IS ORGANISED. Each operator under test appears once in `operators()`;
// every case runs against all of them. Adding a backend means adding one entry
// there, which is the point -- a backend cannot arrive without being held to
// this.
//
// WHAT IT DOES NOT CHECK. Whether the weights are the RIGHT weights: that is
// the remap accuracy and conservation study, which needs meshes and an exact
// solution. This fixture checks the algebra, on the reasoning that an operator
// wrong in the algebra is wrong for every mesh, so failing here is cheaper than
// failing there.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/linear_operator.hpp"
#include "fuse/backend/weights_operator.hpp"

namespace {

using fuse::LinearOperator;
using fuse::RemapWeights;

/// A deliberately NON-SYMMETRIC 4x3 conservative-looking matrix. Non-symmetry
/// is load-bearing: a transpose bug and a composition-order bug are both
/// invisible on a symmetric operator, so a symmetric fixture would pass while
/// the library was wrong.
RemapWeights rectangular() {
  RemapWeights w;
  w.n_rows = 4;
  w.n_cols = 3;
  //          r  c   v
  const int rows[] = {0, 0, 1, 2, 2, 3};
  const int cols[] = {0, 1, 1, 0, 2, 2};
  const double val[] = {0.25, 0.75, 1.0, 0.4, 0.6, 1.0};
  for (int k = 0; k < 6; ++k) {
    w.row.push_back(rows[k]);
    w.col.push_back(cols[k]);
    w.val.push_back(val[k]);
  }
  // Row sums are all exactly 1 by construction above, so this matrix does
  // reproduce a constant; the column sums are not 1, so it is NOT conservative
  // and must not claim to be. The audit fields are what the claims read.
  w.max_row_defect = 0.0;
  w.max_col_defect = 0.35;  // column 0 sums to 0.65
  w.monotone = true;
  w.order = 1;
  return w;
}

/// A second, differently-shaped operator so composition has something to
/// compose with: 3x4, mapping back. Also non-symmetric.
RemapWeights transposed_shape() {
  RemapWeights w;
  w.n_rows = 3;
  w.n_cols = 4;
  const int rows[] = {0, 1, 1, 2, 2};
  const int cols[] = {0, 1, 2, 2, 3};
  const double val[] = {1.0, 0.5, 0.5, 0.3, 0.7};
  for (int k = 0; k < 5; ++k) {
    w.row.push_back(rows[k]);
    w.col.push_back(cols[k]);
    w.val.push_back(val[k]);
  }
  w.max_row_defect = 0.0;
  w.max_col_defect = 0.5;
  w.monotone = true;
  return w;
}

/// Columns sum to exactly 1, rows do not. This is the classic pair a remap can
/// satisfy one of: it conserves the integral but does not reproduce a constant.
/// It exists because without it NO operator in the fixture claimed
/// conservation, so the composition rule (`&&`, not `||`) was never exercised
/// -- a mutation that inherited the claim from either factor survived the whole
/// suite.
RemapWeights conservative() {
  RemapWeights w;
  w.n_rows = 4;
  w.n_cols = 3;
  //                    col 0 -> 0.5 + 0.5, col 1 -> 1.0, col 2 -> 0.7 + 0.3
  const int rows[] = {0, 1, 2, 0, 3};
  const int cols[] = {0, 0, 1, 2, 2};
  const double val[] = {0.5, 0.5, 1.0, 0.7, 0.3};
  for (int k = 0; k < 5; ++k) {
    w.row.push_back(rows[k]);
    w.col.push_back(cols[k]);
    w.val.push_back(val[k]);
  }
  w.max_col_defect = 0.0;   // conserves
  w.max_row_defect = 0.8;   // row 3 sums to 0.3, so it does NOT preserve constants
  w.monotone = false;
  w.order = 1;
  return w;
}

/// One entry per operator under test. Holds the weights alive alongside the
/// operator, because WeightsOperator deliberately does not copy them.
struct Case {
  std::string name;
  std::shared_ptr<RemapWeights> weights;
  std::shared_ptr<const LinearOperator> op;
};

std::vector<Case> operators() {
  std::vector<Case> cases;

  auto w1 = std::make_shared<RemapWeights>(rectangular());
  cases.push_back({"weights-4x3", w1, std::make_shared<fuse::WeightsOperator>(*w1)});

  auto w2 = std::make_shared<RemapWeights>(transposed_shape());
  cases.push_back({"weights-3x4", w2, std::make_shared<fuse::WeightsOperator>(*w2)});

  auto w3 = std::make_shared<RemapWeights>(conservative());
  cases.push_back({"weights-conservative-4x3", w3,
                   std::make_shared<fuse::WeightsOperator>(*w3)});

  // The composition, as a case in its own right: it is an operator, so it is
  // held to every property below rather than only to the composition case.
  auto composed = fuse::compose(cases[0].op, cases[1].op);
  if (composed.ok()) {
    std::shared_ptr<const LinearOperator> c{std::move(composed).value()};
    cases.push_back({"weights-4x3 o weights-3x4", nullptr, c});
  }
  // Conservative composed with non-conservative: the case the `&&` rule is
  // for. It must NOT claim conservation, and the claim check above then holds
  // it to that.
  auto mixed = fuse::compose(cases[2].op, cases[1].op);
  if (mixed.ok()) {
    std::shared_ptr<const LinearOperator> c{std::move(mixed).value()};
    cases.push_back({"weights-conservative-4x3 o weights-3x4", nullptr, c});
  }
  return cases;
}

std::vector<double> ramp(std::size_t n, double start) {
  std::vector<double> v(n);
  for (std::size_t i = 0; i < n; ++i) v[i] = start + 0.37 * static_cast<double>(i);
  return v;
}

}  // namespace

// ---- the adjoint identity ------------------------------------------------

TEST(OperatorConformance, TransposeIsTheAdjoint) {
  for (const Case& c : operators()) {
    for (int components : {1, 3}) {
      const auto nc = static_cast<std::size_t>(components);
      const std::vector<double> x = ramp(c.op->cols() * nc, 1.0);
      const std::vector<double> y = ramp(c.op->rows() * nc, -2.0);
      const auto defect = fuse::adjoint_defect(*c.op, x, y, components);
      ASSERT_TRUE(defect.ok()) << c.name << ": " << defect.error().message;
      // A few rounding steps, not zero: the two sides sum the same products in
      // different orders. The bound is generous relative to double precision
      // and still many orders below the error a genuine transpose bug makes.
      EXPECT_LT(defect.value(), 1e-13)
          << c.name << " with " << components << " components";
    }
  }
}

TEST(OperatorConformance, ADeliberatelyWrongAdjointIsDetected) {
  // The control: without this, the case above passes for an operator whose
  // apply and apply_transpose are both wrong in the same way, and we would not
  // know the measure can fail at all.
  RemapWeights w = rectangular();
  w.val[0] += 0.5;  // perturb one entry
  const fuse::WeightsOperator perturbed(w);
  const RemapWeights base = rectangular();
  const fuse::WeightsOperator original(base);

  const std::vector<double> x = ramp(3, 1.0);
  const std::vector<double> y = ramp(4, -2.0);

  // Mixing the two operators' apply and transpose is exactly the fault the
  // identity exists to catch, so build it by hand: A x from one, A^T y from the
  // other.
  std::vector<double> ax(4, 0.0), aty(3, 0.0);
  ASSERT_TRUE(original.apply(x, ax, 1).ok());
  ASSERT_TRUE(perturbed.apply_transpose(y, aty, 1).ok());
  const double lhs = std::inner_product(y.begin(), y.end(), ax.begin(), 0.0);
  const double rhs = std::inner_product(aty.begin(), aty.end(), x.begin(), 0.0);
  EXPECT_GT(std::abs(lhs - rhs), 1e-3)
      << "a perturbed transpose still satisfied the identity, so the identity "
         "is not testing what it claims";
}

// ---- sums agree with applying to ones ------------------------------------

TEST(OperatorConformance, RowAndColumnSumsAgreeWithApplyingToOnes) {
  for (const Case& c : operators()) {
    std::vector<double> rs(c.op->rows(), 0.0), cs(c.op->cols(), 0.0);
    ASSERT_TRUE(c.op->row_sums(rs).ok()) << c.name;
    ASSERT_TRUE(c.op->col_sums(cs).ok()) << c.name;

    const std::vector<double> ones_in(c.op->cols(), 1.0);
    const std::vector<double> ones_out(c.op->rows(), 1.0);
    std::vector<double> a_ones(c.op->rows(), 0.0), at_ones(c.op->cols(), 0.0);
    ASSERT_TRUE(c.op->apply(ones_in, a_ones, 1).ok()) << c.name;
    ASSERT_TRUE(c.op->apply_transpose(ones_out, at_ones, 1).ok()) << c.name;

    for (std::size_t i = 0; i < rs.size(); ++i) {
      EXPECT_NEAR(rs[i], a_ones[i], 1e-14) << c.name << " row " << i;
    }
    for (std::size_t j = 0; j < cs.size(); ++j) {
      EXPECT_NEAR(cs[j], at_ones[j], 1e-14) << c.name << " column " << j;
    }
  }
}

// ---- claims are held to ---------------------------------------------------

TEST(OperatorConformance, EveryClaimIsSatisfied) {
  for (const Case& c : operators()) {
    const fuse::OperatorClaims cl = c.op->claims();

    if (cl.preserves_constants) {
      std::vector<double> rs(c.op->rows(), 0.0);
      ASSERT_TRUE(c.op->row_sums(rs).ok()) << c.name;
      for (std::size_t i = 0; i < rs.size(); ++i) {
        EXPECT_NEAR(rs[i], 1.0, 1e-10)
            << c.name << " claims to preserve constants but row " << i
            << " sums to " << rs[i];
      }
    }
    if (cl.conservative) {
      std::vector<double> cs(c.op->cols(), 0.0);
      ASSERT_TRUE(c.op->col_sums(cs).ok()) << c.name;
      for (std::size_t j = 0; j < cs.size(); ++j) {
        EXPECT_NEAR(cs[j], 1.0, 1e-10)
            << c.name << " claims conservation but column " << j << " sums to "
            << cs[j];
      }
    }
    if (cl.monotone) {
      // No new extrema: a non-negative combination summing to at most 1 cannot
      // exceed the largest input. Checked on the weights, since that is what
      // the property is about.
      std::vector<double> rs(c.op->rows(), 0.0);
      ASSERT_TRUE(c.op->row_sums(rs).ok()) << c.name;
      const std::vector<double> x = ramp(c.op->cols(), 1.0);
      std::vector<double> y(c.op->rows(), 0.0);
      ASSERT_TRUE(c.op->apply(x, y, 1).ok()) << c.name;
      const double lo = *std::min_element(x.begin(), x.end());
      const double hi = *std::max_element(x.begin(), x.end());
      for (std::size_t i = 0; i < y.size(); ++i) {
        if (rs[i] == 0.0) continue;  // a row with no contributions
        EXPECT_GE(y[i] / rs[i], lo - 1e-12) << c.name << " row " << i;
        EXPECT_LE(y[i] / rs[i], hi + 1e-12) << c.name << " row " << i;
      }
    }
  }
}

TEST(OperatorConformance, AnUnsatisfiedClaimFails) {
  // The control for the case above: an operator whose audit says conservative
  // while its columns do not sum to 1 must be caught. Without this, the claim
  // checks pass trivially whenever nothing claims anything.
  RemapWeights w = rectangular();
  w.max_col_defect = 0.0;  // a false audit -- the columns still do not sum to 1
  const fuse::WeightsOperator lying(w);
  ASSERT_TRUE(lying.claims().conservative);
  std::vector<double> cs(lying.cols(), 0.0);
  ASSERT_TRUE(lying.col_sums(cs).ok());
  bool any_off = false;
  for (const double s : cs) any_off = any_off || std::abs(s - 1.0) > 1e-10;
  EXPECT_TRUE(any_off) << "the fixture's conservation check cannot fail";
}

// ---- composition ---------------------------------------------------------

TEST(OperatorConformance, ComposedApplyEqualsApplyingEachInTurn) {
  const auto cases = operators();
  const auto& outer = cases[0].op;  // 4x3
  const auto& inner = cases[1].op;  // 3x4
  auto composed = fuse::compose(outer, inner);
  ASSERT_TRUE(composed.ok()) << composed.error().message;
  const auto& ab = *composed.value();

  ASSERT_EQ(ab.rows(), outer->rows());
  ASSERT_EQ(ab.cols(), inner->cols());

  const std::vector<double> x = ramp(inner->cols(), 0.5);
  std::vector<double> mid(inner->rows(), 0.0), expect(outer->rows(), 0.0),
      got(ab.rows(), 0.0);
  ASSERT_TRUE(inner->apply(x, mid, 1).ok());
  ASSERT_TRUE(outer->apply(mid, expect, 1).ok());
  ASSERT_TRUE(ab.apply(x, got, 1).ok());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], expect[i], 1e-14) << "entry " << i;
  }
}

TEST(OperatorConformance, ComposedTransposeAppliesTheFactorsInReverse) {
  const auto cases = operators();
  auto composed = fuse::compose(cases[0].op, cases[1].op);
  ASSERT_TRUE(composed.ok());
  const auto& ab = *composed.value();

  // (A B)^T = B^T A^T. If the implementation applied them in the same order as
  // apply, this test fails and the adjoint identity above fails too -- both,
  // deliberately: one names the symptom, the other names the cause.
  const std::vector<double> y = ramp(ab.rows(), -1.0);
  std::vector<double> mid(cases[0].op->cols(), 0.0),
      expect(cases[1].op->cols(), 0.0), got(ab.cols(), 0.0);
  ASSERT_TRUE(cases[0].op->apply_transpose(y, mid, 1).ok());
  ASSERT_TRUE(cases[1].op->apply_transpose(mid, expect, 1).ok());
  ASSERT_TRUE(ab.apply_transpose(y, got, 1).ok());
  for (std::size_t i = 0; i < got.size(); ++i) {
    EXPECT_NEAR(got[i], expect[i], 1e-14) << "entry " << i;
  }
}

TEST(OperatorConformance, ComposingMismatchedShapesIsRefusedAtConstruction) {
  const auto cases = operators();
  // 4x3 after 4x3: the inner produces 4, the outer takes 3.
  auto bad = fuse::compose(cases[0].op, cases[0].op);
  ASSERT_FALSE(bad.ok()) << "a shape mismatch was accepted";
  EXPECT_EQ(bad.error().code, "E-OP-SHAPE");
}

TEST(OperatorConformance, AClaimSurvivesCompositionOnlyIfBothFactorsMakeIt) {
  const auto cases = operators();
  auto composed = fuse::compose(cases[0].op, cases[1].op);
  ASSERT_TRUE(composed.ok());
  const fuse::OperatorClaims c = composed.value()->claims();
  const fuse::OperatorClaims a = cases[0].op->claims();
  const fuse::OperatorClaims b = cases[1].op->claims();
  EXPECT_EQ(c.preserves_constants, a.preserves_constants && b.preserves_constants);
  EXPECT_EQ(c.conservative, a.conservative && b.conservative);
  EXPECT_EQ(c.monotone, a.monotone && b.monotone);
}

// ---- signature -----------------------------------------------------------

TEST(OperatorConformance, SignatureTracksWhatTheOperatorComputes) {
  const RemapWeights base = rectangular();
  const fuse::WeightsOperator a(base);
  const RemapWeights same = rectangular();
  const fuse::WeightsOperator b(same);
  EXPECT_EQ(a.signature(), b.signature())
      << "two operators built identically disagree, so a valid persisted map "
         "would be rejected on reload";

  RemapWeights changed = rectangular();
  changed.val[2] += 1e-15;  // the smallest change that alters the result
  const fuse::WeightsOperator c(changed);
  EXPECT_NE(a.signature(), c.signature())
      << "a changed weight left the signature alone, so a stale persisted map "
         "would be accepted";

  // The audit fields describe the matrix rather than determine what it
  // computes, so they must NOT move the signature -- otherwise a change in the
  // builder's diagnostics invalidates every map on disk.
  RemapWeights reaudited = rectangular();
  reaudited.min_src_coverage = 0.5;
  reaudited.max_col_defect = 0.99;
  reaudited.src_measure_total = 12.0;
  const fuse::WeightsOperator d(reaudited);
  EXPECT_EQ(a.signature(), d.signature())
      << "an audit field moved the signature";
}

TEST(OperatorConformance, CompositionSignatureDependsOnTheOrder) {
  const auto cases = operators();
  auto ab = fuse::compose(cases[0].op, cases[1].op);
  auto ba = fuse::compose(cases[1].op, cases[0].op);
  ASSERT_TRUE(ab.ok());
  ASSERT_TRUE(ba.ok());
  EXPECT_NE(ab.value()->signature(), ba.value()->signature())
      << "A o B and B o A hash the same, so a plan could reuse the wrong map";
}

// ---- shape refusals ------------------------------------------------------

TEST(OperatorConformance, WrongLengthsAreRefusedNotRead) {
  const RemapWeights base = rectangular();
  const fuse::WeightsOperator a(base);
  const std::vector<double> too_short(2, 1.0);
  std::vector<double> out(a.rows(), 0.0);
  const auto r = a.apply(too_short, out, 1);
  ASSERT_FALSE(r.ok()) << "a short input was read anyway";
  EXPECT_EQ(r.error().code, "E-OP-SHAPE");

  std::vector<double> bad_out(1, 0.0);
  const std::vector<double> ok_in(a.cols(), 1.0);
  EXPECT_FALSE(a.apply(ok_in, bad_out, 1).ok());
  EXPECT_FALSE(a.apply(ok_in, out, 0).ok()) << "zero components was accepted";
}

TEST(OperatorConformance, ComposingWithANonConservativeFactorDropsTheClaim) {
  // Named separately from the generic claim rule because this is the asymmetric
  // case: one factor conserves, the other does not, and the composition must
  // not. A rule written as "either factor" passes every other case in this
  // file.
  const auto cases = operators();
  const RemapWeights cw = conservative();
  const fuse::WeightsOperator cons(cw);
  ASSERT_TRUE(cons.claims().conservative) << "the conservative fixture is not conservative";
  ASSERT_FALSE(cases[1].op->claims().conservative);

  auto mixed = fuse::compose(cases[2].op, cases[1].op);
  ASSERT_TRUE(mixed.ok()) << mixed.error().message;
  EXPECT_FALSE(mixed.value()->claims().conservative)
      << "a composition inherited conservation from one factor";

  // And the other order, so the rule is not accidentally one-sided.
  auto other = fuse::compose(cases[1].op, cases[2].op);
  ASSERT_TRUE(other.ok()) << other.error().message;
  EXPECT_FALSE(other.value()->claims().conservative);
}
