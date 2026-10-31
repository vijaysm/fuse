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

#include "fuse/backend/consistent_interp.hpp"
#include "fuse/backend/linear_operator.hpp"
#include "fuse/backend/serial_backend.hpp"
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

  // The consistent-interpolation H on a genuinely
  // non-matching 1D pair (5 beam vertices, 8 fluid cells) -- built through
  // `SerialMeshBackend`'s real "generated:line:n" meshes, not a hand-typed
  // matrix, so a bug in the geometry code (axis choice, node ordering) would
  // show up here too, not only in `ConsistentInterp.*` below. Appended, not
  // inserted, so it does not renumber `cases[0..2]`, which
  // `ComposingWithANonConservativeFactorDropsTheClaim` indexes directly.
  {
    fuse::SerialMeshBackend backend;
    const bool loaded =
        backend.load({.name = "beam", .source = "generated:line:5"}).ok() &&
        backend.load({.name = "fluid", .source = "generated:line:8"}).ok();
    if (loaded) {
      const auto bid = backend.mesh_id("beam");
      const auto fid = backend.mesh_id("fluid");
      if (bid.ok() && fid.ok()) {
        const auto bv = backend.view(bid.value());
        const auto fv = backend.view(fid.value());
        if (bv.ok() && fv.ok()) {
          auto h = fuse::build_consistent_interp_weights(*bv.value(),
                                                          *fv.value(), 1);
          if (h.ok()) {
            auto w4 = std::make_shared<RemapWeights>(std::move(h).value());
            cases.push_back(
                {"consistent-interp-H(5-node-beam,8-cell-fluid)", w4,
                 std::make_shared<fuse::WeightsOperator>(*w4)});
          }
        }
      }
    }
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

TEST(OperatorConformance, AnHStyleColumnSumIsNotClaimedConservative) {
  // The consistent-interpolation (H) builders'
  // `max_col_defect` is an UNWEIGHTED column-sum residual (a nodal degree
  // of freedom has no measure to weight it by), not a conservation
  // residual -- their own headers say H never claims to be conservative in
  // that sense. Without `RemapWeights::col_defect_is_measure_weighted`,
  // `claims()` could not tell this matrix apart from a genuinely
  // conservative one: a coincident node-to-node H pair's column sum is
  // exactly 1 by accident of geometry (every source vertex is also a
  // target vertex, weight exactly 1, nothing else contributes), and that
  // alone would be enough to certify `conservative`.
  RemapWeights w = rectangular();
  // An exact identity-style column sum (every column sums to exactly 1) --
  // what AnUnsatisfiedClaimFails above calls "a false audit" when it is
  // NOT backed by col_defect_is_measure_weighted, but is exactly what an
  // H builder's own max_col_defect legitimately reads on a coincident
  // pair: a real, not fabricated, zero defect.
  w.max_col_defect = 0.0;
  w.col_defect_is_measure_weighted = false;  // what consistent_interp.hpp/general_interp.hpp now set
  const fuse::WeightsOperator h(w);
  EXPECT_FALSE(h.claims().conservative)
      << "H must never certify conservative through its own column-sum "
         "diagnostic, however close to 1 that sum happens to be";

  // The control: the SAME zero defect, but from a builder that DOES mean
  // it (col_defect_is_measure_weighted left at its default true) -- must
  // still claim conservative. Confirms the flag, not the defect value
  // alone, is what changed.
  w.col_defect_is_measure_weighted = true;
  const fuse::WeightsOperator fv(w);
  EXPECT_TRUE(fv.claims().conservative);
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

// ---- shape rejections ------------------------------------------------------

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

// `operators()` adds the consistent-interpolation H case conditionally (in
// the same style the composition cases above use), so a construction
// failure there would otherwise just shrink the fixture silently -- every
// other TEST in this file would keep passing on fewer cases, which is
// exactly the vacuous-pass testing.md warns about. This makes that failure
// loud.
TEST(OperatorConformance, ConsistentInterpCaseIsPresent) {
  const auto cases = operators();
  const bool present =
      std::any_of(cases.begin(), cases.end(), [](const Case& c) {
        return c.name.rfind("consistent-interp-H", 0) == 0;
      });
  EXPECT_TRUE(present) << "the consistent-interpolation H case was not "
                          "constructed (generated:line load or "
                          "build_consistent_interp_weights failed) -- every "
                          "other test in this file silently ran without it";
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

// ===========================================================================
// ConsistentInterp: the domain-specific properties `operators()` above does
// not check (per this file's own "WHAT IT DOES NOT CHECK" -- these need real
// meshes and closed-form solutions, not just algebra). Covers consistent H
// and conservative H^T on 1D interfaces. Uses `SerialMeshBackend`'s real
// "generated:line:n" meshes throughout, exactly as
// `test/verification/interface_transfer_test.cpp` does for the P0 path, so a
// geometry bug (axis choice, node ordering, degenerate-tolerance) shows up
// here on real mesh data, not only against a hand-typed matrix.
// ===========================================================================

namespace {

using fuse::InterfaceDomainPolicy;

/// A `SerialMeshBackend` loaded with one "generated:line:n" mesh, and the
/// resolved `MeshView*` -- the backend must outlive the view, so this bundles
/// them the same way `test/backend/supermesh_test.cpp`'s `Pair` does.
struct LineMesh {
  fuse::SerialMeshBackend backend;
  const fuse::MeshView*   view = nullptr;
};

[[nodiscard]] std::unique_ptr<LineMesh> line_mesh(std::size_t n) {
  auto m = std::make_unique<LineMesh>();
  if (!m->backend.load({.name = "m", .source = "generated:line:" + std::to_string(n)}).ok())
    return nullptr;
  const auto mid = m->backend.mesh_id("m");
  if (!mid.ok()) return nullptr;
  const auto view = m->backend.view(mid.value());
  if (!view.ok()) return nullptr;
  m->view = view.value();
  return m;
}

std::vector<double> apply_H(const RemapWeights& w, const std::vector<double>& f) {
  std::vector<double> g(w.n_rows, 0.0);
  const auto r = fuse::apply_weights(w, f.data(), g.data(), 1);
  if (!r.ok()) ADD_FAILURE() << r.error().message;
  return g;
}

}  // namespace

TEST(ConsistentInterp, ExactForALinearFieldAtFixedNonMatchingResolution) {
  auto beam = line_mesh(5);
  auto fluid = line_mesh(8);
  ASSERT_TRUE(beam && fluid);
  const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid->view, 1);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();
  ASSERT_EQ(H.n_rows, 8u);
  ASSERT_EQ(H.n_cols, 6u);
  EXPECT_LT(H.max_row_defect, 1e-13) << "H's rows must sum to 1 (partition of unity)";

  std::vector<double> v(6);
  for (std::size_t i = 0; i < 6; ++i) v[i] = 1.0 + 2.0 * (static_cast<double>(i) / 5.0);
  const std::vector<double> Hv = apply_H(H, v);
  for (std::size_t j = 0; j < 8; ++j) {
    const double x = (static_cast<double>(j) + 0.5) / 8.0;
    EXPECT_NEAR(Hv[j], 1.0 + 2.0 * x, 1e-12) << "fluid cell " << j;
  }
}

TEST(ConsistentInterp, ObservedOrderIsTwoUnderRefinement) {
  const auto f = [](double x) {
    const double two_pi = 6.283185307179586476925287;
    return std::sin(two_pi * x);
  };
  std::vector<double> hs, errs;
  for (const std::size_t nb : {16u, 32u, 64u, 128u}) {
    const std::size_t nf = nb * 11 / 7;  // deliberately non-matching
    auto beam = line_mesh(nb);
    auto fluid = line_mesh(nf);
    ASSERT_TRUE(beam && fluid);
    const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid->view, 1);
    ASSERT_TRUE(h.ok()) << h.error().message;
    std::vector<double> v(nb + 1);
    for (std::size_t i = 0; i <= nb; ++i)
      v[i] = f(static_cast<double>(i) / static_cast<double>(nb));
    const std::vector<double> Hv = apply_H(h.value(), v);
    double l2 = 0.0;
    for (std::size_t j = 0; j < nf; ++j) {
      const double x = (static_cast<double>(j) + 0.5) / static_cast<double>(nf);
      const double e = Hv[j] - f(x);
      l2 += e * e / static_cast<double>(nf);
    }
    hs.push_back(1.0 / static_cast<double>(nb));
    errs.push_back(std::sqrt(l2));
  }
  double sx = 0, sy = 0, sxx = 0, sxy = 0;
  const double n = static_cast<double>(hs.size());
  for (std::size_t k = 0; k < hs.size(); ++k) {
    const double x = std::log(hs[k]), y = std::log(errs[k]);
    sx += x; sy += y; sxx += x * x; sxy += x * y;
  }
  const double order = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  EXPECT_GT(order, 1.8) << "observed order " << order << ", finest error " << errs.back();
  EXPECT_LT(order, 2.3) << "suspiciously high for a piecewise-linear reconstruction";
}

TEST(ConsistentInterp, HAndItsTransposeAreExactAdjointsToRoundoff) {
  // The generic `TransposeIsTheAdjoint` case above already covers H through
  // `WeightsOperator`; this restates it directly against
  // `build_consistent_interp_weights`'s own output so the property is
  // checked without going through that wrapper too.
  auto beam = line_mesh(5);
  auto fluid = line_mesh(9);
  ASSERT_TRUE(beam && fluid);
  const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid->view, 1);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();

  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> x(H.n_cols), y(H.n_rows);
  for (double& e : x) e = dist(rng);
  for (double& e : y) e = dist(rng);
  const std::vector<double> Hx = apply_H(H, x);
  std::vector<double> Hty(H.n_cols, 0.0);
  ASSERT_TRUE(fuse::WeightsOperator(H).apply_transpose(y, Hty, 1).ok());

  double lhs = 0.0, rhs = 0.0;
  for (std::size_t i = 0; i < y.size(); ++i) lhs += y[i] * Hx[i];
  for (std::size_t i = 0; i < x.size(); ++i) rhs += Hty[i] * x[i];
  EXPECT_NEAR(lhs, rhs, 1e-12 * (std::abs(lhs) + std::abs(rhs) + 1.0))
      << "y.(Hx) = " << lhs << ", (H^T y).x = " << rhs;
}

TEST(ConsistentInterp, DomainEdgeMismatchRefusesByDefault) {
  // The fluid wall's cells extend to x=1; shrink the beam so it stops well
  // short, forcing the fluid's last cell centres to fall past the beam's
  // span.
  auto beam = line_mesh(3);
  auto fluid = line_mesh(8);
  ASSERT_TRUE(beam && fluid);
  // Reduce the beam's own extent by rescaling is not exposed by
  // SerialMeshBackend's generator, so instead evaluate the fluid's points
  // (dim 1, spanning [0,1]) against a beam of only 2 nodes covering
  // [0, 1/3] -- "generated:line:2" gives vertices at 0, 1/2, 1, which DOES
  // span [0,1] fully; a genuinely short domain needs a smaller line count
  // interpreted over a shorter conceptual span, which this generator does
  // not offer. So instead this rejection is exercised the other direction:
  // node-to-node onto a coarser beam evaluated at NODES beyond the fluid
  // wall's own node range is not representable either. The gap is real: see
  // open_questions for a MeshView test double that can present an arbitrary
  // vertex layout, which is what a genuine short-domain fixture needs.
  auto fluid_nodes = line_mesh(20);
  ASSERT_TRUE(fluid_nodes);
  const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid_nodes->view, 1,
                                                        InterfaceDomainPolicy::Refuse);
  // Both meshes span [0, 1] by construction (the generator's only span), so
  // no target point is genuinely out of range and this MUST succeed --
  // asserting that documents the generator's limit rather than papering over
  // it, and the E-INTERP-DOMAIN path itself is exercised directly below via
  // build_consistent_interp_weights' documented contract with a
  // manufactured MeshView.
  EXPECT_TRUE(h.ok()) << "both generated:line meshes span [0,1]; this case "
                         "is not actually out-of-domain -- see the comment "
                         "above";
}

// A minimal MeshView exposing an arbitrary 1D vertex layout, for the one
// property `generated:line:n` cannot exercise: a genuinely short domain.
namespace {
class ArbitraryLine final : public fuse::MeshView {
 public:
  explicit ArbitraryLine(std::vector<double> xs) : xs_(std::move(xs)) {}
  int dimension() const override { return 1; }
  std::size_t entity_count(int dim) const override {
    if (dim == 0) return xs_.size();
    if (dim == 1) return xs_.empty() ? 0 : xs_.size() - 1;
    return 0;
  }
  std::size_t owned_count(int dim) const override { return entity_count(dim); }
  bool is_owned(int, std::size_t) const override { return true; }
  fuse::Point centroid(int dim, std::size_t i) const override {
    if (dim == 0) return fuse::Point{xs_[i], 0, 0};
    return fuse::Point{0.5 * (xs_[i] + xs_[i + 1]), 0, 0};
  }
  double measure(int dim, std::size_t i) const override {
    if (dim == 0) throw std::out_of_range("no vertex measure");
    return xs_[i + 1] - xs_[i];
  }
  std::size_t global_id(int, std::size_t i) const override { return i; }
  std::size_t simplex_count(int dim, std::size_t) const override { return dim == 1 ? 1 : 0; }
  void simplex_points(int, std::size_t i, std::size_t, fuse::Point* out) const override {
    out[0] = fuse::Point{xs_[i], 0, 0};
    out[1] = fuse::Point{xs_[i + 1], 0, 0};
  }

 private:
  std::vector<double> xs_;
};
}  // namespace

TEST(ConsistentInterp, DomainEdgeMismatchRefusesAndClampAccepts) {
  // Beam nodes cover only [0, 0.4]; the fluid's cell centres run to 0.9375,
  // well past it.
  ArbitraryLine beam({0.0, 0.2, 0.4});
  auto fluid = line_mesh(8);
  ASSERT_TRUE(fluid);

  const auto refused =
      fuse::build_consistent_interp_weights(beam, *fluid->view, 1, InterfaceDomainPolicy::Refuse);
  ASSERT_FALSE(refused.ok());
  EXPECT_EQ(refused.error().code, "E-INTERP-DOMAIN");

  const auto clamped = fuse::build_consistent_interp_weights(
      beam, *fluid->view, 1, InterfaceDomainPolicy::ClampToBoundary);
  ASSERT_TRUE(clamped.ok()) << clamped.error().message;
  EXPECT_LT(clamped.value().max_row_defect, 1e-13)
      << "a clamped row is still a single weight of 1, so row-sum-one holds";
}

TEST(ConsistentInterp, TransposeConservesTotalLoadAndBalancesPower) {
  auto beam = line_mesh(5);
  auto fluid = line_mesh(8);
  ASSERT_TRUE(beam && fluid);
  const auto h = fuse::build_consistent_interp_weights(*beam->view, *fluid->view, 1);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();

  // H^T D by hand, exactly as build_consistent_weights_for_plan constructs
  // it: out(beam_dof, fluid_cell) = H(fluid_cell, beam_dof) * fluid_measure.
  RemapWeights HtD;
  HtD.n_rows = H.n_cols;
  HtD.n_cols = H.n_rows;
  for (std::size_t k = 0; k < H.val.size(); ++k) {
    HtD.row.push_back(H.col[k]);
    HtD.col.push_back(H.row[k]);
    HtD.val.push_back(H.val[k] * fluid->view->measure(1, static_cast<std::size_t>(H.row[k])));
  }

  std::mt19937 rng(7);
  std::uniform_real_distribution<double> dist(-3.0, 5.0);
  std::vector<double> rho(8);
  for (double& x : rho) x = dist(rng);
  const std::vector<double> F = apply_H(HtD, rho);

  double sent = 0.0;
  for (std::size_t j = 0; j < 8; ++j) sent += rho[j] * fluid->view->measure(1, j);
  double received = 0.0;
  for (double v : F) received += v;
  EXPECT_NEAR(sent, received, 1e-12 * std::abs(sent))
      << "total force must be conserved to round-off";

  std::vector<double> v(6);
  for (double& x : v) x = dist(rng);
  const std::vector<double> Hv = apply_H(H, v);
  double power_fluid = 0.0;
  for (std::size_t j = 0; j < 8; ++j) power_fluid += Hv[j] * rho[j] * fluid->view->measure(1, j);
  double power_beam = 0.0;
  for (std::size_t i = 0; i < 6; ++i) power_beam += v[i] * F[i];
  EXPECT_NEAR(power_fluid, power_beam, 1e-12 * (std::abs(power_fluid) + std::abs(power_beam)))
      << "discrete interface power must balance exactly: (Hv)^T(D rho) = v^T(H^T D rho)";
}

TEST(ConsistentInterp, CoincidentSourceNodesRefuseAsDegenerate) {
  // Two source vertices at the same x -- the piecewise-linear reconstruction
  // is singular there. `ArbitraryLine` lets this be constructed directly;
  // `generated:line:n` cannot produce a repeated node.
  ArbitraryLine beam({0.0, 0.2, 0.2, 0.6});  // nodes 1 and 2 coincide
  auto fluid = line_mesh(8);
  ASSERT_TRUE(fluid);
  const auto r = fuse::build_consistent_interp_weights(beam, *fluid->view, 1);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTERP-DEGENERATE");
}

TEST(ConsistentInterp, NoSourceGeometryRefuses) {
  ArbitraryLine empty_beam({});  // no vertices at all
  auto fluid = line_mesh(8);
  ASSERT_TRUE(fluid);
  const auto r = fuse::build_consistent_interp_weights(empty_beam, *fluid->view, 1);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTERP-NOGEOMETRY");
}

TEST(ConsistentInterp, NoTargetGeometryRefuses) {
  auto beam = line_mesh(5);
  ASSERT_TRUE(beam);
  ArbitraryLine empty_fluid({});
  const auto r = fuse::build_consistent_interp_weights(*beam->view, empty_fluid, 1);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-INTERP-NOGEOMETRY");
}

TEST(ConsistentInterp, NodeToNodeOntoTheSameMeshIsTheIdentity) {
  auto m = line_mesh(6);
  ASSERT_TRUE(m);
  const auto h = fuse::build_consistent_interp_weights(*m->view, *m->view, 0);
  ASSERT_TRUE(h.ok()) << h.error().message;
  const RemapWeights& H = h.value();
  EXPECT_EQ(H.nnz(), H.n_rows) << "one entry per row: no bracket, an exact hit each time";
  for (const double v : H.val) EXPECT_NEAR(v, 1.0, 1e-14);
}
