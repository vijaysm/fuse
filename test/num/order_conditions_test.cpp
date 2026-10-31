// S8 physics-kit, track "Integrators", Stage B Round 2, task item 1.
//
// A generic order-condition checker (fuse/num/order_conditions.hpp) over Butcher's own
// rooted trees, proved here against known published per-tree conditions before being
// trusted to grade a single tableau (the task's own instruction, "do not hand-check" --
// this file IS the hand-check, run once, of the generic machinery itself).

#include "fuse/num/order_conditions.hpp"

#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

// Same derivation style as butcher_tableau_test.cpp's own consistency_tolerance: the
// order-condition evaluation chains at most 5 nested products/sums of at most `stages`
// terms each (order-5 trees have at most 4 children, each contributing one
// `stages`-term inner sum) -- 1024 * stages^2 * eps is generous against that, MEASURED
// (with this bound removed) to be at least 3 orders of magnitude above every shipped
// classical tableau's own observed defect through its declared order (worst observed
// 1.11e-15, SDIRK543/SDIRK643, 6 stages) and at least 2 orders of magnitude below every
// genuine order-condition failure this file or butcher_tableau.hpp's own top comment
// measures (smallest genuine failure among the existing tableaus: KUTTA44's own
// next-order defect, 8.3e-3; SDIRK44's own finding, exactly 1/18).
double order_tolerance(int stages) {
  return 1024.0 * static_cast<double>(stages) * static_cast<double>(stages) *
         std::numeric_limits<double>::epsilon();
}

bool is_classical_family(ButcherFamily f) {
  switch (f) {
    case ButcherFamily::Erk:
    case ButcherFamily::ErkEmbedded:
    case ButcherFamily::Dirk:
    case ButcherFamily::DirkEmbedded:
    case ButcherFamily::Sdirk:
    case ButcherFamily::SdirkEmbedded:
    case ButcherFamily::FirkNonSingular:
    case ButcherFamily::FirkSingular:
      return true;
    case ButcherFamily::Rosenbrock:
    case ButcherFamily::RosenbrockEmbedded:
      return false;
  }
  return false;
}

constexpr int kMaxCatalogOrder = 5;

}  // namespace

// OEIS A000081 (rooted trees by node count): 1, 1, 2, 4, 9 for n = 1..5. A generator
// that is missing a tree, or double-generating one as two "different" catalog entries,
// is caught here before it ever grades a single tableau.
TEST(OrderConditions, CatalogSizesMatchOeisA000081) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const std::vector<std::size_t> expected{1, 1, 2, 4, 9};
  for (int n = 1; n <= kMaxCatalogOrder; ++n) {
    EXPECT_EQ(cat.of_order(n).size(), expected[static_cast<std::size_t>(n - 1)])
        << "order " << n;
  }
  std::size_t total = 0;
  for (int n = 1; n <= kMaxCatalogOrder; ++n) total += cat.of_order(n).size();
  EXPECT_EQ(total, 17u);
  EXPECT_EQ(cat.trees().size(), 17u);
}

// Butcher's density for the first eight trees (orders 1-4), pinned against the
// published table (Hairer, Norsett & Wanner, Solving ODEs I, Table II.2.1): 1, 2, 3, 6,
// 4, 8, 12, 24 -- independent of any tableau, gamma(t) depends only on the tree.
TEST(OrderConditions, DensityMatchesThePublishedTable) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  std::vector<double> gammas;
  for (int ord = 1; ord <= 4; ++ord) {
    for (int tid : cat.of_order(ord)) gammas.push_back(cat.trees()[static_cast<std::size_t>(tid)].gamma);
  }
  std::sort(gammas.begin(), gammas.end());
  const std::vector<double> expected{1, 2, 3, 4, 6, 8, 12, 24};
  ASSERT_EQ(gammas.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_DOUBLE_EQ(gammas[i], expected[i]);
}

// The catalog's own Phi recursion, proved against four HAND-derived literature
// formulas (Hairer, Norsett & Wanner, Solving ODEs I, Section II.2's own order-4
// condition list: b.c^3=1/4, sum b_i c_i (Ac)_i=1/8, sum b_i (Ac^2)_i=1/12,
// sum b_i (A(Ac))_i=1/24) on the classical 4-stage RK4 tableau, computed TWO
// independent ways and checked to agree -- this is what proves `stage_weights()`'s own
// per-child-independent-product recursion (its own doc comment explains why a
// single shared sum across children is the WRONG formula) is the right one, not merely
// self-consistent.
TEST(OrderConditions, PerChildProductMatchesTheLiteratureConditions) {
  const ButcherTableau t = kutta44_tableau();
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const auto phi = stage_weights(cat, t.A);

  // Hand formulas, directly from A/B/C (no tree machinery).
  const auto& A = t.A;
  const auto& B = t.B;
  const auto& C = t.C;
  const std::size_t s = B.size();
  double hand_c3 = 0.0, hand_cAc = 0.0, hand_Ac2 = 0.0, hand_AAc = 0.0;
  std::vector<double> Ac(s, 0.0), Ac2(s, 0.0), AAc(s, 0.0);
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) Ac[i] += A[i][j] * C[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) Ac2[i] += A[i][j] * C[j] * C[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    for (std::size_t j = 0; j < s; ++j) AAc[i] += A[i][j] * Ac[j];
  }
  for (std::size_t i = 0; i < s; ++i) {
    hand_c3 += B[i] * C[i] * C[i] * C[i];
    hand_cAc += B[i] * C[i] * Ac[i];
    hand_Ac2 += B[i] * Ac2[i];
    hand_AAc += B[i] * AAc[i];
  }

  // Tree-driven: the four order-4 trees are, by children multiset (see
  // order_conditions.hpp's own construction, catalog ids assigned in generation
  // order): three-leaf star [0,0,0], one-leaf-plus-order2-chain [0,1], one order-3
  // "bush" child [2], one order-3 "chain" child [3] -- identified here by sorted
  // children rather than hand-picked ids, so this test does not depend on the
  // catalog's own id-assignment order surviving unchanged.
  double tree_c3 = 0.0, tree_cAc = 0.0, tree_Ac2 = 0.0, tree_AAc = 0.0;
  for (int tid : cat.of_order(4)) {
    const RootedTree& tr = cat.trees()[static_cast<std::size_t>(tid)];
    double val = 0.0;
    for (std::size_t i = 0; i < s; ++i) val += B[i] * phi[static_cast<std::size_t>(tid)][i];
    if (tr.children.size() == 3) {
      tree_c3 = val;
    } else if (tr.children.size() == 2) {
      tree_cAc = val;
    } else {
      // single child: distinguish the order-3 "bush" grandchild (gamma 3) from the
      // order-3 "chain" grandchild (gamma 6) by the CHILD's own gamma, not its id.
      const double child_gamma = cat.trees()[static_cast<std::size_t>(tr.children[0])].gamma;
      if (child_gamma == 3.0) {
        tree_Ac2 = val;
      } else {
        tree_AAc = val;
      }
    }
  }
  constexpr double kTol = 1e-13;
  EXPECT_NEAR(tree_c3, hand_c3, kTol);
  EXPECT_NEAR(tree_cAc, hand_cAc, kTol);
  EXPECT_NEAR(tree_Ac2, hand_Ac2, kTol);
  EXPECT_NEAR(tree_AAc, hand_AAc, kTol);
  // And both agree with the published targets (RK4 is genuinely order 4).
  EXPECT_NEAR(hand_c3, 1.0 / 4.0, kTol);
  EXPECT_NEAR(hand_cAc, 1.0 / 8.0, kTol);
  EXPECT_NEAR(hand_Ac2, 1.0 / 12.0, kTol);
  EXPECT_NEAR(hand_AAc, 1.0 / 24.0, kTol);
}

// Every classical-family tableau (Erk/Dirk/Sdirk/Firk*) in all_karma_tableaus()
// satisfies every order condition through its own declared `order` to round-off, AND
// does not understate it: at least one condition at `order + 1` fails by well above
// round-off (skipped when `order + 1` exceeds the catalog's own max order, 5 --
// DOPRI45 is order 5 already). This is the task's own "prove every new tableau" in its
// general form, run here against the 15 EXISTING classical tableaus first.
TEST(OrderConditions, ClassicalFamiliesSatisfyExactlyTheirDeclaredOrder) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (!is_classical_family(t.family)) continue;
    const double tol = order_tolerance(t.stages);
    const double thru = max_defect_through_order(cat, t.A, t.B, std::min(t.order, kMaxCatalogOrder));
    EXPECT_LT(thru, tol) << t.name << ": defect through order " << t.order << " = " << thru;
    if (t.order < kMaxCatalogOrder) {
      const double next = max_defect_at_order(cat, t.A, t.B, t.order + 1);
      EXPECT_GT(next, 1000.0 * tol)
          << t.name << ": order " << (t.order + 1) << " condition should NOT hold "
          << "(method is declared order " << t.order << " exactly), defect = " << next;
    }
    if (t.has_embedded_estimate()) {
      const auto b_hat = embedded_weights(t);
      const double ethru = max_defect_through_order(
          cat, t.A, b_hat, std::min(*t.embedded_order, kMaxCatalogOrder));
      EXPECT_LT(ethru, tol) << t.name << ": embedded defect through order "
                            << *t.embedded_order << " = " << ethru;
    }
  }
}

// SDIRK44's own finding (butcher_tableau.hpp's own top comment, fifth finding): every
// condition through order 3 holds to round-off, but the order-4 condition
// `sum_i b_i c_i^3 = 1/4` fails by EXACTLY 1/18 -- pinned as an exact number (not just
// "fails") so a future accidental "fix" of the coefficients that happened to also
// satisfy b.c^3=1/4 would be caught by this test actually changing value, prompting a
// second look, rather than the test simply going on passing by the metadata already
// having been corrected to order 3.
TEST(OrderConditions, Sdirk44sDeclaredOrderIsCorrectedFromFourToThree) {
  const ButcherTableau t = sdirk44_tableau();
  EXPECT_EQ(t.order, 3);
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const double thru3 = max_defect_through_order(cat, t.A, t.B, 3);
  EXPECT_NEAR(thru3, 0.0, order_tolerance(t.stages));
  const double at4 = max_defect_at_order(cat, t.A, t.B, 4);
  EXPECT_NEAR(at4, 1.0 / 18.0, 1e-12)
      << "sdirk44.C's own order-4 defect must stay exactly 1/18, not merely nonzero, so "
         "a coefficient edit that happens to also satisfy b.c^3=1/4 is still caught";
}

// The Rosenbrock family's own scope limit (butcher_tableau.hpp's own top comment, fifth
// finding): `rosenbrock_classical_form()`'s (M, b) satisfies the classical order-2
// conditions for EVERY Rosenbrock tableau (the Jacobian-exactness equivalence holds
// that far unconditionally), but the higher-order classical conditions do NOT hold for
// any tableau declared order 3 or above -- this is not a defect, it is why task item 3
// asks for a MEASURED numerical order test on the Rosenbrock family rather than this
// checker; the test exists so a future change that silently started asserting the
// higher-order classical conditions for Rosenbrock (which would be mathematically
// unjustified, Hairer & Wanner Solving ODEs II IV.7) is caught by this test FAILING to
// fail, not by a missing assertion nobody noticed.
TEST(OrderConditions, RosenbrockOrderTwoConditionsHoldButHigherOrderClassicalConditionsDoNot) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_karma_tableaus()) {
    if (is_classical_family(t.family)) continue;
    const ClassicalForm cf = rosenbrock_classical_form(t);
    const double tol = order_tolerance(t.stages);
    const double thru2 = max_defect_through_order(cat, cf.M, cf.b, 2);
    EXPECT_LT(thru2, tol) << t.name << ": classical order-2 defect = " << thru2;
    if (t.order >= 3) {
      const double at_declared = max_defect_at_order(cat, cf.M, cf.b, t.order);
      EXPECT_GT(at_declared, 1000.0 * tol)
          << t.name << ": classical order-" << t.order << " condition should NOT hold "
          << "for a Rosenbrock tableau (Jacobian-exactness equivalence stops at order 2)";
    }
  }
}

// Round 2's own new tableaus (task items 2 and 4): each proved by this SAME generic
// checker before being wired into any integrator, per the task's own "every tableau
// you add must pass it... before it is used anywhere".
TEST(RoundTwoTableaus, EveryNewTableauSatisfiesExactlyItsDeclaredOrder) {
  RootedTreeCatalog cat(kMaxCatalogOrder);
  for (const ButcherTableau& t : all_round2_tableaus()) {
    const double tol = order_tolerance(t.stages);
    const double thru = max_defect_through_order(cat, t.A, t.B, t.order);
    EXPECT_LT(thru, tol) << t.name << ": defect through order " << t.order << " = " << thru;
    const double next = max_defect_at_order(cat, t.A, t.B, t.order + 1);
    EXPECT_GT(next, 1000.0 * tol)
        << t.name << ": order " << (t.order + 1) << " should NOT hold, defect = " << next;
    if (t.has_embedded_estimate()) {
      const auto b_hat = embedded_weights(t);
      const double ethru =
          max_defect_through_order(cat, t.A, b_hat, *t.embedded_order);
      EXPECT_LT(ethru, tol) << t.name << ": embedded defect through order "
                            << *t.embedded_order << " = " << ethru;
    }
  }
}

// SSPRK(10,4)'s own order-4 proof by name, since this is the one new tableau whose
// A/B were derived by hand (Shu-Osher substitution, butcher_tableau.hpp's own
// ssprk104_tableau() doc comment) rather than copied from a published Butcher table --
// a defect here would mean the substitution itself was wrong, not a transcription slip.
TEST(RoundTwoTableaus, Ssprk104SatisfiesOrderFourConditions) {
  const ButcherTableau t = ssprk104_tableau();
  RootedTreeCatalog cat(kMaxCatalogOrder);
  const double thru4 = max_defect_through_order(cat, t.A, t.B, 4);
  EXPECT_LT(thru4, order_tolerance(t.stages)) << "defect through order 4 = " << thru4;
  const double at5 = max_defect_at_order(cat, t.A, t.B, 5);
  EXPECT_GT(at5, 1e-6) << "order 5 should NOT hold for a declared order-4 method";
}

}  // namespace fuse::num
