#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "fuse/num/butcher_tableau.hpp"

/// S8 physics-kit, track "Integrators", Stage B Round 2, task item 1: "Order-condition
/// checker (rooted trees to order 5, Butcher's conditions) as a tier-0 test over all
/// tableaus, with the embedded weights checked to embedded_order. This is also how you
/// prove every new tableau."
///
/// Butcher's theorem (Hairer, Norsett & Wanner, Solving Ordinary Differential Equations
/// I, Theorem II.2.6 and Section II.2's own construction of the elementary weights):
/// a Runge-Kutta method with tableau (A, B) has order >= p IFF, for every rooted tree
/// t with r(t) <= p vertices,
///
///     sum_i B[i] * Phi_i(t) == 1 / gamma(t)
///
/// where Phi_i(*) = 1 for the single-vertex tree, Phi_i([t1,...,tm]) =
/// prod_k ( sum_j A[i][j] * Phi_j(t_k) ) for a tree formed by attaching children
/// t1,...,tm to a new root (EACH child's own independently-summed factor, multiplied
/// together -- see `stage_weights()`'s own doc comment for why a single sum shared
/// across every child is the wrong recursion), and gamma(*) = 1, gamma([t1,...,tm]) = r(t) *
/// prod_k gamma(t_k) (Butcher's own "density"). This file generates every
/// non-isomorphic rooted tree up to a given order ONCE (RootedTreeCatalog, the
/// standard combinations-with-repetition construction: a tree of order n is a root
/// plus an UNORDERED multiset of smaller trees whose orders sum to n-1 -- stored as a
/// non-decreasing sequence of catalog ids so two orderings of the same multiset never
/// generate two catalog entries) and evaluates the condition above directly from a
/// tableau's own A/B, rather than from any hand-derived per-order formula ("b1+b2=1",
/// "sum b_i c_i = 1/2", etc. -- the task's own instruction, "do not hand-check").
///
/// Order-5 trees: OEIS A000081 (rooted trees by node count) gives 1, 1, 2, 4, 9 for
/// n = 1..5 -- 17 trees total up to order 5, pinned by
/// OrderConditions.CatalogSizesMatchOeisA000081 (order_conditions_test.cpp) as a
/// mechanical check that the generator itself is not missing or double-counting any
/// tree before it is trusted to grade a single tableau.
///
/// APPLIES TO EVERY FAMILY, including Rosenbrock: `rosenbrock_classical_form()`
/// (butcher_tableau.hpp) recovers the (M, b) pair whose order conditions are these
/// SAME tree conditions with `A` replaced by `M` -- see that function's own doc comment
/// for the exactness assumption. `order_condition_report()` below takes the tableau
/// directly and makes that substitution itself, so a caller never needs to branch on
/// `family`.
namespace fuse::num {

/// One non-isomorphic rooted tree, identified by its position in a `RootedTreeCatalog`.
/// `children` holds the CATALOG IDS of its own children (sorted ascending, always
/// strictly less than this tree's own id, since nothing can be its own child or a
/// larger-order one) -- never a direct tree-to-tree pointer, so the whole catalog is
/// one flat, by-value, copyable vector.
struct RootedTree {
  int order = 0;
  std::vector<int> children;
  /// Butcher's density, gamma(t): gamma(leaf) = 1, gamma([t1..tm]) = order * prod(gamma(t_k)).
  double gamma = 1.0;
};

/// Builds every rooted tree of order 1..max_order exactly once. Order `n`'s own trees
/// are generated only after every tree of order < n already has a catalog id (the
/// outer loop processes n = 1, 2, ..., max_order in that order), which is what the
/// construction's own "child ids are always smaller" invariant needs.
class RootedTreeCatalog {
 public:
  explicit RootedTreeCatalog(int max_order) {
    by_order_.assign(static_cast<std::size_t>(max_order) + 1, {});
    for (int n = 1; n <= max_order; ++n) {
      std::vector<std::vector<int>> multisets;
      std::vector<int> prefix;
      gen_multisets(0, n - 1, prefix, multisets);
      for (auto& children : multisets) {
        double g = static_cast<double>(n);
        for (int cid : children) g *= trees_[static_cast<std::size_t>(cid)].gamma;
        const int id = static_cast<int>(trees_.size());
        trees_.push_back(RootedTree{n, children, g});
        by_order_[static_cast<std::size_t>(n)].push_back(id);
      }
    }
  }

  [[nodiscard]] const std::vector<RootedTree>& trees() const noexcept { return trees_; }
  [[nodiscard]] const std::vector<int>& of_order(int order) const {
    return by_order_[static_cast<std::size_t>(order)];
  }
  [[nodiscard]] int max_order() const noexcept {
    return static_cast<int>(by_order_.size()) - 1;
  }

 private:
  /// Every non-decreasing sequence of EXISTING catalog ids (repetition allowed --
  /// `id` again after itself is how the SAME smaller tree can appear as two different
  /// children, e.g. order 3's two-leaf tree) summing, by order, to `remaining`,
  /// starting no earlier than `start` -- the "non-decreasing" restriction is exactly
  /// what collapses every permutation of one multiset into a single generated sequence.
  void gen_multisets(int start, int remaining, std::vector<int>& prefix,
                      std::vector<std::vector<int>>& out) const {
    if (remaining == 0) {
      out.push_back(prefix);
      return;
    }
    for (int id = start; id < static_cast<int>(trees_.size()); ++id) {
      const int w = trees_[static_cast<std::size_t>(id)].order;
      if (w > remaining) continue;
      prefix.push_back(id);
      gen_multisets(id, remaining - w, prefix, out);
      prefix.pop_back();
    }
  }

  std::vector<RootedTree> trees_;
  std::vector<std::vector<int>> by_order_;
};

/// Phi_i(t) for every stage i and every tree in the catalog, computed bottom-up: a
/// child's own id is always smaller than its parent's (RootedTreeCatalog's own
/// construction), so iterating ids in increasing order already visits every child
/// before the tree that uses it.
///
/// THE RECURSION, exactly (Hairer, Norsett & Wanner, Solving ODEs I, Theorem II.2.6):
/// for t = [t1, ..., tm] (a new root joining m child subtrees), EACH child
/// contributes its OWN independently-summed factor, and the tree's own Phi_i is the
/// PRODUCT of those factors -- not one sum shared across every child multiplying a
/// product of the children's own Phi_j. Concretely,
///
///     Phi_i(t) = PRODUCT_{k=1}^{m} ( SUM_j  m[i][j] * Phi_j(t_k) )
///
/// Written any other way (one shared j across all children) collapses every leaf
/// child to the SAME factor (Phi_j(leaf) == 1 for every j, so a sum over j of m[i][j]
/// times a product of 1's is just the row sum, independent of how many leaf children
/// there are) -- which cannot be the right formula, since it would make the two-leaf
/// ("bushy") order-3 tree's own condition identical in FORM to the single-child
/// order-2 tree's (both "sum_i b_i * (row sum)_i"), with only `gamma` differing,
/// which no consistent method of order >= 3 could satisfy for both trees
/// simultaneously (order_conditions_test.cpp's own
/// OrderConditions.PerChildProductMatchesTheLiteratureConditions pins the per-tree
/// expressions below against the published ones, e.g. `sum b_i c_i^3 = 1/4` for the
/// order-4 "star" with three leaf children, EXACTLY what the product form gives:
/// `Phi_i(star) = (sum_j m[i][j]*1)^3 = c_i^3`).
[[nodiscard]] inline std::vector<std::vector<double>> stage_weights(
    const RootedTreeCatalog& catalog, const std::vector<std::vector<double>>& m) {
  const std::size_t n_stages = m.size();
  const auto& trees = catalog.trees();
  std::vector<std::vector<double>> phi(trees.size(), std::vector<double>(n_stages, 0.0));
  for (std::size_t tid = 0; tid < trees.size(); ++tid) {
    const RootedTree& t = trees[tid];
    if (t.children.empty()) {
      for (std::size_t i = 0; i < n_stages; ++i) phi[tid][i] = 1.0;
      continue;
    }
    for (std::size_t i = 0; i < n_stages; ++i) {
      double prod = 1.0;
      for (int cid : t.children) {
        double s = 0.0;
        for (std::size_t j = 0; j < n_stages; ++j) {
          s += m[i][j] * phi[static_cast<std::size_t>(cid)][j];
        }
        prod *= s;
      }
      phi[tid][i] = prod;
    }
  }
  return phi;
}

/// `|sum_i b[i]*Phi_i(t) - 1/gamma(t)|` for one tree, given its own precomputed
/// Phi row (`phi[tree_id]`, from `stage_weights`).
[[nodiscard]] inline double order_condition_defect(const RootedTree& t,
                                                    const std::vector<double>& phi_row,
                                                    const std::vector<double>& b) {
  double val = 0.0;
  for (std::size_t i = 0; i < b.size(); ++i) val += b[i] * phi_row[i];
  const double target = 1.0 / t.gamma;
  const double d = val - target;
  return d < 0.0 ? -d : d;
}

/// The worst (maximum) order-condition defect over every tree of EXACTLY `order`
/// vertices -- used both to confirm a declared order holds (every tree up to that
/// order should be near round-off) and to confirm it is not UNDERSTATED (the task's
/// own "prove every new tableau": a method declared order p must have at least one
/// order-(p+1) tree whose condition fails by construction, since "order p" means the
/// largest p for which every condition up to p holds -- OrderConditions::
/// NoShippedTableauIsCreditedBelowItsActualOrder, order_conditions_test.cpp, checks
/// exactly this rather than only the positive direction).
[[nodiscard]] inline double max_defect_at_order(const RootedTreeCatalog& catalog,
                                                const std::vector<std::vector<double>>& m,
                                                const std::vector<double>& b, int order) {
  const auto phi = stage_weights(catalog, m);
  double worst = 0.0;
  for (int tid : catalog.of_order(order)) {
    worst = std::max(worst, order_condition_defect(catalog.trees()[static_cast<std::size_t>(tid)],
                                                    phi[static_cast<std::size_t>(tid)], b));
  }
  return worst;
}

/// The worst defect over every tree with 1 <= r(t) <= up_to_order vertices --
/// `max_defect_at_order`'s own cumulative form, for "does this method satisfy every
/// condition through its declared order" in one call.
[[nodiscard]] inline double max_defect_through_order(const RootedTreeCatalog& catalog,
                                                      const std::vector<std::vector<double>>& m,
                                                      const std::vector<double>& b,
                                                      int up_to_order) {
  double worst = 0.0;
  for (int ord = 1; ord <= up_to_order; ++ord) {
    worst = std::max(worst, max_defect_at_order(catalog, m, b, ord));
  }
  return worst;
}

/// `B - E`, the embedded pair's own weights (`ButcherTableau::E` stores `B - B_hat`,
/// this header's own top comment and butcher_tableau.hpp's doc comment on `E` agree).
[[nodiscard]] inline std::vector<double> embedded_weights(const ButcherTableau& t) {
  std::vector<double> b_hat(t.E->size());
  for (std::size_t i = 0; i < b_hat.size(); ++i) b_hat[i] = t.B[i] - (*t.E)[i];
  return b_hat;
}

}  // namespace fuse::num
