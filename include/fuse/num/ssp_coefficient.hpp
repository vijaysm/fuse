#pragma once

#include <algorithm>
#include <limits>
#include <vector>

/// S8 physics-kit, track "Integrators", Stage B Round 2, task item 4, owner decision
/// (b): "the SSP coefficient is carried as tableau metadata... must be REPRODUCED by
/// the computation, not copied" from the paper. This header is that computation: a
/// Shu-Osher combination's own strong-stability-preserving coefficient (Shu & Osher,
/// "Efficient implementation of essentially non-oscillatory shock-capturing schemes",
/// J. Comput. Phys. 77 (1988) 439-471) is the largest r for which the combination is a
/// convex combination of forward-Euler steps of size h/r -- `min` over every nonzero-
/// beta term of `alpha/beta` -- and a multi-stage method's own coefficient is the
/// minimum over every stage's own combination (Gottlieb, Shu & Tadmor, "Strong
/// Stability-Preserving High-Order Time Discretization Methods", SIAM Review 43
/// (2001), Section 2's own composition argument: a chain is only as SSP as its most
/// restrictive single link). The task's own "the standard optimisation is unnecessary
/// for the methods listed" is why this takes a GIVEN Shu-Osher form (the one the
/// paper itself publishes, reproduced in `ssprk33_shu_osher()`/`ssprk104_shu_osher()`
/// below) rather than searching for the optimal one among every decomposition a given
/// Butcher tableau admits (a harder problem, Ketcheson's own `optimal_shu_osher.py`-
/// style search, out of this round's scope).
namespace fuse::num {

/// One term of a Shu-Osher combination: `alpha * point + beta * h * f(point)`, added
/// into a new combination's own sum. `point` indexes an EARLIER point (0 = y0, i =
/// stage i for i >= 1, by this header's own convention -- the same convention
/// `ssprk33_shu_osher()`/`ssprk104_shu_osher()` use when building one).
struct ShuOsherTerm {
  int point = 0;
  double alpha = 0.0;
  double beta = 0.0;
};
using ShuOsherCombination = std::vector<ShuOsherTerm>;

/// One combination's own SSP coefficient: `min` over every `beta > 0` term of
/// `alpha / beta`. A term with `beta == 0` is a pure convex-combination weight on an
/// already-computed point (no new right-hand-side evaluation), so it imposes no
/// stability restriction of its own and is skipped. A combination with no `beta > 0`
/// term at all returns `+infinity` (unconstrained) rather than a misleadingly small
/// number -- `ssprk33_shu_osher()`'s own first combination and `ssprk104_shu_osher()`'s
/// own "u1 = u0" combination are both this case.
[[nodiscard]] inline double ssp_coefficient_of_combination(const ShuOsherCombination& c) {
  double r = std::numeric_limits<double>::infinity();
  for (const ShuOsherTerm& term : c) {
    if (term.beta > 0.0) r = std::min(r, term.alpha / term.beta);
  }
  return r;
}

/// The whole method's own SSP coefficient: the minimum over every one of its own
/// combinations (every stage, plus a separate final combination when the method has
/// one -- `ssprk104_shu_osher()`'s own 11th entry).
[[nodiscard]] inline double ssp_coefficient_from_shu_osher(
    const std::vector<ShuOsherCombination>& combinations) {
  double r = std::numeric_limits<double>::infinity();
  for (const ShuOsherCombination& c : combinations) {
    r = std::min(r, ssp_coefficient_of_combination(c));
  }
  return r;
}

/// SSPRK(3,3)'s own Shu-Osher form (Shu & Osher 1988): u1 = y0 + h f(y0);
/// u2 = (3/4) y0 + (1/4) u1 + (1/4) h f(u1); u3 = (1/3) y0 + (2/3) u2 + (2/3) h f(u2) --
/// the SAME recursion `ssprk33_tableau()`'s own doc comment states, in the form the
/// SSP coefficient is actually computed from.
[[nodiscard]] inline std::vector<ShuOsherCombination> ssprk33_shu_osher() {
  return {
      ShuOsherCombination{{0, 1.0, 1.0}},
      ShuOsherCombination{{0, 0.75, 0.0}, {1, 0.25, 0.25}},
      ShuOsherCombination{{0, 1.0 / 3.0, 0.0}, {2, 2.0 / 3.0, 2.0 / 3.0}},
  };
}

/// SSPRK(10,4)'s own Shu-Osher form (Ketcheson 2008) -- 10 combinations for its own
/// stages (points 1-10) plus an 11th for the separate final combination, the SAME
/// recursion `ssprk104_tableau()`'s own doc comment states (that factory converts this
/// SAME data to classical Butcher form by direct substitution; this is the form the
/// SSP coefficient is computed from, not the converted one).
[[nodiscard]] inline std::vector<ShuOsherCombination> ssprk104_shu_osher() {
  constexpr double h6 = 1.0 / 6.0;
  std::vector<ShuOsherCombination> c;
  c.push_back(ShuOsherCombination{{0, 1.0, 0.0}});  // u1 = u0, no new RHS term
  for (int i = 2; i <= 5; ++i) c.push_back(ShuOsherCombination{{i - 1, 1.0, h6}});
  c.push_back(ShuOsherCombination{{0, 0.6, 0.0}, {5, 0.4, 0.4 * h6}});  // u6
  for (int i = 7; i <= 10; ++i) c.push_back(ShuOsherCombination{{i - 1, 1.0, h6}});
  c.push_back(ShuOsherCombination{  // the final combination (u_+)
      {0, 1.0 / 25.0, 0.0}, {5, 9.0 / 25.0, (9.0 / 25.0) * h6}, {10, 3.0 / 5.0, (3.0 / 5.0) * h6}});
  return c;
}

}  // namespace fuse::num
