// KNOWN_ISSUES.md D30: Apple clang (and GNU/Clang generally) fuse a*b+c
// into a single-rounded fused multiply-add BY DEFAULT AT EVERY
// OPTIMIZATION LEVEL, including -O0 -- confirmed directly
// (docs/dev/testing.md's "Floating-point contraction policy" has the
// instruction counts), contrary to an earlier, wrong account in this
// file and in D30 itself that claimed -O0 never contracts. This probe's
// own `contracted_expression` below (a single, unvectorized expression,
// not a reduction loop) is therefore sensitive to the flag in EVERY
// build type, Debug included -- hand-proved in both
// (build-evidence/mutations/fp-policy.toml), which is why this row
// carries no `requires` key in tools/mutate.py: unlike the loop-shaped
// comparisons in dist_vector_test.cpp/space_kernels_test.cpp (whose
// Debug/Release difference comes from the optimizer's vectorization or
// unrolling choices, not from the contraction default itself), this
// trivial expression's fusion is governed by -ffp-contract alone, which
// does not vary by optimization level.
#include <fuse/base/config.hpp>

#include <cmath>
#include <cstdio>

#include <gtest/gtest.h>

namespace {

// A triple for which std::fma(a, b, c) (one rounding) and the separately
// rounded (a*b) + c (two roundings) differ by exactly one ulp -- found by
// random search over clang++ -O2 (Apple clang 21.0.0) and recorded here as
// exact hexadecimal floating-point literals (each is an exact power-of-two
// scaled binary fraction, so there is no decimal-to-binary rounding between
// what is written here and what the compiler reads) rather than retyped
// decimal, per docs/dev/testing.md's standing rule on transcribing measured
// values.
constexpr double kA = 0x1.31d89b6e3ad3ap+0;   // 1.1947114128223135
constexpr double kB = 0x1.7d30ccd0c91cap+0;   // 1.4890258798342182
constexpr double kC = -0x1.7627f7d932e2bp+2;  // -5.8461894627649071

// Always ONE rounding, by construction, regardless of any compiler
// contraction policy: the oracle for what a fused multiply-add computes.
double fma_rounded(double a, double b, double c) { return std::fma(a, b, c); }

// Always TWO roundings: the `volatile` store materialises the rounded
// product before the addition is even compiled, so no `-ffp-contract`
// setting can fuse this into one operation. The oracle for what a*b+c
// computes WITHOUT contraction -- the bit pattern FUSE_FP_REPRODUCIBLE=1
// promises a Release build will match.
double two_step_rounded(double a, double b, double c) {
  volatile double t = a * b;
  return t + c;
}

// Ordinary, unannotated source: this expression's rounding IS the build's
// contraction policy -- single-rounded (fused) under the compiler's
// default, double-rounded under -ffp-contract=off. Everything this test
// checks is a comparison against this one function.
double contracted_expression(double a, double b, double c) { return a * b + c; }

}  // namespace

TEST(FpPolicy, ProbeTripleActuallyDistinguishesContraction) {
  // A precondition on the probe itself, ASSERTed rather than EXPECTed:
  // nothing below is informative if this does not hold (a different
  // compiler, or one that folds the multiply differently, could make
  // fma_rounded and two_step_rounded agree on this triple, in which case
  // the tests below would pass trivially and prove nothing about the
  // build's actual contraction policy).
  ASSERT_NE(fma_rounded(kA, kB, kC), two_step_rounded(kA, kB, kC));
}

#if FUSE_FP_REPRODUCIBLE
// FUSE_FP_REPRODUCIBLE == 1: -ffp-contract=off was requested for this
// build. Holds in EVERY build type -- and is hand-proved to FAIL in
// EVERY build type when the flag-applying call is removed (the mutation
// in build-evidence/mutations/fp-policy.toml), because this expression's
// contraction is governed by -ffp-contract alone, which defaults to `on`
// at -O0 exactly as it does at -O2/-O3 (docs/dev/testing.md's corrected
// account of D30).
TEST(FpPolicy, ContractionIsOffWhenReproducible) {
  EXPECT_EQ(contracted_expression(kA, kB, kC), two_step_rounded(kA, kB, kC));
}
#else
// FUSE_FP_REPRODUCIBLE == 0: the compiler's default contraction policy is
// in effect, and no equality claim is made about it -- the point of the
// option being OFF is that the build does not care whether this
// particular expression gets fused. The observed difference from the
// uncontracted evaluation is printed (not asserted) so a human reading
// test output can see whether this build actually exercised contraction.
TEST(FpPolicy, ContractionPolicyIsTheCompilersDefaultWhenNotReproducible) {
  const double contracted = contracted_expression(kA, kB, kC);
  const double uncontracted = two_step_rounded(kA, kB, kC);
  std::printf(
      "[ fp_policy ] FUSE_FP_REPRODUCIBLE=0: a*b+c = %.17g, two-step = %.17g, "
      "difference = %.3e\n",
      contracted, uncontracted, contracted - uncontracted);
  SUCCEED();
}
#endif
