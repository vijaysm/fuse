#include <gtest/gtest.h>
#include <fuse/model/iteration.hpp>
#include <fuse/base/error.hpp>

#include <cmath>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

using fuse::ConvergenceCriterion;
using fuse::ConvergenceMeasure;
using fuse::MeasureKind;
namespace C = fuse::Converged;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

/// A locale whose decimal separator is a comma, so a report formatted through
/// the *global* locale is byte-distinguishable from one formatted through the
/// classic "C" locale. Built from a facet rather than a named locale
/// ("de_DE.UTF-8") because named locales are not installed everywhere.
struct CommaPunct : std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

class ScopedGlobalLocale {
 public:
  explicit ScopedGlobalLocale(const std::locale& loc)
      : saved_(std::locale::global(loc)) {}
  ~ScopedGlobalLocale() { std::locale::global(saved_); }
  ScopedGlobalLocale(const ScopedGlobalLocale&) = delete;
  ScopedGlobalLocale& operator=(const ScopedGlobalLocale&) = delete;

 private:
  std::locale saved_;
};

}  // namespace

TEST(ConvergenceMeasure, RelativeComparesAgainstTheReference) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-7, 1.0));
  EXPECT_FALSE(m.satisfied(1e-5, 1.0));
  EXPECT_TRUE(m.satisfied(1e-5, 1e3));  // relative to a large reference
}

TEST(ConvergenceMeasure, AbsoluteIgnoresTheReference) {
  const auto m = C::Absolute(1e-8, "flux");
  EXPECT_TRUE(m.satisfied(1e-9, 1e6));
  EXPECT_FALSE(m.satisfied(1e-7, 1e-6));
}

TEST(ConvergenceMeasure, AbsoluteOrRelativeAcceptsEither) {
  const auto m = C::AbsoluteOrRelative(1e-8, "T");
  EXPECT_TRUE(m.satisfied(1e-9, 1.0));   // absolute passes
  EXPECT_TRUE(m.satisfied(1e-6, 1e3));   // relative passes
  EXPECT_FALSE(m.satisfied(1e-6, 1.0));  // neither
}

// ResidualRelative runs the same comparison as Relative; the difference is the
// reference the caller supplies. Pinned so the enumerator is not silently
// unreachable or silently divergent from Relative.
TEST(ConvergenceMeasure, ResidualRelativeMatchesRelativeArithmetic) {
  const auto rr = C::ResidualRelative(1e-6, "T");
  const auto rel = C::Relative(1e-6, "T");
  EXPECT_EQ(rr.kind, MeasureKind::ResidualRelative);
  for (const double r : {1e-9, 1e-7, 1e-6, 1e-5, 1.0}) {
    for (const double ref : {0.0, 1e-3, 1.0, 1e3}) {
      EXPECT_EQ(rr.satisfied(r, ref), rel.satisfied(r, ref))
          << "r=" << r << " ref=" << ref;
    }
  }
}

// A zero reference must not produce NaN.
TEST(ConvergenceMeasure, ZeroReferenceFallsBackToAbsolute) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-9, 0.0));
  EXPECT_FALSE(m.satisfied(1.0, 0.0));
  // ... and the same fallback for the OR kind, so the two branches agree.
  const auto o = C::AbsoluteOrRelative(1e-6, "T");
  EXPECT_TRUE(o.satisfied(1e-9, 0.0));
  EXPECT_FALSE(o.satisfied(1.0, 0.0));
}

TEST(ConvergenceMeasure, ExactlyAtToleranceCounts) {
  const auto m = C::Absolute(1e-6, "T");
  EXPECT_TRUE(m.satisfied(1e-6, 1.0));
  // The relative branch is `residual/reference <= tolerance`, exact when the
  // quotient is exact.
  const auto r = C::Relative(1e-6, "T");
  EXPECT_TRUE(r.satisfied(2e-6, 2.0));
  // One ulp above: the quotient (an exact halving) is then one ulp above the
  // tolerance, so it must not count.
  EXPECT_FALSE(r.satisfied(std::nextafter(2e-6, 1.0), 2.0));
}

// A diverging iteration produces NaN norms. `!(residual > tol)` would be true
// for NaN and report the divergence as convergence; every branch is written so
// that it is false instead.
TEST(ConvergenceMeasure, NanResidualIsNeverConverged) {
  for (const auto kind : {MeasureKind::Relative, MeasureKind::Absolute,
                          MeasureKind::AbsoluteOrRelative,
                          MeasureKind::ResidualRelative}) {
    const ConvergenceMeasure m{kind, 1e-6, "T"};
    EXPECT_FALSE(m.satisfied(kNaN, 1.0)) << fuse::to_cstring(kind);
    EXPECT_FALSE(m.satisfied(kNaN, 0.0)) << fuse::to_cstring(kind);
    EXPECT_FALSE(m.satisfied(1e-9, kNaN)) << fuse::to_cstring(kind);
    EXPECT_FALSE(m.satisfied(kNaN, kNaN)) << fuse::to_cstring(kind);
  }
}

TEST(ConvergenceMeasure, InfiniteNormsAreNeverConverged) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_FALSE(m.satisfied(kInf, 1.0));
  // Without the up-front finiteness check this one passes: 1e-9/inf == 0.
  EXPECT_FALSE(m.satisfied(1e-9, kInf));
  EXPECT_FALSE(C::Absolute(1e-6, "T").satisfied(kInf, 1.0));
  EXPECT_FALSE(C::AbsoluteOrRelative(1e-6, "T").satisfied(1e-9, kInf));
}

// A NaN inside a criterion must not let the window advance either.
TEST(ConvergenceCriterion, NanInAnyMeasureBlocksConvergence) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9}, {1.0, 1.0}));
  EXPECT_FALSE(c.all_satisfied({1e-7, kNaN}, {1.0, 1.0}));
  EXPECT_FALSE(c.all_satisfied({kNaN, 1e-9}, {1.0, 1.0}));
}

// A norm cannot be negative; a negative one is a caller bug and must not be
// swallowed under -DNDEBUG, where an assert would vanish.
TEST(ConvergenceMeasure, NegativeNormThrows) {
  const auto m = C::Relative(1e-6, "T");
  EXPECT_THROW((void)m.satisfied(-1e-9, 1.0), fuse::InvalidArgument);
  EXPECT_THROW((void)m.satisfied(1e-9, -1.0), fuse::InvalidArgument);
}

// A non-positive tolerance can never be satisfied by a non-negative norm, so
// the iteration would spin to max_iterations without ever saying why.
TEST(ConvergenceMeasure, NonPositiveToleranceIsRejected) {
  EXPECT_THROW((void)C::Relative(0.0, "T"), fuse::InvalidArgument);
  EXPECT_THROW((void)C::Relative(-1e-6, "T"), fuse::InvalidArgument);
  EXPECT_THROW((void)C::Absolute(kNaN, "T"), fuse::InvalidArgument);
  EXPECT_THROW((void)C::AbsoluteOrRelative(kInf, "T"), fuse::InvalidArgument);
  EXPECT_THROW((void)C::ResidualRelative(0.0, "T"), fuse::InvalidArgument);
  // The struct is an aggregate, so brace-initialisation bypasses the factory;
  // satisfied() re-checks.
  const ConvergenceMeasure bad{MeasureKind::Relative, 0.0, "T"};
  EXPECT_THROW((void)bad.satisfied(0.0, 1.0), fuse::InvalidArgument);
}

// Spec §5.3: `Relative(...) & Absolute(...)`.
TEST(ConvergenceCriterion, ConjunctionRequiresEveryMeasure) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  EXPECT_EQ(c.size(), 2u);
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9}, {1.0, 1.0}));
  EXPECT_FALSE(c.all_satisfied({1e-7, 1e-6}, {1.0, 1.0}));  // second fails
  EXPECT_FALSE(c.all_satisfied({1e-5, 1e-9}, {1.0, 1.0}));  // first fails
}

TEST(ConvergenceCriterion, ChainsMoreThanTwoMeasures) {
  const ConvergenceCriterion c = C::Relative(1e-6, "a") &
                                 C::Absolute(1e-8, "b") &
                                 C::Relative(1e-4, "c");
  EXPECT_EQ(c.size(), 3u);
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9, 1e-5}, {1.0, 1.0, 1.0}));
  EXPECT_FALSE(c.all_satisfied({1e-7, 1e-9, 1e-3}, {1.0, 1.0, 1.0}));
  // `&` appends, so measure i keeps the authoring order.
  ASSERT_EQ(c.measures().size(), 3u);
  EXPECT_EQ(c.measures()[0].field, "a");
  EXPECT_EQ(c.measures()[1].field, "b");
  EXPECT_EQ(c.measures()[2].field, "c");
}

TEST(ConvergenceCriterion, EmptyCriterionIsSatisfied) {
  const ConvergenceCriterion c;
  EXPECT_TRUE(c.empty());
  EXPECT_TRUE(c.all_satisfied({}, {}));
  EXPECT_EQ(c.report({}, {}), "(no measures)");
}

TEST(ConvergenceCriterion, AddAppendsAMeasure) {
  ConvergenceCriterion c;
  c.add(C::Relative(1e-6, "T"));
  c.add(C::Absolute(1e-8, "flux"));
  EXPECT_FALSE(c.empty());
  EXPECT_EQ(c.size(), 2u);
  EXPECT_TRUE(c.all_satisfied({1e-7, 1e-9}, {1.0, 1.0}));
}

// A short residual vector must not be zero-padded: zeros read as converged,
// which is exactly the silent false-positive this class exists to prevent.
TEST(ConvergenceCriterion, MismatchedInputLengthThrows) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  EXPECT_THROW((void)c.all_satisfied({1e-5}, {1.0, 1.0}),
               fuse::InvalidArgument);
  EXPECT_THROW((void)c.all_satisfied({1e-5, 1e-5}, {1.0}),
               fuse::InvalidArgument);
  EXPECT_THROW((void)c.all_satisfied({}, {}), fuse::InvalidArgument);
  EXPECT_THROW((void)c.all_satisfied({1e-7, 1e-9, 1e-9}, {1.0, 1.0, 1.0}),
               fuse::InvalidArgument);
  EXPECT_THROW((void)c.report({1e-5}, {1.0, 1.0}), fuse::InvalidArgument);
}

TEST(ConvergenceCriterion, ReportNamesEveryFieldAndItsState) {
  const ConvergenceCriterion c =
      C::Relative(1e-6, "T") & C::Absolute(1e-8, "flux");
  const std::string s = c.report({1e-7, 1e-6}, {1.0, 1.0});
  EXPECT_NE(s.find("T"), std::string::npos);
  EXPECT_NE(s.find("flux"), std::string::npos);
  EXPECT_EQ(s,
            "T[Relative] 1e-07 vs tol 1e-06 [ok], "
            "flux[Absolute] 1e-06 vs tol 1e-08 [pending]");
}

// The relative report prints the ratio that the decision was made on, not the
// raw residual.
TEST(ConvergenceCriterion, ReportPrintsTheComparedQuantity) {
  ConvergenceCriterion c;
  c.add(C::Relative(1e-3, "T"));
  // residual 2, reference 1000 -> the ratio 0.002 is what is compared.
  EXPECT_EQ(c.report({2.0}, {1000.0}),
            "T[Relative] 0.002 vs tol 0.001 [pending]");
  // A zero reference falls back to absolute, so the residual itself is the
  // compared quantity.
  EXPECT_EQ(c.report({2.0}, {0.0}), "T[Relative] 2 vs tol 0.001 [pending]");
}

// std::to_string(double) and an un-imbued ostringstream follow the global
// locale; the report must not.
TEST(ConvergenceCriterion, ReportIsLocaleIndependent) {
  ConvergenceCriterion c;
  c.add(C::Absolute(0.25, "T"));
  const std::string expected = "T[Absolute] 0.5 vs tol 0.25 [pending]";
  EXPECT_EQ(c.report({0.5}, {1.0}), expected);
  {
    const ScopedGlobalLocale guard{
        std::locale(std::locale::classic(), new CommaPunct)};
    // Sanity: the swapped locale really does change default formatting.
    std::ostringstream probe;
    probe << 0.5;
    ASSERT_EQ(probe.str(), "0,5");
    EXPECT_EQ(c.report({0.5}, {1.0}), expected);
  }
  EXPECT_EQ(c.report({0.5}, {1.0}), expected);
}

TEST(ConvergenceMeasure, DefaultsAreUsableAndRelative) {
  const ConvergenceMeasure m;
  EXPECT_EQ(m.kind, MeasureKind::Relative);
  EXPECT_DOUBLE_EQ(m.tolerance, 1e-6);
  EXPECT_TRUE(m.field.empty());
  EXPECT_TRUE(m.satisfied(1e-7, 1.0));
}

TEST(ConvergenceMeasure, KindNames) {
  EXPECT_STREQ(fuse::to_cstring(MeasureKind::Relative), "Relative");
  EXPECT_STREQ(fuse::to_cstring(MeasureKind::Absolute), "Absolute");
  EXPECT_STREQ(fuse::to_cstring(MeasureKind::AbsoluteOrRelative),
               "AbsoluteOrRelative");
  EXPECT_STREQ(fuse::to_cstring(MeasureKind::ResidualRelative),
               "ResidualRelative");
}
