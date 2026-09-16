// The measure-weighted norm, tested on the three things the plain Euclidean
// norm gets wrong. Each case shows the plain norm failing and the weighted one
// not, because a new norm is only worth having if it fixes something -- and
// showing the failure is how the test stays honest if someone later "fixes"
// the weighted one back into the plain one.

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include <fuse/base/units.hpp>
#include <fuse/num/residual_norm.hpp>

namespace {

using fuse::num::NormKind;
using fuse::num::ResidualField;

/// A uniform interface of `n` cells covering unit total measure, every cell
/// carrying the same residual: the same physical error at any resolution.
ResidualField uniform(std::vector<double>& r, std::vector<double>& w,
                      std::size_t n, double err) {
  r.assign(n, err);
  w.assign(n, 1.0 / static_cast<double>(n));
  return ResidualField{.residual = r.data(), .measure = w.data(),
                       .owned = nullptr, .count = n,
                       .units = fuse::units::kKelvin};
}

double value(NormKind k, const ResidualField& f) {
  const auto v = fuse::num::field_norm(k, f);
  EXPECT_TRUE(v.ok()) << (v.ok() ? "" : v.error().message);
  return v.ok() ? v.value().value : -1.0;
}

}  // namespace

TEST(ResidualNorm, DoesNotChangeWhenTheInterfaceIsRefined) {
  std::vector<double> r1, w1, r2, w2;
  const ResidualField coarse = uniform(r1, w1, 10, 0.25);
  const ResidualField fine   = uniform(r2, w2, 1000, 0.25);

  // What the plain norm does: the same physical error reads ten times larger
  // on a hundred times finer mesh, so a tolerance tuned on one mesh is a
  // different tolerance on the other.
  EXPECT_NEAR(value(NormKind::Euclidean, fine) /
                  value(NormKind::Euclidean, coarse),
              10.0, 1e-9);

  // What the weighted norm does.
  EXPECT_NEAR(value(NormKind::MeasureWeightedRms, coarse), 0.25, 1e-12);
  EXPECT_NEAR(value(NormKind::MeasureWeightedRms, fine), 0.25, 1e-12);
}

TEST(ResidualNorm, IsNotDominatedByASliverCell) {
  // One cell a million times smaller than the rest, holding a large error.
  // Physically it contributes almost nothing; to the plain norm it is the
  // whole residual.
  const std::size_t n = 101;
  std::vector<double> r(n, 0.0), w(n, 1.0 / 100.0);
  r[0] = 100.0;
  w[0] = 1e-8;

  const ResidualField f{.residual = r.data(), .measure = w.data(),
                        .owned = nullptr, .count = n,
                        .units = fuse::units::kKelvin};

  EXPECT_NEAR(value(NormKind::Euclidean, f), 100.0, 1e-9)
      << "the plain norm reports the sliver's error as the whole residual";

  // sqrt(1e-8 * 1e4 / (1e-8 + 1.0)) = 1e-2, to the sliver's share.
  EXPECT_LT(value(NormKind::MeasureWeightedRms, f), 0.011)
      << "the weighted norm reports the sliver in proportion to its size";
}

TEST(ResidualNorm, DoesNotDependOnHowTheInterfaceWasPartitioned) {
  // Six entities. Rank A owns 0-2 and mirrors 3; rank B owns 3-5 and mirrors
  // 2. Summing over local entries double-counts the mirrored ones, so the
  // norm would depend on the halo -- which is a property of the partition,
  // not of the solution.
  const std::vector<double> full_r{0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
  const std::vector<double> full_w(6, 1.0 / 6.0);

  std::vector<double> ra{0.1, 0.2, 0.3, 0.4}, wa(4, 1.0 / 6.0);
  std::vector<bool>   oa_v{true, true, true, false};
  std::vector<double> rb{0.3, 0.4, 0.5, 0.6}, wb(4, 1.0 / 6.0);
  std::vector<bool>   ob_v{false, true, true, true};
  // std::vector<bool> has no data(); the mask is a bool array by design.
  bool oa[4] = {true, true, true, false};
  bool ob[4] = {false, true, true, true};

  const ResidualField whole{.residual = full_r.data(), .measure = full_w.data(),
                            .owned = nullptr, .count = 6,
                            .units = fuse::units::kKelvin};
  const ResidualField a{.residual = ra.data(), .measure = wa.data(),
                        .owned = oa, .count = 4,
                        .units = fuse::units::kKelvin};
  const ResidualField b{.residual = rb.data(), .measure = wb.data(),
                        .owned = ob, .count = 4,
                        .units = fuse::units::kKelvin};

  const auto na = fuse::num::field_norm(NormKind::MeasureWeightedRms, a);
  const auto nb = fuse::num::field_norm(NormKind::MeasureWeightedRms, b);
  ASSERT_TRUE(na.ok());
  ASSERT_TRUE(nb.ok());
  const auto joined = fuse::num::combine(NormKind::MeasureWeightedRms,
                                         {na.value(), nb.value()});
  ASSERT_TRUE(joined.ok());

  EXPECT_NEAR(joined.value().value,
              value(NormKind::MeasureWeightedRms, whole), 1e-12)
      << "the two-rank norm differs from the single-rank norm of the same data";
  EXPECT_EQ(joined.value().owned_count, 6u)
      << "an entity was counted twice or not at all";
}

TEST(ResidualNorm, CarriesTheFieldsUnits) {
  std::vector<double> r, w;
  const ResidualField f = uniform(r, w, 8, 0.5);
  const auto v = fuse::num::field_norm(NormKind::MeasureWeightedRms, f);
  ASSERT_TRUE(v.ok());
  EXPECT_TRUE(v.value().units.same_dimensions(fuse::units::kKelvin))
      << "a norm with units is the point; " << v.value().units.to_string();
}

TEST(ResidualNorm, RefusesToAddUnlikeQuantities) {
  std::vector<double> r1, w1, r2, w2;
  ResidualField t = uniform(r1, w1, 4, 0.1);
  ResidualField q = uniform(r2, w2, 4, 0.1);
  q.units = fuse::units::kHeatFluxDensity;

  const auto nt = fuse::num::field_norm(NormKind::MeasureWeightedRms, t);
  const auto nq = fuse::num::field_norm(NormKind::MeasureWeightedRms, q);
  ASSERT_TRUE(nt.ok());
  ASSERT_TRUE(nq.ok());

  const auto bad = fuse::num::combine(NormKind::MeasureWeightedRms,
                                      {nt.value(), nq.value()});
  ASSERT_FALSE(bad.ok()) << "a temperature norm was added to a flux norm";
  EXPECT_EQ(bad.error().code, "E-NORM-UNITS");
}

TEST(ResidualNorm, ARankWithNoOwnedEntriesIsUndefinedNotZero) {
  // A zero here would read as "converged" inside a reduction over ranks.
  std::vector<double> r{1.0, 2.0}, w{0.5, 0.5};
  bool none[2] = {false, false};
  const ResidualField f{.residual = r.data(), .measure = w.data(),
                        .owned = none, .count = 2,
                        .units = fuse::units::kKelvin};
  const auto v = fuse::num::field_norm(NormKind::MeasureWeightedRms, f);
  ASSERT_TRUE(v.ok());
  EXPECT_EQ(v.value().owned_count, 0u);

  // Combining it with a real contribution must not dilute that contribution.
  std::vector<double> r2, w2;
  const ResidualField real = uniform(r2, w2, 4, 0.3);
  const auto nr = fuse::num::field_norm(NormKind::MeasureWeightedRms, real);
  ASSERT_TRUE(nr.ok());
  const auto joined = fuse::num::combine(NormKind::MeasureWeightedRms,
                                         {v.value(), nr.value()});
  ASSERT_TRUE(joined.ok());
  EXPECT_NEAR(joined.value().value, 0.3, 1e-12);
}

TEST(ResidualNorm, RefusesANonPositiveCellMeasure) {
  std::vector<double> r{1.0, 1.0}, w{1.0, 0.0};
  const ResidualField f{.residual = r.data(), .measure = w.data(),
                        .owned = nullptr, .count = 2,
                        .units = fuse::units::kKelvin};
  const auto v = fuse::num::field_norm(NormKind::MeasureWeightedRms, f);
  ASSERT_FALSE(v.ok()) << "a zero-measure cell was weighted by";
  EXPECT_EQ(v.error().code, "E-NORM-MEASURE");
}
