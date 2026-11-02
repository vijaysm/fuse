// fuse::num::dual and the view of a field as scalars: closed-form derivatives, independence of the number of parameters
// carried, a field of dual numbers crossing the registry as scalars, a
// gradient through the remap kernel equal to the remap of the gradient (bitwise),
// and the Taylor-remainder order of a residual whose derivative action is
// produced by dual numbers through that kernel.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/linalg/derivative_gate.hpp"
#include "fuse/num/dual.hpp"
#include "fuse/num/kernels.hpp"
#include "fuse/num/space_kernels.hpp"
#include "fuse_test_main.hpp"

namespace {

using D1 = fuse::num::dual<double, 1>;
using D2 = fuse::num::dual<double, 2>;
using D3 = fuse::num::dual<double, 3>;
using DF = fuse::num::dual<float, 3>;

static_assert(fuse::CoupledValue<D3>);
static_assert(fuse::CoupledValue<DF>);
static_assert(sizeof(D3) == 4 * sizeof(double));
static_assert(std::is_trivially_copyable_v<D3>);
static_assert(std::is_standard_layout_v<D3>);

// f(x, y) = sin(x y) + exp(x) / y - sqrt(x^2 + y^2) + log(y): every operator and
// every elementary function the type has, except abs and pow (tested apart).
template <class D>
KOKKOS_INLINE_FUNCTION D composite(const D& x, const D& y) {
  return fuse::num::sin(x * y) + fuse::num::exp(x) / y -
         fuse::num::sqrt(x * x + y * y) + fuse::num::log(y);
}

struct Analytic {
  double f, fx, fy;
};
Analytic analytic(double x, double y) {
  const double r = std::sqrt(x * x + y * y);
  return {std::sin(x * y) + std::exp(x) / y - r + std::log(y),
          y * std::cos(x * y) + std::exp(x) / y - x / r,
          x * std::cos(x * y) - std::exp(x) / (y * y) - y / r + 1.0 / y};
}

double tol_for(double reference) {
  return 1e-12 * std::max(1.0, std::fabs(reference));
}

// ------------------------------------------------------------ the number type

TEST(Dual, GradientsOfTheCompositeMatchTheClosedForm) {
  for (double x : {0.3, 0.9, 1.7, 2.4}) {
    for (double y : {0.8, 1.3, 2.9}) {
      const D2 f = composite(D2::variable(x, 0), D2::variable(y, 1));
      const Analytic a = analytic(x, y);
      EXPECT_NEAR(f.value(), a.f, tol_for(a.f)) << x << "," << y;
      EXPECT_NEAR(f.gradient(0), a.fx, tol_for(a.fx)) << x << "," << y;
      EXPECT_NEAR(f.gradient(1), a.fy, tol_for(a.fy)) << x << "," << y;
    }
  }
}

TEST(Dual, GradientsAgreeWithACentralDifferenceOfTheValue) {
  // An independent check that does not use the formulas above: each gradient
  // entry is the partial derivative of the value, whatever the value function is.
  const double x = 1.1, y = 1.9, h = 1e-6;
  const auto val = [](double a, double b) {
    return composite(D1(a), D1(b)).value();
  };
  const D2 f = composite(D2::variable(x, 0), D2::variable(y, 1));
  EXPECT_NEAR(f.gradient(0), (val(x + h, y) - val(x - h, y)) / (2 * h), 1e-8);
  EXPECT_NEAR(f.gradient(1), (val(x, y + h) - val(x, y - h)) / (2 * h), 1e-8);
}

TEST(Dual, AbsAndPowFollowTheBranchTheValueLiesOn) {
  const D1 neg = fuse::num::abs(D1::variable(-2.0, 0));
  EXPECT_EQ(neg.value(), 2.0);
  EXPECT_EQ(neg.gradient(0), -1.0);
  const D1 pos = fuse::num::abs(D1::variable(2.0, 0));
  EXPECT_EQ(pos.gradient(0), 1.0);
  const D1 p = fuse::num::pow(D1::variable(3.0, 0), 4.0);
  EXPECT_NEAR(p.value(), 81.0, 1e-12);
  EXPECT_NEAR(p.gradient(0), 4.0 * 27.0, 1e-10);
}

TEST(Dual, SinAndCosAreDifferentiatedByTheirClosedForms) {
  // The composite above uses sin but not cos; this is the only place the sign
  // of cos's derivative is pinned.
  for (double x : {0.2, 0.9, 2.3, -1.4}) {
    const D1 c = fuse::num::cos(D1::variable(x, 0));
    EXPECT_NEAR(c.value(), std::cos(x), 1e-15) << x;
    EXPECT_NEAR(c.gradient(0), -std::sin(x), 1e-15) << x;
    const D1 s = fuse::num::sin(D1::variable(x, 0));
    EXPECT_NEAR(s.value(), std::sin(x), 1e-15) << x;
    EXPECT_NEAR(s.gradient(0), std::cos(x), 1e-15) << x;
  }
}

TEST(Dual, ScalarOperandsAreConstantsAndConvertImplicitly) {
  const D1 x = D1::variable(2.0, 0);
  const D1 r = 3 * x - 1 + x / 4 + (10 - x) + (6 / x);
  // 3x - 1 + x/4 + 10 - x + 6/x at x = 2, derivative 3 + 1/4 - 1 - 6/4.
  EXPECT_NEAR(r.value(), 6.0 - 1.0 + 0.5 + 8.0 + 3.0, 1e-14);
  EXPECT_NEAR(r.gradient(0), 3.0 + 0.25 - 1.0 - 1.5, 1e-14);
  EXPECT_TRUE(D1(2.0) == x);   // ordered and compared by value only
  EXPECT_TRUE(D1(1.0) < x);
}

TEST(Dual, EachGradientIsIndependentOfHowManyDirectionsAreCarried) {
  const double x = 1.4, y = 2.2;
  const D3 both = composite(D3::variable(x, 0), D3::variable(y, 2));
  const D1 only_x = composite(D1::variable(x, 0), D1(y));
  const D1 only_y = composite(D1(x), D1::variable(y, 0));
  EXPECT_EQ(both.value(), only_x.value());
  EXPECT_EQ(both.gradient(0), only_x.gradient(0));
  EXPECT_EQ(both.gradient(2), only_y.gradient(0));
  EXPECT_EQ(both.gradient(1), 0.0);   // no parameter was seeded in direction 1
}

TEST(DualFieldValue, TraitsDescribeFourBinary64ScalarsOfWhichThreeAreGradients) {
  constexpr auto t = fuse::field_value_type_of<D3>();
  EXPECT_EQ(t.name, "fuse:dual<binary64,3>");
  EXPECT_EQ(t.scalar_name, "binary64");
  EXPECT_EQ(t.scalar_size, sizeof(double));
  EXPECT_EQ(t.size, 4 * sizeof(double));
  EXPECT_EQ(t.scalar_count, 4);
  EXPECT_EQ(t.gradient_size, 3);
  EXPECT_EQ(t.kind, fuse::FieldValueKind::ScalarArray);
  EXPECT_NE(t, fuse::field_value_type_of<D2>());
  EXPECT_NE(t, fuse::field_value_type_of<DF>());
  EXPECT_EQ(fuse::field_value_type_of<DF>().name, "fuse:dual<binary32,3>");
  using D12 = fuse::num::dual<double, 12>;
  EXPECT_EQ(fuse::field_value_type_of<D12>().name, "fuse:dual<binary64,12>");
  EXPECT_EQ(fuse::value_traits<D3>::to_binary64(D3::variable(2.5, 1)), 2.5);
  EXPECT_EQ(fuse::value_traits<D3>::from_binary64(2.5).gradient(1), 0.0);
}

// ------------------------------------------------- a field viewed as scalars

fuse::Layout interleaved(std::size_t count, int components) {
  return fuse::Layout{static_cast<std::size_t>(components), 1, count,
                      components};
}

D3 sample_value(std::size_t i, int c) {
  D3 v;
  v.d[0] = 100.0 * static_cast<double>(i) + 10.0 * c + 0.25;
  for (int k = 1; k <= 3; ++k) {
    v.d[k] = 0.5 * k - static_cast<double>(i) * 0.125 + c;
  }
  return v;
}

TEST(ScalarView, ADualFieldIsItsScalarsComponentsFollowingEachOther) {
  constexpr std::size_t n = 5;
  constexpr int C = 2;
  std::vector<D3> data(n * C);
  for (std::size_t i = 0; i < n; ++i) {
    for (int c = 0; c < C; ++c) data[i * C + c] = sample_value(i, c);
  }
  fuse::FieldRegistry reg;
  const fuse::FieldId id{4};
  ASSERT_TRUE(reg.bind(id, fuse::BasicFieldRef<D3>(data.data(), interleaved(n, C),
                                                   fuse::MemorySpace::Host)));
  EXPECT_TRUE(reg.has_gradient(id));

  // The strict getter still refuses: a dual field is not a binary64 field.
  const auto strict = reg.get(id);
  ASSERT_FALSE(strict);
  EXPECT_EQ(strict.error().code, "E-FIELD-VALUE-TYPE");

  auto flat = reg.get_scalars<double>(id);
  ASSERT_TRUE(flat);
  auto f = flat.value();
  EXPECT_EQ(f.layout().components, C * 4);
  EXPECT_EQ(f.layout().stride, std::size_t{C * 4});
  EXPECT_EQ(f.layout().count, n);
  for (std::size_t i = 0; i < n; ++i) {
    for (int c = 0; c < C; ++c) {
      for (int l = 0; l < 4; ++l) {
        EXPECT_EQ(f.at(i, c * 4 + l), data[i * C + c].d[l]);
      }
    }
  }
  f.at(3, 1 * 4 + 2) = -7.5;                 // writes through to the owner
  EXPECT_EQ(data[3 * C + 1].gradient(1), -7.5);

  const auto as_float = reg.get_scalars<float>(id);
  ASSERT_FALSE(as_float);
  EXPECT_EQ(as_float.error().code, "E-FIELD-VALUE-TYPE");

  // Through the typed view directly, and through the pipeline.
  auto typed = reg.get<D3>(id);
  ASSERT_TRUE(typed);
  auto lv = typed.value().as_scalars();
  ASSERT_TRUE(lv);
  EXPECT_EQ(lv.value().data(), f.data());
}

TEST(ScalarView, OrdinaryAndStorageFieldsAreTheirOwnScalarView) {
  std::vector<double> d(6, 1.5);
  std::vector<float> g(6, 0.5f);
  fuse::FieldRegistry reg;
  ASSERT_TRUE(reg.bind(fuse::FieldId{1}, fuse::FieldRef(d.data(), interleaved(3, 2),
                                                        fuse::MemorySpace::Host)));
  ASSERT_TRUE(reg.bind(fuse::FieldId{2},
                       fuse::BasicFieldRef<float>(g.data(), interleaved(3, 2),
                                                  fuse::MemorySpace::Host)));
  EXPECT_FALSE(reg.has_gradient(fuse::FieldId{1}));
  EXPECT_FALSE(reg.has_gradient(fuse::FieldId{2}));
  EXPECT_FALSE(reg.has_gradient(fuse::FieldId{9}));  // unbound
  auto a = reg.get_scalars(fuse::FieldId{1});
  ASSERT_TRUE(a);
  EXPECT_EQ(a.value().data(), d.data());
  EXPECT_EQ(a.value().layout().components, 2);
  // binary32 is a Storage type, not an array of binary64.
  const auto b = reg.get_scalars(fuse::FieldId{2});
  ASSERT_FALSE(b);
  EXPECT_EQ(b.error().code, "E-FIELD-VALUE-TYPE");
  EXPECT_TRUE(reg.get_scalars<float>(fuse::FieldId{2}));
}

TEST(ScalarView, ABlockedComponentLayoutIsRefusedNotPermuted) {
  std::vector<D3> data(6);
  // Blocked: component 0 of the three elements, then component 1 of them.
  const fuse::Layout blocked{1, 3, 3, 2};
  ASSERT_TRUE(blocked.well_formed());
  fuse::FieldRegistry reg;
  ASSERT_TRUE(reg.bind(fuse::FieldId{3},
                       fuse::BasicFieldRef<D3>(data.data(), blocked,
                                               fuse::MemorySpace::Host)));
  const auto flat = reg.get_scalars<double>(fuse::FieldId{3});
  ASSERT_FALSE(flat);
  EXPECT_EQ(flat.error().code, "E-FIELD-SCALAR-LAYOUT");
  EXPECT_NE(flat.error().message.find("blocked"), std::string::npos);

  // One component has nothing to interleave with, so its stride is irrelevant.
  const fuse::Layout single{2, 5, 3, 1};
  const auto lay = fuse::scalar_layout(single, 4);
  ASSERT_TRUE(lay);
  EXPECT_EQ(lay.value().components, 4);
  EXPECT_EQ(lay.value().stride, std::size_t{8});
  EXPECT_FALSE(fuse::scalar_layout(single, 0));
}

// ------------------------------------------------ per execution space (device)

template <class Space>
class DualOnSpace : public ::testing::Test {
 protected:
  using MemSpace = typename Space::memory_space;
  using Vec = Kokkos::View<double*, MemSpace>;
  using Idx = Kokkos::View<std::size_t*, MemSpace>;

  static Vec upload(const std::vector<double>& v, const std::string& label) {
    Vec d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  static Idx upload_idx(const std::vector<std::size_t>& v,
                        const std::string& label) {
    Idx d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  static std::vector<double> download(const Vec& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    std::vector<double> v(d.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
    return v;
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(DualOnSpace);

namespace {

template <class Space>
struct CompositeKernel {
  Kokkos::View<const double*, typename Space::memory_space> xs, ys;
  Kokkos::View<double*, typename Space::memory_space> out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const D2 f = composite(D2::variable(xs(i), 0), D2::variable(ys(i), 1));
    out(3 * i) = f.value();
    out(3 * i + 1) = f.gradient(0);
    out(3 * i + 2) = f.gradient(1);
  }
};

TYPED_TEST(DualOnSpace, TheCompositeRunsInAKernelAndKeepsItsGradients) {
  // Device and host evaluate sin, exp, log with different libraries, and the
  // device contracts a*b+c into a fused multiply-add the host does not, so
  // this compares to the closed form to rounding, not to the host bitwise.
  const std::size_t n = 64;
  std::vector<double> xs(n), ys(n);
  for (std::size_t i = 0; i < n; ++i) {
    xs[i] = 0.3 + 2.1 * static_cast<double>(i % 17) / 17.0;
    ys[i] = 0.8 + 2.2 * static_cast<double>(i % 11) / 11.0;
  }
  auto dx = TestFixture::upload(xs, "xs");
  auto dy = TestFixture::upload(ys, "ys");
  typename TestFixture::Vec out("out", 3 * n);
  Kokkos::parallel_for(
      "fuse::test::dual_composite", Kokkos::RangePolicy<TypeParam>(0, n),
      CompositeKernel<TypeParam>{dx, dy, out});
  Kokkos::fence();
  const auto got = TestFixture::download(out);
  for (std::size_t i = 0; i < n; ++i) {
    const Analytic a = analytic(xs[i], ys[i]);
    EXPECT_NEAR(got[3 * i], a.f, tol_for(a.f)) << i;
    EXPECT_NEAR(got[3 * i + 1], a.fx, tol_for(a.fx)) << i;
    EXPECT_NEAR(got[3 * i + 2], a.fy, tol_for(a.fy)) << i;
  }
}

/// A small, deterministic, row-sparse remap: row j has three entries with
/// unequal weights, so a wrong index or a reordered sum changes the rounding.
struct Csr {
  std::vector<std::size_t> row_begin, col;
  std::vector<double> val;
};
Csr make_remap(std::size_t rows, std::size_t cols) {
  Csr m;
  m.row_begin.push_back(0);
  for (std::size_t j = 0; j < rows; ++j) {
    for (std::size_t k = 0; k < 3; ++k) {
      m.col.push_back((j * 7 + k * 5 + 1) % cols);
      m.val.push_back(0.1 * (1.0 + static_cast<double>((j * 5 + k * 3) % 7)) /
                      3.1);
    }
    m.row_begin.push_back(m.col.size());
  }
  return m;
}

TYPED_TEST(DualOnSpace, AGradientThroughTheRemapIsTheRemapOfTheGradientBitwise) {
  constexpr std::size_t n_src = 37, n_dst = 29;
  constexpr int C = 2, L = 4, S = C * L;   // scalars per element
  const Csr m = make_remap(n_dst, n_src);

  std::vector<D3> x(n_src * C);
  for (std::size_t i = 0; i < n_src; ++i) {
    for (int c = 0; c < C; ++c) x[i * C + c] = sample_value(i, c);
  }
  fuse::FieldRegistry reg;
  ASSERT_TRUE(reg.bind(fuse::FieldId{1},
                       fuse::BasicFieldRef<D3>(x.data(), interleaved(n_src, C),
                                               fuse::MemorySpace::Host)));
  auto flat = reg.get_scalars<double>(fuse::FieldId{1});
  ASSERT_TRUE(flat);
  ASSERT_EQ(flat.value().layout().components, S);
  const std::vector<double> xs(flat.value().data(),
                               flat.value().data() + n_src * S);

  auto rb = TestFixture::upload_idx(m.row_begin, "rb");
  auto cl = TestFixture::upload_idx(m.col, "col");
  auto vl = TestFixture::upload(m.val, "val");

  // (a) The coupler's path: one multi-component remap over every scalar.
  auto X = TestFixture::upload(xs, "x");
  typename TestFixture::Vec Y("y", n_dst * S);
  fuse::num::remap_apply<TypeParam>(
      fuse::num::IndexSpanOf<TypeParam>(rb.data(), rb.extent(0)),
      fuse::num::IndexSpanOf<TypeParam>(cl.data(), cl.extent(0)),
      fuse::num::ConstSpanOf<TypeParam>(vl.data(), vl.extent(0)),
      fuse::num::ConstSpanOf<TypeParam>(X.data(), X.extent(0)),
      fuse::num::SpanOf<TypeParam>(Y.data(), Y.extent(0)), S);
  Kokkos::fence();
  const auto y_all = TestFixture::download(Y);

  // (b) The reference: the single-component remap applied to each scalar
  // (the value, and each gradient of each component) as a field of its own.
  for (int s = 0; s < S; ++s) {
    std::vector<double> xcol(n_src);
    for (std::size_t i = 0; i < n_src; ++i) xcol[i] = xs[i * S + s];
    auto Xc = TestFixture::upload(xcol, "xc");
    typename TestFixture::Vec Yc("yc", n_dst);
    fuse::num::remap_apply<TypeParam>(
        fuse::num::IndexSpanOf<TypeParam>(rb.data(), rb.extent(0)),
        fuse::num::IndexSpanOf<TypeParam>(cl.data(), cl.extent(0)),
        fuse::num::ConstSpanOf<TypeParam>(vl.data(), vl.extent(0)),
        fuse::num::ConstSpanOf<TypeParam>(Xc.data(), Xc.extent(0)),
        fuse::num::SpanOf<TypeParam>(Yc.data(), Yc.extent(0)));
    Kokkos::fence();
    const auto ycol = TestFixture::download(Yc);
    for (std::size_t j = 0; j < n_dst; ++j) {
      std::uint64_t a, b;
      std::memcpy(&a, &y_all[j * S + s], 8);
      std::memcpy(&b, &ycol[j], 8);
      ASSERT_EQ(a, b) << "row " << j << " scalar " << s;
    }
  }

  // (c) Written back into a field of dual numbers and read as dual numbers.
  std::vector<D3> y(n_dst * C);
  ASSERT_TRUE(reg.bind(fuse::FieldId{2},
                       fuse::BasicFieldRef<D3>(y.data(), interleaved(n_dst, C),
                                               fuse::MemorySpace::Host)));
  auto yflat = reg.get_scalars<double>(fuse::FieldId{2});
  ASSERT_TRUE(yflat);
  std::copy(y_all.begin(), y_all.end(), yflat.value().data());

  // (d) The same product in dual arithmetic, on the host: agrees to the
  // accumulated rounding of the sum (a fused multiply-add changes the last
  // bit, which is why this one is not bitwise).
  for (std::size_t j = 0; j < n_dst; ++j) {
    for (int c = 0; c < C; ++c) {
      D3 acc;
      double mag[L] = {};
      for (std::size_t k = m.row_begin[j]; k < m.row_begin[j + 1]; ++k) {
        const D3& xv = x[m.col[k] * C + c];
        acc += xv * m.val[k];
        for (int l = 0; l < L; ++l) mag[l] += std::fabs(m.val[k] * xv.d[l]);
      }
      for (int l = 0; l < L; ++l) {
        EXPECT_LE(std::fabs(y[j * C + c].d[l] - acc.d[l]),
                  8 * std::numeric_limits<double>::epsilon() * mag[l])
            << "row " << j << " component " << c << " scalar " << l;
      }
    }
  }
}

// -------------------------------------------- Taylor remainder through a remap

/// The Brusselator species residual of docs/dev/derivative-gate.md on a
/// periodic ring of n cells, at a frozen second species v:
///   r(u) = u - u_old - dt * (D * L u + u^2 v - A u),
/// with L the second-difference operator. L is applied as a remap (a sparse
/// matrix) so that the derivative action J dir = dir - dt (D L dir + 2 u v dir
/// - A dir) is produced by pushing a dual number through the remap kernel.
struct Ring {
  static constexpr std::size_t n = 32;
  static constexpr double D = 1e-3, A = 3.4, dt = 0.1;
  std::vector<double> u_old, v;
  Csr lap;
  Ring() : u_old(n), v(n) {
    const double h2 = 1.0 / static_cast<double>(n * n);
    lap.row_begin.push_back(0);
    for (std::size_t i = 0; i < n; ++i) {
      lap.col.push_back((i + n - 1) % n); lap.val.push_back(1.0 / h2);
      lap.col.push_back(i);               lap.val.push_back(-2.0 / h2);
      lap.col.push_back((i + 1) % n);     lap.val.push_back(1.0 / h2);
      lap.row_begin.push_back(lap.col.size());
      u_old[i] = 1.0 + 0.3 * std::sin(6.283185307179586 * i / n);
      v[i] = 3.0 + 0.5 * std::cos(6.283185307179586 * i / n);
    }
  }
  std::vector<double> residual(const std::vector<double>& u) const {
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i) {
      double Lu = 0.0;
      for (std::size_t k = lap.row_begin[i]; k < lap.row_begin[i + 1]; ++k) {
        Lu += lap.val[k] * u[lap.col[k]];
      }
      r[i] = u[i] - u_old[i] - dt * (D * Lu + u[i] * u[i] * v[i] - A * u[i]);
    }
    return r;
  }
};

TYPED_TEST(DualOnSpace, TaylorRemainderOfADualGradientThroughARemapIsOrderTwo) {
  const Ring ring;
  constexpr std::size_t n = Ring::n;
  std::vector<double> u(n), dir(n);
  for (std::size_t i = 0; i < n; ++i) {
    u[i] = 1.0 + 0.3 * std::sin(6.283185307179586 * i / n + 0.4);
    dir[i] = std::cos(3.0 * static_cast<double>(i)) + 0.5;
  }
  auto rb = TestFixture::upload_idx(ring.lap.row_begin, "rb");
  auto cl = TestFixture::upload_idx(ring.lap.col, "col");
  auto vl = TestFixture::upload(ring.lap.val, "val");

  // One dual number per cell with a single gradient: the value u_i and the
  // gradient dir_i, as the two scalars of a one-component field.
  const auto jv = [&](bool include_diffusion_gradient) {
    return [&, include_diffusion_gradient](const std::vector<double>& at,
                                          const std::vector<double>& d) {
      std::vector<D1> x(n);
      for (std::size_t i = 0; i < n; ++i) {
        x[i].d[0] = at[i];
        x[i].d[1] = d[i];
      }
      fuse::FieldRegistry reg;
      EXPECT_TRUE(reg.bind(fuse::FieldId{1},
                           fuse::BasicFieldRef<D1>(x.data(), interleaved(n, 1),
                                                   fuse::MemorySpace::Host)));
      auto flat = reg.get_scalars<double>(fuse::FieldId{1}).value();
      const std::vector<double> xs(flat.data(), flat.data() + 2 * n);
      auto X = TestFixture::upload(xs, "x");
      typename TestFixture::Vec Y("y", 2 * n);
      fuse::num::remap_apply<TypeParam>(
          fuse::num::IndexSpanOf<TypeParam>(rb.data(), rb.extent(0)),
          fuse::num::IndexSpanOf<TypeParam>(cl.data(), cl.extent(0)),
          fuse::num::ConstSpanOf<TypeParam>(vl.data(), vl.extent(0)),
          fuse::num::ConstSpanOf<TypeParam>(X.data(), X.extent(0)),
          fuse::num::SpanOf<TypeParam>(Y.data(), Y.extent(0)), 2);
      Kokkos::fence();
      const auto ly = TestFixture::download(Y);   // (L u, L dir) per cell
      std::vector<double> out(n);
      for (std::size_t i = 0; i < n; ++i) {
        D1 Lx;
        Lx.d[0] = ly[2 * i];
        Lx.d[1] = include_diffusion_gradient ? ly[2 * i + 1] : 0.0;
        const D1 r = x[i] - ring.u_old[i] -
                     Ring::dt * (Ring::D * Lx + x[i] * x[i] * ring.v[i] -
                                 Ring::A * x[i]);
        out[i] = r.gradient(0);
      }
      return out;
    };
  };
  const fuse::linalg::HostVectorFn F = [&](const std::vector<double>& at) {
    return ring.residual(at);
  };

  const auto good = fuse::linalg::taylor_remainder_order(
      F, jv(true), u, dir, /*h0=*/1e-1, /*ndecades=*/9);
  ASSERT_FALSE(good.at_floor_everywhere);
  EXPECT_GT(good.order, 1.8);
  EXPECT_LT(good.order, 2.3);

  // The control: leave the gradient out of the remap and the derivative action
  // is wrong by the whole diffusion term, which the sweep must see as order 1.
  const auto wrong = fuse::linalg::taylor_remainder_order(
      F, jv(false), u, dir, /*h0=*/1e-1, /*ndecades=*/9);
  EXPECT_GT(wrong.order, 0.7);
  EXPECT_LT(wrong.order, 1.3);
}

}  // namespace
