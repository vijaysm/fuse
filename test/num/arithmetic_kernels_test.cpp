// The arithmetic kernels for double-double and quad-double fields on every enabled
// execution space.
//
// Two checks on each kernel, for both accumulator types and for the three limb
// combinations (limbs in, limbs out; one limb in, one out: a binary64 field summed
// in the accumulator type; one limb in, limbs out):
//
//   * the result is the same bits as a plain host loop over the type's operators in
//     the kernel's documented order, so an execution space computes what the host
//     computes (floating-point sums are not associative; this is what shows that a
//     kernel keeps its order);
//   * the result is within the type's round-off of the exact result computed with
//     expansions (test/num/exact_expansion.hpp), which share no code with fuse::num.

#include "fuse/num/arithmetic_kernels.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "exact_expansion.hpp"
#include "fuse_test_main.hpp"

namespace {

using fuse::num::dd;
using fuse::num::qd;
using fuse::test::Expansion;

std::uint64_t bits_of(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

// ---- values of either type as limbs, an exact expansion, and a random generator
template <class A>
std::vector<double> limbs_of_value(const A& a);
template <>
std::vector<double> limbs_of_value<dd>(const dd& a) { return {a.hi, a.lo}; }
template <>
std::vector<double> limbs_of_value<qd>(const qd& a) { return {a[0], a[1], a[2], a[3]}; }

template <class A>
A value_from_limbs(const double* p) { return fuse::num::detail::Limbs<A>::load(p); }

template <class A>
Expansion exact_of(const A& a) {
  Expansion e;
  const auto l = limbs_of_value(a);
  for (std::size_t i = l.size(); i-- > 0;) e = fuse::test::grow(e, l[i]);
  return e;
}

template <class A>
A random_value(std::mt19937_64& rng, int lo_exp, int hi_exp) {
  std::uniform_real_distribution<double> unit(0.5, 1.0), sym(-1.0, 1.0);
  std::uniform_int_distribution<int> expo(lo_exp, hi_exp);
  double l[4];
  l[0] = std::ldexp(unit(rng), expo(rng));
  if (rng() & 1u) l[0] = -l[0];
  for (int i = 1; i < 4; ++i) l[i] = sym(rng) * std::ldexp(std::fabs(l[i - 1]), -54);
  return fuse::num::detail::Limbs<A>::load(l);
}

template <class A>
constexpr double unit_roundoff() { return fuse::num::limb_count<A> == 2 ? 0x1p-104 : 0x1p-208; }

// ---- the data of one run: a CSR matrix, values, weights
struct Csr {
  std::vector<std::size_t> row_begin, col;
  std::vector<double> val;
  std::size_t rows = 0, cols = 0;
};

Csr random_csr(std::mt19937_64& rng, std::size_t rows, std::size_t cols, std::size_t max_nnz) {
  Csr m;
  m.rows = rows;
  m.cols = cols;
  m.row_begin.push_back(0);
  std::uniform_real_distribution<double> w(-1.0, 1.0);
  for (std::size_t j = 0; j < rows; ++j) {
    const std::size_t nnz = rng() % (max_nnz + 1);
    for (std::size_t k = 0; k < nnz; ++k) {
      m.col.push_back(rng() % cols);
      m.val.push_back(w(rng));
    }
    m.row_begin.push_back(m.col.size());
  }
  return m;
}

template <class Space>
class ArithmeticOnSpace : public ::testing::Test {
 protected:
  using Mem = typename Space::memory_space;
  template <class T>
  using Vec = Kokkos::View<T*, Mem>;

  template <class T>
  static Vec<T> upload(const std::vector<T>& v) {
    Vec<T> d(Kokkos::view_alloc(std::string("u"), Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  template <class T>
  static std::vector<T> download(const Vec<T>& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    std::vector<T> v(d.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
    return v;
  }
  static Vec<double> zeros(std::size_t n) {
    Vec<double> d(Kokkos::view_alloc(std::string("z"), Kokkos::WithoutInitializing), n);
    Kokkos::deep_copy(d, 0.0);
    return d;
  }

  // The values: `n` values as `limbs` limbs each (1 limb: the leading limb only).
  template <class A>
  static std::vector<A> make_values(std::mt19937_64& rng, std::size_t n, int lo, int hi, int limbs) {
    std::vector<A> v;
    for (std::size_t i = 0; i < n; ++i) {
      A a = random_value<A>(rng, lo, hi);
      if (limbs == 1) a = A(a.to_double());
      v.push_back(a);
    }
    return v;
  }
  template <class A>
  static std::vector<double> flatten(const std::vector<A>& v, int limbs) {
    std::vector<double> out;
    for (const A& a : v) {
      const auto l = limbs_of_value(a);
      for (int i = 0; i < limbs; ++i) out.push_back(l[static_cast<std::size_t>(i)]);
    }
    return out;
  }

  template <class A>
  static void expect_remap(int limbs_in, int limbs_out) {
    constexpr int L = fuse::num::limb_count<A>;
    ASSERT_TRUE(limbs_in == 1 || limbs_in == L);
    std::mt19937_64 rng(static_cast<std::uint64_t>(1000 + 10 * limbs_in + limbs_out));
    const std::size_t components = 3;
    const Csr m = random_csr(rng, 211, 173, 12);
    const auto xv = make_values<A>(rng, m.cols * components, -10, 10, limbs_in);
    const auto flat = flatten(xv, limbs_in);
    const auto drow = upload(m.row_begin);
    const auto dcol = upload(m.col);
    const auto dval = upload(m.val);
    const auto dx = upload(flat);
    auto dy = zeros(m.rows * components * static_cast<std::size_t>(limbs_out));
    fuse::num::arith_remap_apply<Space, A>(
        fuse::num::IndexSpanOf<Space>(drow.data(), drow.extent(0)),
        fuse::num::IndexSpanOf<Space>(dcol.data(), dcol.extent(0)),
        fuse::num::ConstSpanOf<Space>(dval.data(), dval.extent(0)),
        fuse::num::ConstSpanOf<Space>(dx.data(), dx.extent(0)),
        fuse::num::SpanOf<Space>(dy.data(), dy.extent(0)), components, limbs_in, limbs_out);
    Kokkos::fence();
    const auto y = download(dy);

    double worst = 0.0;
    for (std::size_t j = 0; j < m.rows; ++j) {
      for (std::size_t c = 0; c < components; ++c) {
        // the plain loop, in the documented order
        A acc(0.0);
        Expansion exact;
        double mass = 0.0;  // sum |w x|
        for (std::size_t k = m.row_begin[j]; k < m.row_begin[j + 1]; ++k) {
          const A& x = xv[m.col[k] * components + c];
          acc = acc + x * m.val[k];
          exact = fuse::test::add(exact, fuse::test::mul(Expansion{m.val[k]}, exact_of(x)));
          mass += std::fabs(m.val[k] * x.to_double());
        }
        const double* got = &y[(j * components + c) * static_cast<std::size_t>(limbs_out)];
        if (limbs_out == 1) {
          ASSERT_EQ(bits_of(got[0]), bits_of(acc.to_double())) << "row " << j << " component " << c;
          // one rounding of the exact sum: |exact - got| <= ulp/2 plus the accumulator's round-off
          const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, Expansion{got[0]})));
          const double ulp = std::nextafter(std::fabs(got[0]), INFINITY) - std::fabs(got[0]);
          ASSERT_LE(err, 0.5 * ulp + static_cast<double>(m.row_begin[j + 1] - m.row_begin[j] + 1) *
                                         unit_roundoff<A>() * mass * 4);
        } else {
          const A r = value_from_limbs<A>(got);
          const auto lr = limbs_of_value(r);
          const auto la = limbs_of_value(acc);
          for (std::size_t i = 0; i < lr.size(); ++i) {
            ASSERT_EQ(bits_of(lr[i]), bits_of(la[i])) << "row " << j << " component " << c << " limb " << i;
          }
          const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, exact_of(r))));
          const double bound = static_cast<double>(m.row_begin[j + 1] - m.row_begin[j] + 1) *
                               unit_roundoff<A>() * mass * 4;
          worst = std::fmax(worst, mass > 0 ? err / mass : 0.0);
          ASSERT_LE(err, bound) << "row " << j << " component " << c;
        }
      }
    }
  }

  template <class A>
  static void expect_accumulate(int limbs_in, int limbs_out) {
    using fuse::AccumOp;
    constexpr int L = fuse::num::limb_count<A>;
    ASSERT_TRUE(limbs_in == 1 || limbs_in == L);
    std::mt19937_64 rng(static_cast<std::uint64_t>(2000 + 10 * limbs_in + limbs_out));
    const std::size_t n = 997;
    for (const AccumOp op :
         {AccumOp::Instant, AccumOp::Sum, AccumOp::Mean, AccumOp::Integral, AccumOp::Min, AccumOp::Max}) {
      auto acc_host = make_values<A>(rng, n, -3, 3, limbs_out);
      auto dacc = upload(flatten(acc_host, limbs_out));
      for (int step = 0; step < 4; ++step) {
        const auto sv = make_values<A>(rng, n, -3, 3, limbs_in);
        const auto ds = upload(flatten(sv, limbs_in));
        const bool first = step == 0;
        fuse::num::arith_accumulate<Space, A>(
            op, fuse::num::ConstSpanOf<Space>(ds.data(), ds.extent(0)),
            fuse::num::SpanOf<Space>(dacc.data(), dacc.extent(0)), first, limbs_in, limbs_out);
        Kokkos::fence();
        // the plain loop; a value is stored as one limb or as limbs, as the kernel stores it
        for (std::size_t i = 0; i < n; ++i) {
          const A& s = sv[i];
          A r = s;
          if (!first && op != AccumOp::Instant) {
            const A a = acc_host[i];
            switch (op) {
              case AccumOp::Sum: case AccumOp::Mean: case AccumOp::Integral: r = a + s; break;
              case AccumOp::Min: r = s < a ? s : a; break;
              case AccumOp::Max: r = s > a ? s : a; break;
              case AccumOp::Instant: break;
            }
          }
          acc_host[i] = limbs_out == 1 ? A(r.to_double()) : r;
        }
        const auto got = download(dacc);
        for (std::size_t i = 0; i < n; ++i) {
          const auto want = limbs_of_value(acc_host[i]);
          for (int l = 0; l < limbs_out; ++l) {
            ASSERT_EQ(bits_of(got[i * static_cast<std::size_t>(limbs_out) + static_cast<std::size_t>(l)]),
                      bits_of(want[static_cast<std::size_t>(l)]))
                << "op " << static_cast<int>(op) << " step " << step << " value " << i << " limb " << l;
          }
        }
      }
    }
  }

  template <class A>
  static void expect_scale(int limbs_in, int limbs_out) {
    constexpr int L = fuse::num::limb_count<A>;
    ASSERT_TRUE(limbs_in == 1 || limbs_in == L);
    std::mt19937_64 rng(static_cast<std::uint64_t>(3000 + 10 * limbs_in + limbs_out));
    const std::size_t n = 500;
    const auto sv = make_values<A>(rng, n, -5, 5, limbs_in);
    const auto ds = upload(flatten(sv, limbs_in));
    for (const bool divide : {false, true}) {
      for (const double f : {0.25, 3.0, 0.1, -7.0, 1.0}) {
        auto dd_out = zeros(n * static_cast<std::size_t>(limbs_out));
        fuse::num::arith_scale<Space, A>(f, divide, fuse::num::ConstSpanOf<Space>(ds.data(), ds.extent(0)),
                                         fuse::num::SpanOf<Space>(dd_out.data(), dd_out.extent(0)),
                                         limbs_in, limbs_out);
        Kokkos::fence();
        const auto got = download(dd_out);
        for (std::size_t i = 0; i < n; ++i) {
          const A r = divide ? A(sv[i] / A(f)) : A(sv[i] * A(f));
          const auto want = limbs_of_value(limbs_out == 1 ? A(r.to_double()) : r);
          for (int l = 0; l < limbs_out; ++l) {
            ASSERT_EQ(bits_of(got[i * static_cast<std::size_t>(limbs_out) + static_cast<std::size_t>(l)]),
                      bits_of(want[static_cast<std::size_t>(l)]))
                << "factor " << f << " divide " << divide << " value " << i;
          }
          if (!divide && limbs_in == L && limbs_out == L && (f == 0.25 || f == 1.0)) {
            // a power of two is exact: every limb is the limb times f
            const auto lin = limbs_of_value(sv[i]);
            for (int l = 0; l < L; ++l) {
              ASSERT_EQ(got[i * static_cast<std::size_t>(L) + static_cast<std::size_t>(l)], f * lin[static_cast<std::size_t>(l)]);
            }
          }
        }
      }
    }
  }

  template <class A>
  static void expect_weighted_sum(int limbs) {
    constexpr int L = fuse::num::limb_count<A>;
    ASSERT_TRUE(limbs == 1 || limbs == L);
    std::mt19937_64 rng(static_cast<std::uint64_t>(4000 + limbs));
    std::uniform_real_distribution<double> w(0.0, 2.0);
    for (const std::size_t n : {std::size_t{1}, std::size_t{255}, std::size_t{256}, std::size_t{257}, std::size_t{1000},
                                std::size_t{5000}}) {
      const auto xv = make_values<A>(rng, n, -4, 4, limbs);
      std::vector<double> wt(n);
      for (double& v : wt) v = w(rng);
      const auto dx = upload(flatten(xv, limbs));
      const auto dw = upload(wt);
      const std::size_t chunks = fuse::num::deterministic_chunks(n);
      auto scratch = zeros(chunks * static_cast<std::size_t>(L));
      const A got = fuse::num::arith_weighted_sum<Space, A>(
          fuse::num::ConstSpanOf<Space>(dx.data(), dx.extent(0)),
          fuse::num::ConstSpanOf<Space>(dw.data(), dw.extent(0)),
          fuse::num::SpanOf<Space>(scratch.data(), scratch.extent(0)), limbs);
      // the plain loop in the documented chunked order
      A total(0.0);
      for (std::size_t c = 0; c < chunks; ++c) {
        A s(0.0);
        const std::size_t end = std::min(n, (c + 1) * fuse::num::kDeterministicChunk);
        for (std::size_t i = c * fuse::num::kDeterministicChunk; i < end; ++i) s = s + xv[i] * wt[i];
        total = total + s;
      }
      const auto lg = limbs_of_value(got), lt = limbs_of_value(total);
      for (std::size_t i = 0; i < lg.size(); ++i) ASSERT_EQ(bits_of(lg[i]), bits_of(lt[i])) << "n " << n << " limb " << i;
      // against the exact sum
      Expansion exact;
      double mass = 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        exact = fuse::test::add(exact, fuse::test::mul(Expansion{wt[i]}, exact_of(xv[i])));
        mass += std::fabs(wt[i] * xv[i].to_double());
      }
      const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, exact_of(got))));
      ASSERT_LE(err, static_cast<double>(n + chunks) * unit_roundoff<A>() * mass * 4) << "n " << n;
    }
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(ArithmeticOnSpace);

TYPED_TEST(ArithmeticOnSpace, RemapMatchesThePlainLoopAndTheExactSumForDoubleDouble) {
  TestFixture::template expect_remap<dd>(2, 2);
  TestFixture::template expect_remap<dd>(1, 1);
  TestFixture::template expect_remap<dd>(1, 2);
}

TYPED_TEST(ArithmeticOnSpace, RemapMatchesThePlainLoopAndTheExactSumForQuadDouble) {
  TestFixture::template expect_remap<qd>(4, 4);
  TestFixture::template expect_remap<qd>(1, 1);
  TestFixture::template expect_remap<qd>(1, 4);
}

TYPED_TEST(ArithmeticOnSpace, AccumulateMatchesThePlainLoopForEveryOperation) {
  TestFixture::template expect_accumulate<dd>(2, 2);
  TestFixture::template expect_accumulate<dd>(1, 1);
  TestFixture::template expect_accumulate<qd>(4, 4);
  TestFixture::template expect_accumulate<qd>(1, 1);
}

TYPED_TEST(ArithmeticOnSpace, ScaleAndDivideMatchThePlainOperators) {
  TestFixture::template expect_scale<dd>(2, 2);
  TestFixture::template expect_scale<dd>(1, 1);
  TestFixture::template expect_scale<qd>(4, 4);
  TestFixture::template expect_scale<qd>(1, 1);
}

TYPED_TEST(ArithmeticOnSpace, WeightedSumKeepsItsChunkOrderAndIsAccurate) {
  TestFixture::template expect_weighted_sum<dd>(2);
  TestFixture::template expect_weighted_sum<dd>(1);
  TestFixture::template expect_weighted_sum<qd>(4);
  TestFixture::template expect_weighted_sum<qd>(1);
}

TYPED_TEST(ArithmeticOnSpace, ASumThatCancelsBinary64LosesIsKeptByDoubleDouble) {
  // 1e16 + 1 - 1e16 over many pairs: binary64 gives 0 for each, double-double the 1
  using Space = TypeParam;
  const std::size_t n = 3000;
  std::vector<double> x(n), w(n, 1.0);
  for (std::size_t i = 0; i < n; i += 3) {
    x[i] = 1.0e16;
    x[i + 1] = 1.0;
    x[i + 2] = -1.0e16;
  }
  const auto dx = TestFixture::upload(x);
  const auto dw = TestFixture::upload(w);
  auto scratch = TestFixture::zeros(fuse::num::deterministic_chunks(n) * 2);
  const dd s = fuse::num::arith_weighted_sum<Space, dd>(
      fuse::num::ConstSpanOf<Space>(dx.data(), dx.extent(0)),
      fuse::num::ConstSpanOf<Space>(dw.data(), dw.extent(0)),
      fuse::num::SpanOf<Space>(scratch.data(), scratch.extent(0)), 1);
  EXPECT_EQ(s.to_double(), static_cast<double>(n / 3));
  double plain = 0.0;
  for (const double v : x) plain += v;
  EXPECT_NE(plain, static_cast<double>(n / 3));
}

TEST(ArithmeticTypes, TheClosedListNamesTheTypesAndNothingElse) {
  EXPECT_TRUE(fuse::num::is_arithmetic_type_name("dd"));
  EXPECT_TRUE(fuse::num::is_arithmetic_type_name("qd"));
  EXPECT_FALSE(fuse::num::is_arithmetic_type_name("binary64"));
  EXPECT_FALSE(fuse::num::is_arithmetic_type_name("dual<binary64,1>"));
  EXPECT_FALSE(fuse::num::is_arithmetic_type_name(""));
  int limbs = 0;
  EXPECT_TRUE(fuse::num::visit_arithmetic_type("qd", [&]<class A>(std::type_identity<A>) {
    limbs = fuse::num::limb_count<A>;
  }));
  EXPECT_EQ(limbs, 4);
  EXPECT_EQ(fuse::num::limb_count<dd>, fuse::value_traits<dd>::scalar_count);
  EXPECT_EQ(fuse::num::limb_count<qd>, fuse::value_traits<qd>::scalar_count);
}
