#pragma once

/// The kernels that do arithmetic on Arithmetic field values (double-double and
/// quad-double), on a Kokkos execution space. plans/2026-10-22-universal-number-
/// types.md, stage U4.
///
/// WHAT AN ARITHMETIC FIELD IS HERE. A field of `dd` or `qd` values is a flat
/// array of binary64 limbs, the limbs of one value consecutive: value v of a
/// double-double field is limbs[2v] (the leading one) and limbs[2v + 1]. Every
/// kernel below takes such an array as an ordinary `ConstSpanOf`/`SpanOf` plus
/// the number of limbs per value, so that the same views, staging and device
/// mirrors that carry binary64 data carry these. An operation that only moves
/// values (copy, gather, scatter) is the existing binary64 kernel run over the
/// limbs; the kernels here are the ones that must add or multiply the values.
///
/// ACCUMULATOR TYPE. Each kernel is a template on the accumulator type `A` (dd or
/// qd) and takes the number of limbs of its input and of its output separately,
/// each either 1 or `limb_count<A>`. One limb is a binary64 number, widened to
/// `A` exactly; `limb_count<A>` limbs are an `A`. So the same kernel serves a
/// field of `A` values (limbs in, limbs out) and a binary64 field whose sums are
/// accumulated in `A` and rounded once on the way out (1 limb in, 1 limb out,
/// the accumulation type of the plan).
///
/// ORDER. The summation orders are the ones of the binary64 kernels in
/// space_kernels.hpp (row order of a CSR remap; fixed chunks of
/// kDeterministicChunk values for a reduction, partials combined in ascending
/// chunk order), so a result does not depend on the execution space or on the
/// number of threads: it is the same bits on the host and on a device, which
/// test/num/arithmetic_kernels_test.cpp checks on every enabled space.
///
/// LABELS stay under the 22 characters a small string holds; see
/// space_kernels.hpp for why a longer label allocates on every launch.

#include <cstddef>
#include <string_view>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "fuse/num/dd.hpp"
#include "fuse/num/kernels.hpp"
#include "fuse/num/qd.hpp"
#include "fuse/num/space_kernels.hpp"

namespace fuse::num {

namespace detail {

template <class A>
struct Limbs;

template <>
struct Limbs<dd> {
  static constexpr int count = 2;
  KOKKOS_INLINE_FUNCTION static dd load(const double* p) noexcept { return dd(p[0], p[1]); }
  KOKKOS_INLINE_FUNCTION static void store(double* p, const dd& v) noexcept {
    p[0] = v.hi;
    p[1] = v.lo;
  }
  /// The exact product of two binary64 numbers.
  KOKKOS_INLINE_FUNCTION static dd product(double a, double b) noexcept {
    return dd::from_product(a, b);
  }
  KOKKOS_INLINE_FUNCTION static double narrow(const dd& v) noexcept { return v.to_double(); }
};

template <>
struct Limbs<qd> {
  static constexpr int count = 4;
  KOKKOS_INLINE_FUNCTION static qd load(const double* p) noexcept {
    return qd(p[0], p[1], p[2], p[3]);
  }
  KOKKOS_INLINE_FUNCTION static void store(double* p, const qd& v) noexcept {
    p[0] = v[0];
    p[1] = v[1];
    p[2] = v[2];
    p[3] = v[3];
  }
  KOKKOS_INLINE_FUNCTION static qd product(double a, double b) noexcept {
    return qd(dd::from_product(a, b));
  }
  KOKKOS_INLINE_FUNCTION static double narrow(const qd& v) noexcept { return v.to_double(); }
};

}  // namespace detail

/// Binary64 limbs in one value of the accumulator type `A`.
template <class A>
inline constexpr int limb_count = detail::Limbs<A>::count;

/// The value at `p`: `limbs` is 1 (a binary64 number, widened exactly) or
/// `limb_count<A>` (the limbs of an `A`).
template <class A>
KOKKOS_INLINE_FUNCTION A load_value(const double* p, int limbs) noexcept {
  return limbs == 1 ? A(p[0]) : detail::Limbs<A>::load(p);
}

/// Writes `v` at `p`: as its limbs, or as one binary64 number (rounded to
/// nearest, ties to even, once) when `limbs` is 1.
template <class A>
KOKKOS_INLINE_FUNCTION void store_value(double* p, int limbs, const A& v) noexcept {
  if (limbs == 1) {
    p[0] = detail::Limbs<A>::narrow(v);
  } else {
    detail::Limbs<A>::store(p, v);
  }
}

/// `w * (the value at p)`; exact for a binary64 value, one rounding of `A` for
/// the limbs of an `A`.
template <class A>
KOKKOS_INLINE_FUNCTION A weighted_value(const double* p, int limbs, double w) noexcept {
  return limbs == 1 ? detail::Limbs<A>::product(p[0], w) : A(load_value<A>(p, limbs) * A(w));
}

/// Calls `f(std::type_identity<A>{})` for the Arithmetic type named `name` (the
/// `value_traits` name) and returns true; returns false when `name` is not in
/// the closed list fuse/num/arithmetic_types.def.
template <class F>
bool visit_arithmetic_type(std::string_view name, F&& f) {
#define FUSE_ARITHMETIC_TYPE(Type, Name)       \
  if (name == Name) {                          \
    f(std::type_identity<Type>{});             \
    return true;                               \
  }
#include "fuse/num/arithmetic_types.def"
#undef FUSE_ARITHMETIC_TYPE
  return false;
}

/// True when `name` is in the closed list.
[[nodiscard]] inline bool is_arithmetic_type_name(std::string_view name) {
  return visit_arithmetic_type(name, [](auto) {});
}

// ---------------------------------------------------------------------------
// accumulate

/// `acc <- op(acc, sample)` over `n` values; the arithmetic of the sum is `A`'s.
/// Min and Max compare the values (leading limb first), NaN-handling as the
/// binary64 kernel's.
template <class Space, class A>
struct ArithAccumulateFunctor {
  AccumOp op;
  ConstSpanOf<Space> sample;
  SpanOf<Space> acc;
  bool first;
  int limbs_in;
  int limbs_out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const A s = load_value<A>(sample.data() + i * static_cast<std::size_t>(limbs_in), limbs_in);
    double* const p = acc.data() + i * static_cast<std::size_t>(limbs_out);
    A r = s;
    if (!first && op != AccumOp::Instant) {
      const A a = load_value<A>(p, limbs_out);
      switch (op) {
        case AccumOp::Sum:
        case AccumOp::Mean:
        case AccumOp::Integral:
          r = a + s;
          break;
        case AccumOp::Min:
          r = s < a ? s : a;
          break;
        case AccumOp::Max:
          r = s > a ? s : a;
          break;
        case AccumOp::Instant:
          break;
      }
    }
    store_value<A>(p, limbs_out, r);
  }
};

template <class Space, class A>
void arith_accumulate(AccumOp op, ConstSpanOf<Space> sample, SpanOf<Space> acc, bool first,
                      int limbs_in, int limbs_out) {
  const std::size_t n = acc.extent(0) / static_cast<std::size_t>(limbs_out);
  Kokkos::parallel_for("fuse::accum_a", Kokkos::RangePolicy<Space>(0, n),
                       ArithAccumulateFunctor<Space, A>{op, sample, acc, first, limbs_in, limbs_out});
}

// ---------------------------------------------------------------------------
// scale and divide

/// `dst <- factor * src` (`divide` false) or `dst <- src / factor` (`divide`
/// true), each in `A`. Dividing by a count is one correctly rounded division in
/// `A`, not a multiplication by a rounded reciprocal.
template <class Space, class A>
struct ArithScaleFunctor {
  double factor;
  bool divide;
  ConstSpanOf<Space> src;
  SpanOf<Space> dst;
  int limbs_in;
  int limbs_out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    const A s = load_value<A>(src.data() + i * static_cast<std::size_t>(limbs_in), limbs_in);
    const A r = divide ? A(s / A(factor)) : A(s * A(factor));
    store_value<A>(dst.data() + i * static_cast<std::size_t>(limbs_out), limbs_out, r);
  }
};

template <class Space, class A>
void arith_scale(double factor, bool divide, ConstSpanOf<Space> src, SpanOf<Space> dst, int limbs_in,
                 int limbs_out) {
  const std::size_t n = dst.extent(0) / static_cast<std::size_t>(limbs_out);
  Kokkos::parallel_for("fuse::scale_a", Kokkos::RangePolicy<Space>(0, n),
                       ArithScaleFunctor<Space, A>{factor, divide, src, dst, limbs_in, limbs_out});
}

// ---------------------------------------------------------------------------
// remap

/// `y = W x` for a CSR matrix of binary64 weights: row `j` of `y` is the sum over
/// the row's entries, in ascending entry order, of the weight times the value of
/// the column's entry, accumulated in `A`. `components` is the number of values
/// per row (the binary64 kernel's `components`, in values, not limbs).
template <class Space, class A>
struct ArithRemapFunctor {
  IndexSpanOf<Space> row_begin;
  IndexSpanOf<Space> col;
  ConstSpanOf<Space> val;
  ConstSpanOf<Space> x;
  SpanOf<Space> y;
  std::size_t components;
  int limbs_in;
  int limbs_out;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t j) const {
    const std::size_t row_end = row_begin(j + 1);
    const auto li = static_cast<std::size_t>(limbs_in);
    const auto lo = static_cast<std::size_t>(limbs_out);
    for (std::size_t c = 0; c < components; ++c) {
      A acc(0.0);
      for (std::size_t k = row_begin(j); k < row_end; ++k) {
        acc = acc + weighted_value<A>(x.data() + (col(k) * components + c) * li, limbs_in, val(k));
      }
      store_value<A>(y.data() + (j * components + c) * lo, limbs_out, acc);
    }
  }
};

template <class Space, class A>
void arith_remap_apply(IndexSpanOf<Space> row_begin, IndexSpanOf<Space> col, ConstSpanOf<Space> val,
                       ConstSpanOf<Space> x, SpanOf<Space> y, std::size_t components, int limbs_in,
                       int limbs_out) {
  const std::size_t rows = row_begin.extent(0) - 1;
  Kokkos::parallel_for("fuse::remap_a", Kokkos::RangePolicy<Space>(0, rows),
                       ArithRemapFunctor<Space, A>{row_begin, col, val, x, y, components, limbs_in,
                                                   limbs_out});
}

// ---------------------------------------------------------------------------
// weighted sum

/// `sum_i w_i x_i` in `A`, in the fixed order of `deterministic_dot`: one partial
/// per chunk of kDeterministicChunk values (each accumulated in ascending
/// order), the partials combined in ascending chunk order. `scratch` holds at
/// least `deterministic_chunks(n) * limb_count<A>` binary64 values. The result
/// is an `A`: the caller reduces across ranks in `A` and narrows once.
template <class Space, class A>
struct ArithDotFunctor {
  ConstSpanOf<Space> x;
  ConstSpanOf<Space> w;
  SpanOf<Space> scratch;
  std::size_t n;
  int limbs;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t c) const {
    const std::size_t begin = c * kDeterministicChunk;
    const std::size_t end = (begin + kDeterministicChunk < n) ? begin + kDeterministicChunk : n;
    A s(0.0);
    for (std::size_t i = begin; i < end; ++i) {
      s = s + weighted_value<A>(x.data() + i * static_cast<std::size_t>(limbs), limbs, w(i));
    }
    detail::Limbs<A>::store(scratch.data() + c * static_cast<std::size_t>(limb_count<A>), s);
  }
};

template <class Space, class A>
[[nodiscard]] A arith_weighted_sum(ConstSpanOf<Space> x, ConstSpanOf<Space> w, SpanOf<Space> scratch,
                                   int limbs) {
  const std::size_t n = w.extent(0);
  const std::size_t n_chunks = deterministic_chunks(n);
  Kokkos::parallel_for("fuse::dot_a", Kokkos::RangePolicy<Space>(0, n_chunks),
                       ArithDotFunctor<Space, A>{x, w, scratch, n, limbs});
  constexpr auto L = static_cast<std::size_t>(limb_count<A>);
  A total(0.0);
  if constexpr (Kokkos::SpaceAccessibility<Kokkos::HostSpace, typename Space::memory_space>::accessible) {
    Space().fence("fuse::dot_a_fence");
    for (std::size_t c = 0; c < n_chunks; ++c) total = total + detail::Limbs<A>::load(scratch.data() + c * L);
  } else {
    const auto host_scratch = Kokkos::create_mirror_view(scratch);
    Kokkos::deep_copy(host_scratch, scratch);
    for (std::size_t c = 0; c < n_chunks; ++c) {
      total = total + detail::Limbs<A>::load(host_scratch.data() + c * L);
    }
  }
  return total;
}

}  // namespace fuse::num
