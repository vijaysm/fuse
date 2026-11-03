#pragma once

/// The elementwise kernels the reverse sweep (src/runtime/runner_adjoint.cpp,
/// docs/dev/adjoint-schedule.md) is made of, on any Kokkos execution space.
///
/// Every rule of the sweep except the transpose of a remap is one of these three
/// loops over the values of an adjoint array: add a multiple of one array to
/// another, take an array and set it to zero, fill it with a value. The transpose of
/// a remap is `remap_apply` (space_kernels.hpp) over the transposed CSR matrix.
///
/// Named functors, not KOKKOS_LAMBDA, for the reason space_kernels.hpp gives above
/// `AccumulateFunctor`. Each result element depends on its own inputs only, so a
/// run does not depend on how the range is split across threads, and the multiply
/// and add are not fused on the device (`unfused_multiply_add`) so that the host
/// and the device round the same way.

#include <cstddef>

#include <Kokkos_Core.hpp>

#include "fuse/num/space_kernels.hpp"

namespace fuse::num {

/// `dst[i] <- dst[i] + factor * src[i]`: the adjoint of an input accumulates the
/// contribution of each use of it.
template <class Space>
struct AxpyFunctor {
  double factor;
  ConstSpanOf<Space> src;
  SpanOf<Space> dst;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    dst(i) = unfused_multiply_add(dst(i), factor, src(i));
  }
};

template <class Space>
void axpy(double factor, ConstSpanOf<Space> src, SpanOf<Space> dst) {
  Kokkos::parallel_for("fuse::adjoint_axpy", Kokkos::RangePolicy<Space>(0, dst.extent(0)),
                       AxpyFunctor<Space>{factor, src, dst});
}

/// `taken[i] <- adjoint[i]; adjoint[i] <- 0`: the reverse of an assignment takes
/// the adjoint of the overwritten output, which is dead before the write.
template <class Space>
struct TakeFunctor {
  SpanOf<Space> taken;
  SpanOf<Space> adjoint;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const {
    taken(i) = adjoint(i);
    adjoint(i) = 0.0;
  }
};

template <class Space>
void take(SpanOf<Space> taken, SpanOf<Space> adjoint) {
  Kokkos::parallel_for("fuse::adjoint_take", Kokkos::RangePolicy<Space>(0, adjoint.extent(0)),
                       TakeFunctor<Space>{taken, adjoint});
}

/// `dst[i] <- value`.
template <class Space>
struct FillFunctor {
  double value;
  SpanOf<Space> dst;
  KOKKOS_INLINE_FUNCTION void operator()(const std::size_t i) const { dst(i) = value; }
};

template <class Space>
void fill(double value, SpanOf<Space> dst) {
  Kokkos::parallel_for("fuse::adjoint_fill", Kokkos::RangePolicy<Space>(0, dst.extent(0)),
                       FillFunctor<Space>{value, dst});
}

}  // namespace fuse::num
