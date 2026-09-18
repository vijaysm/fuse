#pragma once

/// The per-step kernels, parameterised on a Kokkos execution space.
///
/// WHICH KERNELS, AND WHY THESE. These are the operations a coupled step
/// actually spends its time in, in the order they run:
///
///   accumulate  once per SUBSTEP, on every coupled field -- the most
///               frequently executed loop in the library
///   gather      packs the values a data move sends, once per exchange
///   remap_apply the sparse matrix-vector product that IS the field transfer
///   scatter     unpacks what arrived
///   relax       once per iteration sweep, on every iterated field
///   dot         the convergence measure's reduction, once per sweep
///
/// Everything else in a step is either O(1) bookkeeping or the component's own
/// solver, which is not ours to parallelise.
///
/// WHY THE SERIAL VERSIONS STAY. `fuse/num/kernels.hpp` keeps the plain-loop
/// forms, and they are not dead code: they are the ORACLE these are tested
/// against. A parallel reduction does not associate the same way a sequential
/// one does, and a parallel_for over a wrongly computed range produces
/// plausible numbers; comparing against a loop nobody can misread is how those
/// are caught. The serial forms are also what runs where a view is a raw
/// pointer into a component's own array and no space has been chosen.
///
/// WHAT IS NOT CLAIMED. No device performance number. This machine has no GPU
/// and its Kokkos reports `THREADS;SERIAL`, so what is verified here is that
/// every kernel compiles and produces the oracle's answer on **every execution
/// space the build enabled** -- locally Serial and Threads, with OpenMP in the
/// CI leg. Parameterisation is a precondition for device performance, not
/// evidence of it, and the two are not conflated.
///
/// LABELS. Every kernel launch is named. A profile of a coupled run should say
/// `fuse::accumulate` rather than `parallel_for`, because the first question
/// asked of a slow run is which phase is slow.

#include <cstddef>

#include <Kokkos_Core.hpp>

#include "fuse/base/config.hpp"
#include "fuse/num/kernels.hpp"

namespace fuse::num {

/// A contiguous, borrowed run of values in `Space`'s memory. Unmanaged so that
/// wrapping a component's own array costs nothing and allocates nothing.
template <class Space>
using SpanOf = Kokkos::View<double*, typename Space::memory_space,
                            Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class Space>
using ConstSpanOf = Kokkos::View<const double*, typename Space::memory_space,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
template <class Space>
using IndexSpanOf =
    Kokkos::View<const std::size_t*, typename Space::memory_space,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

/// `acc <- op(acc, sample)`, the per-substep time accumulation.
///
/// `first` is passed rather than derived from a substep counter inside the
/// kernel: the branch is then uniform across the range instead of per element,
/// and the caller already knows which substep it is on.
template <class Space>
void accumulate(AccumOp op, ConstSpanOf<Space> sample, SpanOf<Space> acc,
                bool first) {
  const std::size_t n = acc.extent(0);
  Kokkos::parallel_for(
      "fuse::accumulate", Kokkos::RangePolicy<Space>(0, n),
      KOKKOS_LAMBDA(const std::size_t i) {
        switch (op) {
          case AccumOp::Instant:
            acc(i) = sample(i);
            break;
          case AccumOp::Sum:
          case AccumOp::Mean:
          case AccumOp::Integral:
            acc(i) = first ? sample(i) : acc(i) + sample(i);
            break;
          case AccumOp::Min:
            acc(i) = first ? sample(i)
                           : (sample(i) < acc(i) ? sample(i) : acc(i));
            break;
          case AccumOp::Max:
            acc(i) = first ? sample(i)
                           : (sample(i) > acc(i) ? sample(i) : acc(i));
            break;
        }
      });
}

/// `out <- omega * current + (1 - omega) * previous`, one iteration sweep's
/// relaxation.
template <class Space>
void relax(double omega, ConstSpanOf<Space> previous,
           ConstSpanOf<Space> current, SpanOf<Space> out) {
  const double one_minus = 1.0 - omega;
  Kokkos::parallel_for(
      "fuse::relax", Kokkos::RangePolicy<Space>(0, out.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) {
        out(i) = omega * current(i) + one_minus * previous(i);
      });
}

/// `dst[k] <- src[index[k]]` -- packing a message, or reading an interface
/// subset out of a whole-mesh field.
template <class Space>
void gather(ConstSpanOf<Space> src, IndexSpanOf<Space> index,
            SpanOf<Space> dst) {
  Kokkos::parallel_for(
      "fuse::gather", Kokkos::RangePolicy<Space>(0, index.extent(0)),
      KOKKOS_LAMBDA(const std::size_t k) { dst(k) = src(index(k)); });
}

/// `dst[index[k]] <- src[k]` -- unpacking what arrived.
///
/// NOT a scatter-add, and the index list must not repeat: a repeated target
/// would make this a race whose result depends on thread scheduling. The
/// caller is the rank-matching graph, whose target indices are distinct by
/// construction (it refuses a plan where they are not), so the precondition is
/// established where it can be checked rather than re-checked per element in
/// the kernel.
template <class Space>
void scatter(ConstSpanOf<Space> src, IndexSpanOf<Space> index,
             SpanOf<Space> dst) {
  Kokkos::parallel_for(
      "fuse::scatter", Kokkos::RangePolicy<Space>(0, index.extent(0)),
      KOKKOS_LAMBDA(const std::size_t k) { dst(index(k)) = src(k); });
}

/// `y <- W x` for the remap matrix in coordinate form, one row per thread.
///
/// The triplets are ROW-MAJOR SORTED, which `build_supermesh_weights`
/// guarantees, so `row_begin` is a prefix scan over the rows and each thread
/// owns a contiguous span of entries. That is what makes this parallel without
/// atomics: a thread writes exactly one `y(j)` and no other thread touches it.
/// A scatter over entries instead would need an atomic add per entry.
template <class Space>
void remap_apply(IndexSpanOf<Space> row_begin, IndexSpanOf<Space> col,
                 ConstSpanOf<Space> val, ConstSpanOf<Space> x,
                 SpanOf<Space> y) {
  const std::size_t rows = y.extent(0);
  Kokkos::parallel_for(
      "fuse::remap_apply", Kokkos::RangePolicy<Space>(0, rows),
      KOKKOS_LAMBDA(const std::size_t j) {
        double s = 0.0;
        for (std::size_t k = row_begin(j); k < row_begin(j + 1); ++k) {
          s += val(k) * x(col(k));
        }
        y(j) = s;
      });
}

/// `sum_i a_i b_i`. The convergence measure's reduction.
///
/// Returns the space's own summation order, which is NOT the sequential one --
/// a parallel reduction associates differently, so this and the serial oracle
/// agree to rounding and not bitwise. Tests compare with a tolerance scaled by
/// the length, and say so.
template <class Space>
[[nodiscard]] double dot(ConstSpanOf<Space> a, ConstSpanOf<Space> b) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "fuse::dot", Kokkos::RangePolicy<Space>(0, a.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double& acc) { acc += a(i) * b(i); },
      result);
  return result;
}

/// `sum_i w_i v_i^2` -- the measure-weighted squared norm the convergence
/// criterion uses. A separate kernel rather than two passes because the
/// weights are the cell measures and reading them twice is the expensive part.
template <class Space>
[[nodiscard]] double weighted_norm_squared(ConstSpanOf<Space> v,
                                           ConstSpanOf<Space> w) {
  double result = 0.0;
  Kokkos::parallel_reduce(
      "fuse::weighted_norm_squared", Kokkos::RangePolicy<Space>(0, v.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i, double& acc) {
        acc += w(i) * v(i) * v(i);
      },
      result);
  return result;
}

}  // namespace fuse::num
