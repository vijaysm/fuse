#pragma once

/// The per-step kernels, parameterised on a Kokkos execution space.
///
/// WHICH KERNELS, AND WHY THESE. These are the operations a coupled step
/// actually spends its time in, in the order they run:
///
///   accumulate  once per subcycling step, on every coupled field -- the most
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
///
/// LABEL LENGTH IS NOT COSMETIC for a kernel the Runner's own hot loop calls
/// every coupling cycle. `Kokkos::parallel_for`/`parallel_reduce` copy their label
/// into a `std::string`, and a label past the standard library's
/// small-string-optimisation capacity (measured: 22 characters on this
/// tree's libc++) heap-allocates one -- per call, every coupling cycle, forever.
/// `runtime.iteration`'s allocation-counted tests
/// (`Iteration.TheSweepLoopAllocatesNothing`/
/// `TheAcceleratorAllocatesNothingPerUpdate`, test/verification/
/// iteration_test.cpp) is exactly what caught this: the first cut of the
/// `deterministic_*` kernels below used long, self-describing labels
/// (`"fuse::deterministic_dot_partial"`, 31 characters) and both tests
/// regressed from 0 allocations to hundreds. Every label a Runner-owned code
/// path can reach stays under that budget; a kernel meant only for a test
/// fixture's own direct calls (nothing in `Impl::execute()`/`run_region()`
/// reaches `dot`/`weighted_norm_squared` above, for instance) is not held to
/// it, but inherits the same convention anyway rather than becoming the
/// exception a later reader has to rediscover why it's there.

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

/// `acc <- op(acc, sample)`, the per-subcycling step time accumulation.
///
/// `first` is passed rather than derived from a subcycling step counter inside the
/// kernel: the branch is then uniform across the range instead of per element,
/// and the caller already knows which subcycling step it is on.
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
/// construction (it rejects a plan where they are not), so the precondition is
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

// ---------------------------------------------------------------------------
// KNOWN_ISSUES.md D11 (closing): additive kernels for the coupling hot loop.
//
// `copy`, `scale` and `residual` below are elementwise -- each output entry
// depends on nothing any other thread writes, so `Kokkos::parallel_for`'s
// partitioning of the range across threads cannot change the answer. They
// need no determinism argument beyond the one `accumulate`/`relax` above
// already carry.
//
// `remap_apply`'s second overload below (with an explicit `components`) is
// the same argument as the single-component one above, generalised: each
// thread still owns one row `j` and writes only `y(j*components + c)` for
// `c` in `[0, components)`, in the SAME triplet order (ascending `k` within
// the row) `apply_weights` (fuse/backend/remap_plan.hpp) processes its COO
// scan in -- so a caller building the CSR ONCE, at plan time, from a
// `RemapWeights`' already row-major-sorted triplets gets the identical
// per-row sum `apply_weights` would have produced, regardless of how many
// threads ran the row loop.
//
// The three `deterministic_*` kernels at the bottom are the exception to
// "elementwise needs no argument": they are REDUCTIONS, and `dot`/
// `weighted_norm_squared` above are explicit that a plain
// `Kokkos::parallel_reduce`'s summation order is the THREADS', not a fixed
// one -- exactly what a runtime whose bit-for-bit reproducibility across
// thread counts is a requirement (not a nicety) cannot use. See their own
// comment for the fixed-chunk scheme and what it does and does not promise.

/// `dst[i] <- src[i]`. The elementwise copy every `num::copy` call site in
/// the Runner (`fuse/num/kernels.hpp`) moves to when the buffer is a
/// component-owned array wrapped as a `SpanOf`/`ConstSpanOf` rather than a
/// raw pointer no space has been chosen for.
template <class Space>
void copy(ConstSpanOf<Space> src, SpanOf<Space> dst) {
  Kokkos::parallel_for(
      "fuse::copy", Kokkos::RangePolicy<Space>(0, dst.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { dst(i) = src(i); });
}

/// `dst[i] <- factor * src[i]`. `OpKind::Scale`'s whole operator.
template <class Space>
void scale(double factor, ConstSpanOf<Space> src, SpanOf<Space> dst) {
  Kokkos::parallel_for(
      "fuse::scale", Kokkos::RangePolicy<Space>(0, dst.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { dst(i) = factor * src(i); });
}

/// `out[i] <- a[i] - b[i]`. `OpKind::Residual`'s two shapes (`a - b` and
/// `a - prev`) are the same subtraction against a different second operand,
/// so one kernel covers both; the caller decides what `b` aliases.
template <class Space>
void residual(ConstSpanOf<Space> a, ConstSpanOf<Space> b, SpanOf<Space> out) {
  Kokkos::parallel_for(
      "fuse::residual", Kokkos::RangePolicy<Space>(0, out.extent(0)),
      KOKKOS_LAMBDA(const std::size_t i) { out(i) = a(i) - b(i); });
}

/// The multi-component sibling of `remap_apply` above: `y <- W x` with `x`
/// `w.n_cols * components` doubles and `y` `w.n_rows * components`, both
/// entity-major with the component index fastest -- `apply_weights`'s own
/// layout (`fuse/backend/remap_plan.hpp`). `row_begin` has `rows + 1`
/// entries (a plain CSR prefix scan, not derivable from `y` alone once
/// `components != 1`, which is why this overload takes it explicitly rather
/// than reusing the single-component one's `y.extent(0)` row count).
template <class Space>
void remap_apply(IndexSpanOf<Space> row_begin, IndexSpanOf<Space> col,
                 ConstSpanOf<Space> val, ConstSpanOf<Space> x,
                 SpanOf<Space> y, std::size_t components) {
  const std::size_t rows = row_begin.extent(0) - 1;
  Kokkos::parallel_for(
      "fuse::remap_multi", Kokkos::RangePolicy<Space>(0, rows),
      KOKKOS_LAMBDA(const std::size_t j) {
        const std::size_t base = j * components;
        for (std::size_t c = 0; c < components; ++c) y(base + c) = 0.0;
        // Deliberately not textually identical to the single-component
        // remap_apply's own row loop above (same bounds, `row_end` hoisted
        // out of the loop instead of re-read each iteration): the two are
        // separate mutate.py anchors in build-evidence/mutations.toml, and
        // an anchor that occurs more than once is one tools/mutate.py
        // --check-anchors reports as STALE for the whole manifest, not just
        // this row.
        const std::size_t row_end = row_begin(j + 1);
        for (std::size_t k = row_begin(j); k < row_end; ++k) {
          const std::size_t cbase = col(k) * components;
          const double v = val(k);
          for (std::size_t c = 0; c < components; ++c) {
            y(base + c) += v * x(cbase + c);
          }
        }
      });
}

/// Fixed-order reductions whose answer does not depend on the execution
/// space's thread count -- unlike `dot`/`weighted_norm_squared` above, whose
/// own comments say plainly that a parallel reduction's summation order is
/// the threads'. KNOWN_ISSUES.md D11: the runtime's OWN reductions
/// (`iteration.cpp`'s block residual/value norms) need bit-for-bit
/// reproducibility across thread counts, which those two cannot give.
///
/// THE SCHEME. `kDeterministicChunk` (a compile-time constant, independent
/// of `Space` and of the thread count) partitions `[0, n)` into
/// `deterministic_chunks(n)` fixed-size groups. `Kokkos::parallel_for` runs
/// over CHUNKS, not elements: each chunk sums its own span sequentially, in
/// ascending index order, into one scratch partial -- the number of
/// independent units of work is fixed by `n` alone, never by how many
/// threads happen to be available. The partials are then combined
/// SEQUENTIALLY, in ascending chunk order, directly out of `scratch`: this
/// second pass, not a second `parallel_reduce`, is what makes the total
/// independent of the first pass's thread partitioning.
///
/// NOT the same bits as `fuse::num::sum` (`kernels.hpp`)'s single-accumulator
/// ascending sum in general: grouping `n` terms into
/// `deterministic_chunks(n)` partials re-associates the moment
/// `n > kDeterministicChunk`, and floating-point addition is not
/// associative. For `n <= kDeterministicChunk` there is exactly one chunk,
/// so the grouped sum and the plain ascending sum are the identical
/// computation and agree bit for bit; a caller whose buffers are all that
/// small (as the runtime's own implicit-coupling-group blocks are, in every
/// existing base-build-configuration fixture) sees no change at all from moving onto this
/// scheme. A caller with genuinely larger `n` gets thread-count
/// independence, not bitwise agreement with a pre-existing single-
/// accumulator reference -- the same trade `dot`/`weighted_norm_squared`
/// already document, just resolved differently.
///
/// `scratch` must hold at least `deterministic_chunks(v.extent(0))` entries;
/// callers size it once, at setup, and reuse it (D11's no-per-coupling-cycle-
/// allocation rule; `iteration.cpp` sizes it from the largest implicit-coupling-group block).
///
/// HOST-ONLY. The finalising sum reads `scratch` with plain host code, so
/// `Space::memory_space` must be host-accessible -- true of every space this
/// tree can build (Serial, Threads, OpenMP). A device space would need a
/// `deep_copy` of the partials to a host mirror before that sum, which nothing
/// here does: closing THAT gap is A5's device work, not this one's.
inline constexpr std::size_t kDeterministicChunk = 256;

[[nodiscard]] inline std::size_t deterministic_chunks(std::size_t n) {
  return (n + kDeterministicChunk - 1) / kDeterministicChunk;
}

/// `sum_i a_i b_i`, fixed-order. `deterministic_norm_squared` below is
/// `deterministic_dot(v, v, scratch)`.
template <class Space>
[[nodiscard]] double deterministic_dot(ConstSpanOf<Space> a,
                                       ConstSpanOf<Space> b,
                                       SpanOf<Space> scratch) {
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 typename Space::memory_space>::accessible,
      "deterministic_dot's fixed-order finalize reads `scratch` directly "
      "from host code; a device memory space needs a deep_copy first "
      "(KNOWN_ISSUES.md D11, remaining work under A5)");
  const std::size_t n = a.extent(0);
  const std::size_t n_chunks = deterministic_chunks(n);
  Kokkos::parallel_for(
      "fuse::det_dot_part", Kokkos::RangePolicy<Space>(0, n_chunks),
      KOKKOS_LAMBDA(const std::size_t c) {
        const std::size_t begin = c * kDeterministicChunk;
        const std::size_t end =
            (begin + kDeterministicChunk < n) ? begin + kDeterministicChunk : n;
        double s = 0.0;
        for (std::size_t i = begin; i < end; ++i) s += a(i) * b(i);
        scratch(c) = s;
      });
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) total += scratch(c);
  return total;
}

/// `sum_i v_i`, fixed-order. See `deterministic_dot`'s comment for the scheme
/// and what it does and does not promise.
template <class Space>
[[nodiscard]] double deterministic_sum(ConstSpanOf<Space> v,
                                       SpanOf<Space> scratch) {
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 typename Space::memory_space>::accessible,
      "deterministic_sum's fixed-order finalize reads `scratch` directly "
      "from host code; a device memory space needs a deep_copy first "
      "(KNOWN_ISSUES.md D11, remaining work under A5)");
  const std::size_t n = v.extent(0);
  const std::size_t n_chunks = deterministic_chunks(n);
  Kokkos::parallel_for(
      "fuse::det_sum_part", Kokkos::RangePolicy<Space>(0, n_chunks),
      KOKKOS_LAMBDA(const std::size_t c) {
        const std::size_t begin = c * kDeterministicChunk;
        const std::size_t end =
            (begin + kDeterministicChunk < n) ? begin + kDeterministicChunk : n;
        double s = 0.0;
        for (std::size_t i = begin; i < end; ++i) s += v(i);
        scratch(c) = s;
      });
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) total += scratch(c);
  return total;
}

/// `sum_i v_i^2`, fixed-order. `deterministic_dot(v, v, scratch)` exactly:
/// a separate name because this is the one `iteration.cpp`'s block value
/// norm calls, and `weighted_norm_squared`'s own precedent above (a distinct
/// name for `dot` specialised to one argument doubled) is the convention
/// this follows.
template <class Space>
[[nodiscard]] double deterministic_norm_squared(ConstSpanOf<Space> v,
                                                SpanOf<Space> scratch) {
  return deterministic_dot<Space>(v, v, scratch);
}

/// `sum_i (a_i - b_i)^2`, fixed-order. `iteration.cpp`'s block RESIDUAL norm:
/// avoids materialising `a - b` into a third buffer (and the allocation, or
/// third scratch array, that would cost) by folding the subtraction into the
/// same per-chunk loop `deterministic_dot` uses.
template <class Space>
[[nodiscard]] double deterministic_norm_squared_diff(ConstSpanOf<Space> a,
                                                      ConstSpanOf<Space> b,
                                                      SpanOf<Space> scratch) {
  static_assert(
      Kokkos::SpaceAccessibility<Kokkos::HostSpace,
                                 typename Space::memory_space>::accessible,
      "deterministic_norm_squared_diff's fixed-order finalize reads "
      "`scratch` directly from host code; a device memory space needs a "
      "deep_copy first (KNOWN_ISSUES.md D11, remaining work under A5)");
  const std::size_t n = a.extent(0);
  const std::size_t n_chunks = deterministic_chunks(n);
  Kokkos::parallel_for(
      "fuse::det_nsd_part",
      Kokkos::RangePolicy<Space>(0, n_chunks),
      KOKKOS_LAMBDA(const std::size_t c) {
        const std::size_t begin = c * kDeterministicChunk;
        const std::size_t end =
            (begin + kDeterministicChunk < n) ? begin + kDeterministicChunk : n;
        double s = 0.0;
        for (std::size_t i = begin; i < end; ++i) {
          const double d = a(i) - b(i);
          s += d * d;
        }
        scratch(c) = s;
      });
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) total += scratch(c);
  return total;
}

}  // namespace fuse::num
