#pragma once

/// The cross-rank sum of the conservation integrals of an Arithmetic field
/// (double-double, quad-double), and of a binary64 field summed in double-double
/// (RunnerConfig::accumulation_type). plans/2026-10-22-universal-number-types.md,
/// stage U4. Internal to the runtime; a header so that the MPI test can drive it
/// without a Runner (test/parallel/arithmetic_reduce_mpi_test.cpp).
///
/// WHY NOT A USER-DEFINED MPI_Op. Addition in dd or qd is not associative, so an
/// MPI_Allreduce with such an operator gives a result that depends on the shape
/// of the reduction tree the library picks, and the ranks need not agree to the
/// last bit. Each rank's integrals instead travel as limbs in one MPI_Allgather
/// and every rank adds them in ascending rank order (an ordered, reproducible
/// reduction): the totals are the same bits on every rank by construction, and
/// the same for a given number of ranks whatever the MPI library. The volume is
/// 2 * limbs + 1 doubles per rank, once per conservation check.

#include <cstdint>
#include <string>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/result.hpp"
#include "fuse/num/arithmetic_kernels.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::detail {

/// Replaces `before` and `after` by their sums over the ranks of the communicator
/// (`communicator_f`: the Fortran handle, as RunnerConfig::base_communicator
/// carries it), and sets `partner_lost` when any rank passed `step_failed`: the
/// buffers of a rank whose period has already failed are whatever the abandoned
/// step left, so the totals are then not to be used. `gather` is the receive
/// buffer, resized (it only grows). Collective over the communicator. `arc` names
/// the arc in the error message.
template <class A>
Result<void> reduce_arithmetic_integrals(int communicator_f, bool step_failed,
                                         std::vector<double>& gather, std::uint32_t arc, A& before,
                                         A& after, bool& partner_lost) {
  partner_lost = false;
#if FUSE_HAVE_MPI
  constexpr int L = num::limb_count<A>;
  constexpr int N = 2 * L + 1;  // the limbs of before, the limbs of after, the failure flag
  double local[N] = {};
  num::detail::Limbs<A>::store(local, before);
  num::detail::Limbs<A>::store(local + L, after);
  local[2 * L] = step_failed ? 1.0 : 0.0;
  const MPI_Comm base = MPI_Comm_f2c(static_cast<MPI_Fint>(communicator_f));
  int size = 0;
  if (MPI_Comm_size(base, &size) != MPI_SUCCESS) {
    return Failure{"E-RT-CONSERVATION-REDUCE",
                   "MPI_Comm_size of the base communicator failed for the conservation "
                   "integrals of arc %" + std::to_string(arc)};
  }
  gather.resize(static_cast<std::size_t>(size) * N);
  if (MPI_Allgather(local, N, MPI_DOUBLE, gather.data(), N, MPI_DOUBLE, base) != MPI_SUCCESS) {
    return Failure{"E-RT-CONSERVATION-REDUCE",
                   "MPI_Allgather of the conservation integrals for arc %" +
                       std::to_string(arc) + " failed"};
  }
  A total_before(0.0), total_after(0.0);
  double flag = 0.0;
  for (int p = 0; p < size; ++p) {
    const double* g = gather.data() + static_cast<std::size_t>(p) * N;
    total_before = total_before + num::detail::Limbs<A>::load(g);
    total_after = total_after + num::detail::Limbs<A>::load(g + L);
    flag += g[2 * L];
  }
  before = total_before;
  after = total_after;
  partner_lost = flag > 0.0;
  return Result<void>::success();
#else
  (void)communicator_f; (void)step_failed; (void)gather; (void)arc; (void)before; (void)after;
  return Failure{"E-RT-CONSERVATION-REDUCE",
                 "a base communicator is configured but this build has no MPI"};
#endif
}

}  // namespace fuse::detail
