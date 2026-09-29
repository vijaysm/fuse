// src/partition/collectives.cpp — the only translation unit in
// fuse::partition that names an MPI type. Every function here is declared
// in the public detail/collectives.hpp without one (docs/dev/layering.md:
// mpi.h never appears in a public header).

#include "fuse/partition/detail/collectives.hpp"

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse::partition::detail {

namespace {

#if FUSE_HAVE_MPI
MPI_Comm resolve(CommHandle comm) {
  FUSE_ASSERT(comm != kNullComm,
             "detail::resolve() called with kNullComm; every caller must short-circuit "
             "locally before reaching here");
  return MPI_Comm_f2c(comm);
}
#else
// A non-null handle in a build with no MPI is a caller error the public
// entry points (rcb_partition, centroids_of) already reject with
// E-PART-NO-MPI before any of these functions are called. Reaching here
// with one anyway is this library's own bug, not the caller's -- hence
// FUSE_ASSERT, not another rejection.
void refuse_no_mpi() {
  FUSE_ASSERT(false,
             "fuse::partition::detail collective called with a non-null CommHandle in a "
             "build with FUSE_HAVE_MPI=0; the public entry point should have refused "
             "E-PART-NO-MPI before reaching here");
}
#endif

}  // namespace

int comm_size(CommHandle comm) {
  if (comm == kNullComm) return 1;
#if FUSE_HAVE_MPI
  int size = 1;
  MPI_Comm_size(resolve(comm), &size);
  return size;
#else
  refuse_no_mpi();
  return 1;
#endif
}

int comm_rank(CommHandle comm) {
  if (comm == kNullComm) return 0;
#if FUSE_HAVE_MPI
  int rank = 0;
  MPI_Comm_rank(resolve(comm), &rank);
  return rank;
#else
  refuse_no_mpi();
  return 0;
#endif
}

void allreduce_sum_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                       CommHandle comm) {
  if (comm == kNullComm) {
    for (int i = 0; i < count; ++i) global_out[i] = local[i];
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Allreduce(local, global_out, count, MPI_INT64_T, MPI_SUM, resolve(comm));
#else
  refuse_no_mpi();
#endif
}

void allreduce_min_f64(const double* local, double* global_out, int count, CommHandle comm) {
  if (comm == kNullComm) {
    for (int i = 0; i < count; ++i) global_out[i] = local[i];
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Allreduce(local, global_out, count, MPI_DOUBLE, MPI_MIN, resolve(comm));
#else
  refuse_no_mpi();
#endif
}

void allreduce_max_f64(const double* local, double* global_out, int count, CommHandle comm) {
  if (comm == kNullComm) {
    for (int i = 0; i < count; ++i) global_out[i] = local[i];
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Allreduce(local, global_out, count, MPI_DOUBLE, MPI_MAX, resolve(comm));
#else
  refuse_no_mpi();
#endif
}

void allreduce_max_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                       CommHandle comm) {
  if (comm == kNullComm) {
    for (int i = 0; i < count; ++i) global_out[i] = local[i];
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Allreduce(local, global_out, count, MPI_INT64_T, MPI_MAX, resolve(comm));
#else
  refuse_no_mpi();
#endif
}

void allreduce_min_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                      CommHandle comm) {
  if (comm == kNullComm) {
    for (int i = 0; i < count; ++i) global_out[i] = local[i];
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Allreduce(local, global_out, count, MPI_INT64_T, MPI_MIN, resolve(comm));
#else
  refuse_no_mpi();
#endif
}

void allgatherv_i64(const std::int64_t* local, int local_count,
                    std::vector<std::int64_t>& gathered_out, int& my_offset_out,
                    CommHandle comm) {
  if (comm == kNullComm) {
    gathered_out.assign(local, local + local_count);
    my_offset_out = 0;
    return;
  }
#if FUSE_HAVE_MPI
  MPI_Comm c = resolve(comm);
  int size = 1;
  MPI_Comm_size(c, &size);
  std::vector<int> counts(static_cast<std::size_t>(size), 0);
  MPI_Allgather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, c);
  std::vector<int> displs(static_cast<std::size_t>(size), 0);
  int total = 0;
  int my_rank = 0;
  MPI_Comm_rank(c, &my_rank);
  for (int r = 0; r < size; ++r) {
    displs[static_cast<std::size_t>(r)] = total;
    total += counts[static_cast<std::size_t>(r)];
  }
  my_offset_out = displs[static_cast<std::size_t>(my_rank)];
  gathered_out.assign(static_cast<std::size_t>(total), 0);
  MPI_Allgatherv(local, local_count, MPI_INT64_T, gathered_out.data(), counts.data(),
                displs.data(), MPI_INT64_T, c);
#else
  refuse_no_mpi();
#endif
}

}  // namespace fuse::partition::detail
