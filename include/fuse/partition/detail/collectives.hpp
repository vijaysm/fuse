#pragma once

#include <cstdint>
#include <vector>

#include "fuse/partition/types.hpp"

/// Implementation detail of fuse::partition: the collective operations
/// rcb_partition's recursion needs, isolated behind functions that take no
/// MPI type -- `mpi.h` appears only in src/partition/collectives.cpp, never
/// in a header, so this layer's public headers stay clean of it (the same
/// rule tools/check-layering.py enforces for every other layer).
///
/// EVERY FUNCTION HERE IS COLLECTIVE over `comm` when `comm != kNullComm`:
/// every rank of that communicator must call it in lockstep. When
/// `comm == kNullComm` each one is a local no-op / identity (there is
/// exactly one participant, so the caller's local value already is the
/// global answer) and touches no MPI symbol -- what keeps fuse::partition
/// callable from a Tier 0 (no-MPI) build. rcb_moab's CLAUDE.md "Global
/// collective discipline" applies here exactly as it does there: a caller
/// must never skip one of these based on a locally-computed condition (an
/// empty local subset is not a reason to skip a reduction every other rank
/// is still going to call).
namespace fuse::partition::detail {

/// Ranks in `comm` (1 for kNullComm).
int comm_size(CommHandle comm);
/// This rank's position in `comm` (0 for kNullComm).
int comm_rank(CommHandle comm);

/// out[i] = sum over ranks of local[i], for i in [0, count). Weight sums are
/// associative in int64, so this is bit-exact regardless of comm size.
void allreduce_sum_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                       CommHandle comm);

/// out[i] = min over ranks of local[i].
void allreduce_min_f64(const double* local, double* global_out, int count, CommHandle comm);
/// out[i] = max over ranks of local[i].
void allreduce_max_f64(const double* local, double* global_out, int count, CommHandle comm);
/// out[i] = max over ranks of local[i] (int64 -- used for small flags/dims,
/// not weights).
void allreduce_max_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                       CommHandle comm);
/// out[i] = min over ranks of local[i] (int64). Paired with
/// `allreduce_max_i64` to detect cross-rank disagreement: max == min means
/// every rank holds the same value.
void allreduce_min_i64(const std::int64_t* local, std::int64_t* global_out, int count,
                       CommHandle comm);

/// Concatenates every rank's `local[0..local_count)` into `gathered_out`, in
/// ascending rank order -- MPI_Allgatherv's exact layout. `my_offset_out` is
/// set to this rank's starting position in `gathered_out` (the sum of every
/// lower-ranked `local_count`), which the caller needs to map a gathered
/// index back to "one of MINE".
void allgatherv_i64(const std::int64_t* local, int local_count,
                    std::vector<std::int64_t>& gathered_out, int& my_offset_out, CommHandle comm);

}  // namespace fuse::partition::detail
