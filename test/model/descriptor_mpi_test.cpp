// The last condition on the waist freeze: every rank must serialize the same
// graph to the same bytes.
//
// This is not pedantry. The descriptor is how a participant states what it is
// coupling, and a graph is authored on every rank. If two ranks produce
// different bytes -- from iteration order over an unordered container, from a
// pointer value reaching the buffer, from a locale-dependent number format --
// then the compiler proves agreement about different graphs on different ranks
// and the run is wrong in a way no single-rank test can see. A single-process
// determinism test (`Descriptor.SerializationIsDeterministic`) cannot catch it.
//
// NOT YET RUN: this sandbox cannot initialize MPI, so this test is registered
// only in a Tier 1 build and has no result here. Running it is a host action.
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse/base/bytes.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/freeze_fixture.hpp"

namespace {

TEST(DescriptorMpi, EveryRankSerializesTheSameBytes) {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto bytes = fuse::serialize(fuse::testing::freeze_fixture());

  // Compare the LENGTH first: a differing length would make the byte compare
  // below read past a shorter buffer on some rank.
  const auto local_size = static_cast<long long>(bytes.size());
  long long min_size = 0;
  long long max_size = 0;
  MPI_Allreduce(&local_size, &min_size, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_size, &max_size, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(min_size, max_size) << "ranks produced descriptors of different lengths";

  // Then the content, by hash, so the message stays small on a large run; a
  // mismatch is then localized by broadcasting rank 0's buffer and diffing.
  const auto local_hash = static_cast<long long>(fuse::bytes::fnv1a(bytes));
  long long min_hash = 0;
  long long max_hash = 0;
  MPI_Allreduce(&local_hash, &min_hash, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_hash, &max_hash, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  ASSERT_EQ(min_hash, max_hash) << "ranks produced different descriptor bytes";

  std::vector<std::byte> reference = bytes;
  MPI_Bcast(reference.data(), static_cast<int>(reference.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
  EXPECT_EQ(reference, bytes) << "rank " << rank << " of " << size
                              << " differs from rank 0's descriptor";
}

}  // namespace
