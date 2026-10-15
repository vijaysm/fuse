// test/partition/repartition_mpi_test.cpp -- fuse::partition::repartition
// under a REAL mpiexec.
//
// NOT YET RUN: this sandbox cannot initialize MPI, exactly the posture
// rcb_mpi_test.cpp documents for itself; running it is a host action.
//
// WHAT THIS CHECKS THAT repartition_test.cpp CANNOT: PS-6's
// num_parts-less-than-comm_size defect can only be reproduced with MORE
// THAN ONE real rank -- at kNullComm, comm_size is always 1, so num_parts
// can never be strictly less than comm_size without also being the
// already-refused num_parts <= 0 case.

#include "fuse/partition/repartition.hpp"

#include <mpi.h>

#include <gtest/gtest.h>

#include "fuse/backend/native_backend.hpp"
#include "fuse_test_main.hpp"

namespace fuse::partition {
namespace {

TEST(RepartitionMpi, RefusesNumPartsLessThanCommSize) {
  int rank = 0, size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  FUSE_REQUIRE_AT_LEAST_RANKS(2);

  auto arrays = fuse::native::generate_native_mesh("hexbox:2x2x2", rank, size);
  ASSERT_TRUE(arrays.ok());
  auto view = fuse::native::NativeMeshView::create_from_host(arrays.value(), rank);
  ASSERT_TRUE(view.ok());

  RepartitionOptions opts;
  opts.comm = MPI_Comm_c2f(MPI_COMM_WORLD);
  opts.rcb.num_parts = size - 1;  // deliberately fewer parts than ranks (>= 1 since size >= 2)
  auto result = repartition(*view.value(), opts);
  ASSERT_FALSE(result.ok()) << "num_parts < comm_size must be refused, not silently leave a "
                               "rank with an empty mesh";
  EXPECT_EQ(result.error().code, "E-PART-REPARTITION-NUM-PARTS");
}

}  // namespace
}  // namespace fuse::partition
