// test/partition/refusals_test.cpp -- deliberate rejections firing
// (docs/dev/testing.md: "counts as verified only when a test makes it
// fire"). Every case here asserts an exception, so this whole target
// carries the THROWS label (see test/CMakeLists.txt).

#include "fuse/partition/rcb.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse_test_main.hpp"

namespace fuse::partition {
namespace {

template <class Space>
Kokkos::View<const double* [3], typename Space::memory_space> centroids_of_size(std::size_t n) {
  Kokkos::View<double* [3], typename Space::memory_space> dev("centroids", n);
  Kokkos::deep_copy(dev, 0.0);
  return dev;
}
template <class Space>
Kokkos::View<const Weight*, typename Space::memory_space> weights_of_size(std::size_t n) {
  Kokkos::View<Weight*, typename Space::memory_space> dev("weights", n);
  Kokkos::deep_copy(dev, Weight{1});
  return dev;
}
template <class Space>
Kokkos::View<const std::int64_t*, typename Space::memory_space> gids_of_size(std::size_t n) {
  Kokkos::View<std::int64_t*, typename Space::memory_space> dev("gids", n);
  Kokkos::deep_copy(dev, std::int64_t{0});
  return dev;
}

using Space = Kokkos::DefaultHostExecutionSpace;

TEST(PartitionRefusals, SizeMismatchBetweenCentroidsAndWeights) {
  auto c = centroids_of_size<Space>(10);
  auto w = weights_of_size<Space>(9);  // deliberately wrong
  auto g = gids_of_size<Space>(10);
  RcbOptions opts;
  opts.num_parts = 2;
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal);
}

TEST(PartitionRefusals, SizeMismatchBetweenCentroidsAndGids) {
  auto c = centroids_of_size<Space>(10);
  auto w = weights_of_size<Space>(10);
  auto g = gids_of_size<Space>(7);  // deliberately wrong
  RcbOptions opts;
  opts.num_parts = 2;
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal);
}

TEST(PartitionRefusals, NegativeNumPartsIsRefused) {
  auto c = centroids_of_size<Space>(10);
  auto w = weights_of_size<Space>(10);
  auto g = gids_of_size<Space>(10);
  RcbOptions opts;
  opts.num_parts = -3;
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal);
}

// A negative weight is not a load-balance metric; without this
// refusal, weighted_median's zero-or-negative-total early return places
// the split at an arbitrary coordinate and silently sorts every tied
// element to the right.
TEST(PartitionRefusals, NegativeWeightIsRefused) {
  auto c = centroids_of_size<Space>(10);
  Kokkos::View<Weight*, typename Space::memory_space> w_dev("weights", 10);
  Kokkos::deep_copy(w_dev, Weight{1});
  auto w_host = Kokkos::create_mirror_view(w_dev);
  Kokkos::deep_copy(w_host, w_dev);
  w_host(3) = Weight{-1};
  Kokkos::deep_copy(w_dev, w_host);
  Kokkos::View<const Weight*, typename Space::memory_space> w = w_dev;
  auto g = gids_of_size<Space>(10);
  RcbOptions opts;
  opts.num_parts = 2;
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal);
}

#if !FUSE_HAVE_MPI
// This build has no MPI at all (the base build configuration) -- a non-null communicator handle
// can only be a caller error, and must be rejected before anything tries to
// resolve it. In an MPI-enabled build this branch does not compile-in at
// all: the corresponding behaviour there is exercised on the host, by
// mpiexec actually disagreeing (see rcb_mpi_test.cpp).
TEST(PartitionRefusals, NonNullCommWithNoMpiBuildIsRefused) {
  auto c = centroids_of_size<Space>(10);
  auto w = weights_of_size<Space>(10);
  auto g = gids_of_size<Space>(10);
  RcbOptions opts;
  opts.num_parts = 2;
  opts.comm = 42;  // any non-zero handle
  EXPECT_THROW((void)(rcb_partition<Space>(opts, c, w, g)), fuse::Refusal);
}
#endif

}  // namespace
}  // namespace fuse::partition
