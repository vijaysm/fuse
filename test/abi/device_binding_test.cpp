// Rank-to-device binding (docs/dev/device-resident-arcs.md):
// `fuse::abi::resolve_device_id()` is pure arithmetic, tested here with a
// mocked device count as well as the real one -- a single-GPU host
// has exactly one GPU, so `local_rank % device_count > 1` is reachable only
// through the mock; the real `resolve_device_id_for_this_process()` glue in
// src/abi/participant.cpp and test/support/fuse_test_main.cpp is not
// re-tested here (it calls MPI and Kokkos, neither mockable from a plain
// GoogleTest row), only the rule it is built on.
#include "fuse/abi/device_binding.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace {

using fuse::abi::resolve_device_id;

TEST(DeviceBinding, ExplicitOverrideWinsRegardlessOfRankOrCount) {
  EXPECT_EQ(resolve_device_id(0, 1, 7), 7);
  EXPECT_EQ(resolve_device_id(3, 4, 0), 0);
  // Even a count of zero (no device backend at all) does not suppress an
  // explicit override: the caller asked for a specific id, and this
  // function's job is the arithmetic, not second-guessing the request.
  EXPECT_EQ(resolve_device_id(0, 0, 2), 2);
}

TEST(DeviceBinding, LocalRankModuloDeviceCountOnTheRealOneGpuHost) {
  // Single-GPU host: every rank binds to device 0, the real device_count=1 case.
  EXPECT_EQ(resolve_device_id(0, 1, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(1, 1, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(4, 1, std::nullopt), 0);
}

TEST(DeviceBinding, LocalRankModuloDeviceCountWithAMockedMultiGpuCount) {
  // No machine in this build's reach has more than one GPU; this is the
  // only coverage the distribution rule (not just "always 0") gets.
  EXPECT_EQ(resolve_device_id(0, 4, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(1, 4, std::nullopt), 1);
  EXPECT_EQ(resolve_device_id(3, 4, std::nullopt), 3);
  EXPECT_EQ(resolve_device_id(4, 4, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(5, 4, std::nullopt), 1);
  // Wrapping past several full cycles, the way a node with many ranks
  // sharing a handful of GPUs would.
  EXPECT_EQ(resolve_device_id(13, 4, std::nullopt), 1);
}

TEST(DeviceBinding, NoDeviceBackendReturnsZeroRatherThanDividingByZero) {
  EXPECT_EQ(resolve_device_id(0, 0, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(5, 0, std::nullopt), 0);
  EXPECT_EQ(resolve_device_id(5, -1, std::nullopt), 0);
}

TEST(DeviceBinding, NegativeLocalRankIsTreatedAsZero) {
  // Not a reachable input from the real glue (a local rank from
  // MPI_Comm_split_type is never negative), but the function's own
  // contract should not propagate a negative index into a modulo.
  EXPECT_EQ(resolve_device_id(-1, 4, std::nullopt), 0);
}

}  // namespace
