// DeviceGhostExchanger
// (include/fuse/backend/device_ghost_exchange.hpp) drives the SAME
// fuse::linalg::GhostExchanger pattern GhostExchanger::update() does, but
// over Space-resident buffers. A ring pattern -- rank r's 4 ghosts are
// rank (r+1)%w's own 4 owned entries -- exercises a genuine cross-rank
// exchange at every world size this file registers at (1, 2, 3; at w==1
// the "next" rank wraps to self, a legitimate self-to-self Alltoallv the
// underlying MpiExchanger already supports). Values are `100*rank + i`,
// so a wrong rank or a wrong local index is distinguishable from a
// correct delivery -- a constant would not be.
//
// Needs a working MPI launcher (docs/dev/testing.md).

#include <fuse/backend/device_ghost_exchange.hpp>
#include <fuse/linalg/comm.hpp>

#include <cstdlib>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>
#include <mpi.h>

#include "fuse_test_main.hpp"

namespace {

int world_size() {
  int n = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &n);
  return n;
}
int world_rank() {
  int r = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &r);
  return r;
}

constexpr std::size_t kOwnedN = 4;

[[nodiscard]] std::shared_ptr<const fuse::linalg::GhostExchanger> build_ring_exchanger() {
  const int w = world_size();
  const int r = world_rank();
  const int next = (r + 1) % w;
  fuse::linalg::GhostPattern pattern;
  pattern.owner_rank.assign(kOwnedN, next);
  pattern.remote_local_index.assign(kOwnedN, 0);
  for (std::size_t k = 0; k < kOwnedN; ++k) {
    pattern.remote_local_index[k] = static_cast<std::int64_t>(k);
  }
  return fuse::linalg::make_mpi_exchanger(MPI_Comm_c2f(MPI_COMM_WORLD), pattern, kOwnedN);
}

// `FUSE_EXECUTION_SPACE=device` (the DEVICE_TWIN re-registration,
// test/CMakeLists.txt's own fuse_add_mpi_test) selects the Device
// instantiation below; its absence (the plain `.N` row) selects Host --
// the same env var Runner::initialize() itself reads
// (src/runtime/runner.cpp), read here directly since this test drives
// DeviceGhostExchanger without going through a Runner at all.
bool want_device_space() {
  const char* env = std::getenv("FUSE_EXECUTION_SPACE");
  return env != nullptr && std::string_view{env} == "device";
}

template <class Space>
void run_ring_check(bool force_host_staging, bool expect_device_direct_possible) {
  if (force_host_staging) {
    ::setenv("FUSE_TRANSPORT_HOST_STAGING", "1", /*overwrite=*/1);
  } else {
    ::unsetenv("FUSE_TRANSPORT_HOST_STAGING");
  }

  const auto exchanger = build_ring_exchanger();
  ASSERT_NE(exchanger, nullptr);
  fuse::backend::DeviceGhostExchanger dge(exchanger);

  const int w = world_size();
  const int r = world_rank();
  const int next = (r + 1) % w;

  std::vector<double> owned_host(kOwnedN);
  for (std::size_t i = 0; i < kOwnedN; ++i) owned_host[i] = 100.0 * r + static_cast<double>(i);
  std::vector<double> ghost_host(kOwnedN, -1.0);

  Kokkos::View<double*, Space> owned_dev(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "owned_dev"), kOwnedN);
  Kokkos::View<double*, Space> ghost_dev(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ghost_dev"), kOwnedN);
  auto owned_dev_host_mirror = Kokkos::create_mirror_view(owned_dev);
  for (std::size_t i = 0; i < kOwnedN; ++i) owned_dev_host_mirror(i) = owned_host[i];
  Kokkos::deep_copy(owned_dev, owned_dev_host_mirror);

  const auto result =
      dge.template update<Space>(owned_dev.data(), kOwnedN, ghost_dev.data(), kOwnedN);
  ASSERT_TRUE(result.ok()) << result.error().code << ": " << result.error().message;

  auto ghost_dev_host_mirror = Kokkos::create_mirror_view(ghost_dev);
  Kokkos::deep_copy(ghost_dev_host_mirror, ghost_dev);
  for (std::size_t i = 0; i < kOwnedN; ++i) ghost_host[i] = ghost_dev_host_mirror(i);

  for (std::size_t i = 0; i < kOwnedN; ++i) {
    const double expected = 100.0 * next + static_cast<double>(i);
    // Pure data movement, no floating-point arithmetic: bitwise, not
    // EXPECT_NEAR, exactly like TheHostStagedFallbackDeliversTheSameValues
    // EvenWhenForced's own rationale (test/parallel/runner_transport_test.cpp).
    EXPECT_EQ(ghost_host[i], expected) << "rank " << r << " ghost " << i;
  }

  // The device-direct path is reachable only under Device space, with no
  // forced host staging, and a genuinely GPU-aware MPI. With an MPI built
  // without CUDA `mpi_is_gpu_aware()` is false and only the else-branch
  // below runs; with a CUDA-aware MPI the probe is true and the TRUE branch
  // is the one measured by the .device twins of this row -- the same test
  // exercises whichever branch the host's MPI makes reachable, never
  // skipping silently.
  if (expect_device_direct_possible && fuse::linalg::mpi_is_gpu_aware() && !force_host_staging) {
    EXPECT_TRUE(dge.last_update_took_device_direct_path())
        << "GPU-aware MPI did not take the device-direct exchange path";
  } else {
    EXPECT_FALSE(dge.last_update_took_device_direct_path());
  }
}

}  // namespace

TEST(DeviceGhostExchangeMpi, RingDeliversTheOwningRanksValues) {
  using HostSpace = Kokkos::DefaultHostExecutionSpace;
  using DeviceSpace = Kokkos::DefaultExecutionSpace;
  if (want_device_space()) {
    run_ring_check<DeviceSpace>(/*force_host_staging=*/false, /*expect_device_direct_possible=*/true);
  } else {
    run_ring_check<HostSpace>(/*force_host_staging=*/false, /*expect_device_direct_possible=*/false);
  }
}

// The host-staged fallback, forced via FUSE_TRANSPORT_HOST_STAGING=1 --
// mirrors runner_transport_test.cpp's own
// TheHostStagedFallbackDeliversTheSameValuesEvenWhenForced: delivery must
// still agree bitwise, and the device-direct path must not have been
// taken.
TEST(DeviceGhostExchangeMpi, TheHostStagedFallbackDeliversTheSameValuesEvenWhenForced) {
  using HostSpace = Kokkos::DefaultHostExecutionSpace;
  using DeviceSpace = Kokkos::DefaultExecutionSpace;
  if (want_device_space()) {
    run_ring_check<DeviceSpace>(/*force_host_staging=*/true, /*expect_device_direct_possible=*/false);
  } else {
    run_ring_check<HostSpace>(/*force_host_staging=*/true, /*expect_device_direct_possible=*/false);
  }
  ::unsetenv("FUSE_TRANSPORT_HOST_STAGING");
}
