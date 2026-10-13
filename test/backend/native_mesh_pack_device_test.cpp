// fuse::native::device::pack_to (include/fuse/backend/native/mesh_pack_device.hpp):
// the Space-templated packing seam docs/dev/device-resident-arcs.md's
// Phase 2 names for native migration and halo. Cross-checked byte-for-byte
// against the EXISTING, already-tested host packers
// (pack_for_migration/pack_halo_cells, both walking NativeMeshView) on the
// SAME generated MeshArrays -- not a separate hand-built fixture -- because
// NativeMeshView::create_from_host preserves cell and vertex index order
// from the MeshArrays it was built from (native_mesh.cpp's own
// construction loop reads arrays.cell_type(cell)/vertex_coords(v,*)
// directly, with no permutation table), so the two code paths describe the
// identical cells in the identical order and a byte-for-byte match is the
// right bar, not merely "equivalent."

#include "fuse/backend/native/mesh_pack_device.hpp"

#include <cstring>
#include <exception>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/backend/native_backend.hpp"  // generate_native_mesh
#include "fuse/backend/native_halo.hpp"
#include "fuse/backend/native_migration.hpp"

namespace {

using namespace fuse::native;  // NOLINT

// Same runtime-skip convention test/runtime/runner_device_test.cpp uses:
// the `.device`-labelled rows compile unconditionally but skip themselves
// at runtime on a build with no Cuda/HIP/SYCL backend, where
// Kokkos::DefaultExecutionSpace::memory_space is HostSpace anyway and the
// row would be a duplicate of the Host-mode one rather than a real check.
constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

template <class Space>
std::vector<std::byte> to_host_bytes(const Kokkos::View<std::byte*, Space>& v) {
  auto h = Kokkos::create_mirror_view(v);
  Kokkos::deep_copy(h, v);
  std::vector<std::byte> out(h.extent(0));
  if (!out.empty()) std::memcpy(out.data(), h.data(), h.extent(0));
  return out;
}

/// `dest_of_owned` splitting the owned cells of `view` round-robin over
/// `num_ranks`, as both a host `std::vector<int>` (for the existing
/// `pack_for_migration`/`pack_halo_cells` entry points) and a
/// `Kokkos::View<int*, Space>` (for `pack_to`) -- the SAME assignment, so
/// the two code paths pack the same cells to the same destinations.
template <class Space>
Kokkos::View<int*, Space> dest_view(const std::vector<int>& dest_host) {
  Kokkos::View<int*, Kokkos::HostSpace> h("dest_h", dest_host.size());
  for (std::size_t i = 0; i < dest_host.size(); ++i) h(i) = dest_host[i];
  Kokkos::View<int*, Space> d(Kokkos::view_alloc("dest_d", Kokkos::WithoutInitializing),
                              dest_host.size());
  Kokkos::deep_copy(d, h);
  return d;
}

/// `to_memory_space<Space>` crossing Host -> a real device (Cuda/HIP/SYCL)
/// space was never exercised before this track (docs/dev/native-mesh.md's
/// own Phase 2 section: "NativeMeshView converts to host at construction
/// and the NativeMesh tests were pinned to host for the CUDA build"). It
/// has a pre-existing defect this track's own device build surfaced for
/// the first time, independent of `pack_to`: `MeshArrays<MemorySpace>`'s
/// multi-dimensional fields (`vertex_coords`, a `double*[3]`) are declared
/// with no explicit layout, so `Kokkos::View`'s PER-SPACE default layout
/// differs (LayoutLeft on Cuda, LayoutRight on Host/OpenMP for a rank >= 2
/// view) between the host source and the device destination, and
/// `Kokkos::deep_copy` across both a different memory space AND a
/// different layout in one call has "no available copy mechanism" --
/// measured as a thrown `Kokkos::Impl::..." exception at runtime, not a
/// compile error. This is `fuse::native::to_memory_space`'s own defect
/// (include/fuse/backend/native_mesh.hpp), not `pack_to`'s, and fixing it
/// (explicit matching layouts, or a mirror-based copy per field) is
/// outside this track's named scope -- see KNOWN_ISSUES.md's
/// weight-builders entry. Skip the Device-mode row with that citation
/// rather than report a `pack_to` failure that isn't one.
template <class Space>
void check_migration_pack_matches_host(const char* generator_spec) {
  auto arrays = generate_native_mesh(generator_spec, /*rank=*/0, /*num_ranks=*/1);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;

  const std::size_t nowned = view.value()->owned_count(view.value()->dimension());
  std::vector<int> dest_host(nowned);
  for (std::size_t i = 0; i < nowned; ++i) dest_host[i] = static_cast<int>(i % 3);

  const auto expected = pack_for_migration(*view.value(), dest_host, /*num_ranks=*/3);
  ASSERT_TRUE(expected.ok()) << expected.error().message;

  // KNOWN_ISSUES.md D54 (fixed): a host -> device copy of the rank-2
  // vertex_coords used to throw on the layout change; the layout is pinned now.
  const MeshArrays<Space> device_mesh = to_memory_space<Space>(arrays.value());
  const auto dest = dest_view<Space>(dest_host);
  auto got = device::pack_to<false, Space>(device_mesh, dest, 3);
  ASSERT_EQ(got.size(), expected.value().size());
  for (std::size_t r = 0; r < got.size(); ++r) {
    const auto got_bytes = to_host_bytes<Space>(got[r]);
    EXPECT_EQ(got_bytes, expected.value()[r]) << "rank " << r << " mismatch for " << generator_spec;
  }
}

template <class Space>
void check_halo_pack_matches_host(const char* generator_spec) {
  auto arrays = generate_native_mesh(generator_spec, 0, 1);
  ASSERT_TRUE(arrays.ok()) << arrays.error().message;
  auto view = NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok()) << view.error().message;
  const int dim = view.value()->dimension();
  const std::size_t nowned = view.value()->owned_count(dim);

  const auto owned_idx_host = owned_block_index(*view.value());  // by LOCAL cell index
  std::vector<std::vector<std::size_t>> send_by_local_index(3);
  std::vector<int> dest_of_owned(nowned);
  for (std::size_t c = 0; c < view.value()->entity_count(dim); ++c) {
    if (!view.value()->is_owned(dim, c)) continue;
    const std::size_t ob = owned_idx_host[c];
    const int r = static_cast<int>(ob % 3);
    dest_of_owned[ob] = r;
    send_by_local_index[static_cast<std::size_t>(r)].push_back(c);
  }
  const auto expected =
      pack_halo_cells(*view.value(), send_by_local_index, owned_idx_host, /*my_rank=*/5, 3);

  // KNOWN_ISSUES.md D54 (fixed): a host -> device copy of the rank-2
  // vertex_coords used to throw on the layout change; the layout is pinned now.
  const MeshArrays<Space> device_mesh = to_memory_space<Space>(arrays.value());
  const auto dest = dest_view<Space>(dest_of_owned);
  std::vector<std::int32_t> owner_rank_host(nowned, 5);
  std::vector<std::int64_t> owned_idx_flat(nowned);
  for (std::size_t i = 0; i < nowned; ++i) owned_idx_flat[i] = static_cast<std::int64_t>(i);
  Kokkos::View<std::int32_t*, Kokkos::HostSpace> or_h("or_h", nowned);
  Kokkos::View<std::int64_t*, Kokkos::HostSpace> oi_h("oi_h", nowned);
  for (std::size_t i = 0; i < nowned; ++i) { or_h(i) = 5; oi_h(i) = static_cast<std::int64_t>(i); }
  Kokkos::View<std::int32_t*, Space> or_d(Kokkos::view_alloc("or_d", Kokkos::WithoutInitializing), nowned);
  Kokkos::View<std::int64_t*, Space> oi_d(Kokkos::view_alloc("oi_d", Kokkos::WithoutInitializing), nowned);
  Kokkos::deep_copy(or_d, or_h);
  Kokkos::deep_copy(oi_d, oi_h);

  auto got = device::pack_to<true, Space>(device_mesh, dest, 3, or_d, oi_d);
  ASSERT_EQ(got.size(), expected.size());
  for (std::size_t r = 0; r < got.size(); ++r) {
    const auto got_bytes = to_host_bytes<Space>(got[r]);
    EXPECT_EQ(got_bytes, expected[r]) << "rank " << r << " mismatch for " << generator_spec;
  }
}

TEST(NativeMeshPackDeviceHost, MigrationPackMatchesPackForMigrationByteForByte_Quad) {
  check_migration_pack_matches_host<Kokkos::HostSpace>("quad:5x4");
}

TEST(NativeMeshPackDeviceHost, MigrationPackMatchesPackForMigrationByteForByte_Polyhedral) {
  check_migration_pack_matches_host<Kokkos::HostSpace>("polyhedral:2x2x2");
}

TEST(NativeMeshPackDeviceHost, HaloPackMatchesPackHaloCellsByteForByte_Quad) {
  check_halo_pack_matches_host<Kokkos::HostSpace>("quad:5x4");
}

TEST(NativeMeshPackDeviceHost, HaloPackMatchesPackHaloCellsByteForByte_Polyhedral) {
  check_halo_pack_matches_host<Kokkos::HostSpace>("polyhedral:2x2x2");
}

using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

TEST(NativeMeshPackDeviceDevice, MigrationPackMatchesPackForMigrationByteForByte_Quad) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_migration_pack_matches_host<DeviceMemSpace>("quad:5x4");
}

TEST(NativeMeshPackDeviceDevice, MigrationPackMatchesPackForMigrationByteForByte_Polyhedral) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_migration_pack_matches_host<DeviceMemSpace>("polyhedral:2x2x2");
}

TEST(NativeMeshPackDeviceDevice, HaloPackMatchesPackHaloCellsByteForByte_Quad) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_halo_pack_matches_host<DeviceMemSpace>("quad:5x4");
}

}  // namespace
