// fuse::native::device::dedup_by_gid (KNOWN_ISSUES D51): the receive-side
// merge decision -- which incoming global ids are already held, and which
// ONE incoming copy of a genuinely new one is canonical -- as a Kokkos
// kernel on Space. Cross-checked against a from-scratch reference built
// with std::unordered_map/std::unordered_set (the same primitives
// unpack_migrated_mesh/merge_halo_cells use on the Host path), not against
// those functions themselves: this header is a reusable primitive, tested
// at its own level (see this header's own "SLOT NUMBERING" comment for
// what wiring it into those functions still needs).

#include "fuse/backend/native/mesh_merge_device.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::native::device::dedup_by_gid;
using fuse::native::device::DedupByGidResult;

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

template <class Space>
Kokkos::View<const std::int64_t*, Space> to_device_const(const std::vector<std::int64_t>& v,
                                                         const std::string& label) {
  Kokkos::View<std::int64_t*, Space> d(label, v.size());
  auto h = Kokkos::create_mirror_view(d);
  for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
  Kokkos::deep_copy(d, h);
  return d;
}

/// A reference, host-only dedup over the SAME two input arrays: existing
/// gids are assumed duplicate-free; for each incoming index (in order),
/// decide whether its gid is already held (by an existing entry, or by an
/// EARLIER incoming entry) and, if not, mark it canonical. Mirrors
/// merge_halo_cells's own "already held" / "first occurrence wins" logic
/// exactly, independent of dedup_by_gid's own implementation.
struct ReferenceDedup {
  std::vector<std::uint8_t> incoming_is_new;
  int n_new = 0;
};

ReferenceDedup reference_dedup(const std::vector<std::int64_t>& existing,
                               const std::vector<std::int64_t>& incoming) {
  std::unordered_set<std::int64_t> held(existing.begin(), existing.end());
  ReferenceDedup out;
  out.incoming_is_new.assign(incoming.size(), 0);
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    if (held.count(incoming[i]) != 0) continue;
    held.insert(incoming[i]);
    out.incoming_is_new[i] = 1;
    ++out.n_new;
  }
  return out;
}

template <class Space>
void check_against_reference(const std::vector<std::int64_t>& existing,
                             const std::vector<std::int64_t>& incoming) {
  const auto existing_d = to_device_const<Space>(existing, "existing");
  const auto incoming_d = to_device_const<Space>(incoming, "incoming");
  const DedupByGidResult<Space> result = dedup_by_gid<Space>(existing_d, incoming_d);

  auto merged_h = Kokkos::create_mirror_view(result.merged_gids);
  Kokkos::deep_copy(merged_h, result.merged_gids);
  auto slot_existing_h = Kokkos::create_mirror_view(result.slot_of_existing);
  Kokkos::deep_copy(slot_existing_h, result.slot_of_existing);
  auto slot_incoming_h = Kokkos::create_mirror_view(result.slot_of_incoming);
  Kokkos::deep_copy(slot_incoming_h, result.slot_of_incoming);
  auto is_new_h = Kokkos::create_mirror_view(result.incoming_is_new);
  Kokkos::deep_copy(is_new_h, result.incoming_is_new);

  // merged_gids is exactly the ascending, deduplicated union.
  std::set<std::int64_t> expected_union(existing.begin(), existing.end());
  expected_union.insert(incoming.begin(), incoming.end());
  ASSERT_EQ(merged_h.extent(0), expected_union.size());
  {
    std::size_t i = 0;
    for (std::int64_t g : expected_union) EXPECT_EQ(merged_h(i++), g);
  }

  // Every existing/incoming entry's slot names its own gid in merged_gids.
  for (std::size_t i = 0; i < existing.size(); ++i) {
    ASSERT_GE(slot_existing_h(i), 0);
    ASSERT_LT(static_cast<std::size_t>(slot_existing_h(i)), merged_h.extent(0));
    EXPECT_EQ(merged_h(slot_existing_h(i)), existing[i]) << "existing index " << i;
  }
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    ASSERT_GE(slot_incoming_h(i), 0);
    ASSERT_LT(static_cast<std::size_t>(slot_incoming_h(i)), merged_h.extent(0));
    EXPECT_EQ(merged_h(slot_incoming_h(i)), incoming[i]) << "incoming index " << i;
  }

  // incoming_is_new / n_new match the reference exactly.
  const ReferenceDedup ref = reference_dedup(existing, incoming);
  ASSERT_EQ(is_new_h.extent(0), ref.incoming_is_new.size());
  for (std::size_t i = 0; i < ref.incoming_is_new.size(); ++i) {
    EXPECT_EQ(is_new_h(i), ref.incoming_is_new[i]) << "incoming index " << i;
  }
  EXPECT_EQ(result.n_new, ref.n_new);

  // Every two incoming entries sharing a gid share a slot, and a slot with
  // no existing claim has EXACTLY one canonical (is_new) member.
  std::unordered_map<int, int> canon_count_by_slot;
  for (std::size_t i = 0; i < incoming.size(); ++i) {
    if (is_new_h(i)) ++canon_count_by_slot[slot_incoming_h(i)];
  }
  for (const auto& [slot, count] : canon_count_by_slot) {
    EXPECT_EQ(count, 1) << "slot " << slot << " has " << count << " canonical incoming records";
  }
}

}  // namespace

TEST(NativeMeshMergeDeviceHost, NoOverlapEveryIncomingIsNew) {
  check_against_reference<Kokkos::HostSpace>({10, 20, 30}, {40, 50, 60});
}

TEST(NativeMeshMergeDeviceHost, IncomingDuplicatesOfExistingAreNotNew) {
  check_against_reference<Kokkos::HostSpace>({10, 20, 30}, {20, 20, 30, 40});
}

TEST(NativeMeshMergeDeviceHost, CrossSenderDuplicatesAmongIncomingCollapseToOneCanonical) {
  // 40 appears three times among incoming (three different senders'
  // buffers claiming the same halo cell/vertex); exactly one must be
  // canonical.
  check_against_reference<Kokkos::HostSpace>({10, 20, 30}, {40, 40, 50, 40, 60, 20});
}

TEST(NativeMeshMergeDeviceHost, EmptyIncomingIsNotARefusal) {
  check_against_reference<Kokkos::HostSpace>({10, 20, 30}, {});
}

TEST(NativeMeshMergeDeviceHost, EmptyExistingEveryDistinctIncomingIsNew) {
  check_against_reference<Kokkos::HostSpace>({}, {5, 5, 7, 9, 7});
}

using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

TEST(NativeMeshMergeDeviceDevice, NoOverlapEveryIncomingIsNew) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_against_reference<DeviceMemSpace>({10, 20, 30}, {40, 50, 60});
}

TEST(NativeMeshMergeDeviceDevice, IncomingDuplicatesOfExistingAreNotNew) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_against_reference<DeviceMemSpace>({10, 20, 30}, {20, 20, 30, 40});
}

TEST(NativeMeshMergeDeviceDevice, CrossSenderDuplicatesAmongIncomingCollapseToOneCanonical) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_against_reference<DeviceMemSpace>({10, 20, 30}, {40, 40, 50, 40, 60, 20});
}

TEST(NativeMeshMergeDeviceDevice, EmptyIncomingIsNotARefusal) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_against_reference<DeviceMemSpace>({10, 20, 30}, {});
}

TEST(NativeMeshMergeDeviceDevice, EmptyExistingEveryDistinctIncomingIsNew) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_against_reference<DeviceMemSpace>({}, {5, 5, 7, 9, 7});
}
