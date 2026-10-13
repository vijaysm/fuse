// fuse::device::candidate_pairs (include/fuse/backend/supermesh_candidates_
// device.hpp): the box-overlap broad phase build_supermesh_weights's own
// brute-force loop performs today, as a Kokkos device kernel. No function
// in src/backend/supermesh.cpp is reachable from a test (its own
// `boxes_overlap` is in an anonymous namespace, translation-unit-local by
// construction -- unlike `inward_plane`, which supermesh_test.cpp reaches
// through the PRIVATE supermesh_detail.hpp for exactly this reason), so
// this test checks the device kernel against its own from-scratch serial
// reference, the same brute-force box test, rather than against
// supermesh.cpp's private function directly.

#include "fuse/backend/supermesh_candidates_device.hpp"

#include <random>
#include <vector>

#include <gtest/gtest.h>

namespace {

using fuse::device::candidate_pairs;

constexpr bool kHasDeviceBackend =
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
    true;
#else
    false;
#endif

template <class Space>
void check_matches_brute_force(int dim, int nt, int ns, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> U(0.0, 10.0);

  // Device views first, host MIRRORS (not independently-typed host views)
  // second: Kokkos::View's default array_layout differs by execution
  // space for a rank >= 2 view (LayoutLeft default on Cuda, LayoutRight
  // on Host/OpenMP), so a host view declared with no explicit layout and
  // a same-shaped device view declared with no explicit layout do NOT
  // generally agree on layout, and Kokkos::deep_copy between different
  // layouts AND different memory spaces in one call has "no available
  // copy mechanism" (measured on this track's own device build).
  // Kokkos::create_mirror_view(device_view) is built to avoid exactly
  // this: the mirror it returns always matches the device view's own
  // layout, so the deep_copy back is a same-layout blit.
  Kokkos::View<double * [3], Space> t_lo(Kokkos::view_alloc("t_lo", Kokkos::WithoutInitializing), nt);
  Kokkos::View<double * [3], Space> t_hi(Kokkos::view_alloc("t_hi", Kokkos::WithoutInitializing), nt);
  Kokkos::View<double * [3], Space> s_lo(Kokkos::view_alloc("s_lo", Kokkos::WithoutInitializing), ns);
  Kokkos::View<double * [3], Space> s_hi(Kokkos::view_alloc("s_hi", Kokkos::WithoutInitializing), ns);
  auto t_lo_h = Kokkos::create_mirror_view(t_lo);
  auto t_hi_h = Kokkos::create_mirror_view(t_hi);
  auto s_lo_h = Kokkos::create_mirror_view(s_lo);
  auto s_hi_h = Kokkos::create_mirror_view(s_hi);

  const auto rand_box = [&](double* lo, double* hi) {
    for (int k = 0; k < dim; ++k) {
      const double x0 = U(rng);
      const double w = U(rng) * 0.3;
      lo[k] = x0;
      hi[k] = x0 + w;
    }
    for (int k = dim; k < 3; ++k) { lo[k] = 0.0; hi[k] = 0.0; }
  };
  for (int t = 0; t < nt; ++t) {
    double lo[3], hi[3];
    rand_box(lo, hi);
    for (int k = 0; k < 3; ++k) { t_lo_h(t, k) = lo[k]; t_hi_h(t, k) = hi[k]; }
  }
  for (int s = 0; s < ns; ++s) {
    double lo[3], hi[3];
    rand_box(lo, hi);
    for (int k = 0; k < 3; ++k) { s_lo_h(s, k) = lo[k]; s_hi_h(s, k) = hi[k]; }
  }

  Kokkos::deep_copy(t_lo, t_lo_h);
  Kokkos::deep_copy(t_hi, t_hi_h);
  Kokkos::deep_copy(s_lo, s_lo_h);
  Kokkos::deep_copy(s_hi, s_hi_h);

  auto result = candidate_pairs<Space>(t_lo, t_hi, s_lo, s_hi, dim);
  auto offsets_h = Kokkos::create_mirror_view(result.offsets);
  Kokkos::deep_copy(offsets_h, result.offsets);
  auto src_index_h = Kokkos::create_mirror_view(result.src_index);
  Kokkos::deep_copy(src_index_h, result.src_index);

  ASSERT_EQ(static_cast<std::size_t>(offsets_h(nt)), src_index_h.extent(0));

  int total_ref = 0;
  for (int t = 0; t < nt; ++t) {
    std::vector<int> ref;
    for (int s = 0; s < ns; ++s) {
      bool ov = true;
      for (int k = 0; k < dim; ++k) {
        if (t_hi_h(t, k) < s_lo_h(s, k) || s_hi_h(s, k) < t_lo_h(t, k)) { ov = false; break; }
      }
      if (ov) ref.push_back(s);
    }
    total_ref += static_cast<int>(ref.size());
    const int b = offsets_h(t), e = offsets_h(t + 1);
    std::vector<int> got(src_index_h.data() + b, src_index_h.data() + e);
    EXPECT_EQ(got, ref) << "target " << t;
  }
  EXPECT_EQ(static_cast<int>(src_index_h.extent(0)), total_ref);
}

TEST(SupermeshCandidatesDeviceHost, MatchesBruteForce2D) {
  check_matches_brute_force<Kokkos::HostSpace>(2, 40, 55, 7);
}

TEST(SupermeshCandidatesDeviceHost, MatchesBruteForce3D) {
  check_matches_brute_force<Kokkos::HostSpace>(3, 25, 30, 11);
}

TEST(SupermeshCandidatesDeviceHost, EmptyWhenNoBoxesOverlap) {
  check_matches_brute_force<Kokkos::HostSpace>(2, 0, 0, 3);
}

using DeviceMemSpace = Kokkos::DefaultExecutionSpace::memory_space;

TEST(SupermeshCandidatesDeviceDevice, MatchesBruteForce2D) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_matches_brute_force<DeviceMemSpace>(2, 40, 55, 7);
}

TEST(SupermeshCandidatesDeviceDevice, MatchesBruteForce3D) {
  if (!kHasDeviceBackend) GTEST_SKIP() << "no Cuda/HIP/SYCL backend in this build";
  check_matches_brute_force<DeviceMemSpace>(3, 25, 30, 11);
}

}  // namespace
