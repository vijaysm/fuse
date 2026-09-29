// fuse::partition::repartition: centroids -> rcb_partition -> migrate, at
// the base build configuration (kNullComm, one process) where it must be the identity.

#include "fuse/partition/repartition.hpp"

#include "fuse/backend/native_backend.hpp"

#include <gtest/gtest.h>

TEST(Repartition, AtKNullCommIsTheIdentityUpToLocalOrdering) {
  auto arrays = fuse::native::generate_native_mesh("hexbox:3x3x3", 0, 1);
  ASSERT_TRUE(arrays.ok());
  auto view = fuse::native::NativeMeshView::create_from_host(arrays.value(), 0);
  ASSERT_TRUE(view.ok());
  const std::size_t ncells = view.value()->cell_count();
  double original_total = 0.0;
  for (std::size_t c = 0; c < ncells; ++c) original_total += view.value()->measure(3, c);

  fuse::partition::RepartitionOptions opts;
  auto result = fuse::partition::repartition(*view.value(), opts);
  ASSERT_TRUE(result.ok()) << result.error().message;
  EXPECT_EQ(result.value()->cell_count(), ncells);
  double total = 0.0;
  for (std::size_t c = 0; c < result.value()->cell_count(); ++c) {
    total += result.value()->measure(3, c);
    EXPECT_TRUE(result.value()->is_owned(3, c));
  }
  EXPECT_NEAR(total, original_total, 1e-10);
}
