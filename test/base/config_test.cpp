// The build's own consistency. Cheap, and it catches the failure mode a green
// unit suite cannot: headers from one Kokkos install, a library from another.
#include <fuse/base/config.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <string>

TEST(Config, KokkosHeadersMatchTheResolvedLibrary) {
  // KOKKOS_VERSION is MAJOR*10000 + MINOR*100 + PATCH.
  const std::string expected = FUSE_KOKKOS_VERSION_EXPECTED;  // "4.4.1" form
  int major = 0, minor = 0, patch = 0;
  ASSERT_EQ(std::sscanf(expected.c_str(), "%d.%d.%d", &major, &minor, &patch), 3) << expected;
  EXPECT_EQ(KOKKOS_VERSION, major * 10000 + minor * 100 + patch)
      << "headers say " << KOKKOS_VERSION << ", CMake resolved " << expected;
}

TEST(Config, ExceptionsFlagIsVisible) {
  static_assert(FUSE_ENABLE_EXCEPTIONS == 0 || FUSE_ENABLE_EXCEPTIONS == 1);
  SUCCEED();
}
