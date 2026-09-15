#include <gtest/gtest.h>
#include <soma/soma.hpp>

#include <string_view>

TEST(Smoke, VersionIsNonEmpty) {
  EXPECT_FALSE(std::string_view{soma::version()}.empty());
}
