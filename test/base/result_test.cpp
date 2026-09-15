#include <gtest/gtest.h>
#include <fuse/base/result.hpp>

#include <string>

using fuse::Diagnostics;
using fuse::Error;
using fuse::Result;
using fuse::Severity;

TEST(Result, HoldsValue) {
  Result<int> r{42};
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(static_cast<bool>(r));
  EXPECT_EQ(r.value(), 42);
}

TEST(Result, HoldsError) {
  Result<int> r{Error{"E001", "bad thing"}};
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E001");
  EXPECT_EQ(r.error().message, "bad thing");
}

TEST(Result, VoidSuccess) {
  auto r = Result<void>::success();
  EXPECT_TRUE(r.ok());
}

TEST(Diagnostics, AccumulatesAndReportsErrors) {
  Diagnostics d;
  EXPECT_FALSE(d.has_errors());
  d.add(Severity::Warning, "W001", "just a warning", "channel[0]");
  EXPECT_FALSE(d.has_errors());
  d.add(Severity::Error, "E002", "real problem", "channel[1]");
  EXPECT_TRUE(d.has_errors());
  EXPECT_EQ(d.size(), 2u);
  EXPECT_EQ(d.items()[1].context, "channel[1]");
}

TEST(Diagnostics, MergePreservesOrder) {
  Diagnostics a, b;
  a.add(Severity::Error, "E1", "first", "");
  b.add(Severity::Error, "E2", "second", "");
  a.merge(std::move(b));
  ASSERT_EQ(a.size(), 2u);
  EXPECT_EQ(a.items()[0].code, "E1");
  EXPECT_EQ(a.items()[1].code, "E2");
}

TEST(Diagnostics, ToStringMentionsEveryCode) {
  Diagnostics d;
  d.add(Severity::Error, "E42", "boom", "here");
  const std::string s = d.to_string();
  EXPECT_NE(s.find("E42"), std::string::npos);
  EXPECT_NE(s.find("boom"), std::string::npos);
  EXPECT_NE(s.find("here"), std::string::npos);
}
