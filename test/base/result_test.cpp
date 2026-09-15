#include <gtest/gtest.h>
#include <fuse/base/result.hpp>

#include <string>

using fuse::Diagnostics;
using fuse::Failure;
using fuse::Result;
using fuse::Severity;

TEST(Result, HoldsValue) {
  Result<int> r{42};
  ASSERT_TRUE(r.ok());
  EXPECT_TRUE(static_cast<bool>(r));
  EXPECT_EQ(r.value(), 42);
}

TEST(Result, HoldsError) {
  Result<int> r{Failure{"E001", "bad thing"}};
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

// ---- transforms (S1.3) --------------------------------------------------

TEST(Diagnostics, FormatsThroughFmtIdenticallyToToString) {
  Diagnostics d;
  d.add(Severity::Error, "E1", "first", "ctx");
  d.add(Severity::Warning, "W2", "second");
  EXPECT_EQ(fmt::format("{}", d), d.to_string());
  EXPECT_EQ(fmt::format("{}", d.items()[0]), "error E1: first [ctx]");
  EXPECT_EQ(fmt::format("{}", Failure{"E9", "gone"}), "[E9] gone");
}

// U2: one terminal call reports EVERY defect, in insertion order. The
// first-failure behaviour is the thing being excluded.
TEST(Diagnostics, AccumulatesEveryDefectInOrder) {
  Diagnostics d;
  for (int i = 0; i < 5; ++i) d.add(Severity::Error, fmt::format("E{}", i), "defect");
  ASSERT_EQ(d.size(), 5u);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(d.items()[static_cast<std::size_t>(i)].code, fmt::format("E{}", i));
}
