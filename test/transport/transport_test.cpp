#include <array>
#include <fuse/transport/tags.hpp>
#include <gtest/gtest.h>
#include <fuse/transport/transport.hpp>

#include <algorithm>
#include <memory>
#include <vector>

using fuse::CopyTransport;
using fuse::TransportRegistry;
using fuse::TransportSpec;

TEST(CopyTransport, MovesContiguousData) {
  const std::vector<double> src{1.0, 2.0, 3.0, 4.0};
  std::vector<double> dst(4, 0.0);

  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.src_rank_set = 0, .dst_rank_set = 0,
                                      .elements = 4}).ok());
  ASSERT_TRUE(t.move(src.data(), dst.data(), 4).ok());
  EXPECT_EQ(src, dst);
}

TEST(CopyTransport, IsLocalAndNamed) {
  const CopyTransport t;
  EXPECT_TRUE(t.is_local());
  EXPECT_EQ(t.name(), "copy");
}

TEST(CopyTransport, SelfMoveIsANoOp) {
  std::vector<double> v{1.0, 2.0, 3.0};
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 3}).ok());
  ASSERT_TRUE(t.move(v.data(), v.data(), 3).ok());
  EXPECT_DOUBLE_EQ(v[1], 2.0);
}

TEST(CopyTransport, RejectsNullPointers) {
  std::vector<double> v(4, 0.0);
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 4}).ok());
  EXPECT_FALSE(t.move(nullptr, v.data(), 4).ok());
  EXPECT_FALSE(t.move(v.data(), nullptr, 4).ok());
}

TEST(CopyTransport, RejectsMoveBeyondPreparedSize) {
  const std::vector<double> src(4, 1.0);
  std::vector<double> dst(4, 0.0);
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 4}).ok());
  const auto r = t.move(src.data(), dst.data(), 8);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

// Standing rule 1: the size guard must be enforcement, not an assert, so it
// has to hold with -O3 -DNDEBUG as well. Multi-component capacity is
// elements * components, and a move one past it is still rejected.
TEST(CopyTransport, SizeGuardIsEnforcedNotAsserted) {
  const std::vector<double> src(6, 1.0);
  std::vector<double> dst(6, 0.0);
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 2, .components = 3}).ok());
  ASSERT_TRUE(t.move(src.data(), dst.data(), 6).ok());
  const auto r = t.move(src.data(), dst.data(), 7);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-TRANSPORT-SIZE");
}

TEST(TransportRegistry, CopyIsAvailableByDefault) {
  const auto names = TransportRegistry::instance().available();
  EXPECT_NE(std::find(names.begin(), names.end(), "copy"), names.end());
}

TEST(TransportRegistry, CreatesByName) {
  const auto t = TransportRegistry::instance().create("copy");
  ASSERT_TRUE(t.ok()) << t.error().message;
  EXPECT_EQ(t.value()->name(), "copy");
}

TEST(TransportRegistry, UnknownBackendIsAnError) {
  const auto t = TransportRegistry::instance().create("does-not-exist");
  ASSERT_FALSE(t.ok());
  EXPECT_EQ(t.error().code, "E-TRANSPORT-BACKEND");
}

// register_backend replaces an existing entry rather than shadowing it, so the
// backend list stays free of duplicates.
TEST(TransportRegistry, ReRegisteringAnExistingNameReplacesIt) {
  TransportRegistry reg;
  reg.register_backend("copy", [] { return std::make_unique<CopyTransport>(); });
  reg.register_backend("copy", [] { return std::make_unique<CopyTransport>(); });
  const auto names = reg.available();
  EXPECT_EQ(std::count(names.begin(), names.end(), "copy"), 1);
}

// ---- MPI tags ----------------------------------------------------------

// Two mechanisms sharing a tag on the same communicator match each other's
// messages, which shows up as a wrong result or a hang rather than an error.
// kEpoch was first chosen as 0x5C13 -- the value MPI_Comm_create_group was
// already being given on the same communicator -- so the uniqueness is pinned
// here rather than left to whoever adds the next one.
TEST(Tags, AreDistinctAndWithinTheGuaranteedBound) {
  using namespace fuse::tags;
  // kIterationRegion (REVIEW.md R-10): the residual-agreement reduction an
  // iteration region's sweep loop runs on the same base communicator.
  const std::array<int, 4> all{kCrystalRouter, kUnionCommCreate, kEpoch,
                               kIterationRegion};

  std::array<int, 4> sorted = all;
  std::sort(sorted.begin(), sorted.end());
  EXPECT_EQ(std::adjacent_find(sorted.begin(), sorted.end()), sorted.end())
      << "two tags share a value; on a shared communicator that is a hang";

  // 32767 is the standard's minimum for MPI_TAG_UB, so a tag above it may be
  // rejected by a conforming implementation. Setup also checks the actual
  // bound at run time.
  for (const int t : all) {
    EXPECT_GT(t, 0);
    EXPECT_LE(t, 32767) << "tag " << t << " may exceed MPI_TAG_UB";
  }
  EXPECT_EQ(kMaxUsed, *std::max_element(all.begin(), all.end()))
      << "kMaxUsed is what setup checks against MPI_TAG_UB; it must be the "
         "largest tag actually used";
}
