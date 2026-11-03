#include <array>
#include <fuse/transport/tags.hpp>
#include "fuse/transport/rank_agreement.hpp"

#include <gtest/gtest.h>
#include <fuse/transport/transport.hpp>

#include <algorithm>
#include <cstdint>
#include <map>
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

// MpiTransport::move() (transport.hpp's own doc)
// accepts null exactly when n is zero -- a rank owning no elements of a mesh
// entity set is ordinary in parallel, and an empty std::vector<double>'s
// data() is permitted to be null. CopyTransport must do the same, so
// the same zero-element, same-process channel is not refused by this backend
// and accepted by the MPI one.
TEST(CopyTransport, AcceptsNullPointersWhenNIsZero) {
  CopyTransport t;
  ASSERT_TRUE(t.prepare(TransportSpec{.elements = 0}).ok());
  EXPECT_TRUE(t.move(nullptr, nullptr, 0).ok());
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
  // kIterationRegion: the residual-agreement reduction an
  // implicit coupling group's sweep loop runs on the same base communicator.
  // kWindowConvergence: a window-scoped region's
  // own per-sweep convergence agreement, same communicator.
  const std::array<int, 5> all{kCrystalRouter, kUnionCommCreate, kEpoch,
                               kIterationRegion, kWindowConvergence};

  std::array<int, 5> sorted = all;
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


// ---- flat source-offer deduplication (pure rule, no MPI) --------------

// The 3-rank shape: rank 0 owns 4
// source cells (global ids 0-3); two different adopting ranks (destination
// peers 1 and 2) each independently committed a remap halo over cell 2,
// so rank 0's own per-peer want records both name cell 2, each under its
// OWN local index (2, the SAME physical entity, since rank 0 holds exactly
// one copy). `CommGraph::build`'s own `emit()` (comm_graph.cpp) builds
// EXACTLY this per-peer shape before calling this function.
TEST(DedupeFlatSourceOffers, TwoDifferentPeersWantingTheSameOwnedCellCollapseToOneDeclaration) {
  const std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>
      by_peer{
          {1, {{0, 0}, {1, 1}, {2, 2}, {3, 3}}},  // peer 1 (6 target cells) wants all 4
          {2, {{2, 2}, {3, 3}}},                  // peer 2 (3 target cells) wants 2 of them
      };
  std::vector<std::size_t> ids;
  std::vector<std::size_t> idxs;
  fuse::dedupe_flat_source_offers(by_peer, &ids, &idxs);

  // Each of the 4 owned cells is declared EXACTLY once, however many peers
  // want it -- not once per (peer, cell) pair (which would be 6, the total
  // number of individual offers above).
  EXPECT_EQ(ids.size(), 4u);
  EXPECT_EQ(idxs.size(), 4u);
  EXPECT_EQ(ids, (std::vector<std::size_t>{0, 1, 2, 3}));
  EXPECT_EQ(idxs, (std::vector<std::size_t>{0, 1, 2, 3}));
}

// Without this function (flatten every peer's own
// entries unconditionally, as the RECV side correctly does), the SAME
// input reproduces the exact duplicate MpiTransport::build_plan's own Phase
// D refuses: global id 2 appears twice in rank 0's own declared list.
TEST(DedupeFlatSourceOffers, WithoutDedupingTheSameInputWouldDeclareTheSharedCellTwice) {
  const std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>
      by_peer{
          {1, {{0, 0}, {1, 1}, {2, 2}, {3, 3}}},
          {2, {{2, 2}, {3, 3}}},
      };
  std::vector<std::size_t> undeduped_ids;
  for (const auto& [peer, offers] : by_peer) {
    (void)peer;
    for (const auto& [gid, idx] : offers) {
      undeduped_ids.push_back(static_cast<std::size_t>(gid));
      (void)idx;
    }
  }
  EXPECT_EQ(undeduped_ids.size(), 6u);
  EXPECT_EQ(std::count(undeduped_ids.begin(), undeduped_ids.end(), 2u), 2)
      << "global id 2, wanted by two different peers, appears twice without "
         "deduplication -- the exact shape MpiTransport::build_plan's Phase "
         "D rejects as \"held by source rank 0 and again by source rank 0\"";
}

// A cell wanted by only ONE peer is unaffected: no duplicate to collapse.
TEST(DedupeFlatSourceOffers, NoFanOutIsAPureFlatten) {
  const std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>
      by_peer{{1, {{10, 0}, {11, 1}}}};
  std::vector<std::size_t> ids, idxs;
  fuse::dedupe_flat_source_offers(by_peer, &ids, &idxs);
  EXPECT_EQ(ids, (std::vector<std::size_t>{10, 11}));
  EXPECT_EQ(idxs, (std::vector<std::size_t>{0, 1}));
}

// No peers at all: an empty list, not a crash.
TEST(DedupeFlatSourceOffers, EmptyInputIsEmptyOutput) {
  const std::map<int, std::vector<std::pair<std::uint64_t, std::size_t>>>
      by_peer;
  std::vector<std::size_t> ids, idxs;
  fuse::dedupe_flat_source_offers(by_peer, &ids, &idxs);
  EXPECT_TRUE(ids.empty());
  EXPECT_TRUE(idxs.empty());
}

// ---- component agreement (pure rule, no MPI) ---------------------------

// A two-rank job: rank 0 is the source with 1
// component, rank 1 the destination with 2. Both ranks gather the same
// array, so both must reach the same outcome -- and that outcome must be a
// disagreement. The rule is checked here, where it can be run without MPI.
TEST(ComponentAgreement, SourceAndDestinationDisagreeing) {
  const auto a = fuse::transport::check_component_agreement(
      /*all_components=*/{1, 2}, /*is_src=*/{1, 0}, /*is_dst=*/{0, 1});
  EXPECT_FALSE(a.agreed());
  EXPECT_EQ(a.first_rank, 0);
  EXPECT_EQ(a.first_value, 1);
  EXPECT_EQ(a.disagreeing_rank, 1);
  EXPECT_EQ(a.disagreeing_value, 2);
}

TEST(ComponentAgreement, AgreeingParticipantsAndIgnoredBystanders) {
  // Every process agrees; a rank in neither set may hold anything,
  // because its count never sizes any count array.
  const auto ok = fuse::transport::check_component_agreement(
      {3, 99, 3}, {1, 0, 0}, {0, 0, 1});
  EXPECT_TRUE(ok.agreed());
  EXPECT_EQ(ok.first_rank, 0);

  // A disagreement between two processes is still found when a bystander
  // sits between them.
  const auto bad = fuse::transport::check_component_agreement(
      {3, 99, 4}, {1, 0, 0}, {0, 0, 1});
  EXPECT_FALSE(bad.agreed());
  EXPECT_EQ(bad.disagreeing_rank, 2);

  // No processes at all is not a disagreement.
  EXPECT_TRUE(fuse::transport::check_component_agreement({1, 2}, {0, 0},
                                                         {0, 0})
                  .agreed());
}
