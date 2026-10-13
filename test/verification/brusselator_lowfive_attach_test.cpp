// S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track. The
// pure, MPI-free and HDF5-free half of the participants' programmatic VOL
// attach -- examples/brusselator/participants/lowfive_attach_decision.hpp's
// own header comment has the full account of what `decide_attach_mode()`
// decides and why. Nothing in this file touches LowFive, HDF5, or a live
// MPI communicator: every RankSet/LaunchMode/ChannelRoles value below is
// hand-built, the same "force a shape by hand, no real communicator"
// discipline KNOWN_ISSUES.md C1 already documents for this codebase's
// other MPI-shaped pure functions.

#include <gtest/gtest.h>

#include "lowfive_attach_decision.hpp"

#include "fuse/model/component.hpp"
#include "fuse/model/launch.hpp"

using fuse::LaunchMode;
using fuse::RankSet;
using fuse_examples::lowfive_attach::AttachMode;
using fuse_examples::lowfive_attach::ChannelRoles;
using fuse_examples::lowfive_attach::decide_attach_mode;

namespace {

// Mirrors the Brusselator mpmd-3 reference graph's own shape: U {0,1}
// (Mpmd, appnum 0), V {1,1} (Mpmd, appnum 1), coupler {2,1} -- the
// coupler never takes part in an HDF5-bound channel, so it never
// appears in a ChannelRoles list at all.
ChannelRoles MakeChannel(const std::string& field_name, RankSet writer_ranks,
                         RankSet reader_ranks, LaunchMode writer_launch = LaunchMode::Mpmd,
                         LaunchMode reader_launch = LaunchMode::Mpmd) {
  ChannelRoles ch;
  ch.field_name = field_name;
  ch.writer_ranks = std::move(writer_ranks);
  ch.reader_ranks = std::move(reader_ranks);
  ch.writer_launch = writer_launch;
  ch.reader_launch = reader_launch;
  return ch;
}

}  // namespace

TEST(DecideAttachMode, WriterRankGetsDistributedWithSplitColorZero) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1})};
  const auto d = decide_attach_mode(channels, /*this_rank=*/0);
  ASSERT_TRUE(d.ok()) << d.error().code << ": " << d.error().message;
  EXPECT_EQ(d.value().mode, AttachMode::kDistributed);
  EXPECT_TRUE(d.value().rendezvous.is_writer);
  EXPECT_EQ(d.value().rendezvous.split_color, 0);
  EXPECT_EQ(d.value().rendezvous.remote_leader, 1);
}

TEST(DecideAttachMode, ReaderRankGetsDistributedWithSplitColorOne) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1})};
  const auto d = decide_attach_mode(channels, /*this_rank=*/1);
  ASSERT_TRUE(d.ok()) << d.error().code << ": " << d.error().message;
  EXPECT_EQ(d.value().mode, AttachMode::kDistributed);
  EXPECT_FALSE(d.value().rendezvous.is_writer);
  EXPECT_EQ(d.value().rendezvous.split_color, 1);
  EXPECT_EQ(d.value().rendezvous.remote_leader, 0);
}

// The Brusselator's own bidirectional exchange: u_out (U writes, V reads)
// and v_out (V writes, U reads) -- same two groups, swapped labels. Rank
// 0 (U) must still reach exactly one, consistent decision.
TEST(DecideAttachMode, BidirectionalChannelsWithSwappedRolesAgree) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}),
      MakeChannel("V.v_out", RankSet{1, 1}, RankSet{0, 1})};
  const auto d_u = decide_attach_mode(channels, /*this_rank=*/0);
  ASSERT_TRUE(d_u.ok()) << d_u.error().code << ": " << d_u.error().message;
  EXPECT_EQ(d_u.value().mode, AttachMode::kDistributed);
  EXPECT_TRUE(d_u.value().rendezvous.is_writer);  // canonical channel's writer side

  const auto d_v = decide_attach_mode(channels, /*this_rank=*/1);
  ASSERT_TRUE(d_v.ok()) << d_v.error().code << ": " << d_v.error().message;
  EXPECT_EQ(d_v.value().mode, AttachMode::kDistributed);
  EXPECT_FALSE(d_v.value().rendezvous.is_writer);  // canonical channel's reader side
}

TEST(DecideAttachMode, NoChannelsIsRefused) {
  const std::vector<ChannelRoles> channels;
  const auto d = decide_attach_mode(channels, /*this_rank=*/0);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-NOCHANNELS");
}

TEST(DecideAttachMode, AcceptModeOnWriterIsRefused) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Connect, LaunchMode::Mpmd)};
  const auto d = decide_attach_mode(channels, /*this_rank=*/0);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-ACCEPT");
}

TEST(DecideAttachMode, AcceptModeOnReaderIsRefused) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Mpmd, LaunchMode::Connect)};
  const auto d = decide_attach_mode(channels, /*this_rank=*/1);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-ACCEPT");
}

// A third channel naming a group outside the canonical pair -- the gap
// src/transport/lowfive_plugin_graph.cpp's own ConfigureDistributed()
// comment names as unchecked there.
TEST(DecideAttachMode, InconsistentPartitionAcrossChannelsIsRefused) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}),
      MakeChannel("W.w_out", RankSet{0, 1}, RankSet{2, 1})};
  const auto d = decide_attach_mode(channels, /*this_rank=*/0);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-INCONSISTENT");
}

// this_rank in neither side -- decide_attach_mode() must not silently
// invent a role; it propagates plan_lowfive_rendezvous()'s own refusal
// verbatim (fuse/transport/lowfive.hpp).
TEST(DecideAttachMode, RankNotInEitherSidePropagatesUnplaced) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1})};
  const auto d = decide_attach_mode(channels, /*this_rank=*/7);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-RENDEZVOUS-UNPLACED");
}

TEST(DecideAttachMode, OverlappingWriterReaderPropagatesOverlap) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 2}, RankSet{1, 1})};  // {0,1} overlaps {1}
  const auto d = decide_attach_mode(channels, /*this_rank=*/1);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-RENDEZVOUS-OVERLAP");
}
