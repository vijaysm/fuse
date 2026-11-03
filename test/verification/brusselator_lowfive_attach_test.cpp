// The pure, MPI-free and HDF5-free half of the participants' programmatic VOL
// attach -- examples/brusselator/participants/lowfive_attach_decision.hpp's
// own header comment has the full account of what `decide_attach_mode()`
// decides and why. Nothing in this file touches LowFive, HDF5, or a live
// MPI communicator: every RankSet/LaunchMode/ChannelRoles value below is
// hand-built, the same "force a shape by hand, no real communicator"
// discipline this codebase's other MPI-shaped pure functions follow.

#include <gtest/gtest.h>

#include "lowfive_attach_decision.hpp"
#include "lowfive_attach_model_check.hpp"

#include "fuse/model/component.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/launch.hpp"

using fuse::LaunchMode;
using fuse::RankSet;
using fuse_examples::lowfive_attach::AttachMode;
using fuse_examples::lowfive_attach::ChannelRoles;
using fuse_examples::lowfive_attach::decide_attach_mode;
using fuse_examples::lowfive_attach::decide_serve_roles;
using fuse_examples::lowfive_attach::BuildIntercommRegistrations;
using fuse_examples::lowfive_attach::decide_vol_settings;
using fuse_examples::lowfive_attach::FindOwnExportedFiles;
using fuse_examples::lowfive_attach::FormatMemoryPatternsForTrace;
using fuse_examples::lowfive_attach::FormatRegistrationsForTrace;
using fuse_examples::lowfive_attach::IntercommRegistration;
using fuse_examples::lowfive_attach::NormalizeDatasetPattern;
using fuse_examples::lowfive_attach::ParseTraceEnabledEnv;
using fuse_examples::lowfive_attach::ServeAction;
using fuse_examples::lowfive_attach::ServeEvent;
using fuse_examples::lowfive_attach::ServeRole;
using fuse_examples::lowfive_attach::ServeState;
using fuse_examples::lowfive_attach::VolSettings;
using fuse_examples::lowfive_attach::VolSettingsSatisfyInvariant;
using fuse_examples::lowfive_attach::DatasetGroupSplit;
using fuse_examples::lowfive_attach::SplitDatasetGroupPath;
using fuse_examples::lowfive_attach::ServeNowOrDefer;
using fuse_examples::lowfive_attach::decide_serve_action;
using fuse_examples::lowfive_attach::ExpectedEventKind;
using fuse_examples::lowfive_attach::ExpectedOrderForExchange;
using fuse_examples::lowfive_attach::OrderTracker;
using fuse_examples::lowfive_attach::CheckAndAdvanceOrder;
using fuse_examples::lowfive_attach::ChannelOrderSignature;
using fuse_examples::lowfive_attach::ComputeGraphOrderSignature;
using fuse_examples::lowfive_attach::ValidatedGraphOrderSignatures;
using fuse_examples::lowfive_attach::CheckGraphOrderValidated;
using fuse_examples::lowfive_attach::LowFiveProcessRole;
using fuse_examples::lowfive_attach::EnvRouteVolVariableNames;
using fuse_examples::lowfive_attach::ExitHygienePlan;
using fuse_examples::lowfive_attach::decide_exit_hygiene_plan;
using fuse_examples::lowfive_attach_model::WouldReuseStaleFileOnRecreate;
using fuse_examples::lowfive_attach_model::EventKind;
using fuse_examples::lowfive_attach_model::ExchangeOrder;
using fuse_examples::lowfive_attach_model::BuildEventSequence;
using fuse_examples::lowfive_attach_model::RunSimulationForOrder;
using fuse_examples::lowfive_attach_model::static_check_accepts;
using fuse_examples::lowfive_attach_model::model_completes;
using fuse_examples::lowfive_attach_model::ProcessPolicy;
using fuse_examples::lowfive_attach_model::PolicyKind;

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

// A Connect-mode channel with a FORMED bridge and a supported topology
// (<=2 Connect-mode components graph-wide) reaches the IDENTICAL
// kDistributed decision Mpmd does, never refused by launch mode alone.
// This is decide_attach_mode()'s own pure logic (the host run that
// exercises it in practice is a separate, host-only question).
TEST(DecideAttachMode, AcceptModeOnWriterWithFormedBridgeIsDistributed) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Connect, LaunchMode::Mpmd)};
  fuse_examples::lowfive_attach::AttachBridgeContext bridge;
  bridge.bridge_formed = true;
  bridge.connect_mode_component_count = 1;
  const auto d = decide_attach_mode(channels, /*this_rank=*/0, bridge);
  ASSERT_TRUE(d.ok()) << d.error().code << ": " << d.error().message;
  EXPECT_EQ(d.value().mode, AttachMode::kDistributed);
  EXPECT_TRUE(d.value().rendezvous.is_writer);
  EXPECT_EQ(d.value().rendezvous.split_color, 0);
}

TEST(DecideAttachMode, AcceptModeOnReaderWithFormedBridgeIsDistributed) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Mpmd, LaunchMode::Connect)};
  fuse_examples::lowfive_attach::AttachBridgeContext bridge;
  bridge.bridge_formed = true;
  bridge.connect_mode_component_count = 1;
  const auto d = decide_attach_mode(channels, /*this_rank=*/1, bridge);
  ASSERT_TRUE(d.ok()) << d.error().code << ": " << d.error().message;
  EXPECT_EQ(d.value().mode, AttachMode::kDistributed);
  EXPECT_FALSE(d.value().rendezvous.is_writer);
  EXPECT_EQ(d.value().rendezvous.split_color, 1);
}

// The ORIGINAL refusal survives, named differently, for the one case
// design (A) never claimed to close: no bridge at all (fuse_setup()
// never given a communicator, or connect() itself refused before this
// attach could run). The DEFAULT AttachBridgeContext{} (bridge_formed
// false) is exactly the one every EXISTING decide_attach_mode() call
// above (with no third argument) already uses, so this is also a
// regression test that the default itself still refuses.
TEST(DecideAttachMode, ConnectModeWithNoBridgeFormedIsRefused) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Connect, LaunchMode::Mpmd)};
  const auto d = decide_attach_mode(channels, /*this_rank=*/0);  // default bridge: not formed
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-ACCEPT-NO-BRIDGE");
}

// connect_pairwise() itself supports exactly two Connect-mode components;
// this is the SAME limit, checked here too, as defence in depth for a
// caller that reaches this function
// without having gone through connect() first.
TEST(DecideAttachMode, MoreThanTwoConnectModeComponentsIsRefused) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Connect, LaunchMode::Mpmd)};
  fuse_examples::lowfive_attach::AttachBridgeContext bridge;
  bridge.bridge_formed = true;
  bridge.connect_mode_component_count = 3;
  const auto d = decide_attach_mode(channels, /*this_rank=*/0, bridge);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY");
}

// The >2-components check fires BEFORE the no-bridge check -- a caller
// should learn about the unsupported topology, not a secondary symptom
// of it (a graph like this would also never form a bridge, since
// connect_pairwise() itself refuses first).
TEST(DecideAttachMode, MoreThanTwoComponentsRefusalPrecedesNoBridgeRefusal) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1}, LaunchMode::Connect, LaunchMode::Mpmd)};
  fuse_examples::lowfive_attach::AttachBridgeContext bridge;
  bridge.bridge_formed = false;
  bridge.connect_mode_component_count = 3;
  const auto d = decide_attach_mode(channels, /*this_rank=*/0, bridge);
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY");
}

// Mpmd-only channels are UNAFFECTED by AttachBridgeContext entirely --
// `any_connect` stays false, so neither new check is even consulted;
// this is the existing WriterRankGetsDistributedWithSplitColorZero
// shape, with an EXPLICIT (still-default, still-unformed) bridge
// context, proving the two new checks are gated correctly.
TEST(DecideAttachMode, MpmdOnlyChannelIgnoresBridgeContextEntirely) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{1, 1})};
  const auto d = decide_attach_mode(channels, /*this_rank=*/0,
                                    fuse_examples::lowfive_attach::AttachBridgeContext{});
  ASSERT_TRUE(d.ok()) << d.error().code << ": " << d.error().message;
  EXPECT_EQ(d.value().mode, AttachMode::kDistributed);
}

// `this_rank` must come from the BRIDGE, never from a rank's own local
// MPI_COMM_WORLD, in an accept (Connect) launch -- MPI_Intercomm_merge's
// own high/low rule offsets V's own position by U's local world size, so
// V's own local-world rank number and its bridge-relative one are
// DIFFERENT integers. This test builds the EXACT accept-3 layout (U
// physics rank 0, a bundled coupler/placeholder rank 1 naming no channel
// at all, V at bridge rank 2) and shows the local-world derivation (V's own
// local MPI_COMM_WORLD rank, 0, since V's own one-rank executable always
// numbers itself from 0) does not refuse and does not crash -- it
// silently misclassifies V as the WRITER instead of the reader, because
// U's own writer_ranks={0} happens to contain 0 too. A refusal would
// be a kinder failure than this.
TEST(DecideAttachMode, WrongThisRankFromLocalWorldInsteadOfBridgeMisclassifiesV) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", /*writer=*/RankSet{0, 1}, /*reader=*/RankSet{2, 1},
                  LaunchMode::Connect, LaunchMode::Connect)};
  fuse_examples::lowfive_attach::AttachBridgeContext bridge;
  bridge.bridge_formed = true;
  bridge.connect_mode_component_count = 2;

  // CORRECT: this_rank = 2, V's own position in the BRIDGE.
  const auto d_correct = decide_attach_mode(channels, /*this_rank=*/2, bridge);
  ASSERT_TRUE(d_correct.ok()) << d_correct.error().code << ": " << d_correct.error().message;
  EXPECT_FALSE(d_correct.value().rendezvous.is_writer) << "V is the reader, correctly";

  // WRONG: this_rank = 0, V's own LOCAL MPI_COMM_WORLD rank.
  const auto d_wrong = decide_attach_mode(channels, /*this_rank=*/0, bridge);
  ASSERT_TRUE(d_wrong.ok()) << d_wrong.error().code << ": " << d_wrong.error().message;
  EXPECT_TRUE(d_wrong.value().rendezvous.is_writer)
      << "the local-world derivation's silent failure mode: V misclassified as the writer";
}

// DecideLocalSplitChoice() -- the pure color/key decision
// bruss_lowfive_join_local_split() turns into a real MPI_Comm_split().
TEST(DecideLocalSplitChoice, EmptyChannelsDoesNotParticipate) {
  const std::vector<ChannelRoles> channels;
  const auto choice = fuse_examples::lowfive_attach::DecideLocalSplitChoice(channels, 1);
  EXPECT_FALSE(choice.participates);
}

TEST(DecideLocalSplitChoice, WriterRankGetsColorZeroKeyedByRankSetPosition) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 2}, RankSet{5, 1})};  // writer = {0,1}
  const auto choice = fuse_examples::lowfive_attach::DecideLocalSplitChoice(channels, 1);
  ASSERT_TRUE(choice.participates);
  EXPECT_EQ(choice.color, 0);
  EXPECT_EQ(choice.key, 1);  // index_of(1) within {0,1}
}

TEST(DecideLocalSplitChoice, ReaderRankGetsColorOne) {
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{5, 2})};  // reader = {5,6}
  const auto choice = fuse_examples::lowfive_attach::DecideLocalSplitChoice(channels, 6);
  ASSERT_TRUE(choice.participates);
  EXPECT_EQ(choice.color, 1);
  EXPECT_EQ(choice.key, 1);  // index_of(6) within {5,6}
}

TEST(DecideLocalSplitChoice, NonMemberRankDoesNotParticipate) {
  // The bundled coupler/placeholder rank (2): neither writer nor reader.
  const std::vector<ChannelRoles> channels{
      MakeChannel("U.u_out", RankSet{0, 1}, RankSet{3, 1})};
  const auto choice = fuse_examples::lowfive_attach::DecideLocalSplitChoice(channels, 2);
  EXPECT_FALSE(choice.participates);
}

namespace {

// The smallest graph that exercises CollectChannelsForRank()
// on a REAL fuse::CouplingGraph -- U (RankSet{0,1}, writer of u_out,
// hdf5_path set), a fieldless Coupler-kind placeholder (RankSet{1,1},
// named "coupler" in a channel neither writer nor reader -- the accept-
// mode bundled rank's own shape), V (RankSet{2,1}, reader of u_out) --
// mirrors the real accept-3 layout's own bridge-relative numbering
// (U physics 0, placeholder 1, V 2).
fuse::CouplingGraph MakeAcceptLikeGraph() {
  using fuse::AccumOp;
  using fuse::ComponentDecl;
  using fuse::ComponentKind;
  using fuse::Conservation;
  using fuse::CouplingGraph;
  using fuse::FieldSpec;
  using fuse::MeshDecl;
  using fuse::Rational;
  using fuse::Region;
  using fuse::RemapFlags;
  namespace u = fuse::units;

  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "grid", .source = "generated:box:4x4x4"});
  const auto r = g.add_region(Region{.name = "all", .mesh = m, .entity_set = "all",
                                     .dimension = 3});

  const auto u_comp = g.add_component(
      ComponentDecl{.name = "U", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  ComponentDecl coupler_decl{.name = "coupler", .ranks = fuse::RankSet{1, 1},
                             .interval = Rational{1}};
  coupler_decl.kind = ComponentKind::Coupler;
  g.add_component(coupler_decl);
  const auto v_comp = g.add_component(
      ComponentDecl{.name = "V", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1}});

  FieldSpec u_out_spec{.name = "u_out", .region = r, .units = u::kDimensionless,
                       .conservation = Conservation::None, .intent = fuse::Intent::Provides};
  u_out_spec.hdf5_path = "brusselator_u.h5/grid/u_out";
  const auto u_out = g.add_field(u_comp, u_out_spec);
  FieldSpec v_in_spec{.name = "u_in", .region = r, .units = u::kDimensionless,
                      .conservation = Conservation::None, .intent = fuse::Intent::Requires};
  const auto v_in = g.add_field(v_comp, v_in_spec);

  const auto ch = g.add_channel("u2v", u_out, v_in);
  g.channel(ch).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);
  return g;
}

}  // namespace

TEST(CollectChannelsForRank, WriterRankSeesTheOneQualifyingChannel) {
  const auto g = MakeAcceptLikeGraph();
  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(g, /*this_rank=*/0);
  ASSERT_EQ(channels.size(), 1u);
  EXPECT_EQ(channels.front().writer_name, "U");
  EXPECT_EQ(channels.front().reader_name, "V");
}

TEST(CollectChannelsForRank, ReaderRankSeesTheOneQualifyingChannel) {
  const auto g = MakeAcceptLikeGraph();
  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(g, /*this_rank=*/2);
  ASSERT_EQ(channels.size(), 1u);
}

// The bundled coupler/placeholder rank (1) is neither writer nor reader of
// the one channel this graph declares -- CollectChannelsForRank() must return it EMPTY, not the
// channel by accident (a Coupler-kind component is never a field owner
// at all, validate()'s own E-COUPLER-FIELDS rule, so this is a property
// of the RankSet membership test alone, not of ComponentKind).
TEST(CollectChannelsForRank, BundledPlaceholderRankSeesNoChannelAtAll) {
  const auto g = MakeAcceptLikeGraph();
  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(g, /*this_rank=*/1);
  EXPECT_TRUE(channels.empty());
}

// The real Brusselator exchange is BIDIRECTIONAL
// (u_out: U writes, V reads; v_out: V writes, U reads) -- MakeAcceptLikeGraph()
// above has only one channel, enough for the three CollectChannelsForRank
// tests it serves, but not enough to compute the FULL two-channel order
// signature CheckGraphOrderValidated() actually checks. Same component
// layout and bridge-relative numbering (U physics 0, placeholder 1, V 2)
// as MakeAcceptLikeGraph(), with the second channel added.
fuse::CouplingGraph MakeAcceptLikeGraphBidirectional() {
  using fuse::AccumOp;
  using fuse::ComponentDecl;
  using fuse::ComponentKind;
  using fuse::Conservation;
  using fuse::CouplingGraph;
  using fuse::FieldSpec;
  using fuse::MeshDecl;
  using fuse::Rational;
  using fuse::Region;
  using fuse::RemapFlags;
  namespace u = fuse::units;

  CouplingGraph g;
  const auto m = g.add_mesh(MeshDecl{.name = "grid", .source = "generated:box:4x4x4"});
  const auto r = g.add_region(Region{.name = "all", .mesh = m, .entity_set = "all",
                                     .dimension = 3});

  const auto u_comp = g.add_component(
      ComponentDecl{.name = "U", .ranks = fuse::RankSet{0, 1}, .interval = Rational{1}});
  ComponentDecl coupler_decl{.name = "coupler", .ranks = fuse::RankSet{1, 1},
                             .interval = Rational{1}};
  coupler_decl.kind = ComponentKind::Coupler;
  g.add_component(coupler_decl);
  const auto v_comp = g.add_component(
      ComponentDecl{.name = "V", .ranks = fuse::RankSet{2, 1}, .interval = Rational{1}});

  FieldSpec u_out_spec{.name = "u_out", .region = r, .units = u::kDimensionless,
                       .conservation = Conservation::None, .intent = fuse::Intent::Provides};
  u_out_spec.hdf5_path = "brusselator_u.h5/grid/u_out";
  const auto u_out = g.add_field(u_comp, u_out_spec);
  FieldSpec v_in_spec{.name = "u_in", .region = r, .units = u::kDimensionless,
                      .conservation = Conservation::None, .intent = fuse::Intent::Requires};
  const auto v_in = g.add_field(v_comp, v_in_spec);
  const auto ch1 = g.add_channel("u2v", u_out, v_in);
  g.channel(ch1).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);

  FieldSpec v_out_spec{.name = "v_out", .region = r, .units = u::kDimensionless,
                       .conservation = Conservation::None, .intent = fuse::Intent::Provides};
  v_out_spec.hdf5_path = "brusselator_v.h5/grid/v_out";
  const auto v_out = g.add_field(v_comp, v_out_spec);
  FieldSpec u_in_spec{.name = "v_in", .region = r, .units = u::kDimensionless,
                      .conservation = Conservation::None, .intent = fuse::Intent::Requires};
  const auto u_in = g.add_field(u_comp, u_in_spec);
  const auto ch2 = g.add_channel("v2u", v_out, u_in);
  g.channel(ch2).accumulate(AccumOp::Mean).remap(RemapFlags::Conservative);

  return g;
}

// ValidatedGraphOrderSignatures() covers the accept layout only because its
// derived signature is identical to the validated MPMD one. PROVEN
// identical here, from an ACTUAL fuse::CouplingGraph matching the real
// accept-3 layout's own component names, RankSets and channel order
// (brusselator_graph_tool.cpp's own accept branch: u_ranks={0,1},
// v_ranks={u_ranks_u+ranks_coupler,1}, both launch modes share the SAME
// add_window_coupling() call that declares u_out/v_out in the SAME order)
// -- so the one {U,V,1,1},{V,U,1,1} entry covers both launch modes,
// because the check is signature-based, never launch-mode-based.
TEST(ComputeGraphOrderSignature, AcceptLayoutPhysicsRanksMatchTheValidatedMpmdSignature) {
  const auto g = MakeAcceptLikeGraphBidirectional();
  for (const int physics_rank : {0, 2}) {  // U's physics rank, then V's
    const auto channels =
        fuse_examples::lowfive_attach::CollectChannelsForRank(g, physics_rank);
    const auto signature =
        fuse_examples::lowfive_attach::ComputeGraphOrderSignature(channels);
    EXPECT_EQ(signature, fuse_examples::lowfive_attach::ValidatedGraphOrderSignatures())
        << "physics_rank=" << physics_rank;
    EXPECT_TRUE(
        fuse_examples::lowfive_attach::CheckGraphOrderValidated(signature).ok())
        << "physics_rank=" << physics_rank;
  }
}

// The bundled placeholder rank's own signature (empty, no qualifying
// channel) does NOT match the validated one -- exactly why the companion
// fix must keep this rank from ever reaching CheckGraphOrderValidated()
// at all (u_main.c's own is_physics_rank gate), not because the check
// itself needs to special-case it.
TEST(ComputeGraphOrderSignature, PlaceholderRankSignatureDoesNotMatchAndWouldBeRefused) {
  const auto g = MakeAcceptLikeGraphBidirectional();
  const auto channels = fuse_examples::lowfive_attach::CollectChannelsForRank(g, 1);
  const auto signature = fuse_examples::lowfive_attach::ComputeGraphOrderSignature(channels);
  EXPECT_TRUE(signature.empty());
  const auto checked = fuse_examples::lowfive_attach::CheckGraphOrderValidated(signature);
  EXPECT_FALSE(checked.ok());
  EXPECT_EQ(checked.error().code, "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED");
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

// ---------------------------------------------------------------------
// Role-ordered serve: decide_serve_roles(). See lowfive_
// attach_decision.hpp's own comment for the deadlock this exists to
// avoid.

TEST(DecideServeRoles, LowerMinRankWinsRegardlessOfArgumentOrder) {
  // U = {0, 1} (ranks 0-1, min_rank 0), V = {2} (min_rank 2): U must win
  // first-server either way the two components are passed in, since the
  // assignment is a property of the two RankSets, not of call order.
  const auto fwd = decide_serve_roles("U", RankSet{0, 2}, "V", RankSet{2, 1});
  ASSERT_TRUE(fwd.ok()) << fwd.error().code << ": " << fwd.error().message;
  EXPECT_EQ(fwd.value().first_server_name, "U");
  EXPECT_EQ(fwd.value().first_reader_name, "V");

  const auto rev = decide_serve_roles("V", RankSet{2, 1}, "U", RankSet{0, 2});
  ASSERT_TRUE(rev.ok()) << rev.error().code << ": " << rev.error().message;
  EXPECT_EQ(rev.value().first_server_name, "U");
  EXPECT_EQ(rev.value().first_reader_name, "V");
}

TEST(DecideServeRoles, OverlappingRankSetsAreRefused) {
  const auto d = decide_serve_roles("U", RankSet{0, 2}, "V", RankSet{1, 2});  // share rank 1
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-SERVE-OVERLAP");
}

TEST(DecideServeRoles, EmptyRankSetIsRefused) {
  const auto d = decide_serve_roles("U", RankSet{0, 0}, "V", RankSet{2, 1});  // U placed nowhere
  ASSERT_FALSE(d.ok());
  EXPECT_EQ(d.error().code, "E-LOWFIVE-ATTACH-SERVE-EMPTY");
}

// ---------------------------------------------------------------------
// serve_state_step(). kImportAboutToOpen is ALWAYS kReceiveOnly (never
// flushes, any role or state) and a separate event, kImportClosed, is
// where a first-reader's own deferred serve is flushed. Exercises
// both roles across the Jacobi row's own repeated write/write-then-
// read/read cycle, plus every refusal the state machine makes.

TEST(ServeStateStep, FirstServerServesThenReceivesThenFinalizesCleanRepeatedly) {
  ServeState st;
  st.role = ServeRole::kFirstServer;
  for (int iter = 0; iter < 3; ++iter) {
    const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
    ASSERT_TRUE(a1.ok()) << a1.error().code;
    EXPECT_EQ(a1.value(), ServeAction::kServeNow);

    // Every read now fires TWO events (both no-ops for a first-server,
    // which never defers): the pre-open hook, then the new post-close one.
    const auto a2 = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
    ASSERT_TRUE(a2.ok()) << a2.error().code;
    EXPECT_EQ(a2.value(), ServeAction::kReceiveOnly);

    const auto a3 = serve_state_step(&st, ServeEvent::kImportClosed);
    ASSERT_TRUE(a3.ok()) << a3.error().code;
    EXPECT_EQ(a3.value(), ServeAction::kNoopAfterImport);
  }
  const auto fin = serve_state_step(&st, ServeEvent::kFinalize);
  ASSERT_TRUE(fin.ok()) << fin.error().code;
  EXPECT_EQ(fin.value(), ServeAction::kNoopFinalizeClean);
}

// Not every iteration's own write defers (lowfive_attach_decision.hpp's
// own comment on the measured order): the host's own measured
// steady-state order shows a first-reader deferring ONLY at the cold
// start, serving immediately every write after. The cycle, repeated:
// iteration 0 defers (cold start, nothing completed yet) and flushes at
// its own kImportClosed; iterations 1+ serve immediately at write
// (decide_serve_action()'s own policy, since a read has already
// completed) and have nothing left to flush.
TEST(ServeStateStep, FirstReaderDefersOnlyTheColdStartThenServesImmediatelyEveryLaterWrite) {
  ServeState st;
  st.role = ServeRole::kFirstReader;

  const auto cold_write = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(cold_write.ok()) << cold_write.error().code;
  EXPECT_EQ(cold_write.value(), ServeAction::kDeferServe);

  const auto cold_about = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
  ASSERT_TRUE(cold_about.ok()) << cold_about.error().code;
  EXPECT_EQ(cold_about.value(), ServeAction::kReceiveOnly);

  const auto cold_closed = serve_state_step(&st, ServeEvent::kImportClosed);
  ASSERT_TRUE(cold_closed.ok()) << cold_closed.error().code;
  EXPECT_EQ(cold_closed.value(), ServeAction::kFlushDeferred);  // the cold start's own flush

  for (int exchange = 0; exchange < 3; ++exchange) {
    const auto w = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
    ASSERT_TRUE(w.ok()) << w.error().code;
    EXPECT_EQ(w.value(), ServeAction::kServeNow) << "exchange " << exchange;  // no longer deferred

    const auto about = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
    ASSERT_TRUE(about.ok()) << about.error().code;
    EXPECT_EQ(about.value(), ServeAction::kReceiveOnly);

    const auto closed = serve_state_step(&st, ServeEvent::kImportClosed);
    ASSERT_TRUE(closed.ok()) << closed.error().code;
    EXPECT_EQ(closed.value(), ServeAction::kNoopAfterImport) << "exchange " << exchange;  // nothing to flush
  }
  const auto fin = serve_state_step(&st, ServeEvent::kFinalize);
  ASSERT_TRUE(fin.ok()) << fin.error().code;
  EXPECT_EQ(fin.value(), ServeAction::kNoopFinalizeClean);
}

// The DIRECT test for the flush placement:
// kImportAboutToOpen must return kReceiveOnly and leave deferred_pending
// UNTOUCHED even when a serve genuinely is pending -- the mutation row
// "serve-state-step-flush-moved-back-to-pre-open" (build-evidence/
// mutations/lowfive-attach.toml) reintroduces exactly the flush this
// test exists to catch directly, independent of the model-check tests
// below (which catch the same mutation a different way -- by failing to
// complete the simulated protocol).
TEST(ServeStateStep, FirstReaderImportAboutToOpenNeverFlushesEvenWithDeferredPending) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(a1.ok());
  EXPECT_EQ(a1.value(), ServeAction::kDeferServe);
  ASSERT_TRUE(st.deferred_pending);

  const auto a2 = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
  ASSERT_TRUE(a2.ok()) << a2.error().code;
  EXPECT_EQ(a2.value(), ServeAction::kReceiveOnly);
  EXPECT_TRUE(st.deferred_pending) << "kImportAboutToOpen must never clear a deferred serve";
}

// A first-reader's own import with nothing deferred (its own first local
// event being an import, not an export) is an ordinary receive/no-op,
// not an error -- at EITHER of the two import events, see serve_state_
// step()'s own doc.
TEST(ServeStateStep, FirstReaderImportWithNothingDeferredIsOrdinaryAtBothEvents) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  const auto a = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
  ASSERT_TRUE(a.ok()) << a.error().code;
  EXPECT_EQ(a.value(), ServeAction::kReceiveOnly);

  const auto b = serve_state_step(&st, ServeEvent::kImportClosed);
  ASSERT_TRUE(b.ok()) << b.error().code;
  EXPECT_EQ(b.value(), ServeAction::kNoopAfterImport);
}

TEST(ServeStateStep, FirstReaderSecondExportBeforeFlushIsRefused) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(a1.ok());
  EXPECT_EQ(a1.value(), ServeAction::kDeferServe);

  // A second export with no intervening kImportClosed -- violates the
  // one-deferred-serve-at-a-time rule.
  const auto a2 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_FALSE(a2.ok());
  EXPECT_EQ(a2.error().code, "E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER");
}

// The double-defer refusal is unaffected by kImportAboutToOpen's own
// no-op status: an about-to-open event between the two exports must NOT
// be mistaken for the flush that would have made the second export legal.
TEST(ServeStateStep, FirstReaderSecondExportAfterOnlyAboutToOpenStillRefused) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(a1.ok());
  EXPECT_EQ(a1.value(), ServeAction::kDeferServe);

  const auto about = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
  ASSERT_TRUE(about.ok());
  EXPECT_EQ(about.value(), ServeAction::kReceiveOnly);

  // kImportClosed never arrived -- still a violation.
  const auto a2 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_FALSE(a2.ok());
  EXPECT_EQ(a2.error().code, "E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER");
}

TEST(ServeStateStep, FirstReaderFinalizeWithDeferredServePendingIsRefused) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(a1.ok());
  EXPECT_EQ(a1.value(), ServeAction::kDeferServe);

  // No kImportClosed ever flushes it -- finalize must refuse, not hang or drop.
  const auto fin = serve_state_step(&st, ServeEvent::kFinalize);
  ASSERT_FALSE(fin.ok());
  EXPECT_EQ(fin.error().code, "E-LOWFIVE-ATTACH-SERVE-UNFLUSHED");
}

TEST(ServeStateStep, FirstServerNeverDefersSoFinalizeIsAlwaysCleanEvenMidCycle) {
  ServeState st;
  st.role = ServeRole::kFirstServer;
  const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(a1.ok());
  EXPECT_EQ(a1.value(), ServeAction::kServeNow);

  // Finalize reached right after an export, with no import in between --
  // still clean, because the first-server's role never defers anything.
  const auto fin = serve_state_step(&st, ServeEvent::kFinalize);
  ASSERT_TRUE(fin.ok()) << fin.error().code;
  EXPECT_EQ(fin.value(), ServeAction::kNoopFinalizeClean);
}

// ---------------------------------------------------------------------
// Model-check tests. lowfive_attach_model_check.hpp's
// own top comment has the full account of what these do and do not
// establish (a check of the PROTOCOL against the blocking semantics AS
// READ from LowFive's source -- not a measurement of LowFive itself).
// The "shipped" tests drive the REAL decide_serve_roles()/serve_state_
// step(), so the mutation row "serve-state-step-flush-moved-back-to-
// pre-open" is caught here too (in addition to the direct unit test
// above), not only by a reimplementation that a mutation could leave
// untouched.

TEST(ModelCheck, ShippedProtocolCompletesFor1Iteration) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(1, ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// Uses `RunSimulationForOrder()` with the MEASURED order
// (lowfive_attach_model::ExchangeOrder::kMeasured) -- the one shape the real
// graph produces for N>=2 exchanges: the two components' own steady-state
// orders are COMPLEMENTARY (lowfive_attach_decision.hpp's own comment on
// the measured order), never identical. A shape that put BOTH roles
// through write-then-read every exchange is reported stuck for N>=2 by
// decide_serve_action() -- see ModelCheck.BothSidesWriteReadSteadyGetsStuck
// below, the DIRECT test for exactly that shape.
TEST(ModelCheck, MeasuredOrderCompletesFor2Exchanges) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, 2,
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// 60: this row's own kMaxIterations (brusselator_graph_tool.cpp) --
// reused here as an exchange count, a stress value, not a claim about any
// measured sweep count (the measured per-run exchange counts are in
// lowfive_attach_decision.hpp's comment on the measured order and are not
// derived here).
TEST(ModelCheck, MeasuredOrderCompletesFor60Exchanges) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, 60,
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// Which SEARCH POSITION (A or B) carries the first-server role must not
// matter -- the search itself has no preferred side.
TEST(ModelCheck, MeasuredOrderCompletesRegardlessOfWhichPositionIsFirstServer) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, 3,
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader},
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// A REJECTED placement (flush at kAboutToOpen instead of
// kImportClosed) -- illustrative evidence that this search is
// discriminating, not evidence about the shipped protocol (which the
// tests above alone speak for). Expressed as a canned policy, NOT
// through serve_state_step() (which cannot take this path at all).
TEST(ModelCheck, RejectedPreOpenFlushPlacementDeadlocks) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r =
      RunSimulation(1, ProcessPolicy{PolicyKind::kRejectedPreOpenFlush, ServeRole::kFirstServer},
                    ProcessPolicy{PolicyKind::kRejectedPreOpenFlush, ServeRole::kFirstReader});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// The "natural placement" (serve at every write, both sides, never defer)
// -- illustrative.
TEST(ModelCheck, OriginalNaturalPlacementDeadlocks) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(1, ProcessPolicy{PolicyKind::kOriginalNatural, ServeRole::kFirstServer},
                                ProcessPolicy{PolicyKind::kOriginalNatural, ServeRole::kFirstServer});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// Strawman: both sides defer, neither ever serves unconditionally --
// nobody ever goes first, so the very first read on either side blocks
// forever. Illustrative: shows asymmetric roles are NECESSARY, not just
// sufficient.
TEST(ModelCheck, BothSidesDeferDeadlocks) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(1, ProcessPolicy{PolicyKind::kBothDefer, ServeRole::kFirstServer},
                                ProcessPolicy{PolicyKind::kBothDefer, ServeRole::kFirstServer});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// ---------------------------------------------------------------------
// The LowFive::DistMetadataVOL::set_intercomm() registration, and
// fuse::split_hdf5_path()'s unslashed dataset component, which must match
// LowFive::Object::fullname()'s own slashed form --
// lowfive_attach_decision.hpp's own comment has the citations for both.

TEST(NormalizeDatasetPattern, UnslashedGetsALeadingSlash) {
  const auto r = NormalizeDatasetPattern("grid/u_out");
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_EQ(r.value(), "/grid/u_out");
}

TEST(NormalizeDatasetPattern, AlreadySlashedIsUnchanged) {
  const auto r = NormalizeDatasetPattern("/grid/u_out");
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_EQ(r.value(), "/grid/u_out");
}

TEST(NormalizeDatasetPattern, EmptyIsRefused) {
  const auto r = NormalizeDatasetPattern("");
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-ATTACH-DATASET-EMPTY-PATTERN");
}

namespace {
// The Brusselator's own two-binding shape (u_out: U writes it, v_out: V
// writes it) -- the SAME field_name/hdf5_path convention lowfive_attach.
// cpp's own ApplyBindings() sees via fuse::build_lowfive_bindings().
fuse::LowFiveFieldBindingTable MakeBrusselatorBindingTable() {
  fuse::LowFiveFieldBindingTable table;
  auto a = table.add({"U.u_out", "brusselator_u.h5/grid/u_out"});
  auto b = table.add({"V.v_out", "brusselator_v.h5/grid/v_out"});
  // add()'s own contract (fuse/transport/lowfive.hpp) already tier-0
  // tests these exact ports succeeding; a failure here would mean that
  // contract broke, not a new input to this file.
  if (!a.ok() || !b.ok()) throw std::runtime_error("MakeBrusselatorBindingTable: unexpected add() failure");
  return table;
}
}  // namespace

TEST(BuildIntercommRegistrations, OneEntryPerBindingWithTheGivenIndexAndLeadingSlash) {
  const auto table = MakeBrusselatorBindingTable();
  const auto r = BuildIntercommRegistrations(table, /*intercomm_index=*/0);
  ASSERT_TRUE(r.ok()) << r.error().code;
  ASSERT_EQ(r.value().size(), 2u);
  for (const IntercommRegistration& reg : r.value()) {
    EXPECT_EQ(reg.intercomm_index, 0);
    EXPECT_EQ(reg.dataset_pattern.front(), '/') << reg.dataset_pattern;
  }
  EXPECT_EQ(r.value()[0].file, "brusselator_u.h5");
  EXPECT_EQ(r.value()[0].dataset_pattern, "/grid/u_out");
  EXPECT_EQ(r.value()[1].file, "brusselator_v.h5");
  EXPECT_EQ(r.value()[1].dataset_pattern, "/grid/v_out");
}

TEST(BuildIntercommRegistrations, IndexIsWhateverTheCallerPasses) {
  const auto table = MakeBrusselatorBindingTable();
  const auto r = BuildIntercommRegistrations(table, /*intercomm_index=*/7);
  ASSERT_TRUE(r.ok()) << r.error().code;
  for (const IntercommRegistration& reg : r.value()) EXPECT_EQ(reg.intercomm_index, 7);
}

TEST(BuildIntercommRegistrations, EmptyTableIsEmptyResultNotARefusal) {
  const fuse::LowFiveFieldBindingTable table;
  const auto r = BuildIntercommRegistrations(table, 0);
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_TRUE(r.value().empty());
}

// ---------------------------------------------------------------------
// VolSettings/decide_vol_settings()/VolSettingsSatisfyInvariant() -- the
// pure decision BOTH LowFive-level settings are routed through, instead of
// being set separately.

TEST(DecideVolSettings, DistributedWantsServeOnCloseFalseAndKeepTrue) {
  const auto s = decide_vol_settings(/*distributed=*/true);
  EXPECT_FALSE(s.serve_on_close);
  EXPECT_TRUE(s.keep);
}

TEST(DecideVolSettings, SingleProcessStillWantsKeepTrue) {
  // serve_on_close is meaningless for the single-process branch's own
  // LowFive::MetadataVOL (no such member) -- only `keep` is ever
  // actually consumed there, and it must be true.
  const auto s = decide_vol_settings(/*distributed=*/false);
  EXPECT_TRUE(s.keep);
}

TEST(VolSettingsSatisfyInvariant, DecideVolSettingsOwnOutputSatisfiesIt) {
  EXPECT_TRUE(VolSettingsSatisfyInvariant(decide_vol_settings(true)));
  EXPECT_TRUE(VolSettingsSatisfyInvariant(decide_vol_settings(false)));
}

TEST(VolSettingsSatisfyInvariant, ServeOnCloseFalseAndKeepFalseViolatesIt) {
  // A wrong configuration, fed directly: a settings pair with
  // serve_on_close=false and keep=false.
  EXPECT_FALSE(VolSettingsSatisfyInvariant(VolSettings{/*serve_on_close=*/false, /*keep=*/false}));
}

TEST(VolSettingsSatisfyInvariant, AnyPairWithKeepTrueSatisfiesIt) {
  EXPECT_TRUE(VolSettingsSatisfyInvariant(VolSettings{false, true}));
  EXPECT_TRUE(VolSettingsSatisfyInvariant(VolSettings{true, true}));
}

TEST(VolSettingsSatisfyInvariant, ServeOnCloseTrueAndKeepFalseSatisfiesIt) {
  // Never a configuration this attach actually ships, but the
  // invariant itself ("serve_on_close==false implies keep==true") does
  // not forbid it -- LowFive's own internal auto-serve covers this
  // pair, per lowfive_attach_decision.hpp's own "ROOT CAUSE" comment.
  EXPECT_TRUE(VolSettingsSatisfyInvariant(VolSettings{true, false}));
}

// ---------------------------------------------------------------------
// FindOwnExportedFiles() -- the binding-table-derived file
// name(s) this rank's own attach must drop after each is served.

TEST(FindOwnExportedFiles, USOwnFileIsBrusselatorU) {
  const auto table = MakeBrusselatorBindingTable();
  const auto r = FindOwnExportedFiles(table, "U");
  ASSERT_TRUE(r.ok()) << r.error().code;
  ASSERT_EQ(r.value().size(), 1u);
  EXPECT_EQ(r.value()[0], "brusselator_u.h5");
}

TEST(FindOwnExportedFiles, VSOwnFileIsBrusselatorV) {
  const auto table = MakeBrusselatorBindingTable();
  const auto r = FindOwnExportedFiles(table, "V");
  ASSERT_TRUE(r.ok()) << r.error().code;
  ASSERT_EQ(r.value().size(), 1u);
  EXPECT_EQ(r.value()[0], "brusselator_v.h5");
}

TEST(FindOwnExportedFiles, UnknownComponentOwnsNothing) {
  const auto table = MakeBrusselatorBindingTable();
  const auto r = FindOwnExportedFiles(table, "W");
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_TRUE(r.value().empty());
}

TEST(FindOwnExportedFiles, EmptyTableOwnsNothing) {
  const fuse::LowFiveFieldBindingTable table;
  const auto r = FindOwnExportedFiles(table, "U");
  ASSERT_TRUE(r.ok()) << r.error().code;
  EXPECT_TRUE(r.value().empty());
}

// ---------------------------------------------------------------------
// The model check's own file-lifetime cases
// (ProcessPolicy::file_deleted_at_close), driven from
// decide_vol_settings() directly, not a hand-set bool disconnected from it
// -- the model must be driven from decide_vol_settings so that a mutation
// of the setting changes the verdict.

TEST(ModelCheck, ShippedConfigurationCompletesFor1Iteration) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto settings = fuse_examples::lowfive_attach::decide_vol_settings(/*distributed=*/true);
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, !settings.keep};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader, !settings.keep};
  const auto r = RunSimulation(1, server, reader);
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// Driven from decide_vol_settings() via RunSimulationForOrder() and the
// measured order (RunSimulation()'s own BuildScript() shape is unrealistic
// for N>=2).
TEST(ModelCheck, MeasuredOrderWithDecideVolSettingsCompletesFor2And60Exchanges) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto settings = fuse_examples::lowfive_attach::decide_vol_settings(/*distributed=*/true);
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, !settings.keep};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader, !settings.keep};
  for (int n : {2, 60}) {
    const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, n, server, reader);
    EXPECT_TRUE(r.completed) << "n=" << n << " " << (r.stuck_example ? *r.stuck_example : "");
  }
}

TEST(ModelCheck, MeasuredOrderWithDecideVolSettingsCompletesRegardlessOfWhichPositionIsFirstServer) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto settings = fuse_examples::lowfive_attach::decide_vol_settings(/*distributed=*/true);
  ProcessPolicy a{PolicyKind::kShipped, ServeRole::kFirstReader, !settings.keep};
  ProcessPolicy b{PolicyKind::kShipped, ServeRole::kFirstServer, !settings.keep};
  const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, 2, a, b);
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// Mutation target (a): if decide_vol_settings() is mutated to return
// keep=false, THIS test's own policy flips to file_deleted_at_close=
// true and the shipped configuration gets stuck -- failing the
// EXPECT_TRUE(r.completed) tests above, not a separate hand-set case.
// Pinned directly here too, for a config that is NOT what decide_vol_
// settings() actually returns, so this mechanism stays pinned even if
// decide_vol_settings() itself is never wrong.
TEST(ModelCheck, FileDeletedAtCloseConfigurationGetsStuck) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, /*file_deleted_at_close=*/true};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader, /*file_deleted_at_close=*/true};
  const auto r = RunSimulation(1, server, reader);
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// Sanity check on the flag itself: leaving it at its own default (false)
// must reproduce the ordinary, already-passing shipped-protocol result
// -- i.e. this field cannot have silently changed the nominal case.
// Uses the measured order (RunSimulation()'s own BuildScript() shape is
// unrealistic for N>=2, as above).
TEST(ModelCheck, ShippedProtocolDefaultsToKeptAndStillCompletes) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader};
  EXPECT_FALSE(server.file_deleted_at_close);
  EXPECT_FALSE(reader.file_deleted_at_close);
  EXPECT_TRUE(server.drop_after_serve);
  EXPECT_TRUE(reader.drop_after_serve);
  const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, 2, server, reader);
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// ---------------------------------------------------------------------
// WouldReuseStaleFileOnRecreate() -- the THIRD named configuration
// ("keep=true without the drop"), reported as a stale-data-reuse failure,
// NOT as a stuck/diverging search result: it is a DIFFERENT failure mode,
// and this header's own doc comment on the function explains why no
// interleaving search is needed or able to detect it (it is not a
// blocking/liveness issue at all).

TEST(WouldReuseStaleFileOnRecreate, KeepTrueWithoutDropOnSecondIterationIsTrue) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, /*file_deleted_at_close=*/false,
                       /*drop_after_serve=*/false};
  EXPECT_TRUE(WouldReuseStaleFileOnRecreate(server, 2));
  EXPECT_TRUE(WouldReuseStaleFileOnRecreate(server, 60));
}

TEST(WouldReuseStaleFileOnRecreate, OneIterationNeverRecreatesAnything) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, /*file_deleted_at_close=*/false,
                       /*drop_after_serve=*/false};
  EXPECT_FALSE(WouldReuseStaleFileOnRecreate(server, 1));
}

TEST(WouldReuseStaleFileOnRecreate, ShippedConfigurationNeverReusesAStaleFile) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto settings = fuse_examples::lowfive_attach::decide_vol_settings(/*distributed=*/true);
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, !settings.keep};
  EXPECT_FALSE(WouldReuseStaleFileOnRecreate(server, 60));
}

TEST(WouldReuseStaleFileOnRecreate, FileDeletedAtCloseHasNothingToReuseEither) {
  // If the file was never kept at all, there is nothing left to reuse stale -- the two
  // failure modes are mutually exclusive, not compounding.
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, /*file_deleted_at_close=*/true,
                       /*drop_after_serve=*/false};
  EXPECT_FALSE(WouldReuseStaleFileOnRecreate(server, 60));
}


// ---------------------------------------------------------------------
// Diagnostic trace: the three PURE functions the real trace call sites
// (lowfive_attach.cpp/shim.cpp's own getenv()/fprintf() calls, not
// testable here; only measurable on a host) go through.

TEST(ParseTraceEnabledEnv, ExactlyOneCharOneIsEnabled) {
  EXPECT_TRUE(ParseTraceEnabledEnv("1"));
}

TEST(ParseTraceEnabledEnv, UnsetIsDisabled) {
  EXPECT_FALSE(ParseTraceEnabledEnv(nullptr));
}

TEST(ParseTraceEnabledEnv, EmptyIsDisabled) {
  EXPECT_FALSE(ParseTraceEnabledEnv(""));
}

TEST(ParseTraceEnabledEnv, ZeroIsDisabled) {
  EXPECT_FALSE(ParseTraceEnabledEnv("0"));
}

TEST(ParseTraceEnabledEnv, WordFormsAreDisabled) {
  // The spec is "FUSE_LOWFIVE_ATTACH_TRACE=1" --
  // deliberately NOT a permissive grammar; "true"/"TRUE"/"yes"/"10"
  // must all be treated the same as unset, not guessed at.
  EXPECT_FALSE(ParseTraceEnabledEnv("true"));
  EXPECT_FALSE(ParseTraceEnabledEnv("TRUE"));
  EXPECT_FALSE(ParseTraceEnabledEnv("yes"));
  EXPECT_FALSE(ParseTraceEnabledEnv("10"));
}

TEST(FormatRegistrationsForTrace, EmptyListIsEmptyBrackets) {
  EXPECT_EQ(FormatRegistrationsForTrace({}), "[]");
}

TEST(FormatRegistrationsForTrace, OneEntryExactFormat) {
  std::vector<IntercommRegistration> regs{{"brusselator_u.h5", "/grid/u_out", 0}};
  EXPECT_EQ(FormatRegistrationsForTrace(regs), "[(brusselator_u.h5, /grid/u_out, 0)]");
}

TEST(FormatRegistrationsForTrace, TwoEntriesCommaSeparated) {
  std::vector<IntercommRegistration> regs{{"brusselator_u.h5", "/grid/u_out", 0},
                                          {"brusselator_v.h5", "/grid/v_out", 0}};
  EXPECT_EQ(FormatRegistrationsForTrace(regs),
           "[(brusselator_u.h5, /grid/u_out, 0), (brusselator_v.h5, /grid/v_out, 0)]");
}

TEST(FormatMemoryPatternsForTrace, EmptyListIsEmptyBrackets) {
  EXPECT_EQ(FormatMemoryPatternsForTrace({}), "[]");
}

TEST(FormatMemoryPatternsForTrace, OneEntryExactFormat) {
  std::vector<std::pair<std::string, std::string>> patterns{{"brusselator_u.h5", "/grid/u_out"}};
  EXPECT_EQ(FormatMemoryPatternsForTrace(patterns), "[(brusselator_u.h5, /grid/u_out)]");
}

TEST(FormatMemoryPatternsForTrace, TwoEntriesCommaSeparated) {
  std::vector<std::pair<std::string, std::string>> patterns{{"brusselator_u.h5", "/grid/u_out"},
                                                             {"brusselator_v.h5", "/grid/v_out"}};
  EXPECT_EQ(FormatMemoryPatternsForTrace(patterns),
           "[(brusselator_u.h5, /grid/u_out), (brusselator_v.h5, /grid/v_out)]");
}

// SplitDatasetGroupPath() tests. "/grid/u_out" is the exact
// path both u_main.c's U_LOWFIVE_OWN_DATASET and v_main.f90's
// kOwnDataset/kPeerDataset use.
TEST(SplitDatasetGroupPath, NestedPathSplitsGroupAndLeaf) {
  DatasetGroupSplit s = SplitDatasetGroupPath("/grid/u_out");
  EXPECT_EQ(s.group_path, "/grid");
  EXPECT_EQ(s.leaf_name, "u_out");
}

TEST(SplitDatasetGroupPath, OtherComponentNestedPath) {
  DatasetGroupSplit s = SplitDatasetGroupPath("/grid/v_out");
  EXPECT_EQ(s.group_path, "/grid");
  EXPECT_EQ(s.leaf_name, "v_out");
}

TEST(SplitDatasetGroupPath, SingleLeadingSlashNeedsNoGroup) {
  DatasetGroupSplit s = SplitDatasetGroupPath("/u_out");
  EXPECT_EQ(s.group_path, "");
  EXPECT_EQ(s.leaf_name, "u_out");
}

TEST(SplitDatasetGroupPath, NoSlashAtAllNeedsNoGroup) {
  DatasetGroupSplit s = SplitDatasetGroupPath("u_out");
  EXPECT_EQ(s.group_path, "");
  EXPECT_EQ(s.leaf_name, "u_out");
}

TEST(SplitDatasetGroupPath, MultiSegmentPathKeepsWholeGroupPrefix) {
  // group_create() (LowFive source) creates intermediate groups one
  // segment at a time on its own, so a single H5Gcreate2() call with
  // this WHOLE multi-segment group_path is sufficient -- this test
  // only pins SplitDatasetGroupPath()'s own output, not LowFive's
  // behavior (which is not under tier-0 test here).
  DatasetGroupSplit s = SplitDatasetGroupPath("/a/b/c");
  EXPECT_EQ(s.group_path, "/a/b");
  EXPECT_EQ(s.leaf_name, "c");
}

// ---------------------------------------------------------------------
// decide_serve_action() -- the policy itself.

TEST(DecideServeAction, FirstServerAlwaysServesNowRegardlessOfHistory) {
  EXPECT_EQ(decide_serve_action(ServeRole::kFirstServer, /*completed_any_read=*/false),
           ServeNowOrDefer::kServeNow);
  EXPECT_EQ(decide_serve_action(ServeRole::kFirstServer, /*completed_any_read=*/true),
           ServeNowOrDefer::kServeNow);
}

TEST(DecideServeAction, FirstReaderDefersOnlyBeforeAnyCompletedRead) {
  EXPECT_EQ(decide_serve_action(ServeRole::kFirstReader, /*completed_any_read=*/false),
           ServeNowOrDefer::kDefer);
  EXPECT_EQ(decide_serve_action(ServeRole::kFirstReader, /*completed_any_read=*/true),
           ServeNowOrDefer::kServeNow);
}

// ---------------------------------------------------------------------
// ExpectedOrderForExchange() / CheckAndAdvanceOrder() -- the LOCAL
// per-event assertion. The unit tracked is one write-then-read (or
// read-then-write) pair (an exchange), not a coupling window
// (lowfive_attach_decision.hpp's own comment has the account).

TEST(ExpectedOrderForExchange, ColdStartIsWriteReadOnBothRoles) {
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstServer, 0)[0], ExpectedEventKind::kWrite);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstServer, 0)[1], ExpectedEventKind::kRead);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstReader, 0)[0], ExpectedEventKind::kWrite);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstReader, 0)[1], ExpectedEventKind::kRead);
}

TEST(ExpectedOrderForExchange, SteadyStateIsWriteReadForFirstServer) {
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstServer, 1)[0], ExpectedEventKind::kWrite);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstServer, 1)[1], ExpectedEventKind::kRead);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstServer, 60)[0], ExpectedEventKind::kWrite);
}

TEST(ExpectedOrderForExchange, SteadyStateIsReadWriteForFirstReader) {
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstReader, 1)[0], ExpectedEventKind::kRead);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstReader, 1)[1], ExpectedEventKind::kWrite);
  EXPECT_EQ(ExpectedOrderForExchange(ServeRole::kFirstReader, 60)[0], ExpectedEventKind::kRead);
}

TEST(CheckAndAdvanceOrder, FirstServerMeasuredSequenceAcceptedAcrossExchanges) {
  OrderTracker t;
  for (int exchange = 0; exchange < 3; ++exchange) {
    auto w = CheckAndAdvanceOrder(ServeRole::kFirstServer, t, ServeEvent::kExportWrittenAndClosed);
    ASSERT_TRUE(w.ok()) << w.error().code;
    t = w.value();
    auto about = CheckAndAdvanceOrder(ServeRole::kFirstServer, t, ServeEvent::kImportAboutToOpen);
    ASSERT_TRUE(about.ok()) << about.error().code;
    t = about.value();
    auto closed = CheckAndAdvanceOrder(ServeRole::kFirstServer, t, ServeEvent::kImportClosed);
    ASSERT_TRUE(closed.ok()) << closed.error().code;
    t = closed.value();
    EXPECT_EQ(t.exchange_index, exchange + 1);
    EXPECT_EQ(t.slot, 0);
  }
}

TEST(CheckAndAdvanceOrder, FirstReaderColdStartThenSteadyAcceptedAcrossExchanges) {
  OrderTracker t;  // exchange 0: write, read (cold start -- same as first-server)
  auto w0 = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(w0.ok()) << w0.error().code;
  t = w0.value();
  auto r0 = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kImportClosed);
  ASSERT_TRUE(r0.ok()) << r0.error().code;
  t = r0.value();
  EXPECT_EQ(t.exchange_index, 1);

  // exchange 1+: read, write (steady state).
  for (int exchange = 1; exchange < 4; ++exchange) {
    auto r = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kImportAboutToOpen);
    ASSERT_TRUE(r.ok()) << r.error().code;
    t = r.value();
    auto rc = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kImportClosed);
    ASSERT_TRUE(rc.ok()) << rc.error().code;
    t = rc.value();
    auto w = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kExportWrittenAndClosed);
    ASSERT_TRUE(w.ok()) << w.error().code;
    t = w.value();
    EXPECT_EQ(t.exchange_index, exchange + 1);
  }
}

TEST(CheckAndAdvanceOrder, FirstReaderWritingTwiceInSteadyStateIsRefused) {
  // exchange 0 (write,read) done; exchange 1 steady state expects read
  // first -- writing again here is the exact regression this check
  // exists to catch (the measured host hang's own cause, one layer up
  // from decide_serve_action() itself).
  OrderTracker t;
  auto w0 = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kExportWrittenAndClosed);
  ASSERT_TRUE(w0.ok());
  t = w0.value();
  auto r0 = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kImportClosed);
  ASSERT_TRUE(r0.ok());
  t = r0.value();

  const auto bad = CheckAndAdvanceOrder(ServeRole::kFirstReader, t, ServeEvent::kExportWrittenAndClosed);
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, "E-LOWFIVE-ATTACH-ORDER-DEVIATION");
}

TEST(CheckAndAdvanceOrder, FirstServerReadingBeforeWritingInSteadyStateIsRefused) {
  OrderTracker t{/*exchange_index=*/1, /*slot=*/0};  // steady state, expects write first
  const auto bad = CheckAndAdvanceOrder(ServeRole::kFirstServer, t, ServeEvent::kImportAboutToOpen);
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, "E-LOWFIVE-ATTACH-ORDER-DEVIATION");
}

TEST(CheckAndAdvanceOrder, FinalizeNeverChangesTheTracker) {
  OrderTracker t{2, 1};
  const auto r = CheckAndAdvanceOrder(ServeRole::kFirstServer, t, ServeEvent::kFinalize);
  ASSERT_TRUE(r.ok());
  EXPECT_EQ(r.value().exchange_index, 2);
  EXPECT_EQ(r.value().slot, 1);
}

// ---------------------------------------------------------------------
// ComputeGraphOrderSignature() / CheckGraphOrderValidated() -- the STATIC
// check, one refusal test per property (channel writer/reader name,
// channel count, rank count).

namespace {
std::vector<ChannelRoles> MeasuredChannels() {
  ChannelRoles u2v;
  u2v.field_name = "U.u_out";
  u2v.writer_ranks = RankSet{0, 1};
  u2v.reader_ranks = RankSet{1, 1};
  u2v.writer_name = "U";
  u2v.reader_name = "V";
  ChannelRoles v2u;
  v2u.field_name = "V.v_out";
  v2u.writer_ranks = RankSet{1, 1};
  v2u.reader_ranks = RankSet{0, 1};
  v2u.writer_name = "V";
  v2u.reader_name = "U";
  return {u2v, v2u};
}
}  // namespace

TEST(ComputeGraphOrderSignature, MeasuredChannelsProduceTheValidatedSignature) {
  const auto sig = ComputeGraphOrderSignature(MeasuredChannels());
  EXPECT_TRUE(CheckGraphOrderValidated(sig).ok());
}

TEST(CheckGraphOrderValidated, SwappedChannelOrderIsRefused) {
  auto channels = MeasuredChannels();
  std::swap(channels[0], channels[1]);  // v2u first, u2v second
  const auto sig = ComputeGraphOrderSignature(channels);
  const auto r = CheckGraphOrderValidated(sig);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED");
}

TEST(CheckGraphOrderValidated, DifferentComponentNamesAreRefused) {
  auto channels = MeasuredChannels();
  channels[0].writer_name = "X";
  const auto sig = ComputeGraphOrderSignature(channels);
  const auto r = CheckGraphOrderValidated(sig);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED");
}

TEST(CheckGraphOrderValidated, ThirdChannelIsRefused) {
  auto channels = MeasuredChannels();
  channels.push_back(channels[0]);  // same pair again: 3 channels total, mismatched count
  const auto sig = ComputeGraphOrderSignature(channels);
  const auto r = CheckGraphOrderValidated(sig);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED");
}

TEST(CheckGraphOrderValidated, MultiRankComponentIsRefused) {
  auto channels = MeasuredChannels();
  channels[0].writer_ranks = RankSet{0, 2};  // U now has 2 ranks, never measured
  const auto sig = ComputeGraphOrderSignature(channels);
  const auto r = CheckGraphOrderValidated(sig);
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED");
}

// ---------------------------------------------------------------------
// The event-sequence-driven model check. Scenarios
// (i)-(iv), each with its own assertion, plus the soundness link
// (static_check_accepts(order) implies model_completes(order)).

TEST(ModelCheck, MeasuredOrderCompletesFor1And2And60And122And25Exchanges) {
  // Scenario (i). 25 is the measured per-rank write count (U and V both,
  // symmetric) of a completed `--periods 2` run; 122 is an earlier
  // measured figure (U writes of a `--periods 2` run) that is no longer
  // the current count. Neither is derived here (that would need the
  // Runner's own sweep-count machinery); the existing N-generic machinery
  // covers both without any new scenario type
  // (lowfive_attach_decision.hpp's own comment on the measured order has
  // the account). 60 is ALSO covered as a stress value (this row's own
  // kMaxIterations, reused as an exchange count, not a claim it equals a
  // Jacobi sweep count).
  for (int n : {1, 2, 25, 60, 122}) {
    const auto r = RunSimulationForOrder(ExchangeOrder::kMeasured, n,
                                       ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                       ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
    EXPECT_TRUE(r.completed) << "n=" << n << " " << (r.stuck_example ? *r.stuck_example : "");
  }
}

TEST(ModelCheck, MeasuredOrderUnderRegressedPolicyReproducesTheMeasuredHang) {
  // Scenario (ii) -- the DIRECT regression test for the host's own
  // measured 120 s timeout: the SAME order the real graph produces,
  // but a falsified "always defer" policy instead of
  // decide_serve_action(). Must get stuck, and the stuck position
  // must be U in read(v#1) (first-server's own kOpenClose at
  // exchange 1) with the first-reader parked at its own exchange-1
  // sync -- exactly the measured cycle.
  const auto r = RunSimulationForOrder(
      ExchangeOrder::kMeasured, 2,
      ProcessPolicy{PolicyKind::kRegressedAlwaysDeferFirstReader, ServeRole::kFirstServer},
      ProcessPolicy{PolicyKind::kRegressedAlwaysDeferFirstReader, ServeRole::kFirstReader});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

TEST(ModelCheck, BothSidesWriteReadSteadyGetsStuck) {
  // Scenario (iii): both components happen to write-then-read in
  // steady state (a shape no real graph under the current channel-
  // order mechanism can produce) -- the shipped
  // policy, applied to this hypothetical shape, reports the
  // mutual-serve stuck state: both block on their OWN write's serve
  // at the SAME exchange, neither has reached the read that would
  // release the other.
  const auto r = RunSimulationForOrder(ExchangeOrder::kBothWriteReadSteady, 3,
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

TEST(ModelCheck, BothSidesReadWriteSteadyGetsStuck) {
  // Scenario (iv): nobody writes first in steady state -- also a
  // shape no real graph produces; reported stuck (neither side's read
  // can find anything served yet).
  const auto r = RunSimulationForOrder(ExchangeOrder::kBothReadWriteSteady, 3,
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                     ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

TEST(ModelCheck, StaticCheckAcceptsImpliesModelCompletesForEveryCatalogueEntry) {
  // The soundness link: the static check must never validate
  // an order the model shows stuck. Checked against EVERY catalogue
  // entry, not only the one real graph -- static_check_accepts() is
  // false for the two strawman shapes, so the implication holds there
  // vacuously; it is TRUE (and must be matched by completion) only for
  // kMeasured.
  for (ExchangeOrder order :
       {ExchangeOrder::kMeasured, ExchangeOrder::kBothWriteReadSteady, ExchangeOrder::kBothReadWriteSteady}) {
    const bool accepts = static_check_accepts(order);
    const bool completes = model_completes(order, 3);
    EXPECT_TRUE(!accepts || completes)
        << "order=" << static_cast<int>(order) << " accepts=" << accepts
        << " completes=" << completes;
  }
}

TEST(ModelCheck, MeasuredOrderIsTheOnlyAcceptedCatalogueEntry) {
  EXPECT_TRUE(static_check_accepts(ExchangeOrder::kMeasured));
  EXPECT_FALSE(static_check_accepts(ExchangeOrder::kBothWriteReadSteady));
  EXPECT_FALSE(static_check_accepts(ExchangeOrder::kBothReadWriteSteady));
}

// ---------------------------------------------------------------------
// BuildEventSequence() itself -- the per-component event sequences are
// DATA (a list of events), pinned directly.

TEST(BuildEventSequence, FirstServerIsWriteReadEveryExchangeIncludingColdStart) {
  const auto seq = BuildEventSequence(ExchangeOrder::kMeasured, ServeRole::kFirstServer, 3);
  const std::vector<EventKind> expected{
      EventKind::kWrite, EventKind::kRead,                                // exchange 0 (cold start)
      EventKind::kWrite, EventKind::kRead, EventKind::kSync,              // exchange 1
      EventKind::kWrite, EventKind::kRead, EventKind::kSync};             // exchange 2
  EXPECT_EQ(seq.size(), expected.size());
  for (std::size_t i = 0; i < expected.size() && i < seq.size(); ++i) {
    EXPECT_EQ(seq[i], expected[i]) << "index " << i;
  }
}

TEST(BuildEventSequence, FirstReaderIsWriteReadAtColdStartThenReadWriteSteady) {
  const auto seq = BuildEventSequence(ExchangeOrder::kMeasured, ServeRole::kFirstReader, 3);
  const std::vector<EventKind> expected{
      EventKind::kWrite, EventKind::kRead,                                // exchange 0 (cold start)
      EventKind::kRead, EventKind::kWrite, EventKind::kSync,              // exchange 1 (steady)
      EventKind::kRead, EventKind::kWrite, EventKind::kSync};             // exchange 2
  EXPECT_EQ(seq.size(), expected.size());
  for (std::size_t i = 0; i < expected.size() && i < seq.size(); ++i) {
    EXPECT_EQ(seq[i], expected[i]) << "index " << i;
  }
}

TEST(BuildEventSequence, ZeroExchangesIsEmpty) {
  EXPECT_TRUE(BuildEventSequence(ExchangeOrder::kMeasured, ServeRole::kFirstServer, 0).empty());
}

// ---------------------------------------------------------------------
// The coupler exit: which teardown steps run for which role, as a PURE
// function (decide_exit_hygiene_plan()), and the literal env-route
// variable list (EnvRouteVolVariableNames()) coupler_main.c's own
// unsetenv() calls mirror by hand (the header's own comment has the
// crash trace, root cause, and fix reasoning).

TEST(EnvRouteVolVariableNames, ListsExactlyTheThreeEnvRouteVariables) {
  const auto names = EnvRouteVolVariableNames();
  ASSERT_EQ(names.size(), 3u);
  EXPECT_EQ(names[0], "HDF5_VOL_CONNECTOR");
  EXPECT_EQ(names[1], "HDF5_PLUGIN_PATH");
  EXPECT_EQ(names[2], "FUSE_LOWFIVE_BINDINGS");
}

TEST(DecideExitHygienePlan, NonAttachingCoordinatorUnregistersAndSkipsExplicitH5close) {
  const auto plan = decide_exit_hygiene_plan(LowFiveProcessRole::kNonAttachingCoordinator);
  EXPECT_TRUE(plan.unregister_env_route_vol);
  EXPECT_FALSE(plan.explicit_h5close_before_mpi_finalize);
}

TEST(DecideExitHygienePlan, AttachingParticipantCallsExplicitH5closeAndDoesNotUnregister) {
  const auto plan = decide_exit_hygiene_plan(LowFiveProcessRole::kAttachingParticipant);
  EXPECT_FALSE(plan.unregister_env_route_vol);
  EXPECT_TRUE(plan.explicit_h5close_before_mpi_finalize);
}


// The accept-layout scenario -- lowfive_attach_model_check.hpp's own
// "RunSimulationAcceptLayout"
// comment has the full account of why this reuses RunSimulationForOrder's
// existing third-actor machinery under an explicit name, rather than a
// new, duplicate actor type.
TEST(ModelCheck, AcceptLayoutPlaceholderJoiningCompletesLikeTheCouplerScenario) {
  for (int n : {1, 2, 25}) {
    const auto r = fuse_examples::lowfive_attach_model::RunSimulationAcceptLayout(
        ExchangeOrder::kMeasured, n, ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
        ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader},
        /*third_actor_joins=*/true);
    EXPECT_TRUE(r.completed) << "n=" << n << " " << (r.stuck_example ? *r.stuck_example : "");
  }
}

// The known-bad placement: a real third actor (the bundled
// coupler/placeholder rank) that EXISTS but never joins the collective at
// all. This is the model-level mirror of
// test/verification/brusselator_lowfive_attach_test.cpp's own
// BundledPlaceholderRankSeesNoChannelAtAll (CollectChannelsForRank)
// test: here it is the SYNC collective, not channel membership, that
// is missing a participant.
TEST(ModelCheck, AcceptLayoutPlaceholderNeverJoiningGetsStuck) {
  const auto r = fuse_examples::lowfive_attach_model::RunSimulationAcceptLayout(
      ExchangeOrder::kMeasured, /*n_exchanges=*/2,
      ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
      ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader},
      /*third_actor_joins=*/false);
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// n_exchanges=1 never reaches a kSyncBarrier at all (BuildEventSequence()
// 's own cold-start-only shape for n<=1) -- the third actor's own
// non-participation therefore has nothing to get stuck ON, a boundary
// stated explicitly (a test that cannot fail is not written) rather than
// silently merged into the n=2 case above.
TEST(ModelCheck, AcceptLayoutPlaceholderNeverJoiningStillCompletesAtOneExchange) {
  const auto r = fuse_examples::lowfive_attach_model::RunSimulationAcceptLayout(
      ExchangeOrder::kMeasured, /*n_exchanges=*/1,
      ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
      ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader},
      /*third_actor_joins=*/false);
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}
