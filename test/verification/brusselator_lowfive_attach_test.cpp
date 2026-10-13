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
#include "lowfive_attach_model_check.hpp"

#include "fuse/model/component.hpp"
#include "fuse/model/launch.hpp"

using fuse::LaunchMode;
using fuse::RankSet;
using fuse_examples::lowfive_attach::AttachMode;
using fuse_examples::lowfive_attach::ChannelRoles;
using fuse_examples::lowfive_attach::decide_attach_mode;
using fuse_examples::lowfive_attach::decide_serve_roles;
using fuse_examples::lowfive_attach::BuildIntercommRegistrations;
using fuse_examples::lowfive_attach::IntercommRegistration;
using fuse_examples::lowfive_attach::NormalizeDatasetPattern;
using fuse_examples::lowfive_attach::ServeAction;
using fuse_examples::lowfive_attach::ServeEvent;
using fuse_examples::lowfive_attach::ServeRole;
using fuse_examples::lowfive_attach::ServeState;

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

// ---------------------------------------------------------------------
// Round 2 ("role-ordered serve"): decide_serve_roles(). See lowfive_
// attach_decision.hpp's own "Round 2" comment for the measured deadlock
// this exists to avoid.

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
// Round 2 REVIEW: serve_state_step(). The event model changed from the
// first round-2 attempt -- kImportAboutToOpen is now ALWAYS kReceiveOnly
// (never flushes, any role or state) and a NEW event, kImportClosed,
// is where a first-reader's own deferred serve is flushed instead. Every
// test below that exercised the OLD (wrong) kReceiveThenFlushDeferred-
// at-kImportAboutToOpen expectation is REPLACED, not deleted outright --
// FirstReaderDefersThenReceivesAndFlushesRepeatedly below is the direct
// replacement for the test of the same name that asserted the bug's own
// expectation, now asserting the corrected one instead (see this file's
// own git history / the round-2-review report for the diff). Exercises
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

// Replaces the test of the same name from the first round-2 attempt,
// which asserted kImportAboutToOpen -> kReceiveThenFlushDeferred (the
// bug this review caught: flushing before this process's own open-and-
// close of the peer's file, not after). The corrected cycle: defer at
// write, plain receive at the pre-open hook (NOT a flush), flush only
// at the new post-close hook.
TEST(ServeStateStep, FirstReaderDefersThenReceivesAndFlushesRepeatedly) {
  ServeState st;
  st.role = ServeRole::kFirstReader;
  for (int iter = 0; iter < 3; ++iter) {
    const auto a1 = serve_state_step(&st, ServeEvent::kExportWrittenAndClosed);
    ASSERT_TRUE(a1.ok()) << a1.error().code;
    EXPECT_EQ(a1.value(), ServeAction::kDeferServe);

    const auto a2 = serve_state_step(&st, ServeEvent::kImportAboutToOpen);
    ASSERT_TRUE(a2.ok()) << a2.error().code;
    EXPECT_EQ(a2.value(), ServeAction::kReceiveOnly);  // NOT a flush -- see comment above

    const auto a3 = serve_state_step(&st, ServeEvent::kImportClosed);
    ASSERT_TRUE(a3.ok()) << a3.error().code;
    EXPECT_EQ(a3.value(), ServeAction::kFlushDeferred);  // flushed HERE instead
  }
  const auto fin = serve_state_step(&st, ServeEvent::kFinalize);
  ASSERT_TRUE(fin.ok()) << fin.error().code;
  EXPECT_EQ(fin.value(), ServeAction::kNoopFinalizeClean);
}

// The DIRECT regression test for the exact bug this review caught:
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

  // A second export with no intervening kImportClosed -- I1 violated.
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
// Round 2 REVIEW: model-check tests. lowfive_attach_model_check.hpp's
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

TEST(ModelCheck, ShippedProtocolCompletesFor2Iterations) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(2, ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// 60: this row's own kMaxIterations (brusselator_graph_tool.cpp).
TEST(ModelCheck, ShippedProtocolCompletesFor60Iterations) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(60, ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer},
                                ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// Which SEARCH POSITION (A or B) carries the first-server role must not
// matter -- the search itself has no preferred side.
TEST(ModelCheck, ShippedProtocolCompletesRegardlessOfWhichPositionIsFirstServer) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r = RunSimulation(3, ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstReader},
                                ProcessPolicy{PolicyKind::kShipped, ServeRole::kFirstServer});
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}

// The REJECTED first round-2 attempt (flush at kAboutToOpen instead of
// kImportClosed) -- illustrative evidence that this search is
// discriminating, not evidence about the shipped protocol (which the
// tests above alone speak for). Expressed as a canned policy, NOT
// through serve_state_step() (today's own function can no longer take
// this path at all, by this round's own fix).
TEST(ModelCheck, RejectedPreOpenFlushPlacementDeadlocks) {
  using namespace fuse_examples::lowfive_attach_model;
  const auto r =
      RunSimulation(1, ProcessPolicy{PolicyKind::kRejectedPreOpenFlush, ServeRole::kFirstServer},
                    ProcessPolicy{PolicyKind::kRejectedPreOpenFlush, ServeRole::kFirstReader});
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// Round 1's own "natural placement" (serve at every write, both sides,
// never defer) -- illustrative; round 1 never tested this directly
// because this header did not exist yet.
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
// Round 4: DEFECT A (missing LowFive::DistMetadataVOL::set_intercomm()
// registration) and DEFECT B (fuse::split_hdf5_path()'s own unslashed
// dataset component never matching LowFive::Object::fullname()'s own
// slashed form) -- lowfive_attach_decision.hpp's own "ROUND 4" comment
// has the full citations for both, confirmed by reading LowFive's
// source before writing either fix.

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
// Round 4: the model check's own DEFECT B case (ProcessPolicy::
// nothing_indexed) -- "a protocol that relies on a serve that never
// serves is a stuck/diverging state, not a silent pass."

TEST(ModelCheck, ShippedProtocolWithNothingIndexedGetsStuck) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer, /*nothing_indexed=*/true};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader, /*nothing_indexed=*/true};
  const auto r = RunSimulation(1, server, reader);
  EXPECT_FALSE(r.completed);
  ASSERT_TRUE(r.stuck_example.has_value());
}

// Sanity check on the flag itself: leaving it at its own default (false)
// must reproduce the ordinary, already-passing shipped-protocol result
// -- i.e. this new field cannot have silently changed the nominal case.
TEST(ModelCheck, ShippedProtocolDefaultsToIndexedAndStillCompletes) {
  using namespace fuse_examples::lowfive_attach_model;
  ProcessPolicy server{PolicyKind::kShipped, ServeRole::kFirstServer};
  ProcessPolicy reader{PolicyKind::kShipped, ServeRole::kFirstReader};
  EXPECT_FALSE(server.nothing_indexed);
  EXPECT_FALSE(reader.nothing_indexed);
  const auto r = RunSimulation(2, server, reader);
  EXPECT_TRUE(r.completed) << (r.stuck_example ? *r.stuck_example : "");
}
