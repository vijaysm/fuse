#pragma once

// S6 phase 3 (Conformance, "LowFive" track), "LowFive attach" track: the
// PURE, MPI-free and HDF5-free half of the participants' programmatic VOL
// attach (lowfive_attach.cpp's own header comment has the full account of
// what wraps this). Decides, from the loaded coupling graph alone, which
// of two outcomes this rank's own attach call must reach, or refuses:
//
//   - every HDF5-bound channel this rank takes part in is Mpmd-launched on
//     both ends -> attach a DistMetadataVOL over an intercommunicator this
//     process still has to form (fuse::plan_lowfive_rendezvous(), already
//     tier-0 tested, supplies the MPI_Comm_split/MPI_Intercomm_create
//     arguments);
//   - no HDF5-bound channel names this rank at all -> refused
//     (E-LOWFIVE-ATTACH-NOCHANNELS): a caller that reaches this file at
//     all already decided (from --transport lowfive) that SOME field
//     needs interception, so a rank with no part in any of them is a
//     graph/launch mismatch, not a legitimate no-op -- the charter's own
//     "no silent fallback";
//   - any HDF5-bound channel this rank takes part in has an ACCEPT-mode
//     (LaunchMode::Connect) component on either end -> refused clearly
//     (E-LOWFIVE-ATTACH-ACCEPT; docs/user/lowfive.md, KNOWN_ISSUES.md
//     D14's own accept-mode intercommunicator question, still open) --
//     never a half-configured VOL;
//   - two or more HDF5-bound channels this rank takes part in do not all
//     partition the world into the SAME two groups -> refused
//     (E-LOWFIVE-ATTACH-INCONSISTENT): src/transport/lowfive_plugin_
//     graph.cpp's own ConfigureDistributed() comment names this exact gap
//     ("a graph with more than one [channel] would need this loop to
//     confirm every ... channel agrees on the SAME writer/reader pair
//     this process belongs to, which is not yet checked here") -- this
//     function is the check that comment asked for, applied to the
//     participants' own bidirectional (u_out/v_out) exchange, where the
//     writer/reader LABELS swap between the two channels but the two
//     GROUPS {U-ranks},{V-ranks} do not.
//
// Nothing here touches MPI or HDF5; every type is fuse::model/fuse::
// transport header-only arithmetic, exercised directly by test/
// verification/brusselator_lowfive_attach_test.cpp with no LowFive, no
// HDF5, and no MPI (mirrors the "force_this_process, no real
// communicator" tier-0 discipline KNOWN_ISSUES.md C1 already documents
// for this codebase's other MPI-shaped pure functions).

#include <cstddef>
#include <string>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/component.hpp"
#include "fuse/model/launch.hpp"
#include "fuse/transport/lowfive.hpp"

namespace fuse_examples::lowfive_attach {

/// One HDF5-bound channel's writer/reader roles, as read from the graph --
/// `field_name` is kept only so a refusal message can name something a
/// human can find in the YAML (the SOURCE field's own qualified name,
/// `fuse::build_lowfive_bindings()`'s own "<component>.<field>"
/// convention).
struct ChannelRoles {
  std::string field_name;
  fuse::RankSet writer_ranks;
  fuse::RankSet reader_ranks;
  fuse::LaunchMode writer_launch = fuse::LaunchMode::InProcess;
  fuse::LaunchMode reader_launch = fuse::LaunchMode::InProcess;
};

enum class AttachMode { kSingleProcess, kDistributed };

struct Decision {
  AttachMode mode = AttachMode::kSingleProcess;
  /// Meaningful only when `mode == kDistributed`.
  fuse::LowFiveRendezvousPlan rendezvous;
};

/// `channels` must already be filtered to the ones naming `this_rank` on
/// either side (lowfive_attach.cpp's own job, mirroring the existing
/// env-route plugin's `ConfigureDistributed()` loop in
/// src/transport/lowfive_plugin_graph.cpp) -- an empty list is refused
/// here, not treated as "nothing to do." `this_rank` is the base
/// communicator's own rank number (MPI_COMM_WORLD's, in MPMD launch,
/// exactly as lowfive_plugin_graph.cpp's own comment states and this
/// track's MEASURED EVIDENCE confirms).
[[nodiscard]] inline fuse::Result<Decision> decide_attach_mode(
    const std::vector<ChannelRoles>& channels, int this_rank) {
  if (channels.empty()) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-NOCHANNELS",
        "rank " + std::to_string(this_rank) +
            " has --transport lowfive but no HDF5-bound channel in the "
            "loaded graph names it on either side; refusing rather than "
            "silently skipping interception"};
  }

  for (const ChannelRoles& ch : channels) {
    if (ch.writer_launch == fuse::LaunchMode::Connect ||
        ch.reader_launch == fuse::LaunchMode::Connect) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-ACCEPT",
          "channel '" + ch.field_name +
              "' has an accept-mode (LaunchMode::Connect) component on one "
              "end; accept mode's own merged base communicator is never "
              "exposed back to a caller (KNOWN_ISSUES.md D14), so this "
              "attach has no intercommunicator to form for it and refuses "
              "rather than falling back to a half-configured VOL"};
    }
  }

  // Canonical channel: the first one, exactly as the existing env-route
  // plugin's ConfigureDistributed() picks "the first LowFive-bound
  // channel found."
  const ChannelRoles& canon = channels.front();
  auto plan = fuse::plan_lowfive_rendezvous(canon.writer_ranks, canon.reader_ranks, this_rank);
  if (!plan.ok()) return plan.error();

  // Every OTHER qualifying channel must partition the world into the
  // IDENTICAL two groups -- same two RankSets, regardless of which one is
  // labelled writer and which reader (the Brusselator's own bidirectional
  // u_out/v_out exchange swaps the labels between the two channels; see
  // lowfive_attach.cpp's own header comment for why that is still one
  // intercommunicator, not two).
  for (std::size_t i = 1; i < channels.size(); ++i) {
    const ChannelRoles& other = channels[i];
    const bool same_order = other.writer_ranks.same_as(canon.writer_ranks) &&
                            other.reader_ranks.same_as(canon.reader_ranks);
    const bool swapped = other.writer_ranks.same_as(canon.reader_ranks) &&
                         other.reader_ranks.same_as(canon.writer_ranks);
    if (!same_order && !swapped) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-INCONSISTENT",
          "channel '" + other.field_name + "' and channel '" + canon.field_name +
              "' do not partition the world into the same two groups; this "
              "attach forms exactly one intercommunicator per process and "
              "cannot serve a third group"};
    }
  }

  Decision d;
  d.mode = AttachMode::kDistributed;
  d.rendezvous = plan.value();
  return d;
}

}  // namespace fuse_examples::lowfive_attach
