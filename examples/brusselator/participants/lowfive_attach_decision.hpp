#pragma once

// The PURE, MPI-free and HDF5-free half of the participants' programmatic VOL
// attach (lowfive_attach.cpp's own header comment has the full account of
// what wraps this). Decides, from the loaded coupling graph alone, which
// of two outcomes this rank's own attach call must reach, or refuses:
//
//   - every HDF5-bound channel this rank takes part in is Mpmd- or
//     Connect(accept)-launched on both ends -> attach a DistMetadataVOL
//     over an intercommunicator this process still has to form
//     (fuse::plan_lowfive_rendezvous(), already tier-0 tested, supplies
//     the MPI_Comm_split/MPI_Intercomm_create arguments; a Connect-mode
//     channel additionally needs the merged base communicator exposed
//     through the base_communicator() ABI entry -- see
//     AttachBridgeContext below);
//   - no HDF5-bound channel names this rank at all -> refused
//     (E-LOWFIVE-ATTACH-NOCHANNELS): a caller that reaches this file at
//     all already decided (from --transport lowfive) that SOME field
//     needs interception, so a rank with no part in any of them is a
//     graph/launch mismatch, not a legitimate no-op -- no silent
//     fallback;
//   - an HDF5-bound channel this rank takes part in has an ACCEPT-mode
//     (LaunchMode::Connect) component on either end, but the caller's
//     own base communicator was never formed -> refused
//     (E-LOWFIVE-ATTACH-ACCEPT-NO-BRIDGE): this refusal fires only when
//     base_communicator() itself reports `formed = 0`, never
//     unconditionally (docs/user/lowfive.md);
//   - more than two Connect-mode components are declared in the whole
//     graph -> refused (E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY):
//     fuse::transport::connect_pairwise() itself supports exactly two,
//     checked here too in defence of this function's own pure-function
//     contract (no reliance on
//     connect() having already refused upstream);
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
// communicator" tier-0 discipline used for this codebase's other
// MPI-shaped pure functions).

#include <array>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/model/component.hpp"
#include "fuse/model/coupling_graph.hpp"
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
  /// The two COMPONENTS' own declared names ("role-ordered
  /// serve") -- distinct from `field_name`'s own "<component>.<field>"
  /// qualification, needed by `decide_serve_roles()` below, which
  /// assigns a role per COMPONENT, not per field. Empty by default so
  /// existing `decide_attach_mode()` callers/tests that never set them
  /// are unaffected -- that function never reads either one.
  std::string writer_name;
  std::string reader_name;
};

/// The per-rank channel-collection loop, factored out of
/// lowfive_attach.cpp's own body (mirroring src/transport/
/// lowfive_plugin_graph.cpp's own ConfigureDistributed() loop) so a
/// SECOND caller -- bruss_lowfive_join_local_split(), which every rank of
/// the bridge communicator calls, including a bundled coupler/placeholder
/// rank that never attaches -- can reach the IDENTICAL, rank-filtered
/// answer without duplicating the loop. PURE: no MPI, no LowFive, no
/// HDF5, only fuse::model/fuse::CouplingGraph. `this_rank` is the base
/// communicator's own rank number (see decide_attach_mode()'s own doc
/// comment below for which communicator that is, by launch mode).
[[nodiscard]] inline std::vector<ChannelRoles> CollectChannelsForRank(
    const fuse::CouplingGraph& graph, int this_rank) {
  std::vector<ChannelRoles> channels;
  for (std::uint32_t i = 0; i < graph.channel_count(); ++i) {
    const fuse::Channel& ch = graph.channel_at(fuse::ChannelId{i});
    const fuse::FieldSpec& src = graph.field(ch.source);
    if (!src.hdf5_path.has_value()) continue;
    const fuse::ComponentId writer_id = graph.owner_of(ch.source);
    const fuse::ComponentId reader_id = graph.owner_of(ch.target);
    if (!writer_id.valid() || !reader_id.valid()) continue;
    const fuse::ComponentDecl& writer_c = graph.component(writer_id);
    const fuse::ComponentDecl& reader_c = graph.component(reader_id);
    if (!writer_c.ranks.contains(this_rank) && !reader_c.ranks.contains(this_rank)) {
      continue;  // this channel's two sides do not include this rank
    }
    ChannelRoles roles;
    roles.field_name = writer_c.name + "." + src.name;
    roles.writer_ranks = writer_c.ranks;
    roles.reader_ranks = reader_c.ranks;
    roles.writer_launch = writer_c.launch.mode;
    roles.reader_launch = reader_c.launch.mode;
    roles.writer_name = writer_c.name;
    roles.reader_name = reader_c.name;
    channels.push_back(std::move(roles));
  }
  return channels;
}

/// LowFive's `local` communicator is NOT the caller's own pre-existing
/// u_comm/v_comm (that would rest on the unconfirmed assumption that
/// u_comm's relative ordering coincides with the bridge's own). Instead,
/// it is a FRESH split of the bridge communicator itself, keyed
/// by which component's RankSet (from the graph, the SAME sets
/// plan_lowfive_rendezvous() below consumes) this rank belongs to -- so
/// the numbering is, BY CONSTRUCTION of the split, the one
/// plan_lowfive_rendezvous() already expects (`split_key =
/// own.index_of(this_rank)`), never a coincidence argued from a SEPARATE
/// communicator's own, differently-derived numbering.
///
/// PURE: returns a DECISION, never calls MPI_Comm_split itself (the one
/// caller that does, bruss_lowfive_join_local_split() in lowfive_attach.
/// cpp, is the ONLY place `MPI_UNDEFINED`'s real value is known -- this
/// header stays MPI-free). `participates = false` is the answer for
/// EVERY rank this channel's own writer/reader RankSets do not name --
/// the bundled coupler/placeholder (accept mode) or the real coupler
/// (Mpmd mode) alike: BOTH still call bruss_lowfive_join_local_split()
/// (a collective must never be skipped on a locally computed condition --
/// MPI_Comm_split is collective over the WHOLE bridge, coupler included,
/// even though the coupler's own output group is empty), but neither ever
/// reaches bruss_lowfive_attach() itself.
struct LocalSplitChoice {
  bool participates = false;
  /// Meaningful only when `participates` -- the caller's own job to
  /// translate into MPI_UNDEFINED for the other case (this header is
  /// MPI-free, so MPI_UNDEFINED's actual value is never named here).
  int color = 0;
  int key = 0;
};

[[nodiscard]] inline LocalSplitChoice DecideLocalSplitChoice(
    const std::vector<ChannelRoles>& channels, int this_rank) {
  if (channels.empty()) return LocalSplitChoice{};
  // The CANONICAL channel (the first one) is enough: decide_attach_mode()
  // 's own INCONSISTENT check (below) is what guarantees every OTHER
  // qualifying channel partitions the world into the SAME two groups, so
  // the canonical channel's own writer/reader RankSets already name the
  // group this_rank belongs to, for ANY qualifying channel.
  const ChannelRoles& canon = channels.front();
  const bool is_writer = canon.writer_ranks.contains(this_rank);
  const bool is_reader = canon.reader_ranks.contains(this_rank);
  // Defence in depth, not merely trusting the documented precondition
  // ("channels must already be filtered to the ones naming this_rank"
  // -- CollectChannelsForRank()'s own job): a this_rank in NEITHER side
  // of the canonical channel is exactly the shape a bug upstream of
  // this call (a stale channels list, a caller skipping
  // CollectChannelsForRank()) would produce, and "does not participate"
  // is the correct, safe answer for it -- never a crash, never a
  // fabricated color.
  if (!is_writer && !is_reader) return LocalSplitChoice{};
  const fuse::RankSet& own = is_writer ? canon.writer_ranks : canon.reader_ranks;
  LocalSplitChoice out;
  out.participates = true;
  out.color = is_writer ? 0 : 1;
  out.key = own.index_of(this_rank);
  return out;
}

enum class AttachMode { kSingleProcess, kDistributed };

struct Decision {
  AttachMode mode = AttachMode::kSingleProcess;
  /// Meaningful only when `mode == kDistributed`.
  fuse::LowFiveRendezvousPlan rendezvous;
};

/// Accept mode's own merged base communicator is exposed back to a caller
/// (`fuse_base_communicator()`, entries.def), so a Connect-mode channel
/// is handled by the SAME
/// plan_lowfive_rendezvous()/INCONSISTENT logic Mpmd already used --
/// PROVIDED a bridge was actually formed for THIS context and the graph's
/// own topology is one this attach supports. Two refusals apply, each
/// named and each verified only by a test that makes it fire:
///   * `connect_mode_component_count` (every component in the WHOLE
///     graph declaring `LaunchMode::Connect`, not just this rank's own
///     qualifying channels -- `fuse::transport::connect_pairwise()`'s own
///     `order.size() != 2` refusal, is a GRAPH-WIDE
///     property a single rank's own channel list cannot always see: a
///     third Connect-mode component elsewhere in the graph, on a channel
///     this rank takes no part in, would still make `connect()` itself
///     refuse for WHATEVER rank tries to pair with it, but never reach
///     this function to say so). More than two is refused here too, by
///     name, as defence in depth for a caller that reaches this function
///     without having gone through `connect()` first.
///   * `bridge_formed` (`fuse_base_communicator_info::formed`, read by
///     the caller BEFORE calling this function): a Connect-mode channel
///     with no formed bridge has no intercommunicator this attach could
///     possibly build -- exactly the cases where `connect()` itself never
///     succeeded (more than two components, or `fuse_setup()` never given
///     a communicator at all).
struct AttachBridgeContext {
  bool bridge_formed = false;
  int connect_mode_component_count = 0;
};

/// `channels` must already be filtered to the ones naming `this_rank` on
/// either side (lowfive_attach.cpp's own job, mirroring the existing
/// env-route plugin's `ConfigureDistributed()` loop in
/// src/transport/lowfive_plugin_graph.cpp) -- an empty list is refused
/// here, not treated as "nothing to do." `this_rank` is the base
/// communicator's own rank number -- MPI_COMM_WORLD's in Mpmd launch
/// (lowfive_plugin_graph.cpp's own comment) and the EXPOSED BRIDGE's in Connect (accept/connect) launch,
/// per fuse_base_communicator()'s own contract -- never MPI_COMM_WORLD
/// there, whose numbering an accept-mode reader (V) does not share with
/// the merged bridge (MPI_Intercomm_
/// merge's own high/low rule offsets the "high" side's local numbers by
/// the "low" side's own world size).
[[nodiscard]] inline fuse::Result<Decision> decide_attach_mode(
    const std::vector<ChannelRoles>& channels, int this_rank,
    const AttachBridgeContext& bridge = AttachBridgeContext{}) {
  if (channels.empty()) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-NOCHANNELS",
        "rank " + std::to_string(this_rank) +
            " has --transport lowfive but no HDF5-bound channel in the "
            "loaded graph names it on either side; refusing rather than "
            "silently skipping interception"};
  }

  bool any_connect = false;
  for (const ChannelRoles& ch : channels) {
    if (ch.writer_launch == fuse::LaunchMode::Connect ||
        ch.reader_launch == fuse::LaunchMode::Connect) {
      any_connect = true;
      break;
    }
  }
  if (any_connect) {
    if (bridge.connect_mode_component_count > 2) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY",
          "the loaded graph declares " +
              std::to_string(bridge.connect_mode_component_count) +
              " Connect-mode (accept/connect) components; "
              "fuse::transport::connect_pairwise() itself supports exactly "
              "two, and this attach does "
              "not attempt to support more -- connect() itself will already "
              "have refused E-CONNECT-SPEC for whichever rank tried to pair "
              "beyond the first two, but this attach refuses by name too, "
              "for a caller that reaches it without having gone through "
              "connect() first"};
    }
    if (!bridge.bridge_formed) {
      return fuse::Failure{
          "E-LOWFIVE-ATTACH-ACCEPT-NO-BRIDGE",
          "an HDF5-bound channel has an accept-mode (LaunchMode::Connect) "
          "component on one end, but no base communicator was formed for "
          "this context (fuse_base_communicator() reports formed=0) -- "
          "either fuse_setup() was never given a nonzero communicator, or "
          "connect() itself refused before this attach could run; a "
          "Connect-mode channel needs a bridge to exist"};
    }
    // A formed bridge and a supported topology: fall through to the SAME
    // canonical-channel/plan_lowfive_rendezvous()/INCONSISTENT logic Mpmd
    // already used: once the bridge exists, Connect-mode needs no
    // different treatment here.
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


// ---------------------------------------------------------------------
// WHY THE SERVE IS ROLE-ORDERED. With LowFive's default behaviour, both
// components of the bidirectional exchange block inside H5Fclose ->
// LowFive::VOLBase::_file_close -> DistMetadataVOL::file_close
// (vol-dist-metadata.cpp:355-358 in the LowFive checkout this was read
// against, /Users/mahadevan/Code/hdf5/LowFive rev 7f3a58b) ->
// DistMetadataVOL::serve_all (:38-72) -> Index::serve
// (vol-dist/index.cpp:106-167), which opens with a `local.barrier()`
// (:110) and then polls each intercomm's RPC server (`s.probe()`/
// `s.process()`, :145-161) until EVERY one of them reports "finish" --
// which only happens once the PEER actively queries this file
// (Query::get_filenames(), vol-dist/query.cpp:106-139, itself a LOCAL
// broadcast, :125, fed by the SAME RPC round trip over the intercomm).
// `DistMetadataVOL::serve_on_close` is a PUBLIC bool member, default true
// (lowfive/vol-dist-metadata.hpp:33), and `file_close()` calls
// `serve_all()` for every local in-memory file whenever it is true
// (vol-dist-metadata.cpp:355, `serve_on_close && match_any(...)`) -- so a
// serve call placed AFTER H5Fclose would be a NO-OP: the serve has already
// happened INSIDE H5Fclose. On the row's own graph (`iteration: Jacobi`,
// `max_iterations 60`, `acceleration None`), both components export at
// the end of each iteration before either imports -- so both close their
// own write, both block inside `serve_all()` waiting to be QUERIED, and
// neither has yet reached the read that would query the other: mutual
// deadlock (both U and V sit inside H5Fclose, the 120 s ctest TIMEOUT).
//
// So `lowfive_attach.cpp` sets `vol.serve_on_close = false` on the
// constructed `DistMetadataVOL`, so `H5Fclose` never serves on its own;
// serving is an explicit decision this namespace's own pure
// functions make. `decide_serve_roles()` below assigns the two
// components in the exchange one of two roles from their RankSets alone
// (the lower `min_rank()` serves first); `serve_state_step()` is the
// per-process state machine that decides, for each write/read event,
// whether to serve immediately, defer, receive-then-flush a deferred
// serve, or refuse. See lowfive_attach.cpp's own header comment for the
// full account of why this is deadlock-free and lowfive_attach.h for the
// follow-up that is not done (a helper thread cannot be sound without
// patching LowFive itself: the same `local` communicator serves BOTH the
// barrier/ibarrier serve path above and the broadcast consumer path, and
// HDF5 is built Threadsafety: no).

/// Which of the two components in a bidirectional LowFive exchange
/// serves its own export IMMEDIATELY at write time (blocking until the
/// peer reads it), and which one defers its own serve until its own
/// next import, flushing the deferred serve only after receiving the
/// peer's file first. Assigned once, from the two components' own
/// RankSets alone (`component_a`/`component_b`, order-independent) --
/// never from a runtime condition, so every rank of EITHER component
/// reaches the IDENTICAL assignment (the collective-safety requirement
/// `serve_all()`'s own `local.barrier()` places on every rank of a
/// component: the standing rule that a collective must be reached from
/// data every rank shares, never a locally computed condition).
struct ServeRoleAssignment {
  std::string first_server_name;
  std::string first_reader_name;
};

/// Errors: `E-LOWFIVE-ATTACH-SERVE-OVERLAP` (the two components' own
/// RankSets share a rank -- this protocol needs two DISJOINT sides, one
/// per role), `E-LOWFIVE-ATTACH-SERVE-EMPTY` (either RankSet is empty --
/// a component with no ranks can neither serve nor read).
[[nodiscard]] inline fuse::Result<ServeRoleAssignment> decide_serve_roles(
    const std::string& a_name, const fuse::RankSet& a_ranks, const std::string& b_name,
    const fuse::RankSet& b_ranks) {
  if (a_ranks.overlaps(b_ranks)) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-SERVE-OVERLAP",
        "component '" + a_name + "' and component '" + b_name +
            "' own RankSets overlap; the role-ordered serve protocol needs two DISJOINT "
            "sides, one first-server and one first-reader"};
  }
  if (a_ranks.empty() || b_ranks.empty()) {
    return fuse::Failure{"E-LOWFIVE-ATTACH-SERVE-EMPTY",
                        "component '" + (a_ranks.empty() ? a_name : b_name) +
                            "' has an empty RankSet; it can neither serve nor read"};
  }
  ServeRoleAssignment out;
  if (a_ranks.min_rank() < b_ranks.min_rank()) {
    out.first_server_name = a_name;
    out.first_reader_name = b_name;
  } else if (b_ranks.min_rank() < a_ranks.min_rank()) {
    out.first_server_name = b_name;
    out.first_reader_name = a_name;
  } else {
    // UNREACHABLE given RankSet's own semantics, kept only as a
    // deterministic, defensive tie-break rather than an assert: two
    // DISJOINT, NONEMPTY RankSets cannot share a minimum rank (the
    // global minimum of their union belongs to exactly one of them,
    // which is exactly what the overlap check above already rules out
    // sharing). Not exercised by any test -- there is no reachable
    // input that takes this branch (a test for it could not fail).
    if (a_name < b_name) {
      out.first_server_name = a_name;
      out.first_reader_name = b_name;
    } else {
      out.first_server_name = b_name;
      out.first_reader_name = a_name;
    }
  }
  return out;
}

enum class ServeRole { kFirstServer, kFirstReader };

// ---------------------------------------------------------------------
// RECEIVE AND RELEASE. Two facts, read from
// /Users/mahadevan/Code/hdf5/LowFive/src/vol-dist-metadata.cpp (rev
// 7f3a58b), determine where each serve may be flushed:
//
//   - The CONSUMER's H5Fopen of a remote in-memory file IS the receive.
//     `DistMetadataVOL::file_open()` (line 230 on) builds a `Query` and
//     calls the producer's "get_file_hierarchy" RPC (line 278); its own
//     comment says plainly: "force the producer to enter serve_all to
//     get file hierarchy" (lines 255-257). This RPC round trip is what
//     actually moves data across the intercomm -- nothing in shim.cpp's
//     own PRE-open hook does or can substitute for it; `H5Fopen` itself
//     performs it, automatically, as an ordinary VOL callback dispatch.
//   - The consumer's H5Fclose of that remote file is what ENDS the
//     producer's serve. `DistMetadataVOL::file_close()`'s `RemoteFile`
//     branch (line ~333 on) calls `f->obj.call<void>("file_close")`
//     (line 337), then `MPI_Barrier(local)` (line 339), then root calls
//     `f->obj.self_->finish(0)` (line 342) -- `finish(0)` is exactly
//     what lets `Index::serve()`'s own probe/process loop (vol-dist/
//     index.cpp:145-161) exit for that client. The producer's `serve_
//     all()` call therefore blocks until the CONSUMER has done BOTH its
//     own open AND its own close of that specific file -- not until the
//     consumer has merely attempted to open it.
//
// So a first-reader's own deferred serve (a second, explicit
// `serve_all()` for its OWN file) must NOT be flushed in a hook that runs
// BEFORE this process's own H5Fopen of the PEER's file: that is before
// this process has done anything the peer's own pending `serve_all()`
// could possibly be released by. Both sides would then block: the first-
// server is stuck in its original `kServeNow` from its first write,
// waiting for the first-reader's own open+close; the first-reader is
// stuck in the premature flush, waiting for the first-SERVER's
// open+close of the newly-served file -- which the first-server cannot
// perform, because it is still blocked in its own first `kServeNow`.
// Circular wait, both inside `Index::serve`'s probe loop. A state machine
// tested only against its OWN event model (hook call order) cannot see
// this: the peer's serve is released by MY open AND close, not by MY
// merely calling a hook named "before read".
//
// So the flush is fired AFTER this process's own H5Fclose of the peer's
// file -- `kImportClosed`, action `kFlushDeferred` -- so a first-reader's
// own deferred serve is only entered once this process has actually done
// the open+close that, symmetrically, is what WOULD release a peer's own
// pending serve. `kImportAboutToOpen` is a plain, unconditional
// `kReceiveOnly` for every role and every state: it does nothing
// LowFive-level at all (see the next paragraph for why), kept only as a
// validated event in the sequence for symmetry with the real shim.cpp
// call sites and so a mutation that reintroduces a flush there is caught
// by this file's own model check below, not just by a human re-reading
// this comment.
//
// `broadcast_files()` does NOT belong in either hook. Read directly from
// source (src/vol-dist-metadata.cpp:75-103): it operates ENTIRELY over
// `local` (never touches an intercomm), serializing `files` on one local
// rank (`root`, default 0) and broadcasting it to the OTHER ranks of the
// SAME local communicator -- propagating an ALREADY-KNOWN file registry
// across a MULTI-RANK component's own sibling ranks, not communicating
// with the peer component at all. `grep -rn broadcast_files` across this
// LowFive checkout's own `src/`/`include/` finds zero internal callers;
// its only caller anywhere in the repository is the Python binding
// (`bindings/python/lowfive.cpp:126`), a user-facing utility, never
// invoked by `file_open()`/`file_close()` themselves. The Brusselator's
// own `lowfive.mpmd.3` row places exactly one rank per component (U{1},
// V{1}), so even a harmless call would have nothing to propagate to;
// keeping an unjustified collective "for safety" is worse than omitting
// it, since every future reader would have to re-derive this same
// analysis to confirm it is inert. A future multi-rank-per-component
// layout may need an explicit `broadcast_files()` call so that only one
// local rank pays the RPC round trip per file; that is not done here.

/// One process's own write/read lifecycle events, in the SAME terms
/// lowfive_attach.cpp's own bruss_lowfive_serve_after_write()/
/// bruss_lowfive_receive_before_read()/bruss_lowfive_after_read_close()/
/// bruss_lowfive_finalize() see them. `kImportClosed` is fired once per
/// read, after this process's OWN H5Fclose of the peer's file, which is
/// the event LowFive's own source ties the serve release to (this
/// header's own "RECEIVE AND RELEASE" comment above).
enum class ServeEvent { kExportWrittenAndClosed, kImportAboutToOpen, kImportClosed, kFinalize };

/// What `serve_state_step()` decided THIS process must do next. The
/// caller (lowfive_attach.cpp) maps each one onto the real
/// `DistMetadataVOL::serve_all()` call (or onto nothing at all, for
/// `kReceiveOnly`/`kNoopAfterImport`/`kNoopFinalizeClean` -- see this
/// header's own "RECEIVE AND RELEASE" comment for why `broadcast_files()`
/// is not called from either hook).
enum class ServeAction {
  kServeNow,            // kExportWrittenAndClosed, first-server: serve_all() now, blocking
  kDeferServe,          // kExportWrittenAndClosed, first-reader: remember a serve is owed
  kReceiveOnly,         // kImportAboutToOpen (always, any role/state): no LowFive call at all
  kFlushDeferred,       // kImportClosed, first-reader with a deferred serve: serve_all() now, blocking
  kNoopAfterImport,     // kImportClosed, nothing to flush (first-server always; first-reader with nothing pending)
  kNoopFinalizeClean,   // kFinalize, nothing left pending
};

/// Per-process state the machine threads across calls -- exactly the
/// `deferred_pending` bit `serve_state_step()` needs and nothing else.
/// Default-constructed with `role = kFirstServer`; a real caller always
/// sets `role` from `decide_serve_roles()`'s own answer before the first
/// event.
struct ServeState {
  ServeRole role = ServeRole::kFirstServer;
  bool deferred_pending = false;
  // Has this process completed (reached kImportClosed for) ANY
  // read yet -- decide_serve_action()'s own "history" input. False
  // only before this process's own very first completed read (the cold
  // start, on either role); true forever after, on either role (a
  // first-server's own decision never consults it, but it is still
  // maintained uniformly, never role-gated at the point it is SET).
  bool completed_any_read = false;
};

// ---------------------------------------------------------------------
// THE PER-EXCHANGE ORDER DIFFERS BETWEEN THE TWO COMPONENTS. Deferring
// EVERY export a first-reader ever writes, flushed only at its own NEXT
// import's close, holds only at the COLD START, where it happens to
// coincide with "this process has not read anything yet." The measured
// steady-state event sequences (docs/user/lowfive.md) show the two
// components running DIFFERENT per-exchange orders once past the cold
// start: U (first-server) = write, read, sync, every exchange; V
// (first-reader) = write, read ONLY at the cold start (exchange 0) --
// then READ, WRITE, sync from exchange 1 onward, the READ now coming
// FIRST. A policy that deferred V's OWN later writes by `role` alone
// (never looking at anything that happens in between) would stack two
// never-flushed defers against V's own read-then-write steady state and
// produce the mutual-serve cycle (a 120 s timeout).
//
// WHY THE STEADY-STATE ORDER DIFFERS BETWEEN THE TWO COMPONENTS, read
// from source, not inferred from the measured sequences:
// `fuse::CouplingGraph::add_channel()` (include/
// fuse/model/coupling_graph.hpp) assigns `ChannelId` values strictly in
// call order (`channels_.push_back(...); return ChannelId{channels_.
// size() - 1};`); `brusselator::add_window_coupling()` (examples/
// brusselator/brusselator_graph.hpp) calls `g.add_channel("u2v", u_out,
// v_in)` BEFORE `g.add_channel("v2u", v_out, u_in)` -- channel u2v is
// ChannelId{0}, v2u is ChannelId{1}, and `decl.channels = {ch_u2v,
// ch_v2u}` lists them in that SAME order. `Runner::Impl::deliver_
// external()` (src/runtime/iteration.cpp) suspends the EXPORTER of a
// channel's own delivery BEFORE the IMPORTER, for that ONE channel
// ("Suspended FIRST, matching the real time order ('export done')
// precedes 'import due')"); the TWO channels' own ExternalDelivery
// instructions are placed in the lowered DAG by `lower_channel()` (src/
// ir/rules_channel.cpp) with an ordering edge carrying `source_
// producer` as input specifically so the scheduler places each one
// "after the exporting member's own last substep... before the
// importer's own advance" -- the SAME slot `Transport` would occupy for
// an ordinary channel (`OpKind::ExternalDelivery`'s own doc comment,
// include/fuse/ir/op.hpp). Combined: whichever channel is scheduled
// FIRST (u2v, by declaration order) is the one whose EXPORTER sees its
// own suspension first in ITS OWN stream (U: write u_out, i.e. this
// channel's export, comes first) and whose IMPORTER ALSO sees that SAME
// channel's suspension first in ITS OWN stream (V: read u_out, i.e.
// this channel's import, comes first) -- U's own per-exchange order is
// therefore [write(u_out), read(v_out)] and V's is [read(u_out),
// write(v_out)], matching the measured sequences exactly. COMPONENT
// ORDER and RANK SETS do not independently enter this: `decl.channels`
// is listed BY THE CALLER, not derived from component declaration
// order (the two happen to agree in this graph, but are logically
// separate knobs); rank sets decide which process hosts which half of
// a suspension pair (already covered by `exporter_hosted`/`importer_
// hosted` in `deliver_external()`), not the RELATIVE order of the two
// channels' own suspensions. ITERATION KIND (Jacobi) and ACCELERATION
// KIND do not change this mechanism either -- the ordering-edge
// placement in `lower_channel()` is applied identically regardless of
// `WaveformIteration` kind, per the same cited comment.
//
// EXCHANGE vs. PERIOD. The unit this header and the functions below
// track is ONE write-then-read (or read-then-write) PAIR -- an EXCHANGE
// -- not a coupling WINDOW. ExternalDelivery fires once per Jacobi SWEEP,
// not once per coupling WINDOW: with the real species physics a window
// converges in 12 sweeps, so a `--periods 2` run completes with 25
// writes and 25 reads per rank (1 cold start + 12 sweeps x 2 periods = 25
// exchanges each way; the collective trace shows 12 sweeps per period on
// all three ranks). "period" is reserved, in this file and in
// docs/user/lowfive.md, for the Brusselator's own `--periods` CLI flag
// and u_main.c's/v_main.f90's own outer loop -- a coupling WINDOW -- and
// is never used for the write/read-pair unit tracked here. The sweep
// count is a property of the physics' convergence, not derived from
// first principles here; the order-check below does not depend on it.
//
// Cold start is a SEPARATE, PRE-LOOP priming call on BOTH mains
// (u_main.c, v_main.f90's own mirror), not exchange 0 of either
// component's own `fuse_begin_step()`/`fuse_end_step()` loop -- it calls
// `u_lowfive_write_export()`/`u_lowfive_read_import()` directly, with no
// step request/report and no coupler sync of any kind. Both mains' own
// cold-start call order is write-then-read, unconditionally, by
// construction (hardcoded call order in the C/Fortran source, not
// derived from the IR schedule at all) -- this is WHY the cold start
// shows write-then-read for BOTH components even though V's own
// STEADY-STATE order (inside the loop, IR-schedule-derived) is
// read-then-write: two independent mechanisms that happen to produce the
// identical write-then-read SHAPE at the cold start for unrelated
// reasons (U's: the IR schedule, same as every later exchange; V's: the
// priming call's own hardcoded order, true ONLY at the cold start). The
// model below (and ExpectedOrderForExchange()'s own exchange_index == 0
// special case) treats the cold start as "exchange 0" for bookkeeping,
// while it is mechanically a different code path.
enum class ServeNowOrDefer { kServeNow, kDefer };

/// PURE POLICY: a write is deferred ONLY if this process is
/// the first-reader and has not yet completed ANY read (the cold
/// start); every other write -- first-server, always, and first-reader
/// once past its own cold start -- serves immediately ("first-reader always
/// defers" is falsified by the measured steady-state order; see this
/// header's own "THE PER-EXCHANGE ORDER DIFFERS" comment above).
[[nodiscard]] inline ServeNowOrDefer decide_serve_action(ServeRole role, bool completed_any_read) {
  if (role == ServeRole::kFirstReader && !completed_any_read) return ServeNowOrDefer::kDefer;
  return ServeNowOrDefer::kServeNow;
}

/// One top-level event kind in a process's own per-exchange order
/// -- distinct from `ServeEvent`, which is
/// LowFive-hook-shaped (an import is TWO ServeEvents,
/// kImportAboutToOpen then kImportClosed); this is the coarser,
/// order-table-shaped kind each hook's own event maps onto for the
/// purpose of checking against the measured order.
enum class ExpectedEventKind { kWrite, kRead };

/// PURE: this role's own two per-exchange
/// slots, for the ONE measured, validated configuration -- cold start
/// (`exchange_index == 0`, EITHER role) and first-server (every
/// exchange) are [write, read]; first-reader's own STEADY state
/// (`exchange_index >= 1`) is [read, write]. See this header's own
/// "THE PER-EXCHANGE ORDER DIFFERS" comments above for the full
/// derivation and citations.
[[nodiscard]] inline std::array<ExpectedEventKind, 2> ExpectedOrderForExchange(
    ServeRole role, int exchange_index) {
  if (exchange_index == 0 || role == ServeRole::kFirstServer) {
    return {ExpectedEventKind::kWrite, ExpectedEventKind::kRead};
  }
  return {ExpectedEventKind::kRead, ExpectedEventKind::kWrite};
}

/// Per-process position within `ExpectedOrderForExchange()`'s own
/// table -- `slot` is 0 or 1 within the CURRENT `exchange_index`; an
/// exchange completes (exchange_index increments, slot resets to 0)
/// once slot 1's own event has been observed, whichever top-level kind
/// (write or read) that happens to be for this role/exchange.
struct OrderTracker {
  int exchange_index = 0;
  int slot = 0;
};

/// PURE: checks `event` against `ExpectedOrderForExchange(role,
/// tracker.exchange_index)[tracker.slot]` and, if it matches, returns
/// the ADVANCED tracker (never mutates its argument -- this header's
/// own "PURITY" convention elsewhere); refuses
/// (`E-LOWFIVE-ATTACH-ORDER-DEVIATION`), naming the expected event, the
/// observed one, and the exchange index, on a mismatch.
/// `kImportAboutToOpen` and `kImportClosed` both check against the SAME
/// `kRead` slot (a read is one logical event spanning two ServeEvents)
/// but only `kImportClosed` (and `kExportWrittenAndClosed`) ADVANCE the
/// tracker -- `kImportAboutToOpen` only checks, since advancing early
/// would let a SECOND write slip in without its own read ever
/// completing. `kFinalize` is not a per-exchange slot at all and
/// always leaves the tracker unchanged. Called BEFORE `serve_state_
/// step()` at every real call site (lowfive_attach.cpp), so a
/// deviation is caught before the blocking call `serve_state_step()`'s
/// own answer might otherwise enter -- never after.
[[nodiscard]] inline fuse::Result<OrderTracker> CheckAndAdvanceOrder(ServeRole role,
                                                                     OrderTracker tracker,
                                                                     ServeEvent event) {
  const std::array<ExpectedEventKind, 2> expected =
      ExpectedOrderForExchange(role, tracker.exchange_index);
  auto refuse = [&](ExpectedEventKind want, ExpectedEventKind got) -> fuse::Failure {
    auto name = [](ExpectedEventKind k) { return k == ExpectedEventKind::kWrite ? "write" : "read"; };
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-ORDER-DEVIATION",
        "exchange " + std::to_string(tracker.exchange_index) + ", slot " +
            std::to_string(tracker.slot) + ": expected " + std::string(name(want)) +
            ", observed " + std::string(name(got)) +
            " -- the serve policy has only been validated against the measured order"};
  };

  switch (event) {
    case ServeEvent::kExportWrittenAndClosed: {
      if (expected[tracker.slot] != ExpectedEventKind::kWrite) {
        return refuse(expected[tracker.slot], ExpectedEventKind::kWrite);
      }
      OrderTracker next = tracker;
      if (next.slot == 1) {
        next.exchange_index += 1;
        next.slot = 0;
      } else {
        next.slot = 1;
      }
      return next;
    }
    case ServeEvent::kImportAboutToOpen: {
      if (expected[tracker.slot] != ExpectedEventKind::kRead) {
        return refuse(expected[tracker.slot], ExpectedEventKind::kRead);
      }
      return tracker;  // checked, not advanced -- see doc comment above
    }
    case ServeEvent::kImportClosed: {
      if (expected[tracker.slot] != ExpectedEventKind::kRead) {
        return refuse(expected[tracker.slot], ExpectedEventKind::kRead);
      }
      OrderTracker next = tracker;
      if (next.slot == 1) {
        next.exchange_index += 1;
        next.slot = 0;
      } else {
        next.slot = 1;
      }
      return next;
    }
    case ServeEvent::kFinalize:
      return tracker;
  }
  return fuse::Failure{"E-LOWFIVE-ATTACH-ORDER-DEVIATION", "unrecognised ServeEvent"};
}

/// One LowFive-bound channel's own contribution to the static order
/// signature -- writer/reader COMPONENT names (not field
/// names: `ChannelRoles::writer_name`/`reader_name` above, already
/// populated by lowfive_attach.cpp's own graph walk, in `ChannelId`
/// order) plus each side's own rank COUNT (not which ranks -- a
/// multi-rank-per-component layout was never measured, kept
/// defensively, not because it is known to change the ORDER
/// mechanism itself).
struct ChannelOrderSignature {
  std::string writer_name;
  std::string reader_name;
  std::size_t writer_rank_count = 0;
  std::size_t reader_rank_count = 0;
};

[[nodiscard]] inline bool operator==(const ChannelOrderSignature& a,
                                      const ChannelOrderSignature& b) {
  return a.writer_name == b.writer_name && a.reader_name == b.reader_name &&
         a.writer_rank_count == b.writer_rank_count && a.reader_rank_count == b.reader_rank_count;
}

/// PURE: this graph's own order signature, in `ChannelId` order --
/// `channels` must be the SAME, already-filtered vector
/// `decide_attach_mode()` takes (lowfive_attach.cpp's own job to build
/// it in that order).
[[nodiscard]] inline std::vector<ChannelOrderSignature> ComputeGraphOrderSignature(
    const std::vector<ChannelRoles>& channels) {
  std::vector<ChannelOrderSignature> out;
  out.reserve(channels.size());
  for (const ChannelRoles& ch : channels) {
    out.push_back(ChannelOrderSignature{ch.writer_name, ch.reader_name, ch.writer_ranks.size(),
                                        ch.reader_ranks.size()});
  }
  return out;
}

/// The ONE configuration this policy/order-check pair has
/// been validated against: the measured verification.brusselator.
/// participants.lowfive.mpmd.3 row -- channel u2v (U writes, V reads)
/// THEN channel v2u (V writes, U reads), one rank per component. Any
/// OTHER graph -- a third channel, a swapped channel order, a
/// multi-rank component -- is refused by `CheckGraphOrderValidated()`
/// below, by name, rather than silently assumed to behave the same way.
///
/// Does the accept-mode
/// 3-rank layout need its OWN entry here? NO, proven rather than
/// argued -- test/verification/brusselator_lowfive_attach_test.cpp's
/// own ComputeGraphOrderSignature.AcceptLayoutPhysicsRanksMatchTheValid
/// atedMpmdSignature builds the real accept-3 component layout
/// (brusselator_graph_tool.cpp's own accept branch: U's RankSet is its
/// physics ranks alone, V's starts right after U's local
/// world) and computes, from each physics rank's own
/// CollectChannelsForRank() result, EXACTLY this function's own return
/// value -- both launch modes share the SAME add_window_coupling() call
/// that declares u_out/v_out in the SAME order, so the signature is a
/// property of the GRAPH, not of launch.mode. The bundled
/// coupler/placeholder rank's own signature is empty and does NOT
/// match (PlaceholderRankSignatureDoesNotMatchAndWouldBeRefused, same
/// file) -- irrelevant in practice because that rank never calls
/// decide_attach_mode() at all.
[[nodiscard]] inline std::vector<ChannelOrderSignature> ValidatedGraphOrderSignatures() {
  return {ChannelOrderSignature{"U", "V", 1, 1}, ChannelOrderSignature{"V", "U", 1, 1}};
}

/// PURE: refuses (`E-LOWFIVE-ATTACH-ORDER-UNVALIDATED`),
/// naming the first differing property, unless `signature` is EXACTLY
/// `ValidatedGraphOrderSignatures()` -- this makes the LowFive
/// participants usable only for the one validated configuration until a
/// thread route (lowfive_attach.h's own not-done follow-up) or a plan-derived
/// order exists. Collective-safe: every rank of both
/// components computes `signature` from the SAME loaded graph, the
/// identical data `decide_attach_mode()`'s own INCONSISTENT check
/// already relies on being rank-independent.
[[nodiscard]] inline fuse::Result<void> CheckGraphOrderValidated(
    const std::vector<ChannelOrderSignature>& signature) {
  const std::vector<ChannelOrderSignature> validated = ValidatedGraphOrderSignatures();
  if (signature.size() != validated.size()) {
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED",
        "this graph has " + std::to_string(signature.size()) +
            " LowFive-bound channel(s); the serve policy has only been validated against " +
            std::to_string(validated.size())};
  }
  for (std::size_t i = 0; i < signature.size(); ++i) {
    if (signature[i] == validated[i]) continue;
    std::string what;
    if (signature[i].writer_name != validated[i].writer_name) {
      what = "channel " + std::to_string(i) + "'s writer is '" + signature[i].writer_name +
             "', validated only for '" + validated[i].writer_name + "'";
    } else if (signature[i].reader_name != validated[i].reader_name) {
      what = "channel " + std::to_string(i) + "'s reader is '" + signature[i].reader_name +
             "', validated only for '" + validated[i].reader_name + "'";
    } else if (signature[i].writer_rank_count != validated[i].writer_rank_count) {
      what = "channel " + std::to_string(i) + "'s writer has " +
             std::to_string(signature[i].writer_rank_count) + " rank(s), validated only for " +
             std::to_string(validated[i].writer_rank_count);
    } else {
      what = "channel " + std::to_string(i) + "'s reader has " +
             std::to_string(signature[i].reader_rank_count) + " rank(s), validated only for " +
             std::to_string(validated[i].reader_rank_count);
    }
    return fuse::Failure{
        "E-LOWFIVE-ATTACH-ORDER-UNVALIDATED",
        "graph order signature does not match the measured configuration: " + what +
            "; the serve policy has only been validated against the "
            "measured order"};
  }
  return fuse::Result<void>::success();
}

/// The protocol, event by event (see this header's own "RECEIVE AND
/// RELEASE" comment above for the LowFive-source citations it is built
/// from):
///
///   - kExportWrittenAndClosed, first-server: kServeNow (blocks until
///     the peer's own open+close of THIS file releases it -- see the
///     INVARIANT below for why that peer always arrives, and in time).
///   - kExportWrittenAndClosed, first-reader before its first completed
///     read (the cold start; see decide_serve_action()): kDeferServe. Once
///     past it, kServeNow, as for the first-server. Refused
///     (`E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER`) if a PREVIOUS deferred
///     serve was never flushed by an intervening kImportClosed -- the
///     alternation this protocol depends on (see INVARIANT) was
///     violated, and silently replacing the first deferred serve with
///     the second would drop the first file's data with no diagnostic.
///   - kImportAboutToOpen, ANY role or state: kReceiveOnly,
///     unconditionally. Nothing LowFive-level happens at this event at
///     all (this header's own "RECEIVE AND RELEASE" comment has the full
///     account of why not, and of why a flush here specifically would
///     deadlock); the real receive is `H5Fopen` itself, which this
///     event merely precedes and does not need to assist.
///   - kImportClosed, first-server: kNoopAfterImport (the first-server
///     never defers, so there is never anything of its own to flush).
///   - kImportClosed, first-reader, nothing deferred: kNoopAfterImport
///     (an import with no outstanding export of its own -- not an
///     error; e.g. the very first event of a run on a graph laid out so
///     the first-reader's own first delivery stop is an import).
///   - kImportClosed, first-reader, a deferred serve pending:
///     kFlushDeferred -- THIS process has just completed its own
///     open-and-close of the peer's file (the action that, symmetrically,
///     is what releases a peer's own pending serve -- see INVARIANT), so
///     entering this process's own blocking serve now is safe: the peer
///     is guaranteed to reach ITS next import (an ordinary kReceiveOnly
///     is not blocking, so it moves straight on to its own open+close)
///     and release it in turn.
///   - kFinalize: refused (`E-LOWFIVE-ATTACH-SERVE-UNFLUSHED`) if a
///     first-reader still has a deferred serve with no `kImportClosed`
///     left to flush it -- the data was written and closed but never
///     served, and refusing loudly is the only acceptable outcome (never
///     a silent drop, never a hang nothing will ever resolve). Clean
///     (`kNoopFinalizeClean`) otherwise, including always for a
///     first-server (which never defers).
///
/// INVARIANT (deadlock-freedom), stated in terms of what the PEER
/// observes, not in terms of this process's own hook-call order. I1
/// (progress) rests on three pieces of evidence -- the measured stack,
/// u_main.c's/v_main.f90's single-branch dispatch, and `include/fuse/
/// runtime/window_coupler.hpp`'s own documented Jacobi snapshot contract
/// -- establishing that on one process, an export is always eventually
/// followed by an import before that SAME process's own next export.
/// I2 (role exclusivity, this state machine's own contract above): the
/// first-server serves at every export and never defers; the
/// first-reader defers its cold-start export and flushes only at its own
/// next `kImportClosed`. THE RELEASE CONDITION, read from LowFive's own
/// source (this header's own "RECEIVE AND RELEASE" comment): a process
/// blocked inside `serve_all()` for file K is released exactly when the
/// PEER completes ITS OWN open-and-close of file K -- not by the peer
/// merely reaching an import, and not by anything this process itself
/// does. So: whenever the first-server blocks inside its own `kServeNow`
/// for file K (entered at its own export), I1 guarantees the first-
/// reader's own NEXT local event is an import, which performs an
/// ordinary, non-blocking open-and-close of file K (kImportAboutToOpen
/// does nothing to delay it) -- exactly the release condition for the
/// first-server's pending serve. Symmetrically, the first-reader's own
/// `kImportClosed` for that SAME import is what then flushes ITS own
/// deferred serve (entered only now, after its own open-and-close is
/// already done) -- and I1 guarantees the first-server's own NEXT local
/// event is ITS import, whose open-and-close is exactly the release
/// condition for the first-reader's newly-entered serve. Each side's own
/// single blocking serve call is therefore always matched by the OTHER
/// side's very next open-and-close, in that order -- never by the hook
/// that merely PRECEDES it -- and never left unconsumed except at
/// `kFinalize`, which this function refuses loudly rather than hanging
/// or dropping. This DEPENDS on I1 holding for BOTH components every
/// iteration; the double-defer refusal remains the
/// check that catches a violation instead of hanging silently. See
/// lowfive_attach_model_check.hpp for an explicit-state simulation that
/// checks this argument directly against the blocking semantics above,
/// rather than taking the argument's own word for it.
[[nodiscard]] inline fuse::Result<ServeAction> serve_state_step(ServeState* state,
                                                                 ServeEvent event) {
  switch (event) {
    case ServeEvent::kExportWrittenAndClosed: {
      // The policy decision is decide_serve_action()'s own job, not a
      // hardcoded "first-reader always defers" -- see this header's own
      // "THE PER-EXCHANGE ORDER DIFFERS" comment above for the citations
      // behind the rule.
      if (decide_serve_action(state->role, state->completed_any_read) ==
          ServeNowOrDefer::kServeNow) {
        return ServeAction::kServeNow;
      }
      if (state->deferred_pending) {
        return fuse::Failure{
            "E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER",
            "a second export was written and closed while a FIRST deferred serve was "
            "still pending (no kImportClosed arrived between them) -- the write/read "
            "alternation this protocol depends on (I1) was violated"};
      }
      state->deferred_pending = true;
      return ServeAction::kDeferServe;
    }
    case ServeEvent::kImportAboutToOpen:
      // Unconditional, any role or state: this header's own "RECEIVE AND
      // RELEASE" comment has the full account of why NOTHING is flushed
      // here, no matter what.
      return ServeAction::kReceiveOnly;
    case ServeEvent::kImportClosed:
      // Every completed read counts, on EITHER role -- this is
      // decide_serve_action()'s own "history" input, and it must be set
      // unconditionally here (before the role branch below), not only
      // for a first-reader, so a role that later changes behaviour
      // based on it is never fed a stale value.
      state->completed_any_read = true;
      if (state->role == ServeRole::kFirstServer) return ServeAction::kNoopAfterImport;
      if (state->deferred_pending) {
        state->deferred_pending = false;
        return ServeAction::kFlushDeferred;
      }
      return ServeAction::kNoopAfterImport;
    case ServeEvent::kFinalize:
      if (state->role == ServeRole::kFirstReader && state->deferred_pending) {
        return fuse::Failure{
            "E-LOWFIVE-ATTACH-SERVE-UNFLUSHED",
            "finalize reached with a deferred serve still pending -- the last export "
            "was written and closed but no matching kImportClosed ever arrived to "
            "flush it; refusing rather than dropping the data silently or leaving the "
            "peer to hang waiting for it"};
      }
      return ServeAction::kNoopFinalizeClean;
  }
  return fuse::Failure{"E-LOWFIVE-ATTACH-SERVE-BADEVENT", "unrecognised ServeEvent"};
}

// ---------------------------------------------------------------------
// WHY THIS ATTACH CALLS set_intercomm() AND SLASH-NORMALISES DATASET
// PATTERNS. Both facts were confirmed by reading LowFive's own source
// (/Users/mahadevan/Code/hdf5/LowFive, rev 7f3a58b), not inferred from
// an error message alone.
//
// set_intercomm(). Without it, both brusselator_u_main and
// brusselator_v_main die at their own FIRST remote H5Fopen,
// `LowFive::MetadataError: file_open(): no intercomm found for
// brusselator_{v,u}.h5`. `DistMetadataVOL::file_open()`
// (src/vol-dist-metadata.cpp:263) calls `find_matches(name, "",
// intercomm_locations, true)`, which is empty unless something calls
// `DistMetadataVOL::set_intercomm()` (include/lowfive/vol-dist-metadata.
// hpp:59-63, which pushes a `LocationPattern{filename, full_path}` onto
// `intercomm_locations` and the given index onto `intercomm_indices` --
// the ONLY two places either vector is ever touched). So this attach
// registers every binding itself (BuildIntercommRegistrations() below).
// (`fuse::transport::LowFiveTransport::set_intercomm()` is a same-named
// but unrelated wrapper around forming the MPI intercommunicator itself,
// not registering it with the VOL.)
//
// Slash normalisation. `fuse::split_hdf5_path()` (include/fuse/transport/
// lowfive.hpp:61) returns the dataset component WITHOUT a leading slash
// (`hdf5_path.substr(first + sep.size())`, where `sep = ".h5/"` -- the
// slash is consumed as part of the separator, never re-added).
// `LowFive::Object::fullname()` (src/metadata/object.hpp:169-184), by
// contrast, ALWAYS produces a full dataset path WITH a leading slash: it
// prefixes every path segment with `/` unless already present
// (`(!o->name.empty() && o->name[0] != '/') ? "/" : ""`, applied walking
// up to the File root AND to the child path argument).
// `MetadataVOL::dataset_create()` (src/vol-metadata/dataset.cpp:30-33)
// computes `is_memory = match_any(filepath, memory)` -- the DEFAULT
// 2-argument overload, `partial = false` -- where
// `filepath = obj->fullname(name_str)`, i.e. the slashed form. `match()`
// (include/lowfive/vol-metadata.hpp:135-150) is a wildcard matcher
// walking both strings position by position; at the very first
// character, an unslashed pattern ('g' at position 0 of "grid/u_out")
// against a slashed candidate ('/' at position 0 of "/grid/u_out") is
// neither an exact character match nor a `*`/`?` wildcard, so the match
// falls through to `return partial;` = `false`. An unslashed pattern
// passed to `vol.set_memory()` (as `split_hdf5_path()` returns it) would
// therefore make `is_memory` `false` for every Brusselator dataset,
// meaning `dataset_create()` never attaches an in-memory `Dataset`
// object to the file's own tree, `Index`'s own constructor's
// `find_datasets()` traversal (src/vol-dist/index.cpp:17-28) never finds
// it, `indexed_datasets` stays 0, and `serve_all()`
// (src/vol-dist-metadata.cpp:61, `if (index.indexed_datasets ||
// !perform_indexing) index.serve();`) returns WITHOUT EVER CALLING
// `index.serve()` -- a first-server's own "blocking" `serve_all()` call
// would not block at all, and would not actually serve anything either.
// So `ApplyBindings()` passes the normalised form
// (NormalizeDatasetPattern() below) to `vol.set_memory()`.
//
// This attach assumes exactly one intercomm, matching the
// single-intercomm `create_DistMetadataVOL(local, intercomm)` overload
// used here.

/// PURE: normalizes a dataset pattern to the leading-slash form `Object::
/// fullname()` always produces (see the "Slash normalisation" comment
/// above). Already-slashed
/// input is returned UNCHANGED (idempotent); an empty pattern is refused
/// (`E-LOWFIVE-ATTACH-DATASET-EMPTY-PATTERN`) -- nothing to register.
/// Deliberately NOT a change to `fuse::split_hdf5_path()` itself: that
/// function has its own tests and the single-process path reaches
/// LowFive through a `"*"` wildcard, never through this specific
/// unslashed string, so changing its own contract would be an unrelated
/// behaviour change for a caller this does not concern.
[[nodiscard]] inline fuse::Result<std::string> NormalizeDatasetPattern(const std::string& pattern) {
  if (pattern.empty()) {
    return fuse::Failure{"E-LOWFIVE-ATTACH-DATASET-EMPTY-PATTERN",
                         "a dataset pattern is empty; nothing to register with LowFive's own "
                         "set_memory()/set_intercomm()"};
  }
  if (pattern.front() == '/') return pattern;
  return "/" + pattern;
}

/// One (file, normalized dataset pattern, intercomm index) entry this
/// rank's own attach must register via `LowFive::DistMetadataVOL::
/// set_intercomm()` (see the "set_intercomm()" comment above) so a CONSUMER's own remote
/// `file_open()` can find which intercomm serves a given filename.
struct IntercommRegistration {
  std::string file;
  std::string dataset_pattern;
  int         intercomm_index;
};

/// PURE, MPI/HDF5-free: builds the full registration list for one
/// binding table against a SINGLE, EXPLICIT intercomm index -- never
/// hardcoded inside this function, so a future multi-intercomm design
/// needs a different SIGNATURE here (e.g. a per-binding-group index),
/// not a different value silently threaded through this one. See
/// lowfive_attach.cpp's own call site for why it always passes `0`
/// under THIS attach's own architecture (exactly one intercomm is ever
/// formed: `decide_attach_mode()` already refuses, by name, any graph
/// where this rank's own qualifying channels do not all partition the
/// world into the SAME two groups -- `E-LOWFIVE-ATTACH-INCONSISTENT` --
/// so there is structurally no way for this attach to need a second
/// intercomm today; a design that ever did would need to extend THIS
/// function, not silently bump the value its caller passes).
///
/// Mirrors `ApplyBindings()`'s own loop and its own defensive split_
/// hdf5_path() re-check (`build_lowfive_bindings()` already validated
/// every binding; a failure here would mean that contract broke, not a
/// new input) -- refuses with the propagated `E-LOWFIVE-PORT` in that
/// case, or `NormalizeDatasetPattern()`'s own refusal if some future
/// binding's dataset component is ever empty.
///
/// Builds one entry per BINDING in the whole table, not filtered by
/// whether THIS rank writes or reads that binding's own file -- grep
/// confirms `intercomm_locations`/`intercomm_indices` (the two vectors
/// `set_intercomm()` populates) are consulted ONLY inside `DistMetadataVOL
/// ::file_open()` (src/vol-dist-metadata.cpp:263,269), the REMOTE-OPEN
/// path; a rank's own `H5Fcreate` of a file it writes reaches `file_
/// create()`, an entirely different VOL callback that never touches
/// either vector. So registering a file's own intercomm location on the
/// rank that only ever CREATES it (never opens it as a consumer) is
/// harmless by construction: `file_open()`'s own `find_matches()` first
/// checks `match(x.filename, filename)` per entry (src/vol-dist-
/// metadata.cpp's own helper in vol-metadata.hpp:177-187), so an entry
/// for a filename THIS rank's own `file_open()` call is never made
/// against simply never matches, costing one extra linear-scan
/// comparison and nothing else.
[[nodiscard]] inline fuse::Result<std::vector<IntercommRegistration>> BuildIntercommRegistrations(
    const fuse::LowFiveFieldBindingTable& table, int intercomm_index) {
  std::vector<IntercommRegistration> out;
  for (const fuse::LowFiveFieldBinding& b : table.bindings()) {
    auto split = fuse::split_hdf5_path(b.hdf5_path);
    if (!split.ok()) {
      return fuse::Failure{split.error().code, split.error().message};
    }
    auto normalized = NormalizeDatasetPattern(split.value().second);
    if (!normalized.ok()) {
      return fuse::Failure{normalized.error().code, normalized.error().message};
    }
    out.push_back(IntercommRegistration{split.value().first, normalized.value(), intercomm_index});
  }
  return out;
}


// DIAGNOSTIC TRACE SEAM. When a host run hangs past its first remote
// H5Fopen with BOTH U and V inside file_open()'s own
// "get_file_hierarchy" RPC and neither in serve_all(), the first-server's
// own serve_after_write() -> serve_all() call either returned WITHOUT
// BLOCKING or never ran. The trace distinguishes: whether the cold-start
// write's own view guard was false (so no H5Fcreate ever happened);
// whether serve_all() returned at once because indexed_datasets==0;
// whether the hook returned an action other than kServeNow, or never ran;
// and whether serve_all() served the wrong intercomm index (or
// serve_indices/intercomms ended up empty). The functions below are the
// PURE, testable seam that the trace-line formatting and env-parsing go
// through; the actual getenv()/fprintf() calls (not pure, not
// unit-testable) live in lowfive_attach.cpp/shim.cpp and call these.

/// Pure env-value parser for `FUSE_LOWFIVE_ATTACH_TRACE`: enabled iff
/// the raw value is the single character `"1"` -- unset (`raw ==
/// nullptr`), empty, `"0"`, or anything else (`"true"`, `"TRUE"`,
/// `"yes"`) is disabled ("enabled by env FUSE_LOWFIVE_ATTACH_TRACE=1"),
/// not a more permissive grammar.
/// The CALLER (lowfive_attach.cpp's own `bruss_lowfive_trace_enabled()`)
/// reads `getenv()` exactly once per process and caches the result in
/// a function-local static -- that caching is not reachable from a
/// pure function and is not tested here; this function is the pure
/// half, tested directly against literal strings.
[[nodiscard]] inline bool ParseTraceEnabledEnv(const char* raw) {
  return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

/// Pure formatter for the attach-time trace line's own registration
/// list (one line summarizing every `(file, dataset_pattern,
/// intercomm_index)` this process registered via `LowFive::
/// DistMetadataVOL::set_intercomm()` -- settles whether a "wrong
/// intercomm index" is even possible for THIS row's own single-
/// intercomm architecture, and whether the registration list is empty
/// (which would mean `BuildIntercommRegistrations()`'s own table
/// argument was empty, a DIFFERENT failure). Deterministic,
/// pure string building -- no I/O, no env access -- so a unit test can
/// assert the exact output for a known input vector.
[[nodiscard]] inline std::string FormatRegistrationsForTrace(
    const std::vector<IntercommRegistration>& regs) {
  std::string out = "[";
  for (std::size_t i = 0; i < regs.size(); ++i) {
    if (i > 0) out += ", ";
    out += "(" + regs[i].file + ", " + regs[i].dataset_pattern + ", " +
           std::to_string(regs[i].intercomm_index) + ")";
  }
  out += "]";
  return out;
}

/// Pure formatter for the attach-time trace line's own memory-pattern
/// list (one line summarizing every `(file, dataset)` pattern this
/// process registered via `LowFive::MetadataVOL::set_memory()`, i.e.
/// `ApplyBindings()`'s own loop AFTER `NormalizeDatasetPattern()` --
/// the EXACT pair `MetadataVOL::dataset_create()`'s own `is_memory =
/// match_any(filepath, memory)` check compares a
/// real dataset's `fullname()` against. Same determinism/testability
/// rationale as `FormatRegistrationsForTrace()` above.
[[nodiscard]] inline std::string FormatMemoryPatternsForTrace(
    const std::vector<std::pair<std::string, std::string>>& patterns) {
  std::string out = "[";
  for (std::size_t i = 0; i < patterns.size(); ++i) {
    if (i > 0) out += ", ";
    out += "(" + patterns[i].first + ", " + patterns[i].second + ")";
  }
  out += "]";
  return out;
}


// WHY `keep=true` IS REQUIRED. Symptom without it: `serve_all()` enters
// and exits in tens of microseconds, LowFive's own log printing
// "enter serve_all, files.size = 0", while the writer's H5Fcreate
// succeeded, the first-server took kServeNow, and exactly one intercomm
// is registered. (`is_memory` does not gate whether a dataset is
// COUNTED, so the empty file table is not a pattern-matching problem.)
//
// ROOT CAUSE, read directly from LowFive's source:
// - `include/lowfive/vol-metadata.hpp:49`: `bool keep = false; // whether
//   to keep files in the metadata after they are closed`; `set_keep
//   (bool)` at `:106`. PUBLIC, default false.
// - `src/vol-metadata/file.cpp` (`MetadataVOL::file_close()`): `if
//   (File* f = dynamic_cast<File*>(...)) { ...; if (!keep) { files.erase
//   (f->name); delete f; /* erases all the children too */ } }` -- runs
//   regardless of whether this VOL is a plain `MetadataVOL` or a
//   `DistMetadataVOL` (the distributed subclass's own `file_close()`
//   falls through to this EXACT function at its own tail,
//   `src/vol-dist-metadata.cpp`: `return MetadataVOL::file_close(file,
//   dxpl_id, req);`).
// - `src/vol-dist-metadata.cpp` (`DistMetadataVOL::file_close()`, the
//   LOCAL-file branch): the ONLY place this function ever calls
//   `serve_all()` on its own is `else if (serve_on_close && match_any
//   (f->name, "", memory, true)) serve_all();` -- gated on `serve_on_
//   close`, which lowfive_attach.cpp sets `false`
//   specifically so NOTHING auto-serves inside `H5Fclose()`. That
//   `if`/`else if` chain runs BEFORE the tail call into `MetadataVOL::
//   file_close()` above.
// - Putting these two together: with `serve_on_close == false` and
//   `keep` left at its own default `false`, `H5Fclose()` neither serves
//   internally NOR keeps the file -- it is ERASED AND DELETED,
//   unconditionally, the moment `H5Fclose()` returns, before this
//   process's own explicit `bruss_lowfive_serve_after_write()` hook
//   (called by `shim.cpp` AFTER `H5Fclose()` returns) ever runs a
//   line. By the time that hook's own `serve_all()` call happens,
//   `files` is ALREADY EMPTY -- exactly the measured `files.size = 0`,
//   `enter serve_all, files.size = 0`, and the 52-microsecond return
//   (LowFive own confirms, from its own source, why: `serve_all()`'s
//   own `Index` constructor counts `indexed_datasets` by walking
//   `files`, which has nothing left to walk; `index.indexed_datasets ||
//   !perform_indexing` is then false at `serve_all()`'s own default
//   `perform_indexing=true`, so `index.serve()` -- the part that would
//   actually block polling for the peer -- is never called at all).
//   LowFive therefore only works with `serve_on_close=true`, OR with
//   `serve_on_close=false` AND `keep=true`; this attach uses the second.
// - `MetadataVOL::drop(std::string filename)` (`src/vol-metadata.cpp`):
//   `auto it = files.find(filename); delete it->second; files.erase
//   (it);` -- NO check against `files.end()`; calling this for an
//   ABSENT name dereferences `files.end()`, undefined behaviour. Guarded
//   by `GuardedDropOwnExportedFile()` (`lowfive_attach.cpp`) before ever
//   calling this.
// - `MetadataVOL::clear_files()` (`src/vol-metadata.cpp`): deletes EVERY
//   entry in `files`, not just one -- never call this to release a
//   single file; it would also delete the PEER's own file this rank
//   may have just received.
// - `MetadataVOL::file_create()` (`src/vol-metadata/file.cpp`): `File*
//   f = new File(name_, ...); files.emplace(name_, f);` -- `std::map::
//   emplace()` is a NO-OP on a DUPLICATE key: if `name_` is still in
//   `files` from a PREVIOUS create that was never dropped, the OLD
//   `File*` stays exactly where it is, and the FRESH `f` this call just
//   allocated is never stored in `files` at all (only on this NEW open
//   handle's own `mdata_obj`, which nothing else ever looks up by
//   filename). The practical effect, read from this call site directly
//   (not assumed): SILENT REUSE OF STALE DATA, not an error and not a
//   crash -- a LATER `serve_all()` keyed off this filename walks
//   `files`, finds the OLD `File`'s own (now stale) children, and
//   serves THOSE, never the fresh write this handle just performed
//   through its own orphaned object. This is exactly why the file must
//   be dropped after it is served, before this component's own NEXT
//   `H5Fcreate` of the same name.

/// The only two LowFive-level settings `bruss_lowfive_attach()` applies
/// to a constructed VOL -- `serve_on_close` (meaningful for `LowFive::
/// DistMetadataVOL` only; `LowFive::MetadataVOL` has no such member at
/// all) and `keep` (meaningful for either). Decided together, by ONE
/// function, rather than set separately at each of `bruss_lowfive_
/// attach()`'s own two branches, so the invariant below is checked
/// once, not re-derived at each call site.
struct VolSettings {
  bool serve_on_close;
  bool keep;
};

/// PURE: the settings this attach applies, for either branch
/// (`distributed` tells the CALLER which fields are actually consumed --
/// the single-process branch's own `LowFive::MetadataVOL` has no `serve_
/// on_close` member, so it simply never reads that field of the
/// returned struct, exactly as it already never changed its own
/// existing `vol.set_keep(true)` call). BOTH branches need
/// `keep=true`: on the single-process branch, without it the read leg's
/// own `H5Fopen` finds nothing (the SAME `MetadataVOL::file_close()` this
/// header's own "ROOT CAUSE" comment above cites for the distributed
/// case); the distributed branch needs it for exactly the same
/// underlying reason. The distributed branch additionally needs
/// `serve_on_close=false` (`true` serves ANY two writes racing into
/// `serve_all()` together, inside `H5Fclose()`, before either side has
/// read the other).
[[nodiscard]] inline VolSettings decide_vol_settings(bool distributed) {
  (void)distributed;  // both branches currently want the identical pair
  return VolSettings{/*serve_on_close=*/false, /*keep=*/true};
}

/// PURE: the invariant `decide_vol_settings()` must never violate --
/// "`serve_on_close == false` implies `keep == true`" (this header's own
/// "ROOT CAUSE" comment above has the full account of why: with `serve_
/// on_close` false, NOTHING inside `DistMetadataVOL::file_close()` ever
/// serves a local file on its own, so the application's own explicit
/// serve call is the ONLY server that file will ever get -- if `keep` is
/// also false, the file is erased and deleted before that explicit
/// serve call ever runs). Takes an ARBITRARY pair, not only `decide_vol_
/// settings()`'s own return value, so a future mutation of either field
/// is caught directly by a test feeding it the bad pair, not only
/// indirectly through whatever `decide_vol_settings()` happens to
/// return.
[[nodiscard]] inline bool VolSettingsSatisfyInvariant(VolSettings s) {
  return s.serve_on_close || s.keep;
}

/// PURE: which file(s) `this_name` (the SAME canonical component name
/// `bruss_lowfive_attach()` already derives from `plan.is_writer`) is
/// the WRITER of, among `table`'s own bindings -- derived from `fuse::
/// build_lowfive_bindings()`'s own `"<component>.<field>"` qualification
/// scheme (`include/fuse/transport/lowfive_graph.hpp`), never a
/// hardcoded literal. Returns one file per matching binding, in table
/// order (the Brusselator graph has exactly one per
/// component, but a future graph where one component writes more than
/// one HDF5-bound field is not refused by `decide_attach_mode()` by
/// name, so this function does not assume it away either); empty if
/// `this_name` writes nothing qualifying (a read-only component, same
/// reasoning). These are the files this rank's own attach must drop,
/// one at a time, after each is served -- never the PEER's own file,
/// which this rank only ever reads and never owns the lifetime of.
[[nodiscard]] inline fuse::Result<std::vector<std::string>> FindOwnExportedFiles(
    const fuse::LowFiveFieldBindingTable& table, const std::string& this_name) {
  std::vector<std::string> out;
  const std::string prefix = this_name + ".";
  for (const fuse::LowFiveFieldBinding& b : table.bindings()) {
    if (b.field_name.size() <= prefix.size() ||
        b.field_name.compare(0, prefix.size(), prefix) != 0) {
      continue;
    }
    auto split = fuse::split_hdf5_path(b.hdf5_path);
    if (!split.ok()) {
      return fuse::Failure{split.error().code, split.error().message};
    }
    out.push_back(split.value().first);
  }
  return out;
}

/// WHY THE DATASET'S PARENT GROUP IS CREATED. Without it, the reader
/// fails in under a second with an abort, inside V's own `H5Dget_space()`
/// on a REMOTE dataset: `H5S.c ... H5Scopy(): not a dataspace` ->
/// `LowFive::MetadataError: Unexpected no class in dataspace = -1`, thrown
/// from `Dataspace(space_id)`'s own ctor (LowFive/src/metadata/
/// dataspace.hpp) evaluated as a logger argument inside
/// `MetadataVOL::dataset_get()`'s GET_SPACE branch (LowFive/src/
/// vol-metadata/dataset.cpp): `auto& dataspace = static_cast<Dataset*>
/// (dset_->mdata_obj)->space; hid_t space_id = dataspace.copy();` --
/// `dataspace.copy()` (`H5Scopy(id)`) fails because `id == 0`.
///
/// Root cause, read from source: `bruss_lowfive_write()` calls
/// `H5Dcreate2(fid, "/grid/u_out", ...)`. With NO prior
/// `H5Gcreate2(fid, "/grid", ...)` this flattens the dataset into ONE
/// child of the FILE/root literally named `"grid/u_out"`:
/// `LowFive::Object::search()`, src/metadata/object.hpp, ALWAYS splits
/// on the first `/` before attempting any match -- `auto pos =
/// path.find("/"); ...; for (child : children) if (child->name ==
/// first_name) ...; return ObjectPath{this, path};` -- it never tries a
/// whole-string match first. The flat name breaks `H5Lexists()` (a
/// COSMETIC trace artifact, since the write-side `is_memory` check
/// compares the whole string via `fullname()` and is unaffected). It
/// ALSO breaks `MetadataVOL::dataset_open()` (src/vol-metadata/
/// dataset.cpp) -- NOT cosmetic there: `auto obj_path =
/// parent->search(name); if (obj_path.path.empty()) result->mdata_obj =
/// obj_path.obj; if (!result->mdata_obj) { result->mdata_obj =
/// parent->add_child(new DummyDataset(name)); }` -- since
/// `search("grid/u_out")` never finds a child named exactly `"grid"`,
/// `obj_path.path` comes back as the WHOLE unconsumed string
/// (non-empty), so `result->mdata_obj` is left null and a `DummyDataset`
/// placeholder (src/metadata/dummy.hpp: `struct DummyDataset : public
/// DummyObject { Datatype type; Dataspace space; ... }` -- NOT a
/// subclass of `Dataset`, default-constructed `space` with `id == 0`)
/// is substituted. `DistMetadataVOL::dataset_open()`'s own promotion to
/// a RPC-backed `RemoteDataset` (`dynamic_cast<Dataset*>(mdata_obj) &&
/// RemoteObject::query(mdata_obj)`) never fires, because `dynamic_cast
/// <Dataset*>` on a `DummyDataset*` is correctly null (unrelated
/// classes, both children of `Object` but not of each other) -- so the
/// placeholder's own, never-populated `space` (id == 0) is what later
/// `H5Dget_space()` reads. `H5Scopy(0)` fails ("not a dataspace"),
/// returning LowFive's own documented failure sentinel `H5I_INVALID_HID
/// == -1`; the immediately following `log->trace("copied space id =
/// {}, space = {}", space_id, Dataspace(space_id))` constructs
/// `Dataspace(-1)` as a LOGGER ARGUMENT (evaluated regardless of the
/// active log level, C++'s own eager-argument-evaluation rule, not a
/// LowFive choice) -- its ctor's `if (cls_ == H5S_NO_CLASS) throw
/// MetadataError(fmt::format("Unexpected no class in dataspace = {}",
/// id));` fires with `id == -1`, the EXACT message above.
///
/// Not the cause: `RemoteDataset`'s own ctor (LowFive/src/metadata/
/// remote.hpp) DOES fetch `space` via a real RPC (`std::tie(dim, type,
/// space, decomposer) = obj.call<...>("metadata")`) -- this machinery is
/// real and, per LowFive's own upstream `examples/prod-con-multidata/`
/// demo (producer: `H5Gcreate(file, "/group1", ...)` then
/// `H5Dcreate2(group, "grid", ...)` with a BARE name relative to the
/// GROUP handle; consumer: calls `H5Dget_space()` on the resulting
/// remote dataset successfully), is a SUPPORTED, WORKING LowFive feature
/// -- it is simply never reached when the flat name defeats
/// `make_remote_dataset()`.
///
/// The fix belongs in this shim, not in LowFive: create the dataset's
/// own parent group(s) before `H5Dcreate2()`, so `search()`'s segment
/// walk succeeds at every level on both the writer's own in-process
/// lookup and -- via the identical, separately-deserialized tree built
/// from the SAME `serialize()`d byte stream -- the reader's `dataset_
/// open()` lookup too. `LowFive::MetadataVOL::group_create()` (src/
/// vol-metadata/group.cpp) already walks its OWN unconsumed `search()`
/// remainder segment by segment, creating one `Group` per segment --
/// unlike `dataset_create()`, which never does -- so ONE `H5Gcreate2()`
/// call for the dataset's full parent-group path is sufficient even if
/// that path has more than one segment. `Object::fullname()` (used by
/// the write-side `is_memory`/pattern-matching check) walks `parent`
/// pointers and re-concatenates with `/`, so nesting the dataset under a
/// real `Group` named "grid" reproduces the IDENTICAL `"/grid/u_out"`
/// string that check already matches against.
struct DatasetGroupSplit {
  std::string group_path;  // "" if no H5Gcreate2() call is needed (the
                            // dataset belongs directly under whatever
                            // location H5Dcreate2/H5Dopen2 was given)
  std::string leaf_name;   // the dataset's own bare name once
                            // group_path's own Group chain exists; not
                            // used by the shim's own H5Dcreate2 call
                            // (which still passes the FULL path -- see
                            // above for why that is sufficient once the
                            // group exists), returned so a test can
                            // assert on both halves independently
};

/// PURE: splits a dataset path (as passed to `H5Dcreate2`/`H5Dopen2`)
/// into the parent group path to `H5Gcreate2()` before creating the
/// dataset, and the dataset's own leaf name. A path with no `/` at all,
/// or whose only `/` is a single leading one (`"/u_out"`), needs no
/// group (`group_path == ""`) -- the dataset belongs directly under
/// whatever location was passed to `H5Dcreate2`/`H5Dopen2`.
[[nodiscard]] inline DatasetGroupSplit SplitDatasetGroupPath(
    const std::string& dataset_path) {
  const auto last_slash = dataset_path.find_last_of('/');
  if (last_slash == std::string::npos) {
    return DatasetGroupSplit{"", dataset_path};
  }
  const std::string before = dataset_path.substr(0, last_slash);
  const std::string leaf = dataset_path.substr(last_slash + 1);
  if (before.empty()) {
    // "/u_out": a single leading slash, nothing between it and the
    // leaf -- root-level dataset, no intermediate group.
    return DatasetGroupSplit{"", leaf};
  }
  return DatasetGroupSplit{before, leaf};
}

// ---------------------------------------------------------------------
// EXIT HYGIENE: THE COUPLER EXIT CRASH. Symptom: brusselator_coupler_main
// aborts at process exit, AFTER MPI_Finalize, with `libc++abi: terminating
// due to uncaught exception of type std::__1::system_error: mutex lock
// failed: Invalid argument`. Symbolised trace (innermost first):
// __pthread_kill <- abort <- __abort_message <- std::terminate <-
// __cxa_throw <- std::__throw_system_error <- std::mutex::lock <-
// std::lock_guard<std::mutex> (in liblowfive.dylib) <-
// spdlog::details::registry::get <- spdlog::get <- LowFive::get_logger()
// (log-private.hpp:53) <- LowFive::VOLBase::_term() (base.cpp:210) <-
// H5VL__free_cls <- ... <- H5_term_library <- __cxa_finalize_ranges <-
// exit <- dyld start. That is: a registered "lowfive" VOL connector class
// is torn down by H5_term_library during exit(), after spdlog's registry
// mutex has already been destroyed by static destruction. With this
// abort the row still completes its data path: FUSE_COLLECTIVE_TRACE
// agrees exactly across all three ranks, U's and V's completed-row counts
// and `u.log`/`v.log` are byte-identical to the transport-only twin's,
// and `brusselator_check` run on them reports `--check: OK`; the coupler
// exit abort is the only remaining failure (mpiexec reports BAD
// TERMINATION and kills U/V before their own exit codes are observed).
//
// (a) Why the coupler has a LowFive VOL class registered at all is NOT
// ESTABLISHED. `examples/CMakeLists.txt`'s own `verification.
// brusselator.participants.lowfive.{mpmd,accept}.3` row sets
// `ENVIRONMENT "HDF5_VOL_CONNECTOR=lowfive;HDF5_PLUGIN_PATH=...;
// FUSE_LOWFIVE_BINDINGS=..."` on the ONE ctest COMMAND
// (`run_participants.sh`), which launches ALL THREE MPMD app segments
// (U, V, coupler) from a SINGLE `mpiexec` invocation sharing ONE
// environment (`test/parallel/brusselator_participants/run_
// participants.sh`'s own ':'-separated segments). U's and V's own
// `H5Fcreate`/`H5Fopen` calls bypass the env-route resolution (they pass
// an explicit `fapl_id`); the coupler never calls `bruss_lowfive_attach()`
// (`lowfive_attach.cpp` is not one of `brusselator_coupler_main`'s own
// sources). Removing the three env-route variables from the coupler's
// own environment before MPI_Init (the `unsetenv()` calls under (b)) does
// NOT stop the abort: the coupler's own trace line ("unregistered
// env-route VOL vars... before MPI_Init (role=kNonAttachingCoordinator)")
// prints, the unsetenv() calls run before MPI_Init exactly as designed,
// and the SAME abort still happens, byte-identical message, on BOTH MOAB
// installs measured. So environment inheritance from the ctest row is
// NOT (or not solely) the mechanism that registers the connector. The
// two-HDF5-image hazard (`tools/probe-hdf5-images.{c,sh}`) is a
// DIFFERENT, already-measured-PASS condition on this host (exactly one
// `libhdf5` image for this MOAB+LowFive pairing) and is NOT this crash's
// cause: its own symptom is a SILENT `H5Pset_vol(): "not a file VOL ID"`
// failure from a split id table, not a correctly-registered connector's
// own exit-time teardown -- the crash trace shows a SINGLE,
// correctly-resolving `H5VL__free_cls` call chain, consistent with one
// image, not two.
//
// (b) Mitigations, each with the reason it is or is not adopted.
// - A non-attaching process (the coupler) does not register the
//   connector: `unsetenv()` on the three env-route variables, before
//   MPI_Init, before any native HDF5 call. Kept as sound hygiene (the
//   coupler never calls `bruss_lowfive_attach()` and has no legitimate
//   use for the connector regardless of what else registers it), but it
//   is NOT the fix for this crash, per (a).
// - A deterministic teardown in `bruss_lowfive_finalize()`, ADOPTED for U
//   and V, which DO deliberately attach: an explicit, SYNCHRONOUS
//   `H5close()` call, after `serve_state_step(kFinalize)`'s own
//   unflushed-serve check succeeds and BEFORE the caller's own
//   `MPI_Finalize()` (both u_main.c and v_main.f90 call
//   `bruss_lowfive_finalize()` strictly before `MPI_Finalize()`). It is
//   sound because it runs the SAME `VOLBase::_term()` -> `get_logger()`
//   call chain while the process is still in ORDINARY execution -- no
//   exit()-time static-destruction unwind has begun at all, so spdlog's
//   own registry cannot have been torn down yet, regardless of atexit()
//   registration order. This is STRICTER than relying solely on the
//   `std::atexit(+[]{ H5close(); });` registered in
//   `bruss_lowfive_attach()` (the SAME mechanism `VolSingleton`'s own
//   constructor uses, per docs/user/lowfive.md's "HDF5_VOL_CONNECTOR
//   abort" section); that registration is KEPT as a backstop for an
//   abrupt-exit path that skips `bruss_lowfive_finalize()` (e.g. a later
//   `MPI_Abort()`), since `H5close()` is documented idempotent and a
//   second call after the first already tore everything down is a safe
//   no-op.
// - An explicit `spdlog::shutdown()` ordering is NOT adopted: it would
//   tear down the registry itself, which does not fix the hazard (it
//   only moves WHICH line crashes if `_term()` is still reachable
//   afterward) and orders a DIFFERENT library's own static object from
//   FUSE's side, no more reliable than the atexit race it replaces.
// - Making LowFive's logger a `shared_ptr` the VOL object itself holds
//   would be a change to LowFive's OWN source (`log-private.hpp`/
//   `vol-base.hpp`) and is not made here.
//
// (c) Enumeration of every FUSE-side path that could reach a LowFive
// connector registration or construct a VOLBase-derived object, and
// whether the coupler (role kNonAttachingCoordinator) can reach it.
// EXCLUDED, with citations (the coupler CANNOT reach these):
// (1) `src/transport/lowfive_transport.cpp`'s own `VolSingleton`/
// `DistVolSingleton` -- the ONLY callers of `vol_singleton()`/
// `dist_vol_singleton()` anywhere in the tree are `LowFiveTransport`'s
// OWN member functions (`set_location`, `serve_as_producer`,
// `receive_as_consumer`, `build_plan`, the `Impl` destructor). A
// `LowFiveTransport` object is constructed ONLY via
// `TransportRegistry::instance().create(config.transport)`
// (`src/runtime/runner.cpp`) when `config.transport == "lowfive"`.
// `RunnerConfig::transport` is assigned in exactly ONE place in the
// tree, `src/abi/participant.cpp`: `config.transport = "mpi";`, inside
// the `if (c.communicator_set)` branch -- the Brusselator's own MPMD
// `fuse_setup()` call ALWAYS sets `setup.communicator`
// (`coupler_main.c`'s own `setup.communicator = (int64_t)MPI_Comm_c2f
// (MPI_COMM_WORLD);`, identically in u_main.c/v_main.f90), so
// `config.transport` is `"mpi"` for EVERY rank of THIS graph, on EVERY
// role -- `TransportRegistry::instance().create("mpi")` returns an
// `MpiTransport`, never a `LowFiveTransport`.
// (2) `examples/brusselator/participants/lowfive_attach.cpp`'s own
// direct `LowFive::MetadataVOL::create_MetadataVOL()`/
// `LowFive::DistMetadataVOL::create_DistMetadataVOL()` calls --
// `lowfive_attach.cpp` is not one of `brusselator_coupler_main`'s own
// sources (`examples/CMakeLists.txt`: `target_link_libraries
// (brusselator_coupler_main PRIVATE fuse::abi MPI::MPI_C)`, no
// `target_sources` adding `lowfive_attach.cpp` the way U's/V's own
// targets get it) -- this translation unit is not even COMPILED into
// the coupler executable.
// (3) The graph's own fieldless "lowfive" marker component
// (`brusselator_graph_tool.cpp`: `kind = ComponentKind::Coupler`,
// `ranks = RankSet{0, total_ranks}` -- spans every rank, including the
// coupler's own) -- `fuse_initialize()`'s own wiring loop
// (`src/abi/participant.cpp`: `if (decl.kind == fuse::ComponentKind::
// Coupler) continue;`) skips BOTH the real "coupler" component AND this
// fieldless marker identically, since both share `kind ==
// ComponentKind::Coupler` -- this loop constructs nothing for either.
// The marker is declarative only: nothing in the ABI or Runtime consumes
// it to construct a transport or VOL object.
//
// CONFIRMED but NOT sufficient on its own to explain registration
// (linkage is not the same as invocation): `liblowfive`/
// `liblowfive-dist`/MOAB/TempestRemap are linked into EVERY FUSE
// executable, including the coupler, UNCONDITIONALLY, by build-time
// design, independent of role or graph content -- `src/CMakeLists.txt`:
// `fuse_link_lowfive_with_fmt_fix(fuse-coupler-transport)`, called
// whenever `FUSE_ENABLE_LOWFIVE` is ON; `target_link_libraries
// (fuse-coupler-backend PUBLIC fuse::moab)`, called whenever
// `FUSE_ENABLE_MOAB` is ON. Since `brusselator_coupler_main` links
// `fuse::abi`, which pulls in the WHOLE `fuse-coupler-*` static library
// stack, it transitively links `liblowfive`/`liblowfive-dist`/MOAB/
// TempestRemap REGARDLESS of whether its own graph or role ever uses the
// LowFive transport -- this is WHY `otool -L` on the coupler shows these
// libraries; but dyld LINKING a dylib is not the same as that dylib's
// own static initialisers doing anything HDF5-related; no eager/global-
// constructor call into `H5VLregister_connector`/`H5open` was found by
// reading LowFive's own source (below).
//
// READ IN LOWFIVE'S OWN SOURCE (/Users/mahadevan/Code/hdf5/LowFive, rev
// 7f3a58b -- may not be the EXACT revision installed at the host's own
// LowFive prefix): `LowFive::VOLBase::register_plugin()` (`src/vol-base/
// base.cpp:155-163`) is the ONLY call site in the whole LowFive tree for
// the UNVALUED `H5VLregister_connector` (not `_by_name`, not
// `_by_value`) -- it is an ORDINARY MEMBER FUNCTION, called only from
// `LowFive::H5VOLProperty`'s own constructor (`include/lowfive/
// H5VOLProperty.hpp:14`: `vol_id = vol_plugin.register_plugin();`),
// itself only constructed by explicit application code (`VolSingleton`/
// `DistVolSingleton` in FUSE, or `lowfive_attach.cpp` -- both excluded
// above). `LowFive::VOLBase::_str_to_info()` (`src/vol-base/info.cpp`,
// around line 160) calls `H5VLregister_connector_by_value`, NOT the
// unvalued form -- this is HDF5's OWN env-route string parser, invoked by
// HDF5 itself when `HDF5_VOL_CONNECTOR` names a connector not yet
// registered. `LowFive::VOLBase::connector_id` (`base.cpp:5`) and
// `LowFive::VOLBase::connector` (`base.cpp:7-...`) are GLOBAL,
// NAMESPACE-SCOPE static definitions -- but both are plain DATA (a
// `hid_t` initialised to `-1`; an aggregate of FUNCTION POINTERS), not
// calls; neither executes any code at load time. `H5PLget_plugin_type()`/
// `H5PLget_plugin_info()` (`base.cpp:118-`; LowFive itself ships BOTH a
// `lowfive` AND a SEPARATE `lowfive-dist` library target) are ORDINARY
// FUNCTIONS HDF5's OWN plugin loader would call AFTER finding and
// `dlopen`-ing a plugin file via a PATH search -- `liblowfive.dylib` is
// DIRECTLY LINKED into the coupler, not discovered via
// `HDF5_PLUGIN_PATH` (unset before `MPI_Init`), so this specific route is
// unlikely, but not excluded with certainty (whether HDF5's OWN default,
// compiled-in plugin search directories happen to also contain this file
// is outside what source reading alone can settle). No
// `__attribute__((constructor))` function and no non-trivial GLOBAL
// (namespace-scope, not function-local) C++ object was found anywhere in
// LowFive's source tree.
//
// NOT EXCLUDED: MOAB's own and TempestRemap's own source were not read --
// whether EITHER has a static initialiser that calls `H5open()` (directly,
// or indirectly through a netCDF/HDF5 convenience call during its OWN
// library init) is unknown. Given every FUSE-side and every LowFive-side
// path that could be read is excluded, a pre-`main()` static initialiser
// in MOAB, TempestRemap, or netCDF -- running before `coupler_main()`'s
// own `unsetenv()` calls have a chance to execute at all -- is the
// LEADING candidate, but it is a candidate, not a settled claim.
//
// (d) The host experiment that would settle it: an `lldb` batch session
// on the COUPLER's own MPMD segment only (U and V run normally alongside
// it, through the SAME `mpiexec` invocation `test/parallel/
// brusselator_participants/run_participants.sh`'s own `mpmd` branch
// builds), breaking on FOUR symbols: `main` (an ordering anchor),
// `H5open` (library init), `H5VLregister_connector` (the UNVALUED form --
// programmatic registration, e.g. `register_plugin()`), and
// `H5VLregister_connector_by_value` (the env-route form, `_str_to_info`'s
// own call). Each breakpoint auto-prints a backtrace and continues, so
// the session runs to the real crash (or to a clean exit, once fixed)
// without manual stepping. Outcomes and what each would show:
// (i) `H5open`/either register_connector variant fires BEFORE `main` --
// a pre-`main` static initialiser in a linked library (MOAB/TempestRemap/
// netCDF/LowFive's own dylib load) is the cause; the backtrace's own top
// frame (even if from an anonymous `.init` thunk `lldb` cannot fully
// symbolise) narrows WHICH library by which `lldb image list`/`image
// lookup -a <pc>` reports at that address. (ii) `H5VLregister_connector`
// (unvalued) fires AFTER `main`, with a backtrace rooted in
// `coupler_main`'s own call chain -- DISPROVES the exclusions (1)/(2)/(3)
// above and names the exact FUSE function that was missed. (iii)
// `H5VLregister_connector_by_value` fires AFTER `main`, despite the
// `unsetenv()` -- the env-route parser still ran from SOME value (a
// stale environment snapshot taken before the `unsetenv()`, or
// `HDF5_VOL_CONNECTOR` propagated through a DIFFERENT channel than the
// three names checked, e.g. an MPI-launcher-level `-genv` not visible to
// a plain `getenv()` audit) -- names the env route as the cause after
// all, through a different mechanism than (a) assumed. (iv) NEITHER
// register-connector breakpoint ever fires, yet the crash still happens
// -- the SAME symbol name resolves differently in the installed build (a
// static/internal-linkage variant, or symbol versioning) and the
// experiment should re-target the breakpoint at `H5_term_library`/
// `H5VL_term_package` instead, read backward from there.
//
// U's and V's own H5close()/finalize path (whether it ALSO exits cleanly
// once the coupler stops aborting) is unmeasured, because mpiexec kills
// them before their exit codes are observed. To observe it: run U and V
// under the SAME `mpiexec` MPMD command with each rank's own
// `--outfile-pattern`-redirected stdout/stderr kept (rather than
// discarded on a killed run), and add one trace line immediately AFTER
// each rank's own `MPI_Finalize()` call in `u_main.c`/`v_main.f90`/
// `bruss_lowfive_finalize()` reporting that rank's own process is about
// to exit 0 cleanly -- printed only if execution reaches that point, so
// its ABSENCE on a killed run is itself informative (distinguishes
// "U/V exited cleanly, killed after" from "U/V never reached their own
// clean-exit line").
enum class LowFiveProcessRole { kAttachingParticipant, kNonAttachingCoordinator };

/// PURE data: the three env-route variables a non-attaching
/// coordinator process (the coupler) must `unsetenv()` before its own
/// first native HDF5 call, so that HDF5's own env-route default-VOL
/// resolution never registers the "lowfive" connector class in a
/// process that has no deliberate use for it at all. The SAME three
/// names `examples/CMakeLists.txt`'s own ctest `ENVIRONMENT` property
/// sets process-wide for the whole MPMD launch (this header's own
/// "EXIT HYGIENE" comment above has the full citation).
[[nodiscard]] inline std::vector<std::string> EnvRouteVolVariableNames() {
  return {"HDF5_VOL_CONNECTOR", "HDF5_PLUGIN_PATH", "FUSE_LOWFIVE_BINDINGS"};
}

struct ExitHygienePlan {
  bool unregister_env_route_vol = false;
  bool explicit_h5close_before_mpi_finalize = false;
};

/// PURE: which of the two mitigations applies to which
/// role -- a non-attaching coordinator (the coupler) never calls
/// `bruss_lowfive_attach()`, so its own fix is to prevent the env-route
/// connector from ever registering at all; an attaching participant
/// (U or V) deliberately attaches and needs a deterministic, explicit
/// `H5close()` before its own `MPI_Finalize()` instead (this header's
/// own "EXIT HYGIENE" comment above has the full reasoning for
/// why each role gets the fix it gets, not the other one).
[[nodiscard]] inline ExitHygienePlan decide_exit_hygiene_plan(LowFiveProcessRole role) {
  if (role == LowFiveProcessRole::kNonAttachingCoordinator) {
    return ExitHygienePlan{/*unregister_env_route_vol=*/true,
                           /*explicit_h5close_before_mpi_finalize=*/false};
  }
  return ExitHygienePlan{/*unregister_env_route_vol=*/false,
                         /*explicit_h5close_before_mpi_finalize=*/true};
}

}  // namespace fuse_examples::lowfive_attach
