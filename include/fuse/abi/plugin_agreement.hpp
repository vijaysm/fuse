#pragma once

/// KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
/// 2026-10-22, verbatim: "...when a multi-executable launch has no base
/// communicator spanning all executables before connect(), validate a
/// priori during init and throw an error with details on why this cannot
/// be run"). SPLIT FOR TESTABILITY, the same reason include/fuse/abi/
/// collective_reconcile.hpp's own module comment gives for the
/// reconciliation arithmetic: whether THIS process's own base
/// communicator can be trusted to span every executable a Plugin
/// component's mount agreement (src/abi/participant.cpp's
/// reconcile_plugin_agreement(), KNOWN_ISSUES.md D15) would need to
/// reconcile over is a deterministic function of the loaded graph and this
/// process's own setup() options -- no MPI call in it, so a tier-0 test
/// fires every branch directly.
///
/// WHY THIS CANNOT BE A COLLECTIVE CHECK. The whole reason a priori
/// validation exists is that, in the one launch shape this header refuses,
/// there IS no communicator that spans every executable -- not yet, and
/// (because fuse::transport::connect() is the ONLY thing that ever forms
/// one, and this process's own setup() options tell it not to call
/// connect() at all) never. A collective call needs a communicator to be
/// collective OVER; asking every rank of a communicator that does not
/// include the ranks this check is worried about would not detect the
/// problem, it would just omit them from the vote. The check below is
/// therefore local: every process that will ever call fuse_initialize()
/// on this graph decides the SAME answer from the SAME graph bytes and
/// its OWN setup() options (per include/fuse/model/launch.hpp's own
/// comment, a component's LaunchSpec is round-tripped identically to
/// every process that loads this graph), so "every process refuses
/// identically, with no rank left to disagree or to hang waiting for a
/// peer" follows from determinism, not from a reduction.
///
/// WHICH LAUNCH SHAPE IS REFUSED, precisely. `fuse::LaunchMode::Mpmd`'s
/// own doc comment (include/fuse/model/launch.hpp) states the property
/// that makes it SAFE here: "MPI_COMM_WORLD already contains every rank
/// of every program" before fuse_setup() is ever called, so whatever
/// communicator a caller hands to setup() already spans every executable
/// an Mpmd-launched component's partner lives in, with no connect() call
/// needed for that to be true. `fuse::LaunchMode::Connect`'s own doc
/// comment states the opposite property: "independent MPI universes, no
/// common parent communicator" until connect()'s own MPI_Open_port/
/// MPI_Comm_accept/MPI_Comm_connect handshake runs. `fuse::LaunchMode::
/// Plugin` (and InProcess, and LowFive today) never call connect() at all
/// (entries.def's own comment on `my_component`: "for InProcess or
/// Plugin, behaves exactly as when NULL ... there is nothing to connect
/// and the given communicator already is the base one"), so a process
/// whose OWN my_component is Plugin (or unset) never gains a merged
/// communicator -- its own `communicator` is whatever was handed to
/// setup(), unmodified. Put together: a graph that declares BOTH a
/// Plugin component AND a Connect component asks this process -- if its
/// own my_component is not itself the Connect half of that handshake --
/// to reconcile a plugin-mount verdict over a communicator that was never
/// grown to include the Connect-launched executable at all. Refused here,
/// before fuse_initialize() does any of the (wasted) work reaching the
/// component-wiring loop would otherwise do first.
///
/// WHAT THIS DOES NOT COVER (recorded, not guessed past). A graph with
/// MORE THAN ONE Connect-launched component, paired into more than one
/// independent two-executable handshake, is not fully modelled: this
/// check only asks "does my own my_component connect at all", not "does
/// its own connect() result include the SPECIFIC executable that hosts
/// the Plugin component" -- a graph shaped that way could still reconcile
/// over a communicator that spans the WRONG subset of executables,
/// undetected by this function. Flagged in this round's own open
/// questions (KNOWN_ISSUES.md D15) rather than solved here: the fully
/// general version needs the launch topology connect() itself validates
/// (fuse::transport::validate_launch_layout(), include/fuse/transport/
/// connect.hpp) consulted BEFORE connect() runs, which this function
/// deliberately does not attempt.

#include <cstdint>
#include <optional>

#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/launch.hpp"

namespace fuse::abi_detail {

/// True iff `graph` declares at least one component whose own LaunchSpec
/// has `mode`. A plain linear scan of `graph.component_count()` -- no
/// index by LaunchMode exists or is worth building for a check that runs
/// once, at most, per fuse_initialize() call.
[[nodiscard]] inline bool graph_has_launch_mode(const fuse::CouplingGraph& graph,
                                                fuse::LaunchMode mode) noexcept {
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(graph.component_count()); ++i) {
    if (graph.component(fuse::ComponentId{i}).launch.mode == mode) return true;
  }
  return false;
}

/// True exactly when `fuse_initialize()` must refuse THIS process's own
/// attempt to mount a Plugin component a priori, per this header's own
/// module comment: the graph declares a Plugin component, the graph ALSO
/// declares a Connect-launched one (a genuinely separate executable with
/// no shared communicator before its own handshake), and THIS process's
/// own `my_component` will not itself call fuse::transport::connect() --
/// so there is no way for this process's own base communicator to have
/// been grown to include the Connect-launched executable.
///
/// `my_component_mode` is the LaunchMode of the component THIS process's
/// own `my_component` setup() option names (std::nullopt when
/// `my_component` was never set, or named a component this graph does
/// not declare -- impl_setup()'s own E-ABI-SETUP-COMPONENT already
/// refuses the latter before this function would ever be reached, but
/// std::nullopt is the same answer either way: no connect() call is
/// coming). `communicator_set` mirrors fuse_ctx::communicator_set: with
/// no communicator configured at all, there is only one process in this
/// run by construction (context.hpp's own comment on `backend`), so there
/// is no OTHER executable for a Plugin component's agreement to ever need
/// to reach -- refusing here would reject a legitimate, single-process
/// exercise of a graph that merely DECLARES a Connect component without
/// ever running it that way.
[[nodiscard]] inline bool plugin_mount_agreement_needs_a_missing_spanning_communicator(
    const fuse::CouplingGraph& graph, std::optional<fuse::LaunchMode> my_component_mode,
    bool communicator_set) noexcept {
  if (!communicator_set) return false;
  const bool my_component_connects =
      my_component_mode.has_value() &&
      (*my_component_mode == fuse::LaunchMode::Mpmd ||
       *my_component_mode == fuse::LaunchMode::Connect);
  if (my_component_connects) return false;
  return graph_has_launch_mode(graph, fuse::LaunchMode::Plugin) &&
         graph_has_launch_mode(graph, fuse::LaunchMode::Connect);
}

}  // namespace fuse::abi_detail
