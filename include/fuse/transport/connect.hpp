#pragma once

/// Forms the ONE base communicator `RunnerConfig::base_communicator`
/// (include/fuse/runtime/runner.hpp) and `MpiTransport::set_communicator`
/// (include/fuse/transport/transport.hpp) already assume: every rank of
/// every component in the graph, numbered exactly as the graph's `RankSet`s
/// say. Two launch styles produce it (fuse/model/launch.hpp's `LaunchMode`):
/// a single MPMD `mpiexec` invocation, where the work is reading
/// `MPI_APPNUM` and checking it against the graph; and separately launched
/// executables, where the work is a connection handshake (`MPI_Open_port`/
/// `MPI_Comm_accept`/`MPI_Comm_connect`) followed by `MPI_Intercomm_merge`.
///
/// SPLIT FOR TESTABILITY. Everything that can be decided from the graph
/// alone, or from already-gathered data, involves no MPI call and is an
/// ordinary function below -- the same reasoning as
/// include/fuse/transport/rank_agreement.hpp, written down there in full.
/// Only the functions that actually call an MPI routine are behind
/// `#if FUSE_HAVE_MPI`, so `validate_launch_layout()`, `check_mpmd_layout()`
/// and `wait_for_port_file()` compile and run at the base build
/// configuration, where most of this project's tests run, instead of only
/// inside a two-rank job this sandbox cannot start (KNOWN_ISSUES.md C1).
///
/// COLLECTIVE AGREEMENT, THE EASY WAY. `connect()`'s refusals
/// (`E-CONNECT-SPEC`, `E-CONNECT-APPNUM`) are computed from data that is
/// already identical on every rank before the check runs -- the graph
/// itself (every rank parses the same descriptor) and, for
/// `check_mpmd_layout()`, an `MPI_Allgather`ed array rather than any rank's
/// private view. A deterministic function of identical input returns an
/// identical answer on every rank without a further reduction; see
/// KNOWN_ISSUES.md D3 for why a collective path must never return on a
/// locally computed condition, and how `MpiTransport::build_plan` earns the
/// same property in a harder case (its validation is NOT already uniform,
/// so it reduces with `MPI_Allreduce` before anyone acts on it).

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "fuse/base/config.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/launch.hpp"

namespace fuse::transport {

/// Whether this build offers the `Connect` launch mode, decided at configure
/// time by `tools/probe-mpi-dynamic.sh`: an MPI may provide
/// `MPI_Comm_accept`/`MPI_Comm_connect` and still be unable to use them
/// between two jobs on one node (`KNOWN_ISSUES.md` D23). `Mpmd` is not
/// affected and is never gated on this.
inline constexpr bool kConnectAcceptEnabled =
#if defined(FUSE_CONNECT_ACCEPT_ENABLED) && FUSE_CONNECT_ACCEPT_ENABLED
    true;
#else
    false;
#endif

/// The refusal `connect()` returns for a `Connect`-mode component when this
/// build does not offer that mode. Separate from the switch that calls it so
/// that it can be made to fire without rebuilding: the condition is settled
/// at configure time, and a refusal nothing can exercise is a refusal nobody
/// has checked.
[[nodiscard]] Result<void> check_connect_accept_available(
    bool enabled, const std::string& component_name);

// (Deliberately above the `#if FUSE_HAVE_MPI` below: the verdict is a build
// setting, not an MPI call, and a tier-0 build must be able to exercise the
// refusal. Everything after the guard needs a real MPI to say anything.)


/// What `connect()` needs beyond the graph and this process's local world.
struct ConnectOptions {
  /// Which component of `g` this process's local `MPI_COMM_WORLD` (or, in
  /// `Mpmd` mode, this process's slice of the shared one) forms. Required:
  /// an executable built for `test/parallel/connect` links one graph but
  /// plays one component's part, and `connect()` has no other way to know
  /// which.
  ComponentId my_component;

  /// `Connect` mode, the non-acceptor side only: how long to wait for the
  /// acceptor's port name to appear at `LaunchSpec::port_file` before
  /// refusing with `E-CONNECT-TIMEOUT`. A partner that never launches must
  /// become a named refusal, not a process that blocks forever in
  /// `MPI_Comm_connect` -- see connect.hpp's own module comment and
  /// KNOWN_ISSUES.md D3's "a collective must agree before anyone returns".
  double timeout_s = 30.0;

  /// How often to re-check for the port file while waiting. The residual
  /// blocking window this leaves is real and is written down rather than
  /// hidden: between a poll finding the file and `MPI_Comm_connect`
  /// actually being called, nothing times out `MPI_Comm_connect` itself --
  /// that call blocks until the acceptor's own `MPI_Comm_accept` is
  /// reached, which is a synchronous MPI call and not a window this
  /// library can poll around.
  double poll_interval_s = 0.05;
};

/// The result of a successful `connect()`: a Fortran communicator handle
/// (`MPI_Comm_c2f` of the merged base communicator), in exactly the form
/// `RunnerConfig::base_communicator` and `MpiTransport::set_communicator`
/// already take.
struct ConnectedWorld {
  int base_comm_f = -1;

  /// What `disconnect()` must do with `base_comm_f`. The two launch modes
  /// produce communicators with different obligations, and the difference
  /// is not visible from the handle itself, so it is recorded here.
  enum class Teardown {
    None,        ///< Nothing to release (a failed or default-constructed handle).
    Free,        ///< `Mpmd`: a duplicate of the caller's own world.
    Disconnect,  ///< `Connect`: merged from an intercommunicator, so the two
                 ///< separately launched programs are CONNECTED in the MPI
                 ///< sense until this is done. `MPI_Finalize` is collective
                 ///< over connected programs; leaving this undone is what
                 ///< crashed the acceptor inside MPICH's own communicator
                 ///< bookkeeping (`MPII_CommL_forget`, reading a freed
                 ///< communicator) on the host, 2026-10-08. The connector
                 ///< then failed too, in whatever way it happened to be
                 ///< waiting -- an MPICH-internal communicator failure in
                 ///< the `ctest` run, `E-CONNECT-TIMEOUT` in a later manual
                 ///< run -- always as collateral of the dead peer, never on
                 ///< its own account.
  };
  Teardown teardown = Teardown::None;
};

/// Pure, MPI-free. Checks that the graph's launched components' `RankSet`s
/// -- every component whose `LaunchSpec::mode` is not `InProcess` -- form a
/// **partition of the declared world into consecutive blocks, in the
/// graph's own component order**. Not merely "disjoint and covering": the
/// stronger, consecutive-block requirement is what the two launch
/// mechanisms actually produce --
///
///   * `MPI_Intercomm_merge` concatenates one side's ranks after the
///     other's, in whichever order the two sides agree to merge in (which
///     `connect()` fixes to the graph's component order, see that
///     function's own comment);
///   * launcher-style MPMD assigns rank 0..k-1 to the first `mpiexec`
///     segment, k..k+m-1 to the second, and so on, in command-line order,
///     which this library requires to match the graph's own component
///     order so the mapping is the same thing stated twice rather than two
///     independent claims that could disagree.
///
/// A declaration that is merely disjoint-and-covering but not consecutive
/// (e.g. two components interleaved on alternating ranks) could never
/// actually be produced by either mechanism, so it is refused here before
/// `connect()` goes anywhere near MPI.
///
/// Refuses `E-CONNECT-SPEC` when: no component declares a launch mode other
/// than `InProcess` (nothing for `connect()` to do); a launched component's
/// `RankSet` is empty or not contiguous; or a launched component does not
/// start exactly where the previous launched component (in graph order)
/// left off.
[[nodiscard]] inline Result<void> validate_launch_layout(const CouplingGraph& g) {
  int expected_start = 0;
  bool any = false;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    // Plugin, like InProcess, shares the host process's own communicator
    // (launch.hpp's own doc on LaunchMode::Plugin): neither needs a rank
    // block in the mpmd/connect layout this function checks.
    if (c.launch.mode == LaunchMode::InProcess || c.launch.mode == LaunchMode::Plugin) continue;
    any = true;
    if (c.ranks.empty() || !c.ranks.is_contiguous()) {
      return Failure{
          "E-CONNECT-SPEC",
          "component '" + c.name +
              "' declares an mpmd or connect launch but its ranks are not a "
              "single contiguous, non-empty block; MPI_Intercomm_merge and "
              "MPMD's own rank numbering both assign one contiguous block "
              "per program, so a non-contiguous declaration could never be "
              "produced by either"};
    }
    if (c.ranks.min_rank() != expected_start) {
      return Failure{
          "E-CONNECT-SPEC",
          "component '" + c.name + "' starts at rank " +
              std::to_string(c.ranks.min_rank()) +
              ", but the launched components before it in the graph's own "
              "order occupy ranks [0, " + std::to_string(expected_start) +
              "); the launched components' rank sets must be consecutive "
              "blocks in the graph's component order, because that order is "
              "what fixes MPI_Intercomm_merge's and MPMD's own rank "
              "numbering"};
    }
    expected_start = c.ranks.max_rank() + 1;
  }
  if (!any) {
    return Failure{
        "E-CONNECT-SPEC",
        "no component in the graph declares an mpmd or connect launch; "
        "connect() has nothing to form a base communicator for"};
  }
  return Result<void>::success();
}

/// Pure, MPI-free. `appnum_by_rank[r]` is the `MPI_APPNUM` rank `r` of the
/// MPMD world reported (already gathered -- every rank sees the same
/// array, which is what makes the refusal this computes agreed without a
/// further reduction; see this header's module comment).
///
/// Checks, for every `Mpmd`-mode component in `g`: its `appnum` is unique
/// among `Mpmd` components; some rank actually reports that appnum; and the
/// set of ranks that report it is EXACTLY that component's declared
/// `RankSet` (not a superset, not a subset -- `RankSet::operator==`, both
/// sorted and de-duplicated by construction). Also checks the converse:
/// every appnum some rank reports names a declared `Mpmd` component.
///
/// Refuses `E-CONNECT-APPNUM` for any of the above; refuses `E-CONNECT-SPEC`
/// if two `Mpmd` components declare the same `appnum` (a graph-level
/// defect, not a mismatch against the runtime layout).
[[nodiscard]] inline Result<void> check_mpmd_layout(
    const CouplingGraph& g, const std::vector<int>& appnum_by_rank) {
  std::unordered_map<int, ComponentId> component_by_appnum;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    const auto& c = g.component(ComponentId{i});
    if (c.launch.mode != LaunchMode::Mpmd) continue;
    const auto [it, inserted] = component_by_appnum.emplace(c.launch.appnum, ComponentId{i});
    if (!inserted) {
      return Failure{
          "E-CONNECT-SPEC",
          "components '" + g.component(it->second).name + "' and '" + c.name +
              "' both declare mpmd appnum " + std::to_string(c.launch.appnum)};
    }
  }

  std::unordered_map<int, std::vector<int>> ranks_by_appnum;
  for (std::size_t r = 0; r < appnum_by_rank.size(); ++r) {
    ranks_by_appnum[appnum_by_rank[r]].push_back(static_cast<int>(r));
  }

  for (const auto& [appnum, ranks] : ranks_by_appnum) {
    const auto it = component_by_appnum.find(appnum);
    if (it == component_by_appnum.end()) {
      return Failure{
          "E-CONNECT-APPNUM",
          "rank(s) report MPI_APPNUM " + std::to_string(appnum) +
              ", which no mpmd component in the graph declares"};
    }
    const ComponentDecl& comp = g.component(it->second);
    const RankSet actual{ranks};
    if (!(comp.ranks == actual)) {
      return Failure{
          "E-CONNECT-APPNUM",
          "component '" + comp.name + "' declares appnum " +
              std::to_string(appnum) +
              " with a rank set that does not match the ranks that "
              "actually report that appnum at this launch"};
    }
  }
  for (const auto& [appnum, cid] : component_by_appnum) {
    if (ranks_by_appnum.find(appnum) == ranks_by_appnum.end()) {
      return Failure{
          "E-CONNECT-APPNUM",
          "component '" + g.component(cid).name + "' declares appnum " +
              std::to_string(appnum) + ", which no rank in this launch reports"};
    }
  }
  return Result<void>::success();
}

/// Pure, MPI-free. Polls for a non-empty first line at `path`, the way the
/// `Connect`-mode non-acceptor side waits for the acceptor's published port
/// name. Returns that line (the port name) or `E-CONNECT-TIMEOUT` once
/// `timeout_s` has elapsed since the call began -- checked BEFORE the first
/// sleep, so `timeout_s == 0` refuses immediately rather than sleeping once
/// and only then noticing, which is what makes the refusal path reachable
/// without a filesystem race in a tier-0 test.
[[nodiscard]] inline Result<std::string> wait_for_port_file(
    const std::string& path, double timeout_s, double poll_interval_s) {
  const auto deadline = std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
          std::chrono::duration<double>(timeout_s));
  for (;;) {
    std::ifstream in(path);
    if (in) {
      std::string line;
      if (std::getline(in, line) && !line.empty()) return line;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      return Failure{
          "E-CONNECT-TIMEOUT",
          "no port name appeared at '" + path + "' within " +
              std::to_string(timeout_s) + "s"};
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(poll_interval_s));
  }
}

#if FUSE_HAVE_MPI
/// Forms the base communicator for the component named by
/// `opts.my_component`, from this process's local world
/// (`MPI_Comm_f2c(my_world_comm_f)`), per that component's declared
/// `LaunchSpec`:
///
///   * `InProcess`: refused (`E-CONNECT-SPEC`) -- there is nothing to
///     connect, `my_world_comm_f` already IS the base communicator, and a
///     caller that reaches here for an in-process component has
///     misconfigured something.
///   * `Mpmd`: `my_world_comm_f` must already be the single MPMD world
///     (every rank of every program). Reads `MPI_APPNUM`, `MPI_Allgather`s
///     it, checks the result with `check_mpmd_layout()`, and returns a
///     duplicate of the given communicator -- already numbered correctly,
///     nothing to merge.
///   * `Connect`: `my_world_comm_f` is this executable's OWN world only.
///     Supports exactly the two-component case `test/parallel/connect`
///     exercises: finds the one other `Connect`-mode component, opens or
///     waits for the connection, and merges the resulting intercommunicator
///     with `MPI_Intercomm_merge`, fixing high/low by the two components'
///     order in the graph (not by which one is the acceptor) so the merged
///     numbering is the same thing on both sides regardless of which side
///     happens to publish the port.
///
/// Every path calls `validate_launch_layout()` first, so `E-CONNECT-SPEC`
/// is reported identically regardless of launch mode.
[[nodiscard]] Result<ConnectedWorld> connect(const CouplingGraph& g,
                                             int my_world_comm_f,
                                             const ConnectOptions& opts);

/// Releases what `connect()` formed, and MUST be called before
/// `MPI_Finalize` by every caller of `connect()`.
///
/// For a `Connect`-mode handle this is `MPI_Comm_disconnect`, and it is not
/// housekeeping: `MPI_Comm_accept`/`MPI_Comm_connect` leave the two
/// separately launched programs CONNECTED, which makes `MPI_Finalize`
/// collective across both of them. A program that finalizes while still
/// connected is not doing something merely untidy -- it is leaving an
/// operation half-finished on a peer that is no longer listening.
/// For an `Mpmd` handle it is `MPI_Comm_free` of the duplicate `connect()`
/// made; for a default-constructed or already-released handle it is a
/// no-op success, so calling it twice, or on the result of a failed
/// `connect()`, is safe.
///
/// `base_comm_f` is set to -1 and the teardown to `None` on success, so the
/// handle cannot be released twice. The communicator must be out of use by
/// then: anything that duplicated it (`MpiTransport::set_communicator`, a
/// `Runner` built over it) must be destroyed FIRST. Called after
/// `MPI_Finalize` it refuses with `E-CONNECT-FINALIZED` rather than
/// reaching into a torn-down MPI.
[[nodiscard]] Result<void> disconnect(ConnectedWorld& w);
#endif  // FUSE_HAVE_MPI

}  // namespace fuse::transport
