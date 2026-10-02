#pragma once

#include <string>

namespace fuse {

/// How a component's ranks come to exist in the base communicator the
/// graph's `RankSet`s are numbered over
/// (`RunnerConfig::base_communicator`, include/fuse/runtime/runner.hpp).
///
/// `InProcess` (the default): every component of the graph runs in the same
/// executable and already shares `MPI_COMM_WORLD` -- or runs with no MPI at
/// all. `fuse::transport::connect()` is never called for a component
/// declared this way.
///
/// `Mpmd`: launcher-style multiple-program-multiple-data
/// (`mpiexec -np 2 ./a : -np 3 ./b`, upstream/soma's comm-graph design
/// calls this spec §6 mode 2). `MPI_COMM_WORLD` already contains every rank
/// of every program; `connect()` reads `MPI_APPNUM` to tell this program's
/// ranks from the others' and checks the result against `appnum` and the
/// declared `RankSet`s.
///
/// `Connect`: true separate-executable MPMD (spec §6 mode 3 -- independent
/// MPI universes, no common parent communicator). This component's
/// executable was launched with its own `mpiexec`, with `MPI_COMM_WORLD`
/// containing only its own ranks. `connect()` forms an intercommunicator to
/// its partner with `MPI_Open_port`/`MPI_Comm_accept` (the `is_acceptor`
/// side) or `MPI_Comm_connect` (the other side), the port name exchanged
/// through `port_file` or `service_name`, then merges that intercommunicator
/// into a base communicator with `MPI_Intercomm_merge`.
enum class LaunchMode { InProcess, Mpmd, Connect };

[[nodiscard]] constexpr const char* to_cstring(LaunchMode m) noexcept {
  switch (m) {
    case LaunchMode::InProcess: return "InProcess";
    case LaunchMode::Mpmd:      return "Mpmd";
    case LaunchMode::Connect:   return "Connect";
  }
  return "Unknown";
}

/// How THIS component's executable finds the rest of the graph. Declared
/// per component, not once for the whole graph, because a run can mix
/// launch styles -- today's default (`InProcess`) and the two added here
/// are additive launch styles a component picks, not alternative modes the
/// whole graph is locked into.
///
/// Deliberately NOT validated against its partners here: whether a
/// `Connect` component's declared partner actually declares the matching
/// half of the pair, and whether every launched component's `RankSet`s
/// together partition the declared world, are graph-wide questions that
/// `fuse::transport::validate_launch_layout()` (include/fuse/transport/
/// connect.hpp) answers once connect() is reached, where the error can
/// carry which component or which rank is implicated. Per-member checks
/// that need no other component's declaration (a non-negative `appnum`, a
/// non-empty connection target) are `validate()`'s job instead -- see
/// `E-LAUNCH-APPNUM` and `E-LAUNCH-CONNECT-TARGET` there.
struct LaunchSpec {
  LaunchMode mode = LaunchMode::InProcess;

  /// `Mpmd` only: the `MPI_APPNUM` this component's executable is launched
  /// as (the zero-based index of its segment of the `mpiexec ... : ...`
  /// command line). Negative is "not set"; an `Mpmd` component with a
  /// negative `appnum` is refused at graph build (`E-LAUNCH-APPNUM`) rather
  /// than left to fail, opaquely, the first time `connect()` reads the real
  /// `MPI_APPNUM` and has nothing to compare it against.
  int appnum = -1;

  /// `Connect` only: whether this component's executable calls
  /// `MPI_Open_port` + `MPI_Comm_accept` (true) rather than
  /// `MPI_Comm_connect` (false, the default -- the "client" side of the
  /// pair).
  bool is_acceptor = false;

  /// `Connect` only: the file path through which the acceptor's port name
  /// is published (it writes the name `MPI_Open_port` returned once that
  /// name is known; the other side polls for the file and reads it). Also
  /// the key the two sides use to agree they are opening a connection to
  /// EACH OTHER and not to some third component that also uses `Connect` --
  /// a pair's two `LaunchSpec`s must name the same `port_file` (or the same
  /// `service_name`). At least one of `port_file` or `service_name` is
  /// required for `Connect` (`E-LAUNCH-CONNECT-TARGET`).
  std::string port_file;

  /// `Connect` only: alternative to `port_file` -- an `MPI_Publish_name`
  /// service name, looked up with `MPI_Lookup_name` instead of a file. Not
  /// exercised by test/parallel/connect (no name service is assumed present
  /// on a test host), but declared and round-tripped because the graph
  /// model is the ABI's narrow waist (docs/dev/abi-design.md) and a
  /// consumer that does have one should not need a second way to say so.
  std::string service_name;
};

}  // namespace fuse
