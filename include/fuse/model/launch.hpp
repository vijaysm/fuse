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
///
/// `Plugin` (S6 phase 4, "Adapter host" track): this component is not
/// launched at all -- it is MOUNTED, by `fuse::abi_detail::Registry`, from a
/// shared object named by `plugin_path` that exports exactly one symbol,
/// `fuse_provider_abi` (include/fuse/abi/provider.h). A plugin-launched
/// component shares the host process and its communicator exactly the way
/// an `InProcess` one does -- `fuse::transport::connect()` is never called
/// for it either, for the identical reason -- so it is a FOURTH value of
/// this enum, not a flag combined with one of the other three: "where do my
/// ranks come from" (the three above) and "where does my code come from"
/// (in-process native vs. a dlopen()'d plugin) are different questions for
/// every OTHER mode, but collapse to one for Plugin, which is always
/// in-process by construction.
enum class LaunchMode { InProcess, Mpmd, Connect, Plugin };

[[nodiscard]] constexpr const char* to_cstring(LaunchMode m) noexcept {
  switch (m) {
    case LaunchMode::InProcess: return "InProcess";
    case LaunchMode::Mpmd:      return "Mpmd";
    case LaunchMode::Connect:   return "Connect";
    case LaunchMode::Plugin:    return "Plugin";
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

  /// `Plugin` only: the path to a shared object exporting `fuse_provider_abi`
  /// (include/fuse/abi/provider.h). Required and non-empty for `Plugin`
  /// (`E-LAUNCH-PLUGIN-PATH` otherwise, `fuse/model/validate.hpp`); resolved
  /// however the running process resolves a dlopen() path (absolute, or
  /// relative to the process's own working directory -- this model carries
  /// the string verbatim and states no policy of its own).
  std::string plugin_path;
};

}  // namespace fuse
