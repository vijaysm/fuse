#include "fuse/transport/connect.hpp"

#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace fuse::transport {
namespace {

[[nodiscard]] std::string mpi_error_text(int rc) {
  char buf[MPI_MAX_ERROR_STRING];
  int len = 0;
  if (MPI_Error_string(rc, buf, &len) != MPI_SUCCESS || len <= 0) {
    return "MPI error code " + std::to_string(rc);
  }
  return std::string(buf, static_cast<std::size_t>(len));
}

/// Every `Connect`-mode component, in the graph's own component order. That
/// order, not acceptor/connector role, is what `connect()` fixes
/// `MPI_Intercomm_merge`'s high/low argument by, so the merged numbering is
/// the same thing on both sides regardless of which side happens to
/// publish the port.
[[nodiscard]] std::vector<ComponentId> connect_mode_components(const CouplingGraph& g) {
  std::vector<ComponentId> out;
  for (std::uint32_t i = 0; i < g.component_count(); ++i) {
    if (g.component(ComponentId{i}).launch.mode == LaunchMode::Connect) {
      out.push_back(ComponentId{i});
    }
  }
  return out;
}

/// Writes `port_name` to `path` so a concurrent reader (another process's
/// `wait_for_port_file`) never observes a partially written file: write to
/// a sibling temporary name on the same directory, then atomically rename
/// it into place. A reader that opened `path` mid-write under a direct
/// `std::ofstream(path)` could see a truncated line and fail to parse a
/// port name that was, in full, perfectly valid.
[[nodiscard]] Result<void> publish_port_file(const std::string& path,
                                             const std::string& port_name) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) {
      return Failure{"E-CONNECT-MPI", "could not open '" + tmp + "' for writing"};
    }
    out << port_name << '\n';
    if (!out) {
      return Failure{"E-CONNECT-MPI", "could not write the port name to '" + tmp + "'"};
    }
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    return Failure{"E-CONNECT-MPI",
                   "could not rename '" + tmp + "' to '" + path + "'"};
  }
  return Result<void>::success();
}

[[nodiscard]] Result<ConnectedWorld> connect_mpmd(const CouplingGraph& g, MPI_Comm world) {
  int size = 0;
  int rc = MPI_Comm_size(world, &size);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Comm_size failed: " + mpi_error_text(rc)};
  }

  int* appnum_ptr = nullptr;
  int flag = 0;
  rc = MPI_Comm_get_attr(world, MPI_APPNUM, &appnum_ptr, &flag);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Comm_get_attr(MPI_APPNUM) failed: " + mpi_error_text(rc)};
  }
  if (flag == 0) {
    return Failure{"E-CONNECT-SPEC",
                   "MPI_APPNUM is not set on this communicator; this process "
                   "was not launched as part of an MPMD mpiexec invocation"};
  }
  const int my_appnum = *appnum_ptr;

  // Gathered onto every rank, so check_mpmd_layout()'s refusal is a
  // deterministic function of identical data everywhere -- see
  // connect.hpp's own module comment for why that makes it agreed without
  // a further reduction.
  std::vector<int> appnum_by_rank(static_cast<std::size_t>(size));
  rc = MPI_Allgather(&my_appnum, 1, MPI_INT, appnum_by_rank.data(), 1, MPI_INT, world);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Allgather of MPI_APPNUM failed: " + mpi_error_text(rc)};
  }

  if (auto r = check_mpmd_layout(g, appnum_by_rank); !r.ok()) return r.error();

  MPI_Comm dup = MPI_COMM_NULL;
  rc = MPI_Comm_dup(world, &dup);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Comm_dup failed: " + mpi_error_text(rc)};
  }
  ConnectedWorld out;
  out.base_comm_f = static_cast<int>(MPI_Comm_c2f(dup));
  out.teardown = ConnectedWorld::Teardown::Free;
  return out;
}

[[nodiscard]] Result<ConnectedWorld> connect_pairwise(const CouplingGraph& g,
                                                      ComponentId my_id,
                                                      MPI_Comm world,
                                                      const ConnectOptions& opts) {
  const std::vector<ComponentId> order = connect_mode_components(g);
  if (order.size() != 2) {
    return Failure{"E-CONNECT-SPEC",
                   "pairwise connect() supports exactly two connect-mode "
                   "components; the graph declares " +
                       std::to_string(order.size())};
  }
  const bool i_am_first = (order[0] == my_id);
  const ComponentId partner_id = i_am_first ? order[1] : order[0];
  const ComponentDecl& mine = g.component(my_id);
  const ComponentDecl& partner = g.component(partner_id);

  if (mine.launch.is_acceptor == partner.launch.is_acceptor) {
    return Failure{"E-CONNECT-SPEC",
                   "components '" + mine.name + "' and '" + partner.name +
                       "' must have exactly one acceptor between them"};
  }
  const std::string& connect_file =
      !mine.launch.port_file.empty() ? mine.launch.port_file : partner.launch.port_file;
  if (connect_file.empty()) {
    return Failure{"E-CONNECT-SPEC",
                   "neither '" + mine.name + "' nor '" + partner.name +
                       "' declares a port_file; service_name connection is "
                       "not implemented by this connect()"};
  }

  int rank = 0;
  int rc = MPI_Comm_rank(world, &rank);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Comm_rank failed: " + mpi_error_text(rc)};
  }

  MPI_Comm intercomm = MPI_COMM_NULL;
  if (mine.launch.is_acceptor) {
    char port_name[MPI_MAX_PORT_NAME] = {};
    if (rank == 0) {
      rc = MPI_Open_port(MPI_INFO_NULL, port_name);
      if (rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI", "MPI_Open_port failed: " + mpi_error_text(rc)};
      }
      if (auto r = publish_port_file(connect_file, port_name); !r.ok()) return r.error();
    }
    // Every rank of `world` must call MPI_Comm_accept with the SAME port
    // name (it is collective over this side's local group), so rank 0's
    // name is broadcast rather than reopened per rank.
    rc = MPI_Bcast(port_name, MPI_MAX_PORT_NAME, MPI_CHAR, 0, world);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI", "MPI_Bcast of the port name failed: " + mpi_error_text(rc)};
    }
    rc = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, world, &intercomm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI", "MPI_Comm_accept failed: " + mpi_error_text(rc)};
    }
    if (rank == 0) MPI_Close_port(port_name);
  } else {
    const auto port_result =
        wait_for_port_file(connect_file, opts.timeout_s, opts.poll_interval_s);
    if (!port_result.ok()) return port_result.error();
    rc = MPI_Comm_connect(port_result.value().c_str(), MPI_INFO_NULL, 0, world, &intercomm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI", "MPI_Comm_connect failed: " + mpi_error_text(rc)};
    }
  }

  // Fixed by component order, not acceptor/connector role (this file's own
  // connect_pairwise() comment above and connect.hpp's `connect()` doc).
  MPI_Comm merged = MPI_COMM_NULL;
  rc = MPI_Intercomm_merge(intercomm, i_am_first ? 0 : 1, &merged);
  // MPI_Comm_disconnect, not MPI_Comm_free: this intercommunicator is the
  // connection itself, and freeing it locally leaves the peer's half in
  // place. Measured on the host 2026-10-08 -- with MPI_Comm_free here the
  // acceptor took SIGSEGV inside MPICH's own communicator bookkeeping
  // (MPII_CommL_forget, reading a freed communicator) about a tenth of a
  // second in, long before MPI_Finalize, and the connector then failed
  // dup'ing the merged communicator. Both sides reach this point together,
  // so the collective is matched. The merged communicator survives it and
  // is what the caller gets; it is released separately by disconnect().
  MPI_Comm_disconnect(&intercomm);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI", "MPI_Intercomm_merge failed: " + mpi_error_text(rc)};
  }

  ConnectedWorld out;
  out.base_comm_f = static_cast<int>(MPI_Comm_c2f(merged));
  // Not Free: this communicator carries the connection between two
  // separately launched programs (connect.hpp's own Teardown comment).
  out.teardown = ConnectedWorld::Teardown::Disconnect;
  return out;
}

}  // namespace

Result<ConnectedWorld> connect(const CouplingGraph& g, int my_world_comm_f,
                               const ConnectOptions& opts) {
  if (auto r = validate_launch_layout(g); !r.ok()) return r.error();

  if (!opts.my_component.valid() || opts.my_component.value() >= g.component_count()) {
    return Failure{"E-CONNECT-SPEC",
                   "ConnectOptions::my_component does not name a component in the graph"};
  }
  const ComponentDecl& mine = g.component(opts.my_component);

  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return Failure{"E-CONNECT-MPI",
                   "MPI is not initialised; MPI_Comm_f2c on a handle from an "
                   "uninitialised MPI is not meaningful"};
  }
  const MPI_Comm world = MPI_Comm_f2c(static_cast<MPI_Fint>(my_world_comm_f));
  if (world == MPI_COMM_NULL) {
    return Failure{"E-CONNECT-SPEC",
                   "my_world_comm_f converts to MPI_COMM_NULL"};
  }

  switch (mine.launch.mode) {
    case LaunchMode::InProcess:
      return Failure{"E-CONNECT-SPEC",
                     "connect() called for component '" + mine.name +
                         "', which is declared in_process; there is nothing "
                         "to connect, and my_world_comm_f already is the base "
                         "communicator"};
    case LaunchMode::Mpmd:
      return connect_mpmd(g, world);
    case LaunchMode::Connect:
      if (auto r = check_connect_accept_available(kConnectAcceptEnabled,
                                                  mine.name);
          !r.ok()) {
        return r.error();
      }
      return connect_pairwise(g, opts.my_component, world, opts);
    case LaunchMode::Plugin:
      // A plugin is mounted (fuse::abi_detail::Registry), never launched: it
      // shares the host process and its communicator exactly like an
      // InProcess component, so connect() has nothing to do here either
      // (launch.hpp's own doc on why Plugin is not merely InProcess with a
      // path attached does not change this call's answer).
      return Failure{"E-CONNECT-SPEC",
                     "connect() called for component '" + mine.name +
                         "', which is declared plugin; there is nothing to "
                         "connect, and my_world_comm_f already is the base "
                         "communicator"};
    case LaunchMode::LowFive:
      // S6 phase 3 ("LowFive" track), Unit 3. Today's only implemented
      // case (launch.hpp's own doc on this value) is in-process: the
      // ranks a LowFive-launched component runs on come from whatever
      // OTHER launch story this run uses (or none at all), never from
      // this call -- same refusal shape as InProcess/Plugin above. The
      // distributed, cross-executable extension this value's own doc
      // comment names (DistMetadataVOL's own intercommunicator) is
      // explicitly NOT this function's job either way
      // (docs/user/lowfive.md's own "forming the intercommunicator is
      // explicitly NOT this transport's job" ruling for Demo B's
      // identical case).
      return Failure{"E-CONNECT-SPEC",
                     "connect() called for component '" + mine.name +
                         "', which is declared lowfive; there is nothing to "
                         "connect, and my_world_comm_f already is the base "
                         "communicator"};
  }
  return Failure{"E-CONNECT-SPEC", "unreachable launch mode"};
}


Result<void> disconnect(ConnectedWorld& w) {
  if (w.teardown == ConnectedWorld::Teardown::None || w.base_comm_f < 0) {
    // Already released, or never formed. Idempotent on purpose: a caller
    // unwinding from a failure should not have to work out whether it got
    // far enough to owe a teardown.
    w.base_comm_f = -1;
    w.teardown = ConnectedWorld::Teardown::None;
    return Result<void>::success();
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized != 0) {
    return Failure{"E-CONNECT-FINALIZED",
                   "disconnect() called after MPI_Finalize; the communicator "
                   "connect() formed can no longer be released, and for a "
                   "Connect-mode handle MPI_Finalize itself was collective "
                   "over a program this one was still connected to"};
  }
  MPI_Comm c = MPI_Comm_f2c(static_cast<MPI_Fint>(w.base_comm_f));
  if (c == MPI_COMM_NULL) {
    w.base_comm_f = -1;
    w.teardown = ConnectedWorld::Teardown::None;
    return Result<void>::success();
  }
  const bool is_connection =
      w.teardown == ConnectedWorld::Teardown::Disconnect;
  // Both are collective over the communicator, and MPI_Comm_disconnect is
  // additionally collective over the OTHER program: it returns only once
  // the peer reaches its own call, which is what makes the two programs
  // independently finalizable afterwards.
  const int rc = is_connection ? MPI_Comm_disconnect(&c) : MPI_Comm_free(&c);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-CONNECT-MPI",
                   std::string(is_connection ? "MPI_Comm_disconnect"
                                             : "MPI_Comm_free") +
                       " failed while releasing the base communicator: " +
                       mpi_error_text(rc)};
  }
  w.base_comm_f = -1;
  w.teardown = ConnectedWorld::Teardown::None;
  return Result<void>::success();
}

}  // namespace fuse::transport

#endif  // FUSE_HAVE_MPI

namespace fuse::transport {

Result<void> check_connect_accept_available(bool enabled,
                                            const std::string& component_name) {
  if (enabled) return Result<void>::success();
  return Failure{
      "E-CONNECT-ACCEPT-UNAVAILABLE",
      "component '" + component_name +
          "' is declared launch mode 'connect', but this build does not offer "
          "that mode: tools/probe-mpi-dynamic.sh measured that this MPI cannot "
          "connect two separately launched jobs on one node, so "
          "FUSE_ENABLE_CONNECT_ACCEPT is OFF (KNOWN_ISSUES.md D23). Launch the "
          "components as MPMD segments of one mpiexec instead, which needs "
          "nothing this MPI lacks, or build against an MPI whose dynamic "
          "process management works between same-node jobs -- for MPICH that "
          "means --with-device=ch4."};
}

}  // namespace fuse::transport

