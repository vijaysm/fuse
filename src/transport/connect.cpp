#include "fuse/transport/connect.hpp"

#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI

#include <mpi.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

// Optional trace of which rendezvous step each process is in. A chain failure
// (a fast error versus a 120s timeout) is distinguishable, on a host, only from
// WHICH rendezvous step each process was in when it stopped, which the launcher
// logs do not capture (they carry nothing past MPICH's own leak report). The
// accept rows this chain reuses (connect_pairwise(), run_pair.sh) are known to
// be intermittent on some hosts; this trace is what lets a host run tell "the
// chain itself deadlocks" apart from "the same accept/connect intermittency,
// now on a 3- or 4-way chain". OFF by default -- the style
// examples/brusselator/participants/lowfive_attach_decision.hpp follows for
// FUSE_LOWFIVE_ATTACH_TRACE: `fuse::transport` cannot depend on that file at
// all (it sits ABOVE this layer), so this is a separate, local instance of the
// identical pattern, not a reuse of it.
[[nodiscard]] inline bool ParseConnectChainTraceEnabledEnv(const char* raw) noexcept {
  return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

[[nodiscard]] bool ChainTraceEnabled() {
  static const bool enabled =
      ParseConnectChainTraceEnabledEnv(std::getenv("FUSE_CONNECT_CHAIN_TRACE"));
  return enabled;
}

// `rank` is THIS step's own local-communicator rank (`current`, not the
// chain's eventual merged rank) -- the one piece of context that tells
// two lines from the SAME step apart (e.g. "rank 0 opened the port" vs
// "rank 0 is still waiting after MPI_Bcast", never printed by a rank
// that is not rank 0 for an accept-side action).
void TraceChainStep(const char* fn, std::size_t step_index, int rank, const char* what) {
  if (!ChainTraceEnabled()) return;
  std::fprintf(stderr,  // NOLINT(fuse-raw-io)
               "FUSE_CONNECT_CHAIN_TRACE: %s step=%zu rank=%d: %s\n", fn, step_index, rank,
               what);
  std::fflush(stderr);
}

// The same discipline mpi_transport.cpp's and comm_graph.cpp's own header
// comments state: a failure that cannot be agreed across ranks (because it
// happens inside, or strictly between, collectives every rank already entered
// together) aborts loudly with a message rather than returning a Result one
// rank's peers never see -- returning instead leaves them in whatever
// COLLECTIVE this communicator's caller performs next, waiting for a rank that
// already left.
[[noreturn]] void abort_job(MPI_Comm comm, const char* what) {
  std::fprintf(stderr,  // NOLINT(fuse-raw-io)
               "FUSE: connect() cannot continue (%s). Aborting rather than "
               "deadlocking the job.\n",
               what);
  std::fflush(stderr);
  MPI_Abort(comm, 1);
  std::abort();
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
    // Every rank already agreed check_mpmd_layout()'s verdict above
    // from identical gathered data, so a per-rank-divergent MPI_Comm_dup
    // outcome here cannot be reconciled by a further reduction -- the
    // ranks that succeeded go on to perform MORE collectives over `dup`
    // (CommGraph::build, MpiTransport::build_plan, ...) that a rank
    // returning an error here will never reach. Abort rather than hand
    // back a Result its peers have no way to learn about.
    abort_job(world, ("MPI_Comm_dup failed: " + mpi_error_text(rc)).c_str());
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
  // The two-component rule itself lives in connect.hpp, pure and MPI-free,
  // so a tier-0 test can fire its refusal directly.
  //
  // connect_mode_components() may return MORE than two components (the N-way
  // chain generalisation) -- this function is ALWAYS the two-party merge of the
  // first two in that order, used standalone for a plain two-component graph
  // and as connect_chain()'s own step 1 for a longer one; connect_chain() never
  // calls this with `my_id` outside {order[0], order[1]}, so looking only at
  // those two here, regardless of how many more the graph declares, is correct.
  const auto order_result = ::fuse::transport::connect_mode_components(g);
  if (!order_result.ok()) return order_result.error();
  const std::vector<ComponentId>& order = order_result.value();
  const ComponentId first_id = order[0];
  const ComponentId second_id = order[1];
  const bool i_am_first = (first_id == my_id);
  const ComponentId partner_id = i_am_first ? second_id : first_id;
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
    // rank 0 alone opens the port and publishes its name; if a failure in
    // either call returned from rank 0 right here, every other acceptor rank
    // would go on into the MPI_Bcast below and block in it forever. So rank 0's
    // outcome is itself broadcast first, so every acceptor rank learns it and
    // returns the same E-CONNECT-MPI together before any rank reaches the
    // port-name Bcast.
    int         open_rc = MPI_SUCCESS;
    std::string open_detail;
    if (rank == 0) {
      rc = MPI_Open_port(MPI_INFO_NULL, port_name);
      if (rc != MPI_SUCCESS) {
        open_rc = rc;
        open_detail = "MPI_Open_port failed: " + mpi_error_text(rc);
      } else if (auto r = publish_port_file(connect_file, port_name); !r.ok()) {
        open_rc = MPI_ERR_OTHER;  // not a real MPI code; the sentinel only needs != MPI_SUCCESS
        open_detail = r.error().message;
      }
    }
    rc = MPI_Bcast(&open_rc, 1, MPI_INT, 0, world);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI",
                     "MPI_Bcast of the port-open outcome failed: " + mpi_error_text(rc)};
    }
    if (open_rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI",
                     rank == 0 ? open_detail
                               : "rank 0 failed to open or publish the acceptor port; its own "
                                 "diagnostic carries the detail"};
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
  // connection itself, and freeing it locally leaves the peer's half in place.
  // With MPI_Comm_free here the acceptor took SIGSEGV inside MPICH's own
  // communicator bookkeeping (MPII_CommL_forget, reading a freed communicator)
  // about a tenth of a second in, long before MPI_Finalize, and the connector
  // then failed dup'ing the merged communicator. Both sides reach this point
  // together, so the collective is matched. The merged communicator survives it
  // and is what the caller gets; it is released separately by disconnect().
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

// Generalises connect_pairwise() above to any number of Connect-mode components
// (>= 2), per connect_chain_plan()'s own rule (connect.hpp): step 1 is
// connect_pairwise() ITSELF, verbatim, between order[0] and order[1]; each
// later step i merges the chain already formed from order[0..i-1] with order[i]
// alone, with order[i] always the acceptor (connect_chain_plan() already
// refused a graph where that is not declared, before this function is ever
// called). For the REAL two-component graph (order.size() == 2) this performs
// EXACTLY one merge -- step 1 -- so behaviour for every existing two-component
// caller is unchanged.
//
// Built, not run: a sandbox that cannot start mpiexec cannot exercise it.
// The host rows this entry names (`parallel.connect.chain.{3,4}`) are not
// registered: the matching N-executable test harness does not exist yet,
// and test/parallel/connect/connect_common.hpp's own
// build_graph()/run_participant() would need to grow a third and fourth
// participant executable.
[[nodiscard]] Result<ConnectedWorld> connect_chain(const CouplingGraph& g,
                                                   ComponentId my_id,
                                                   MPI_Comm world,
                                                   const ConnectOptions& opts) {
  const auto plan_result = connect_chain_plan(g);
  if (!plan_result.ok()) return plan_result.error();
  const std::vector<ChainMergeStep>& steps = plan_result.value();
  const auto order_result = connect_mode_components(g);
  if (!order_result.ok()) return order_result.error();
  const std::vector<ComponentId>& order = order_result.value();

  // Step 1: today's two-component path, unchanged, and ONLY for a
  // process playing order[0] or order[1] -- connect_pairwise() is
  // collective over `world`, and a process playing some LATER component
  // in the chain (order[2+]) has a `world` that is its own separate
  // executable's MPI_COMM_WORLD, sharing no rank with order[0]'s or
  // order[1]'s; calling connect_pairwise() there would mean this
  // process's own world entering a collective its real step-1 peers
  // never reach. For a plain two-component graph this is the only step
  // and the result IS the final merged world; for a longer chain it is
  // the seed every later step grows. A process that is neither order[0]
  // nor order[1] waits for its OWN joining step, below, carrying `world`
  // forward unchanged until then.
  MPI_Comm current = world;
  if (my_id == order[0] || my_id == order[1]) {
    TraceChainStep("connect_chain", 0, -1, "entering step 1 (connect_pairwise)");
    Result<ConnectedWorld> chain_result = connect_pairwise(g, my_id, world, opts);
    if (!chain_result.ok()) return chain_result.error();
    current = MPI_Comm_f2c(static_cast<MPI_Fint>(chain_result.value().base_comm_f));
    TraceChainStep("connect_chain", 0, -1, "step 1 (connect_pairwise) returned");
  }

  for (std::size_t s = 1; s < steps.size(); ++s) {
    const ChainMergeStep& step = steps[s];
    const bool i_am_joining = (my_id == step.joining);
    // Whether THIS process is already part of the chain formed by steps
    // [1 .. step.step_index - 1]: true for order[0..step.step_index-1]'s
    // own ranks, which by connect_chain_plan()'s own numbering is every
    // component strictly before the one joining at this step.
    bool i_am_in_chain_so_far = false;
    for (std::size_t k = 0; k < step.step_index; ++k) {
      if (my_id == order[k]) { i_am_in_chain_so_far = true; break; }
    }
    if (!i_am_joining && !i_am_in_chain_so_far) continue;  // not part of this step

    const ComponentDecl& joining_decl = g.component(step.joining);
    const std::string& connect_file = joining_decl.launch.port_file;

    int rank = 0;
    int rc = MPI_Comm_rank(current, &rank);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI", "MPI_Comm_rank failed: " + mpi_error_text(rc)};
    }
    TraceChainStep("connect_chain", step.step_index, rank,
                   i_am_joining ? "entering step as the JOINING (acceptor) side"
                                : "entering step as the CHAIN-SO-FAR (connector) side");

    MPI_Comm intercomm = MPI_COMM_NULL;
    if (i_am_joining) {
      // The joining component is always the acceptor for its own step
      // (connect_chain_plan()'s own rule) -- identical shape to
      // connect_pairwise()'s own acceptor branch above, just reached for
      // a LATER step's single joining component instead of order[1].
      char port_name[MPI_MAX_PORT_NAME] = {};
      int         open_rc = MPI_SUCCESS;
      std::string open_detail;
      if (rank == 0) {
        rc = MPI_Open_port(MPI_INFO_NULL, port_name);
        if (rc != MPI_SUCCESS) {
          open_rc = rc;
          open_detail = "MPI_Open_port failed: " + mpi_error_text(rc);
        } else if (auto r = publish_port_file(connect_file, port_name); !r.ok()) {
          open_rc = MPI_ERR_OTHER;
          open_detail = r.error().message;
        } else {
          TraceChainStep("connect_chain", step.step_index, rank,
                         ("opened and published port file " + connect_file).c_str());
        }
      }
      rc = MPI_Bcast(&open_rc, 1, MPI_INT, 0, current);
      if (rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI",
                       "MPI_Bcast of the port-open outcome failed: " + mpi_error_text(rc)};
      }
      if (open_rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI",
                       rank == 0 ? open_detail
                                 : "rank 0 failed to open or publish the chain step's "
                                   "acceptor port; its own diagnostic carries the detail"};
      }
      rc = MPI_Bcast(port_name, MPI_MAX_PORT_NAME, MPI_CHAR, 0, current);
      if (rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI", "MPI_Bcast of the port name failed: " + mpi_error_text(rc)};
      }
      TraceChainStep("connect_chain", step.step_index, rank,
                     "calling MPI_Comm_accept -- blocks until the connector side arrives");
      rc = MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, current, &intercomm);
      if (rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI", "MPI_Comm_accept failed: " + mpi_error_text(rc)};
      }
      TraceChainStep("connect_chain", step.step_index, rank, "MPI_Comm_accept returned");
      if (rank == 0) MPI_Close_port(port_name);
    } else {
      // Already part of the chain formed so far: the connector side,
      // waiting on the joining component's own published port.
      TraceChainStep("connect_chain", step.step_index, rank,
                     ("polling for port file " + connect_file).c_str());
      const auto port_result =
          wait_for_port_file(connect_file, opts.timeout_s, opts.poll_interval_s);
      if (!port_result.ok()) return port_result.error();
      TraceChainStep("connect_chain", step.step_index, rank,
                     "found the port file -- calling MPI_Comm_connect");
      rc = MPI_Comm_connect(port_result.value().c_str(), MPI_INFO_NULL, 0, current, &intercomm);
      if (rc != MPI_SUCCESS) {
        return Failure{"E-CONNECT-MPI", "MPI_Comm_connect failed: " + mpi_error_text(rc)};
      }
      TraceChainStep("connect_chain", step.step_index, rank, "MPI_Comm_connect returned");
    }

    // The chain formed so far is always "low" (0) and the joining
    // component is always "high" (1): the chain's own ranks keep their
    // existing numbering and the joining component's ranks are appended
    // after them, preserving the graph's own component order across
    // every step, the same reasoning connect_pairwise() states for its
    // own single merge.
    MPI_Comm merged = MPI_COMM_NULL;
    rc = MPI_Intercomm_merge(intercomm, i_am_joining ? 1 : 0, &merged);
    MPI_Comm_disconnect(&intercomm);
    if (rc != MPI_SUCCESS) {
      return Failure{"E-CONNECT-MPI", "MPI_Intercomm_merge failed: " + mpi_error_text(rc)};
    }
    TraceChainStep("connect_chain", step.step_index, rank, "MPI_Intercomm_merge done for this step");
    // `current` before this step was either `world` (this process's own
    // local world, for a process joining at this step with no prior
    // merge of its own to free) or a PRIOR merged communicator this same
    // function formed and is responsible for; only the latter is this
    // function's own to release here.
    if (current != world) MPI_Comm_free(&current);
    current = merged;
  }

  ConnectedWorld out;
  out.base_comm_f = static_cast<int>(MPI_Comm_c2f(current));
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
      // connect_chain() subsumes the plain two-component case as its own single
      // step (connect_chain_plan()'s own rule, connect.hpp), so every existing
      // two-component caller's behaviour is unchanged -- this is
      // connect_pairwise()'s own body, run once, for order.size() == 2.
      return connect_chain(g, opts.my_component, world, opts);
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
      // The only implemented case (launch.hpp's own doc on this value) is
      // in-process: the ranks a LowFive-launched component runs on come from
      // whatever OTHER launch story this run uses (or none at all), never from
      // this call -- same refusal shape as InProcess/Plugin above. The
      // distributed, cross-executable extension this value's own doc comment
      // names (DistMetadataVOL's own intercommunicator) is explicitly NOT this
      // function's job either way (docs/user/lowfive.md: forming the
      // intercommunicator is explicitly not this transport's job, for the
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
          "FUSE_ENABLE_CONNECT_ACCEPT is OFF. Launch the "
          "components as MPMD segments of one mpiexec instead, which needs "
          "nothing this MPI lacks, or build against an MPI whose dynamic "
          "process management works between same-node jobs -- for MPICH that "
          "means --with-device=ch4."};
}

}  // namespace fuse::transport

