#pragma once

// Demo B's OWN minimal two-process rendezvous: MPI_Open_port/Comm_accept on
// the producer side, MPI_Comm_connect on the consumer side, exchanging the
// port name through a file both processes can see (a shared filesystem path,
// passed on each executable's command line).
//
// DELIBERATELY NOT fuse/transport/connect.cpp. That file forms the single,
// graph-aware base communicator the WHOLE coupled run's rank numbering
// depends on (merging via MPI_Intercomm_merge in the graph's own component
// order, per its own header comment) -- considerably more than a
// self-contained two-process demo needs. Whether Demo B's rendezvous should
// eventually BECOME a `connect()` call instead of this local one is an open
// question; see docs/user/lowfive.md.
//
// Exercised by the ctest row example.demo_b; see docs/user/demo-b.md.

#include <mpi.h>

#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace demo_b {

/// Producer side. Opens a port, writes its name to `port_file`, blocks in
/// MPI_Comm_accept (a real, synchronous MPI call -- not a window either
/// side can time out around, same caveat fuse/transport/connect.hpp's own
/// ConnectOptions::timeout_s documents for the connecting side only).
/// Returns the formed intercommunicator. Throws std::runtime_error on any
/// MPI failure.
inline MPI_Comm accept_as_producer(const std::string& port_file) {
  char port_name[MPI_MAX_PORT_NAME] = {0};
  if (MPI_Open_port(MPI_INFO_NULL, port_name) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Open_port failed");
  }
  {
    std::ofstream out(port_file, std::ios::trunc);
    if (!out) {
      throw std::runtime_error("cannot write port file '" + port_file + "'");
    }
    out << port_name;
  }
  MPI_Comm intercomm = MPI_COMM_NULL;
  if (MPI_Comm_accept(port_name, MPI_INFO_NULL, 0, MPI_COMM_SELF,
                      &intercomm) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_accept failed");
  }
  return intercomm;
}

/// Consumer side. Polls for `port_file` to appear (the producer may not
/// have reached accept_as_producer() yet), reads the port name, and calls
/// MPI_Comm_connect. Throws std::runtime_error on a timeout or any MPI
/// failure.
inline MPI_Comm connect_as_consumer(const std::string& port_file,
                                    double timeout_s = 30.0,
                                    double poll_interval_s = 0.05) {
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_s);
  std::string port_name;
  for (;;) {
    std::ifstream in(port_file);
    if (in) {
      std::ostringstream buf;
      buf << in.rdbuf();
      port_name = buf.str();
      if (!port_name.empty()) break;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      throw std::runtime_error("timed out waiting for port file '" +
                               port_file + "'");
    }
    std::this_thread::sleep_for(
        std::chrono::duration<double>(poll_interval_s));
  }
  MPI_Comm intercomm = MPI_COMM_NULL;
  if (MPI_Comm_connect(port_name.c_str(), MPI_INFO_NULL, 0, MPI_COMM_SELF,
                       &intercomm) != MPI_SUCCESS) {
    throw std::runtime_error("MPI_Comm_connect failed");
  }
  return intercomm;
}

}  // namespace demo_b
