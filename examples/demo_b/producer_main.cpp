// Demo B producer driver (S6 phase 3, "LowFive" track). This file, unlike
// producer_hdf5.cpp, IS allowed to mention the framework -- it is the FUSE
// side of the split described in docs/user/lowfive.md: attach the
// intercepting connector, then call into the unmodified HDF5 payload.
//
// UNEXECUTED: MPI cannot start in the sandbox this was written in. Compiles
// against real MOAB/LowFive/MPI headers on this host (2026-10-02); has
// never been run. See docs/user/lowfive.md and this track's report.
//
// Usage: demo_b_producer <port_file> <hdf5_file> <dataset_name>

#include "producer_hdf5.h"
#include "rendezvous.hpp"

#include <fuse/transport/transport.hpp>

#include <mpi.h>

#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr,
                 "usage: %s <port_file> <hdf5_file> <dataset_name>\n",
                 argv[0]);
    return 2;
  }
  const std::string port_file = argv[1];
  const std::string hdf5_file = argv[2];
  const std::string dataset = argv[3];

  MPI_Init(&argc, &argv);

  int status = 0;
  try {
    MPI_Comm intercomm = demo_b::accept_as_producer(port_file);

    // The same fixed, deterministic values the consumer checks against
    // (consumer_main.cpp) -- Demo B's dataset is small and known on both
    // sides so the demo needs no separate "expected values" channel.
    const std::vector<double> data{1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0};

    fuse::LowFiveTransport transport;
    auto r = transport.set_intercomm(MPI_Comm_c2f(MPI_COMM_SELF),
                                     MPI_Comm_c2f(intercomm));
    if (!r.ok()) {
      std::fprintf(stderr, "set_intercomm: [%s] %s\n", r.error().code.c_str(),
                   r.error().message.c_str());
      status = 1;
    }
    if (status == 0) {
      r = transport.set_location(hdf5_file, dataset);
      if (!r.ok()) {
        std::fprintf(stderr, "set_location: [%s] %s\n",
                     r.error().code.c_str(), r.error().message.c_str());
        status = 1;
      }
    }
    if (status == 0) {
      r = transport.build_plan(
          fuse::TransportSpec{.elements = data.size()});
      if (!r.ok()) {
        std::fprintf(stderr, "build_plan: [%s] %s\n", r.error().code.c_str(),
                     r.error().message.c_str());
        status = 1;
      }
    }
    if (status == 0) {
      // The unmodified payload: it has no idea this property list carries
      // an interception VOL connector.
      const int rc = demo_b_produce(hdf5_file.c_str(), dataset.c_str(),
                                    transport.access_plist_handle(),
                                    data.data(),
                                    static_cast<long>(data.size()));
      if (rc != 0) {
        std::fprintf(stderr, "demo_b_produce failed, rc=%d\n", rc);
        status = 1;
      }
    }
    if (status == 0) {
      r = transport.serve_as_producer();
      if (!r.ok()) {
        std::fprintf(stderr, "serve_as_producer: [%s] %s\n",
                     r.error().code.c_str(), r.error().message.c_str());
        status = 1;
      }
    }
    if (status == 0) {
      std::printf("demo_b_producer: wrote %zu values, served across the "
                  "intercommunicator\n",
                  data.size());
    }
    MPI_Comm_disconnect(&intercomm);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "demo_b_producer: %s\n", e.what());
    status = 1;
  }

  MPI_Finalize();
  return status;
}
