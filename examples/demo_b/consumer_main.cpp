// Demo B consumer driver (S6 phase 3, "LowFive" track). See
// producer_main.cpp's own comment -- the same reasoning applies here.
//
// Run by the ctest row example.demo_b; see producer_main.cpp.
//
// Usage: demo_b_consumer <port_file> <hdf5_file> <dataset_name>
//
// Exit code is the load-bearing assertion, not a printed message: 0 iff (a)
// the read-back values match the producer's bit for bit, AND (b) `stat()`
// on <hdf5_file> fails (no real file reached disk).

#include "consumer_hdf5.h"
#include "rendezvous.hpp"

#include <fuse/transport/transport.hpp>

#include <mpi.h>
#include <sys/stat.h>

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

  // The producer's own values (producer_main.cpp) -- known on both sides,
  // see that file's comment for why Demo B needs no separate channel for
  // this.
  const std::vector<double> expected{1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0};

  MPI_Init(&argc, &argv);

  int status = 0;
  try {
    MPI_Comm intercomm =
        demo_b::connect_as_consumer(port_file, /*timeout_s=*/60.0);

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
          fuse::TransportSpec{.elements = expected.size()});
      if (!r.ok()) {
        std::fprintf(stderr, "build_plan: [%s] %s\n", r.error().code.c_str(),
                     r.error().message.c_str());
        status = 1;
      }
    }
    if (status == 0) {
      r = transport.receive_as_consumer();
      if (!r.ok()) {
        std::fprintf(stderr, "receive_as_consumer: [%s] %s\n",
                     r.error().code.c_str(), r.error().message.c_str());
        status = 1;
      }
    }
    std::vector<double> got(expected.size(), 0.0);
    if (status == 0) {
      // The unmodified payload: it has no idea this property list carries
      // an interception VOL connector.
      const int rc = demo_b_consume(hdf5_file.c_str(), dataset.c_str(),
                                    transport.access_plist_handle(),
                                    got.data(), static_cast<long>(got.size()));
      if (rc != 0) {
        std::fprintf(stderr, "demo_b_consume failed, rc=%d\n", rc);
        status = 1;
      }
    }
    if (status == 0 && got != expected) {
      std::fprintf(stderr,
                   "demo_b_consumer: read-back values do not match the "
                   "producer's bit for bit\n");
      status = 1;
    }
    struct ::stat st {};
    const bool on_disk = ::stat(hdf5_file.c_str(), &st) == 0;
    if (on_disk) {
      std::fprintf(stderr,
                   "demo_b_consumer: '%s' exists on disk -- interception "
                   "failed and this run degraded to real file I/O\n",
                   hdf5_file.c_str());
      status = 1;
    }
    if (status == 0) {
      std::printf("demo_b_consumer: read %zu values, bit-for-bit match, "
                  "zero bytes of HDF5 on disk\n",
                  got.size());
    }
    MPI_Comm_disconnect(&intercomm);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "demo_b_consumer: %s\n", e.what());
    status = 1;
  }

  MPI_Finalize();
  return status;
}
