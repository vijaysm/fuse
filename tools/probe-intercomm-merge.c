/* Minimal reproducer for the accept/connect/merge intermittency recorded in
 * KNOWN_ISSUES.md D23 (measured 2026-10-19 on the owner's machine, MPICH
 * 5.0.0a1, FI_PROVIDER="tcp;ofi_rxm" FI_TCP_IFACE=en0). No FUSE code: one
 * MPI_Comm_accept / MPI_Comm_connect pair over a port file, then
 * MPI_Intercomm_merge. Build: mpicc -O0 -g probe-intercomm-merge.c -o merge.
 * Run: (mpiexec -n NA ./merge accept port.txt 0 &); mpiexec -n NB ./merge
 * connect port.txt 1. Every rank prints "formed" after the rendezvous and
 * "merged rank" after the merge; a run that prints neither on some rank hung
 * in connect/accept, one that prints "formed" everywhere but not "merged
 * rank" hung in the merge. Measured rates (OK of 6): 1+1: 5, 2+1: 3, 2+3: 2,
 * 3+2: 1. With the default libfabric provider connect never completes.
 *
 * KNOWN_ISSUES.md D23's final account: tools/probe-mpi-dynamic.c (the
 * program cmake/FuseDependencies.cmake actually compiles and launches at
 * configure time, via tools/probe-mpi-dynamic.sh) now repeats the SAME
 * accept/connect/merge sequence at every rank layout a registered row
 * uses, several times each, so the ad hoc by-hand measurements this file
 * was built for are no longer the only record of the intermittency -- the
 * configure-time probe's own per-layout record (FUSE_MPI_DYNAMIC_LAYOUTS)
 * is. This file stays: it is the exact reproducer the measurements already
 * written into this entry name, with no cmake, no shell driver, and no
 * dependence on the other file's CLI or exit-code conventions -- a human
 * re-running ONE specific layout by hand, with a debugger attached, wants
 * exactly this, not a six-layout probe that also decides a cmake cache
 * entry. They are not merged into one program: this one takes its
 * acceptor/connector role as an explicit argument (the `high` parameter)
 * and never distinguishes "could not test" from "measured broken", which
 * is fine for a by-hand tool and wrong for a configure-time one. */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  if (argc < 4) {
    fprintf(stderr, "usage: %s accept|connect <portfile> <high 0|1>\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int rank = 0, size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const char* role = argv[1];
  const char* portfile = argv[2];
  const int high = atoi(argv[3]);
  char port[MPI_MAX_PORT_NAME] = {0};
  MPI_Comm inter = MPI_COMM_NULL, merged = MPI_COMM_NULL;
  if (strcmp(role, "accept") == 0) {
    if (rank == 0) {
      MPI_Open_port(MPI_INFO_NULL, port);
      FILE* f = fopen(portfile, "w");
      fprintf(f, "%s\n", port);
      fclose(f);
    }
    MPI_Comm_accept(port, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &inter);
  } else {
    for (;;) {
      FILE* f = fopen(portfile, "r");
      if (f != NULL) {
        if (fgets(port, (int)sizeof port, f) != NULL) port[strcspn(port, "\n")] = 0;
        fclose(f);
        if (port[0] != 0) break;
      }
      usleep(50000);
    }
    MPI_Comm_connect(port, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &inter);
  }
  fprintf(stderr, "%s rank %d/%d: formed, merging high=%d\n", role, rank, size, high);
  MPI_Intercomm_merge(inter, high, &merged);
  int mr = 0, ms = 0;
  MPI_Comm_rank(merged, &mr);
  MPI_Comm_size(merged, &ms);
  fprintf(stderr, "%s rank %d/%d: merged rank %d of %d\n", role, rank, size, mr, ms);
  MPI_Comm_free(&merged);
  MPI_Comm_disconnect(&inter);
  MPI_Finalize();
  return 0;
}
