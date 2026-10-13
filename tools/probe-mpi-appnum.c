/* tools/probe-mpi-appnum.c -- does this MPI report a distinct MPI_APPNUM per
 * segment of an MPMD (colon-syntax) launch?
 *
 * fuse::transport::check_mpmd_layout() (include/fuse/transport/connect.hpp)
 * validates an "mpmd" multi-executable launch (docs/user/
 * multiple-executables.md) by comparing each component's declared RankSet
 * against the ranks that actually report its MPI_APPNUM. That only works if
 * MPI_APPNUM differs per segment of the launch -- the standard requires it be
 * SET when the launcher provides one (MPI-3.1 11.9.2), but does not require
 * launchers to differentiate it, and this host's Open MPI (4.1.5) measured to
 * report appnum 0 for every rank in every segment of both a colon-syntax and
 * an --app-file launch (KNOWN_ISSUES.md D41).
 *
 * This probe does the real thing: it is launched as two colon-syntax
 * segments, each told (via argv) which appnum it should see, and reports
 * whether it actually did.
 *
 * Usage: probe-mpi-appnum <expected-appnum>
 * Exit:  0 this rank's MPI_APPNUM equals <expected-appnum>.
 *        1 MPI_APPNUM is unset or does not match -- measured broken.
 *        2 usage/MPI_Init-level failure, unrelated to the capability.
 */

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <expected-appnum>\n", argv[0]);
    return 2;
  }
  const int expected = atoi(argv[1]);

  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  void* appnum_ptr = NULL;
  int   flag       = 0;
  const int rc = MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_APPNUM, &appnum_ptr, &flag);

  int result = 0;
  if (rc != MPI_SUCCESS || !flag) {
    fprintf(stderr, "probe[%d]: MPI_APPNUM not set (rc=%d flag=%d)\n", rank, rc, flag);
    result = 1;
  } else {
    const int appnum = *(int*)appnum_ptr;
    if (appnum != expected) {
      fprintf(stderr, "probe[%d]: MPI_APPNUM=%d, expected %d\n", rank, appnum, expected);
      result = 1;
    } else {
      printf("probe[%d]: MPI_APPNUM=%d as expected\n", rank, appnum);
    }
  }

  MPI_Finalize();
  return result;
}
