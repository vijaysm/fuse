/* tools/probe-mpi-dynamic.c -- can this MPI connect two separately launched
 * programs on one node?
 *
 * The accept/connect launch mode (docs/user/multiple-executables.md) needs
 * three things to work, and an MPI may provide the first two without the
 * third:
 *
 *   1. dynamic process management at all -- MPI_Open_port, MPI_Comm_accept,
 *      MPI_Comm_connect. Optional in the standard, and some builds omit it.
 *   2. a merge of the resulting intercommunicator into one intracommunicator,
 *      which is how a coupled run gets a single base communicator with one
 *      rank numbering across every program.
 *   3. both of those between two jobs that happen to share a NODE, AT EVERY
 *      RANK COUNT a registered row actually launches -- not just one rank a
 *      side. KNOWN_ISSUES.md D23's final account: a device whose context-id
 *      agreement for the merged communicator is intermittent under load (not
 *      the ch3:nemesis defect this file's comment originally described, which
 *      cleared with a device change) can pass a 1+1 run reliably while the
 *      3+2/2+3/chain layouts the real rows use still time out.
 *
 * So this probe does the real thing rather than reading a version string: it
 * is launched twice, as two separate jobs on this machine -- at whatever rank
 * counts the caller gives each side (tools/probe-mpi-dynamic.sh's own
 * <na>+<nb> argument), repeated several times -- and performs the exact
 * sequence src/transport/connect.cpp performs. Nothing is inferred from the
 * implementation's name.
 *
 * tools/probe-intercomm-merge.c is a SEPARATE, FUSE-free program with the
 * same shape (accept/connect over a port file, then MPI_Intercomm_merge),
 * kept for its own reason: it is the exact reproducer KNOWN_ISSUES.md D23
 * names for the ad hoc multi-rank measurements recorded there, run and
 * re-run by hand outside any build, with no cmake, no shell driver, and no
 * dependence on this file's own CLI or exit-code conventions. This file is
 * the one cmake/FuseDependencies.cmake actually compiles and launches at
 * configure time (via tools/probe-mpi-dynamic.sh); that one is a manual
 * diagnostic tool whose own measurements this file's own multi-layout
 * probing was built to stop needing. They are not interchangeable: this one
 * distinguishes "could not test" from "measured broken" (the `selftest`
 * mode below) and is driven entirely by command-line argument and exit
 * code, which the configure-time caller depends on; the other takes its
 * acceptor/connector role as an explicit argument and reports only to
 * stderr, which is what a human re-running it by hand wants.
 *
 * Usage: probe-mpi-dynamic {accept|connect} <port-file> | probe-mpi-dynamic selftest
 * Exit:  0 the sequence completed; non-zero or a signal, it did not.
 */

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { kPortWaitSeconds = 30 };

int main(int argc, char** argv) {
  /* `selftest` takes no port file; the other two require one. Getting this
   * wrong made the probe report "cannot launch an MPI process here" on a
   * machine whose launcher was fine (host, 2026-10-09) -- the preflight was
   * failing on its own usage message, which is exactly the confusion between
   * "could not test" and "measured" that the three verdicts exist to avoid. */
  const int selftest = (argc == 2 && strcmp(argv[1], "selftest") == 0);
  if (!selftest && argc != 3) {
    fprintf(stderr, "usage: %s {accept|connect} <port-file> | %s selftest\n",
            argv[0], argv[0]);
    return 2;
  }
  const char* mode = argv[1];
  const char* path = selftest ? "" : argv[2];

  MPI_Init(&argc, &argv);
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char     port[MPI_MAX_PORT_NAME];
  MPI_Comm inter = MPI_COMM_NULL;
  memset(port, 0, sizeof(port));

  if (strcmp(mode, "accept") == 0) {
    if (rank == 0) {
      if (MPI_Open_port(MPI_INFO_NULL, port) != MPI_SUCCESS) {
        fprintf(stderr, "probe: MPI_Open_port failed\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      /* Write to a temporary and rename, so the connector never reads a
       * half-written name -- the same discipline connect.cpp uses. */
      char tmp[4096];
      snprintf(tmp, sizeof(tmp), "%s.partial", path);
      FILE* f = fopen(tmp, "w");
      if (f == NULL) {
        fprintf(stderr, "probe: cannot write '%s'\n", tmp);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fprintf(f, "%s", port);
      fclose(f);
      if (rename(tmp, path) != 0) {
        fprintf(stderr, "probe: cannot rename to '%s'\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }
    if (MPI_Comm_accept(port, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &inter) !=
        MPI_SUCCESS) {
      fprintf(stderr, "probe: MPI_Comm_accept failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } else if (strcmp(mode, "connect") == 0) {
    if (rank == 0) {
      FILE* f = NULL;
      for (int waited = 0; waited < kPortWaitSeconds; ++waited) {
        f = fopen(path, "r");
        if (f != NULL) break;
        sleep(1);
      }
      if (f == NULL) {
        fprintf(stderr, "probe: no port file at '%s'\n", path);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      if (fgets(port, (int)sizeof(port), f) == NULL) {
        fprintf(stderr, "probe: port file '%s' is empty\n", path);
        fclose(f);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
      fclose(f);
    }
    MPI_Bcast(port, MPI_MAX_PORT_NAME, MPI_CHAR, 0, MPI_COMM_WORLD);
    if (MPI_Comm_connect(port, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &inter) !=
        MPI_SUCCESS) {
      fprintf(stderr, "probe: MPI_Comm_connect failed\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  } else if (strcmp(mode, "selftest") == 0) {
    /* MPI_Init and MPI_Finalize and nothing else: the driver runs this
     * first, so that a launcher which cannot start ANY process here is
     * reported as "could not test" rather than as a missing capability.
     * The two verdicts call for opposite actions, and a probe that
     * confuses them is worse than no probe. */
    MPI_Finalize();
    if (rank == 0) printf("probe selftest: ok\n");
    return 0;
  } else {
    fprintf(stderr, "probe: unknown mode '%s'\n", mode);
    MPI_Finalize();
    return 2;
  }

  /* THE STEP THAT FAILS on a device that cannot do this: one intracommunicator
   * spanning both jobs. A probe that stopped at the intercommunicator would
   * report success on an installation where a coupled run still cannot start. */
  MPI_Comm merged = MPI_COMM_NULL;
  const int high  = (strcmp(mode, "accept") == 0) ? 0 : 1;
  if (MPI_Intercomm_merge(inter, high, &merged) != MPI_SUCCESS) {
    fprintf(stderr, "probe: MPI_Intercomm_merge failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* And a collective over it, since merging can succeed while the merged
   * communicator is unusable. */
  int merged_size = 0;
  int one         = 1;
  int total       = 0;
  MPI_Comm_size(merged, &merged_size);
  if (MPI_Allreduce(&one, &total, 1, MPI_INT, MPI_SUM, merged) != MPI_SUCCESS ||
      total != merged_size) {
    fprintf(stderr, "probe: collective over the merged communicator failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Comm_disconnect(&inter);
  MPI_Comm_disconnect(&merged);
  if (strcmp(mode, "accept") == 0 && rank == 0) MPI_Close_port(port);
  MPI_Finalize();
  if (rank == 0) printf("probe %s: ok, merged size %d\n", mode, merged_size);
  return 0;
}