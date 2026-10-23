/* Time windows track (Conformance, Unit 2): the coupler's OWN MPMD app
 * segment. PURE C. Only needed in mpmd mode -- in accept mode the
 * coupler's ranks are bundled into U's own executable instead (Q1's
 * ruling, KNOWN_ISSUES.md D18; u_main.c's own header comment has the
 * full account). A ComponentKind::Coupler placement owns no fields, has
 * no clock, and is never advanced (docs/dev/coupler-placement.md's own
 * enumerated call sites), so this program's own job is almost nothing:
 * create the context, load the SAME graph u_main.c/v_main.f90 load,
 * call fuse_setup() with my_component="coupler" (its own LaunchSpec is
 * Mpmd here, with its own appnum -- brusselator_graph_tool.cpp's own
 * mpmd branch), call fuse_initialize() (collective -- every rank of the
 * compiled graph's base communicator participates, including this one),
 * then fuse_run() for the SAME --periods U's and V's own executables
 * step, then destroy. It registers no field and never calls
 * fuse_begin_step/fuse_end_step at all: there is no step for a coupler
 * to take, but it must still execute the period's own collectives
 * (Unit 2b, D19, step (d)) -- fuse_run() is legal here because this
 * rank hosts no externally driven component at all (Runner::hosts_
 * component() says so; fuse_run()'s own E-ABI-RUN-EXTERNAL refusal only
 * fires when one IS hosted on THIS process).
 *
 * BEFORE Unit 2b step (d): this rank called neither fuse_run() nor
 * fuse_begin_step()/fuse_end_step() at all between fuse_initialize()
 * and fuse_destroy() -- the Runner's own internal Transport/Remap
 * collectives routed through this rank during U's and V's own
 * begin_step/end_step calls had no partner here, so U's and V's own
 * ranks would block forever in the first Transport that needed this
 * rank's participation (KNOWN_ISSUES.md D19, evidence 4). fuse_run()
 * is the fix: it executes exactly those collectives, with nothing else
 * to drive.
 *
 * NOT YET HOST-VERIFIED (MPI cannot start in this sandbox,
 * KNOWN_ISSUES.md C1): the mechanism is argued (fuse_run() executes
 * the period's own instruction stream, which is where the coupler's
 * own Transport/Remap participation lives) and build-verified only;
 * confirming it actually unblocks U's/V's own ranks under real MPI is
 * verification.window.mpi.{2,3}'s/the Brusselator host rows' own job.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuse/abi/participant.h"
#include "lowfive_attach.h"

static void fail(fuse_ctx* ctx, const char* what, fuse_status rc) {
  char buf[2048];
  fuse_error err;
  memset(&err, 0, sizeof(err));
  err.struct_size = sizeof(err);
  err.buffer = buf;
  err.capacity = sizeof(buf);
  buf[0] = '\0';
  if (ctx) {
    const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
    if (t) t->last_error(ctx, NULL, &err);
  }
  fprintf(stderr, "FAIL(coupler_main): %s: status=%d %s\n", what, (int)rc, buf);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char** argv) {
  const char* graph_path = NULL;
  int periods = 2;
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
      graph_path = argv[++i];
    } else if (strcmp(argv[i], "--periods") == 0 && i + 1 < argc) {
      periods = atoi(argv[++i]);
    } else {
      fprintf(stderr, "coupler_main: unrecognized argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (!graph_path) {
    fprintf(stderr, "usage: coupler_main --graph graph.yaml [--periods N]\n");
    return 1;
  }

  /* ROUND 9, task 1(a): this process never calls bruss_lowfive_attach()
   * (D14 Stage B: lowfive_attach.cpp IS now one of this target's own
   * sources -- examples/CMakeLists.txt -- but ONLY for
   * bruss_lowfive_join_local_split(), the collective split every rank
   * of the bridge must join; this rank never calls bruss_lowfive_
   * attach() itself, the ONE function that constructs a real LowFive
   * VOL object), so it has no deliberate use for a LowFive VOL
   * connector at all. The THREE env-route variables named here are
   * nonetheless still live in this process's own environment --
   * examples/CMakeLists.txt's own ctest ENVIRONMENT property sets them
   * on the WHOLE MPMD launch (run_participants.sh's own single
   * mpiexec invocation, one shared environment across all three
   * ':'-separated app segments), not per-segment -- and HDF5's own
   * env-route default-VOL resolution would otherwise register the
   * "lowfive" connector class here too, with nothing that ever wanted
   * it, the measured cause of this process's own exit-time SIGABRT
   * (lowfive_attach_decision.hpp's own "ROUND 9, task 1" comment has
   * the full crash trace, root cause, and the ranked alternatives this
   * fix was chosen over). Removing these before MPI_Init, before any
   * native HDF5 call this process might make, prevents the connector
   * from EVER registering here -- the authoritative list of which
   * three variables is fuse_examples::lowfive_attach::
   * EnvRouteVolVariableNames() (lowfive_attach_decision.hpp); kept in
   * sync by hand, since this file is plain C with no dependency on
   * that C++ header (duplicating three literal names is smaller and
   * less coupling than adding a whole new C++ translation unit and
   * CMake target just to share them). */
  {
    unsetenv("HDF5_VOL_CONNECTOR");
    unsetenv("HDF5_PLUGIN_PATH");
    unsetenv("FUSE_LOWFIVE_BINDINGS");
    const char* trace = getenv("FUSE_LOWFIVE_ATTACH_TRACE");
    if (trace && trace[0] == '1' && trace[1] == '\0') {
      fprintf(stderr,
              "TRACE(coupler_main): unregistered env-route VOL vars "
              "(HDF5_VOL_CONNECTOR, HDF5_PLUGIN_PATH, FUSE_LOWFIVE_BINDINGS) "
              "before MPI_Init (role=kNonAttachingCoordinator)\n");
    }
  }

  MPI_Init(&argc, &argv);

  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) {
    fprintf(stderr, "FAIL(coupler_main): fuse_coupler_abi() returned NULL\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  /* Physics track: COLLECTIVE over MPI_COMM_WORLD, matching u_main.c's
   * and v_main.f90's own MPI_Comm_split -- MPI_COMM_WORLD here is the
   * SHARED world every ':'-segment of this mpmd mpiexec invocation
   * numbers into (coupler_main is MPMD-only, this file's own header
   * comment), so each program's own split is a collective over every
   * rank of every segment, not just its own; a coupler rank that never
   * called a matching split would strand the other two programs' own
   * ranks inside theirs (this track's own standing lesson: a collective
   * must never be skipped).
   *
   * COLOR = COMPONENT INDEX (U 0, V 1, coupler 2), key = world rank --
   * ONE uniform split shape across all three programs, each naming its
   * own fixed color, rather than the earlier "U's own color plus
   * everyone else opts out with MPI_UNDEFINED" form: that form put the
   * ENTIRE burden of "did every non-U program remember to opt out at the
   * right point" on files that have nothing else to do with U's own
   * split, which is exactly the class of mismatch this choreography
   * already found once (see this comment's own ORDER MATTERS paragraph
   * below) -- naming a REAL, fixed color here is no harder to get right
   * than MPI_UNDEFINED was, and reads the same in all three files. The
   * resulting communicator is still never used by this program (the
   * coupler has no physics of its own to decompose), but it is a real,
   * valid one now, not MPI_COMM_NULL -- freed unconditionally below.
   *
   * ORDER MATTERS: this split comes right after MPI_Init and BEFORE
   * load_graph/setup, where u_main.c and v_main.f90 place theirs.
   * setup() runs connect(), itself a collective over the same world
   * (connect_mpmd's MPI_Allgather of one int per rank); the first host
   * run of this choreography (2026-10-10) had the coupler split AFTER
   * setup, so the coupler's 4-byte Allgather met U's 8-byte (color,
   * key) exchange from inside MPI_Comm_split: "Message from rank 0 and
   * tag 7 truncated; 8 bytes received but buffer size is 4". Every
   * program must enter the same collectives in the same order. */
  MPI_Comm coupler_comm = MPI_COMM_NULL;
  {
    int world_rank_for_split = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_for_split);
    MPI_Comm_split(MPI_COMM_WORLD, /*color=*/2, world_rank_for_split, &coupler_comm);
  }


  fuse_context_out co;
  memset(&co, 0, sizeof(co));
  co.struct_size = sizeof(co);
  fuse_status rc = t->create(NULL, NULL, &co);
  if (rc != FUSE_OK) fail(NULL, "create", rc);
  fuse_ctx* ctx = co.ctx;

  fuse_graph_source src;
  memset(&src, 0, sizeof(src));
  src.struct_size = sizeof(src);
  src.yaml_path = graph_path;
  rc = t->load_graph(ctx, &src, NULL);
  if (rc != FUSE_OK) fail(ctx, "load_graph", rc);

  fuse_setup_options setup;
  memset(&setup, 0, sizeof(setup));
  setup.struct_size = sizeof(setup);
  setup.communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
  /* KNOWN_ISSUES.md D76: MPI_Comm_c2f(MPI_COMM_WORLD) is 0 under Open MPI,
   * which this struct's older sentinel read as "no communicator" -- set
   * has_communicator so fuse_setup() knows 0 really is this process's own
   * world, not an absent one. */
  setup.has_communicator = 1;
  setup.my_component = "coupler";
  rc = t->setup(ctx, &setup, NULL);
  if (rc != FUSE_OK) fail(ctx, "setup", rc);

  /* D14 Stage B (KNOWN_ISSUES.md), decision 2/3: this rank joins the
   * SAME collective split U's and V's own physics ranks join
   * (bruss_lowfive_join_local_split()'s own header comment in
   * lowfive_attach.h has the full account) -- MPI_Comm_split() is
   * collective over the WHOLE bridge communicator, and this rank is
   * part of it (it shares MPI_COMM_WORLD with U and V in this track's
   * own mpmd-only placement -- this file's own header comment), even
   * though a Coupler-kind component owns no field and is never the
   * writer or reader of any LowFive-bound channel, so its own color is
   * always MPI_UNDEFINED and local_comm_f always comes back 0. This
   * process still never calls bruss_lowfive_attach() itself (round 9's
   * own ruling, unchanged) -- only the collective every rank of the
   * bridge must join, never the VOL registration itself. Safe and cheap
   * even for an ordinary, non-LowFive row: no MPI call at all when the
   * graph carries no hdf5_path field (bruss_lowfive_join_local_split()'s
   * own doc comment has the full account). */
  {
    long long local_comm_f = 0;
    char split_errbuf[2048];
    const int split_rc = bruss_lowfive_join_local_split(
        graph_path, (long long)MPI_Comm_c2f(MPI_COMM_WORLD), &local_comm_f,
        split_errbuf, (long long)sizeof(split_errbuf));
    if (split_rc != 0) {
      fprintf(stderr, "FAIL(coupler_main): bruss_lowfive_join_local_split: rc=%d %s\n",
              split_rc, split_errbuf);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    (void)local_comm_f; /* always 0 on this rank; nothing more to do with it */
  }

  rc = t->initialize(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(ctx, "initialize", rc);

  /* No field, no step -- a Coupler component is never advanced. But the
   * period's own collectives (the Transport/Remap routed through this
   * rank) still need executing, the SAME --periods U's/V's own
   * executables step -- Unit 2b (D19), step (d)'s own fix. */
  {
    fuse_run_request req;
    memset(&req, 0, sizeof(req));
    req.struct_size = sizeof(req);
    req.periods = periods;
    rc = t->run(ctx, &req, NULL);
    if (rc != FUSE_OK) fail(ctx, "run", rc);
  }

  rc = t->destroy(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(NULL, "destroy", rc);

  /* D14 Stage B: harmless no-op on this rank (bruss_lowfive_attach() is
   * never called here, so there is nothing for this to flush or free --
   * lowfive_attach.h's own doc comment states the "regardless of
   * whether bruss_lowfive_attach() was ever called" contract plainly)
   * -- called anyway, for the same uniform hygiene u_main.c/v_main.f90
   * already apply, before tearing down MPI. */
  {
    char finalize_errbuf[2048];
    const int finalize_rc =
        bruss_lowfive_finalize(finalize_errbuf, (long long)sizeof(finalize_errbuf));
    if (finalize_rc != 0) {
      fprintf(stderr, "FAIL(coupler_main): bruss_lowfive_finalize: rc=%d %s\n", finalize_rc,
              finalize_errbuf);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Comm_free(&coupler_comm);
  MPI_Finalize();
  return 0;
}
