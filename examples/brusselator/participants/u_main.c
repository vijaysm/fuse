/* Time windows track (Conformance, Unit 2): the C participant for
 * component "U". PURE C -- no C++ header, no C++ call, anywhere in this
 * file (the lead's own constraint: "a foreign code never touches C++").
 * Links fuse::abi (fuse/abi/participant.h, the one-exported-symbol
 * table), the shim's own C-ABI physics wrapper (shim.h/shim.cpp -- a
 * C++ TRANSLATION UNIT, but its own header is extern "C", so this file
 * never sees a C++ type) and loop_step.h (the SAME u_loop_step() the
 * tier-0 harness, test/verification/brusselator_participants_test.cpp,
 * drives).
 *
 * GRAPH AUTHORING vs PARTICIPATING (Q2): this program never builds a
 * fuse::CouplingGraph. It calls fuse_load_graph() on a YAML path a
 * SEPARATE tool, brusselator_graph_tool, already wrote (--graph on the
 * command line), and nothing else constructs the graph.
 *
 * LAUNCH MODE FROM THE GRAPH, NOT FROM A FLAG (the brief's own "LaunchSpec
 * from the YAML, connect() per the YAML's launch mode"): this program
 * always passes `communicator = MPI_Comm_c2f(MPI_COMM_WORLD)` and
 * `my_component = "U"` to fuse_setup(); fuse_setup() itself resolves "U"'s
 * LaunchSpec from the loaded graph and only calls fuse::transport::
 * connect() when that LaunchSpec says to (fuse_setup_options::my_component
 *'s own comment in fuse/abi/types.h has the full account) -- this
 * executable's own code is IDENTICAL whether the graph says mpmd, accept,
 * or (for a tier-0 smoke run with no MPI at all) in-process.
 *
 * BUNDLED COUPLER RANKS (accept mode only, KNOWN_ISSUES.md D18 / Q1):
 * connect_pairwise() supports exactly two Connect-mode components, so in
 * accept mode the coupler's ranks are NOT a separate executable -- they
 * are EXTRA ranks of this same executable's own MPI_COMM_WORLD, declared
 * in the graph as an overlapping subset of U's own RankSet
 * (brusselator_graph_tool.cpp's own header comment has the full
 * rationale). `--ranks-u N` tells this binary how many of ITS OWN local
 * ranks (0..N-1) are real U-physics ranks; any remaining local ranks
 * (N..world_size-1, present only in accept mode with a nonzero
 * --ranks-coupler) are coupler placeholder ranks: they call fuse_setup()
 * identically (the call is collective over this executable's WHOLE local
 * world, exactly matching what MPI_Comm_accept/MPI_Intercomm_merge
 * requires of every rank of the local group passed to them) but never
 * create a bruss_u_handle, never register a field, and never call
 * u_loop_step() -- there is no "U" work for them to drive, and the
 * coupler itself has no clock, no step, and no user code of its own
 * (docs/dev/coupler-placement.md's own enumerated call sites). In mpmd
 * mode --ranks-u defaults to this executable's own full world size (no
 * bundled coupler at all -- the coupler is coupler_main.c's own separate
 * appnum segment there instead).
 *
 * NOT YET HOST-VERIFIED (MPI cannot start in this sandbox, KNOWN_ISSUES.md
 * C1): whether a coupler placeholder rank skipping begin_step/end_step
 * entirely (rather than calling them with no field registered) is what
 * the Runner's own internal remap collectives actually expect. Flagged
 * in the Unit 2 report as the one open mechanical question beyond
 * Q1-Q3; build-verified only.
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fuse/abi/participant.h"
#include "u_loop.h"

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
  fprintf(stderr, "FAIL(u_main): %s: status=%d %s\n", what, (int)rc, buf);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

int main(int argc, char** argv) {
  const char* graph_path = NULL;
  const char* log_path = NULL;
  int ranks_u = -1; /* -1 = "all of my local world" (mpmd default) */
  int periods = 2;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
      graph_path = argv[++i];
    } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
      log_path = argv[++i];
    } else if (strcmp(argv[i], "--ranks-u") == 0 && i + 1 < argc) {
      ranks_u = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--periods") == 0 && i + 1 < argc) {
      periods = atoi(argv[++i]);
    } else {
      fprintf(stderr, "u_main: unrecognized argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (!graph_path || !log_path) {
    fprintf(stderr,
            "usage: u_main --graph graph.yaml --log u.log [--ranks-u N] "
            "[--periods N]\n");
    return 1;
  }

  MPI_Init(&argc, &argv);
  int world_rank = 0, world_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  if (ranks_u < 0) ranks_u = world_size; /* mpmd: every local rank is U */
  const int is_physics_rank = (world_rank < ranks_u);

  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) {
    fprintf(stderr, "FAIL(u_main): fuse_coupler_abi() returned NULL\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
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

  /* The ONE field a foreign caller sets to take part in a multi-
   * executable run: my_component. communicator is this executable's
   * OWN world (already MPI_Init'd); fuse_setup() resolves "U"'s
   * LaunchSpec from the loaded graph and connects accordingly (this
   * file's own header comment has the full account). */
  fuse_setup_options setup;
  memset(&setup, 0, sizeof(setup));
  setup.struct_size = sizeof(setup);
  setup.communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
  setup.my_component = "U";
  rc = t->setup(ctx, &setup, NULL);
  if (rc != FUSE_OK) fail(ctx, "setup", rc);

  bruss_u_handle* uh = NULL;
  double* u_in = NULL;
  double* u_out = NULL;
  int nu = 0;
  FILE* log = NULL;

  if (is_physics_rank) {
    fuse_field_ref u_out_ref, u_in_ref;
    memset(&u_out_ref, 0, sizeof(u_out_ref));
    u_out_ref.struct_size = sizeof(u_out_ref);
    fuse_field_name fname;
    memset(&fname, 0, sizeof(fname));
    fname.struct_size = sizeof(fname);
    fname.component = "U";
    fname.field = "u_out";
    rc = t->field_handle(ctx, &fname, &u_out_ref);
    if (rc != FUSE_OK) fail(ctx, "field_handle(u_out)", rc);

    /* NOTE: this reference implementation's own grid shape (nx, ny) must
     * match brusselator_graph_tool's own hardcoded kNxU/kNyU
     * (reproduced here as a build-time constant rather than queried
     * through the ABI, since the participant ABI has no "field length"
     * query beyond the view the caller itself supplies -- the caller is
     * expected to know its own grid, the same way u_main's own
     * `--ranks-u` is a caller-supplied fact about this run's layout,
     * not something fuse_load_graph() reports back). */
    const int nx = 3, ny = 3; /* brusselator_graph_tool.cpp's own kNxU/kNyU */
    nu = nx * ny;
    u_out = (double*)calloc((size_t)nu, sizeof(double));
    u_in = (double*)calloc((size_t)nu, sizeof(double));

    uh = bruss_u_create(nx, ny, /*diffusion=*/10.0, /*A=*/3.4, /*B=*/1.0,
                        /*forcing=*/1, u_in /* initial_v_on_u: zero is a
                                               valid, if not bit-identical
                                               to the tier-0 harness's own
                                               seeded, starting point for
                                               a standalone host run */);
    if (!uh) fail(ctx, "bruss_u_create", FUSE_ERR_RUNTIME);

    fuse_field_view view;
    memset(&view, 0, sizeof(view));
    view.struct_size = sizeof(view);
    view.handle = u_out_ref.handle;
    view.element_type = FUSE_ELEMENT_F64;
    view.memspace = FUSE_MEMSPACE_HOST;
    view.ownership = FUSE_OWNERSHIP_OWNED;
    view.components = 1;
    view.count = (uint64_t)nu;
    view.owned_count = (uint64_t)nu;
    view.stride = 1;
    view.data = u_out;
    rc = t->register_field(ctx, &view, NULL);
    if (rc != FUSE_OK) fail(ctx, "register_field(u_out)", rc);

    fname.field = "u_in";
    rc = t->field_handle(ctx, &fname, &u_in_ref);
    if (rc != FUSE_OK) fail(ctx, "field_handle(u_in)", rc);
    view.handle = u_in_ref.handle;
    view.data = u_in;
    rc = t->register_field(ctx, &view, NULL);
    if (rc != FUSE_OK) fail(ctx, "register_field(u_in)", rc);

    log = fopen(log_path, "w");
    if (!log) fail(ctx, "fopen(log)", FUSE_ERR_RUNTIME);
  }

  rc = t->initialize(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(ctx, "initialize", rc);

  if (is_physics_rank) {
    double* baseline_state = (double*)calloc((size_t)nu, sizeof(double));
    bruss_u_state(uh, baseline_state);
    double baseline_mean_u = 0.0;
    for (int i = 0; i < nu; ++i) baseline_mean_u += baseline_state[i];
    baseline_mean_u /= (double)nu;
    free(baseline_state);
    fprintf(log, "window=0 mean_u=%a external_total=%a exchange_out=%a\n", baseline_mean_u,
            bruss_u_external_total(uh), bruss_u_exchange_out(uh));

    fuse_step_request req;
    memset(&req, 0, sizeof(req));
    req.struct_size = sizeof(req);
    req.component = "U";

    for (int p = 0; p < periods; ++p) {
      fuse_ongoing ongoing;
      memset(&ongoing, 0, sizeof(ongoing));
      ongoing.struct_size = sizeof(ongoing);
      /* This process hosts only "U" (shim.h's own precedent, species_
       * component.hpp's single-component-per-rank convention) -- unlike
       * the tier-0 harness, which drives BOTH U and V in one process and
       * therefore does need next_component, u_main.c calls begin_step/
       * end_step for "U" every time it is this process's own turn, and
       * simply loops until the PERIOD closes; it never has to decide
       * which OTHER component is due, because no other component is
       * ever due on this process's own ranks. */
      fuse_step_report rep, begin_rep;
      for (;;) {
        rc = u_loop_step(ctx, uh, &req, u_in, u_out, &rep, &begin_rep);
        if (rc != FUSE_OK) fail(ctx, "u_loop_step", rc);
        if (rep.next_component == -1) break; /* period complete */
      }
      /* u_out holds q (the reaction term), NOT U's own state -- the log
       * format's own mean_u is the STATE mean (bruss_u_state), matching
       * main.cpp's own pu->state().mean(). */
      double* state = (double*)calloc((size_t)nu, sizeof(double));
      bruss_u_state(uh, state);
      double mean_u = 0.0;
      for (int i = 0; i < nu; ++i) mean_u += state[i];
      mean_u /= (double)nu;
      free(state);

      fprintf(log, "window=%d mean_u=%a external_total=%a exchange_out=%a\n",
              p + 1, mean_u, bruss_u_external_total(uh), bruss_u_exchange_out(uh));
    }
    fclose(log);
    bruss_u_destroy(uh);
    free(u_in);
    free(u_out);
  }

  rc = t->destroy(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(NULL, "destroy", rc);

  MPI_Finalize();
  return 0;
}
