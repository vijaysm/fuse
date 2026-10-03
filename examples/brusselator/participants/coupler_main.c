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
 * then destroy. It registers no field and never calls fuse_begin_step/
 * fuse_end_step at all: there is no step for a coupler to take.
 *
 * NOT YET HOST-VERIFIED (MPI cannot start in this sandbox,
 * KNOWN_ISSUES.md C1): whether fuse_initialize() alone (with no
 * subsequent begin_step/end_step call from this rank) is sufficient for
 * the Runner's own internal remap collectives that use the coupler's
 * ranks during U's and V's own begin_step/end_step calls -- flagged in
 * the Unit 2 report as the same open mechanical question u_main.c's own
 * header names for its bundled-coupler-rank case, applying here too.
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "fuse/abi/participant.h"

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
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) {
      graph_path = argv[++i];
    } else {
      fprintf(stderr, "coupler_main: unrecognized argument: %s\n", argv[i]);
      return 1;
    }
  }
  if (!graph_path) {
    fprintf(stderr, "usage: coupler_main --graph graph.yaml\n");
    return 1;
  }

  MPI_Init(&argc, &argv);

  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) {
    fprintf(stderr, "FAIL(coupler_main): fuse_coupler_abi() returned NULL\n");
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

  fuse_setup_options setup;
  memset(&setup, 0, sizeof(setup));
  setup.struct_size = sizeof(setup);
  setup.communicator = (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD);
  setup.my_component = "coupler";
  rc = t->setup(ctx, &setup, NULL);
  if (rc != FUSE_OK) fail(ctx, "setup", rc);

  rc = t->initialize(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(ctx, "initialize", rc);

  /* No field, no step -- a Coupler component is never advanced. */

  rc = t->destroy(ctx, NULL, NULL);
  if (rc != FUSE_OK) fail(NULL, "destroy", rc);

  MPI_Finalize();
  return 0;
}
