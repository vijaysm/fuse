/* The C ABI under a real mpiexec, with more than one rank.
 *
 * WHY THIS EXISTS. soma_end_step() decides convergence from norm_l2_diff over
 * THIS RANK'S buffer and then advances this rank's clock on that verdict.
 * Nothing in src/capi/soma_c.cpp reduces anything across ranks. Run two ranks
 * whose residuals differ and the component clocks diverge -- and before the
 * gate this test guards, soma_initialize() returned SOMA_OK on both while it
 * happened. That is a hang or a mistimed exchange in a real coupling, and it
 * needs no communicator to trigger: the run below never calls
 * soma_set_communicator at all.
 *
 * WHY IT IS C, not C++. Same reason as test/unit/capi_test.c: compiling the
 * public header with a C compiler is the only thing that proves it is C.
 * <mpi.h> here is the TEST's dependency, not the header's -- soma_c.h still
 * names no MPI type and takes the communicator as a plain int32_t.
 *
 * WHY IT IS NOT A soma_add_test(). It calls MPI_Init and is launched by
 * mpiexec, exactly like moab_parallel_test and mpi_transport_test. A rank that
 * fails calls MPI_Abort rather than returning, so a broken build fails the
 * CTest entry in seconds instead of hanging on a peer's collective.
 *
 * RUN AT -n 1 AND -n 2, and the -n 1 leg is not decoration: it is the positive
 * control that proves the gate refuses on rank COUNT and not merely on the
 * presence of MPI. Everything the -n 2 leg refuses, the -n 1 leg must accept.
 */

#include <soma/soma_c.h>

#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
static int g_rank = 0;
static int g_size = 1;

#define CHECK(cond, msg)                                                  \
  do {                                                                    \
    if (!(cond)) {                                                        \
      printf("FAIL [rank %d/%d] %s:%d: %s\n", g_rank, g_size, __FILE__,   \
             __LINE__, (msg));                                            \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

#define CHECK_RC(expr, expected, msg)                                     \
  do {                                                                    \
    const int rc_ = (expr);                                               \
    if (rc_ != (expected)) {                                              \
      printf("FAIL [rank %d/%d] %s:%d: %s (got %d, expected %d)\n",       \
             g_rank, g_size, __FILE__, __LINE__, (msg), rc_, (expected)); \
      ++failures;                                                         \
    }                                                                     \
  } while (0)

static double g_q[16];
static double g_in[16];

static int message_contains(soma_ctx ctx, const char* needle) {
  char buffer[4096];
  buffer[0] = '\0';
  if (soma_last_error(ctx, buffer, sizeof(buffer)) != SOMA_OK) return 0;
  return strstr(buffer, needle) != NULL;
}

/* pair_iterated.yaml: fluid.q -> solid.q_in inside an iteration region with a
   convergence measure, i.e. exactly the graph for which soma_end_step() takes
   a rank-local early-exit decision. */
static soma_ctx load_iterated(const char* data_dir) {
  soma_ctx ctx = NULL;
  int32_t  hq = -1;
  int32_t  hi = -1;
  char     path[4096];

  snprintf(path, sizeof(path), "%s/pair_iterated.yaml", data_dir);
  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, path), SOMA_OK, "load pair_iterated");
  CHECK_RC(soma_field_handle(ctx, "fluid", "q", &hq), SOMA_OK, "handle q");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_in", &hi), SOMA_OK, "handle q_in");
  CHECK_RC(soma_register_field(ctx, hq, g_q, 16, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_OK, "bind q");
  CHECK_RC(soma_register_field(ctx, hi, g_in, 16, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_OK, "bind q_in");
  return ctx;
}

/* pair.yaml declares `iteration_regions: []`, so no component converges and
   soma_end_step() takes no rank-local decision at all. */
static soma_ctx load_plain(const char* data_dir) {
  soma_ctx ctx = NULL;
  int32_t  hf = -1;
  int32_t  hw = -1;
  char     path[4096];

  snprintf(path, sizeof(path), "%s/pair.yaml", data_dir);
  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, path), SOMA_OK, "load pair");
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &hf), SOMA_OK, "hf");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_wall", &hw), SOMA_OK, "hw");
  CHECK_RC(soma_register_field(ctx, hf, g_q, 16, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_OK, "bind heat_flux");
  CHECK_RC(soma_register_field(ctx, hw, g_in, 16, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_OK, "bind q_wall");
  return ctx;
}

/* THE ONE THAT MATTERS. A rank-local convergence test on more than one rank,
   with no communicator ever set. Refused above 1 rank; accepted at exactly 1,
   where there is nobody to disagree with. */
static void test_rank_local_convergence(const char* data_dir) {
  soma_ctx ctx = load_iterated(data_dir);

  if (g_size > 1) {
    CHECK_RC(soma_initialize(ctx), SOMA_ERR_UNSUPPORTED,
             "a rank-local convergence early exit was accepted on >1 rank");
    CHECK(message_contains(ctx, "rank-local"),
          "the refusal must say why, not merely refuse");
    CHECK(message_contains(ctx, "fluid"),
          "the refusal must name the component whose window would diverge");
    /* Refused leaves LOADED, so the caller can still act on the advice. */
    CHECK_RC(soma_set_communicator(ctx, 0), SOMA_OK,
             "a refused initialize() must leave the context in LOADED");
  } else {
    CHECK_RC(soma_initialize(ctx), SOMA_OK,
             "a single-rank run has nothing to disagree with and must be "
             "accepted");
  }
  soma_destroy(ctx);
}

/* An explicitly serial scope inside a parallel job. MPI_COMM_SELF has size 1
   on every rank, so this is a correct serial run and must be accepted even at
   -n 2. Refusing it -- which the first version of this gate did -- turns
   working code into a hard failure. */
static void test_comm_self_is_accepted(const char* data_dir) {
  soma_ctx ctx = load_iterated(data_dir);
  CHECK_RC(soma_set_communicator(ctx, (int32_t)MPI_Comm_c2f(MPI_COMM_SELF)),
           SOMA_OK, "set_communicator(MPI_COMM_SELF)");
  CHECK_RC(soma_initialize(ctx), SOMA_OK,
           "MPI_COMM_SELF scopes this context to one rank; a correct serial "
           "run was refused");
  /* And the lifecycle rule still applies on top of a successful initialize. */
  CHECK_RC(soma_set_communicator(ctx, 0), SOMA_ERR_STATE,
           "set_communicator after initialize accepted");
  soma_destroy(ctx);
}

/* MPI_COMM_WORLD explicitly handed over. Same verdict as setting nothing --
   the predicate is the rank count, not whether the caller mentioned a
   communicator. */
static void test_comm_world_matches_the_default(const char* data_dir) {
  soma_ctx ctx = load_iterated(data_dir);
  CHECK_RC(soma_set_communicator(ctx, (int32_t)MPI_Comm_c2f(MPI_COMM_WORLD)),
           SOMA_OK, "set_communicator(MPI_COMM_WORLD)");
  CHECK_RC(soma_initialize(ctx), g_size > 1 ? SOMA_ERR_UNSUPPORTED : SOMA_OK,
           "an explicit MPI_COMM_WORLD must get the same verdict as setting "
           "no communicator at all");
  soma_destroy(ctx);
}

/* More than one rank, but no iteration region: nothing rank-local decides
   anything, the clocks cannot diverge, and the run is correct. Accepted, with
   an advisory recorded rather than an error returned. */
static void test_many_ranks_without_an_iteration_region(const char* data_dir) {
  soma_ctx ctx = load_plain(data_dir);
  CHECK_RC(soma_initialize(ctx), SOMA_OK,
           "a graph with no iteration region has nothing to diverge and must "
           "be accepted on any rank count");
  if (g_size > 1) {
    CHECK(message_contains(ctx, "note (not an error)"),
          "a multi-rank run must still be told the ABI reduces nothing");
  }
  soma_destroy(ctx);
}

/* The demonstration itself, at -n 2: drive the two ranks to DIFFERENT verdicts
   and show the clocks diverge. This is what the gate above is protecting
   against, and it runs on a context that deliberately bypasses the gate by
   scoping itself to MPI_COMM_SELF -- so the divergence is observed, not
   argued. Reported, not asserted as a failure: it is a known defect of the
   rank-local early exit, recorded here so that the day someone adds a
   reduction this prints identical windows and the CHECK below starts to fail. */
static void test_the_divergence_is_real(const char* data_dir) {
  soma_ctx ctx;
  double   t0 = -1.0;
  double   t1 = -1.0;
  int      sweep;
  int      i;

  if (g_size != 2) return;

  ctx = load_iterated(data_dir);
  CHECK_RC(soma_set_communicator(ctx, (int32_t)MPI_Comm_c2f(MPI_COMM_SELF)),
           SOMA_OK, "set_communicator(MPI_COMM_SELF)");
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize under MPI_COMM_SELF");

  for (sweep = 0; sweep < 2; ++sweep) {
    CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin");
    /* rank 0 repeats a value: residual 0 on sweep 2, so it converges and its
       clock advances. rank 1 moves every sweep and never converges. */
    for (i = 0; i < 16; ++i) {
      g_in[i] = (g_rank == 0) ? 100.0 : (double)(sweep + 1) * 100.0;
    }
    CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end");
  }
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin");
  printf("[rank %d/%d] after 2 sweeps the window is [%g, %g)\n", g_rank, g_size,
         t0, t1);
  fflush(stdout);
  CHECK(t0 == (g_rank == 0 ? 1.0 : 0.0),
        "the rank-local early exit no longer diverges -- if this fires, "
        "soma_end_step() has gained a reduction and the gate in "
        "soma_initialize() should be relaxed to match");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end");
  soma_destroy(ctx);
}

int main(int argc, char** argv) {
  const char* data_dir = (argc > 1) ? argv[1] : ".";

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_size);

  test_rank_local_convergence(data_dir);
  test_comm_self_is_accepted(data_dir);
  test_comm_world_matches_the_default(data_dir);
  test_many_ranks_without_an_iteration_region(data_dir);
  test_the_divergence_is_real(data_dir);

  fflush(stdout);
  if (failures != 0) {
    printf("capi_parallel_test: %d failure(s) on rank %d\n", failures, g_rank);
    fflush(stdout);
    /* Abort rather than return: a peer blocked in a later collective would
       otherwise hang the CTest entry instead of failing it. */
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  if (g_rank == 0) printf("capi_parallel_test: all checks passed\n");
  MPI_Finalize();
  return 0;
}
