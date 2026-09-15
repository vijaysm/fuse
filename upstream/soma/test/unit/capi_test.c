/* Tests for SOMA's C ABI.
 *
 * Compiled as C99, not C++, and linked without GoogleTest. That is the point:
 * a .cpp test would happily accept a header full of references, default
 * arguments, `bool`, and templates. This translation unit is the only thing
 * that proves include/soma/soma_c.h is what it claims to be.
 */

#include <soma/soma_c.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg)); \
      ++failures;                                            \
    }                                                        \
  } while (0)

#define CHECK_RC(expr, expected, msg)                                   \
  do {                                                                  \
    const int rc_ = (expr);                                             \
    if (rc_ != (expected)) {                                            \
      printf("FAIL %s:%d: %s (got %d, expected %d)\n", __FILE__,        \
             __LINE__, (msg), rc_, (expected));                         \
      ++failures;                                                       \
    }                                                                   \
  } while (0)

/* ------------------------------------------------------------------------ */
/* Helpers                                                                   */
/* ------------------------------------------------------------------------ */

/* pair.yaml: fluid.heat_flux (Provides, interval 1/2) -> solid.q_wall
   (Requires, interval 1). Both must be registered before initialize. */
static double g_flux[16];
static double g_wall[16];

static soma_ctx loaded_pair(void) {
  soma_ctx ctx = NULL;
  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml"), SOMA_OK,
           "load pair.yaml");
  return ctx;
}

static void bind_pair_fields(soma_ctx ctx) {
  int32_t hf = -1;
  int32_t hw = -1;
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &hf), SOMA_OK,
           "handle fluid.heat_flux");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_wall", &hw), SOMA_OK,
           "handle solid.q_wall");
  CHECK_RC(soma_register_field(ctx, hf, g_flux, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_OK, "register fluid.heat_flux");
  CHECK_RC(soma_register_field(ctx, hw, g_wall, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_OK, "register solid.q_wall");
}

static int message_nonempty(soma_ctx ctx) {
  char buffer[1024];
  buffer[0] = 'x';
  if (soma_last_error(ctx, buffer, sizeof(buffer)) != SOMA_OK) return 0;
  return strlen(buffer) > 0;
}

/* Two distinct defects both come back as SOMA_ERR_INVALID, so a test that
   checks only the code cannot tell them apart -- and one of the tests below
   provably could not: with the empty-criterion check disabled, the
   every-channel-must-be-measured check fires on the same graph and returns
   the same code. The diagnostic carries the SOMA error code that decided it,
   so the tests name the one they mean. */
static int message_contains(soma_ctx ctx, const char* needle) {
  char buffer[4096];
  buffer[0] = '\0';
  if (soma_last_error(ctx, buffer, sizeof(buffer)) != SOMA_OK) return 0;
  return strstr(buffer, needle) != NULL;
}

/* ------------------------------------------------------------------------ */
/* Version                                                                   */
/* ------------------------------------------------------------------------ */

static void test_version(void) {
  CHECK(soma_version() != NULL, "version string is null");
  CHECK(strlen(soma_version()) > 0, "version string is empty");
  CHECK(soma_abi_version() >= 1, "abi version must be at least 1");
  CHECK(soma_abi_version() == SOMA_ABI_VERSION,
        "library ABI version disagrees with the header it was built from");
  /* The pointer must be stable: it is documented as static storage. */
  CHECK(soma_version() == soma_version(), "version string is not stable");
}

/* ------------------------------------------------------------------------ */
/* Lifetime and handle validation                                            */
/* ------------------------------------------------------------------------ */

static void test_create_destroy(void) {
  soma_ctx ctx = NULL;
  CHECK_RC(soma_create(&ctx), SOMA_OK, "create failed");
  CHECK(ctx != NULL, "create returned a null context");
  CHECK_RC(soma_destroy(ctx), SOMA_OK, "destroy failed");
}

static void test_null_arguments_are_rejected(void) {
  /* A handle SOMA never issued. Validation is a lookup in a table of live
     pointers, so this must be rejected without ever being dereferenced. */
  soma_ctx bogus = (soma_ctx)(uintptr_t)0x10;
  soma_ctx ctx = NULL;

  CHECK_RC(soma_create(NULL), SOMA_ERR_INVALID, "null out-param accepted");
  CHECK_RC(soma_destroy(NULL), SOMA_ERR_INVALID, "null context accepted");
  CHECK_RC(soma_destroy(bogus), SOMA_ERR_INVALID, "bogus handle accepted");
  CHECK_RC(soma_initialize(bogus), SOMA_ERR_INVALID, "bogus handle accepted");

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK(ctx != NULL, "create returned null");
  CHECK_RC(soma_destroy(ctx), SOMA_OK, "destroy");
}

static void test_double_destroy_and_stale_handles(void) {
  soma_ctx ctx = NULL;
  int32_t  handle = 0;
  double   a = 0.0;
  double   b = 0.0;
  int32_t  flag = 0;
  uint64_t len = 0;
  char     buffer[64];

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_destroy(ctx), SOMA_OK, "first destroy");

  /* Every one of these must be a clean rejection, not a use-after-free. */
  CHECK_RC(soma_destroy(ctx), SOMA_ERR_INVALID, "double destroy accepted");
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml"),
           SOMA_ERR_INVALID, "stale handle accepted by load_graph_yaml");
  CHECK_RC(soma_load_graph(ctx, "xxxx", 4), SOMA_ERR_INVALID,
           "stale handle accepted by load_graph");
  CHECK_RC(soma_save_graph(ctx, NULL, 0, &len), SOMA_ERR_INVALID,
           "stale handle accepted by save_graph");
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &handle),
           SOMA_ERR_INVALID, "stale handle accepted by field_handle");
  CHECK_RC(soma_register_field(ctx, 0, g_flux, 1, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "stale handle accepted by register_field");
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_INVALID,
           "stale handle accepted by initialize");
  CHECK_RC(soma_begin_step(ctx, "fluid", &a, &b), SOMA_ERR_INVALID,
           "stale handle accepted by begin_step");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_ERR_INVALID,
           "stale handle accepted by end_step");
  CHECK_RC(soma_requires_checkpoint(ctx, "fluid", &flag), SOMA_ERR_INVALID,
           "stale handle accepted by requires_checkpoint");
  CHECK_RC(soma_requires_restore(ctx, "fluid", &flag), SOMA_ERR_INVALID,
           "stale handle accepted by requires_restore");
  CHECK_RC(soma_is_coupling_ongoing(ctx, &flag), SOMA_ERR_INVALID,
           "stale handle accepted by is_coupling_ongoing");
  CHECK_RC(soma_run_period(ctx), SOMA_ERR_INVALID,
           "stale handle accepted by run_period");
  CHECK_RC(soma_run(ctx, 1), SOMA_ERR_INVALID, "stale handle accepted by run");
  CHECK_RC(soma_set_communicator(ctx, 0), SOMA_ERR_INVALID,
           "stale handle accepted by set_communicator");
  CHECK_RC(soma_set_end_time(ctx, 1, 1), SOMA_ERR_INVALID,
           "stale handle accepted by set_end_time");
  CHECK_RC(soma_last_error(ctx, buffer, sizeof(buffer)), SOMA_ERR_INVALID,
           "stale handle accepted by last_error");
  CHECK_RC(soma_last_error_length(ctx, &len), SOMA_ERR_INVALID,
           "stale handle accepted by last_error_length");
}

/* ------------------------------------------------------------------------ */
/* Error retrieval                                                           */
/* ------------------------------------------------------------------------ */

static void test_error_message_is_retrievable(void) {
  soma_ctx ctx = NULL;
  char     buffer[512];
  uint64_t len = 0;

  soma_create(&ctx);
  CHECK_RC(soma_load_graph_yaml(ctx, "/definitely/not/a/file.yaml"),
           SOMA_ERR_INVALID, "a missing YAML file must be rejected");

  CHECK_RC(soma_last_error(ctx, buffer, sizeof(buffer)), SOMA_OK,
           "last_error failed");
  CHECK(strlen(buffer) > 0, "error message is empty");
  CHECK(strstr(buffer, "definitely") != NULL,
        "error message does not mention the offending path");

  CHECK_RC(soma_last_error_length(ctx, &len), SOMA_OK,
           "last_error_length failed");
  CHECK(len == (uint64_t)strlen(buffer),
        "last_error_length disagrees with the copied message");

  soma_destroy(ctx);
}

static void test_error_message_truncates_safely(void) {
  soma_ctx ctx = NULL;
  char     small[8];
  uint64_t len = 0;
  size_t   i;

  soma_create(&ctx);
  soma_load_graph_yaml(ctx, "/definitely/not/a/file.yaml");
  CHECK_RC(soma_last_error_length(ctx, &len), SOMA_OK, "length");
  CHECK(len > sizeof(small), "test needs a message longer than the buffer");

  for (i = 0; i < sizeof(small); ++i) small[i] = 'Z';
  CHECK_RC(soma_last_error(ctx, small, sizeof(small)), SOMA_OK,
           "truncated last_error must still succeed");
  CHECK(strlen(small) == sizeof(small) - 1,
        "truncation must fill the buffer exactly");
  CHECK(small[sizeof(small) - 1] == '\0', "truncated message is not NUL-terminated");

  CHECK_RC(soma_last_error(ctx, small, 0), SOMA_ERR_INVALID,
           "zero-size buffer accepted");
  CHECK_RC(soma_last_error(ctx, NULL, 16), SOMA_ERR_INVALID,
           "null buffer accepted");
  CHECK_RC(soma_last_error_length(ctx, NULL), SOMA_ERR_INVALID,
           "null out_length accepted");

  soma_destroy(ctx);
}

/* Without this path a caller can never learn why soma_create() failed: there
   is no context to ask. */
static void test_thread_local_error_without_a_context(void) {
  char buffer[256];
  buffer[0] = '\0';
  CHECK_RC(soma_create(NULL), SOMA_ERR_INVALID, "null out-param accepted");
  CHECK_RC(soma_last_error(NULL, buffer, sizeof(buffer)), SOMA_OK,
           "thread-local last_error failed");
  CHECK(strlen(buffer) > 0, "thread-local error message is empty");
  CHECK(strstr(buffer, "soma_create") != NULL,
        "thread-local message does not name the failing call");
}

/* ------------------------------------------------------------------------ */
/* The narrow waist                                                          */
/* ------------------------------------------------------------------------ */

static void test_yaml_load_and_handles(void) {
  soma_ctx ctx = loaded_pair();
  int32_t  handle = -1;
  int32_t  missing = 7;

  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &handle), SOMA_OK,
           "field handle lookup failed");
  CHECK(handle >= 0, "handle is negative");

  CHECK_RC(soma_field_handle(ctx, "fluid", "nope", &missing),
           SOMA_ERR_NOTFOUND, "missing field should report NOTFOUND");
  CHECK(missing == -1, "a failed lookup must clear the out-parameter");
  CHECK_RC(soma_field_handle(ctx, "nosuch", "heat_flux", &missing),
           SOMA_ERR_NOTFOUND, "missing component should report NOTFOUND");
  CHECK_RC(soma_field_handle(ctx, NULL, "heat_flux", &missing),
           SOMA_ERR_INVALID, "null component name accepted");
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", NULL),
           SOMA_ERR_INVALID, "null out_handle accepted");

  /* One graph per context. */
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml"),
           SOMA_ERR_STATE, "a second graph load must be refused");

  soma_destroy(ctx);
}

/* The descriptor is the ABI boundary (spec 3.2): what YAML produces must go
   through soma_load_graph() and come out the same graph. */
static void test_descriptor_round_trip(void) {
  soma_ctx  from_yaml = loaded_pair();
  soma_ctx  from_desc = NULL;
  uint64_t  bytes = 0;
  uint64_t  again = 0;
  unsigned char* blob = NULL;
  int32_t   h1 = -1;
  int32_t   h2 = -1;
  unsigned char tiny[4];

  CHECK_RC(soma_save_graph(from_yaml, NULL, 0, &bytes), SOMA_OK, "size query");
  CHECK(bytes > 0, "descriptor is empty");

  CHECK_RC(soma_save_graph(from_yaml, tiny, sizeof(tiny), &again),
           SOMA_ERR_INVALID, "a too-small buffer must be refused");
  CHECK(again == bytes,
        "a refused save must still report the size the caller needs");

  blob = (unsigned char*)malloc((size_t)bytes);
  CHECK(blob != NULL, "malloc");
  CHECK_RC(soma_save_graph(from_yaml, blob, bytes, &again), SOMA_OK, "save");
  CHECK(again == bytes, "save reported a different size than the query");

  CHECK_RC(soma_create(&from_desc), SOMA_OK, "create");
  CHECK_RC(soma_load_graph(from_desc, blob, bytes), SOMA_OK,
           "descriptor load failed");

  CHECK_RC(soma_field_handle(from_yaml, "solid", "q_wall", &h1), SOMA_OK,
           "handle via yaml");
  CHECK_RC(soma_field_handle(from_desc, "solid", "q_wall", &h2), SOMA_OK,
           "handle via descriptor");
  CHECK(h1 == h2, "the descriptor round trip changed a field handle");

  free(blob);
  soma_destroy(from_desc);
  soma_destroy(from_yaml);
}

static void test_descriptor_rejects_garbage(void) {
  soma_ctx ctx = NULL;
  uint64_t bytes = 0;
  const char garbage[16] = "not a desc";

  soma_create(&ctx);
  CHECK_RC(soma_save_graph(ctx, NULL, 0, &bytes), SOMA_ERR_STATE,
           "save with no graph accepted");
  CHECK_RC(soma_load_graph(ctx, NULL, 8), SOMA_ERR_INVALID,
           "null descriptor accepted");
  CHECK_RC(soma_load_graph(ctx, garbage, 0), SOMA_ERR_INVALID,
           "zero-length descriptor accepted");
  CHECK_RC(soma_load_graph(ctx, garbage, sizeof(garbage)), SOMA_ERR_INVALID,
           "garbage descriptor accepted");
  CHECK(message_nonempty(ctx), "rejecting a garbage descriptor left no message");
  /* A rejected load must leave the context EMPTY, not half-loaded. */
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair.yaml"), SOMA_OK,
           "a failed load must not consume the context's single graph slot");
  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* Field registration                                                        */
/* ------------------------------------------------------------------------ */

static void test_register_field(void) {
  soma_ctx ctx = loaded_pair();
  int32_t  handle = -1;
  double   data[16];
  int      i;

  for (i = 0; i < 16; ++i) data[i] = (double)i;
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &handle), SOMA_OK,
           "handle");
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_OK, "register_field failed");

  /* Re-registering must be refused, not silently accepted. */
  CHECK(soma_register_field(ctx, handle, data, 16, 1, 0, 1,
                            SOMA_MEMSPACE_HOST) != SOMA_OK,
        "rebinding should be refused");

  soma_destroy(ctx);
}

static void test_register_field_rejects_bad_arguments(void) {
  soma_ctx ctx = loaded_pair();
  int32_t  handle = -1;
  double   data[16];

  soma_field_handle(ctx, "fluid", "heat_flux", &handle);

  CHECK_RC(soma_register_field(ctx, handle, NULL, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "null data accepted");
  CHECK_RC(soma_register_field(ctx, -1, data, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "negative handle accepted");
  CHECK_RC(soma_register_field(ctx, 9999, data, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "out-of-range handle accepted");
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 1, 0, 1, 42),
           SOMA_ERR_INVALID, "bogus memspace accepted");
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 1, 0, 0,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "zero components accepted");
  /* heat_flux is declared with 1 component; claiming 3 is silent wrong
     physics, not a layout question. */
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 3, 1, 3,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "component-count mismatch accepted");
  /* An aliasing layout: 16 elements of stride 0 all land on the same slot. */
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 0, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_INVALID, "aliasing layout accepted");
  CHECK(message_nonempty(ctx), "a rejected registration left no message");

  /* None of the failures above may have bound anything. */
  CHECK_RC(soma_register_field(ctx, handle, data, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_OK, "a rejected registration left the field bound");

  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* State machine                                                             */
/* ------------------------------------------------------------------------ */

static void test_state_errors(void) {
  soma_ctx ctx = NULL;
  double   a = -1.0;
  double   b = -1.0;
  int32_t  flag = -1;

  soma_create(&ctx);
  /* run before initialize */
  CHECK_RC(soma_run_period(ctx), SOMA_ERR_STATE, "run before init accepted");
  CHECK_RC(soma_run(ctx, 3), SOMA_ERR_STATE, "run before init accepted");
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_STATE,
           "initialize with no graph accepted");
  CHECK_RC(soma_field_handle(ctx, "fluid", "heat_flux", &flag), SOMA_ERR_STATE,
           "field lookup with no graph accepted");
  CHECK_RC(soma_register_field(ctx, 0, g_flux, 1, 1, 0, 1, SOMA_MEMSPACE_HOST),
           SOMA_ERR_STATE, "registration with no graph accepted");
  CHECK_RC(soma_begin_step(ctx, "fluid", &a, &b), SOMA_ERR_STATE,
           "begin_step before init accepted");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_ERR_STATE,
           "end_step before init accepted");
  CHECK_RC(soma_is_coupling_ongoing(ctx, &flag), SOMA_ERR_STATE,
           "is_coupling_ongoing before init accepted");
  CHECK(a == -1.0 && b == -1.0, "a failed begin_step wrote its out-params");
  soma_destroy(ctx);
}

static void test_initialize_requires_registered_fields(void) {
  soma_ctx ctx = loaded_pair();
  int32_t  hf = -1;

  /* Only the source is registered; the target is not. */
  soma_field_handle(ctx, "fluid", "heat_flux", &hf);
  soma_register_field(ctx, hf, g_flux, 16, 1, 0, 1, SOMA_MEMSPACE_HOST);

  CHECK_RC(soma_initialize(ctx), SOMA_ERR_STATE,
           "initialize accepted an unregistered channel field");
  {
    char buffer[512];
    soma_last_error(ctx, buffer, sizeof(buffer));
    CHECK(strstr(buffer, "q_wall") != NULL,
          "the error must name the unregistered field");
  }
  soma_destroy(ctx);
}

static void test_initialize_and_step(void) {
  soma_ctx ctx = loaded_pair();
  double   t0 = -1.0;
  double   t1 = -1.0;
  int32_t  flag = -1;

  bind_pair_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize failed");
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_STATE,
           "a second initialize must be refused");
  CHECK_RC(soma_register_field(ctx, 0, g_flux, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST),
           SOMA_ERR_STATE, "registration after initialize accepted");
  CHECK_RC(soma_set_end_time(ctx, 1, 1), SOMA_ERR_STATE,
           "set_end_time after initialize accepted");

  /* fluid's interval is 1/2, solid's is 1. */
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin fluid");
  CHECK(t0 == 0.0 && t1 == 0.5, "fluid's first window is not [0, 1/2)");
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_ERR_STATE,
           "a second begin_step without an end_step accepted");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end fluid");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_ERR_STATE,
           "end_step without a begin_step accepted");
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin fluid 2");
  CHECK(t0 == 0.5 && t1 == 1.0, "fluid's second window is not [1/2, 1)");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end fluid 2");

  CHECK_RC(soma_begin_step(ctx, "solid", &t0, &t1), SOMA_OK, "begin solid");
  CHECK(t0 == 0.0 && t1 == 1.0, "solid's first window is not [0, 1)");
  CHECK_RC(soma_end_step(ctx, "solid"), SOMA_OK, "end solid");

  CHECK_RC(soma_begin_step(ctx, "nosuch", &t0, &t1), SOMA_ERR_NOTFOUND,
           "begin_step on an unknown component accepted");
  CHECK_RC(soma_end_step(ctx, "nosuch"), SOMA_ERR_NOTFOUND,
           "end_step on an unknown component accepted");
  CHECK_RC(soma_begin_step(ctx, "fluid", NULL, &t1), SOMA_ERR_INVALID,
           "null t_begin accepted");

  /* No component of pair.yaml is in an iteration region. */
  CHECK_RC(soma_requires_checkpoint(ctx, "fluid", &flag), SOMA_OK,
           "requires_checkpoint");
  CHECK(flag == 0, "a non-iterated component must not require a checkpoint");
  CHECK_RC(soma_requires_restore(ctx, "fluid", &flag), SOMA_OK,
           "requires_restore");
  CHECK(flag == 0, "a non-iterated component must not require a restore");

  /* Reserved; never SOMA_OK in this build, but diagnosed rather than silent. */
  CHECK_RC(soma_run_period(ctx), SOMA_ERR_UNSUPPORTED, "run_period");
  CHECK_RC(soma_run(ctx, 4), SOMA_ERR_UNSUPPORTED, "run");
  CHECK_RC(soma_run(ctx, -1), SOMA_ERR_INVALID, "negative period count");
  CHECK(message_nonempty(ctx), "an unsupported call left no message");

  soma_destroy(ctx);
}

static void test_coupling_ongoing(void) {
  soma_ctx ctx = loaded_pair();
  int32_t  ongoing = -1;
  double   t0 = 0.0;
  double   t1 = 0.0;

  bind_pair_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize");
  /* No end time: the caller decides when to stop. */
  CHECK_RC(soma_is_coupling_ongoing(ctx, &ongoing), SOMA_OK, "ongoing");
  CHECK(ongoing == 1, "without an end time the run is always ongoing");
  soma_destroy(ctx);

  ctx = loaded_pair();
  bind_pair_fields(ctx);
  CHECK_RC(soma_set_end_time(ctx, 0, 0), SOMA_ERR_INVALID,
           "zero denominator accepted");
  CHECK_RC(soma_set_end_time(ctx, 1, 1), SOMA_OK, "set_end_time");
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize");

  CHECK_RC(soma_is_coupling_ongoing(ctx, &ongoing), SOMA_OK, "ongoing");
  CHECK(ongoing == 1, "the run should be ongoing at t = 0 with end time 1");

  /* Advance fluid to 1 (two half steps) and solid to 1 (one step). */
  soma_begin_step(ctx, "fluid", &t0, &t1);
  soma_end_step(ctx, "fluid");
  soma_begin_step(ctx, "fluid", &t0, &t1);
  soma_end_step(ctx, "fluid");
  CHECK_RC(soma_is_coupling_ongoing(ctx, &ongoing), SOMA_OK, "ongoing");
  CHECK(ongoing == 1, "solid has not reached the end time yet");

  soma_begin_step(ctx, "solid", &t0, &t1);
  soma_end_step(ctx, "solid");
  CHECK_RC(soma_is_coupling_ongoing(ctx, &ongoing), SOMA_OK, "ongoing");
  CHECK(ongoing == 0, "every clock has reached the end time");

  CHECK_RC(soma_is_coupling_ongoing(ctx, NULL), SOMA_ERR_INVALID,
           "null out_ongoing accepted");
  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* Iteration regions: the checkpoint/restore contract                        */
/* ------------------------------------------------------------------------ */

static void test_iteration_checkpoint_contract(void) {
  /* pair_iterated.yaml: fluid.q -> solid.q_in, both interval 1, in an
     iteration region with max_iterations 30. */
  soma_ctx ctx = NULL;
  int32_t  hq = -1;
  int32_t  hi = -1;
  int32_t  need = -1;
  double   t0 = -1.0;
  double   t1 = -1.0;
  int      sweep;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair_iterated.yaml"),
           SOMA_OK, "load pair_iterated.yaml");
  CHECK_RC(soma_field_handle(ctx, "fluid", "q", &hq), SOMA_OK, "handle q");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_in", &hi), SOMA_OK,
           "handle q_in");
  CHECK_RC(soma_register_field(ctx, hq, g_flux, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q");
  CHECK_RC(soma_register_field(ctx, hi, g_wall, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q_in");
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize");

  /* Sweep 0: checkpoint, no restore. */
  CHECK_RC(soma_requires_checkpoint(ctx, "fluid", &need), SOMA_OK, "ckpt");
  CHECK(need == 1, "an iterated component must checkpoint before sweep 0");
  CHECK_RC(soma_requires_restore(ctx, "fluid", &need), SOMA_OK, "restore");
  CHECK(need == 0, "nothing to restore before sweep 0");

  for (sweep = 0; sweep < 30; ++sweep) {
    int i;
    CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin");
    CHECK(t0 == 0.0 && t1 == 1.0,
          "every sweep of an iteration region covers the same window");
    /* pair_iterated.yaml declares Converged::Relative(1e-6, "q"), and
       soma_end_step() now tests it (this is the gap this change closes --
       see soma_c.cpp). The measured quantity is solid.q_in (g_wall), the
       channel's target. This test exists to pin the checkpoint/restore
       contract across a FULL max_iterations window, so g_wall is driven to a
       new value every sweep: the residual between consecutive sweeps is then
       always 100 per element (a fixed step), while the reference norm grows
       with the sweep count, so the ratio is 1/(sweep+1) -- at its smallest,
       around sweep 28, still ~0.034, far above the 1e-6 tolerance -- so this
       window is guaranteed to run all 30 sweeps without ever satisfying the
       criterion early. (Leaving g_wall untouched, as a
       prior version of this test did, made every sweep's residual exactly
       0 and converged after sweep 1 -- a real, if accidental, test of the
       very bug this change fixes. See the task report.) */
    for (i = 0; i < 16; ++i) g_wall[i] = (double)(sweep + 1) * 100.0;
    CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end");

    soma_requires_restore(ctx, "fluid", &need);
    if (sweep < 29) {
      CHECK(need == 1, "a non-final sweep must be followed by a restore");
      soma_requires_checkpoint(ctx, "fluid", &need);
      CHECK(need == 0, "a mid-window sweep must not re-checkpoint");
    } else {
      CHECK(need == 0, "the final sweep must not ask for a restore");
      soma_requires_checkpoint(ctx, "fluid", &need);
      CHECK(need == 1, "the next window must start with a checkpoint");
    }
  }

  /* Only after the last sweep does the clock move on. */
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin");
  CHECK(t0 == 1.0 && t1 == 2.0,
        "the clock must advance exactly once per iteration region window");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end");

  CHECK_RC(soma_requires_checkpoint(ctx, "nosuch", &need), SOMA_ERR_NOTFOUND,
           "requires_checkpoint on an unknown component accepted");
  CHECK_RC(soma_requires_restore(ctx, "fluid", NULL), SOMA_ERR_INVALID,
           "null out-param accepted");
  soma_destroy(ctx);
}

/* Loads pair_iterated.yaml (fluid.q -> solid.q_in, max_iterations 30,
   Converged::Relative(1e-6, "q")) and drives "fluid" through exactly two
   sweeps that deliver the SAME value into solid.q_in. The residual (the
   change in solid.q_in between what the sweep started with and what it
   ends with) is then exactly 0 on the second sweep, which satisfies any
   positive tolerance -- so this proves soma_end_step() stops the window
   there, 28 sweeps short of max_iterations, rather than sweeping to 30
   regardless of convergence. Evidence is: (1) soma_requires_checkpoint() /
   soma_requires_restore() flip back to "start of a fresh window" after only
   2 sweeps, and (2) the *next* soma_begin_step() reports the advanced
   window [1, 2), not a third sweep of [0, 1). */
static void test_iteration_convergence_early_exit(void) {
  soma_ctx ctx = NULL;
  int32_t  hq = -1;
  int32_t  hi = -1;
  int32_t  need = -1;
  double   t0 = -1.0;
  double   t1 = -1.0;
  int      i;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair_iterated.yaml"),
           SOMA_OK, "load pair_iterated.yaml");
  CHECK_RC(soma_field_handle(ctx, "fluid", "q", &hq), SOMA_OK, "handle q");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_in", &hi), SOMA_OK,
           "handle q_in");
  CHECK_RC(soma_register_field(ctx, hq, g_flux, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q");
  CHECK_RC(soma_register_field(ctx, hi, g_wall, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q_in");
  CHECK_RC(soma_initialize(ctx), SOMA_OK, "initialize");

  /* Sweep 1: y0 (captured when this window opened, at iteration 0) is all
     zero -- nothing has written to g_wall yet. Deliver a large jump so this
     sweep alone plainly has not converged. */
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin sweep 1");
  CHECK(t0 == 0.0 && t1 == 1.0, "sweep 1 window is [0, 1)");
  for (i = 0; i < 16; ++i) g_wall[i] = 100.0;
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end sweep 1");

  CHECK_RC(soma_requires_restore(ctx, "fluid", &need), SOMA_OK, "restore1");
  CHECK(need == 1, "sweep 1 alone must not look converged");

  /* Sweep 2: deliver the SAME value again -- a residual of exactly 0. */
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin sweep 2");
  CHECK(t0 == 0.0 && t1 == 1.0, "sweep 2 is still the same window");
  for (i = 0; i < 16; ++i) g_wall[i] = 100.0;
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end sweep 2");

  /* Evidence #1: the checkpoint/restore contract already reads "fresh
     window", not "28 sweeps still to go". */
  CHECK_RC(soma_requires_restore(ctx, "fluid", &need), SOMA_OK, "restore2");
  CHECK(need == 0,
        "converging on sweep 2 must not ask for a restore into sweep 3");
  CHECK_RC(soma_requires_checkpoint(ctx, "fluid", &need), SOMA_OK, "ckpt2");
  CHECK(need == 1,
        "converging on sweep 2 must checkpoint for the NEXT window");

  /* Evidence #2: the clock has actually moved. */
  CHECK_RC(soma_begin_step(ctx, "fluid", &t0, &t1), SOMA_OK, "begin next");
  CHECK(t0 == 1.0 && t1 == 2.0,
        "early convergence must advance the clock past the first window "
        "instead of re-opening [0, 1) for a third sweep");
  CHECK_RC(soma_end_step(ctx, "fluid"), SOMA_OK, "end next");

  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* Iteration regions: the graph checks soma_initialize() makes on its own     */
/*                                                                            */
/* The Runner reaches these through resolve_regions() (src/runtime/           */
/* iteration.cpp), which needs a ComponentSet this ABI never builds. Both     */
/* refusals below are therefore soma_initialize()'s own, and both are         */
/* mutation-checked: disabling the check in src/capi/soma_c.cpp makes each of */
/* these load, initialize, and report nothing.                                */
/* ------------------------------------------------------------------------ */

/* Bind the two fields of the pair_iterated* family, whose channel is
   fluid.q -> solid.q_in. Returns nothing; every failure is a CHECK. */
static void bind_iterated_fields(soma_ctx ctx) {
  int32_t hq = -1;
  int32_t hi = -1;
  CHECK_RC(soma_field_handle(ctx, "fluid", "q", &hq), SOMA_OK, "handle q");
  CHECK_RC(soma_field_handle(ctx, "solid", "q_in", &hi), SOMA_OK,
           "handle q_in");
  CHECK_RC(soma_register_field(ctx, hq, g_flux, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q");
  CHECK_RC(soma_register_field(ctx, hi, g_wall, 16, 1, 0, 1,
                               SOMA_MEMSPACE_HOST), SOMA_OK, "bind q_in");
}

/* An iteration region with an empty ConvergenceCriterion. all_satisfied() on
   one is trivially true, so the region's stopping rule is "always stop": the
   window would end after one sweep and report convergence. That is the state
   E-ITER-NOCONVERGE exists to refuse, and until this change the C ABI was the
   one path that reached State::Initialized with it. */
static void test_initialize_rejects_an_empty_convergence_criterion(void) {
  soma_ctx ctx = NULL;
  int32_t  need = -1;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(
               ctx, SOMA_TEST_DATA_DIR "/pair_iterated_no_convergence.yaml"),
           SOMA_OK, "load pair_iterated_no_convergence.yaml");
  bind_iterated_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_INVALID,
           "an iteration region with no convergence measure was accepted");
  CHECK(message_contains(ctx, "E-ITER-NOCONVERGE"),
        "the refusal must name E-ITER-NOCONVERGE, not merely be a refusal");

  /* Refused means refused: the context is still LOADED, so nothing that needs
     INITIALIZED works and a second initialize() is not a STATE error. */
  CHECK_RC(soma_requires_checkpoint(ctx, "fluid", &need), SOMA_ERR_STATE,
           "a refused initialize() left the context usable");
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_INVALID,
           "a refused initialize() must stay refused, not become STATE");
  soma_destroy(ctx);
}

/* A convergence measure naming a field no channel of the region carries. It
   would never be tested against anything; resolve_regions() refuses it with
   E-ITER-MEASURE. Before this change soma_initialize() accepted it and left
   CompRuntime::converges false, i.e. silently declined to honour the
   criterion the author wrote. */
static void test_initialize_rejects_an_unmatched_convergence_measure(void) {
  soma_ctx ctx = NULL;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(
               ctx, SOMA_TEST_DATA_DIR "/pair_iterated_bad_measure.yaml"),
           SOMA_OK, "load pair_iterated_bad_measure.yaml");
  bind_iterated_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_INVALID,
           "a convergence measure matching no channel of its region was "
           "accepted");
  CHECK(message_contains(ctx, "E-ITER-MEASURE"),
        "the refusal must name E-ITER-MEASURE, not merely be a refusal");
  CHECK(message_contains(ctx, "t_wall"),
        "the refusal must name the measure that matched nothing");
  soma_destroy(ctx);
}

/* max_iterations below 1. The Runner refuses it with E-ITER-MAXITER; the C ABI
   used to clamp it to 1 and run one silent sweep. Measured before the fix:
   rc 0, empty message, requires_checkpoint 0, window [0,1) then [1,2). */
static void test_initialize_rejects_zero_max_iterations(void) {
  soma_ctx ctx = NULL;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(
               ctx, SOMA_TEST_DATA_DIR "/pair_iterated_zero_iterations.yaml"),
           SOMA_OK, "load pair_iterated_zero_iterations.yaml");
  bind_iterated_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_ERR_INVALID,
           "an iteration region with max_iterations 0 was accepted");
  CHECK(message_contains(ctx, "E-ITER-MAXITER"),
        "the refusal must name E-ITER-MAXITER, not merely be a refusal");
  soma_destroy(ctx);
}

/* The graph that IS well formed still initializes. Without this the three
   refusals above would be satisfied by a soma_initialize() that rejected
   every iteration region. */
static void test_initialize_still_accepts_a_well_formed_region(void) {
  soma_ctx ctx = NULL;

  CHECK_RC(soma_create(&ctx), SOMA_OK, "create");
  CHECK_RC(soma_load_graph_yaml(ctx, SOMA_TEST_DATA_DIR "/pair_iterated.yaml"),
           SOMA_OK, "load pair_iterated.yaml");
  bind_iterated_fields(ctx);
  CHECK_RC(soma_initialize(ctx), SOMA_OK,
           "a region with a measure naming its own source field was refused");
  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* Communicator                                                              */
/* ------------------------------------------------------------------------ */

static void test_set_communicator(void) {
  soma_ctx ctx = loaded_pair();
  const int rc = soma_set_communicator(ctx, 0);
  /* SOMA_OK on an MPI build, SOMA_ERR_UNSUPPORTED otherwise -- never a
     silent success that drops the communicator on the floor. */
  CHECK(rc == SOMA_OK || rc == SOMA_ERR_UNSUPPORTED,
        "set_communicator returned neither OK nor UNSUPPORTED");
  if (rc == SOMA_ERR_UNSUPPORTED) {
    CHECK(message_nonempty(ctx), "UNSUPPORTED must explain itself");
  }
  bind_pair_fields(ctx);
  /* Recording a communicator is NOT by itself a reason to refuse: this
     process is serial, so whatever was recorded resolves to one rank and
     nothing about the run is wrong. (Handle 0 is Fortran MPI_COMM_NULL; an
     explicitly serial MPI_COMM_SELF is covered on real ranks by
     test/parallel/capi_parallel_test.c, which also covers the > 1 case this
     single-process test cannot reach.) */
  CHECK_RC(soma_initialize(ctx), SOMA_OK,
           "initialize() refused a serial context that recorded a "
           "communicator");
  /* The lifecycle assertion, which must hold on EVERY build: swapping a
     communicator under an already-initialized context is a sequencing error.
     Reached here on both MPI and non-MPI builds precisely because
     initialize() above succeeds on both. */
  CHECK_RC(soma_set_communicator(ctx, 0), SOMA_ERR_STATE,
           "set_communicator after initialize accepted");
  soma_destroy(ctx);
}

/* ------------------------------------------------------------------------ */
/* A graph that does not validate must be reported, not run                  */
/* ------------------------------------------------------------------------ */

static void test_invalid_graph_is_reported(void) {
  /* A descriptor whose header is right but whose body is truncated. */
  soma_ctx       ctx = NULL;
  soma_ctx       src = loaded_pair();
  uint64_t       bytes = 0;
  unsigned char* blob = NULL;

  soma_save_graph(src, NULL, 0, &bytes);
  blob = (unsigned char*)malloc((size_t)bytes);
  CHECK(blob != NULL, "malloc");
  soma_save_graph(src, blob, bytes, &bytes);

  soma_create(&ctx);
  CHECK_RC(soma_load_graph(ctx, blob, bytes - 1), SOMA_ERR_INVALID,
           "a truncated descriptor was accepted");
  CHECK(message_nonempty(ctx), "a truncated descriptor left no message");

  free(blob);
  soma_destroy(ctx);
  soma_destroy(src);
}

/* ------------------------------------------------------------------------ */
/* No C++ exception may cross the boundary                                   */
/* ------------------------------------------------------------------------ */

/* soma::Rational's constructor throws when a fraction cannot be normalised
   into two int64s. Reaching that through the ABI must produce an error code,
   not std::terminate. */
static void test_exception_from_a_direct_argument(void) {
  soma_ctx ctx = NULL;
  soma_create(&ctx);
  /* -(-2^63)/1 is not representable, so Rational's overflow guard throws. */
  CHECK_RC(soma_set_end_time(ctx, INT64_MIN, -1), SOMA_ERR_INVALID,
           "a non-representable end time must be rejected, not fatal");
  CHECK(message_nonempty(ctx), "the caught exception left no message");
  CHECK_RC(soma_set_end_time(ctx, 3, 4), SOMA_OK,
           "a rejected end time must leave the context usable");
  soma_destroy(ctx);
}

/* The same guard, reached from deep inside deserialize() rather than from an
   argument this file checks directly: a hand-patched descriptor whose
   component interval is -(-2^63)/1. This is the case that would be
   std::terminate without entry()'s catch-all. */
static void test_exception_from_deep_inside_a_descriptor(void) {
  soma_ctx       src = loaded_pair();
  soma_ctx       ctx = NULL;
  uint64_t       bytes = 0;
  unsigned char* blob = NULL;
  /* uint32 length 5, then "fluid": unique to the component table, since the
     mesh "fluid_mesh" and region "fluid_wall" have length 10. */
  const unsigned char needle[9] = {5, 0, 0, 0, 'f', 'l', 'u', 'i', 'd'};
  size_t         at = 0;
  int            found = -1;
  int            i;

  soma_save_graph(src, NULL, 0, &bytes);
  blob = (unsigned char*)malloc((size_t)bytes);
  CHECK(blob != NULL, "malloc");
  soma_save_graph(src, blob, bytes, &bytes);

  for (at = 0; at + sizeof(needle) <= (size_t)bytes; ++at) {
    if (memcmp(blob + at, needle, sizeof(needle)) == 0) {
      found = (int)at;
      break;
    }
  }
  CHECK(found >= 0, "could not find the fluid component in the descriptor");
  if (found >= 0) {
    /* name(9) + ranks.first(4) + ranks.count(4) -> interval numerator. */
    unsigned char* num = blob + found + 9 + 4 + 4;
    unsigned char* den = num + 8;
    for (i = 0; i < 7; ++i) num[i] = 0x00;
    num[7] = 0x80;                              /* INT64_MIN, little-endian */
    for (i = 0; i < 8; ++i) den[i] = 0xFF;      /* -1 */

    soma_create(&ctx);
    CHECK_RC(soma_load_graph(ctx, blob, bytes), SOMA_ERR_RUNTIME,
             "an exception from deserialize() must become SOMA_ERR_RUNTIME");
    CHECK(message_nonempty(ctx),
          "a caught exception must leave its what() retrievable");
    soma_destroy(ctx);
  }

  free(blob);
  soma_destroy(src);
}

int main(void) {
  test_version();
  test_create_destroy();
  test_null_arguments_are_rejected();
  test_double_destroy_and_stale_handles();
  test_error_message_is_retrievable();
  test_error_message_truncates_safely();
  test_thread_local_error_without_a_context();
  test_yaml_load_and_handles();
  test_descriptor_round_trip();
  test_descriptor_rejects_garbage();
  test_register_field();
  test_register_field_rejects_bad_arguments();
  test_state_errors();
  test_initialize_requires_registered_fields();
  test_initialize_and_step();
  test_coupling_ongoing();
  test_iteration_checkpoint_contract();
  test_iteration_convergence_early_exit();
  test_initialize_rejects_an_empty_convergence_criterion();
  test_initialize_rejects_an_unmatched_convergence_measure();
  test_initialize_rejects_zero_max_iterations();
  test_initialize_still_accepts_a_well_formed_region();
  test_set_communicator();
  test_invalid_graph_is_reported();
  test_exception_from_a_direct_argument();
  test_exception_from_deep_inside_a_descriptor();

  if (failures == 0) {
    printf("capi_test: all checks passed\n");
    return 0;
  }
  printf("capi_test: %d failure(s)\n", failures);
  return 1;
}
