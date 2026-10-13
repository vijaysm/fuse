/* A provider plugin for test/abi/conformance_test.cpp: a pure-C
 * implementation of the SAME physics the native C++ "upstream" component
 * and the Fortran plugin (test/abi/plugins/linear_provider.f90) implement,
 * so EXPECT_EQ against the native run is a real cross-language check, not a
 * tautology. Physics: the one field this provider exports, "out", is
 * written as t_end (the exact rational cast to double) at every advance() --
 * deterministic IEEE754 double division, bit-identical across a C, a
 * Fortran and a C++ translation unit on the same hardware, which is the
 * property this conformance harness needs and nothing more elaborate buys.
 *
 * Behaviours are selected at compile time, each its own CMake target
 * (test/CMakeLists.txt), so every one of build-evidence/abi-matrix.toml's
 * negative provider rows exercises a REAL dlopen()'d shared object rather
 * than a hand-built in-memory table:
 *   (default)                 the correct plugin, used for the EXPECT_EQ
 *                              cross-kind comparison.
 *   FUSE_TEST_BAD_VERSION      fuse_provider_abi() refuses FUSE_ABI_VERSION
 *                              (returns NULL), as if it only spoke v2.
 *   FUSE_TEST_BAD_DESCRIBE     describe() reports the wrong component count
 *                              for "out".
 *   FUSE_TEST_BAD_ADVANCE      advance() always returns FUSE_ERR_RUNTIME.
 *   FUSE_TEST_BAD_CREATE       create() always returns FUSE_ERR_RUNTIME
 *                              (Review 2026-10-11 TD-6: E-ABI-PROVIDER-CREATE
 *                              had no test).
 *   FUSE_TEST_BAD_BIND         bind() always returns FUSE_ERR_INVALID
 *                              (TD-6: E-ABI-PROVIDER-BIND).
 *   FUSE_TEST_BAD_WINDOW       advance() echoes a window shifted from the
 *                              one it was asked to run (TD-6:
 *                              E-ABI-PROVIDER-WINDOW).
 *   FUSE_TEST_BAD_ELEMENT      describe() reports an element type other than
 *                              FUSE_ELEMENT_F64 (TD-6: E-ABI-PROVIDER-ELEMENT).
 *   FUSE_TEST_BAD_CAPABILITIES capabilities() always returns FUSE_ERR_RUNTIME
 *                              (Review 2026-10-11 AB-6(a):
 *                              E-ABI-PROVIDER-CAPABILITIES).
 *   FUSE_TEST_BAD_DESCRIBE_EXTRA describe() reports "out" correctly PLUS an
 *                              extra field the graph never declares
 *                              (AB-6(b): the describe() superset direction
 *                              of E-ABI-DESCRIBE).
 *   FUSE_TEST_BAD_SAVE         save_state() always returns FUSE_ERR_RUNTIME
 *                              (TD-6: E-ABI-PROVIDER-SAVE).
 *   FUSE_TEST_BAD_RESTORE      restore_state() always returns
 *                              FUSE_ERR_RUNTIME (TD-6: E-ABI-PROVIDER-RESTORE).
 */
#include "fuse/abi/provider.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double* out_ptr;
} instance_t;

static fuse_status p_version(fuse_provider_ctx* self, const fuse_null* in, fuse_version_info* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_version_info);
  out->library = (fuse_semantic_version){1, 0, 0};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = "fuse test linear_provider (C)";
  return FUSE_OK;
}

static fuse_status p_create(fuse_provider_ctx* self, const fuse_null* in, fuse_provider_instance* out) {
  (void)self; (void)in;
#ifdef FUSE_TEST_BAD_CREATE
  (void)out;
  return FUSE_ERR_RUNTIME;
#else
  if (out == NULL) return FUSE_ERR_INVALID;
  instance_t* inst = (instance_t*)calloc(1, sizeof(instance_t));
  if (inst == NULL) return FUSE_ERR_RUNTIME;
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = (fuse_provider_ctx*)inst;
  return FUSE_OK;
#endif
}

static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)in; (void)out;
  free((instance_t*)self);
  return FUSE_OK;
}

static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
#if defined(FUSE_TEST_BAD_DESCRIBE_EXTRA)
  out->written = 2;  /* "out" plus a field the graph never declares */
#else
  out->written = 1;
#endif
  if (out->buffer != NULL && out->capacity >= 1) {
    out->buffer[0].struct_size = sizeof(fuse_field_spec);
    out->buffer[0].name = "out";
#ifdef FUSE_TEST_BAD_DESCRIBE
    out->buffer[0].components = 2;  /* the graph declares 1: E-ABI-DESCRIBE */
#else
    out->buffer[0].components = 1;
#endif
    out->buffer[0].intent = 0;  /* Provides */
#ifdef FUSE_TEST_BAD_ELEMENT
    out->buffer[0].element_type = FUSE_ELEMENT_I64;  /* not F64: E-ABI-PROVIDER-ELEMENT */
#else
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
#endif
  }
#if defined(FUSE_TEST_BAD_DESCRIBE_EXTRA)
  if (out->buffer != NULL && out->capacity >= 2) {
    out->buffer[1].struct_size = sizeof(fuse_field_spec);
    out->buffer[1].name = "extra_undeclared_field";
    out->buffer[1].components = 1;
    out->buffer[1].intent = 0;
    out->buffer[1].element_type = FUSE_ELEMENT_F64;
  }
#endif
  return FUSE_OK;
}

static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
#ifdef FUSE_TEST_BAD_BIND
  (void)self; (void)in;
  return FUSE_ERR_INVALID;
#else
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || in->data == NULL) return FUSE_ERR_INVALID;
  inst->out_ptr = (double*)in->data;
  return FUSE_OK;
#endif
}

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
#ifdef FUSE_TEST_BAD_ADVANCE
  (void)inst;
  return FUSE_ERR_RUNTIME;
#else
  if (inst->out_ptr != NULL) {
    *inst->out_ptr = (double)in->t_end.numerator / (double)in->t_end.denominator;
  }
#ifdef FUSE_TEST_OLD_STEP_REPORT
  /* Review 2026-10-11 AB-3: a plugin built against the pre-growth
   * fuse_step_report layout (c-abi.md's growth table, 64 bytes, before
   * next_component/step_opened/period/delivery_pending/delivery_field
   * were appended) never had those members to write, and writes the
   * OLD struct_size its own header's sizeof() gave it -- offsetof(t_end)
   * + sizeof(fuse_exact_time), the same expression
   * step_report_struct_size_ok() (src/abi/provider_adapter.cpp) checks
   * against, computed independently here rather than taken from the
   * (newer, shared) header this file also includes. */
  out->struct_size = (uint32_t)(offsetof(fuse_step_report, t_end) + sizeof(fuse_exact_time));
#else
  out->struct_size = sizeof(fuse_step_report);
#endif
  out->requires_checkpoint = 0;
  out->requires_restore = 0;
  out->step = in->step;
  out->iteration = in->iteration;
  out->reserved = 0;
  out->t_begin = in->t_begin;
#ifdef FUSE_TEST_BAD_WINDOW
  /* Echo a window whose end is one time unit past the one actually requested:
   * rule 6 requires EXACT agreement, so this is refused E-ABI-PROVIDER-WINDOW
   * rather than accepted as "close enough". */
  out->t_end.numerator = in->t_end.numerator + in->t_end.denominator;
  out->t_end.denominator = in->t_end.denominator;
#else
  out->t_end = in->t_end;
#endif
  return FUSE_OK;
#endif
}

static fuse_status p_save_state(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
#ifdef FUSE_TEST_BAD_SAVE
  (void)self; (void)in; (void)out;
  return FUSE_ERR_RUNTIME;
#else
  (void)self; (void)in; (void)out;
  return FUSE_OK;
#endif
}

static fuse_status p_restore_state(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
#ifdef FUSE_TEST_BAD_RESTORE
  (void)self; (void)in; (void)out;
  return FUSE_ERR_RUNTIME;
#else
  (void)self; (void)in; (void)out;
  return FUSE_OK;
#endif
}

static fuse_status p_capabilities(fuse_provider_ctx* self, const fuse_null* in, fuse_capability_set* out) {
  (void)self; (void)in;
#ifdef FUSE_TEST_BAD_CAPABILITIES
  (void)out;
  return FUSE_ERR_RUNTIME;
#else
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
  out->flags = 0;
  return FUSE_OK;
#endif
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1),
    FUSE_ABI_VERSION,
    p_version, p_create, p_destroy, p_describe, p_bind, p_advance,
    p_save_state, p_restore_state, p_capabilities,
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
#ifdef FUSE_TEST_BAD_VERSION
  (void)abi_version;
  return NULL;  /* refuses every version, as if it only spoke v2 */
#else
  if (abi_version != FUSE_ABI_VERSION) return NULL;
  return &g_table;
#endif
}
