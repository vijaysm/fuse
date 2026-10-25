/* A non-hosting rank must never dlopen() a LaunchMode::Plugin
 * component's shared object at all. This plugin's ONLY purpose
 * is to let a tier-0 test tell, from OUTSIDE the plugin, that NONE of
 * create()/describe()/bind()/advance()/destroy() was ever called -- the
 * direct evidence that a non-hosting rank's own ProviderAdapter never
 * reached this shared object, as distinct from reaching it but finding
 * nothing to do (which is what every rank used to do).
 *
 * A file-static counter per entry, incremented once per call, read back
 * through the one exported probe function at the bottom -- NOT part of
 * the frozen provider ABI (fuse_provider_abi_v1, include/fuse/abi/
 * provider.h, carries no such entry, and growing it to add one is
 * explicitly out of scope). A test reads the counters
 * through its OWN dlsym() on this SAME shared object -- the identical
 * mechanism fuse_provider_abi() itself uses to bootstrap -- never
 * through any FUSE ABI table.
 *
 * Physics: identical to linear_provider.c's own "out = t_end" -- this
 * file exists to count calls, not to test a different computation; no
 * FUSE_TEST_BAD_* compile-time variants, unlike linear_provider.c, since
 * this plugin's only job is being dlopen()'d (or not) and counted.
 */
#include "fuse/abi/provider.h"

#include <stddef.h>
#include <stdlib.h>

typedef struct {
  double* out_ptr;
} instance_t;

static int g_create_calls = 0;
static int g_describe_calls = 0;
static int g_bind_calls = 0;
static int g_advance_calls = 0;
static int g_destroy_calls = 0;

static fuse_status p_create(fuse_provider_ctx* self, const fuse_null* in,
                            fuse_provider_instance* out) {
  (void)self; (void)in;
  ++g_create_calls;
  if (out == NULL) return FUSE_ERR_INVALID;
  instance_t* inst = (instance_t*)calloc(1, sizeof(instance_t));
  if (inst == NULL) return FUSE_ERR_RUNTIME;
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = (fuse_provider_ctx*)inst;
  return FUSE_OK;
}

static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)in; (void)out;
  ++g_destroy_calls;
  free((instance_t*)self);
  return FUSE_OK;
}

static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in,
                              fuse_field_table* out) {
  (void)self; (void)in;
  ++g_describe_calls;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
  out->written = 1;
  if (out->buffer != NULL && out->capacity >= 1) {
    out->buffer[0].struct_size = sizeof(fuse_field_spec);
    out->buffer[0].name = "out";
    out->buffer[0].components = 1;
    out->buffer[0].intent = 0;  /* Provides */
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
  }
  return FUSE_OK;
}

static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  ++g_bind_calls;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || in->data == NULL) return FUSE_ERR_INVALID;
  inst->out_ptr = (double*)in->data;
  return FUSE_OK;
}

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in,
                             fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  ++g_advance_calls;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
  if (inst->out_ptr != NULL) {
    *inst->out_ptr = (double)in->t_end.numerator / (double)in->t_end.denominator;
  }
  out->struct_size = sizeof(fuse_step_report);
  out->requires_checkpoint = 0;
  out->requires_restore = 0;
  out->step = in->step;
  out->iteration = in->iteration;
  out->reserved = 0;
  out->t_begin = in->t_begin;
  out->t_end = in->t_end;
  return FUSE_OK;
}

static fuse_status p_save_state(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)self; (void)in; (void)out;
  return FUSE_OK;
}

static fuse_status p_restore_state(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)self; (void)in; (void)out;
  return FUSE_OK;
}

static fuse_status p_capabilities(fuse_provider_ctx* self, const fuse_null* in,
                                  fuse_capability_set* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
  out->flags = 0;
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1),
    FUSE_ABI_VERSION,
    NULL /* version */, p_create, p_destroy, p_describe, p_bind, p_advance,
    p_save_state, p_restore_state, p_capabilities,
    NULL /* save_restart_blob */, NULL /* load_restart_blob */,
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  if (abi_version != FUSE_ABI_VERSION) return NULL;
  return &g_table;
}

/* Test-only probe, NOT part of the frozen provider ABI (see this file's
 * own header comment) -- reached by a test's OWN dlsym() on this shared
 * object, never through fuse_provider_abi_v1's own table. Any argument
 * may be NULL (the caller only wants a subset of the five counts). */
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
int fuse_test_hosting_probe_calls(int* create_calls, int* describe_calls,
                                  int* bind_calls, int* advance_calls,
                                  int* destroy_calls) {
  if (create_calls != NULL) *create_calls = g_create_calls;
  if (describe_calls != NULL) *describe_calls = g_describe_calls;
  if (bind_calls != NULL) *bind_calls = g_bind_calls;
  if (advance_calls != NULL) *advance_calls = g_advance_calls;
  if (destroy_calls != NULL) *destroy_calls = g_destroy_calls;
  return 0;
}
