/* A provider plugin with an adjoint, for test/abi/adjoint_plugin_test.cpp: a pure-C implementation
 * of the linear step the Runner's own adjoint test uses,
 *
 *     out <- A out + C in,
 *
 * and of its adjoint (fuse_provider_abi_v1::bind_adjoint and advance_adjoint, capability bit
 * FUSE_CAP_ADJOINT). One source, built twice by CMake: the plugin of component "p" (fields out,
 * in; A = 0.9, C = 0.4) and the plugin of component "q" (fields in, out; A = 0.8, C = 0.3), the
 * constants and the field order chosen by FUSE_TEST_ADJOINT_ROLE, 0 for p and 1 for q. The plugin
 * is dlopen()ed by the participant ABI as a graph-declared `plugin` component, so the reverse
 * sweep reaches it through the same path a real plugin takes.
 *
 * bind() and bind_adjoint() come in the order the graph declares the fields: p out, in; q in, out.
 */
#include "fuse/abi/provider.h"

#include <stddef.h>
#include <stdlib.h>

#if FUSE_TEST_ADJOINT_ROLE == 0
#define FUSE_TEST_A 0.9
#define FUSE_TEST_C 0.4
#else
#define FUSE_TEST_A 0.8
#define FUSE_TEST_C 0.3
#endif

typedef struct {
  double *out, *in, *out_bar, *in_bar;
  size_t n;
  int binds, adjoint_binds;
} instance_t;

/* the position (0 or 1) of the field `out` among the two this role binds */
static int out_position(void) { return FUSE_TEST_ADJOINT_ROLE == 0 ? 0 : 1; }

static fuse_status p_version(fuse_provider_ctx* self, const fuse_null* in, fuse_version_info* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_version_info);
  out->library = (fuse_semantic_version){1, 0, 0};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = "fuse test adjoint_provider (C)";
  return FUSE_OK;
}

static fuse_status p_create(fuse_provider_ctx* self, const fuse_null* in, fuse_provider_instance* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  instance_t* inst = (instance_t*)calloc(1, sizeof(instance_t));
  if (inst == NULL) return FUSE_ERR_RUNTIME;
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = (fuse_provider_ctx*)inst;
  return FUSE_OK;
}

static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)in; (void)out;
  free((instance_t*)self);
  return FUSE_OK;
}

static void put_row(fuse_field_spec* r, const char* name, int components, int intent) {
  r->struct_size = sizeof(fuse_field_spec);
  r->name = name;
  r->components = components;
  r->intent = intent;
  r->element_type = FUSE_ELEMENT_F64;
}

static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
  out->written = 2;
  if (out->buffer != NULL && out->capacity >= 2) {
    /* one component per entity: the graph of the test declares one */
    if (FUSE_TEST_ADJOINT_ROLE == 0) {
      put_row(&out->buffer[0], "out", 1, 0);
      put_row(&out->buffer[1], "in", 1, 1);
    } else {
      put_row(&out->buffer[0], "in", 1, 1);
      put_row(&out->buffer[1], "out", 1, 0);
    }
  }
  return FUSE_OK;
}

static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || in->data == NULL) return FUSE_ERR_INVALID;
  if (inst->binds >= 2) return FUSE_ERR_INVALID;
  if (inst->binds == out_position()) inst->out = (double*)in->data; else inst->in = (double*)in->data;
  inst->n = (size_t)in->count * (size_t)in->components;
  inst->binds += 1;
  return FUSE_OK;
}

static fuse_status p_bind_adjoint(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || in->data == NULL) return FUSE_ERR_INVALID;
  if (inst->adjoint_binds >= 2) return FUSE_ERR_INVALID;
  if (inst->adjoint_binds == out_position()) inst->out_bar = (double*)in->data; else inst->in_bar = (double*)in->data;
  inst->adjoint_binds += 1;
  return FUSE_OK;
}

static void echo(const fuse_step_request* in, fuse_step_report* out) {
  out->struct_size = sizeof(fuse_step_report);
  out->requires_checkpoint = 0;
  out->requires_restore = 0;
  out->step = in->step;
  out->iteration = in->iteration;
  out->reserved = 0;
  out->t_begin = in->t_begin;
  out->t_end = in->t_end;
}

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  size_t i;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
  for (i = 0; i < inst->n; ++i) {
    inst->out[i] = FUSE_TEST_A * inst->out[i] + FUSE_TEST_C * inst->in[i];
  }
  echo(in, out);
  return FUSE_OK;
}

/* The adjoint of the step: the adjoint of `in` receives C times that of `out`, which is then scaled
 * by A (the step overwrote out, so the adjoint of its value before the step replaces it). */
static fuse_status p_advance_adjoint(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  size_t i;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
  for (i = 0; i < inst->n; ++i) {
    inst->in_bar[i] += FUSE_TEST_C * inst->out_bar[i];
    inst->out_bar[i] *= FUSE_TEST_A;
  }
  echo(in, out);
  return FUSE_OK;
}

static fuse_status p_capabilities(fuse_provider_ctx* self, const fuse_null* in, fuse_capability_set* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
  out->flags = FUSE_CAP_ADJOINT;
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1),
    FUSE_ABI_VERSION,
    p_version, p_create, p_destroy, p_describe, p_bind, p_advance,
    NULL /* save_state */, NULL /* restore_state */, p_capabilities,
    NULL /* save_restart_blob */, NULL /* load_restart_blob */,
    p_bind_adjoint, p_advance_adjoint,
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  if (abi_version != FUSE_ABI_VERSION) return NULL;
  return &g_table;
}
