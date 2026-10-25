/* A provider plugin for test/abi/conformance_test.cpp. Declares TWO Provides fields --
 * "host_out" (FUSE_MEMSPACE_HOST) and "device_out" (FUSE_MEMSPACE_DEVICE)
 * -- on the SAME provider instance, to exercise the mixed host/device
 * path fuse::abi_detail::ProviderAdapter::register_fields() supports:
 * each field's own describe() row decides its own memory space
 * independently, with no uniform, provider-wide choice.
 *
 * Physics: identical to test/abi/plugins/linear_provider.c's own single
 * field, applied to BOTH fields -- advance() writes t_end (the exact
 * rational cast to double) through each pointer it was bind()-bound. This
 * is deliberately the SAME deterministic, cross-language-stable formula
 * linear_provider.c already uses, so a conformance test can assert BOTH
 * fields hold the identical, predictable value after advance() with no
 * new arithmetic to reason about.
 *
 * ON A TIER-0 BUILD (no CUDA/HIP/SYCL compiled into Kokkos):
 * Kokkos::DefaultExecutionSpace::memory_space legally equals
 * Kokkos::HostSpace (device-resident-arcs.md's own documented "Device mode
 * on a CPU-only build" mechanism, the SAME one runtime.runner_device/
 * backend.supermesh.device's own ctest twins already rely on). ON A REAL
 * DEVICE BUILD, the pointer bind() hands this plugin for "device_out" is
 * a genuine accelerator address: it is written through a Kokkos kernel
 * launch, not an ordinary C store (mixed_memspace_device_write.cpp, this
 * same directory -- Kokkos has no C binding, so that one function is its
 * own C++ translation unit, exported as a plain C-callable function and
 * compiled into this plugin's shared object alongside this file; see that
 * file's own header comment for why the fence this write relies on lives
 * in src/abi/provider_adapter.cpp and not here). Measured on ssh:higgs
 * (GTX 1050 Ti, sm_61): an earlier version of this file's own advance()
 * (an ordinary C store through the device pointer) segfaulted on that
 * hardware (exit code 139), because that store is valid only on tier 0,
 * where the pointer is secretly host-dereferenceable. What this plugin
 * now proves,
 * on either tier 0 or a real device build: the describe()/capabilities()/
 * bind()/fence MACHINERY around a device-resident field is wired
 * correctly end to end, AND the field is actually touched through a
 * kernel launch appropriate to wherever it really resides.
 *
 * capabilities() reports FUSE_CAP_DEVICE_FIELDS, required before
 * ProviderAdapter will honour "device_out"'s own describe()-row request
 * for FUSE_MEMSPACE_DEVICE at all (the capability gate).
 *
 * FUSE_TEST_NO_DEVICE_CAPABILITY (its own CMake target, mirroring
 * linear_provider.c's own "behaviours selected at compile time" table):
 * describe() still requests FUSE_MEMSPACE_DEVICE for "device_out", but
 * capabilities() no longer sets FUSE_CAP_DEVICE_FIELDS -- an internally
 * inconsistent plugin ProviderAdapter::register_fields() must refuse by
 * name (E-ABI-PROVIDER-DEVICE-UNDECLARED) before allocating anything.
 *
 * FUSE_TEST_HOST_ONLY_FIELDS (its own CMake target): describe() reports
 * ONLY "host_out" (written=1, "device_out" dropped), while capabilities()
 * STILL sets FUSE_CAP_DEVICE_FIELDS -- the opposite direction (bit set, no
 * field asks, is accepted): a provider is always free to offer more than any
 * one graph uses, so this must register successfully with no refusal at all.
 */
#include "fuse/abi/provider.h"

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  double* host_ptr;
  double* device_ptr;
} instance_t;

static fuse_status p_version(fuse_provider_ctx* self, const fuse_null* in, fuse_version_info* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_version_info);
  out->library = (fuse_semantic_version){1, 0, 0};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = "fuse test mixed_memspace_provider (C)";
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
  free(self);
  return FUSE_OK;
}

static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
#ifdef FUSE_TEST_HOST_ONLY_FIELDS
  out->written = 1;
#else
  out->written = 2;
#endif
  if (out->buffer != NULL && out->capacity >= 1) {
    out->buffer[0].struct_size  = sizeof(fuse_field_spec);
    out->buffer[0].name         = "host_out";
    out->buffer[0].components   = 1;
    out->buffer[0].intent       = 0; /* Provides */
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
    out->buffer[0].memspace     = FUSE_MEMSPACE_HOST;
    out->buffer[0].reserved     = 0;
  }
#ifndef FUSE_TEST_HOST_ONLY_FIELDS
  if (out->buffer != NULL && out->capacity >= 2) {
    out->buffer[1].struct_size  = sizeof(fuse_field_spec);
    out->buffer[1].name         = "device_out";
    out->buffer[1].components   = 1;
    out->buffer[1].intent       = 0; /* Provides */
    out->buffer[1].element_type = FUSE_ELEMENT_F64;
    out->buffer[1].memspace     = FUSE_MEMSPACE_DEVICE;
    out->buffer[1].reserved     = 0;
  }
#endif
  return FUSE_OK;
}

/* bind() is called once per field, in the GRAPH's own declared field
 * order for this component (src/abi/provider_adapter.cpp's own
 * register_fields(), measured directly -- see test/abi/plugins/
 * affine_provider.c's own header comment for the full account of why
 * this is bind() call order, not describe()'s row order). The graph
 * fixture this plugin serves (test/abi/graph_fixture.hpp's own
 * build_mixed_memspace_graph()) declares
 * "host_out" before "device_out" -- matching describe()'s own row order
 * above, so this plugin's own bind_calls counter and describe() table
 * agree, the same way affine_provider.c's own two fields do. */
static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL) return FUSE_ERR_INVALID;
  if (in->count > 0 && in->data == NULL) return FUSE_ERR_INVALID;
  if (inst->host_ptr == NULL && inst->device_ptr == NULL) {
    inst->host_ptr = (double*)in->data;
  } else if (inst->device_ptr == NULL) {
    inst->device_ptr = (double*)in->data;
    if (inst->device_ptr == NULL) inst->device_ptr = (double*)(void*)1; /* sentinel: "bound, empty" */
  } else {
    return FUSE_ERR_INVALID;
  }
  return FUSE_OK;
}

/* Declared here, defined in mixed_memspace_device_write.cpp (this same
 * directory): Kokkos has no C binding, so the one line that must launch
 * a Kokkos kernel cannot live in this C translation unit. Plain
 * `extern` is enough for C linkage (C itself has no name mangling to
 * disambiguate); the defining file exports it as `extern "C"` so both
 * sides agree on the symbol name. */
extern void fuse_test_mixed_memspace_write_device(double* ptr, double value);

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
  const double t_end = (double)in->t_end.numerator / (double)in->t_end.denominator;
  if (inst->host_ptr != NULL) *inst->host_ptr = t_end;
  /* See this file's own header comment and mixed_memspace_device_
   * write.cpp's own: this pointer may be a genuine accelerator address,
   * so it is written through a Kokkos kernel launch, never an ordinary
   * C store. The sentinel value p_bind() may have stored for an
   * UNBOUND (count==0) device field is never dereferenced. */
  if (inst->device_ptr != NULL && inst->device_ptr != (double*)(void*)1) {
    fuse_test_mixed_memspace_write_device(inst->device_ptr, t_end);
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

static fuse_status p_capabilities(fuse_provider_ctx* self, const fuse_null* in, fuse_capability_set* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
#ifdef FUSE_TEST_NO_DEVICE_CAPABILITY
  out->flags = 0;  /* deliberately inconsistent with describe()'s own
                     * "device_out" / FUSE_MEMSPACE_DEVICE row above --
                     * this CMake target exists to make ProviderAdapter's
                     * own capability gate refuse it by name. */
#else
  out->flags = FUSE_CAP_DEVICE_FIELDS;
#endif
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1), FUSE_ABI_VERSION,
    p_version, p_create, p_destroy, p_describe, p_bind, p_advance,
    NULL /* save_state */, NULL /* restore_state */, p_capabilities,
    NULL /* save_restart_blob */, NULL /* load_restart_blob */,
};

const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  return abi_version == FUSE_ABI_VERSION ? &g_table : NULL;
}
