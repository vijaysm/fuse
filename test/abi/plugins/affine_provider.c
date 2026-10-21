/* A provider plugin for test/abi/conformance_test.cpp: a faithful, pure-C
 * reimplementation of test/abi/participant_test.cpp's own AffineComponent
 * (lines ~106-168 there), used so a mixed native/plugin iteration region
 * (KNOWN_ISSUES.md D15, "checkpoint/restore through a mixed iteration
 * region") can be compared bit-for-bit against the all-native reference
 * ImplicitCouplingConvergesLikeInProcess already measures for the SAME
 * physics (participant_test.cpp:785).
 *
 * Physics (identical to AffineComponent::advance()): this provider PROVIDES
 * one field ("b_out") and REQUIRES another ("b_in"), and at every advance()
 * computes, per entity i,
 *
 *     b_out[i] = FUSE_TEST_AFFINE_OFFSET + FUSE_TEST_AFFINE_SCALE * b_in[i]
 *
 * one multiply, one add, IEEE754 double, no accumulation-order sensitivity
 * -- the same category of deterministic expression linear_provider.c's own
 * header comment already states is "bit-identical across a C, a Fortran and
 * a C++ translation unit on the same hardware" for a simpler one-field
 * computation; -ffp-contract=off (FUSE_FP_REPRODUCIBLE, project-wide) removes
 * the one compiler freedom that could otherwise make this expression round
 * differently between this translation unit and AffineComponent's own C++
 * one.
 *
 * Field names and coefficients are fixed at compile time via macros, the
 * SAME "behaviours selected at compile time, each its own CMake target"
 * convention linear_provider.c's own header comment states (there is no
 * runtime-configuration channel create() has an argument slot for -- its
 * own `in` parameter is `fuse_null`, entries.def's own
 * FUSE_ABI_PROVIDER_ENTRY(create, fuse_null, fuse_provider_instance, ...)).
 * Defaults match test/abi/participant_test.cpp:812's own "beta" coefficients
 * exactly (scale=0.5, offset=0.0), so the default build of this plugin is a
 * drop-in replacement for the native "beta" AffineComponent in
 * build_implicit_coupling_graph()'s own two-component iteration region.
 *
 * UNLIKE linear_provider.c's own single-scalar p_advance() (which writes
 * exactly one double, correct only for the one-cell meshes its own fixtures
 * use), this plugin loops over every entity bind() told it about -- the
 * fixture this plugin serves uses a 4-cell mesh (build_implicit_coupling_
 * graph()'s own default `cells = 4`), and AffineComponent's own advance()
 * loops over the whole of `provides_`/`requires_`, not just entity 0.
 *
 * save_state()/restore_state() are REAL, not no-ops (unlike
 * linear_provider.c's own, which have nothing to checkpoint): a malloc'd
 * snapshot buffer, sized to the bound "b_out" field's own entity count,
 * copied from on save_state() and copied back to on restore_state() --
 * the exact round trip AffineComponent::save_state()/restore_state() do
 * for `provides_`/`saved_`. This is deliberately the ONE thing this
 * plugin's own checkpoint/restore test exercises: whether a provider's own
 * save/restore round-trips correctly through fuse::abi_detail::
 * ProviderAdapter and the runtime's OWN existing rollback-verification
 * check (src/runtime/iteration.cpp's header comment, "THE RUNTIME VERIFIES
 * THE ROLLBACK RATHER THAN TRUSTING IT" -- E-ITER-ROLLBACK fires on a byte-
 * for-byte mismatch after any restore_state(), whether the mismatch comes
 * from native or plugin code).
 *
 * bind() is called once per field, in the GRAPH's OWN declared field
 * order for this component -- MEASURED by reading src/abi/provider_
 * adapter.cpp's own register_fields(): its per-field loop walks
 * `graph.component(cid).fields` (the order add_field() was called in,
 * test/abi/graph_fixture.hpp's own build_implicit_coupling_graph() and
 * build_implicit_coupling_graph_with_plugin_beta()), using describe()'s
 * own rows only to VALIDATE each field by name, never to decide bind()
 * call order. This does not appear to be written down anywhere in
 * docs/user/providers.md or entries.def's own "bind" row (both are
 * silent on ORDER across more than one field) -- flagged as an open
 * question for that doc, not fixed here (Stage B-1 does not touch
 * entries.def; docs/user/providers.md's own wording is a separate,
 * narrower edit).
 *
 * fuse_field_view carries no field NAME at all, only a handle this
 * plugin never interprets, so order is the only way to tell "b_out"
 * apart from "b_in" at bind() time. Both fixtures above declare "beta"'s
 * Requires field ("b_in") before its Provides field ("b_out") --
 * describe() below reports them in THAT SAME order (row 0 = Requires,
 * row 1 = Provides) so this plugin's own two tables agree, and bind()'s
 * own internal counter relies on it exactly.
 */
#include "fuse/abi/provider.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef FUSE_TEST_AFFINE_SCALE
#define FUSE_TEST_AFFINE_SCALE 0.5
#endif
#ifndef FUSE_TEST_AFFINE_OFFSET
#define FUSE_TEST_AFFINE_OFFSET 0.0
#endif
#ifndef FUSE_TEST_AFFINE_PROVIDES_FIELD
#define FUSE_TEST_AFFINE_PROVIDES_FIELD "b_out"
#endif
#ifndef FUSE_TEST_AFFINE_REQUIRES_FIELD
#define FUSE_TEST_AFFINE_REQUIRES_FIELD "b_in"
#endif

typedef struct {
  double* out_ptr;      /* bound storage for FUSE_TEST_AFFINE_PROVIDES_FIELD */
  size_t  out_count;    /* entities in out_ptr; 0 until bind() runs */
  double* in_ptr;       /* bound storage for FUSE_TEST_AFFINE_REQUIRES_FIELD */
  size_t  in_count;      /* entities in in_ptr; 0 until bind() runs */
  int     bind_calls;   /* 0 before any bind(); counts which row's turn it is */
  double* saved;        /* malloc'd snapshot of out_ptr, NULL until save_state() */
  size_t  saved_count;  /* 0 until save_state() has actually captured something */
} instance_t;

static fuse_status p_version(fuse_provider_ctx* self, const fuse_null* in, fuse_version_info* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_version_info);
  out->library = (fuse_semantic_version){1, 0, 0};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = "fuse test affine_provider (C)";
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
  instance_t* inst = (instance_t*)self;
  if (inst != NULL) {
    free(inst->saved);
  }
  free(inst);
  return FUSE_OK;
}

static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
  out->written = 2;
  /* Row order matches bind()'s own expected call order -- the GRAPH's
   * declared field order for "beta" (Requires field first, Provides
   * field second; see this file's own header comment above) -- not the
   * Provides-then-Requires order a cleaner-looking table might suggest. */
  if (out->buffer != NULL && out->capacity >= 1) {
    out->buffer[0].struct_size  = sizeof(fuse_field_spec);
    out->buffer[0].name         = FUSE_TEST_AFFINE_REQUIRES_FIELD;
    out->buffer[0].components   = 1;
    out->buffer[0].intent       = 1; /* Requires */
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
  }
  if (out->buffer != NULL && out->capacity >= 2) {
    out->buffer[1].struct_size  = sizeof(fuse_field_spec);
    out->buffer[1].name         = FUSE_TEST_AFFINE_PROVIDES_FIELD;
    out->buffer[1].components   = 1;
    out->buffer[1].intent       = 0; /* Provides */
    out->buffer[1].element_type = FUSE_ELEMENT_F64;
  }
  return FUSE_OK;
}

static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || in->data == NULL) return FUSE_ERR_INVALID;
  /* Row 0 is the Requires field, row 1 the Provides field -- the GRAPH's
   * own declared order for "beta" (this file's own header comment), not
   * describe()'s row order (which happens to agree here, by construction
   * above, but is not what bind() call order actually follows). */
  if (inst->bind_calls == 0) {
    inst->in_ptr   = (double*)in->data;
    inst->in_count = (size_t)in->count;
  } else if (inst->bind_calls == 1) {
    inst->out_ptr   = (double*)in->data;
    inst->out_count = (size_t)in->count;
  } else {
    /* describe() above reports exactly two fields; a third bind() call
     * would mean ProviderAdapter and this plugin have disagreed about how
     * many fields describe() declared -- refuse rather than silently
     * ignoring the extra call. */
    return FUSE_ERR_INVALID;
  }
  inst->bind_calls += 1;
  return FUSE_OK;
}

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  if (inst == NULL || in == NULL || out == NULL) return FUSE_ERR_INVALID;
  if (inst->out_ptr != NULL && inst->in_ptr != NULL) {
    /* AffineComponent::advance()'s own loop, reproduced exactly: one
     * multiply, one add, per entity, in increasing index order -- no
     * reduction, so no order-of-operations question beyond this one line
     * (abi-design.md's own "deterministic IEEE754" posture, linear_
     * provider.c's own header comment states the measured property for a
     * simpler expression; FUSE_FP_REPRODUCIBLE's -ffp-contract=off removes
     * the one compiler freedom that could make this one differ by ULP
     * between this translation unit and AffineComponent's own). */
    size_t n = inst->out_count < inst->in_count ? inst->out_count : inst->in_count;
    for (size_t i = 0; i < n; ++i) {
      inst->out_ptr[i] = (double)FUSE_TEST_AFFINE_OFFSET +
                         (double)FUSE_TEST_AFFINE_SCALE * inst->in_ptr[i];
    }
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
  (void)in; (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL) return FUSE_ERR_INVALID;
  if (inst->out_ptr == NULL) {
    /* Nothing bound yet (e.g. this rank does not host the field): nothing
     * to snapshot, and nothing for restore_state() to need either -- the
     * same "not hosted" shape ProviderAdapter's own count==0 path already
     * handles for bind() itself. */
    return FUSE_OK;
  }
  if (inst->saved == NULL || inst->saved_count != inst->out_count) {
    free(inst->saved);
    inst->saved = (double*)malloc(inst->out_count * sizeof(double));
    if (inst->saved == NULL && inst->out_count > 0) {
      inst->saved_count = 0;
      return FUSE_ERR_RUNTIME;
    }
    inst->saved_count = inst->out_count;
    /* Round 2 (KNOWN_ISSUES.md D15, Stage B-1): poison the freshly
     * allocated buffer with a quiet NaN BEFORE the real capture below,
     * rather than leaving it at whatever malloc() happens to hand back.
     * Measured round 1: a provider-parity mutation that skips the
     * memcpy just below survived undetected, because this fixture's
     * own "beta" provides field is genuinely 0.0 at the one checkpoint
     * this region takes (its pristine pre-advance value), and a fresh
     * malloc() on this platform happened to also read back as zero --
     * a coincidence of allocator behaviour and this fixture's own
     * initial conditions, not a property of the code under test. A
     * quiet NaN's bit pattern can never equal ANY finite value
     * offset + scale * b_in produces (FUSE_TEST_AFFINE_OFFSET/_SCALE
     * are both finite, and b_in is never NaN in this fixture), so the
     * runtime's own byte-level rollback check (src/runtime/
     * iteration.cpp, "THE RUNTIME VERIFIES THE ROLLBACK RATHER THAN
     * TRUSTING IT") cannot mistake a skipped capture for a correct one
     * ever again, regardless of what any given allocator or platform
     * happens to hand back. Written once, right after allocation, not
     * on every save_state() call: the real memcpy below overwrites it
     * immediately in CORRECT code, so this changes nothing observable
     * when save_state() actually captures -- only a mutation that skips
     * that capture ever sees the poison instead of a real value. */
    for (size_t i = 0; i < inst->saved_count; ++i) {
      inst->saved[i] = NAN;
    }
  }
  if (inst->out_count > 0) {
    memcpy(inst->saved, inst->out_ptr, inst->out_count * sizeof(double));
  }
  return FUSE_OK;
}

static fuse_status p_restore_state(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)in; (void)out;
  instance_t* inst = (instance_t*)self;
  if (inst == NULL) return FUSE_ERR_INVALID;
  if (inst->out_ptr == NULL) return FUSE_OK;  /* not hosted; nothing to restore */
  if (inst->saved == NULL || inst->saved_count != inst->out_count) {
    /* restore_state() called with no matching save_state() -- mirrors
     * AffineComponent::restore_state()'s own named refusal ("E-TEST-
     * RESTORE") for the identical condition; the provider ABI has no
     * per-call message channel, so the status code alone is this
     * plugin's own signal -- fuse::abi_detail::ProviderAdapter's own
     * wrapper names it E-ABI-PROVIDER-RESTORE. */
    return FUSE_ERR_RUNTIME;
  }
  if (inst->out_count > 0) {
    memcpy(inst->out_ptr, inst->saved, inst->out_count * sizeof(double));
  }
  return FUSE_OK;
}

static fuse_status p_capabilities(fuse_provider_ctx* self, const fuse_null* in, fuse_capability_set* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
  /* FUSE_CAP_CHECKPOINT set, nothing else: entries.def's own doc on this
   * bit ("capabilities() means save_state/restore_state are real, not
   * the default no-op") is exactly what this plugin provides -- a real
   * malloc()'d snapshot round trip over its own Provides output, the
   * same contract AffineComponent::capabilities() (participant_test.cpp,
   * conformance_test.cpp) declares through fuse::make_capabilities()'s
   * rollback bit. src/abi/provider_adapter.cpp's own capabilities()
   * wrapper maps FUSE_CAP_CHECKPOINT to fuse::kCapabilityRollback --
   * without it, a component re-advanced on every sweep of an iteration
   * region is refused outright (E-ITER-NOROLLBACK, src/runtime/
   * iteration.cpp) before a single sweep runs, since the runtime cannot
   * tell a provider that merely forgot to declare this from one that
   * genuinely has no save_state()/restore_state() to call.
   * No FUSE_CAP_DEVICE_FIELDS (this plugin binds fuse::MemorySpace::Host
   * storage only -- Stage B-1 builds item 2 only, not item 1), no
   * FUSE_CAP_RESTART_FORM_B (save_restart_blob()/load_restart_blob()
   * below are the documented no-op default, not a real implementation). */
  out->flags = FUSE_CAP_CHECKPOINT;
  return FUSE_OK;
}

static fuse_status p_save_restart_blob(fuse_provider_ctx* self, const fuse_null* in,
                                       fuse_restart_blob_sink* out) {
  (void)self; (void)in;
  if (out == NULL) return FUSE_ERR_INVALID;
  out->struct_size  = sizeof(fuse_restart_blob_sink);
  out->written      = 0;
  out->blob_version = 0;
  return FUSE_OK;  /* "nothing to capture" -- entries.def's own documented
                    * default for a provider that does not implement this,
                    * stated explicitly rather than left to a NULL entry
                    * to avoid any struct_size-of-the-table ambiguity. */
}

static fuse_status p_load_restart_blob(fuse_provider_ctx* self,
                                       const fuse_restart_blob_source* in,
                                       fuse_null* out) {
  (void)self; (void)out;
  if (in == NULL) return FUSE_ERR_INVALID;
  return FUSE_OK;  /* nothing was ever saved, so nothing to reject either */
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1),
    FUSE_ABI_VERSION,
    p_version, p_create, p_destroy, p_describe, p_bind, p_advance,
    p_save_state, p_restore_state, p_capabilities,
    p_save_restart_blob, p_load_restart_blob,
};

#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  if (abi_version != FUSE_ABI_VERSION) return NULL;
  return &g_table;
}
