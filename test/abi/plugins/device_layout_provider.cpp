/* A provider plugin for test/abi/device_layout_test.cpp (D15 item 1,
 * "Device layout" track, 2026-10-21): unlike test/abi/plugins/
 * mixed_memspace_provider.c, this plugin genuinely launches a Kokkos
 * kernel to touch a device-resident field instead of dereferencing the
 * pointer with an ordinary C store. That distinction matters only on a
 * real CUDA/HIP/SYCL backend: a plain store through a device pointer is
 * undefined behaviour there (measured directly on this track: the
 * EXISTING DISABLED_MixedHostAndDeviceFieldsOnOneProviderConverge-
 * ThroughChannels row segfaults inside mixed_memspace_provider.c's own
 * advance() on ssh:higgs, exit code 139, because that plugin's own
 * header comment already names the gap -- "a build with a REAL device
 * backend ... would need a device kernel launch to touch that pointer
 * instead"). This file is that device kernel launch, so it is C++, not
 * C: Kokkos has no C binding. include/fuse/abi/provider.h already
 * wraps its declarations in `#ifdef __cplusplus extern "C"`, so a
 * plugin written in C++ speaks the exact same ABI table as every other
 * plugin in this directory; nothing about the dlopen()/fuse_provider_abi
 * mechanism changes.
 *
 * Three fields on one component ("mixed"), to exercise both the layout
 * question (3a) and both copy directions (3b) with one provider:
 *
 *   "device_out" (Provides, components = kComponents > 1): advance()
 *   writes an injective function of (entity, component),
 *   f(e, c) = 1000*(e+1) + (c+1), through the device pointer bind() was
 *   given -- using fuse_field_view's OWN reported stride/component_stride
 *   rather than assuming array-of-structures, so a Runner-side stride or
 *   AoS/SoA mix-up changes what lands at each (entity, component) and the
 *   test's own exact per-element assertion catches it. This is the
 *   device -> host copy direction too: the consumer
 *   (device_layout_test.cpp's Sink) registers its own field in host
 *   memory, so the Runner must cross the boundary delivering it.
 *
 *   "device_in" (Requires, components = 1): fed by a host-registered
 *   producer's own Provides field through a channel -- the host -> device
 *   copy direction. advance() reads it back with Kokkos::deep_copy (never
 *   a raw host dereference of the device pointer) and copies the value,
 *   unchanged, into:
 *
 *   "echo_out" (Provides, components = 1, ALWAYS host memory regardless
 *   of the FUSE_TEST_DEVICE_LAYOUT_FORCE_HOST build below): a plain
 *   host -> host echo, so the test can compare what arrived at
 *   "device_in" against what the host producer sent, through the SAME
 *   channel-delivery machinery every other field in this tree is
 *   verified through, rather than an out-of-band accessor.
 *
 * capabilities() reports FUSE_CAP_DEVICE_FIELDS, required before
 * ProviderAdapter will honour "device_out"/"device_in"'s own describe()
 * rows requesting FUSE_MEMSPACE_DEVICE at all.
 *
 * FUSE_TEST_DEVICE_LAYOUT_FORCE_HOST (its own CMake target, the same
 * "behaviours selected at compile time, each its own CMake target"
 * convention test/abi/plugins/linear_provider.c's own header comment
 * states): describe() reports FUSE_MEMSPACE_HOST for "device_out" and
 * "device_in" instead, and the write/read kernels below run on
 * Kokkos::DefaultHostExecutionSpace explicitly rather than
 * Kokkos::DefaultExecutionSpace -- an all-host run of the IDENTICAL
 * kernel code and the IDENTICAL arithmetic, used as
 * device_layout_test.cpp's own reference run (never a hand-recomputed
 * expected value) to compare the real device run's delivered values
 * against, bit for bit.
 */
#include "fuse/abi/provider.h"

#include <Kokkos_Core.hpp>

#include <cstdint>
#include <new>

namespace {

#if defined(FUSE_TEST_DEVICE_LAYOUT_FORCE_HOST)
using KernelExecSpace = Kokkos::DefaultHostExecutionSpace;
constexpr int32_t kMemspace = FUSE_MEMSPACE_HOST;
#else
using KernelExecSpace = Kokkos::DefaultExecutionSpace;
constexpr int32_t kMemspace = FUSE_MEMSPACE_DEVICE;
#endif

constexpr int32_t kComponents = 3;  // > 1: exercises the AoS stride, not just a scalar.

/// What bind() recorded for one field: the raw pointer plus the view's OWN
/// reported layout, read back at advance() time rather than assumed --
/// this is the fuse_field_view the Runner actually handed us, the same
/// struct device_layout_test.cpp's own accessor (below) reports to the
/// test, so "what the plugin wrote through" and "what the test asserts
/// fuse_field_view said" are provably the same view, not two guesses that
/// happen to agree.
struct FieldSlot {
  double* data = nullptr;
  int64_t stride = 0;
  int64_t component_stride = 0;
  uint64_t count = 0;
  int32_t components = 0;
  int32_t memspace = FUSE_MEMSPACE_HOST;
  bool bound = false;  // false until bind() runs; also false for count==0 (D24: not hosted here)
};

struct instance_t {
  FieldSlot device_out;
  FieldSlot device_in;
  FieldSlot echo_out;
  int bind_calls = 0;
};

/// Exposed to the test via dlsym() on this same shared object (the test
/// holds the dlopen() handle fuse::abi_detail::Registry::mount_dynamic()
/// already opened for the graph-declared plugin path, the same handle
/// test/abi/conformance_test.cpp's own Table() wraps for the well-known
/// `fuse_provider_abi` symbol) -- a second instance_t pointer would be
/// private to this translation unit otherwise, with no ABI-sanctioned way
/// for a test to reach it. One instance only: every test in
/// device_layout_test.cpp creates exactly one "mixed" component.
instance_t* g_last_instance = nullptr;

}  // namespace

extern "C" {

static fuse_status p_version(fuse_provider_ctx*, const fuse_null*, fuse_version_info* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_version_info);
  out->library = fuse_semantic_version{1, 0, 0};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = "fuse test device_layout_provider (C++/Kokkos)";
  return FUSE_OK;
}

static fuse_status p_create(fuse_provider_ctx*, const fuse_null*, fuse_provider_instance* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  instance_t* inst = new (std::nothrow) instance_t();
  if (inst == nullptr) return FUSE_ERR_RUNTIME;
  g_last_instance = inst;
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = reinterpret_cast<fuse_provider_ctx*>(inst);
  return FUSE_OK;
}

static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null*, fuse_null*) {
  instance_t* inst = reinterpret_cast<instance_t*>(self);
  if (inst == g_last_instance) g_last_instance = nullptr;
  delete inst;
  return FUSE_OK;
}

static fuse_status p_describe(fuse_provider_ctx*, const fuse_null*, fuse_field_table* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_field_table);
  out->written = 3;
  if (out->buffer != nullptr && out->capacity >= 3) {
    out->buffer[0] = fuse_field_spec{sizeof(fuse_field_spec), "device_out", kComponents,
                                      /*intent=*/0 /* Provides */, FUSE_ELEMENT_F64, kMemspace, 0};
    out->buffer[1] = fuse_field_spec{sizeof(fuse_field_spec), "device_in", 1,
                                      /*intent=*/1 /* Requires */, FUSE_ELEMENT_F64, kMemspace, 0};
    out->buffer[2] = fuse_field_spec{sizeof(fuse_field_spec), "echo_out", 1,
                                      /*intent=*/0 /* Provides */, FUSE_ELEMENT_F64,
                                      FUSE_MEMSPACE_HOST, 0};
  }
  return FUSE_OK;
}

/// bind() call order mirrors describe()'s own row order (mixed_memspace_
/// provider.c's own header comment measured this directly for
/// ProviderAdapter::register_fields(); unchanged here): device_out, then
/// device_in, then echo_out.
static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null*) {
  instance_t* inst = reinterpret_cast<instance_t*>(self);
  if (inst == nullptr || in == nullptr) return FUSE_ERR_INVALID;
  if (in->count > 0 && in->data == nullptr) return FUSE_ERR_INVALID;
  FieldSlot* slot = nullptr;
  switch (inst->bind_calls) {
    case 0: slot = &inst->device_out; break;
    case 1: slot = &inst->device_in; break;
    case 2: slot = &inst->echo_out; break;
    default: return FUSE_ERR_INVALID;
  }
  ++inst->bind_calls;
  slot->data = static_cast<double*>(in->data);
  slot->stride = in->stride;
  slot->component_stride = in->component_stride;
  slot->count = in->count;
  slot->components = in->components;
  slot->memspace = in->memspace;
  slot->bound = in->count > 0;  // D24/round5: count==0 means not hosted on this rank
  return FUSE_OK;
}

static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in,
                             fuse_step_report* out) {
  instance_t* inst = reinterpret_cast<instance_t*>(self);
  if (inst == nullptr || in == nullptr || out == nullptr) return FUSE_ERR_INVALID;

  if (inst->device_out.bound) {
    double* base = inst->device_out.data;
    const int64_t stride = inst->device_out.stride;
    const int64_t cstride = inst->device_out.component_stride;
    const int32_t ncomp = inst->device_out.components;
    const int64_t n = static_cast<int64_t>(inst->device_out.count);
    // Writes through the VIEW's OWN reported stride/component_stride, not
    // an assumed array-of-structures layout -- a Runner-side stride or
    // AoS/SoA disagreement changes what lands at (e, c), which the test's
    // own exact per-(entity, component) assertion then catches.
    Kokkos::parallel_for(
        "device_layout_provider.write",
        Kokkos::RangePolicy<KernelExecSpace>(0, n),
        KOKKOS_LAMBDA(int64_t e) {
          for (int32_t c = 0; c < ncomp; ++c) {
            base[e * stride + c * cstride] =
                1000.0 * static_cast<double>(e + 1) + static_cast<double>(c + 1);
          }
        });
    Kokkos::fence("device_layout_provider.write_fence");
  }

  if (inst->device_in.bound && inst->echo_out.bound) {
    const int64_t n = static_cast<int64_t>(inst->device_in.count);
    // Kokkos::deep_copy, never a raw host dereference of a device
    // pointer: that raw-dereference mistake is exactly what the existing
    // DISABLED_MixedHostAndDeviceFieldsOnOneProviderConvergeThrough-
    // Channels row's own plugin makes, and it segfaults for it on a real
    // device backend (host_pending in this track's own report).
    using UnmanagedDevice =
        Kokkos::View<const double*, typename KernelExecSpace::memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UnmanagedHost =
        Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Kokkos::View<double*, Kokkos::HostSpace> mirror("device_layout_provider.echo_mirror", n);
    const UnmanagedDevice src(inst->device_in.data, n);
    Kokkos::deep_copy(mirror, src);  // blocking; the fence above already covers device_out
    double* echo_base = inst->echo_out.data;
    const int64_t echo_stride = inst->echo_out.stride;
    for (int64_t e = 0; e < n; ++e) {
      echo_base[e * echo_stride] = mirror(static_cast<std::size_t>(e));
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

static fuse_status p_capabilities(fuse_provider_ctx*, const fuse_null*, fuse_capability_set* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  out->struct_size = sizeof(fuse_capability_set);
  out->flags = FUSE_CAP_DEVICE_FIELDS;
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1), FUSE_ABI_VERSION,
    p_version, p_create, p_destroy, p_describe, p_bind, p_advance,
    nullptr /* save_state */, nullptr /* restore_state */, p_capabilities,
    nullptr /* save_restart_blob */, nullptr /* load_restart_blob */,
};

const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  return abi_version == FUSE_ABI_VERSION ? &g_table : nullptr;
}

/// Test-only accessor, NOT part of the fuse_provider_abi_v1 table: the
/// test dlsym()s this symbol on the SAME shared object (it already holds
/// the handle from mounting the graph-declared plugin) to read back the
/// fuse_field_view "device_out"'s own bind() call actually received, and
/// asserts it against an independently-derived expected layout --
/// ProviderAdapter::register_fields()'s own allocation convention
/// (stride = components, component_stride = 1, count = the region's
/// entity count), not a value read off a run and pasted in.
fuse_field_view fuse_test_device_layout_provider_last_device_out_view(void) {
  fuse_field_view v{sizeof(fuse_field_view)};
  if (g_last_instance != nullptr) {
    const FieldSlot& s = g_last_instance->device_out;
    v.handle = 0;
    v.element_type = FUSE_ELEMENT_F64;
    v.memspace = s.memspace;
    v.ownership = FUSE_OWNERSHIP_OWNED;
    v.components = s.components;
    v.count = s.count;
    v.owned_count = s.count;
    v.stride = s.stride;
    v.component_stride = s.component_stride;
    v.data = s.data;
    v.window_samples = 0;
  }
  return v;
}

}  // extern "C"
