/// Implementation of FUSE's participant C ABI (docs/dev/abi-design.md,
/// include/fuse/abi/{entries.def,types.h,participant.h}). See context.hpp's
/// header comment for how begin_step()/end_step() drive the real
/// fuse::Runner without any change to it.
///
/// Two rules govern every function in this file, the same two
/// upstream/soma/src/capi/soma_c.cpp states for its own ABI and for the same
/// reasons:
///
///   1. No C++ exception escapes. `with_ctx()` below supplies the catch-all
///      once, for every entry that takes a context; version()/capabilities()/
///      create() wrap their own bodies the same way.
///   2. No handle is dereferenced before it is proven live: `is_live()` is a
///      lookup in a table of pointers, never a read through the pointer
///      itself.

#include "fuse/abi/participant.h"

#include "fuse/base/config.hpp"

#include <cstddef>
#include <cstring>
#include <mutex>
#include <new>
#include <span>
#include <string>
#include <unordered_set>

#include <Kokkos_Core.hpp>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/version.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/component.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/field_spec.hpp"
#include "fuse/model/validate.hpp"
#include "fuse/model/yaml_io.hpp"
#include "fuse/schedule/plan.hpp"

#include "context.hpp"
#include "external_step_component.hpp"

namespace {

using fuse::abi_detail::ExternalStepComponent;
using fuse::abi_detail::ExternalState;

// ----------------------------------------------------------- struct_size

/// abi-design.md rule 2. `T`'s CURRENT (and, so far, only) frozen layout is
/// exactly `sizeof(T)`: every argument struct in entries.def has shipped one
/// version so far, so "smaller than the oldest known layout" and "smaller
/// than sizeof(T)" are the same test today. A second frozen layout would
/// change this to a per-type minimum, not this function's shape.
template <class T>
[[nodiscard]] fuse_status check_struct_size(const T* s) noexcept {
  if (s == nullptr) return FUSE_ERR_INVALID;
  if (s->struct_size < sizeof(T) ||
      s->struct_size > static_cast<std::uint32_t>(FUSE_ABI_MAX_STRUCT_SIZE)) {
    return FUSE_ERR_VERSION;
  }
  return FUSE_OK;
}

// ----------------------------------------------------- handle validity

/// The set of live contexts (upstream/soma/src/capi/soma_c.cpp's
/// HandleTable, same reasoning: constructed on first use in static storage
/// and never destroyed, so a global object whose destructor calls
/// fuse_destroy() after this translation unit's static storage would
/// otherwise have begun tearing down still validates against a live table).
struct HandleTable {
  std::mutex                      mu;
  std::unordered_set<const void*> live;
};

alignas(HandleTable) unsigned char g_table_storage[sizeof(HandleTable)];

HandleTable& table() {
  static HandleTable* t =
      new (static_cast<void*>(g_table_storage)) HandleTable();
  return *t;
}

[[nodiscard]] bool is_live(fuse_ctx* ctx) {
  if (ctx == nullptr) return false;
  HandleTable&                 t = table();
  std::lock_guard<std::mutex>  lock(t.mu);
  return t.live.contains(static_cast<const void*>(ctx));
}

/// The calling thread's most recent error, for fuse_last_error(NULL, ...)
/// and for a failed fuse_create() (which has no context to store into).
std::string& thread_error_code() {
  static thread_local std::string code;
  return code;
}
std::string& thread_error_message() {
  static thread_local std::string msg;
  return msg;
}

void store_error(fuse_ctx* ctx, const std::string& code,
                 const std::string& msg) noexcept {
  try {
    thread_error_code()    = code;
    thread_error_message() = msg;
  } catch (...) {  // NOLINT(bugprone-empty-catch): nothing better is possible
  }
  if (ctx != nullptr) {
    try {
      ctx->last_error_code    = code;
      ctx->last_error_message = msg;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }
}

fuse_status fail(fuse_ctx* ctx, fuse_status code, const std::string& fcode,
                 const std::string& msg) noexcept {
  store_error(ctx, fcode, msg);
  return code;
}

/// The single shape every context-taking entry point has: validate the
/// handle without dereferencing it, then run `body` under a catch-all that
/// also recognises fuse::Error (so its E-code reaches last_error(), not just
/// its what()).
template <class F>
fuse_status with_ctx(fuse_ctx* ctx, F&& body) noexcept {
  bool live = false;
  try {
    live = is_live(ctx);
  } catch (const std::exception& e) {
    return fail(nullptr, FUSE_ERR_RUNTIME, "E-ABI-LOCK",
               std::string("fuse: handle table lock failed: ") + e.what());
  }
  if (!live) {
    return fail(nullptr, FUSE_ERR_INVALID, "E-ABI-HANDLE",
               "fuse: invalid, null, or already-destroyed fuse_ctx handle");
  }
  try {
    return body(*ctx);
  } catch (const fuse::Error& e) {
    return fail(ctx, FUSE_ERR_RUNTIME, std::string(e.code()),
               std::string(e.what()));
  } catch (const std::exception& e) {
    return fail(ctx, FUSE_ERR_RUNTIME, "E-ABI-EXCEPTION", e.what());
  } catch (...) {
    return fail(ctx, FUSE_ERR_RUNTIME, "E-ABI-EXCEPTION",
               "fuse: unknown non-std exception at the C ABI boundary");
  }
}

// ------------------------------------------------------- time conversion

/// fuse_exact_time and fuse::ExactTimeRecord (fuse/base/rational.hpp) are
/// the same two-int64 layout by design (abi-design.md); this is the one
/// place that is asserted and the one place the fields are copied across,
/// so a Rational never crosses this boundary through a double (rule 6).
static_assert(sizeof(fuse_exact_time) == sizeof(fuse::ExactTimeRecord));
static_assert(offsetof(fuse_exact_time, numerator) ==
             offsetof(fuse::ExactTimeRecord, numerator));
static_assert(offsetof(fuse_exact_time, denominator) ==
             offsetof(fuse::ExactTimeRecord, denominator));

[[nodiscard]] fuse_exact_time to_abi_time(const fuse::Rational& r) noexcept {
  const fuse::ExactTimeRecord rec = r.to_record();
  return fuse_exact_time{rec.numerator, rec.denominator};
}

[[nodiscard]] fuse::Result<fuse::Rational> from_abi_time(
    fuse_exact_time t) {
  return fuse::Rational::from_record(fuse::ExactTimeRecord{t.numerator, t.denominator});
}

// --------------------------------------------------------------- entries

fuse_status impl_version(fuse_ctx*, const fuse_null*, fuse_version_info* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  const fuse_status sz = check_struct_size(out);
  if (sz != FUSE_OK) return sz;
  static const std::string kVersionString =
      std::string("FUSE ") + FUSE_VERSION_STRING + " (participant ABI v" +
      std::to_string(FUSE_ABI_VERSION) + ")";
  out->struct_size = sizeof(fuse_version_info);
  out->library = fuse_version{FUSE_PROJECT_VERSION_MAJOR, FUSE_PROJECT_VERSION_MINOR,
                              FUSE_PROJECT_VERSION_PATCH};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = kVersionString.c_str();
  return FUSE_OK;
}

fuse_status impl_capabilities(fuse_ctx*, const fuse_null*,
                              fuse_capabilities* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  const fuse_status sz = check_struct_size(out);
  if (sz != FUSE_OK) return sz;
  std::uint64_t flags = FUSE_CAP_YAML;
#if FUSE_HAVE_MPI
  flags |= FUSE_CAP_MPI;
#endif
  // Not yet implemented through this ABI (open question in the task
  // report): device fields (register_field refuses FUSE_MEMSPACE_DEVICE),
  // checkpoint/restore (no iteration-region support through begin_step/
  // end_step), MPI connect/accept or MPMD, and the LowFive in-memory path.
  out->struct_size = sizeof(fuse_capabilities);
  out->flags = flags;
  return FUSE_OK;
}

/// Kokkos has global, process-wide initialization state (SerialMeshBackend
/// and every space kernel need it) that NOTHING outside this library has
/// any way to set up: a pure-C or Fortran caller has no Kokkos header and
/// no way to call Kokkos::initialize() itself, so this library's own entry
/// points are the only place it can happen. Lazily, on the first
/// fuse_create(), guarded so a second context in the same process is a
/// no-op rather than Kokkos's own "already initialized" refusal.
/// NEVER finalized from fuse_destroy(): Kokkos does not support
/// initialize() again after finalize() in the same process, and this
/// library's contexts are created and destroyed far more often than once
/// per process (every test in this suite does), so finalizing on the
/// first context's destroy() would strand every later one. Left for the
/// process's own exit, exactly like fuse_test_main.cpp's Kokkos::ScopeGuard
/// does for every other test binary in this tree -- the cost stated
/// plainly rather than discovered later.
void ensure_kokkos_initialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::initialize();
    }
  });
}

fuse_status impl_create(fuse_ctx* ctx, const fuse_null*,
                        fuse_context_out* out) {
  if (ctx != nullptr) {
    return fail(nullptr, FUSE_ERR_INVALID, "E-ABI-CREATE",
               "fuse_create: ctx must be NULL");
  }
  if (out == nullptr) return FUSE_ERR_INVALID;
  const fuse_status sz = check_struct_size(out);
  if (sz != FUSE_OK) return sz;
  try {
    ensure_kokkos_initialized();
    auto* c = new fuse_ctx();
    try {
      HandleTable&                t = table();
      std::lock_guard<std::mutex> lock(t.mu);
      t.live.insert(static_cast<const void*>(c));
    } catch (...) {
      delete c;
      throw;
    }
    out->struct_size = sizeof(fuse_context_out);
    out->ctx = c;
    return FUSE_OK;
  } catch (const std::exception& e) {
    return fail(nullptr, FUSE_ERR_RUNTIME, "E-ABI-CREATE", e.what());
  }
}

fuse_status impl_destroy(fuse_ctx* ctx, const fuse_null*, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    {
      HandleTable&                t = table();
      std::lock_guard<std::mutex> lock(t.mu);
      t.live.erase(static_cast<const void*>(&c));
    }
    delete &c;
    return FUSE_OK;
  });
}

fuse_status impl_last_error(fuse_ctx* ctx, const fuse_null*, fuse_error* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  const fuse_status sz = check_struct_size(out);
  if (sz != FUSE_OK) return sz;

  const std::string* code = nullptr;
  const std::string* msg = nullptr;
  if (ctx == nullptr) {
    code = &thread_error_code();
    msg  = &thread_error_message();
  } else if (!is_live(ctx)) {
    return FUSE_ERR_INVALID;
  } else {
    code = &ctx->last_error_code;
    msg  = &ctx->last_error_message;
  }

  out->struct_size = sizeof(fuse_error);
  out->status      = FUSE_OK;  // the ABI never reconstructs the ORIGINAL
                               // failing status; only the message/code.
  out->length      = msg->size();
  out->code        = code->empty() ? nullptr : code->c_str();
  if (out->buffer == nullptr) {
    out->truncated = msg->empty() ? 0 : 1;
    return FUSE_OK;
  }
  if (out->capacity < msg->size() + 1) {
    out->truncated = 1;
    return FUSE_OK;
  }
  std::memcpy(out->buffer, msg->c_str(), msg->size() + 1);
  out->truncated = 0;
  return FUSE_OK;
}

fuse_status impl_load_graph(fuse_ctx* ctx, const fuse_graph_source* in,
                           fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Empty) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-LOAD-STATE",
                 "fuse_load_graph: a graph is already loaded on this context");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    const bool have_yaml  = in->yaml_path != nullptr;
    const bool have_bytes = in->bytes != nullptr;
    if (have_yaml == have_bytes) {
      return fail(&c, FUSE_ERR_INVALID, "E-ABI-LOAD-SOURCE",
                 "fuse_load_graph: exactly one of yaml_path and bytes must "
                 "be set");
    }
    fuse::Result<fuse::CouplingGraph> loaded =
        have_yaml ? fuse::load_yaml_file(in->yaml_path)
                  : fuse::deserialize(std::span<const std::byte>(
                        static_cast<const std::byte*>(in->bytes),
                        static_cast<std::size_t>(in->byte_count)));
    if (!loaded.ok()) {
      return fail(&c, FUSE_ERR_INVALID, std::string(loaded.error().code),
                 loaded.error().message);
    }
    c.graph  = std::move(loaded).value();
    c.state  = fuse_ctx::State::Loaded;
    return FUSE_OK;
  });
}

fuse_status impl_save_graph(fuse_ctx* ctx, const fuse_null*,
                           fuse_graph_sink* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state == fuse_ctx::State::Empty) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-SAVE-STATE",
                 "fuse_save_graph: no graph is loaded on this context");
    }
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status sz = check_struct_size(out);
    if (sz != FUSE_OK) return sz;
    const std::vector<std::byte> bytes = fuse::serialize(c.graph);
    out->struct_size = sizeof(fuse_graph_sink);
    out->written     = bytes.size();
    if (out->buffer == nullptr || out->capacity < bytes.size()) {
      return FUSE_OK;  // query-size path, or too small: nothing written
    }
    std::memcpy(out->buffer, bytes.data(), bytes.size());
    return FUSE_OK;
  });
}

fuse_status impl_setup(fuse_ctx* ctx, const fuse_setup* in, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Loaded) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-SETUP-STATE",
                 "fuse_setup: must be called after fuse_load_graph and "
                 "before fuse_initialize");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (in->communicator != 0) {
#if !FUSE_HAVE_MPI
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-SETUP-NOMPI",
                 "fuse_setup: a non-zero communicator was supplied, but this "
                 "build has no MPI");
#endif
      c.communicator     = in->communicator;
      c.communicator_set = true;
    }
    if (in->has_end_time) {
      const fuse::Result<fuse::Rational> t = from_abi_time(in->end_time);
      if (!t.ok()) {
        return fail(&c, FUSE_ERR_INVALID, std::string(t.error().code),
                   t.error().message);
      }
      c.end_time     = t.value();
      c.has_end_time = true;
    }
    return FUSE_OK;
  });
}

fuse_status impl_field_handle(fuse_ctx* ctx, const fuse_field_name* in,
                             fuse_field_ref* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state == fuse_ctx::State::Empty) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-FIELD-HANDLE-STATE",
                 "fuse_field_handle: no graph is loaded on this context");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    if (in->component == nullptr || in->field == nullptr) {
      return fail(&c, FUSE_ERR_INVALID, "E-ABI-FIELD-HANDLE-NULL",
                 "fuse_field_handle: component or field name is null");
    }
    const auto cid = c.graph.find_component(in->component);
    if (!cid.has_value()) {
      return fail(&c, FUSE_ERR_NOTFOUND, "E-ABI-FIELD-HANDLE-COMPONENT",
                 std::string("fuse_field_handle: no component named '") +
                     in->component + "'");
    }
    const auto fid = c.graph.find_field(*cid, in->field);
    if (!fid.has_value()) {
      return fail(&c, FUSE_ERR_NOTFOUND, "E-ABI-FIELD-HANDLE-FIELD",
                 std::string("fuse_field_handle: component '") +
                     in->component + "' has no field named '" + in->field +
                     "'");
    }
    const fuse::FieldSpec& spec = c.graph.field(*fid);
    out->struct_size = sizeof(fuse_field_ref);
    out->handle      = static_cast<std::int32_t>(fid->value());
    out->components  = spec.components;
    out->intent       = static_cast<std::int32_t>(spec.intent);  // Provides=0,
                                                                 // Requires=1,
                                                                 // Bidirectional=2
                                                                 // agree by
                                                                 // construction
    switch (spec.conservation) {
      case fuse::Conservation::None:      out->conservation = 0; break;
      case fuse::Conservation::Intensive: out->conservation = 1; break;
      case fuse::Conservation::Extensive: out->conservation = 2; break;
    }
    return FUSE_OK;
  });
}

fuse_status impl_register_field(fuse_ctx* ctx, const fuse_field_view* in,
                               fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Loaded) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-REGISTER-STATE",
                 "fuse_register_field: must be called after fuse_load_graph "
                 "and before fuse_initialize");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (in->element_type != FUSE_ELEMENT_F64) {
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-REGISTER-ELEMENT",
                 "fuse_register_field: only FUSE_ELEMENT_F64 is supported "
                 "through this ABI");
    }
    if (in->memspace != FUSE_MEMSPACE_HOST) {
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-REGISTER-MEMSPACE",
                 "fuse_register_field: only FUSE_MEMSPACE_HOST is supported "
                 "through this ABI (open question: device fields)");
    }
    if (in->ownership == FUSE_OWNERSHIP_GHOSTED) {
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-REGISTER-OWNERSHIP",
                 "fuse_register_field: FUSE_OWNERSHIP_GHOSTED is not yet "
                 "supported through this ABI (open question)");
    }
    if (in->owned_count != in->count && in->ownership != FUSE_OWNERSHIP_GHOSTED) {
      return fail(&c, FUSE_ERR_INVALID, "E-ABI-REGISTER-OWNED-COUNT",
                 "fuse_register_field: owned_count must equal count unless "
                 "ownership is FUSE_OWNERSHIP_GHOSTED");
    }
    const fuse::Layout layout{.stride = static_cast<std::size_t>(in->stride),
                              .component_stride =
                                  static_cast<std::size_t>(in->component_stride),
                              .count = static_cast<std::size_t>(in->count),
                              .components = in->components};
    const fuse::FieldRef ref{static_cast<double*>(in->data), layout,
                             fuse::MemorySpace::Host};
    const fuse::Result<void> bound =
        c.fields.bind(fuse::FieldId{static_cast<std::uint32_t>(in->handle)}, ref);
    if (!bound.ok()) {
      return fail(&c, FUSE_ERR_INVALID, std::string(bound.error().code),
                 bound.error().message);
    }
    return FUSE_OK;
  });
}

fuse_status impl_initialize(fuse_ctx* ctx, const fuse_null*, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Loaded) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-INIT-STATE",
                 "fuse_initialize: no graph has been loaded, or the context "
                 "is already initialized");
    }
    const fuse::Diagnostics diags = fuse::validate(c.graph);
    if (diags.has_errors()) {
      return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-VALIDATE",
                 "fuse_initialize: the graph does not validate:\n" +
                     diags.to_string());
    }
    for (std::uint32_t i = 0; i < c.graph.mesh_count(); ++i) {
      const fuse::Result<void> r =
          c.backend.load(c.graph.mesh(fuse::MeshId{i}));
      if (!r.ok()) {
        return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                   r.error().message);
      }
    }
    const fuse::LoweringResult lowered = fuse::lower(c.graph);
    if (!lowered.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-LOWER",
                 "fuse_initialize: the graph does not lower:\n" +
                     lowered.diags.to_string());
    }
    c.dag    = lowered.dag;
    c.clocks = fuse::ClockSet{c.graph};
    const fuse::RegionSizes sizes = fuse::make_region_sizes(c.graph, c.backend);
    const fuse::ScheduleResult sched =
        fuse::schedule(c.graph, *c.dag, *c.clocks, sizes);
    if (!sched.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-SCHEDULE",
                 "fuse_initialize: the graph does not schedule:\n" +
                     sched.diags.to_string());
    }
    c.plan = sched.plan;

    // Wire every graph component to either the test-injected real
    // fuse::Component (driven in-process, through fuse_run()) or a fresh
    // ExternalStepComponent (driven through fuse_begin_step()/
    // fuse_end_step()); see context.hpp's header comment.
    c.ext.assign(c.graph.component_count(), ExternalState{});
    c.external_count = 0;
    for (std::uint32_t i = 0; i < c.graph.component_count(); ++i) {
      const fuse::ComponentId id{i};
      const std::string& name = c.graph.component(id).name;
      auto it = c.injected.find(name);
      if (it != c.injected.end()) {
        std::unique_ptr<fuse::Component> comp = std::move(it->second);
        c.injected.erase(it);
        const fuse::Result<void> r = comp->register_fields(c.fields, c.graph);
        if (!r.ok()) {
          return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                     r.error().message);
        }
        c.components.add(std::move(comp));
        c.ext[i].in_process = true;
      } else {
        c.components.add(std::make_unique<ExternalStepComponent>(name));
        c.ext[i].in_process = false;
        ++c.external_count;
        const fuse::Rational interval = c.graph.component(id).interval;
        const fuse::Rational ticks = c.plan->period() / interval;
        if (!ticks.is_integer()) {
          return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-PERIOD",
                     "fuse_initialize: component '" + name +
                         "' interval does not divide the common period "
                         "evenly (a scheduler invariant; should be "
                         "unreachable)");
        }
        c.ext[i].ticks_per_period = ticks.num();
      }
    }

    fuse::RunnerConfig config{};
    if (c.communicator_set) {
      config.base_communicator = static_cast<int>(c.communicator);
    }
    c.runner.emplace(c.graph, *c.dag, *c.plan, c.backend);
    const fuse::Result<void> r =
        c.runner->initialize(c.components, c.fields, config);
    if (!r.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                 r.error().message);
    }
    c.state = fuse_ctx::State::Initialized;
    return FUSE_OK;
  });
}

fuse_status impl_is_coupling_ongoing(fuse_ctx* ctx, const fuse_null*,
                                    fuse_ongoing* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Initialized) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-ONGOING-STATE",
                 "fuse_is_coupling_ongoing: the context is not initialized");
    }
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status sz = check_struct_size(out);
    if (sz != FUSE_OK) return sz;
    out->struct_size = sizeof(fuse_ongoing);
    out->period      = c.periods_completed;
    const fuse::Rational now = c.plan->period() * fuse::Rational{c.periods_completed};
    out->now = to_abi_time(now);
    if (c.graph.component_count() == 0) {
      out->ongoing = 0;
    } else if (!c.has_end_time) {
      out->ongoing = 1;
    } else {
      out->ongoing = (now < c.end_time) ? 1 : 0;
    }
    return FUSE_OK;
  });
}

/// Shared body of fuse_begin_step()/fuse_end_step(): resolve `in->component`
/// to an externally-driven ComponentId, or report exactly why not.
/// `*out_id` is set only on success; on failure the caller's `fail()` call
/// should use `out_code`/`out_message`, which are always set.
bool resolve_external(fuse_ctx& c, const char* name, const char* fn,
                      fuse::ComponentId* out_id, std::string* out_code,
                      std::string* out_message) {
  if (c.state != fuse_ctx::State::Initialized) {
    *out_code = "E-ABI-STEP-STATE";
    *out_message = std::string(fn) + ": the context is not initialized";
    return false;
  }
  if (name == nullptr) {
    *out_code = "E-ABI-STEP-NULL";
    *out_message = std::string(fn) + ": component name is null";
    return false;
  }
  const auto cid = c.graph.find_component(name);
  if (!cid.has_value()) {
    *out_code = "E-ABI-STEP-COMPONENT";
    *out_message = std::string(fn) + ": no component named '" + name + "'";
    return false;
  }
  if (c.ext[cid->value()].in_process) {
    *out_code = "E-ABI-STEP-IN-PROCESS";
    *out_message = std::string(fn) + ": component '" + name +
                   "' is driven in-process (fuse_run), not through "
                   "fuse_begin_step/fuse_end_step";
    return false;
  }
  *out_id = *cid;
  return true;
}

fuse_status impl_begin_step(fuse_ctx* ctx, const fuse_step_request* in,
                           fuse_step_report* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    fuse::ComponentId cid;
    std::string err_code, err_message;
    if (!resolve_external(c, in->component, "fuse_begin_step", &cid,
                          &err_code, &err_message)) {
      return fail(&c, FUSE_ERR_STATE, err_code, err_message);
    }
    ExternalState& st = c.ext[cid.value()];
    if (st.open) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-STEP-ALREADY-OPEN",
                 std::string("fuse_begin_step: a step is already open on "
                             "component '") +
                     in->component + "'; call fuse_end_step() first");
    }
    const fuse::Clock& clock = c.clocks->clock(cid);
    st.t_begin = clock.now();
    st.t_end   = clock.next();
    st.open    = true;

    out->struct_size          = sizeof(fuse_step_report);
    out->requires_checkpoint  = 0;  // no iteration-region support through
                                    // this ABI yet (open question)
    out->requires_restore     = 0;
    out->step                 = clock.step();
    out->iteration            = 0;
    out->reserved             = 0;
    out->t_begin               = to_abi_time(st.t_begin);
    out->t_end                 = to_abi_time(st.t_end);
    return FUSE_OK;
  });
}

fuse_status impl_end_step(fuse_ctx* ctx, const fuse_step_request* in,
                         fuse_step_report* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    fuse::ComponentId cid;
    std::string err_code, err_message;
    if (!resolve_external(c, in->component, "fuse_end_step", &cid, &err_code,
                          &err_message)) {
      return fail(&c, FUSE_ERR_STATE, err_code, err_message);
    }
    ExternalState& st = c.ext[cid.value()];
    if (!st.open) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-STEP-NOT-OPEN",
                 std::string("fuse_end_step: no step is open on component '") +
                     in->component + "'");
    }
    st.open = false;
    c.clocks->clock(cid).advance();
    ++st.ticks_this_period;

    out->struct_size         = sizeof(fuse_step_report);
    out->requires_checkpoint = 0;
    out->requires_restore    = 0;
    out->step                = c.clocks->clock(cid).step() - 1;
    out->iteration           = 0;
    out->reserved            = 0;
    out->t_begin              = to_abi_time(st.t_begin);
    out->t_end                = to_abi_time(st.t_end);

    if (st.ticks_this_period >= st.ticks_per_period) {
      st.ticks_this_period = 0;
      ++c.ready_for_period;
    }
    if (c.ready_for_period >= c.external_count && c.external_count > 0) {
      c.ready_for_period = 0;
      const fuse::Result<void> r = c.runner->run_period(c.components);
      if (!r.ok()) {
        return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                   r.error().message);
      }
      ++c.periods_completed;
    }
    return FUSE_OK;
  });
}

fuse_status impl_run(fuse_ctx* ctx, const fuse_run_request* in, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Initialized) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-RUN-STATE",
                 "fuse_run: the context is not initialized");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (c.external_count > 0) {
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-RUN-EXTERNAL",
                 "fuse_run: at least one component in this graph is driven "
                 "from outside (fuse_begin_step/fuse_end_step); drive the "
                 "run with those instead");
    }
    if (in->periods < -1) {
      return fail(&c, FUSE_ERR_INVALID, "E-ABI-RUN-PERIODS",
                 "fuse_run: periods must be >= 0, or exactly -1 for \"until "
                 "the end time\"");
    }
    if (in->periods == -1) {
      if (!c.has_end_time) {
        return fail(&c, FUSE_ERR_INVALID, "E-ABI-RUN-NO-END-TIME",
                   "fuse_run: periods == -1 requires an end_time set by "
                   "fuse_setup()");
      }
      while (true) {
        const fuse::Rational now =
            c.plan->period() * fuse::Rational{c.periods_completed};
        if (!(now < c.end_time)) break;
        const fuse::Result<void> r = c.runner->run_period(c.components);
        if (!r.ok()) {
          return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                     r.error().message);
        }
        ++c.periods_completed;
      }
      return FUSE_OK;
    }
    const fuse::Result<void> r = c.runner->run(c.components, in->periods);
    if (!r.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                 r.error().message);
    }
    c.periods_completed += in->periods;
    return FUSE_OK;
  });
}

// ----------------------------------------------------------- the table

// Member-count cross-check (abi-design.md point 5 / task instructions):
// counts FUSE_ABI_ENTRY expansions in entries.def independently of the
// table initialiser below, so a hand-edited participant.h that silently
// drops or duplicates an entry is caught here at compile time rather than
// only by the golden-header test (which needs tools/gen-abi.py on PATH).
constexpr int kEntryCountFromDef = 0
#define FUSE_ABI_ENTRY(name, in, out, doc) +1
#include "fuse/abi/entries.def"
#undef FUSE_ABI_ENTRY
    ;

static_assert(
    (sizeof(fuse_participant_abi_v1) -
     offsetof(fuse_participant_abi_v1, version)) /
        sizeof(void*) ==
    static_cast<std::size_t>(kEntryCountFromDef),
    "fuse_participant_abi_v1's member count disagrees with entries.def's "
    "entry count; regenerate participant.h (tools/gen-abi.py --write)");

#define FUSE_ABI_ENTRY(name, in, out, doc) impl_##name,
const fuse_participant_abi_v1 g_table_v1 = {
    sizeof(fuse_participant_abi_v1),
    FUSE_ABI_VERSION,
#include "fuse/abi/entries.def"
};
#undef FUSE_ABI_ENTRY

}  // namespace

// The ONE exported symbol (abi-design.md point 1). `-fvisibility=hidden`
// (src/CMakeLists.txt's fuse-abi-objects target) hides everything in this
// translation unit BY DEFAULT, participant.h's own declaration carries no
// visibility attribute (it is frozen and must not gain one), so the
// definition is where this single exception is stated -- belt-and-braces
// with the link-time allowlist (src/abi/exported_symbols.{map,sym}), which
// is what actually keeps every statically-linked dependency's own
// default-visibility symbols (Kokkos, fmt, ...) out of the built .so/.dylib.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((visibility("default")))
#endif
const fuse_participant_abi_v1* fuse_coupler_abi(std::uint32_t abi_version) {
  if (abi_version != FUSE_ABI_VERSION) return nullptr;
  return &g_table_v1;
}
