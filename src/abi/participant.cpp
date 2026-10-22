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
#include <cstdint>
#include <cstring>
#include <cstdlib>   // std::atexit, for the Kokkos finalize hook; std::getenv/atoi below
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>

#include <Kokkos_Core.hpp>
#include <fmt/format.h>

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
// Kokkos' own pre-initialization device enumeration (used internally by its
// `map_device_id_by("mpi_rank")` option); the non-test overload
// (`get_visible_devices()`, no argument) is the one its own header comment
// says to use. Not raw CUDA: this is Kokkos' API, just under `Impl`.
#include <impl/Kokkos_DeviceManagement.hpp>
#endif

#include "fuse/abi/collective_reconcile.hpp"
#include "fuse/abi/device_binding.hpp"
#include "fuse/abi/plugin_agreement.hpp"
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
#include "fuse/transport/connect.hpp"
#include "fuse/transport/transport.hpp"

#include "context.hpp"
#include "external_step_component.hpp"
#include "provider_adapter.hpp"

namespace {

using fuse::abi_detail::ExternalStepComponent;
using fuse::abi_detail::ExternalState;

// ----------------------------------------------------------- struct_size

/// abi-design.md rule 2's "first frozen layout" minimum, per struct type.
/// Defaults to the struct's own CURRENT `sizeof`, which is exactly right
/// for every struct in entries.def that has shipped exactly one layout so
/// far (there was, until Defect 3's `next_component`, no OTHER case to get
/// wrong). A struct that has since grown by appending a member specializes
/// this template to its ORIGINAL size, computed from its own last
/// pre-growth member so there is no magic number to drift from the
/// header -- see the fuse_step_report/fuse_ongoing specializations below,
/// just after their own growth was appended 2026-10-03 (entries.def's
/// comment on `next_component` explains why).
template <class T>
struct FirstFrozenStructSize {
  static constexpr std::size_t value = sizeof(T);
};

template <>
struct FirstFrozenStructSize<fuse_step_report> {
  static constexpr std::size_t value =
      offsetof(fuse_step_report, t_end) + sizeof(fuse_exact_time);
};

template <>
struct FirstFrozenStructSize<fuse_ongoing> {
  static constexpr std::size_t value =
      offsetof(fuse_ongoing, now) + sizeof(fuse_exact_time);
};

template <>
struct FirstFrozenStructSize<fuse_setup_options> {
  // The layout shipped before my_component was appended 2026-10-03; see
  // entries.def's own comment on that field.
  static constexpr std::size_t value =
      offsetof(fuse_setup_options, end_time) + sizeof(fuse_exact_time);
};

template <>
struct FirstFrozenStructSize<fuse_field_view> {
  // The layout shipped before window_samples was appended 2026-10-04;
  // see entries.def's own comment on that field. `data` is the last
  // pre-growth member.
  static constexpr std::size_t value =
      offsetof(fuse_field_view, data) + sizeof(void*);
};

template <>
struct FirstFrozenStructSize<fuse_step_request> {
  // Review 2026-10-11 AB-2: this specialization was missing, so an old
  // caller's struct_size=16 (the layout shipped before t_begin/t_end/step/
  // iteration/requires_checkpoint/requires_restore/reserved were appended
  // in 612b153 for the provider direction, per c-abi.md's growth table
  // "16 -> 72") was refused FUSE_ERR_VERSION by check_struct_size's old
  // `>= sizeof(current)` bound, even though begin_step()/end_step() never
  // read any of the grown members (entries.def's comments on each mark
  // them provider-direction-only). `component` is the last pre-growth
  // member.
  static constexpr std::size_t value =
      offsetof(fuse_step_request, component) + sizeof(const char*);
};

/// abi-design.md rule 2. A `struct_size` smaller than `T`'s first frozen
/// layout, or larger than FUSE_ABI_MAX_STRUCT_SIZE, is FUSE_ERR_VERSION.
/// Before Defect 3 this bound was always `sizeof(T)` (current); now that
/// fuse_step_report/fuse_ongoing have a second, larger frozen layout, an
/// OLD caller's smaller-but-still-valid struct_size must keep passing here
/// -- see FirstFrozenStructSize above, and struct_size_covers() below for
/// the matching write-side guard every appended member needs.
template <class T>
[[nodiscard]] fuse_status check_struct_size(const T* s) noexcept {
  if (s == nullptr) return FUSE_ERR_INVALID;
  if (s->struct_size < FirstFrozenStructSize<T>::value ||
      s->struct_size > static_cast<std::uint32_t>(FUSE_ABI_MAX_STRUCT_SIZE)) {
    return FUSE_ERR_VERSION;
  }
  return FUSE_OK;
}

/// The write-side mirror of rule 2, for an appended member: `declared` is
/// the OUT struct_size the caller passed in, captured before this entry
/// point's own unconditional `out->struct_size = sizeof(T)` overwrites it
/// (that overwrite is always safe on its own -- struct_size sits at offset
/// 0, which every valid caller's allocation covers -- but it would hide the
/// caller's REAL declared size from this check if read afterwards).
/// `through` is `offsetof(T, member) + sizeof(member)` at each call site.
/// A caller whose declared struct_size does not reach `through` allocated
/// no storage for `member`; writing it anyway would be this ABI boundary's
/// own version of Defect 2's heap overflow, just on the other side of the
/// call.
[[nodiscard]] inline bool struct_size_covers(std::uint32_t declared,
                                             std::size_t through) noexcept {
  return declared >= through;
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
/// The fuse_status the most recent failing call on this thread returned;
/// the thread-local twin of fuse_ctx::last_error_status (context.hpp).
fuse_status& thread_error_status() {
  static thread_local fuse_status status = FUSE_OK;
  return status;
}

void store_error(fuse_ctx* ctx, fuse_status status, const std::string& code,
                 const std::string& msg) noexcept {
  try {
    thread_error_code()    = code;
    thread_error_message() = msg;
    thread_error_status()  = status;
  } catch (...) {  // NOLINT(bugprone-empty-catch): nothing better is possible
  }
  if (ctx != nullptr) {
    try {
      ctx->last_error_code    = code;
      ctx->last_error_message = msg;
      ctx->last_error_status  = status;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }
}

fuse_status fail(fuse_ctx* ctx, fuse_status code, const std::string& fcode,
                 const std::string& msg) noexcept {
  store_error(ctx, code, fcode, msg);
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
  out->library = fuse_semantic_version{FUSE_PROJECT_VERSION_MAJOR, FUSE_PROJECT_VERSION_MINOR,
                              FUSE_PROJECT_VERSION_PATCH};
  out->abi_version = FUSE_ABI_VERSION;
  out->reserved = 0;
  out->string = kVersionString.c_str();
  return FUSE_OK;
}

fuse_status impl_capabilities(fuse_ctx*, const fuse_null*,
                              fuse_capability_set* out) {
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
  out->struct_size = sizeof(fuse_capability_set);
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
/// ...but "left for the process's own exit" has to mean an exit this
/// library controls the ORDER of. Kokkos::initialize() registers its own
/// exit hook, and that hook runs at a point where the libraries Kokkos
/// itself sits on -- the OpenMP runtime above all -- may already have been
/// torn down. With a statically linked Kokkos that ordering happened to
/// hold; with a shared one it does not, and the process aborts AFTER
/// finishing its work with "mutex lock failed: Invalid argument", locking
/// a mutex in an image that is already gone. Measured 2026-10-09, when
/// Kokkos was rebuilt as a shared library on this lab's machine: the
/// installed C and Fortran consumers both printed their results and every
/// expected value, then died in exit.
///
/// std::atexit is LIFO, so a hook registered HERE -- after
/// Kokkos::initialize() has registered its own -- runs BEFORE it, while
/// the runtime underneath is still intact. Kokkos's own hook then finds
/// itself already finalized and does nothing. This is not finalizing from
/// fuse_destroy(), which the paragraph above rules out for good reason: by
/// the time an atexit hook runs, no caller can create another context.
namespace {

/// An explicit `--kokkos-device-id=N` argument or the `KOKKOS_DEVICE_ID`
/// environment variable, checked in that order -- the same precedence
/// Kokkos' own command-line parser gives the two spellings. `argc==0` (this
/// file's own call site, which has no argv at all: `ensure_kokkos_initialized()`
/// runs from `fuse_create()`, not from a `main()`) simply skips the argv
/// scan and falls through to the environment variable.
std::optional<int> explicit_device_id_override(int argc, char** argv) {
  if (const char* env = std::getenv("KOKKOS_DEVICE_ID");
      env != nullptr && env[0] != '\0') {
    return std::atoi(env);
  }
  constexpr std::string_view kPrefix = "--kokkos-device-id=";
  for (int i = 0; i < argc; ++i) {
    if (argv == nullptr || argv[i] == nullptr) continue;
    const std::string_view a{argv[i]};
    if (a.substr(0, kPrefix.size()) == kPrefix) {
      return std::atoi(std::string(a.substr(kPrefix.size())).c_str());
    }
  }
  return std::nullopt;
}

/// This process's rank within its node: `MPI_COMM_TYPE_SHARED` splits
/// `MPI_COMM_WORLD` into one communicator per node, so a rank's position in
/// its own split is exactly "how many other ranks on this node come before
/// me" -- which device index 0, 1, 2, ... it should try to claim if devices
/// are assigned round-robin. 0 without MPI, and 0 before `MPI_Init` (this
/// function is called from `ensure_kokkos_initialized()`, which MUST run
/// before `MPI_Init` in a library-embedding caller is not guaranteed by
/// anything this library controls -- `MPI_Initialized()` is the one
/// well-defined way to ask "is it safe to call `MPI_Comm_split_type` yet").
int local_rank_in_node() {
#if FUSE_HAVE_MPI
  int mpi_started = 0;
  if (MPI_Initialized(&mpi_started) != MPI_SUCCESS || mpi_started == 0) {
    return 0;
  }
  MPI_Comm local = MPI_COMM_NULL;
  if (MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                          MPI_INFO_NULL, &local) != MPI_SUCCESS) {
    return 0;
  }
  int rank = 0;
  MPI_Comm_rank(local, &rank);
  MPI_Comm_free(&local);
  return rank;
#else
  return 0;
#endif
}

/// How many devices this build's Kokkos backend reports visible, queried
/// BEFORE `Kokkos::initialize()` -- the same enumeration Kokkos' own
/// `map_device_id_by("mpi_rank")` option uses internally for this exact
/// purpose. 0 on a Serial/OpenMP-only Kokkos (no device backend at all).
int visible_device_count() {
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_SYCL)
  return static_cast<int>(Kokkos::Impl::get_visible_devices().size());
#else
  return 0;
#endif
}

/// Gathers the real inputs `fuse::abi::resolve_device_id()` needs and
/// returns the device id THIS process should bind to, or `nullopt` when
/// there is no device backend at all and no explicit override either --
/// the caller then passes nothing to Kokkos and gets ordinary host-only
/// behaviour, unchanged from before this function existed.
std::optional<int> resolve_device_id_for_this_process(int argc, char** argv) {
  const std::optional<int> override_id = explicit_device_id_override(argc, argv);
  const int count = visible_device_count();
  if (count <= 0 && !override_id.has_value()) return std::nullopt;
  const int local_rank = local_rank_in_node();
  const int id = fuse::abi::resolve_device_id(local_rank, count, override_id);
  if (std::getenv("FUSE_DEVICE_DEBUG") != nullptr) {
    fmt::print(stderr,
               "[device-bind] local_rank={} device_count={} override={} -> "
               "device_id={}\n",
               local_rank, count,
               override_id.has_value() ? std::to_string(*override_id) : "none",
               id);
  }
  return id;
}

}  // namespace

void ensure_kokkos_initialized() {
  static std::once_flag once;
  std::call_once(once, [] {
    if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
      Kokkos::InitializationSettings settings;
      // No argv at this call site (fuse_create() is a library entry point,
      // not a main()): only the KOKKOS_DEVICE_ID environment variable can
      // override here, not a command-line flag.
      if (const std::optional<int> dev =
              resolve_device_id_for_this_process(0, nullptr)) {
        settings.set_device_id(*dev);
      }
      Kokkos::initialize(settings);
      std::atexit([] {
        if (Kokkos::is_initialized() && !Kokkos::is_finalized()) {
          Kokkos::finalize();
        }
      });
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
    // Taken out of the context BEFORE the delete, and released AFTER it:
    // the Runner and its transport hold duplicates of this communicator and
    // must be destroyed first, and the error cannot be reported through a
    // context that no longer exists -- hence the thread-local error, the
    // same channel fuse_last_error() reads for a null handle.
    fuse::transport::ConnectedWorld owned = c.connected;
    c.connected = fuse::transport::ConnectedWorld{};
    // D61 round 3: the component-local reconciliation communicator
    // (context.hpp's own comment on `collective_comm_f`), freed the same
    // way -- extracted before delete, released after, 0 meaning there is
    // nothing to free (the default, or any of the "never formed" cases
    // that field's own comment lists).
    const int owned_collective_comm_f = c.collective_comm_f;
    c.collective_comm_f = 0;
    delete &c;
#if FUSE_HAVE_MPI
    if (auto r = fuse::transport::disconnect(owned); !r.ok()) {
      thread_error_code()    = r.error().code;
      thread_error_message() = r.error().message;
      thread_error_status()  = FUSE_ERR_RUNTIME;
      return FUSE_ERR_RUNTIME;
    }
    if (owned_collective_comm_f != 0) {
      MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(owned_collective_comm_f));
      if (comm != MPI_COMM_NULL && MPI_Comm_free(&comm) != MPI_SUCCESS) {
        thread_error_code()    = "E-ABI-COLLECTIVE-COMM";
        thread_error_message() = "fuse_destroy: MPI_Comm_free failed on the "
                                 "component-local reconciliation communicator";
        thread_error_status()  = FUSE_ERR_RUNTIME;
        return FUSE_ERR_RUNTIME;
      }
    }
#else
    (void)owned;
    (void)owned_collective_comm_f;
#endif
    return FUSE_OK;
  });
}

fuse_status impl_last_error(fuse_ctx* ctx, const fuse_null*, fuse_error* out) {
  if (out == nullptr) return FUSE_ERR_INVALID;
  const fuse_status sz = check_struct_size(out);
  if (sz != FUSE_OK) return sz;

  const std::string* code = nullptr;
  const std::string* msg = nullptr;
  const fuse_status* status = nullptr;
  if (ctx == nullptr) {
    code   = &thread_error_code();
    msg    = &thread_error_message();
    status = &thread_error_status();
  } else if (!is_live(ctx)) {
    return FUSE_ERR_INVALID;
  } else {
    code   = &ctx->last_error_code;
    msg    = &ctx->last_error_message;
    status = &ctx->last_error_status;
  }

  out->struct_size = sizeof(fuse_error);
  out->status      = *status;  // the fuse_status store_error() recorded
                               // alongside code/message for the call that
                               // set them (FUSE_OK if nothing has failed
                               // yet on this context/thread).
  out->length      = msg->size();
  out->code        = code->empty() ? nullptr : code->c_str();
  if (out->buffer == nullptr) {
    out->truncated = msg->empty() ? 0 : 1;
    return FUSE_OK;
  }
  if (out->capacity < msg->size() + 1) {
    // Copy the prefix that fits, NUL-terminated, as the header promises
    // ("as much of the message as fits"); the first version copied nothing
    // on truncation and left the caller printing an uninitialised buffer.
    out->truncated = 1;
    if (out->capacity > 0) {
      const std::size_t n = static_cast<std::size_t>(out->capacity) - 1;
      std::memcpy(out->buffer, msg->c_str(), n);
      out->buffer[n] = '\0';
    }
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

fuse_status impl_setup(fuse_ctx* ctx, const fuse_setup_options* in, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Loaded) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-SETUP-STATE",
                 "fuse_setup_options: must be called after fuse_load_graph and "
                 "before fuse_initialize");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    // Review 2026-10-11 AB-11: entries.def documents `reserved` as "must
    // be 0", but nothing read it to check -- a caller that left it
    // uninitialised (or copy-pasted a struct with something else in that
    // slot) was silently accepted. Cheap to check, so checked.
    if (in->reserved != 0) {
      return fail(&c, FUSE_ERR_INVALID, "E-ABI-SETUP-RESERVED",
                 "fuse_setup_options: reserved must be 0");
    }
    // entries.def's own comment on my_component: an old caller's smaller
    // struct_size never allocated this member, so it is read ONLY when
    // struct_size_covers it -- exactly the write-side guard's own mirror,
    // on the read side this time (abi-design.md rule 2).
    const char* my_component =
        struct_size_covers(in->struct_size, offsetof(fuse_setup_options, my_component) +
                                                sizeof(in->my_component))
            ? in->my_component
            : nullptr;
    if (in->communicator != 0) {
#if !FUSE_HAVE_MPI
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-SETUP-NOMPI",
                 "fuse_setup_options: a non-zero communicator was supplied, but this "
                 "build has no MPI");
#endif
      c.communicator = in->communicator;
#if FUSE_HAVE_MPI
      // my_component's own comment (entries.def): form the real base
      // communicator via fuse::transport::connect() when the loaded
      // graph's own LaunchSpec for this component says so -- the ONE
      // thing that keeps a C or Fortran caller from ever having to link
      // or call into C++ to take part in a multi-executable run.
      if (my_component != nullptr) {
        const auto cid = c.graph.find_component(my_component);
        if (!cid.has_value()) {
          return fail(&c, FUSE_ERR_INVALID, "E-ABI-SETUP-COMPONENT",
                     std::string("fuse_setup_options: my_component names '") +
                         my_component + "', which the loaded graph does not declare");
        }
        // D61 round 3 (KNOWN_ISSUES.md): kept for initialize() to split
        // the component-local reconciliation communicator by -- see
        // context.hpp's own comment on `my_component`/`collective_comm_f`.
        c.my_component = cid;
        const fuse::LaunchMode mode = c.graph.component(*cid).launch.mode;
        if (mode == fuse::LaunchMode::Mpmd || mode == fuse::LaunchMode::Connect) {
          fuse::transport::ConnectOptions copts;
          copts.my_component = *cid;
          const fuse::Result<fuse::transport::ConnectedWorld> connected =
              fuse::transport::connect(c.graph, static_cast<int>(in->communicator), copts);
          if (!connected.ok()) {
            return fail(&c, FUSE_ERR_RUNTIME, std::string(connected.error().code),
                       connected.error().message);
          }
          c.communicator = connected.value().base_comm_f;
          // destroy() releases it (context.hpp's own comment on this
          // field): until disconnect() runs, this program and its partner
          // are connected, and neither can finalize on its own.
          c.connected = connected.value();
        }
        // InProcess/Plugin/LowFive: nothing to do -- in->communicator
        // already is the base communicator, exactly as when my_component
        // is NULL (fuse::transport::connect() itself refuses to be
        // called for any of the three, for the identical reason).
      }
#endif
      // D14 (KNOWN_ISSUES.md), found while implementing base_communicator():
      // set ONLY here, after every early return above has had its chance to
      // fire -- a FAILED connect() (Mpmd/Connect launch) must leave
      // communicator_set FALSE, never stranded true with the caller's own
      // raw, UNMERGED communicator. Before this fix, a failed connect()
      // left communicator_set true (set eagerly, above the connect() call)
      // with c.communicator still the caller's own unmerged in->communicator
      // -- harmless everywhere that already gated on c.state (setup() itself
      // returned a failure, so no caller proceeds), but exactly wrong for
      // base_communicator()'s own `formed` flag, whose contract is "a
      // TRUSTWORTHY base communicator exists", not "setup() was called".
      c.communicator_set = true;
    } else if (my_component != nullptr) {
      // Review 2026-10-11 AB-4: this whole my_component handling (above)
      // lived inside `if (in->communicator != 0)`, so a caller that named
      // an Mpmd/Connect component but left communicator == 0 fell through
      // silently -- the run proceeded single-process on a graph that
      // declares a multi-executable launch, with no refusal at all.
      // my_component's own comment (entries.def) requires a non-zero
      // communicator for exactly these two LaunchModes, to form the real
      // base communicator from; InProcess/Plugin need no communicator at
      // all, so only Mpmd/Connect are refused here.
      const auto cid = c.graph.find_component(my_component);
      if (cid.has_value()) {
        // D61 round 3: stored even on this no-communicator path -- see
        // context.hpp's own comment -- though collective_comm_f (below,
        // initialize()) only ever forms when communicator_set is ALSO
        // true, so this has no effect on a single-process run beyond
        // recording which component this process named.
        c.my_component = cid;
        const fuse::LaunchMode mode = c.graph.component(*cid).launch.mode;
        if (mode == fuse::LaunchMode::Mpmd || mode == fuse::LaunchMode::Connect) {
          return fail(&c, FUSE_ERR_INVALID, "E-ABI-SETUP-COMPONENT",
                     std::string("fuse_setup_options: my_component names '") +
                         my_component + "', whose LaunchSpec is Mpmd or Connect and "
                         "therefore requires a non-zero communicator to form the real "
                         "base communicator from (communicator == 0 was supplied)");
        }
      }
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

// Unit 3 (S6 phase 3, Conformance): the reverse of impl_register_field
// above -- FUSE hands back a view of its own storage for `in->handle`,
// rather than the caller handing FUSE a pointer. Before initialize(), or
// for a field with no window scope at all, that storage is exactly the
// native buffer register_field() was given (c.fields' own copy); after
// initialize(), a window-scoped field instead gets the window path's own
// private buffer, through Runner::field_window_view() (window_samples'
// own comment, entries.def, has the full account of which arc that is).
fuse_status impl_view_field(fuse_ctx* ctx, const fuse_field_ref* in,
                           fuse_field_view* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    const std::uint32_t declared_out_size = out->struct_size;

    if (in->handle < 0 ||
        static_cast<std::uint32_t>(in->handle) >= c.graph.field_count()) {
      return fail(&c, FUSE_ERR_NOTFOUND, "E-ABI-VIEW-FIELD-HANDLE",
                 "fuse_view_field: handle does not name a field in the "
                 "loaded graph");
    }
    const fuse::FieldId field{static_cast<std::uint32_t>(in->handle)};

    const double* data = nullptr;
    std::uint64_t count = 0;
    std::int32_t window_samples = 0;
    if (c.runner.has_value()) {
      const fuse::Result<fuse::Runner::FieldWindowView> wv =
          c.runner->field_window_view(field);
      if (!wv.ok()) {
        return fail(&c, FUSE_ERR_RUNTIME, std::string(wv.error().code),
                   wv.error().message);
      }
      if (wv.value().window_samples > 0) {
        data = wv.value().data;
        count = wv.value().count;
        window_samples = wv.value().window_samples;
      }
    }
    if (data == nullptr) {
      // Not window-scoped (or not initialized yet): the native storage
      // this context's own register_field() was given, same as
      // impl_register_field binds from above.
      const fuse::Result<fuse::FieldRef> ref = c.fields.get(field);
      if (!ref.ok()) {
        return fail(&c, FUSE_ERR_INVALID, std::string(ref.error().code),
                   ref.error().message);
      }
      data = ref.value().data();
      count = static_cast<std::uint64_t>(ref.value().size());
    }

    const fuse::FieldSpec& spec = c.graph.field(field);
    // `count` from the window branch above is slot.total() = entities *
    // window_samples (the arc's own type.components is spec.components *
    // window_samples, and a window member's field is constrained to
    // spec.components == 1 -- src/runtime/iteration.cpp's own
    // E-RT-WINDOW-LAYOUT check -- so this division is exact, never a
    // truncation); from the ordinary branch, `count` is already entities
    // and window_samples is 0, so dividing by 1 is a no-op.
    const std::uint64_t entities =
        count / static_cast<std::uint64_t>(window_samples > 0 ? window_samples : 1);
    // Reported the SAME way an ordinary multi-component field already
    // is: `components` widened by the window's own sample count,
    // `count` unchanged (entities), array-of-structures (component_
    // stride = 1, stride = components) -- which for a window view is
    // exactly the entity-major, then-sample layout this entry's own
    // window_samples doc measures against Runner::Impl::resolve_
    // regions()'s priming loop (`data[e * K + s]`, src/runtime/
    // iteration.cpp).
    out->struct_size   = sizeof(fuse_field_view);
    out->handle         = in->handle;
    out->element_type   = FUSE_ELEMENT_F64;
    out->memspace       = FUSE_MEMSPACE_HOST;
    out->ownership      = FUSE_OWNERSHIP_OWNED;
    out->components     = spec.components *
                          static_cast<int32_t>(window_samples > 0 ? window_samples : 1);
    out->count          = entities;
    out->owned_count    = out->count;
    out->stride         = out->components;
    out->component_stride = 1;
    out->data           = const_cast<double*>(data);
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_field_view, window_samples) +
                               sizeof(out->window_samples))) {
      out->window_samples = window_samples;
    }
    return FUSE_OK;
  });
}

// Forward declaration: defined further down, grouped with reconcile_
// collective_status() (both are "agree a verdict across the base or a
// component-local communicator" glue, KNOWN_ISSUES.md D61 and D15), but
// needed here because impl_initialize_local() calls it before that
// point in the file.
fuse_status reconcile_plugin_agreement(fuse_ctx& c, int rank, fuse_status local_status,
                                       const char* fn) noexcept;

fuse_status impl_initialize_local(fuse_ctx* ctx, const fuse_null*, fuse_null*) {
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
    // KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
    // 2026-10-22): a priori, before any of the work below runs at all --
    // fuse::abi_detail::plugin_mount_agreement_needs_a_missing_spanning_
    // communicator() (include/fuse/abi/plugin_agreement.hpp) is a pure
    // function of this process's own already-loaded graph and its own
    // my_component (set at setup() time, read here, never recomputed),
    // so every process that will ever call fuse_initialize() on this
    // graph decides the SAME refusal, from the SAME bytes, with nothing
    // to reconcile and no rank left to hang waiting for a peer -- see
    // that header's own module comment for why a collective check cannot
    // do this job at all (there is, by construction, no communicator
    // that spans everyone to be collective OVER).
    {
      std::optional<fuse::LaunchMode> my_component_mode;
      if (c.my_component.has_value()) {
        my_component_mode = c.graph.component(*c.my_component).launch.mode;
      }
      if (fuse::abi_detail::plugin_mount_agreement_needs_a_missing_spanning_communicator(
              c.graph, my_component_mode, c.communicator_set)) {
        std::string plugin_name;
        std::string connect_name;
        for (std::uint32_t i = 0; i < c.graph.component_count(); ++i) {
          const fuse::ComponentDecl& decl = c.graph.component(fuse::ComponentId{i});
          if (plugin_name.empty() && decl.launch.mode == fuse::LaunchMode::Plugin) {
            plugin_name = decl.name;
          }
          if (connect_name.empty() && decl.launch.mode == fuse::LaunchMode::Connect) {
            connect_name = decl.name;
          }
        }
        return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-PLUGIN-NO-SPANNING-COMM",
                   "fuse_initialize: component '" + plugin_name +
                       "' is launched as Plugin, and this graph also declares component '" +
                       connect_name + "' launched as Connect (a separate-executable MPMD with "
                       "no common parent communicator until its own handshake); this process's "
                       "own my_component does not call fuse::transport::connect(), so there is "
                       "no way to confirm its base communicator spans the Connect-launched "
                       "executable, and the plugin-mount agreement (KNOWN_ISSUES.md D15) cannot "
                       "be reconciled across executables that have never shared a communicator. "
                       "Workaround: host the plugin component in a process that is itself "
                       "Connect-launched (so its own connect() call forms the spanning "
                       "communicator), or declare the multi-executable component as Mpmd instead "
                       "of Connect (Mpmd's MPI_COMM_WORLD already spans every program before "
                       "fuse_setup() is ever called, docs/user/providers.md's own known failure "
                       "case).");
      }
    }
    // This process's own (rank, num_ranks) -- NativeMeshBackend's own
    // construction argument, needed before the mesh-loading loop below
    // can run at all. Derived from the communicator fuse_setup_options()
    // recorded, if any (consumed here, same as the RunnerConfig block
    // further down uses it); (0, 1) otherwise -- correct by construction
    // for the single-process case (context.hpp's own comment on
    // `backend`), and the only case this sandbox can ever execute.
    int rank = 0, num_ranks = 1;
#if FUSE_HAVE_MPI
    if (c.communicator_set) {
      const MPI_Comm base = MPI_Comm_f2c(static_cast<MPI_Fint>(c.communicator));
      if (base == MPI_COMM_NULL || MPI_Comm_rank(base, &rank) != MPI_SUCCESS ||
          MPI_Comm_size(base, &num_ranks) != MPI_SUCCESS) {
        return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-COMM",
                   "fuse_initialize: could not read this process's rank/size "
                   "from the communicator fuse_setup_options() recorded");
      }
    }
#endif
    c.backend.emplace(rank, num_ranks);
    for (std::uint32_t i = 0; i < c.graph.mesh_count(); ++i) {
      const fuse::Result<void> r =
          c.backend->load(c.graph.mesh(fuse::MeshId{i}), c.graph);
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
    const fuse::RegionSizes sizes = fuse::make_region_sizes(c.graph, *c.dag, *c.backend);
    const fuse::ScheduleResult sched =
        fuse::schedule(c.graph, *c.dag, *c.clocks, sizes);
    if (!sched.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-SCHEDULE",
                 "fuse_initialize: the graph does not schedule:\n" +
                     sched.diags.to_string());
    }
    c.plan = sched.plan;

    // Wire every graph component to either the test-injected real
    // fuse::Component (driven in-process, through fuse_run()), a
    // LaunchMode::Plugin component mounted from its own shared object
    // (S6 phase 4, "Adapter host" track -- also driven in-process: a
    // provider shares the host's communicator exactly like an InProcess
    // component, launch.hpp's own doc), or a fresh ExternalStepComponent
    // (driven through fuse_begin_step()/fuse_end_step()); see context.hpp's
    // header comment.
    c.ext.assign(c.graph.component_count(), ExternalState{});
    c.external_count = 0;
    // KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
    // 2026-10-22, option (b)): the Plugin branch below no longer
    // `return fail(...)`s immediately on a hosting rank's own mount/
    // extract/register failure -- every such early return is
    // restructured to record its verdict here and `break` the loop
    // instead, so that EVERY rank (hosting or not) still reaches the
    // one-collective reconciliation after the loop, reconcile_plugin_
    // agreement() below. Without this restructuring, a hosting rank that
    // failed would return from inside the loop before a non-hosting rank
    // (which never fails locally -- there is nothing for it to fail on)
    // ever learned about it, exactly the gap this round closes
    // (KNOWN_ISSUES.md D15's own "Open after the mount-everywhere fix"
    // paragraph).
    fuse_status plugin_mount_status = FUSE_OK;
    bool saw_plugin_component = false;
    for (std::uint32_t i = 0; i < c.graph.component_count(); ++i) {
      const fuse::ComponentId id{i};
      const fuse::ComponentDecl& decl = c.graph.component(id);
      // A ComponentKind::Coupler entry is a placement, not a physics
      // participant: it has no register_fields()/advance() to supply, gets
      // no Tick, and is never stepped through fuse_begin_step()/
      // fuse_end_step() (src/runtime/runner.cpp's own step-list builder
      // skips it for the identical reason, in the comment right above its
      // own `if (decl.kind == ComponentKind::Coupler) continue;`). Before
      // this fix, this loop built an ExternalStepComponent for it anyway,
      // which that same runner.cpp step-list builder then rejected with
      // E-RT-COMPONENT ("component 'coupler' is not declared in the
      // coupling graph") the moment any graph loaded through this ABI had
      // an add_coupler() placement -- found this session, driving the
      // Brusselator participant graph (examples/brusselator/participants/
      // participant_graph.hpp) through fuse_coupler_abi() for the first
      // time anywhere in the suite.
      if (decl.kind == fuse::ComponentKind::Coupler) continue;
      const std::string& name = decl.name;
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
      } else if (decl.launch.mode == fuse::LaunchMode::Plugin) {
        // KNOWN_ISSUES.md D15 ("Registry hosting" track, owner decision
        // 2026-10-22, verbatim: "NO. This should not happen."): a rank
        // that is not a member of this component's own RankSet must
        // never dlopen() its shared object at all -- not even to
        // immediately discard the result, which is what EVERY rank did
        // before this round (Registry::mount_dynamic() below was called
        // unconditionally). `!c.communicator_set` keeps every single-
        // process caller's own behaviour exactly unchanged: there is
        // only one rank, and it always hosts whatever it is handed --
        // the SAME "assume hosted everywhere" rule provider_adapter.
        // cpp's own `hosted` local variable already documents, for the
        // identical reason.
        saw_plugin_component = true;
        const bool hosted = !c.communicator_set || decl.ranks.contains(rank);
        std::unique_ptr<fuse::Component> comp;
        if (hosted) {
          const fuse::Result<void> mounted =
              c.plugin_registry.mount_dynamic(name, decl.launch.plugin_path);
          if (!mounted.ok()) {
            // KNOWN_ISSUES.md D15 ("Registry agreement" track): recorded,
            // not returned -- see the `break` right after for why, and
            // this loop's own header comment above for the rule every
            // one of this branch's four early-return sites now follows.
            plugin_mount_status = fail(&c, FUSE_ERR_RUNTIME, std::string(mounted.error().code),
                                       mounted.error().message);
            break;
          }
          fuse::Result<std::unique_ptr<fuse::Component>> extracted =
              c.plugin_registry.extract(name);
          if (!extracted.ok()) {
            plugin_mount_status = fail(&c, FUSE_ERR_RUNTIME, std::string(extracted.error().code),
                                       extracted.error().message);
            break;
          }
          comp = std::move(extracted).value();
          // KNOWN_ISSUES.md D15 round 3: tell the adapter which rank it is
          // on the REAL base communicator, if any -- see provider_adapter.
          // hpp's own comment on set_hosting_rank() and D24 (this file's
          // own `rank` above, the SAME value backend construction used).
          // Always a ProviderAdapter here (mount_dynamic() above only ever
          // constructs one); the dynamic_cast exists so a future second
          // concrete type mounted the same way fails loudly, not silently.
          if (c.communicator_set) {
            auto* adapter = dynamic_cast<fuse::abi_detail::ProviderAdapter*>(comp.get());
            if (adapter == nullptr) {
              plugin_mount_status = fail(&c, FUSE_ERR_RUNTIME, "E-ABI-PROVIDER-TYPE",
                                         "fuse_initialize: component '" + name +
                                             "' was mounted via mount_dynamic() but is not a "
                                             "ProviderAdapter");
              break;
            }
            adapter->set_hosting_rank(rank);
          }
        } else {
          // KNOWN_ISSUES.md D15 ("Registry hosting"): build an UNHOSTED
          // ProviderAdapter shell directly, bypassing Registry::
          // mount_dynamic() entirely -- no dlopen(), no dlsym(), no
          // fuse_provider_abi() call, and create()/describe()/bind()/
          // advance()/destroy() never reach the plugin's own shared
          // object from this rank. register_fields() (called the same
          // way below for every branch) recognises the shell and
          // registers exactly the empty views the OLD, mount-everywhere
          // path already produced -- see that function's own comment
          // for the one thing this loses (docs/user/providers.md, "A
          // plugin launched on a subset").
          comp = std::make_unique<fuse::abi_detail::ProviderAdapter>(name);
        }
        const fuse::Result<void> r = comp->register_fields(c.fields, c.graph);
        if (!r.ok()) {
          plugin_mount_status = fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                                     r.error().message);
          break;
        }
        c.components.add(std::move(comp));
        c.ext[i].in_process = true;
      } else {
        auto ext_comp = std::make_unique<ExternalStepComponent>(name);
        const fuse::Result<void> r = ext_comp->register_fields(c.fields, c.graph);
        if (!r.ok()) {
          return fail(&c, FUSE_ERR_RUNTIME, std::string(r.error().code),
                     r.error().message);
        }
        c.components.add(std::move(ext_comp));
        c.ext[i].in_process = false;
        ++c.external_count;
        // KNOWN_ISSUES.md D12: no separate per-component tick count is
        // kept any more -- fuse::Runner::resume_period() (through the
        // SAME fuse::ClockSet-derived schedule every component, in-process
        // or external, is built from) is now the single source of when
        // this component is due, so there is nothing here left to
        // validate that the real scheduler has not already checked itself
        // when it built the plan above.
      }
    }

    // KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
    // 2026-10-22, option (b)): the ONE collective this round adds, run
    // exactly once per fuse_initialize() call and ONLY when the graph
    // declares a Plugin component -- `saw_plugin_component` is set above
    // inside the loop's own Plugin branch and nowhere else, so a graph
    // with none triggers NO extra collective at all, satisfying this
    // round's own requirement. See reconcile_plugin_agreement()'s own
    // doc comment (below) for why this cannot deadlock the way
    // KNOWN_ISSUES.md D61's REVERTED round did: fuse_initialize() has no
    // per-step cadence to mismatch, and every early return the Plugin
    // branch above used to take was restructured (this loop's own header
    // comment) so every rank that reaches this point has finished the
    // SAME wiring loop over the SAME graph.
    if (saw_plugin_component) {
      const fuse_status reconciled =
          reconcile_plugin_agreement(c, rank, plugin_mount_status, "fuse_initialize");
      if (reconciled != FUSE_OK) return reconciled;
    } else if (plugin_mount_status != FUSE_OK) {
      // Defensive only: plugin_mount_status can only be set inside the
      // Plugin branch above, which also sets saw_plugin_component in the
      // SAME iteration before it can fail -- this should never run on
      // the real setup()->initialize() path.
      return plugin_mount_status;
    }

    // Unit 3 (S6 phase 3, Conformance): a LowFive-bound channel's own
    // ExternalDelivery instruction (src/ir/rules_channel.cpp's
    // lower_channel()) suspends the Runner's own coroutine chain for the
    // IMPORTING component (deliver_external(), src/runtime/iteration.cpp)
    // -- which only resumes through fuse_begin_step()/fuse_end_step(), so
    // an importer driven in-process (c.ext[...].in_process, set just
    // above) would leave the suspension permanently pending: it never
    // calls either entry point. Refused here, at initialize(), rather
    // than discovered as a hang -- the same reasoning E-ABI-STEP-IN-
    // PROCESS already applies to an in-process component NAMED directly
    // in a fuse_begin_step()/fuse_end_step() call (resolve_external()
    // above in this file), just checked against the graph instead of a
    // caller's own argument, since nothing the application passes names
    // the importer here.
    for (std::size_t ch = 0; ch < c.graph.channel_count(); ++ch) {
      const fuse::Channel& channel =
          c.graph.channel_at(fuse::ChannelId{static_cast<std::uint32_t>(ch)});
      if (!c.graph.field(channel.source).hdf5_path.has_value()) continue;
      const fuse::ComponentId importer = c.graph.owner_of(channel.target);
      if (importer.value() < c.ext.size() && c.ext[importer.value()].in_process) {
        return fail(&c, FUSE_ERR_INVALID, "E-ABI-DELIVERY-INPROCESS",
                   "fuse_initialize: channel '" + channel.name +
                       "' is LowFive-bound (its source field sets "
                       "FieldSpec::hdf5_path), but its importer '" +
                       c.graph.component(importer).name +
                       "' is driven in-process; an ExternalDelivery "
                       "suspension can only resume through "
                       "fuse_begin_step()/fuse_end_step(), which an "
                       "in-process component never calls");
      }
    }

    fuse::RunnerConfig config{};
    if (c.communicator_set) {
      config.base_communicator = static_cast<int>(c.communicator);
      // A communicator means the components may live on different ranks,
      // and data between ranks moves only through the "mpi" transport --
      // the default "copy" is the single-address-space one. Setting the
      // communicator alone, as this block did until 2026-10-09, builds the
      // exchange plan and then refuses at the first Transport instruction:
      // `a CommGraph-backed Transport plan requires the "mpi" transport
      // backend, but the configured transport is 'copy'`, measured on the
      // host from every MPMD participant at once. The in-process C++
      // callers (test/parallel/connect/connect_common.hpp) have always set
      // all three together; the ABI was written when every participant it
      // drove was in one process, and nothing exercised this until the
      // Brusselator's three programs did.
      //
      // The transport's own setup is the Runner's `transport_setup` hook,
      // since `Transport&` cannot express a communicator handle: the hook
      // hands the base communicator to the concrete MpiTransport, which
      // duplicates it (collective, every rank, in the same order -- the
      // Runner calls the hook exactly once, before any instruction).
      config.transport = "mpi";
#if FUSE_HAVE_MPI
      const int base_f = static_cast<int>(c.communicator);
      config.transport_setup = [base_f](fuse::Transport& t) -> fuse::Result<void> {
        auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
        if (mpi == nullptr) {
          return fuse::Failure{"E-ABI-TRANSPORT",
                               "fuse_initialize: the \"mpi\" transport was requested but "
                               "the registry built something else"};
        }
        return mpi->set_communicator(base_f);
      };
#endif  // FUSE_HAVE_MPI  (without it a non-zero communicator was already refused in setup)
    }
    // D61 round 4 (KNOWN_ISSUES.md): the component-local communicator
    // reconcile_collective_status() (below) reconciles initialize()/
    // begin_step()/end_step()/run() over -- formed EXACTLY ONCE, here,
    // inside the SAME `if (c.communicator_set)` guard the Runner's own
    // transport configuration above already uses, so it can never be
    // confused with a communicator fuse::abi_test::force_communicator_
    // set() fakes (that seam only ever runs AFTER a real initialize()
    // has already completed -- see context.hpp's own comment on
    // `collective_comm_f`, and this entry's REVERTED paragraph for the
    // host crash that conflating the two caused).
    //
    // ROUND 3's OWN VERSION OF THIS BLOCK WAS WRONG, MEASURED ON THE HOST
    // (KNOWN_ISSUES.md D61's own round-4 paragraph carries the stack
    // samples). It called `fuse::linalg::make_participant_set_reducer()`
    // -- written for a REGION, where every rank of the base communicator
    // passes the IDENTICAL participant list, so "am I in it" (colour 0)
    // vs "not" (MPI_UNDEFINED) is the only distinction that can ever
    // exist. Here, each rank passes its OWN component's RankSet, not a
    // shared one: a U-rank calls it with U's own ranks, a V-rank with
    // V's own -- and a rank is trivially always a member of its OWN
    // component's set, so EVERY rank computed colour 0 and
    // `MPI_Comm_split` placed U, V and the coupler into ONE shared
    // communicator -- exactly the base-communicator reconciliation
    // this entry's round-2 revert already measured as a deadlock, by a
    // different route. The lesson this page's own abi-design.md records
    // ("a collective that pairs calls across components can only run on
    // a communicator whose every member makes the SAME sequence of
    // calls") applies to the SPLIT ITSELF, not only to the reconcile
    // call it feeds.
    //
    // THE FIX: `colour` must be DISTINCT PER COMPONENT, not a uniform 0,
    // so `MPI_Comm_split` partitions the base communicator by WHICH
    // component each rank plays, not by "did I pass a valid set at all"
    // (every rank always does). `my_component`'s own `ComponentId::
    // value()` is exactly that -- stable, distinct per component, known
    // identically on every rank from the SAME loaded graph. `key` orders
    // ranks within the resulting sub-communicator by this rank's
    // position in `my_component`'s own declared `RankSet` (unchanged
    // from round 3). A rank with NO component at all (`my_component`
    // never named at setup() time -- e.g. accept mode's bundled coupler-
    // placeholder rank when it is not itself named my_component; see
    // examples/brusselator/participants/u_main.c's own placeholder
    // branch and lowfive_attach_decision.hpp's account of the identical
    // posture) passes `MPI_UNDEFINED` and gets `MPI_COMM_NULL` back --
    // `collective_comm_f` stays 0, "nothing to reconcile", precisely
    // the semantics `ParticipantSetReducerResult::comm_f` documented for
    // the SAME convention, reused here without the type this function no
    // longer calls.
    //
    // `MPI_Comm_split` IS COLLECTIVE OVER THE WHOLE BASE COMMUNICATOR:
    // every rank must call it TOGETHER, including one passing
    // `MPI_UNDEFINED` -- round 3's own version called this ONLY `if (c.
    // my_component.has_value())`, which would itself have left a
    // componentless rank never entering the split at all, hanging every
    // OTHER rank that did. Fixed by moving the `my_component.has_value()`
    // check INSIDE (deciding `colour`), not around, the call.
#if FUSE_HAVE_MPI
    if (c.communicator_set) {
      const MPI_Comm base = MPI_Comm_f2c(static_cast<MPI_Fint>(c.communicator));
      int base_rank = 0;
      if (base == MPI_COMM_NULL || MPI_Comm_rank(base, &base_rank) != MPI_SUCCESS) {
        return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-COMM-SPLIT",
                   "fuse_initialize: could not read this rank's own position in the "
                   "base communicator to split a component-local communicator from it");
      }
      int colour = MPI_UNDEFINED;
      int key = 0;
      std::optional<std::uint32_t> component_value;
      int position = -1;
      if (c.my_component.has_value()) {
        const fuse::ComponentDecl& mc = c.graph.component(*c.my_component);
        position = mc.ranks.index_of(base_rank);
        component_value = c.my_component->value();
      }
      // fuse::abi_detail::decide_split_colour_key (include/fuse/abi/
      // collective_reconcile.hpp): the pure half of this decision, tested
      // directly at tier 0 with no MPI dependency -- see that header's own
      // doc comment for why round 3's version of this call (`make_
      // participant_set_reducer`) computed the wrong thing here, and what
      // `position < 0` defensively covers (should never happen on the
      // real setup()->initialize() path).
      if (const auto ck = fuse::abi_detail::decide_split_colour_key(component_value, position);
          ck.has_value()) {
        colour = ck->colour;
        key = ck->key;
      }
      MPI_Comm sub = MPI_COMM_NULL;
      const int split_rc = MPI_Comm_split(base, colour, key, &sub);
      if (split_rc != MPI_SUCCESS) {
        return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-INIT-COMM-SPLIT",
                   "fuse_initialize: MPI_Comm_split (for the per-component "
                   "reconciliation communicator) returned error code " +
                       std::to_string(split_rc));
      }
      if (sub != MPI_COMM_NULL) {
        // A fresh communicator, set to return errors rather than abort
        // the job: every OTHER MPI call this file makes is already
        // reported through fail() when it fails, and a communicator this
        // function itself just created is exactly the one place where
        // doing the same costs nothing (no caller outside this file ever
        // sees the handle to disagree about its error-handling policy).
        MPI_Comm_set_errhandler(sub, MPI_ERRORS_RETURN);
        c.collective_comm_f = static_cast<int>(MPI_Comm_c2f(sub));
      }
      // colour == MPI_UNDEFINED: sub is MPI_COMM_NULL, collective_comm_f
      // stays 0 -- reconcile_collective_status() already treats that as
      // "nothing to reconcile".
    }
#endif  // FUSE_HAVE_MPI
    // Unit 2b (D19), step (d): fuse::abi_test::force_this_process()'s own
    // seam -- std::nullopt (the default) leaves config exactly what it
    // was before this field existed.
    config.force_this_process = c.test_force_this_process;
    c.runner.emplace(c.graph, *c.dag, *c.plan, *c.backend);
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

std::int32_t first_external_component_due_at_period_start(fuse_ctx& c) {
  for (const fuse::ComponentId cid : c.runner->components_due_at_period_start()) {
    if (cid.value() < c.ext.size() && !c.ext[cid.value()].in_process) {
      return static_cast<std::int32_t>(cid.value());
    }
  }
  return -1;
}

/// Unit 3 (S6 phase 3, Conformance), the lead's own corrected ruling
/// (2026-10-04), ITSELF SUPERSEDED this round (item 2 -- see
/// `impl_begin_step()`'s own is_delivery_due branch above for the full
/// account): resolves the `next_component` value a CACHED request
/// (`c.next_request`, assumed to already have a value) reports --
/// shared by `fuse_is_coupling_ongoing()`, `fuse_end_step()`, and
/// `fuse_begin_step()`'s own is_delivery_due branch, the three entry
/// points that report this field. `next_component` has exactly ONE
/// meaning everywhere: the component whose `fuse_begin_step()` the
/// Runner expects next. Under the OLD (single-direction, importer-
/// only-suspends) delivery design, an `is_delivery_due` request was
/// "purely informational" -- its own `component` named the importer,
/// but the CALLER needed no action beyond skipping ahead to the next
/// REAL advance, resolved via a forward scan, `Runner::
/// next_real_advance_after()`. That rule is now wrong: `delivery_field`
/// (entries.def's own comment) requires the CALLER to act (H5Dwrite if
/// producing the named field, H5Dread if consuming it) before its own
/// next `fuse_begin_step()`, and `deliver_external()` (src/runtime/
/// iteration.cpp) now suspends for BOTH the exporter and the importer
/// of a channel, each its own request naming the SPECIFIC component
/// that must act -- so `req.component` is always meaningful and always
/// reported directly, exactly like a real (non-delivery) request;
/// there is no longer a separate "real next stepper" to scan ahead to.
std::int32_t resolve_cached_next_component(fuse_ctx& c) {
  const fuse::ExternalAdvanceRequest& req = *c.next_request;
  const auto cid = c.graph.find_component(req.component);
  return cid.has_value() ? static_cast<std::int32_t>(cid->value()) : -1;
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
    const std::uint32_t declared_out_size = out->struct_size;
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
    // Defect 3, refined (entries.def's comment on this field). This
    // function must stay side-effect-free -- it always has been, and a
    // caller is free to call it as many times as it likes between steps.
    // Resolving a MID-PERIOD answer would mean calling
    // prime_next_request(), which calls runner->resume_period(): a
    // genuine side effect that actually executes ordinary instructions,
    // not a cache lookup -- unsafe here for exactly the reason an
    // earlier version of this fix demonstrated by corrupting
    // AbiParticipant.ExternallyDrivenWindowMemberMatchesInProcessRunner
    // (fuse_is_coupling_ongoing() called once after a period had already
    // fully run, which re-entered resume_period() and silently re-ran
    // part of the window's iteration on top of its own converged
    // output). BUT a period's own START is not a mid-period question at
    // all: the compiled instruction stream is static once schedule() has
    // built it, and the first externally driven component of a period is
    // the SAME one every period (the schedule's own common-period
    // invariant) -- knowable from the plan alone, for free, no
    // resumption required. So: if something is already cached from a
    // prior fuse_begin_step()/fuse_end_step() THIS period, report that
    // (free, as before); otherwise -- nothing cached means exactly "a
    // period is about to start, whichever one" -- read the answer
    // straight off Runner::components_due_at_period_start() (a pure,
    // non-resuming query over the plan) and report the first entry that
    // is NOT in_process, i.e. the first one this process's own caller
    // could possibly be asked to step. -2 ("undetermined without
    // stepping") is no longer reachable from here at all; it stays
    // fuse_begin_step()'s own answer for the genuinely mid-period case
    // (this component's own advance() has not run yet, so even the
    // compiled plan cannot say what comes right after it).
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_ongoing, next_component) +
                               sizeof(out->next_component))) {
      if (c.external_count == 0) {
        out->next_component = -1;
      } else if (c.next_request.has_value()) {
        out->next_component = resolve_cached_next_component(c);
      } else {
        // Unreachable for a well-formed graph with external_count > 0:
        // every externally driven component has a substep at period-time
        // 0 (either an ordinary Tick{k=0,...} or a window member's first
        // instance's own first substep), so
        // first_external_component_due_at_period_start()'s own loop
        // always finds one and its -1 default is never actually used
        // here. Left as the safe fallback rather than an assert,
        // matching this function's own "fail gracefully, never a new
        // hard-failure mode" convention elsewhere in this file.
        out->next_component = first_external_component_due_at_period_start(c);
      }
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

/// KNOWN_ISSUES.md D12: calls `runner->resume_period()` if nothing is
/// already cached in `c.next_request` (see context.hpp's own header
/// comment), and returns the request it finds -- WITHOUT consuming it;
/// the caller decides whether it matches `name` and, if so, moves it out
/// itself. Returns false (and fills `out_code`/`out_message`) only for a
/// genuine runtime failure; a name mismatch is the CALLER's concern,
/// checked afterwards, because leaving the cache populated on a mismatch
/// is exactly what lets the correctly-named begin_step() that follows see
/// the same request.
///
/// The lead's own ruling (2026-10-04), SUPERSEDING an earlier one the
/// SAME day that treated `resume_period()` returning nothing as always
/// unreachable: that was right for `fuse_is_coupling_ongoing()`'s own
/// "nothing cached" branch and for a GENUINE mismatch rejected earlier
/// (resolve_external() already refuses `name` if it is in-process or
/// unknown, and open_component's own check in impl_begin_step rules out
/// a second begin_step() for the same component with no intervening
/// end_step()), but wrong for THIS call site once a scheduling-order fix
/// (`OpKind::ExternalDelivery`'s own comment, `include/fuse/ir/op.hpp`)
/// made it possible for `resume_period()`, resumed from a delivery
/// proxy's own suspend point, to run an iteration region through to real
/// convergence and on into the period's own end: every externally driven
/// component still ticks at least once per period (the scheduler's own
/// common-period invariant), but THIS period already collected that
/// tick earlier, before the proxy was cached -- resuming past the proxy
/// can legitimately find nothing left to do here, not because the
/// invariant broke, but because this period is simply over. That is a
/// legitimate outcome, not an error: `*rolled_over` is set, this
/// period's own bookkeeping is completed exactly as `fuse_run()` would
/// at a period's end (`++c.periods_completed`), and the next period is
/// started and its own first request primed immediately, in the SAME
/// call -- `fuse_begin_step()`'s own caller (below) decides, from
/// whether the newly primed request's component matches the one it was
/// asked for, whether to open a step as usual or report `fuse_step_
/// report::step_opened = 0` instead (that field's own comment, `entries.
/// def`, has the full account). Only if THAT second `resume_period()`
/// ALSO finds nothing is the common-period invariant genuinely violated,
/// which remains unreachable for a well-formed graph and is still
/// reported as `E-ABI-STEP-ORDER`.
///
/// The lead's own ruling (2026-10-04), the SAME day, closing the one way
/// this rollover could otherwise run away: nothing above bounds how many
/// times a period can roll into the next one, and a graph whose window
/// never fails to converge (the common case once state reaches a fixed
/// point) would let it happen without end, exactly the hang a caller
/// driving by `fuse_is_coupling_ongoing()`'s own `ongoing` flag must
/// never see from inside the library. So: before the SECOND
/// `resume_period()` call below actually starts a new period,
/// `c.has_end_time` is checked the SAME way `fuse_is_coupling_ongoing()`
/// already does (`c.plan->period() * Rational{c.periods_completed} <
/// c.end_time`) -- using the JUST-INCREMENTED `periods_completed`,
/// because that product is exactly the NEW period's own start time, the
/// same quantity `fuse_run()`'s own `periods == -1` loop tests before
/// calling `run_period()` again. At or past `end_time`, the run is
/// simply over: `c.next_request` is left empty (not an error) and this
/// function still returns `true` -- `fuse_begin_step()`'s own caller
/// reads the empty cache as "nothing more, ever" (distinct from a
/// request that merely mismatches `C`) and reports `step_opened = 0`,
/// `next_component = -1`, exactly what `fuse_is_coupling_ongoing()`
/// already reports as `ongoing = 0` once resumed this far. Without a
/// `has_end_time` bound at all, there is nothing to refuse against --
/// the run is unbounded by the caller's own declared intent, and rolling
/// forever is exactly what an unbounded run means.
bool prime_next_request(fuse_ctx& c, const char* fn, bool* rolled_over,
                        std::string* out_code, std::string* out_message) {
  *rolled_over = false;
  if (c.next_request.has_value()) return true;
  // The lead's own ruling (2026-10-04): this call might be reached fresh
  // (not through the rollover below) after a PRIOR call already closed
  // the run's own last period -- fuse_end_step()'s own "the period just
  // ran to completion" branch (below, this file) leaves c.next_request
  // empty exactly that way for a graph where it is reachable at all.
  // Checked here, before EITHER resume_period() call, so a begin_step()
  // the caller makes anyway (ignoring fuse_is_coupling_ongoing()'s own
  // `ongoing = 0`) never starts a new period's own coroutine at all --
  // the SAME `c.plan->period() * Rational{c.periods_completed} <
  // c.end_time` test that function already uses.
  if (c.has_end_time &&
      !(c.plan->period() * fuse::Rational{c.periods_completed} < c.end_time)) {
    return true;
  }
  fuse::Result<std::optional<fuse::ExternalAdvanceRequest>> outcome =
      c.runner->resume_period(c.components);
  if (!outcome.ok()) {
    *out_code = outcome.error().code;
    *out_message = outcome.error().message;
    return false;
  }
  if (!outcome.value().has_value()) {
    // This period is over (see this function's own comment above for
    // why that is expected here, not a violation of the "ticks at least
    // once" invariant): complete its bookkeeping exactly as fuse_run()
    // would at a period's end, then start the next one and prime ITS
    // OWN first request, in this same call -- UNLESS the next period's
    // own start would be at or past the run's own declared end_time
    // (this function's own comment above), in which case the run is
    // simply over and c.next_request is left empty.
    ++c.periods_completed;
    *rolled_over = true;
    if (c.has_end_time) {
      const fuse::Rational now =
          c.plan->period() * fuse::Rational{c.periods_completed};
      if (!(now < c.end_time)) {
        c.next_request.reset();
        return true;
      }
    } else if (c.communicator_set) {
      // KNOWN_ISSUES.md D57. No end_time was declared at all (fuse_setup
      // ()'s own fuse_setup_options::has_end_time left at its zero
      // default) -- the ONLY bound the branch above can test against is
      // absent, so there is nothing here to check a new period's own
      // start against. Unlike the exceeded-end_time case, this is not
      // "the run is over": a well-formed caller bounds an undeclared-
      // end_time run itself, by its own loop over a known period count
      // (examples/brusselator/participants/u_main.c's own `for (p = 0;
      // p < periods; ++p)`, mirrored by v_main.f90 and by coupler_main.c
      // 's own fuse_run(periods) call) -- but THIS function has no way
      // to learn that count, and fuse_run()'s own loop never asks this
      // one to look ahead at all (Runner::run() calls run_period()
      // exactly `periods` times and stops; it never peeks into period
      // `periods` the way the eager continuation below would). Starting
      // period c.periods_completed's own coroutine here regardless would
      // run REAL instructions of a period the caller's own bound never
      // authorized.
      //
      // Gated on `c.communicator_set` -- NOT on `!c.has_end_time` alone
      // -- because the hazard this guards against is specifically a
      // base-communicator collective (OpKind::Constrain's reduction,
      // gated on `config.base_communicator.has_value()`, KNOWN_ISSUES.md
      // D10) that an in-process-only PARTNER rank bounded by
      // Runner::run(periods) will never also enter, since its own loop
      // already stopped -- a hazard that exists only when a base
      // communicator is configured at all. `c.communicator_set` is set
      // once, identically, from `fuse_setup_options::communicator` at
      // setup() time (not derived from any rank-local runtime state), so
      // this is the same value on every rank that shares the
      // communicator -- consistent with the standing rule that a
      // collective path never returns on a locally computed condition.
      // Without this narrowing, the single-process conformance case
      // (`AbiParticipant.MismatchedRolloverNamesTheRealPeriodStartComp
      // onent`, test/abi/participant_test.cpp) -- which never configures
      // a communicator and relies on EXACTLY the eager continuation this
      // branch would otherwise skip, to resolve a still-converging
      // window's real next-period owner ahead of the caller asking for
      // it, per "the lead's own ruling" that test's own header cites --
      // regressed the first time this fix was written unconditionally
      // (measured: the full tier-0 ctest suite caught it, `abi.
      // participant` FAILED where it had passed on HEAD). With no
      // communicator at all, there is no partner rank to disagree with,
      // so the eager continuation below is harmless and that test's own
      // behaviour is preserved exactly, unchanged by this entry.
      //
      // Reported exactly like the exceeded-end_time case: c.next_request
      // stays empty, fuse_begin_step()'s own caller (below) sees
      // next_component == -1 / step_opened == 0 for THIS component, for
      // THIS now-completed period -- a true, non-speculative answer, not
      // a lie, since the rolled-over count above already reflects this
      // period finishing -- and the decision whether a period beyond it
      // should run is left entirely to whether the caller's own loop
      // makes another fuse_begin_step() call at all. A caller that does
      // (because its own loop has more periods to go) gets the fresh
      // period's real first request from THAT later call's own ordinary
      // (non-rollover) resume_period(), exactly as if this rollover had
      // never been entered -- unchanged behaviour for every period but
      // the last one requested.
      c.next_request.reset();
      return true;
    }
    outcome = c.runner->resume_period(c.components);
    if (!outcome.ok()) {
      *out_code = outcome.error().code;
      *out_message = outcome.error().message;
      return false;
    }
    if (!outcome.value().has_value()) {
      // Two periods in a row with no externally driven component to
      // stop at: THIS is the genuine invariant violation, unreachable
      // for a well-formed graph with at least one such component.
      *out_code = "E-ABI-STEP-ORDER";
      *out_message =
          std::string(fn) +
          ": the current period has no externally driven component left "
          "to step; this should be unreachable for a graph with at least "
          "one such component";
      return false;
    }
  }
  c.next_request = std::move(outcome.value());
  return true;
}

fuse_status impl_begin_step_local(fuse_ctx* ctx, const fuse_step_request* in,
                                  fuse_step_report* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    const std::uint32_t declared_out_size = out->struct_size;
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
    // KNOWN_ISSUES.md D12: at most one externally driven component may be
    // open at a time -- see context.hpp's own header comment for why.
    if (c.open_component.has_value() && *c.open_component != cid) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-STEP-ALREADY-OPEN",
                 std::string("fuse_begin_step: component '") + in->component +
                     "' cannot begin a step while component '" +
                     c.graph.component(*c.open_component).name +
                     "' still has one open; call its fuse_end_step() first");
    }
    // Review 2026-10-11 AB-12: `rolled_over` was declared `false` and
    // then OR'd with `this_call_rolled_over` immediately below with
    // nothing in between able to set it -- a no-op vestige of an
    // earlier loop (this is the only prime_next_request() call in this
    // function). `this_call_rolled_over` carries the real value from
    // here on; no behaviour change.
    bool this_call_rolled_over = false;
    if (!prime_next_request(c, "fuse_begin_step", &this_call_rolled_over,
                            &err_code, &err_message)) {
      return fail(&c, FUSE_ERR_RUNTIME, err_code, err_message);
    }
    const bool rolled_over = this_call_rolled_over;
    // S6 phase 3 (Conformance, Unit 3b), item 2 of this round,
    // SUPERSEDING the lead's own earlier (2026-10-04) ruling that this
    // call "consumes EVERY pending delivery proxy it encounters while
    // resuming, WHATEVER component each one names" and drains silently
    // to the first REAL advance request: that was right as long as a
    // delivery proxy carried no information beyond its own existence,
    // but is wrong now that `delivery_field` (entries.def's own
    // comment) must name WHICH field each one is about -- draining
    // silently discards every proxy's own field identity except the
    // last one found, the exact bug this round's own report traced
    // (KNOWN_ISSUES.md D14): `deliver_external()` (src/runtime/
    // iteration.cpp) now suspends for the EXPORTER as well as the
    // importer, so a single channel's own delivery produces TWO
    // back-to-back is_delivery_due requests whenever both ends are
    // hosted on this process, with nothing real between them for a
    // draining loop to stop at.
    //
    // The corrected rule: EVERY is_delivery_due request is its own
    // externally visible stop, reported in full and consumed ONE at a
    // time -- never looped past. `resolve_cached_next_component()`
    // (this file, above) is the SAME forward-scan `fuse_end_step()`/
    // `fuse_is_coupling_ongoing()` already use for an identical cached
    // request, so all three entry points resolve `next_component`
    // identically for a delivery-due request; it must run BEFORE
    // `c.next_request.reset()` below, since it reads directly off
    // `c.next_request`.
    if (c.next_request.has_value() && c.next_request->is_delivery_due) {
      const std::int32_t delivery_next = resolve_cached_next_component(c);
      const std::int32_t delivery_field_value = c.next_request->delivery_field;
      c.next_request.reset();
      // Deliberately NOT re-primed here (an earlier version of this
      // round's own fix did, and was wrong): `prime_next_request()`
      // calls `resume_period()`, which RESUMES THE COROUTINE -- for an
      // importer's own is_delivery_due stop, whatever comes next
      // (another sweep, or the window's own convergence check) reads
      // the import buffer the CALLER has not yet refreshed (its own
      // H5Dread, this report's entire reason for existing, has not run
      // yet). Resuming here reads that buffer BEFORE the caller's own
      // update -- observed as a window "converging" after a single
      // stale-data sweep instead of its real iteration count
      // (`RunExternalDeliveryMatchesInProcessRunner`'s own sweep-count
      // diagnostic caught this: harness max sweep index 0 against a
      // reference of 11). Left reset, the CALLER's own next
      // `fuse_begin_step()` re-primes it (the existing call at this
      // function's own top, above) -- AFTER that caller has acted on
      // THIS report, the only point priming is safe. A caller needing
      // `next_component` before then (e.g. `fuse_is_coupling_ongoing()`
      // sandwiched in between) must use THIS report's own value, not
      // re-query -- the harness's own loop does exactly that (`next_
      // hint`, test/verification/brusselator_lowfive_harness_test.cpp).
      out->struct_size         = sizeof(fuse_step_report);
      out->requires_checkpoint = 0;
      out->requires_restore    = 0;
      out->step                = 0;
      out->iteration            = 0;
      out->reserved             = 0;
      out->t_begin               = to_abi_time(fuse::Rational{0});
      out->t_end                 = to_abi_time(fuse::Rational{0});
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, next_component) +
                                 sizeof(out->next_component))) {
        out->next_component = delivery_next;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, step_opened) +
                                 sizeof(out->step_opened))) {
        out->step_opened = 0;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, period) +
                                 sizeof(out->period))) {
        out->period = static_cast<std::int32_t>(c.periods_completed);
      }
      // delivery_pending's own comment, entries.def: this report exists
      // BECAUSE a LowFive-bound delivery proxy was just consumed -- the
      // one case, distinct from an ordinary rollover, this bit signals.
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_pending) +
                                 sizeof(out->delivery_pending))) {
        out->delivery_pending = 1;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_field) +
                                 sizeof(out->delivery_field))) {
        out->delivery_field = delivery_field_value;
      }
      return FUSE_OK;
    }
    if (!c.next_request.has_value()) {
      // The lead's own ruling (2026-10-04): prime_next_request()'s own
      // rollover refused to start a period past the run's declared
      // end_time (that function's own comment has the full account) --
      // this run is simply over, the SAME fact fuse_is_coupling_ongoing()
      // already reports as `ongoing = 0` once resumed this far (the
      // SAME `c.plan->period() * Rational{c.periods_completed} <
      // c.end_time` test). No step opens on ANY component; -1 names
      // nothing because there is nothing left, not because of a
      // mismatch (fuse_step_report::step_opened's own comment,
      // entries.def).
      out->struct_size         = sizeof(fuse_step_report);
      out->requires_checkpoint = 0;
      out->requires_restore    = 0;
      out->step                = 0;
      out->iteration            = 0;
      out->reserved             = 0;
      out->t_begin               = to_abi_time(fuse::Rational{0});
      out->t_end                 = to_abi_time(fuse::Rational{0});
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, next_component) +
                                 sizeof(out->next_component))) {
        out->next_component = -1;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, step_opened) +
                                 sizeof(out->step_opened))) {
        out->step_opened = 0;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, period) +
                                 sizeof(out->period))) {
        out->period = static_cast<std::int32_t>(c.periods_completed);
      }
      // delivery_pending's own comment, entries.def: the run simply
      // ended (nothing left, not a suspension) -- nothing is pending.
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_pending) +
                                 sizeof(out->delivery_pending))) {
        out->delivery_pending = 0;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_field) +
                                 sizeof(out->delivery_field))) {
        out->delivery_field = -1;
      }
      return FUSE_OK;
    }
    const fuse::ExternalAdvanceRequest& req = *c.next_request;
    // Unit 3a (S6 phase 3, Conformance), this round's own correction
    // (2026-10-06): a MATCHED rollover -- `req.component == in->
    // component`, but `rolled_over` is true -- used to fall through to
    // the step_opened=1 path below unconditionally ("whether or not a
    // period rollover happened first to reach it", this function's own
    // comment there, now corrected). MEASURED (this session, UCore::
    // step()'s own temporary per-call trace, test/verification/
    // brusselator_lowfive_harness_test.cpp's own ExternalDeliveryMatches
    // InProcessRunner): that let THIS SAME call run the new period's
    // own first physics substep before any caller outside the Runner
    // could observe the period having changed at all (fuse_is_coupling_
    // ongoing()'s own `period` field does not advance until the
    // FOLLOWING call, by which point the physics already ran) -- there
    // was no external vantage point from which a window member's own
    // committed state could be read at the true period boundary. A
    // matched rollover is no longer special-cased: it is folded into
    // the SAME `treat_as_mismatch` branch an unmatched rollover already
    // takes (step_opened=0, request left unconsumed, `next_component`
    // names `req.component` so the caller's own next call picks the
    // SAME request back up and actually runs it) -- one extra, cheap
    // begin_step()/end_step() round trip at a period boundary, in
    // exchange for every window member's own state being observable
    // from outside at the period's own true end, matched or not.
    const bool treat_as_mismatch = (req.component != in->component) || rolled_over;
    if (treat_as_mismatch) {
      if (req.component != in->component && !rolled_over) {
        // Not consumed: the stream's own next tick belongs to a
        // DIFFERENT externally driven component, named here so the
        // caller can step it first. Unaffected by this day's rollover
        // correction: no period boundary was crossed to reach this
        // request, so it is the ordinary out-of-turn mistake it always
        // was.
        return fail(&c, FUSE_ERR_STATE, "E-ABI-STEP-ORDER",
                   std::string("fuse_begin_step: component '") +
                       in->component +
                       "' asked out of turn; the Runner's instruction "
                       "stream expects component '" + req.component +
                       "' next");
      }
      // Resuming past a proxy rolled this call into a NEW period -- a
      // legitimate outcome (a time-window member on one side of an
      // implicit coupling iteration cannot know from outside whether
      // that iteration has converged), not an error, whether the new
      // period's own first request names a DIFFERENT component (the
      // `req.component != in->component` disjunct) or the SAME one
      // (this fix's own disjunct, `rolled_over` alone -- see the
      // comment above `treat_as_mismatch`). No step opens on ANY
      // component either way; the caller's own loop body must call
      // fuse_begin_step() again with the component named below, which
      // in the matched case is simply `in->component` again -- the
      // SAME, still-unconsumed `req` runs on that very next call.
      out->struct_size         = sizeof(fuse_step_report);
      out->requires_checkpoint = 0;
      out->requires_restore    = 0;
      out->step                = 0;
      out->iteration            = 0;
      out->reserved             = 0;
      out->t_begin               = to_abi_time(fuse::Rational{0});
      out->t_end                 = to_abi_time(fuse::Rational{0});
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, next_component) +
                                 sizeof(out->next_component))) {
        // req is a real advance request (is_delivery_due false), by the
        // SAME common-period invariant prime_next_request()'s own
        // comment relies on: a period's own first instruction is never
        // a delivery, so resolve_cached_next_component()'s delivery-due
        // branch is not needed here.
        const auto resolved_cid = c.graph.find_component(req.component);
        out->next_component =
            resolved_cid.has_value()
                ? static_cast<std::int32_t>(resolved_cid->value())
                : -1;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, step_opened) +
                                 sizeof(out->step_opened))) {
        out->step_opened = 0;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, period) +
                                 sizeof(out->period))) {
        out->period = static_cast<std::int32_t>(c.periods_completed);
      }
      // delivery_pending's own comment, entries.def: `treat_as_mismatch`
      // is reached here ONLY via `rolled_over` (the other disjunct,
      // `req.component != in->component && !rolled_over`, already
      // returned via `fail()` above with E-ABI-STEP-ORDER) -- an
      // ORDINARY period rollover (prime_next_request()'s own "period is
      // over, start the next one" mechanism), never a delivery: this
      // round's own correction (item 2) intercepts every is_delivery_due
      // request earlier, in its own dedicated branch above, and returns
      // from there directly -- `c.next_request` reaching this point is
      // never delivery-due by construction. 0 here, always.
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_pending) +
                                 sizeof(out->delivery_pending))) {
        out->delivery_pending = 0;
      }
      if (struct_size_covers(declared_out_size,
                             offsetof(fuse_step_report, delivery_field) +
                                 sizeof(out->delivery_field))) {
        out->delivery_field = -1;
      }
      return FUSE_OK;
    }
    st.request = req;
    st.open = true;
    c.open_component = cid;
    c.next_request.reset();

    out->struct_size         = sizeof(fuse_step_report);
    out->requires_checkpoint = req.requires_checkpoint ? 1 : 0;
    out->requires_restore    = req.requires_restore ? 1 : 0;
    out->step                = req.ctx.step;
    out->iteration            = req.ctx.iteration;
    out->reserved             = 0;
    out->t_begin               = to_abi_time(req.ctx.t_begin);
    out->t_end                 = to_abi_time(req.ctx.t_end);
    // Defect 3 (entries.def's comment on this field). Unlike
    // fuse_is_coupling_ongoing()/fuse_end_step(), begin_step() cannot
    // resolve a real answer here: THIS component's own advance() has not
    // run yet, so the Runner has not been asked to resume past it, and the
    // real "next" answer only exists once the MATCHING fuse_end_step()
    // calls resume_period() again. -2 says exactly that, distinct from -1
    // ("period complete") so a caller never confuses "ask again later"
    // with "nothing left this period".
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, next_component) +
                               sizeof(out->next_component))) {
      out->next_component = -2;
    }
    // step_opened's own comment, entries.def: 1 here always -- reached
    // only when `!treat_as_mismatch`, i.e. `req.component == in->
    // component && !rolled_over` (this round's own correction, above):
    // a step just opened on the component this call was asked for, and
    // no period rollover was needed to reach it at all -- a MATCHED
    // rollover no longer falls through to this branch; see
    // `treat_as_mismatch`'s own comment for why.
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, step_opened) +
                               sizeof(out->step_opened))) {
      out->step_opened = 1;
    }
    // period's own comment, entries.def: set here too, for uniformity
    // with every other return path in this function -- `rolled_over`
    // is always false on this branch now (above), so this is simply
    // `c.periods_completed`'s own current value, nothing left for a
    // caller to otherwise miss.
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, period) +
                               sizeof(out->period))) {
      out->period = static_cast<std::int32_t>(c.periods_completed);
    }
    // delivery_pending's own comment, entries.def: a step just opened
    // on the component this call was asked for -- nothing is pending.
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, delivery_pending) +
                               sizeof(out->delivery_pending))) {
      out->delivery_pending = 0;
    }
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, delivery_field) +
                               sizeof(out->delivery_field))) {
      out->delivery_field = -1;
    }
    return FUSE_OK;
  });
}

fuse_status impl_end_step_local(fuse_ctx* ctx, const fuse_step_request* in,
                                fuse_step_report* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status osz = check_struct_size(out);
    if (osz != FUSE_OK) return osz;
    const std::uint32_t declared_out_size = out->struct_size;
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

    // Echo exactly what the matching begin_step() reported -- the request
    // this component's own step was opened with, not whatever the Runner
    // may have already resumed to below.
    out->struct_size         = sizeof(fuse_step_report);
    out->requires_checkpoint = st.request.requires_checkpoint ? 1 : 0;
    out->requires_restore    = st.request.requires_restore ? 1 : 0;
    out->step                = st.request.ctx.step;
    out->iteration            = st.request.ctx.iteration;
    out->reserved             = 0;
    out->t_begin               = to_abi_time(st.request.ctx.t_begin);
    out->t_end                 = to_abi_time(st.request.ctx.t_end);
    // step_opened's own comment, entries.def: always 1 from
    // fuse_end_step() -- this call only ever echoes a step that WAS
    // open (the E-ABI-STEP-NOT-OPEN refusal above rules out the only
    // other case), so it is never the one place `step_opened = 0` is
    // reported.
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, step_opened) +
                               sizeof(out->step_opened))) {
      out->step_opened = 1;
    }

    // KNOWN_ISSUES.md D12: resume the Runner PAST this component's own
    // (no-op) advance(), to the next externally driven component's
    // suspension point or the period's end -- caching whichever it finds
    // for the NEXT fuse_begin_step() to read, exactly as
    // prime_next_request() does. next_request is known empty here: this
    // component was open, which only happens after a begin_step() that
    // consumed it.
    const fuse::Result<std::optional<fuse::ExternalAdvanceRequest>> outcome =
        c.runner->resume_period(c.components);
    if (!outcome.ok()) {
      return fail(&c, FUSE_ERR_RUNTIME, std::string(outcome.error().code),
                 outcome.error().message);
    }
    st.open = false;
    c.open_component.reset();
    if (outcome.value().has_value()) {
      c.next_request = std::move(outcome.value());
    } else {
      // The period just ran to completion.
      c.next_request.reset();
      ++c.periods_completed;
    }
    // Defect 3 (entries.def's comment on this field). Resolved from the
    // SAME resume_period() outcome this call already ran to refill
    // c.next_request above -- no extra cost, and no extra failure mode
    // beyond the one `outcome.ok()` already guards. Read c.next_request
    // itself rather than `outcome.value()`: the branch above may have
    // moved the engaged optional's contents out of it, and an
    // already-moved-from fuse::ExternalAdvanceRequest is not a value this
    // function may read again.
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, next_component) +
                               sizeof(out->next_component))) {
      if (!c.next_request.has_value()) {
        out->next_component = -1;
      } else {
        out->next_component = resolve_cached_next_component(c);
      }
    }
    // period's own comment, entries.def: the SAME counter
    // fuse_ongoing::period reports, read after the resume_period() call
    // above that may just have incremented it (the "period just ran to
    // completion" branch).
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, period) +
                               sizeof(out->period))) {
      out->period = static_cast<std::int32_t>(c.periods_completed);
    }
    // delivery_pending's own comment, entries.def: always 0 from
    // fuse_end_step(), the same reason step_opened is always 1 here --
    // this call only ever echoes a step that WAS open, never the call
    // that suspends at a delivery proxy (that is always reported by a
    // FOLLOWING fuse_begin_step(), its own dedicated is_delivery_due
    // branch, above).
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, delivery_pending) +
                               sizeof(out->delivery_pending))) {
      out->delivery_pending = 0;
    }
    if (struct_size_covers(declared_out_size,
                           offsetof(fuse_step_report, delivery_field) +
                               sizeof(out->delivery_field))) {
      out->delivery_field = -1;
    }
    return FUSE_OK;
  });
}

fuse_status impl_run_local(fuse_ctx* ctx, const fuse_run_request* in, fuse_null*) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (c.state != fuse_ctx::State::Initialized) {
      return fail(&c, FUSE_ERR_STATE, "E-ABI-RUN-STATE",
                 "fuse_run: the context is not initialized");
    }
    const fuse_status sz = check_struct_size(in);
    if (sz != FUSE_OK) return sz;
    // Unit 2b (D19), step (d): narrowed from "the GRAPH declares an
    // externally driven component anywhere at all" to "one IS HOSTED ON
    // THIS PROCESS" -- Runner::hosts_component()'s own rule (step (a)'s
    // hosting formula, generalised past window members). A process
    // hosting none at all -- a dedicated coupler rank, or a placeholder
    // rank in a bundled MPMD run (KNOWN_ISSUES.md D19's own evidence 4)
    // -- has nothing here for fuse_begin_step()/fuse_end_step() to drive
    // in the first place, so fuse_run() is the ONLY way it can ever
    // execute its own share of the period's collectives (Transport/
    // Epoch) at all; refusing it unconditionally just because SOME
    // OTHER rank's own component is external is exactly the gap that
    // left coupler_main.c's/u_main.c's own placeholder ranks calling
    // neither fuse_run() nor fuse_begin_step()/fuse_end_step() between
    // fuse_initialize() and fuse_destroy() -- never executing the
    // Transports the physics ranks were waiting on.
    bool external_hosted_here = false;
    for (std::size_t i = 0; i < c.ext.size(); ++i) {
      const fuse::ComponentId cid{static_cast<std::uint32_t>(i)};
      // A ComponentKind::Coupler entry gets no ExternalStepComponent at
      // all (fuse_initialize()'s own `if (decl.kind == ComponentKind::
      // Coupler) continue;`) -- its own c.ext[i] stays at ExternalState{}
      // 's default, in_process == false, which means nothing here: it is
      // not "externally driven", it is simply never populated. Checking
      // hosts_component() for it anyway would refuse fuse_run() on a
      // rank that hosts ONLY a coupler placement, the exact case this
      // fix exists to allow.
      if (cid.value() < c.graph.component_count() &&
          c.graph.component(cid).kind == fuse::ComponentKind::Coupler) {
        continue;
      }
      if (c.ext[i].in_process) continue;
      if (c.runner->hosts_component(cid)) {
        external_hosted_here = true;
        break;
      }
    }
    if (external_hosted_here) {
      return fail(&c, FUSE_ERR_UNSUPPORTED, "E-ABI-RUN-EXTERNAL",
                 "fuse_run: an externally driven component (fuse_begin_step/"
                 "fuse_end_step) is hosted on this process; drive the run "
                 "with those instead");
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
                   "fuse_setup_options()");
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

// D14 (KNOWN_ISSUES.md), appended 2026-10-13: a local, non-collective read of
// c.communicator/c.communicator_set -- entries.def's own doc comment on this
// entry has the full account of why it exists (the merged base communicator
// Connect-mode's own accept/connect + MPI_Intercomm_merge forms, previously
// held only inside this opaque fuse_ctx and never returned to any caller).
// Legal at ANY context state, including before fuse_setup() is ever called
// (formed=0, communicator=0, matching context.hpp's own field defaults) --
// unlike fuse_is_coupling_ongoing() above, this reads nothing that depends
// on fuse_initialize() or fuse_setup() having run; `with_ctx()` is the only
// gate (a live, non-destroyed handle).
fuse_status impl_base_communicator(fuse_ctx* ctx, const fuse_null*,
                                   fuse_base_communicator_info* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (out == nullptr) return FUSE_ERR_INVALID;
    const fuse_status sz = check_struct_size(out);
    if (sz != FUSE_OK) return sz;
    out->struct_size  = sizeof(fuse_base_communicator_info);
    out->communicator = c.communicator_set ? c.communicator : 0;
    out->formed       = c.communicator_set ? 1 : 0;
    out->reserved     = 0;
    return FUSE_OK;
  });
}

/// Restart track round 2, item 4's participant-side request
/// (entries.def's own doc comment on this entry, fuse/runtime/
/// runner.hpp's Runner::request_restart_point()). A no-op, not an
/// error, before initialize(): no Runner exists yet to hold the flag,
/// and a participant calling this too early has asked for nothing this
/// entry can act on -- the SAME "nothing to do yet" posture
/// fuse_base_communicator() takes before setup() (formed=0).
fuse_status impl_request_restart_point(fuse_ctx* ctx, const fuse_null*,
                                       fuse_null* out) {
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    if (out != nullptr) {
      const fuse_status sz = check_struct_size(out);
      if (sz != FUSE_OK) return sz;
      out->struct_size = sizeof(fuse_null);
    }
    if (c.runner.has_value()) {
      c.runner->request_restart_point();
    }
    return FUSE_OK;
  });
}

// --------------------------------------------- D61 collective reconciliation
//
// KNOWN_ISSUES.md D61: begin_step()/end_step()/initialize()/run() each
// decided a refusal from this rank's own local state only, with nothing
// agreeing that decision across the rest of the component's ranks before
// returning -- violating the standing rule a collective path never
// returns on a locally computed condition (KNOWN_ISSUES.md D3, docs/dev/
// invariants.md I2/A4), which src/runtime/runner.cpp's own
// reconcile_epoch() already upholds for the C++ terminals, through the
// Epoch instruction. The ABI boundary meets the SAME contract here, by
// the same means A4 names for a terminal: one MPI_Allreduce of an
// encoded status before anyone acts on it, not a further round after the
// fact.
//
// ROUND 3, REPLACING A REVERTED FIX (KNOWN_ISSUES.md D61's own REVERTED
// paragraph carries the measurements). The first attempt reconciled over
// TWO wrong communicators:
//   1. The base communicator (`c.communicator`), which in a multi-
//      executable run spans every component (U, V and the coupler all
//      merged into one world) -- different components make different
//      numbers of begin_step()/end_step() calls per window, so an
//      Allreduce keyed to the WRONG set of ranks never finds a partner on
//      the other side and the whole run hangs. Fixed by reconciling over
//      a COMPONENT-LOCAL communicator instead (see impl_initialize_local()
//      above, where it is split once from the real base communicator by
//      `my_component`'s own RankSet). ROUND 3's OWN FIRST ATTEMPT AT THIS
//      reused `fuse::linalg::make_participant_set_reducer()` (D32's own
//      `MPI_Comm_split`) directly -- wrong for a DIFFERENT reason, measured
//      on the host and fixed in round 4: that function colours a rank by
//      "is it a member of THE participant list passed in", written for a
//      caller where every rank passes the SAME list; here, every rank
//      passes its OWN component's list and is trivially always a member of
//      it, so every rank computed colour 0 and landed back in ONE shared
//      communicator -- the identical bug, by a different route. Round 4's
//      own fix calls `MPI_Comm_split` directly, colouring by `my_component`'s
//      own `ComponentId::value()` (distinct per component) instead.
//   2. Reading `c.communicator`/`c.communicator_set` directly, which
//      `fuse::abi_test::force_communicator_set()` can set to a value that
//      is not a real MPI handle at all (its own doc comment: deliberately
//      callable AFTER a successful initialize(), to reach a LATER gate in
//      isolation, never meant to feed MPI itself) -- `MPI_Comm_f2c` on
//      that fake value, then `MPI_Comm_rank` on the result, aborted three
//      tier-0 rows on the host (MPICH's default error handler on an
//      invalid communicator; this sandbox has no MPI runtime at all, so
//      it could not have caught this). Fixed by reconciling over
//      `c.collective_comm_f` instead -- a SEPARATE field `force_
//      communicator_set()` never touches, which stays 0 (its default,
//      "nothing to reconcile") for every context that seam builds, BY
//      CONSTRUCTION: the real split only ever runs inside initialize()'s
//      own `if (c.communicator_set)` block, before the seam has had any
//      chance to set a fake value (or not at all, if the seam is the
//      ONLY thing that ever sets communicator_set on that context).
//
// WHICH ENTRIES, AND WHY SETUP() IS NOT ONE OF THEM. Four of entries.def's
// fifteen participant-table members: initialize, begin_step, end_step,
// run. `setup()` is explicitly EXCLUDED this round, unlike round 2:
// `my_component` -- the only thing that tells this function WHICH
// component's ranks to split a communicator by -- is an ARGUMENT to
// setup() itself (`fuse_setup_options::my_component`), resolved partway
// through its own body; there is no component-local communicator to
// reconcile setup()'s own verdict over until setup() has already decided
// one. Reconciling setup() over the raw, UNSPLIT base communicator (round
// 2's own choice) is exactly bug 1 above, generalised: setup() is called
// once per component, by definition, before any other component in the
// same merged world has necessarily called it too, so there is no single
// communicator setup() could reconcile over that is guaranteed to span
// only ITS OWN component's ranks. Left rank-local, same as every entry
// NOT in the list below: setup()'s own refusals are structural (a
// malformed `fuse_setup_options`, an unknown `my_component` name, a
// missing communicator for an Mpmd/Connect launch) -- the SAME bytes on
// every rank that is supposed to agree would also decide them the same
// way independently, which is why KNOWN_ISSUES.md D61's own first
// paragraph names the hazard as an OUT-OF-TURN or OMITTED call (begin_
// step/end_step), not a setup()-time configuration mismatch.
//
// Of the remaining four:
//   - initialize/begin_step/end_step are marked COLLECTIVE in entries.
//     def's own doc string (initialize "COLLECTIVE"; begin_step/end_step
//     "COLLECTIVE over the component's processes").
//   - run's own doc string is not marked with that literal word, but
//     states it "still executes this call's own share of the period's
//     collectives" even when refusing nothing itself, and KNOWN_ISSUES.md
//     D61 names it explicitly alongside the other four -- a rank that
//     refuses run() locally (E-ABI-RUN-EXTERNAL) while its peers proceed
//     into the period's own Transport/Epoch/Constrain collectives hangs
//     them exactly as an unreconciled begin_step()/end_step() refusal
//     does.
//   - The other eleven entries (now including setup) are NOT reconciled
//     here, for the reasons round 2's own version of this comment already
//     gave: none is marked COLLECTIVE in entries.def; each is a purely
//     local operation on this rank's own handle, copy of the (identically
//     loaded) graph, or storage; register_field was checked explicitly
//     (per the original round's own task) and found not collective
//     either; there is no "finalize" entry at all (destroy() is the
//     nearest analogue and is, per entries.def, a per-handle operation).
//
// THE RULE. reconcile_collective_status() below runs AFTER the entry's
// own impl_*_local() body has computed its local verdict, and BEFORE that
// verdict reaches the caller: every rank of the communicator being
// reconciled over contributes its own (fuse_status, rank) pair to one
// MPI_Allreduce with MPI_MINLOC (the one-call equivalent of "MIN over an
// encoded status" that ALSO reports which rank the minimum came from,
// needed to name the refusing rank in the rewritten message every other
// rank gets). FUSE_OK is 0 and every FUSE_ERR_* code is negative
// (entries.def), so the plain minimum already propagates any refusal
// over FUSE_OK, and two different ranks refusing with two different
// codes agree on the more negative one, deterministically -- the pure
// arithmetic lives in fuse::abi_detail::reconcile_local_statuses
// (include/fuse/abi/collective_reconcile.hpp), so this glue and its own
// tier-0 test (collective_reconcile_test.cpp) cannot drift apart on what
// "agreed" means.
//
// WHICH COMMUNICATOR, restated precisely: `c.collective_comm_f`,
// `int`-valued, `0` meaning "nothing to reconcile" (see context.hpp's
// own comment on the field for every way it stays 0). The TOP-LEVEL
// gate is still `c.communicator_set` -- a context with no communicator
// configured at all (single-process, or a tier-0 test that never calls
// force_communicator_set()) has nothing to reconcile, exactly as
// reconcile_epoch() treats an unconfigured base_communicator.
//
// WHAT THIS DOES NOT COVER. A rank that is live but passes an invalid
// `ctx` handle to a DIFFERENT call than its peers (so with_ctx() refuses
// E-ABI-HANDLE before any body runs) has no communicator this function
// can read at all -- there is no fuse_ctx to ask. That specific
// misuse -- confusing which handle to call, not a refusal this entry's
// own body decided -- is not rescued here; KNOWN_ISSUES.md D61's own
// entry records this as the residual scope, consistent with D61's "the
// ABI boundary... said nothing about it" being about THIS call's own
// refusal, not about a caller passing the wrong object entirely. A
// process whose own base communicator carries a real multi-rank run but
// never names `my_component` at setup() time also gets no reconciliation
// (`collective_comm_f` stays 0, with no component to split by) -- a
// narrow, documented gap: `my_component` is REQUIRED at setup() time for
// Mpmd/Connect launches already (E-ABI-SETUP-COMPONENT), so this can only
// arise for an InProcess/Plugin-only graph handed a real communicator
// directly, without going through fuse::transport::connect() at all.
//
// COST, ACCEPTED AS PERMANENT (the lead's own decision, round 2, restated
// here since it still applies): begin_step()/end_step() are step-boundary
// calls (abi-design.md's own "costs" section: "every entry point is a
// setup or step-boundary call, never per-degree-of-freedom"), so one
// small MPI_Allreduce added to each is bounded, not asymptotic -- but it
// IS one more real collective per step, on top of whatever Transport/
// Epoch/Constrain instructions the compiled plan already runs for that
// step. Follow-up, recorded and not pursued further: a cheaper mechanism
// would piggyback the encoded status onto a collective the step already
// performs instead of a separate MPI_Allreduce.
fuse_status reconcile_collective_status(fuse_ctx& c, fuse_status local_status,
                                        const char* fn) noexcept {
  if (!c.communicator_set) return local_status;  // nothing configured: rank-local, unchanged

  // Test-only seam (fuse::abi_test::force_fake_collective_peers): simulate
  // N ranks with no MPI runtime, through the SAME pure arithmetic the real
  // MPI glue below uses. Gated on communicator_set (above) exactly like
  // the real path, so setting fake peers with NO force_communicator_set()
  // call still reconciles nothing (test/abi/participant_test.cpp's own
  // NoCommunicatorConfiguredMeansNoReconciliationEvenWithFakePeersSet).
  if (c.test_fake_collective_self_rank.has_value()) {
    const int self = *c.test_fake_collective_self_rank;
    std::vector<std::int32_t> all;
    all.reserve(c.test_fake_collective_peer_statuses.size() + 1);
    for (std::size_t i = 0; i < c.test_fake_collective_peer_statuses.size() + 1; ++i) {
      if (static_cast<int>(i) == self) {
        all.push_back(local_status);
      } else {
        const std::size_t peer_index =
            i < static_cast<std::size_t>(self) ? i : i - 1;
        all.push_back(c.test_fake_collective_peer_statuses.at(peer_index));
      }
    }
    const fuse::abi_detail::ReconciledStatus agreed =
        fuse::abi_detail::reconcile_local_statuses(all);
    if (agreed.origin_rank == self) return local_status;  // this rank's own message stands
    if (agreed.status == FUSE_OK) return FUSE_OK;
    return fail(&c, static_cast<fuse_status>(agreed.status), "E-ABI-COLLECTIVE-REFUSED",
               std::string(fn) + ": rank " + std::to_string(agreed.origin_rank) +
                   " of this component refused (status " +
                   std::to_string(agreed.status) +
                   "); reconciled across every rank of the component's own "
                   "communicator so every rank returns the same code "
                   "(docs/dev/abi-design.md)");
  }

#if FUSE_HAVE_MPI
  if (c.collective_comm_f == 0) return local_status;  // no component-local
                                                       // communicator was ever formed
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(c.collective_comm_f));
  if (comm == MPI_COMM_NULL) return local_status;  // nothing sane to reconcile over
  int rank = 0;
  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-COLLECTIVE-COMM",
               std::string(fn) + ": MPI_Comm_rank failed while reconciling status "
                                  "across the component's ranks");
  }
  // MPI_2INT packs two plain `int`s; fuse_status is int32_t. Every
  // platform this library builds on has sizeof(int) == 4, so the two
  // agree, but it is cheap to say so rather than assume it.
  static_assert(sizeof(int) == sizeof(std::int32_t),
               "MPI_2INT packs two `int`s; fuse_status is int32_t -- the two must agree");
  struct { std::int32_t status; int rank; } in{local_status, rank}, out{0, 0};
  if (MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MINLOC, comm) != MPI_SUCCESS) {
    return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-COLLECTIVE-COMM",
               std::string(fn) + ": MPI_Allreduce(MPI_MINLOC) failed while "
                                  "reconciling status across the component's ranks");
  }
  if (out.rank == rank) return local_status;  // this rank's own message stands (incl. FUSE_OK)
  if (out.status == FUSE_OK) return FUSE_OK;
  return fail(&c, static_cast<fuse_status>(out.status), "E-ABI-COLLECTIVE-REFUSED",
             std::string(fn) + ": rank " + std::to_string(out.rank) +
                 " of this component refused (status " + std::to_string(out.status) +
                 "); reconciled across every rank of the component's own "
                 "communicator so every rank returns the same code "
                 "(docs/dev/abi-design.md)");
#else
  return local_status;
#endif
}

// KNOWN_ISSUES.md D15 ("Registry agreement" track, owner decision
// 2026-10-22, option (b)). Reconciles the plugin-mount verdict
// impl_initialize_local()'s own component-wiring loop (Plugin branch,
// above) computes, across EVERY rank of the BASE communicator
// (c.communicator) -- NOT the per-component collective_comm_f
// reconcile_collective_status() above uses, because collective_comm_f
// never forms for a graph whose my_component is unset (Plugin/InProcess
// launch never resolves my_component to anything Mpmd/Connect --
// entries.def's own my_component comment), which is exactly the shape a
// Plugin-only, direct-communicator graph takes -- the gap KNOWN_ISSUES.md
// D15's own "Open after the mount-everywhere fix" paragraph names and
// this function closes.
//
// CANNOT DEADLOCK THE WAY D61's ROUND-2 FIX DID (abi-design.md's own "How
// the participant ABI reconciles a collective refusal", "LESSON (round
// 3, the hard way)"): that fix reconciled begin_step()/end_step()/run()
// -- PER-STEP entries a multi-component graph's different components
// call a DIFFERENT NUMBER OF TIMES -- over the base communicator, so two
// ranks hosting different components never paired the same Allreduce
// call. fuse_initialize() has no such cadence to mismatch: it is called
// EXACTLY ONCE, synchronously, by every rank of the base communicator,
// before any step boundary exists at all -- there is no "which call
// number is this" question for two ranks to disagree about. The other
// half of what made D61's round 2 unsafe -- touching a FAKED
// (fuse::abi_test::force_communicator_set()) communicator handle with a
// real MPI_Comm_rank/MPI_Allreduce under FUSE_HAVE_MPI, which aborted
// three host rows -- is avoided the same way the pre-existing rank/
// num_ranks derivation at the top of impl_initialize_local() already
// relies on: every test that fakes a communicator with no real MPI
// behind it is, by established convention, excluded from MPI-enabled
// test binaries (test/abi/conformance_test.cpp's own
// NonHostingRankNeverDlopensThePluginOrCallsAnyOfItsEntries, "Registry
// hosting" track, is the precedent).
//
// Reuses fuse::abi_detail::reconcile_local_statuses (include/fuse/abi/
// collective_reconcile.hpp) for the MIN-with-origin-rank decision -- the
// SAME pure arithmetic reconcile_collective_status() above uses -- so the
// two reconciliations cannot disagree about what "agreed" means, only
// about which communicator and which local verdict they reconcile.
// `rank` is this rank's own position on `c.communicator`, ALREADY derived
// by impl_initialize_local() (the identical MPI_Comm_f2c/MPI_Comm_rank
// call this function would otherwise have to repeat) and passed in
// rather than re-read.
//
// Test-only seam: fuse::abi_test::force_fake_plugin_agreement_peers() --
// a DEDICATED pair of fields (context.hpp's own comment on
// test_fake_plugin_agreement_self_rank explains why this is not a reuse
// of force_fake_collective_peers()). test_plugin_agreement_reconcile_
// calls (context.hpp) is incremented on EVERY call, real or faked,
// before any early return below -- the white-box signal a tier-0 test
// uses to confirm this function was actually reached (impl_initialize_
// local()'s own restructuring did its job) versus merely that the
// outcome would have looked the same either way.
fuse_status reconcile_plugin_agreement(fuse_ctx& c, int rank, fuse_status local_status,
                                       const char* fn) noexcept {
  ++c.test_plugin_agreement_reconcile_calls;
  if (!c.communicator_set) return local_status;  // single process: nothing to reconcile

  if (c.test_fake_plugin_agreement_self_rank.has_value()) {
    const int self = *c.test_fake_plugin_agreement_self_rank;
    std::vector<std::int32_t> all;
    all.reserve(c.test_fake_plugin_agreement_peer_statuses.size() + 1);
    for (std::size_t i = 0; i < c.test_fake_plugin_agreement_peer_statuses.size() + 1; ++i) {
      if (static_cast<int>(i) == self) {
        all.push_back(local_status);
      } else {
        const std::size_t peer_index =
            i < static_cast<std::size_t>(self) ? i : i - 1;
        all.push_back(c.test_fake_plugin_agreement_peer_statuses.at(peer_index));
      }
    }
    const fuse::abi_detail::ReconciledStatus agreed = fuse::abi_detail::reconcile_local_statuses(all);
    if (agreed.origin_rank == self) return local_status;  // this rank's own message stands
    if (agreed.status == FUSE_OK) return FUSE_OK;
    return fail(&c, static_cast<fuse_status>(agreed.status), "E-ABI-PLUGIN-AGREEMENT-REFUSED",
               std::string(fn) + ": rank " + std::to_string(agreed.origin_rank) +
                   " refused the plugin-mount agreement (status " +
                   std::to_string(agreed.status) +
                   "); reconciled across every rank of the base communicator because the "
                   "graph declares a Plugin component (docs/dev/abi-design.md, "
                   "KNOWN_ISSUES.md D15)");
  }

#if FUSE_HAVE_MPI
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(c.communicator));
  if (comm == MPI_COMM_NULL) return local_status;  // defensive: impl_initialize_local()'s
                                                   // own derivation above already refused
                                                   // E-ABI-INIT-COMM before this point if so
  static_assert(sizeof(int) == sizeof(std::int32_t),
               "MPI_2INT packs two `int`s; fuse_status is int32_t -- the two must agree");
  struct { std::int32_t status; int rank; } in{local_status, rank}, out{0, 0};
  if (MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MINLOC, comm) != MPI_SUCCESS) {
    return fail(&c, FUSE_ERR_RUNTIME, "E-ABI-PLUGIN-AGREEMENT-COMM",
               std::string(fn) + ": MPI_Allreduce(MPI_MINLOC) failed while reconciling the "
                                  "plugin-mount agreement across the base communicator");
  }
  if (out.rank == rank) return local_status;  // this rank's own message stands (incl. FUSE_OK)
  if (out.status == FUSE_OK) return FUSE_OK;
  return fail(&c, static_cast<fuse_status>(out.status), "E-ABI-PLUGIN-AGREEMENT-REFUSED",
             std::string(fn) + ": rank " + std::to_string(out.rank) +
                 " refused the plugin-mount agreement (status " + std::to_string(out.status) +
                 "); reconciled across every rank of the base communicator because the "
                 "graph declares a Plugin component (docs/dev/abi-design.md, "
                 "KNOWN_ISSUES.md D15)");
#else
  return local_status;
#endif
}

fuse_status impl_initialize(fuse_ctx* ctx, const fuse_null* in, fuse_null* out) {
  const fuse_status local = impl_initialize_local(ctx, in, out);
  if (!is_live(ctx)) return local;  // no ctx to reconcile through; see "WHAT THIS DOES NOT COVER"
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    return reconcile_collective_status(c, local, "fuse_initialize");
  });
}

fuse_status impl_begin_step(fuse_ctx* ctx, const fuse_step_request* in,
                            fuse_step_report* out) {
  const fuse_status local = impl_begin_step_local(ctx, in, out);
  if (!is_live(ctx)) return local;
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    return reconcile_collective_status(c, local, "fuse_begin_step");
  });
}

fuse_status impl_end_step(fuse_ctx* ctx, const fuse_step_request* in,
                          fuse_step_report* out) {
  const fuse_status local = impl_end_step_local(ctx, in, out);
  if (!is_live(ctx)) return local;
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    return reconcile_collective_status(c, local, "fuse_end_step");
  });
}

fuse_status impl_run(fuse_ctx* ctx, const fuse_run_request* in, fuse_null* out) {
  const fuse_status local = impl_run_local(ctx, in, out);
  if (!is_live(ctx)) return local;
  return with_ctx(ctx, [&](fuse_ctx& c) -> fuse_status {
    return reconcile_collective_status(c, local, "fuse_run");
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
