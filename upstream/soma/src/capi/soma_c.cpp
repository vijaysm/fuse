// Implementation of SOMA's stable C ABI (spec 3.2). See include/soma/soma_c.h
// for the contract; this file is only about honouring it.
//
// Two rules govern every function here.
//
//   1. No exception escapes. `extern "C"` has no exception specification that
//      helps: an exception that propagates out of one of these functions is
//      std::terminate. The C++ core throws by design -- Rational's overflow
//      guards, FieldRef::at(), OperatorDAG::add_node()'s SSA check,
//      ClockSet's interval guard, ComponentSet::add() -- so the catch-all is
//      load-bearing, not ceremony. `entry()` below supplies it once.
//
//   2. No handle is dereferenced before it is proven live. Validation is a
//      lookup in a table of pointers, never a read through the pointer
//      itself, so a stale or garbage handle cannot fault or read freed
//      memory.

#include "soma/soma_c.h"

// The THIRD translation unit in the project that may name an MPI type, on
// exactly the same terms as the other two (src/mesh/moab_backend.cpp and
// src/transport/mpi_transport.cpp): soma_core already links MPI::MPI_CXX
// PRIVATE (src/CMakeLists.txt), and no SOMA header names an MPI type, so
// mpi.h stays off a consumer's include path. soma_c.h above is unchanged and
// still crosses the boundary with a plain int32_t Fortran handle.
//
// It is needed because soma_initialize() has to answer one question it cannot
// answer any other way: how many ranks will run this context's
// soma_end_step() decisions? See the parallel-safety gate there.
#ifdef SOMA_HAVE_MPI
#include <mpi.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/field/registry.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/graph/descriptor.hpp"
#include "soma/graph/validate.hpp"
#include "soma/graph/yaml_io.hpp"
#include "soma/ir/dag.hpp"
#include "soma/ir/lowering.hpp"
#include "soma/num/convergence.hpp"
#include "soma/num/kernels.hpp"
#include "soma/schedule/clock.hpp"

#ifndef SOMA_VERSION_STRING
#define SOMA_VERSION_STRING "0.0.0-unconfigured"
#endif

namespace {

enum class State { Empty, Loaded, Initialized };

/// Per-component driver state for the begin_step / end_step contract.
///
/// CONVERGENCE EARLY-EXIT (the fields below `in_step`). Populated by
/// soma_initialize() only when this component's governing iteration region
/// (the one that set `max_iterations`) has a *non-empty* ConvergenceCriterion
/// whose every measure resolves to exactly one channel of that region -- see
/// soma_initialize() for why "non-empty" and "resolves" are not guaranteed by
/// construction on this path, and soma_end_step() for how these are used.
/// When `converges` is false (the default, and every pre-existing graph),
/// soma_end_step() does exactly what it always has: sweep to max_iterations.
///
/// `criterion` points into `soma_ctx_s::graph`'s iteration-region table. That
/// table is populated once, by soma_load_graph()/soma_load_graph_yaml(),
/// before soma_initialize() ever runs, and is never mutated or reallocated
/// afterwards -- so the pointer is stable for the lifetime of the context,
/// the same lifetime CompRuntime itself has.
struct CompRuntime {
  std::int32_t iteration      = 0;  // index of the current sweep
  std::int32_t max_iterations = 1;  // 1 outside every iteration region
  bool         in_step        = false;

  bool                               converges = false;
  const soma::ConvergenceCriterion*  criterion = nullptr;

  // The distinct fields any measure of `criterion` names (deduplicated: two
  // measures on the same field share one snapshot), and, for each, the
  // previous-iterate snapshot needed to form the residual `current -
  // previous` -- every MeasureKind needs this, since iteration.cpp forms the
  // residual as the delta between sweeps regardless of kind; only the
  // reference norm differs by kind. Gathered in canonical (element, then
  // component) order via FieldRef::offset(), so it is correct for both AoS
  // and SoA registrations.
  std::vector<soma::FieldId>        snapshot_fields;
  std::vector<std::vector<double>>  snapshot_prev;
  // Parallel to `criterion->measures()`: index into snapshot_fields /
  // snapshot_prev for that measure's field.
  std::vector<std::size_t>          measure_slot;
  // Parallel to `criterion->measures()`; meaningful only where the measure's
  // kind is ResidualRelative, which divides by the residual of the window's
  // first sweep rather than by the current value's norm.
  std::vector<double>               measure_first_residual;

  // True once snapshot_prev holds "y0", the value the fields held before the
  // window's first sweep ran. Captured in soma_begin_step() when a fresh
  // window opens (iteration == 0) and invalidated the moment the window ends
  // (converged or exhausted), so a stale snapshot can never leak into the
  // next window.
  bool snapshots_valid    = false;
  // True once measure_first_residual[] holds the first sweep's residuals for
  // the window currently in progress.
  bool have_first_residual = false;
};

/// The calling thread's copy of the most recent error message.
///
/// Needed because soma_create() and any call made with an invalid handle have
/// no context to store into, and a C caller must still be able to find out
/// why. Per-thread rather than global so two threads driving two contexts do
/// not overwrite each other.
std::string& thread_error() {
  static thread_local std::string msg;
  return msg;
}

/// The set of live contexts.
///
/// Constructed on first use in static storage and deliberately never
/// destroyed. A function-local `static HandleTable t;` would be destroyed at
/// exit in reverse construction order; a caller whose *global* object holds a
/// soma_ctx and calls soma_destroy() from its destructor would then validate
/// against a destroyed table. Placement-new into a static array avoids both
/// that and a leak report under LeakSanitizer (the storage is a static
/// object, and the set's nodes are reachable from it).
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

/// True when `v` is representable as a std::size_t on this platform. Written
/// with `if constexpr` rather than a plain comparison so the 64-bit case does
/// not produce a tautological-comparison warning.
[[nodiscard]] bool fits_size(std::uint64_t v) noexcept {
  if constexpr (sizeof(std::size_t) >= sizeof(std::uint64_t)) {
    (void)v;
    return true;
  } else {
    return v <= static_cast<std::uint64_t>(
                    std::numeric_limits<std::size_t>::max());
  }
}

}  // namespace

/// The context. Defined here, not in the header: C callers must not be able
/// to see, size, or allocate one.
struct soma_ctx_s {
  State                             state = State::Empty;
  soma::CouplingGraph               graph;
  soma::FieldRegistry               fields;
  std::optional<soma::OperatorDAG>  dag;
  std::optional<soma::ClockSet>     clocks;
  std::vector<CompRuntime>          comps;
  std::string                       last_error;
  std::int32_t                      communicator = 0;
  bool                              communicator_set = false;
  std::optional<soma::Rational>     end_time;
};

namespace {

/// Records `msg` on the context (when there is one) and always on the calling
/// thread. Must not throw: it runs on the failure path of noexcept wrappers,
/// and a bad_alloc here would defeat the whole point of the boundary.
void store_error(soma_ctx ctx, const std::string& msg) noexcept {
  try {
    thread_error() = msg;
  } catch (...) {  // NOLINT(bugprone-empty-catch): nothing better is possible
  }
  if (ctx != nullptr) {
    try {
      ctx->last_error = msg;
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }
}

/// Store `msg` and return `code`. Every failing return in this file goes
/// through here, so no error can reach the caller without a message.
int fail(soma_ctx ctx, int code, const std::string& msg) noexcept {
  store_error(ctx, msg);
  return code;
}

[[nodiscard]] bool is_live(soma_ctx ctx) {
  if (ctx == nullptr) return false;
  HandleTable&                t = table();
  std::lock_guard<std::mutex> lock(t.mu);
  return t.live.find(static_cast<const void*>(ctx)) != t.live.end();
}

/// The single shape every context-taking entry point has: validate the
/// handle without dereferencing it, then run the body under a catch-all.
template <class F>
int entry(soma_ctx ctx, F&& body) noexcept {
  bool live = false;
  try {
    live = is_live(ctx);
  } catch (const std::exception& e) {
    return fail(nullptr, SOMA_ERR_RUNTIME,
                std::string("soma: handle table lock failed: ") + e.what());
  } catch (...) {
    return fail(nullptr, SOMA_ERR_RUNTIME, "soma: handle table lock failed");
  }
  if (!live) {
    return fail(nullptr, SOMA_ERR_INVALID,
                "soma: invalid, null, or already-destroyed soma_ctx handle");
  }
  try {
    return body(*ctx);
  } catch (const std::exception& e) {
    return fail(ctx, SOMA_ERR_RUNTIME, e.what());
  } catch (...) {
    return fail(ctx, SOMA_ERR_RUNTIME,
                "soma: unknown non-std exception at the C ABI boundary");
  }
}

[[nodiscard]] std::string describe(const soma::Error& e) {
  return e.code + ": " + e.message;
}

/// Resolve a component name, or report why not. On success `*out` is the id.
int resolve_component(soma_ctx_s& c, const char* name, const char* fn,
                      soma::ComponentId* out) {
  if (name == nullptr) {
    return fail(&c, SOMA_ERR_INVALID,
                std::string(fn) + ": component name is null");
  }
  if (c.state != State::Initialized) {
    return fail(&c, SOMA_ERR_STATE,
                std::string(fn) +
                    ": the context is not initialized; call soma_initialize() "
                    "first");
  }
  const auto id = c.graph.find_component(name);
  if (!id.has_value()) {
    return fail(&c, SOMA_ERR_NOTFOUND,
                std::string(fn) + ": no component named '" + name +
                    "' in the loaded graph");
  }
  *out = *id;
  return SOMA_OK;
}

/// Copy `src` into a caller-provided buffer, truncating and NUL-terminating.
int copy_out(const std::string& src, char* buffer, std::uint64_t buffer_size) {
  const std::uint64_t room = buffer_size - 1;  // caller checked size >= 1
  const std::uint64_t n =
      std::min<std::uint64_t>(static_cast<std::uint64_t>(src.size()), room);
  if (n > 0) std::memcpy(buffer, src.data(), static_cast<std::size_t>(n));
  buffer[n] = '\0';
  return SOMA_OK;
}

/// How many ranks will independently run this context's soma_begin_step() /
/// soma_end_step() sequence.
///
/// 1 means "this decision is nobody else's business": a serial process, a
/// build without MPI, an MPI that was never initialised, or a caller who
/// explicitly scoped the context with MPI_COMM_SELF. Anything greater means
/// several ranks are each about to advance their own copy of the clocks with
/// no reduction between them -- which is only safe when nothing makes those
/// clocks depend on rank-local data. See the gate in soma_initialize().
///
/// WHICH COMMUNICATOR. The one the caller set, if it set one: that is its
/// explicit statement of this context's scope, and MPI_COMM_SELF is a
/// legitimate way to say "just me" inside a larger job. Otherwise
/// MPI_COMM_WORLD, because a caller who never mentioned a communicator at all
/// is exactly the case that has to be caught -- the hazard is live whether or
/// not soma_set_communicator() was ever called.
///
/// LOCAL, NOT COLLECTIVE. MPI_Comm_size and the two state queries are all
/// local, so this cannot deadlock a rank that reaches soma_initialize() when
/// another does not. Every failure answers 1, which is the answer that
/// refuses nothing.
///
/// Residual risk, the same one src/transport/mpi_transport.cpp accepts and for
/// the same reason: a caller that passes a handle which is neither
/// MPI_COMM_NULL nor a live communicator is in undefined behaviour the moment
/// anything looks at it, and no MPI call can tell the difference.
[[nodiscard]] int coupled_rank_count([[maybe_unused]] const soma_ctx_s& c)
    noexcept {
#ifdef SOMA_HAVE_MPI
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return 1;
  }
  int finalized = 0;
  if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized != 0) return 1;

  const MPI_Comm comm =
      c.communicator_set
          ? MPI_Comm_f2c(static_cast<MPI_Fint>(c.communicator))
          : MPI_COMM_WORLD;
  if (comm == MPI_COMM_NULL) return 1;

  int size = 1;
  if (MPI_Comm_size(comm, &size) != MPI_SUCCESS) return 1;
  return size < 1 ? 1 : size;
#else
  return 1;
#endif
}

/// Gather a registered field's elements into a contiguous, canonically
/// ordered (element, then component) buffer, using the same strided offsets
/// the rest of the ABI already trusts (FieldRef::offset()). This is what lets
/// soma::num::norm_l2 / norm_l2_diff -- which assume a flat contiguous
/// buffer, like the Runner's own DAG-arc storage -- run correctly regardless
/// of whether the caller registered AoS or SoA storage.
[[nodiscard]] std::vector<double> gather_field(const soma::FieldRef& ref) {
  const soma::Layout& layout = ref.layout();
  std::vector<double> out;
  out.reserve(layout.count * static_cast<std::size_t>(layout.components));
  for (std::size_t i = 0; i < layout.count; ++i) {
    for (int comp = 0; comp < layout.components; ++comp) {
      out.push_back(ref.data()[ref.offset(i, comp)]);
    }
  }
  return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

extern "C" const char* soma_version(void) { return SOMA_VERSION_STRING; }

extern "C" int soma_abi_version(void) { return SOMA_ABI_VERSION; }

// ---------------------------------------------------------------------------
// Lifetime
// ---------------------------------------------------------------------------

extern "C" int soma_create(soma_ctx* out) {
  if (out == nullptr) {
    return fail(nullptr, SOMA_ERR_INVALID,
                "soma_create: the out parameter is null");
  }
  *out = nullptr;
  soma_ctx_s* p = nullptr;
  try {
    p = new soma_ctx_s();
    HandleTable& t = table();
    {
      std::lock_guard<std::mutex> lock(t.mu);
      t.live.insert(static_cast<const void*>(p));
    }
  } catch (const std::exception& e) {
    delete p;
    return fail(nullptr, SOMA_ERR_RUNTIME,
                std::string("soma_create: ") + e.what());
  } catch (...) {
    delete p;
    return fail(nullptr, SOMA_ERR_RUNTIME,
                "soma_create: unknown non-std exception");
  }
  *out = p;
  return SOMA_OK;
}

extern "C" int soma_destroy(soma_ctx ctx) {
  if (ctx == nullptr) {
    return fail(nullptr, SOMA_ERR_INVALID, "soma_destroy: null handle");
  }
  try {
    HandleTable& t = table();
    {
      // Erasing under the lock is what makes a double destroy safe: exactly
      // one caller can win the erase, and only that caller deletes.
      std::lock_guard<std::mutex> lock(t.mu);
      if (t.live.erase(static_cast<const void*>(ctx)) == 0) {
        return fail(nullptr, SOMA_ERR_INVALID,
                    "soma_destroy: invalid or already-destroyed soma_ctx "
                    "handle");
      }
    }
    delete ctx;
    return SOMA_OK;
  } catch (const std::exception& e) {
    return fail(nullptr, SOMA_ERR_RUNTIME,
                std::string("soma_destroy: ") + e.what());
  } catch (...) {
    return fail(nullptr, SOMA_ERR_RUNTIME,
                "soma_destroy: unknown non-std exception");
  }
}

// ---------------------------------------------------------------------------
// Error retrieval
// ---------------------------------------------------------------------------

extern "C" int soma_last_error(soma_ctx ctx, char* buffer,
                               std::uint64_t buffer_size) {
  if (buffer == nullptr || buffer_size == 0) return SOMA_ERR_INVALID;
  buffer[0] = '\0';
  try {
    if (ctx == nullptr) return copy_out(thread_error(), buffer, buffer_size);
    if (!is_live(ctx)) return SOMA_ERR_INVALID;
    return copy_out(ctx->last_error, buffer, buffer_size);
  } catch (...) {
    // Deliberately does not call fail(): overwriting the message the caller
    // is trying to read would be perverse.
    return SOMA_ERR_RUNTIME;
  }
}

extern "C" int soma_last_error_length(soma_ctx ctx, std::uint64_t* out_length) {
  if (out_length == nullptr) return SOMA_ERR_INVALID;
  *out_length = 0;
  try {
    if (ctx == nullptr) {
      *out_length = static_cast<std::uint64_t>(thread_error().size());
      return SOMA_OK;
    }
    if (!is_live(ctx)) return SOMA_ERR_INVALID;
    *out_length = static_cast<std::uint64_t>(ctx->last_error.size());
    return SOMA_OK;
  } catch (...) {
    return SOMA_ERR_RUNTIME;
  }
}

// ---------------------------------------------------------------------------
// The narrow waist
// ---------------------------------------------------------------------------

extern "C" int soma_load_graph(soma_ctx ctx, const void* descriptor,
                               std::uint64_t bytes) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state != State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_load_graph: this context already has a graph; a "
                  "context accepts exactly one");
    }
    if (descriptor == nullptr) {
      return fail(&c, SOMA_ERR_INVALID, "soma_load_graph: descriptor is null");
    }
    if (bytes == 0) {
      return fail(&c, SOMA_ERR_INVALID, "soma_load_graph: bytes is zero");
    }
    if (!fits_size(bytes)) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_load_graph: descriptor length " +
                      std::to_string(bytes) +
                      " exceeds this platform's address space");
    }
    const auto* first = static_cast<const std::byte*>(descriptor);
    auto        r     = soma::deserialize(
        std::span<const std::byte>(first, static_cast<std::size_t>(bytes)));
    if (!r.ok()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_load_graph: " + describe(r.error()));
    }
    c.graph = std::move(r.value());
    c.state = State::Loaded;
    return SOMA_OK;
  });
}

extern "C" int soma_load_graph_yaml(soma_ctx ctx, const char* path) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state != State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_load_graph_yaml: this context already has a graph; a "
                  "context accepts exactly one");
    }
    if (path == nullptr) {
      return fail(&c, SOMA_ERR_INVALID, "soma_load_graph_yaml: path is null");
    }
    auto r = soma::load_yaml_file(std::string(path));
    if (!r.ok()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_load_graph_yaml: " + describe(r.error()));
    }
    c.graph = std::move(r.value());
    c.state = State::Loaded;
    return SOMA_OK;
  });
}

extern "C" int soma_save_graph(soma_ctx ctx, void* buffer,
                               std::uint64_t buffer_size,
                               std::uint64_t* out_bytes) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (out_bytes == nullptr) {
      return fail(&c, SOMA_ERR_INVALID, "soma_save_graph: out_bytes is null");
    }
    *out_bytes = 0;
    if (c.state == State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_save_graph: no graph has been loaded");
    }
    const std::vector<std::byte> blob = soma::serialize(c.graph);
    *out_bytes = static_cast<std::uint64_t>(blob.size());
    if (buffer == nullptr) return SOMA_OK;  // size query
    if (buffer_size < static_cast<std::uint64_t>(blob.size())) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_save_graph: buffer of " + std::to_string(buffer_size) +
                      " bytes is too small for a descriptor of " +
                      std::to_string(blob.size()));
    }
    if (!blob.empty()) std::memcpy(buffer, blob.data(), blob.size());
    return SOMA_OK;
  });
}

// ---------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------

extern "C" int soma_field_handle(soma_ctx ctx, const char* component,
                                 const char* field, std::int32_t* out_handle) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (out_handle == nullptr) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_field_handle: out_handle is null");
    }
    *out_handle = -1;
    if (component == nullptr || field == nullptr) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_field_handle: component or field name is null");
    }
    if (c.state == State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_field_handle: no graph has been loaded");
    }
    const auto comp = c.graph.find_component(component);
    if (!comp.has_value()) {
      return fail(&c, SOMA_ERR_NOTFOUND,
                  std::string("soma_field_handle: no component named '") +
                      component + "'");
    }
    const auto fid = c.graph.find_field(*comp, field);
    if (!fid.has_value()) {
      return fail(&c, SOMA_ERR_NOTFOUND,
                  std::string("soma_field_handle: component '") + component +
                      "' has no field named '" + field + "'");
    }
    // A graph with more than 2^31-1 fields cannot be expressed through this
    // ABI. Refuse rather than wrap into a negative handle.
    if (fid->value() >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
      return fail(&c, SOMA_ERR_UNSUPPORTED,
                  "soma_field_handle: field index exceeds the int32 handle "
                  "range");
    }
    *out_handle = static_cast<std::int32_t>(fid->value());
    return SOMA_OK;
  });
}

extern "C" int soma_register_field(soma_ctx ctx, std::int32_t handle,
                                   double* data, std::uint64_t count,
                                   std::uint64_t stride,
                                   std::uint64_t component_stride,
                                   std::int32_t components,
                                   std::int32_t memspace) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state == State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_register_field: no graph has been loaded");
    }
    if (c.state == State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_register_field: the context is already initialized; "
                  "register every field before soma_initialize()");
    }
    if (data == nullptr) {
      return fail(&c, SOMA_ERR_INVALID, "soma_register_field: data is null");
    }
    if (handle < 0 || static_cast<std::uint64_t>(handle) >=
                          static_cast<std::uint64_t>(c.graph.field_count())) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_register_field: handle " + std::to_string(handle) +
                      " is outside [0, " +
                      std::to_string(c.graph.field_count()) + ")");
    }
    if (memspace != SOMA_MEMSPACE_HOST && memspace != SOMA_MEMSPACE_DEVICE) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_register_field: memspace " + std::to_string(memspace) +
                      " is neither SOMA_MEMSPACE_HOST nor "
                      "SOMA_MEMSPACE_DEVICE");
    }
    if (components < 1) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_register_field: components must be >= 1, got " +
                      std::to_string(components));
    }
    const soma::FieldId    fid{static_cast<std::uint32_t>(handle)};
    const soma::FieldSpec& spec = c.graph.field(fid);
    if (components != spec.components) {
      // Silently reinterpreting a 3-vector as a scalar is wrong physics with
      // no symptom, so this is refused rather than tolerated.
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_register_field: field '" + spec.name + "' is declared "
                  "with " + std::to_string(spec.components) +
                      " component(s) but was registered with " +
                      std::to_string(components));
    }
    if (!fits_size(count) || !fits_size(stride) ||
        !fits_size(component_stride)) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_register_field: count/stride/component_stride exceed "
                  "this platform's address space");
    }

    const soma::Layout layout{static_cast<std::size_t>(stride),
                              static_cast<std::size_t>(component_stride),
                              static_cast<std::size_t>(count), components};
    const soma::MemorySpace space = (memspace == SOMA_MEMSPACE_DEVICE)
                                        ? soma::MemorySpace::Device
                                        : soma::MemorySpace::Host;
    const auto r = c.fields.bind(fid, soma::FieldRef{data, layout, space});
    if (!r.ok()) {
      // A rebind is a sequencing mistake (STATE); everything else bind()
      // rejects is a malformed argument (INVALID).
      const int code =
          (r.error().code == "E-FIELD-REBIND") ? SOMA_ERR_STATE
                                               : SOMA_ERR_INVALID;
      return fail(&c, code, "soma_register_field: " + describe(r.error()));
    }
    return SOMA_OK;
  });
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

extern "C" int soma_set_communicator(soma_ctx ctx, std::int32_t fortran_comm) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state == State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_set_communicator: the context is already initialized");
    }
#ifdef SOMA_HAVE_MPI
    c.communicator     = fortran_comm;
    c.communicator_set = true;
    return SOMA_OK;
#else
    (void)fortran_comm;
    return fail(&c, SOMA_ERR_UNSUPPORTED,
                "soma_set_communicator: this build of SOMA has no MPI "
                "transport; reconfigure with -DSOMA_ENABLE_MPI=ON");
#endif
  });
}

extern "C" int soma_set_end_time(soma_ctx ctx, std::int64_t numerator,
                                 std::int64_t denominator) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state == State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_set_end_time: the context is already initialized");
    }
    if (denominator == 0) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_set_end_time: denominator is zero");
    }
    // Rational's constructor throws on a non-representable normalisation
    // (e.g. INT64_MIN); that becomes SOMA_ERR_INVALID rather than RUNTIME
    // because it is a bad argument, not an internal failure.
    try {
      c.end_time = soma::Rational{numerator, denominator};
    } catch (const std::exception& e) {
      return fail(&c, SOMA_ERR_INVALID,
                  std::string("soma_set_end_time: ") + e.what());
    }
    return SOMA_OK;
  });
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

extern "C" int soma_initialize(soma_ctx ctx) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state == State::Empty) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_initialize: no graph has been loaded");
    }
    if (c.state == State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_initialize: the context is already initialized");
    }

    // 1. Structural validation. Every diagnostic, not just the first.
    const soma::Diagnostics diags = soma::validate(c.graph);
    if (diags.has_errors()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_initialize: the graph does not validate:\n" +
                      diags.to_string());
    }

    // 2. Every field a channel reads or writes must have storage. The
    //    exchange dereferences these pointers; an unbound one is a null
    //    dereference later instead of a diagnosable error now.
    std::vector<soma::FieldId> required;
    required.reserve(2 * c.graph.channel_count());
    for (std::uint32_t i = 0; i < c.graph.channel_count(); ++i) {
      const soma::Channel& ch = c.graph.channel_at(soma::ChannelId{i});
      required.push_back(ch.source);
      required.push_back(ch.target);
    }
    const std::vector<soma::FieldId> missing = c.fields.unbound(required);
    if (!missing.empty()) {
      std::string names;
      for (const soma::FieldId f : missing) {
        const std::string n = c.graph.field(f).name;
        if (names.find("'" + n + "'") != std::string::npos) continue;
        if (!names.empty()) names += ", ";
        names += "'" + n + "'";
      }
      return fail(&c, SOMA_ERR_STATE,
                  "soma_initialize: field(s) used by a channel have no "
                  "registered storage: " +
                      names);
    }

    // 3. Lowering. Catches everything validation cannot: an unclaimed
    //    channel, an SSA violation, an unsupported operator combination.
    soma::LoweringResult lowered = soma::lower(c.graph);
    if (!lowered.ok()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_initialize: the graph does not lower:\n" +
                      lowered.diags.to_string());
    }

    // 4. Clocks. ClockSet's constructor throws on a non-positive interval;
    //    validate() already rejected that with E-INTERVAL, and entry()'s
    //    catch-all covers the case where it somehow did not.
    soma::ClockSet clocks{c.graph};
    const auto     period = clocks.checked_common_period();
    if (!period.ok()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_initialize: " + describe(period.error()));
    }

    // 5. Per-component sweep counts, derived from the iteration regions.
    //    Also records each component's *governing* region -- the one that
    //    set its max_iterations -- so the loop just below can try to resolve
    //    a convergence early-exit for it. A component named by more than one
    //    region ties to whichever is encountered first among those with the
    //    largest max_iterations; that ambiguity pre-dates this change (the
    //    old code already only kept a single std::max()'d count) and is not
    //    newly introduced here.
    std::vector<CompRuntime> comps(c.graph.component_count());
    std::vector<std::optional<soma::IterationRegionIdx>> governing(
        c.graph.component_count());
    for (std::uint32_t i = 0; i < c.graph.iteration_region_count(); ++i) {
      const auto region_id = soma::IterationRegionIdx{i};
      const auto& region = c.graph.iteration_region(region_id);
      const std::int32_t sweeps =
          std::max(1, static_cast<int>(region.max_iterations));
      for (const soma::ChannelId cid : region.channels) {
        if (!cid.valid() ||
            cid.value() >= static_cast<std::uint32_t>(c.graph.channel_count())) {
          return fail(&c, SOMA_ERR_INVALID,
                      "soma_initialize: iteration region '" + region.name +
                          "' references channel " +
                          std::to_string(cid.value()) +
                          ", which does not exist");
        }
        const soma::Channel& ch = c.graph.channel_at(cid);
        for (const soma::ComponentId owner :
             {c.graph.owner_of(ch.source), c.graph.owner_of(ch.target)}) {
          CompRuntime& rt = comps.at(owner.value());
          if (sweeps > rt.max_iterations) {
            rt.max_iterations = sweeps;
            governing.at(owner.value()) = region_id;
          }
        }
      }
    }

    // 6. THE ITERATION-REGION GATE. Spec §5.3's conditions on a region's
    //    stopping rule, enforced here because nothing else on this path
    //    enforces them: soma::validate() does not look at iteration regions at
    //    all, soma::lower() only rejects a region that covers no node
    //    (E-IR-REGION-EMPTY), and verify_iteration_region() /
    //    resolve_regions() -- which do -- are reachable only through a Runner,
    //    which needs a ComponentSet and a MeshBackend this ABI never builds.
    //
    //    Both predicates are the Runner's own, shared from
    //    src/ir/checkpoint.cpp rather than reimplemented, so the two entry
    //    points cannot drift and a C or Fortran caller reads the same
    //    diagnostic a C++ caller reads:
    //
    //      * E-ITER-NOCONVERGE -- an empty ConvergenceCriterion. all_satisfied()
    //        on one is trivially true, so the region's stopping rule is "always
    //        stop": the window ends after one sweep and reports that it
    //        converged. There is no safe way to accept this. Testing the
    //        criterion truncates every window to one sweep; ignoring it (what
    //        this ABI did until now) silently declines to honour a criterion
    //        the author never wrote in the first place, and hides a graph the
    //        Runner would refuse outright.
    //      * E-ITER-MAXITER -- max_iterations < 1. Measured through this ABI
    //        before it was checked here: `max_iterations: 0` gave rc 0, an
    //        empty error message, soma_requires_checkpoint() == 0, then ONE
    //        sweep and a clock advance -- because step 5 below clamps with
    //        std::max(1, ...). That is verbatim the E-ITER-NOCONVERGE failure
    //        mode reached by a different route, so it is refused by the same
    //        gate rather than clamped into silence.
    //      * E-ITER-MEASURE -- a measure that names no channel of its region or
    //        more than one, or a channel of the region that no measure names.
    //
    //    THIS TIGHTENS soma_initialize(): a graph in any of those states used
    //    to reach State::Initialized and now does not. It is documented as such
    //    in soma_c.h. The trade is a loud SOMA_ERR_INVALID in place of a run
    //    whose convergence behaviour is not the one the graph asks for.
    //
    //    ONE MESSAGE SHAPE for all three codes: every failure here is collected
    //    into one Diagnostics and rendered with to_string(), so a caller
    //    parsing soma_last_error() sees the same "error <CODE>: <text>" layout
    //    whichever check fired, and a region with two problems reports both.
    //
    //    The candidate names come from the region's declared channels; see
    //    match_convergence_measures() in include/soma/ir/lowering.hpp for why
    //    that list and the DAG's residual list are the same list. Channel ids
    //    were bounds-checked in step 5.
    std::vector<std::vector<std::size_t>> region_match(
        c.graph.iteration_region_count());
    soma::Diagnostics region_diags;
    for (std::uint32_t i = 0; i < c.graph.iteration_region_count(); ++i) {
      const auto& region = c.graph.iteration_region(soma::IterationRegionIdx{i});

      soma::Diagnostics one = soma::verify_iteration_declaration(
          region.name, region.max_iterations, region.convergence);

      // Only worth matching when the declaration itself stands up: an empty
      // criterion has no measures to match, and reporting "your zero measures
      // cover none of your channels" on top of E-ITER-NOCONVERGE would bury
      // the diagnostic that matters under one that follows from it.
      if (!one.has_errors()) {
        std::vector<soma::MeasureTarget> targets;
        targets.reserve(region.channels.size());
        for (const soma::ChannelId cid : region.channels) {
          const soma::Channel& ch = c.graph.channel_at(cid);
          targets.push_back(soma::MeasureTarget{ch.name,
                                                c.graph.field(ch.source).name,
                                                c.graph.field(ch.target).name});
        }

        soma::Result<std::vector<std::size_t>> matched =
            soma::match_convergence_measures(region.name, region.convergence,
                                             targets);
        if (matched.ok()) {
          region_match[i] = std::move(matched.value());
        } else {
          one.add(soma::Severity::Error, matched.error().code,
                  matched.error().message,
                  "iteration region '" + region.name + "'");
        }
      }
      region_diags.merge(std::move(one));
    }
    if (region_diags.has_errors()) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_initialize: the graph's iteration regions do not "
                  "verify:\n" +
                      region_diags.to_string());
    }

    // 7. Convergence early-exit resolution; see soma_end_step(). For every
    //    component whose governing region declares more than one sweep, bind
    //    that region's criterion to the registered storage it measures. Step 6
    //    has already proven the criterion is non-empty and that every measure
    //    resolves to exactly one channel, so there is no "unresolved" fallback
    //    left: `converges` is now true for every such component, which is the
    //    point -- it used to be false, silently, for precisely the graphs the
    //    Runner refuses.
    for (std::size_t ci = 0; ci < comps.size(); ++ci) {
      CompRuntime& rt = comps[ci];
      if (rt.max_iterations <= 1 || !governing[ci].has_value()) continue;
      const auto& region = c.graph.iteration_region(*governing[ci]);
      const std::vector<std::size_t>& matched =
          region_match.at(governing[ci]->value());

      const auto& measures = region.convergence.measures();
      std::vector<std::size_t>   measure_slot(measures.size());
      std::vector<soma::FieldId> snapshot_fields;
      for (std::size_t mi = 0; mi < measures.size(); ++mi) {
        // The measured quantity is the channel's TARGET field: that is what
        // build_iteration_regions() (src/ir/regions.cpp) attaches each region's
        // Residual node to (it reads ctx.field_arc for ch.target), i.e. the
        // same value the Runner's residual measures.
        const soma::FieldId hit =
            c.graph.channel_at(region.channels.at(matched.at(mi))).target;
        std::size_t slot = snapshot_fields.size();
        for (std::size_t k = 0; k < snapshot_fields.size(); ++k) {
          if (snapshot_fields[k] == hit) {
            slot = k;
            break;
          }
        }
        if (slot == snapshot_fields.size()) snapshot_fields.push_back(hit);
        measure_slot[mi] = slot;
      }

      rt.converges  = true;
      rt.criterion  = &region.convergence;
      rt.snapshot_fields = std::move(snapshot_fields);
      rt.snapshot_prev.assign(rt.snapshot_fields.size(), std::vector<double>{});
      rt.measure_slot = std::move(measure_slot);
      rt.measure_first_residual.assign(measures.size(), 0.0);
    }

    // 8. THE PARALLEL-SAFETY GATE. Task 5.4's requirement, keyed on the
    //    condition that actually makes a run wrong rather than on whether
    //    soma_set_communicator() happened to be called.
    //
    //    WHAT IS WRONG. soma_end_step() decides convergence from
    //    soma::num::norm_l2_diff over THIS RANK'S registered buffer and then
    //    calls clock(id).advance() on that purely local verdict. There is no
    //    reduction anywhere in this file. So on n > 1 ranks each rank ends the
    //    window whenever its own residual happens to satisfy the criterion,
    //    and the component clocks diverge. Measured, on two ranks, with
    //    test/data/pair_iterated.yaml and soma_set_communicator NEVER CALLED:
    //    after the same two sweeps rank 0 reported the window [1, 2) and
    //    rank 1 reported [0, 1), and soma_initialize() had returned SOMA_OK on
    //    both. In a real coupling that is a hang, or an exchange at the wrong
    //    time. test/parallel/capi_parallel_test.c pins it.
    //
    //    WHY THE PREDICATE IS `converges`, NOT `communicator_set`. Two
    //    corrections to the first version of this gate, both demonstrated:
    //
    //      * The hazard does not need a communicator. The run above set none.
    //        Gating on communicator_set misses the entire failure and, worse,
    //        the advice that follows from it ("just do not call
    //        soma_set_communicator") steers callers straight into it.
    //      * A communicator is not itself a hazard. soma_set_communicator(
    //        MPI_Comm_c2f(MPI_COMM_SELF)) followed by soma_initialize() is a
    //        correct serial run; refusing it turned working code into a hard
    //        failure. Size 1 is now accepted unconditionally, from any source.
    //
    //    Without an early exit every rank sweeps to max_iterations and
    //    advances in lockstep, which is deterministic and identical
    //    everywhere -- so `rt.converges` is exactly the predicate, and step 7
    //    above is what makes it trustworthy.
    //
    //    n > 1 WITH NO EARLY EXIT IS NOT REFUSED. Nothing is wrong with that
    //    run: the clocks cannot diverge. It leaves an advisory in the error
    //    slot instead, which is the only diagnostic channel this ABI has; see
    //    soma_last_error() in soma_c.h, which now says a successful
    //    soma_initialize() may leave one.
    const int ranks = coupled_rank_count(c);
    if (ranks > 1) {
      std::string iterated;
      for (std::size_t ci = 0; ci < comps.size(); ++ci) {
        if (!comps[ci].converges) continue;
        if (!iterated.empty()) iterated += ", ";
        iterated += "'" +
                    c.graph.component(soma::ComponentId{
                                          static_cast<std::uint32_t>(ci)})
                        .name +
                    "'";
      }
      if (!iterated.empty()) {
        return fail(
            &c, SOMA_ERR_UNSUPPORTED,
            "soma_initialize: this context will run on " +
                std::to_string(ranks) +
                " ranks, and component(s) " + iterated +
                " sit in an iteration region whose convergence SOMA would test "
                "for you in soma_end_step(). That test is rank-local -- it "
                "reduces nothing across ranks -- so each rank would end the "
                "window on its own residual and the component clocks would "
                "diverge, which is a hang or a mistimed exchange rather than a "
                "wrong number you could see. This ABI has no collective to fix "
                "that with. Either run this context on one rank (MPI_COMM_SELF "
                "is accepted), or drop the region's convergence measure and "
                "sweep to max_iterations, which every rank does identically, "
                "or use the C++ Runner, whose iteration is not driven a step "
                "at a time from outside");
      }
      // Advisory only: no early exit, so nothing here can diverge. Recorded
      // rather than returned, because the run is correct.
      store_error(&c,
                  "soma_initialize: note (not an error): this context will run "
                  "on " +
                      std::to_string(ranks) +
                      " ranks. soma_begin_step()/soma_end_step() are entirely "
                      "rank-local and reduce nothing; no iteration region here "
                      "asks SOMA to test convergence, so the clocks stay in "
                      "step, but any agreement your components need across "
                      "ranks is yours to arrange");
    }

    // 9. WHAT IS DELIBERATELY *NOT* CHECKED HERE, AND WHY -- the remap.
    //
    //    A graph may name a conservative remap that soma::build_weights()
    //    (src/remap/intx_weights.cpp) would refuse on the Runner path. This
    //    repository's own test/data/pair.yaml is such a graph: `fluid_wall`
    //    (384 wall faces of a generated 8x8x8 box) -> `solid_wall` (150 faces
    //    of a 5x5x5 box), Conservative|Monotone. soma_initialize() accepts it,
    //    and will keep accepting it. Two reasons, both measured rather than
    //    assumed:
    //
    //    (a) THE REFUSAL IS ONLY PARTLY VISIBLE FROM THE GRAPH. What actually
    //        rejects pair.yaml is the SECOND test in resolve() (the region-id
    //        range check at intx_weights.cpp:237 precedes it): at :246 the
    //        regions name entity set "wall" rather than the whole mesh, and
    //        MeshBackend does not say whether a sub-set region's buffer is
    //        indexed by set position or by mesh-local index.
    //
    //        resolve() has roughly fifteen refusal paths across seven codes.
    //        FOUR are decidable from the graph alone -- :237 (region id out of
    //        range), :246 (sub-set region), :290 (mesh id out of range) and
    //        :312 (declared dimension outside 1..3) -- and the rest are not:
    //        a backend that is not MOAB (E-REMAP-BACKEND), a partitioned read
    //        (E-REMAP-PARALLEL), a mesh name the backend never loaded, a cell
    //        count that disagrees with the MeshView, an int32 overflow, and
    //        MOAB's own errors. Checking the four decidable ones and staying
    //        silent about the others would hand callers a guarantee that reads
    //        as "initialize accepted it, so the remap is buildable" and is not.
    //        (E-REMAP-UNSUPPORTED, for an unknown method or a non-FV
    //        discretization, is not in resolve() at all -- it is raised later,
    //        in the plan dispatch around :1386-1482.)
    //
    //        Note also that NON-COINCIDENCE ITSELF IS NOT A REFUSAL: differing
    //        meshes are the ordinary case a conservative remap exists for, and
    //        test/unit/remap_test.cpp's `non_matching()` fixture builds weights
    //        across exactly that. In pair.yaml the two walls in fact coincide
    //        geometrically -- both meshes are generated boxes -- and differ
    //        only in discretization, 384 faces against 150.
    //
    //    (b) REFUSING WOULD BREAK A GRAPH THIS ABI SERVES CORRECTLY. This ABI
    //        performs no remap, exchange or transport on any path, by design
    //        (see the top of soma_c.h): the caller is the component and does
    //        its own transfer by whatever means it and its peers agreed on.
    //        SOMA's inability to BUILD weights for a graph says nothing about
    //        the caller's ability to couple it. pair.yaml is driven correctly
    //        through soma_begin_step()/soma_end_step() today, by
    //        test/unit/capi_test.c and test/fortran/capi_test.f90 among others;
    //        rejecting it would turn a working path into a refused one, which
    //        is not the "silent wrong answer -> loud error" trade the check in
    //        step 1 above is.
    //
    //    So the honest statement is documentation, not a refusal, and it is
    //    made where a caller will read it: soma_c.h's "WHAT THIS ABI DOES NOT
    //    DO" now says in as many words that reaching State::Initialized is not
    //    a claim that SOMA could perform the graph's remap.

    // Commit only once everything above has succeeded: a failed
    // soma_initialize() leaves the context in LOADED, not half-initialized.
    c.dag    = std::move(lowered.dag);
    c.clocks = std::move(clocks);
    c.comps  = std::move(comps);
    c.state  = State::Initialized;
    return SOMA_OK;
  });
}

extern "C" int soma_begin_step(soma_ctx ctx, const char* component,
                               double* t_begin, double* t_end) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (t_begin == nullptr || t_end == nullptr) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_begin_step: t_begin or t_end is null");
    }
    soma::ComponentId id;
    const int rc = resolve_component(c, component, "soma_begin_step", &id);
    if (rc != SOMA_OK) return rc;

    CompRuntime& rt = c.comps.at(id.value());
    if (rt.in_step) {
      return fail(&c, SOMA_ERR_STATE,
                  std::string("soma_begin_step: a step is already open on "
                              "component '") +
                      component + "'; call soma_end_step() first");
    }

    // A fresh window opens exactly when iteration == 0 (the same instant
    // soma_requires_checkpoint() reports 1). Capture "y0" -- the value the
    // measured fields hold before this window's first sweep runs -- here,
    // before the caller does any work for sweep 0. Capturing it any later
    // (e.g. in soma_end_step()) would already be post-sweep-0 data, which is
    // exactly the checkpoint-timing hazard iteration.cpp's header comment
    // warns about for the C++ Runner's own save_state(): a snapshot taken
    // after the first advance is not the window's starting point.
    if (rt.converges && rt.iteration == 0) {
      bool ok = true;
      for (std::size_t k = 0; k < rt.snapshot_fields.size() && ok; ++k) {
        const auto ref = c.fields.get(rt.snapshot_fields[k]);
        if (!ref.ok()) { ok = false; break; }
        rt.snapshot_prev[k] = gather_field(ref.value());
      }
      rt.snapshots_valid     = ok;
      rt.have_first_residual = false;
    }

    const soma::Clock& clock = c.clocks->clock(id);
    *t_begin                 = clock.now().to_double();
    *t_end                   = clock.next().to_double();
    rt.in_step               = true;
    return SOMA_OK;
  });
}

extern "C" int soma_end_step(soma_ctx ctx, const char* component) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    soma::ComponentId id;
    const int rc = resolve_component(c, component, "soma_end_step", &id);
    if (rc != SOMA_OK) return rc;

    CompRuntime& rt = c.comps.at(id.value());
    if (!rt.in_step) {
      return fail(&c, SOMA_ERR_STATE,
                  std::string("soma_end_step: no step is open on component '") +
                      component + "'");
    }
    rt.in_step = false;
    ++rt.iteration;

    // Convergence early exit. Only attempted strictly before the declared
    // cap: at iteration == max_iterations the window ends unconditionally
    // below, exactly as before, so there is nothing to gain by testing here
    // (and the "not converged, but continue" branch's snapshot-advance would
    // have nothing left to feed).
    bool converged_early = false;
    if (rt.converges && rt.snapshots_valid &&
        rt.iteration < rt.max_iterations) {
      std::vector<std::vector<double>> current(rt.snapshot_fields.size());
      bool gathered = true;
      for (std::size_t k = 0; k < rt.snapshot_fields.size() && gathered;
           ++k) {
        const auto ref = c.fields.get(rt.snapshot_fields[k]);
        if (!ref.ok()) { gathered = false; break; }
        current[k] = gather_field(ref.value());
      }
      if (gathered) {
        const auto&         measures = rt.criterion->measures();
        std::vector<double> residual_norms(measures.size());
        std::vector<double> reference_norms(measures.size());
        for (std::size_t mi = 0; mi < measures.size(); ++mi) {
          const std::size_t slot = rt.measure_slot[mi];
          const double      residual = soma::num::norm_l2_diff(
              current[slot].data(), rt.snapshot_prev[slot].data(),
              current[slot].size());
          residual_norms[mi] = residual;
          if (measures[mi].kind == soma::MeasureKind::ResidualRelative) {
            // Reference is the *first* sweep's residual of this window, not
            // this sweep's -- matches iteration.cpp's R.block_first, captured
            // once at k == 0 and held fixed for the rest of the window.
            if (!rt.have_first_residual) {
              rt.measure_first_residual[mi] = residual;
            }
            reference_norms[mi] = rt.measure_first_residual[mi];
          } else {
            reference_norms[mi] =
                soma::num::norm_l2(current[slot].data(), current[slot].size());
          }
        }
        rt.have_first_residual = true;
        // Throws only on a size mismatch, which the parallel construction in
        // soma_initialize() above makes impossible; entry()'s catch-all is
        // the backstop if it somehow still does.
        converged_early =
            rt.criterion->all_satisfied(residual_norms, reference_norms);
        if (!converged_early) {
          // Not done: what was "current" becomes "previous" for the next
          // sweep's residual, mirroring iteration.cpp's x_old.swap(x_out).
          for (std::size_t k = 0; k < rt.snapshot_fields.size(); ++k) {
            rt.snapshot_prev[k] = std::move(current[k]);
          }
        }
      }
    }

    if (converged_early || rt.iteration >= rt.max_iterations) {
      // Last sweep of this window (or no iteration region at all): the
      // component's clock moves to the next tick. Reached either because the
      // convergence criterion is satisfied (above) or because max_iterations
      // -- the declared upper bound -- was reached without it.
      rt.iteration       = 0;
      rt.snapshots_valid = false;  // re-baseline in soma_begin_step() next
      c.clocks->clock(id).advance();
    }
    return SOMA_OK;
  });
}

namespace {

int rollback_query(soma_ctx ctx, const char* component,
                   std::int32_t* out_required, const char* fn, bool want_restore) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (out_required == nullptr) {
      return fail(&c, SOMA_ERR_INVALID,
                  std::string(fn) + ": out_required is null");
    }
    *out_required = 0;
    soma::ComponentId id;
    const int         rc = resolve_component(c, component, fn, &id);
    if (rc != SOMA_OK) return rc;

    const CompRuntime& rt = c.comps.at(id.value());
    const bool iterated = rt.max_iterations > 1;
    const bool yes = iterated && (want_restore ? (rt.iteration > 0)
                                               : (rt.iteration == 0));
    *out_required = yes ? 1 : 0;
    return SOMA_OK;
  });
}

}  // namespace

extern "C" int soma_requires_checkpoint(soma_ctx ctx, const char* component,
                                        std::int32_t* out_required) {
  return rollback_query(ctx, component, out_required,
                        "soma_requires_checkpoint", false);
}

extern "C" int soma_requires_restore(soma_ctx ctx, const char* component,
                                     std::int32_t* out_required) {
  return rollback_query(ctx, component, out_required, "soma_requires_restore",
                        true);
}

extern "C" int soma_is_coupling_ongoing(soma_ctx ctx,
                                        std::int32_t* out_ongoing) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (out_ongoing == nullptr) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_is_coupling_ongoing: out_ongoing is null");
    }
    *out_ongoing = 0;
    if (c.state != State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_is_coupling_ongoing: the context is not initialized");
    }
    const std::size_t n = c.clocks->size();
    if (n == 0) return SOMA_OK;  // nothing to advance; the run is over
    if (!c.end_time.has_value()) {
      *out_ongoing = 1;  // no horizon: the caller decides when to stop
      return SOMA_OK;
    }
    soma::Rational earliest =
        c.clocks->clock(soma::ComponentId{0}).now();
    for (std::uint32_t i = 1; i < static_cast<std::uint32_t>(n); ++i) {
      const soma::Rational t = c.clocks->clock(soma::ComponentId{i}).now();
      if (t < earliest) earliest = t;
    }
    *out_ongoing = (earliest < *c.end_time) ? 1 : 0;
    return SOMA_OK;
  });
}

extern "C" int soma_run_period(soma_ctx ctx) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state != State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_run_period: the context is not initialized");
    }
    return fail(&c, SOMA_ERR_UNSUPPORTED,
                "soma_run_period: SOMA cannot drive the loop through this "
                "ABI, which defines no way to register a component callback; "
                "drive the run with soma_begin_step()/soma_end_step()");
  });
}

extern "C" int soma_run(soma_ctx ctx, std::int64_t periods) {
  return entry(ctx, [&](soma_ctx_s& c) -> int {
    if (c.state != State::Initialized) {
      return fail(&c, SOMA_ERR_STATE,
                  "soma_run: the context is not initialized");
    }
    if (periods < 0) {
      return fail(&c, SOMA_ERR_INVALID,
                  "soma_run: periods must be >= 0, got " +
                      std::to_string(periods));
    }
    return fail(&c, SOMA_ERR_UNSUPPORTED,
                "soma_run: SOMA cannot drive the loop through this ABI, which "
                "defines no way to register a component callback; drive the "
                "run with soma_begin_step()/soma_end_step()");
  });
}
