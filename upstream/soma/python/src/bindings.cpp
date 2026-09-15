// SOMA's Python front-end (spec §6: in-process Python / ML surrogates are peer
// components through the pybind11 boundary; spec §3.2: they build the same
// graph object, and the same flat descriptor, as C++ and YAML).
//
// -----------------------------------------------------------------------------
// WHAT CROSSES WHICH BOUNDARY, AND WHY
// -----------------------------------------------------------------------------
// Two boundaries are bound here, deliberately, because they answer two
// different questions:
//
//   1. `soma.Context` is the C ABI of include/soma/soma_c.h, one Python method
//      per entry point. It is the narrow waist (§3.2) and the thing an external
//      driver would use. It is bound rather than bypassed so that the Python
//      front-end exercises exactly the surface Fortran does, including the
//      `SOMA_ERR_UNSUPPORTED` that `soma_run`/`soma_run_period` return.
//
//   2. `soma.CouplingGraph`, `soma.Session` and friends bind the C++ authoring
//      and runtime layers directly. The C ABI cannot express "here is a Python
//      object; call it back once per step", which is precisely what a surrogate
//      is (soma_run_period()'s own documentation says so). Peer components
//      therefore go through `Session`, which drives `soma::Runner`.
//
// Both produce and consume the same `soma_graph_v1` descriptor, so a graph
// authored in Python can be handed to either.
//
// -----------------------------------------------------------------------------
// ERRORS
// -----------------------------------------------------------------------------
// Nothing here ever returns an error code to Python. Every failure raises, and
// every exception derives from `soma.SomaError`. The five C ABI codes map onto
// five subclasses, each of which ALSO derives from the closest builtin, so
// idiomatic Python (`except ValueError`) keeps working:
//
//   SOMA_ERR_INVALID     -> soma.InvalidArgument  (SomaError, ValueError)
//   SOMA_ERR_NOTFOUND    -> soma.NotFound         (SomaError, LookupError)
//   SOMA_ERR_STATE       -> soma.StateError       (SomaError, RuntimeError)
//   SOMA_ERR_RUNTIME     -> soma.SomaRuntimeError (SomaError, RuntimeError)
//   SOMA_ERR_UNSUPPORTED -> soma.Unsupported      (SomaError, NotImplementedError)
//
// Every instance carries `.code` (the negative int, or None for errors raised
// above the ABI). `soma.ValidationError` additionally carries `.diagnostics`.
//
// No C++ exception escapes into the interpreter: pybind11 wraps every bound
// function in a catch-all, and the translator registered in PYBIND11_MODULE
// converts anything the built-in translators do not recognise into
// `soma.SomaRuntimeError` rather than letting it reach `std::terminate`.
//
// No Python exception propagates into C++ either: `ForwardingComponent` catches
// `py::error_already_set` at every callback (which clears the Python error
// indicator), converts it to a `soma::Result<void>` error for the C++ runtime
// to unwind through, stashes the original exception object, and re-raises that
// exact object once control is back at the binding boundary.
//
// -----------------------------------------------------------------------------
// THE GIL
// -----------------------------------------------------------------------------
// Released, with `py::gil_scoped_release`, around exactly the calls that do
// real work and touch no Python object:
//   soma.lower(), soma.schedule(), Session.initialize()'s validate/lower/
//   schedule and Runner setup, Session.run_period()/run(), Context.initialize().
// `Session.run()` is the interesting one: it calls back into Python through
// `ForwardingComponent`, which reacquires with `py::gil_scoped_acquire` before
// touching anything Python. Everything else bound here is O(1) bookkeeping or
// setup-time graph editing and holds the GIL, which is cheaper than releasing
// it.
//
// -----------------------------------------------------------------------------
// NUMPY BUFFER CONTRACT
// -----------------------------------------------------------------------------
// SOMA never owns field storage (soma_c.h, "FIELD STORAGE"), so a bound array
// must satisfy all of:
//   * it is a `numpy.ndarray` -- not a list, not a memoryview, not a scalar;
//   * dtype is exactly float64 (native byte order); no casting, ever, because a
//     cast is a copy and a copy is not the component's storage;
//   * it is C-contiguous -- `a[::2]`, `a.T` and any other strided view are
//     refused, because SOMA's Layout describes a flat run and would otherwise
//     silently read the wrong doubles;
//   * it is writeable -- SOMA writes into a Requires field;
//   * it is 1-D `(count,)` for a scalar field or 2-D `(count, components)` for a
//     vector field; C-contiguous 2-D is interleaved, i.e. stride = components
//     and component_stride = 1, matching soma_register_field()'s AoS case.
// The binding then holds a strong reference to the array for as long as the
// registry (or context) lives, which is what makes a temporary expression such
// as `reg.bind_numpy(f, np.zeros(4) + 1)` safe rather than a dangling pointer,
// and which makes numpy refuse a subsequent in-place `ndarray.resize()` --
// numpy will not reallocate an array that something else references.

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "soma/base/ids.hpp"
#include "soma/base/rational.hpp"
#include "soma/base/result.hpp"
#include "soma/base/units.hpp"
#include "soma/field/registry.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/graph/descriptor.hpp"
#include "soma/graph/validate.hpp"
#include "soma/graph/yaml_io.hpp"
#include "soma/ir/dump.hpp"
#include "soma/ir/lowering.hpp"
#include "soma/mesh/serial_backend.hpp"
#include "soma/runtime/component.hpp"
#include "soma/runtime/runner.hpp"
#include "soma/schedule/clock.hpp"
#include "soma/schedule/plan.hpp"
#include "soma/soma_c.h"

namespace py = pybind11;

namespace {

// ---------------------------------------------------------------------------
// Exception types
// ---------------------------------------------------------------------------

// Held for the lifetime of the interpreter. `py::object` at namespace scope is
// deliberately avoided (its destructor would run at static-destruction time,
// after the interpreter may already be finalized); a leaked reference to a
// module-level exception class is the standard, and correct, trade.
struct ExcTypes {
  py::handle base;
  py::handle invalid;
  py::handle notfound;
  py::handle state;
  py::handle runtime;
  py::handle unsupported;
  py::handle validation;
};

ExcTypes& exc() {
  static ExcTypes e;
  return e;
}

/// PyErr_NewException with a *tuple* of bases, which pybind11's
/// `py::exception<>` cannot express. The second base is what makes
/// `except ValueError:` catch `soma.InvalidArgument`.
py::handle make_exception(py::module_& m, const char* name,
                          const std::vector<py::handle>& bases,
                          const char* doc) {
  py::tuple base_tuple(bases.size());
  for (std::size_t i = 0; i < bases.size(); ++i) {
    base_tuple[i] = py::reinterpret_borrow<py::object>(bases[i]);
  }
  const std::string qualified = std::string("soma.") + name;
  PyObject* raw = PyErr_NewExceptionWithDoc(
      qualified.c_str(), doc, base_tuple.ptr(), nullptr);
  if (raw == nullptr) throw py::error_already_set();
  py::object obj = py::reinterpret_steal<py::object>(raw);
  m.add_object(name, obj);
  // Intentionally leaked: see ExcTypes.
  return py::handle(obj.release());
}

[[noreturn]] void raise_soma(py::handle type, const std::string& message,
                             std::optional<int> code) {
  py::object inst = py::reinterpret_borrow<py::object>(type)(message);
  inst.attr("code") = code.has_value() ? py::cast(*code) : py::none();
  PyErr_SetObject(type.ptr(), inst.ptr());
  throw py::error_already_set();
}

[[noreturn]] void raise_invalid(const std::string& message) {
  raise_soma(exc().invalid, message, std::nullopt);
}

[[noreturn]] void raise_state(const std::string& message) {
  raise_soma(exc().state, message, std::nullopt);
}

[[noreturn]] void raise_runtime(const std::string& message) {
  raise_soma(exc().runtime, message, std::nullopt);
}

[[noreturn]] void raise_notfound(const std::string& message) {
  raise_soma(exc().notfound, message, std::nullopt);
}

py::handle exception_for_code(int code) {
  switch (code) {
    case SOMA_ERR_INVALID:     return exc().invalid;
    case SOMA_ERR_NOTFOUND:    return exc().notfound;
    case SOMA_ERR_STATE:       return exc().state;
    case SOMA_ERR_UNSUPPORTED: return exc().unsupported;
    case SOMA_ERR_RUNTIME:
    default:                   return exc().runtime;
  }
}

[[noreturn]] void raise_for_code(int code, const std::string& message) {
  raise_soma(exception_for_code(code), message, code);
}

/// Raise a `soma.ValidationError` carrying the whole diagnostic list. Spec
/// Global Constraints: report every problem together, so the Python caller gets
/// the object, not just the first line of text.
[[noreturn]] void raise_validation(const std::string& what,
                                   const soma::Diagnostics& diags) {
  py::object inst = py::reinterpret_borrow<py::object>(exc().validation)(
      what + ":\n" + diags.to_string());
  inst.attr("code") = py::none();
  inst.attr("diagnostics") = py::cast(diags);
  PyErr_SetObject(exc().validation.ptr(), inst.ptr());
  throw py::error_already_set();
}

/// A `soma::Error` from the C++ core, which carries a stable code string.
[[noreturn]] void raise_core_error(const std::string& what,
                                   const soma::Error& e) {
  py::object inst = py::reinterpret_borrow<py::object>(exc().runtime)(
      what + ": " + e.code + ": " + e.message);
  inst.attr("code") = py::none();
  inst.attr("core_code") = py::cast(e.code);
  PyErr_SetObject(exc().runtime.ptr(), inst.ptr());
  throw py::error_already_set();
}

// ---------------------------------------------------------------------------
// Rational coercion
// ---------------------------------------------------------------------------

/// Accepts `soma.Rational`, `int`, `"1/3"`, `fractions.Fraction`, and anything
/// else exposing integral `.numerator` / `.denominator`.
///
/// `float` is deliberately NOT accepted. Spec §4.3 requires exact coupling
/// times, and 0.1 is not 1/10; silently taking the nearest double is exactly
/// the drift the rational clock exists to prevent. `Fraction(1, 10)` or "1/10"
/// says what was meant.
soma::Rational coerce_rational(py::handle value, const char* what) {
  if (value.is_none()) {
    raise_invalid(std::string(what) + ": expected a rational, got None");
  }
  if (py::isinstance<soma::Rational>(value)) {
    return value.cast<soma::Rational>();
  }
  if (py::isinstance<py::bool_>(value)) {
    raise_invalid(std::string(what) + ": a bool is not a coupling interval");
  }
  if (py::isinstance<py::float_>(value)) {
    raise_invalid(
        std::string(what) +
        ": a float is not accepted, because spec 4.3 requires exact coupling "
        "times and a binary double cannot represent 1/3 or 1/10. Pass an int, "
        "a string such as '1/3', or a fractions.Fraction.");
  }
  if (py::isinstance<py::int_>(value)) {
    return soma::Rational{value.cast<std::int64_t>()};
  }
  if (py::isinstance<py::str>(value)) {
    const std::string text = value.cast<std::string>();
    const auto        slash = text.find('/');
    try {
      if (slash == std::string::npos) {
        return soma::Rational{static_cast<std::int64_t>(std::stoll(text))};
      }
      const std::int64_t n = std::stoll(text.substr(0, slash));
      const std::int64_t d = std::stoll(text.substr(slash + 1));
      return soma::Rational{n, d};
    } catch (const std::exception& e) {
      raise_invalid(std::string(what) + ": cannot read '" + text +
                    "' as a rational: " + e.what());
    }
  }
  if (py::hasattr(value, "numerator") && py::hasattr(value, "denominator")) {
    try {
      const auto n = value.attr("numerator").cast<std::int64_t>();
      const auto d = value.attr("denominator").cast<std::int64_t>();
      return soma::Rational{n, d};
    } catch (const py::error_already_set&) {
      throw;
    } catch (const std::exception& e) {
      raise_invalid(std::string(what) + ": " + e.what());
    }
  }
  raise_invalid(std::string(what) + ": expected soma.Rational, int, str or " +
                "fractions.Fraction, got " +
                py::str(value.get_type().attr("__name__")).cast<std::string>());
}

// ---------------------------------------------------------------------------
// NumPy buffer inspection
// ---------------------------------------------------------------------------

/// What a validated array tells us, in SOMA's own vocabulary. Strides are in
/// doubles, matching soma::Layout and soma_register_field().
struct ArrayView {
  py::object  owner;              ///< strong reference; the lifetime guarantee
  double*     data = nullptr;
  std::size_t count = 0;
  std::size_t stride = 1;
  std::size_t component_stride = 0;  // 0 means 1 (interleaved)
  int         components = 1;
};

ArrayView inspect_array(py::handle obj, const std::string& what) {
  if (!py::isinstance<py::array>(obj)) {
    raise_invalid(what + ": expected a numpy.ndarray, got " +
                  py::str(obj.get_type().attr("__name__")).cast<std::string>() +
                  ". SOMA binds the component's own storage; a list or a "
                  "sequence would have to be copied, and a copy is not the "
                  "component's storage.");
  }
  py::array arr = py::reinterpret_borrow<py::array>(obj);

  if (!arr.dtype().equal(py::dtype::of<double>())) {
    raise_invalid(what + ": dtype must be float64, got " +
                  py::str(arr.dtype()).cast<std::string>() +
                  ". SOMA fields are double; casting would silently copy.");
  }

  const py::object flags = arr.attr("flags");
  if (!flags.attr("c_contiguous").cast<bool>()) {
    raise_invalid(
        what + ": the array is not C-contiguous (shape=" +
        py::str(arr.attr("shape")).cast<std::string>() +
        ", strides=" + py::str(arr.attr("strides")).cast<std::string>() +
        "). A strided view such as a[::2] or a transpose does not describe a "
        "soma::Layout run, and binding it would read the wrong doubles. Pass "
        "numpy.ascontiguousarray(a) -- and keep THAT array alive, since it is "
        "a different buffer.");
  }
  if (!flags.attr("writeable").cast<bool>()) {
    raise_invalid(what +
                  ": the array is read-only. SOMA writes into a Requires "
                  "field, so a read-only buffer cannot be bound.");
  }

  const auto ndim = static_cast<int>(arr.ndim());
  if (ndim != 1 && ndim != 2) {
    raise_invalid(what +
                  ": expected a 1-D (count,) or 2-D (count, components) "
                  "array, got " +
                  std::to_string(ndim) + " dimensions");
  }

  ArrayView v;
  v.owner = py::reinterpret_borrow<py::object>(obj);
  v.count = static_cast<std::size_t>(arr.shape(0));
  if (ndim == 1) {
    v.components = 1;
    v.stride = 1;
    v.component_stride = 0;
  } else {
    const auto ncomp = static_cast<std::int64_t>(arr.shape(1));
    if (ncomp < 1) {
      raise_invalid(what +
                    ": a 2-D array must have at least one component; got "
                    "shape " +
                    py::str(arr.attr("shape")).cast<std::string>());
    }
    if (ncomp > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      raise_invalid(what + ": component count does not fit an int");
    }
    v.components = static_cast<int>(ncomp);
    // C-contiguous (count, components) is interleaved: one element's
    // components sit next to each other.
    v.stride = static_cast<std::size_t>(ncomp);
    v.component_stride = 1;
  }
  v.data = static_cast<double*>(arr.mutable_data());
  if (v.data == nullptr && v.count != 0) {
    raise_invalid(what + ": the array has no data pointer");
  }
  return v;
}

/// A numpy array that ALIASES `ref`'s storage. `base` is what tells numpy not
/// to own the memory; without it pybind11 would copy, which would break the
/// "view, not a copy" contract silently.
py::object numpy_view(const soma::FieldRef& ref, py::object base) {
  const soma::Layout& l = ref.layout();
  auto*               data = const_cast<double*>(ref.data());
  const auto          elem = static_cast<py::ssize_t>(sizeof(double));

  if (base.is_none()) {
    // Nothing to tie the lifetime to: a capsule with a no-op deleter marks the
    // array as non-owning without inventing an owner.
    base = py::capsule(data, [](void*) {});
  }

  if (l.components == 1) {
    return py::array_t<double>(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.count)},
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.stride) * elem},
        data, base);
  }
  return py::array_t<double>(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.count),
                               static_cast<py::ssize_t>(l.components)},
      std::vector<py::ssize_t>{
          static_cast<py::ssize_t>(l.stride) * elem,
          static_cast<py::ssize_t>(l.effective_component_stride()) * elem},
      data, base);
}

// ---------------------------------------------------------------------------
// FieldRegistry, numpy-aware
// ---------------------------------------------------------------------------

/// Keeps every bound array alive. Shared, because a `PyFieldRegistry` handed to
/// `Component.register_fields` is a temporary view onto the session's registry
/// and must not take the references with it when it dies.
struct KeepAlive {
  std::unordered_map<std::uint32_t, py::object> refs;
};

/// `soma.FieldRegistry`. Either owns a `soma::FieldRegistry` (standalone use)
/// or points at somebody else's (the session's, during register_fields).
class PyFieldRegistry {
 public:
  PyFieldRegistry()
      : owned_(std::make_shared<soma::FieldRegistry>()),
        reg_(owned_.get()),
        keep_(std::make_shared<KeepAlive>()) {}

  /// A borrowed view onto a registry this object does NOT own. Valid only for
  /// as long as `reg` is; used when something other than a `Session` supplies
  /// the `soma::FieldRegistry&` to `Component::register_fields`. The owning
  /// case goes through the copy constructor instead, which shares `owned_` and
  /// so cannot dangle.
  PyFieldRegistry(soma::FieldRegistry* reg, std::shared_ptr<KeepAlive> keep)
      : reg_(reg), keep_(std::move(keep)) {}

  soma::FieldRegistry& raw() { return *reg_; }

  /// The graph, when one is known, so a shape mismatch is diagnosed HERE --
  /// against the declared component count -- rather than being discovered from
  /// the text of an error message further down.
  void set_graph(const soma::CouplingGraph* g) { graph_ = g; }

  void bind_numpy(soma::FieldId id, py::handle array) {
    const std::string what =
        "FieldRegistry.bind_numpy(field " + std::to_string(id.value()) + ")";
    ArrayView v = inspect_array(array, what);

    if (graph_ != nullptr) {
      if (!id.valid() || id.value() >= graph_->field_count()) {
        raise_invalid(what + ": field id " + std::to_string(id.value()) +
                      " is outside [0, " +
                      std::to_string(graph_->field_count()) + ")");
      }
      const soma::FieldSpec& spec = graph_->field(id);
      if (spec.components != v.components) {
        raise_invalid(
            what + ": field '" + spec.name + "' is declared with " +
            std::to_string(spec.components) + " component(s) but the array " +
            "supplies " + std::to_string(v.components) +
            ". A 1-D array is one component per entity; a 2-D (count, "
            "components) array is C-contiguous interleaved.");
      }
    }

    const soma::Layout layout{v.stride, v.component_stride, v.count,
                              v.components};
    const auto         r = reg_->bind(
        id, soma::FieldRef{v.data, layout, soma::MemorySpace::Host});
    if (!r.ok()) {
      // Same split soma_register_field() makes: a rebind is a sequencing
      // mistake, everything else is a malformed argument.
      if (r.error().code == "E-FIELD-REBIND") {
        raise_state(what + ": " + r.error().code + ": " + r.error().message);
      }
      raise_invalid(what + ": " + r.error().code + ": " + r.error().message);
    }
    keep_->refs[id.value()] = std::move(v.owner);
  }

  [[nodiscard]] py::object get_numpy(soma::FieldId id) const {
    const auto r = reg_->get(id);
    if (!r.ok()) {
      raise_notfound("FieldRegistry.get_numpy: " + r.error().code + ": " +
                     r.error().message);
    }
    const auto it = keep_->refs.find(id.value());
    py::object base =
        it == keep_->refs.end() ? py::none() : py::object(it->second);
    return numpy_view(r.value(), std::move(base));
  }

  [[nodiscard]] bool is_bound(soma::FieldId id) const {
    return reg_->is_bound(id);
  }
  [[nodiscard]] std::size_t bound_count() const { return reg_->bound_count(); }

  [[nodiscard]] std::shared_ptr<KeepAlive> keep() const { return keep_; }

 private:
  std::shared_ptr<soma::FieldRegistry> owned_;  // null when wrapping
  soma::FieldRegistry*                 reg_ = nullptr;
  std::shared_ptr<KeepAlive>           keep_;
  const soma::CouplingGraph*           graph_ = nullptr;
};

// ---------------------------------------------------------------------------
// soma.PyComponent and the adapter that makes it a peer
// ---------------------------------------------------------------------------

/// The Python-facing base class. Deliberately NOT a pybind11 trampoline over
/// `soma::Component`: `ComponentSet::add` takes a `unique_ptr<Component>`, and
/// handing it a trampoline whose real owner is the Python object is a
/// double-ownership bug waiting for the first refcount to go the wrong way.
/// `ForwardingComponent` below holds a strong reference to the Python object
/// instead, which has exactly one owner and one lifetime.
class PyComponent {
 public:
  explicit PyComponent(std::string name) : name_(std::move(name)) {
    if (name_.empty()) {
      raise_invalid(
          "PyComponent: the name must be non-empty and must match a component "
          "name in the coupling graph");
    }
  }
  virtual ~PyComponent() = default;
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  std::string name_;
};

/// Bridges a Python object into `soma::Component`.
///
/// Every method here is called from the C++ runtime, which may be running with
/// the GIL released, so each reacquires it before touching Python. Each also
/// catches `py::error_already_set` -- whose constructor FETCHES the Python
/// error, leaving the interpreter's error indicator clear -- and turns it into
/// a `soma::Result<void>` failure so the C++ runtime unwinds through its own
/// error channel. The original exception object is stashed and re-raised
/// verbatim once control reaches the binding boundary, so the user sees their
/// own traceback rather than a stringified copy.
class ForwardingComponent final : public soma::Component {
 public:
  ForwardingComponent(py::object obj, std::string name,
                      PyFieldRegistry registry)
      : obj_(std::move(obj)),
        name_(std::move(name)),
        registry_(std::move(registry)),
        keep_(registry_.keep()) {
    has_save_ = py::hasattr(obj_, "save_state");
    has_restore_ = py::hasattr(obj_, "restore_state");
    rollback_ = has_save_ && has_restore_;
  }

  ~ForwardingComponent() override {
    // py::object's destructor decrefs, which requires the GIL. The session that
    // owns this component is destroyed from Python (GIL held), but a stray
    // destruction order is cheap to defend against.
    //
    // Everything holding a Python reference is released INSIDE the guard,
    // members included: `gil` is a local of this body and is released before
    // member destructors run, so a `registry_` still holding the last
    // reference to a bound numpy array would be decref'd outside it.
    py::gil_scoped_acquire gil;
    obj_ = py::object();
    pending_ = py::object();
    registry_ = PyFieldRegistry{};
    keep_.reset();
  }

  // Returns the name captured at construction: no Python call, so this is safe
  // to call from the runtime's hot loop with or without the GIL.
  [[nodiscard]] std::string name() const override { return name_; }

  soma::Result<void> register_fields(
      soma::FieldRegistry& registry,
      const soma::CouplingGraph& graph) override {
    return call("register_fields", [&](py::object& fn) {
      // A COPY of the session's registry, not a raw pointer to it: the copy
      // shares the owning shared_ptr, so a component that squirrels the
      // `registry` argument away and uses it after the session is gone gets a
      // live object rather than a dangling one. The borrowed-pointer form is
      // only reached if some other driver supplies a registry this component
      // was not built with, and it is valid only for the duration of the call.
      PyFieldRegistry view =
          (&registry == &registry_.raw())
              ? registry_
              : PyFieldRegistry{&registry, keep_};
      view.set_graph(&graph);
      fn(py::cast(std::move(view)),
         py::cast(graph, py::return_value_policy::reference));
    });
  }

  soma::Result<void> advance(const soma::StepContext& ctx) override {
    return call("advance", [&](py::object& fn) { fn(py::cast(ctx)); });
  }

  soma::Result<void> save_state() override {
    if (!has_save_) return soma::Result<void>::success();
    return call("save_state", [](py::object& fn) { fn(); });
  }

  soma::Result<void> restore_state() override {
    if (!has_restore_) return soma::Result<void>::success();
    return call("restore_state", [](py::object& fn) { fn(); });
  }

  [[nodiscard]] bool supports_rollback() const override { return rollback_; }

  /// Re-raise the Python exception this component swallowed, if any. Called at
  /// the binding boundary with the GIL held.
  void rethrow_pending() {
    if (!pending_) return;
    py::object value = pending_;
    pending_ = py::object();
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())),
                    value.ptr());
    throw py::error_already_set();
  }

 private:
  template <class F>
  soma::Result<void> call(const char* method, F&& body) {
    py::gil_scoped_acquire gil;
    try {
      py::object fn = obj_.attr(method);
      body(fn);
      return soma::Result<void>::success();
    } catch (py::error_already_set& e) {
      pending_ = e.value();
      return soma::Error{"E-PY-CALLBACK", "component '" + name_ + "'." +
                                              method +
                                              "() raised: " + e.what()};
    } catch (const std::exception& e) {
      return soma::Error{"E-PY-CALLBACK", "component '" + name_ + "'." +
                                              method +
                                              "() failed: " + e.what()};
    } catch (...) {
      return soma::Error{"E-PY-CALLBACK", "component '" + name_ + "'." +
                                              method +
                                              "() threw a non-std exception"};
    }
  }

  py::object                 obj_;
  std::string                name_;
  PyFieldRegistry            registry_;  // shares the session's registry
  std::shared_ptr<KeepAlive> keep_;
  py::object                 pending_;
  bool                       has_save_ = false;
  bool                       has_restore_ = false;
  bool                       rollback_ = false;
};

// ---------------------------------------------------------------------------
// soma.Session -- the C++ runtime, driven from Python
// ---------------------------------------------------------------------------

/// Owns one copy of the graph and everything derived from it. The copy matters:
/// `soma::Runner` holds `const CouplingGraph&` for its whole life, so a session
/// that aliased the caller's Python object would dangle the moment that object
/// was rebound.
class Session {
 public:
  explicit Session(const soma::CouplingGraph& graph) {
    // An explicit temporary, cast as an rvalue. `py::cast(graph)` would NOT
    // copy: pybind11 looks the source pointer up in its registry first, finds
    // the caller's live Python object wrapping that very CouplingGraph, and
    // hands it straight back whatever the return-value policy says (observed:
    // a component added to the caller's graph after Session construction
    // appeared inside the session). Casting a temporary has no registered
    // instance to find, so a genuinely new one is made.
    graph_obj_ = py::cast(soma::CouplingGraph(graph));
    registry_obj_ = py::cast(PyFieldRegistry{});
    registry().set_graph(&this->graph());
  }

  soma::CouplingGraph& graph() {
    return graph_obj_.cast<soma::CouplingGraph&>();
  }
  PyFieldRegistry& registry() { return registry_obj_.cast<PyFieldRegistry&>(); }
  [[nodiscard]] py::object graph_object() const { return graph_obj_; }
  [[nodiscard]] py::object registry_object() const { return registry_obj_; }

  void add_component(py::object obj) {
    if (initialized_) {
      raise_state("Session.add_component: the session is already initialized");
    }
    if (!py::isinstance<PyComponent>(obj)) {
      raise_invalid(
          "Session.add_component: expected a soma.PyComponent subclass "
          "instance, got " +
          py::str(obj.get_type().attr("__name__")).cast<std::string>());
    }
    const std::string name = obj.cast<PyComponent&>().name();
    if (!graph().find_component(name).has_value()) {
      raise_notfound(
          "Session.add_component: the graph declares no component named '" +
          name + "'");
    }
    auto owned = std::make_unique<ForwardingComponent>(std::move(obj), name,
                                                       registry());
    ForwardingComponent* raw = owned.get();
    try {
      components_.add(std::move(owned));
    } catch (const std::invalid_argument& e) {
      raise_invalid(std::string("Session.add_component: ") + e.what());
    }
    forwarders_.push_back(raw);
  }

  void initialize(const soma::RunnerConfig& config) {
    if (initialized_) raise_state("Session.initialize: already initialized");
    soma::CouplingGraph& g = graph();

    // 1. Meshes, in declaration order, so the backend's MeshIds match the
    //    graph's (MeshBackend::load's MESH ID CONTRACT).
    for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
      const auto r = backend_.load(g.mesh(soma::MeshId{i}));
      if (!r.ok()) {
        raise_core_error("Session.initialize: mesh load failed", r.error());
      }
    }

    // 2. Validate, lower and schedule. Pure C++ from here to the end of the
    //    block, so the GIL goes.
    soma::LoweringResult lowered{soma::OperatorDAG{}, soma::Diagnostics{}};
    soma::ScheduleResult scheduled;
    soma::Diagnostics    validation;
    {
      py::gil_scoped_release release;
      validation = soma::validate(g);
      if (!validation.has_errors()) {
        lowered = soma::lower(g);
        if (lowered.ok()) {
          const soma::ClockSet clocks{g};
          scheduled = soma::schedule(g, lowered.dag, clocks,
                                     soma::make_region_sizes(g, backend_));
        }
      }
    }
    if (validation.has_errors()) {
      raise_validation("Session.initialize: the graph does not validate",
                       validation);
    }
    if (!lowered.ok()) {
      raise_validation("Session.initialize: the graph does not lower",
                       lowered.diags);
    }
    if (!scheduled.ok()) {
      raise_validation("Session.initialize: the graph does not schedule",
                       scheduled.diags);
    }
    dag_ = std::move(lowered.dag);
    plan_ = std::move(scheduled.plan);

    // 3. Let every component bind its storage. This calls into Python, so the
    //    GIL is held; a raising component is re-raised verbatim below.
    for (soma::Component& c : components_) {
      const auto r = c.register_fields(registry().raw(), g);
      if (!r.ok()) {
        rethrow_pending();
        raise_core_error("Session.initialize: register_fields failed",
                         r.error());
      }
    }

    // 4. Runner setup. No Python is reachable from here.
    runner_ = std::make_unique<soma::Runner>(g, dag_, plan_, backend_);
    soma::Result<void> r = soma::Result<void>::success();
    {
      py::gil_scoped_release release;
      r = runner_->initialize(components_, registry().raw(), config);
    }
    if (!r.ok()) {
      runner_.reset();
      raise_core_error("Session.initialize", r.error());
    }
    initialized_ = true;
  }

  void run_period() { run(1); }

  void run(std::int64_t periods) {
    if (!initialized_) raise_state("Session.run: call initialize() first");
    if (periods < 0) {
      raise_invalid("Session.run: periods must be >= 0, got " +
                    std::to_string(periods));
    }
    soma::Result<void> r = soma::Result<void>::success();
    {
      py::gil_scoped_release release;
      r = runner_->run(components_, periods);
    }
    rethrow_pending();
    if (!r.ok()) raise_core_error("Session.run", r.error());
  }

  [[nodiscard]] const soma::RunStats& stats() const {
    if (!initialized_) raise_state("Session.stats: call initialize() first");
    return runner_->stats();
  }

  [[nodiscard]] const soma::OperatorDAG& dag() const {
    if (!initialized_) raise_state("Session.dag: call initialize() first");
    return dag_;
  }
  [[nodiscard]] const soma::ExecutionPlan& plan() const {
    if (!initialized_) raise_state("Session.plan: call initialize() first");
    return plan_;
  }

 private:
  void rethrow_pending() {
    for (ForwardingComponent* f : forwarders_) f->rethrow_pending();
  }

  py::object                        graph_obj_;
  py::object                        registry_obj_;
  soma::SerialMeshBackend           backend_;
  soma::OperatorDAG                 dag_;
  soma::ExecutionPlan               plan_;
  soma::ComponentSet                components_;
  std::vector<ForwardingComponent*> forwarders_;  // non-owning
  std::unique_ptr<soma::Runner>     runner_;
  bool                              initialized_ = false;
};

// ---------------------------------------------------------------------------
// soma.Context -- the C ABI of include/soma/soma_c.h
// ---------------------------------------------------------------------------

/// One Python object per `soma_ctx`.
///
/// A mirror copy of the graph is kept alongside the handle. That is what closes
/// the shape-query gap on the Python side without touching the frozen header:
/// the ABI has no `soma_field_components()`, so a binding that only had the
/// handle would have to discover a component-count mismatch by attempting the
/// registration and reading the English text of the resulting error. Because
/// every path into `load_graph`/`load_yaml` here also produces a
/// `soma::CouplingGraph` on the Python side, `register_field` can compare the
/// declared count with the array's shape BEFORE calling the ABI, and report
/// both numbers itself.
class Context {
 public:
  Context() {
    const int rc = soma_create(&ctx_);
    if (rc != SOMA_OK) {
      // soma_create() has no context to attach the message to; the thread-local
      // slot is the only place it exists.
      raise_for_code(rc, "soma_create failed: " + thread_error());
    }
  }

  ~Context() { close_noexcept(); }

  Context(const Context&) = delete;
  Context& operator=(const Context&) = delete;

  void close() {
    if (ctx_ == nullptr) return;
    const int rc = soma_destroy(ctx_);
    ctx_ = nullptr;
    keep_.clear();
    graph_.reset();
    if (rc != SOMA_OK) {
      raise_for_code(rc, "soma_destroy failed: " + thread_error());
    }
  }

  [[nodiscard]] bool closed() const noexcept { return ctx_ == nullptr; }

  [[nodiscard]] std::string last_error() const {
    if (ctx_ == nullptr) return thread_error();
    std::uint64_t n = 0;
    if (soma_last_error_length(ctx_, &n) != SOMA_OK) return {};
    std::string buf(static_cast<std::size_t>(n) + 1, '\0');
    if (soma_last_error(ctx_, buf.data(), n + 1) != SOMA_OK) return {};
    buf.resize(static_cast<std::size_t>(n));
    return buf;
  }

  void load_graph(py::handle source) {
    soma_ctx               c = live("load_graph");
    std::vector<std::byte> bytes;
    soma::CouplingGraph    mirror;

    if (py::isinstance<soma::CouplingGraph>(source)) {
      mirror = source.cast<const soma::CouplingGraph&>();
      bytes = soma::serialize(mirror);
    } else if (py::isinstance<py::bytes>(source) ||
               py::isinstance<py::bytearray>(source)) {
      const std::string raw = source.cast<std::string>();
      bytes.resize(raw.size());
      if (!raw.empty()) std::memcpy(bytes.data(), raw.data(), raw.size());
      auto parsed = soma::deserialize(bytes);
      if (!parsed.ok()) {
        raise_invalid("Context.load_graph: " + parsed.error().code + ": " +
                      parsed.error().message);
      }
      mirror = std::move(parsed.value());
    } else {
      raise_invalid(
          "Context.load_graph: expected a soma.CouplingGraph or a bytes "
          "descriptor, got " +
          py::str(source.get_type().attr("__name__")).cast<std::string>());
    }

    check(soma_load_graph(c, bytes.data(),
                          static_cast<std::uint64_t>(bytes.size())),
          "soma_load_graph");
    graph_ = std::move(mirror);
  }

  void load_yaml(const std::string& path) {
    soma_ctx c = live("load_yaml");
    check(soma_load_graph_yaml(c, path.c_str()), "soma_load_graph_yaml");
    // The ABI accepted the file; parse it again with the same reader so the
    // Python side has the shape information the ABI cannot report. A
    // disagreement here would be a genuine library bug, not a user error.
    auto parsed = soma::load_yaml_file(path);
    if (!parsed.ok()) {
      raise_runtime("Context.load_yaml: soma_load_graph_yaml() accepted '" +
                    path +
                    "' but re-reading it for the shape mirror failed: " +
                    parsed.error().code + ": " + parsed.error().message);
    }
    graph_ = std::move(parsed.value());
  }

  [[nodiscard]] py::bytes save_graph() {
    soma_ctx      c = live("save_graph");
    std::uint64_t n = 0;
    check(soma_save_graph(c, nullptr, 0, &n), "soma_save_graph (sizing)");
    std::string buf(static_cast<std::size_t>(n), '\0');
    check(soma_save_graph(c, buf.data(), n, &n), "soma_save_graph");
    return py::bytes(buf);
  }

  [[nodiscard]] std::int32_t field_handle(const std::string& component,
                                          const std::string& field) {
    soma_ctx     c = live("field_handle");
    std::int32_t h = -1;
    check(soma_field_handle(c, component.c_str(), field.c_str(), &h),
          "soma_field_handle");
    return h;
  }

  /// The declared component count for a handle, from the mirrored graph. The C
  /// ABI has no equivalent; see the class comment.
  [[nodiscard]] int field_components(std::int32_t handle) {
    live("field_components");
    return mirror_field(handle, "field_components").components;
  }

  [[nodiscard]] std::string field_name(std::int32_t handle) {
    live("field_name");
    return mirror_field(handle, "field_name").name;
  }

  void register_field(std::int32_t handle, py::handle array,
                      std::int32_t memspace) {
    soma_ctx          c = live("register_field");
    const std::string what =
        "Context.register_field(handle " + std::to_string(handle) + ")";
    ArrayView v = inspect_array(array, what);

    if (graph_.has_value()) {
      const soma::FieldSpec& spec = mirror_field(handle, "register_field");
      if (spec.components != v.components) {
        // Reported from the declared count directly. Nothing here reads the
        // ABI's error string to find out what the graph says.
        raise_invalid(what + ": field '" + spec.name + "' is declared with " +
                      std::to_string(spec.components) +
                      " component(s) but the array supplies " +
                      std::to_string(v.components) +
                      ". A 1-D array is one component per entity; a 2-D "
                      "(count, components) array is C-contiguous interleaved.");
      }
    }

    check(soma_register_field(c, handle, v.data,
                              static_cast<std::uint64_t>(v.count),
                              static_cast<std::uint64_t>(v.stride),
                              static_cast<std::uint64_t>(v.component_stride),
                              v.components, memspace),
          "soma_register_field");
    // Only after the ABI accepted it: a refused registration must not pin a
    // reference to an array SOMA is not using.
    keep_[handle] = std::move(v.owner);
  }

  void set_communicator(std::int32_t fortran_comm) {
    check(soma_set_communicator(live("set_communicator"), fortran_comm),
          "soma_set_communicator");
  }

  void set_end_time(py::handle value) {
    const soma::Rational r = coerce_rational(value, "Context.set_end_time");
    check(soma_set_end_time(live("set_end_time"), r.num(), r.den()),
          "soma_set_end_time");
  }

  void initialize() {
    soma_ctx c = live("initialize");
    int      rc = SOMA_OK;
    {
      // Validation, lowering and clock construction: real work, no Python.
      py::gil_scoped_release release;
      rc = soma_initialize(c);
    }
    check(rc, "soma_initialize");
  }

  [[nodiscard]] std::pair<double, double> begin_step(
      const std::string& component) {
    double t0 = 0.0;
    double t1 = 0.0;
    check(soma_begin_step(live("begin_step"), component.c_str(), &t0, &t1),
          "soma_begin_step");
    return {t0, t1};
  }

  void end_step(const std::string& component) {
    check(soma_end_step(live("end_step"), component.c_str()), "soma_end_step");
  }

  [[nodiscard]] bool requires_checkpoint(const std::string& component) {
    std::int32_t out = 0;
    check(soma_requires_checkpoint(live("requires_checkpoint"),
                                   component.c_str(), &out),
          "soma_requires_checkpoint");
    return out != 0;
  }

  [[nodiscard]] bool requires_restore(const std::string& component) {
    std::int32_t out = 0;
    check(soma_requires_restore(live("requires_restore"), component.c_str(),
                                &out),
          "soma_requires_restore");
    return out != 0;
  }

  [[nodiscard]] bool is_coupling_ongoing() {
    std::int32_t out = 0;
    check(soma_is_coupling_ongoing(live("is_coupling_ongoing"), &out),
          "soma_is_coupling_ongoing");
    return out != 0;
  }

  void run_period() {
    check(soma_run_period(live("run_period")), "soma_run_period");
  }

  void run(std::int64_t periods) {
    check(soma_run(live("run"), periods), "soma_run");
  }

 private:
  static std::string thread_error() {
    std::uint64_t n = 0;
    if (soma_last_error_length(nullptr, &n) != SOMA_OK) return {};
    std::string buf(static_cast<std::size_t>(n) + 1, '\0');
    if (soma_last_error(nullptr, buf.data(), n + 1) != SOMA_OK) return {};
    buf.resize(static_cast<std::size_t>(n));
    return buf;
  }

  soma_ctx live(const char* what) {
    if (ctx_ == nullptr) {
      raise_state(std::string("Context.") + what +
                  ": the context has been closed");
    }
    return ctx_;
  }

  const soma::FieldSpec& mirror_field(std::int32_t handle, const char* what) {
    if (!graph_.has_value()) {
      raise_state(std::string("Context.") + what +
                  ": no graph has been loaded");
    }
    if (handle < 0 ||
        static_cast<std::size_t>(handle) >= graph_->field_count()) {
      raise_invalid(std::string("Context.") + what + ": handle " +
                    std::to_string(handle) + " is outside [0, " +
                    std::to_string(graph_->field_count()) + ")");
    }
    return graph_->field(soma::FieldId{static_cast<std::uint32_t>(handle)});
  }

  void check(int rc, const char* call) {
    if (rc == SOMA_OK) return;
    raise_for_code(rc, std::string(call) + " failed: " + last_error());
  }

  void close_noexcept() noexcept {
    if (ctx_ == nullptr) return;
    (void)soma_destroy(ctx_);
    ctx_ = nullptr;
    try {
      keep_.clear();
      graph_.reset();
    } catch (...) {  // NOLINT(bugprone-empty-catch)
    }
  }

  soma_ctx                                     ctx_ = nullptr;
  std::optional<soma::CouplingGraph>           graph_;
  std::unordered_map<std::int32_t, py::object> keep_;
};

// ---------------------------------------------------------------------------
// Small helpers for the module body
// ---------------------------------------------------------------------------

template <class IdT>
void bind_id(py::module_& m, const char* name, const char* doc) {
  py::class_<IdT>(m, name, doc)
      .def(py::init<>())
      .def(py::init<std::uint32_t>(), py::arg("value"))
      .def_property_readonly("value", &IdT::value)
      .def_property_readonly("valid", &IdT::valid)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def("__hash__", [](const IdT& id) { return std::hash<IdT>{}(id); })
      .def("__repr__", [name](const IdT& id) {
        return std::string(name) + "(" +
               (id.valid() ? std::to_string(id.value())
                           : std::string("invalid")) +
               ")";
      });
}

}  // namespace

PYBIND11_MODULE(_soma, m) {
  m.doc() =
      "SOMA's Python front-end. See soma/__init__.py for the public surface.";

  // --- exceptions ----------------------------------------------------------
  ExcTypes& e = exc();
  e.base = make_exception(m, "SomaError", {py::handle(PyExc_Exception)},
                          "Base class of every error SOMA raises. Carries "
                          "`.code`, the SOMA_ERR_* integer, or None when the "
                          "failure was detected above the C ABI.");
  e.invalid = make_exception(
      m, "InvalidArgument", {e.base, py::handle(PyExc_ValueError)},
      "SOMA_ERR_INVALID: a malformed argument -- a null pointer, an "
      "out-of-range value, a bad handle, an array that is not C-contiguous "
      "float64.");
  e.notfound = make_exception(
      m, "NotFound", {e.base, py::handle(PyExc_LookupError)},
      "SOMA_ERR_NOTFOUND: a well-formed lookup that found nothing.");
  e.state = make_exception(
      m, "StateError", {e.base, py::handle(PyExc_RuntimeError)},
      "SOMA_ERR_STATE: the call is valid but not in this state.");
  e.runtime = make_exception(
      m, "SomaRuntimeError", {e.base, py::handle(PyExc_RuntimeError)},
      "SOMA_ERR_RUNTIME: an internal operation failed, or an exception was "
      "caught at a boundary.");
  e.unsupported = make_exception(
      m, "Unsupported", {e.base, py::handle(PyExc_NotImplementedError)},
      "SOMA_ERR_UNSUPPORTED: the operation exists in this ABI but this build "
      "cannot perform it.");
  e.validation = make_exception(
      m, "ValidationError", {e.invalid},
      "A graph that does not validate, lower or schedule. `.diagnostics` is "
      "the full soma.Diagnostics list -- every problem, not just the first.");

  // Anything the built-in translators do not recognise becomes a SomaError
  // rather than reaching std::terminate through an extern "C" frame.
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const std::exception&) {
      throw;  // let pybind11's built-in mapping handle std:: exceptions
    } catch (...) {
      PyErr_SetString(exc().runtime.ptr(),
                      "soma: an exception that does not derive from "
                      "std::exception crossed the binding boundary");
    }
  });

  // --- version -------------------------------------------------------------
  m.def("version", []() { return std::string(soma_version()); },
        "The library version string, from soma_version().");
  m.def("abi_version", []() { return soma_abi_version(); },
        "SOMA_ABI_VERSION as compiled into the library.");
  m.attr("MEMSPACE_HOST") = SOMA_MEMSPACE_HOST;
  m.attr("MEMSPACE_DEVICE") = SOMA_MEMSPACE_DEVICE;
  m.attr("OK") = SOMA_OK;
  m.attr("ERR_INVALID") = SOMA_ERR_INVALID;
  m.attr("ERR_NOTFOUND") = SOMA_ERR_NOTFOUND;
  m.attr("ERR_STATE") = SOMA_ERR_STATE;
  m.attr("ERR_RUNTIME") = SOMA_ERR_RUNTIME;
  m.attr("ERR_UNSUPPORTED") = SOMA_ERR_UNSUPPORTED;

  // --- enums ---------------------------------------------------------------
  py::enum_<soma::Discretization>(m, "Discretization")
      .value("FV", soma::Discretization::FV)
      .value("ContinuousGLL", soma::Discretization::ContinuousGLL)
      .value("DiscontinuousGLL", soma::Discretization::DiscontinuousGLL)
      .value("PointCloud", soma::Discretization::PointCloud);

  // `None` is a Python keyword, so the C++ enumerator `Conservation::None`
  // cannot keep its spelling here. `Nil` is the Python name.
  py::enum_<soma::Conservation>(m, "Conservation")
      .value("Nil", soma::Conservation::None)
      .value("Extensive", soma::Conservation::Extensive)
      .value("Intensive", soma::Conservation::Intensive);

  py::enum_<soma::Intent>(m, "Intent")
      .value("Provides", soma::Intent::Provides)
      .value("Requires", soma::Intent::Requires)
      .value("Bidirectional", soma::Intent::Bidirectional);

  py::enum_<soma::AccumOp>(m, "AccumOp")
      .value("Instant", soma::AccumOp::Instant)
      .value("Sum", soma::AccumOp::Sum)
      .value("Mean", soma::AccumOp::Mean)
      .value("Min", soma::AccumOp::Min)
      .value("Max", soma::AccumOp::Max)
      .value("Integral", soma::AccumOp::Integral);

  py::enum_<soma::TemporalBasis>(m, "TemporalBasis")
      .value("Nil", soma::TemporalBasis::None)
      .value("Constant", soma::TemporalBasis::Constant)
      .value("Linear", soma::TemporalBasis::Linear)
      .value("DG", soma::TemporalBasis::DG)
      .value("Spline", soma::TemporalBasis::Spline);

  // The bitwise operators are defined by hand, not obtained from
  // py::arithmetic(). pybind11 only synthesises __and__/__or__/__xor__ when
  // the enum is implicitly convertible to its underlying type, which a C++
  // `enum class` is not -- so `RemapFlags.Conservative | RemapFlags.Monotone`
  // raised TypeError with py::arithmetic() alone (observed). They are defined
  // over RemapFlags, not int, so composing two flags yields a RemapFlags that
  // ChannelBuilder.remap() accepts.
  py::enum_<soma::RemapFlags> remap_flags(m, "RemapFlags", py::arithmetic());
  remap_flags.value("Nil", soma::RemapFlags::None)
      .value("Conservative", soma::RemapFlags::Conservative)
      .value("Monotone", soma::RemapFlags::Monotone)
      .value("HighOrder", soma::RemapFlags::HighOrder)
      .def("__or__",
           [](soma::RemapFlags a, soma::RemapFlags b) { return a | b; },
           py::is_operator())
      .def("__and__",
           [](soma::RemapFlags a, soma::RemapFlags b) { return a & b; },
           py::is_operator())
      .def("__xor__",
           [](soma::RemapFlags a, soma::RemapFlags b) {
             return static_cast<soma::RemapFlags>(
                 static_cast<std::uint32_t>(a) ^ static_cast<std::uint32_t>(b));
           },
           py::is_operator())
      .def("__bool__",
           [](soma::RemapFlags a) {
             return static_cast<std::uint32_t>(a) != 0u;
           })
      .def("has", &soma::has, py::arg("bit"),
           "True when this set contains `bit`.");

  py::enum_<soma::Severity>(m, "Severity")
      .value("Warning", soma::Severity::Warning)
      .value("Error", soma::Severity::Error);

  py::enum_<soma::MemorySpace>(m, "MemorySpace")
      .value("Host", soma::MemorySpace::Host)
      .value("Device", soma::MemorySpace::Device);

  py::enum_<soma::ExpandMode>(m, "ExpandMode")
      .value("Auto", soma::ExpandMode::Auto)
      .value("ForceGeneral", soma::ExpandMode::ForceGeneral);

  // --- ids -----------------------------------------------------------------
  bind_id<soma::MeshId>(m, "MeshId", "Index of a mesh in a CouplingGraph.");
  bind_id<soma::RegionId>(m, "RegionId", "Index of a region.");
  bind_id<soma::ComponentId>(m, "ComponentId", "Index of a component.");
  bind_id<soma::FieldId>(m, "FieldId", "Index of a field.");
  bind_id<soma::ChannelId>(m, "ChannelId", "Index of a channel.");
  bind_id<soma::RegionIdx>(m, "IterationRegionIdx",
                           "Index of an iteration region.");

  // --- Rational ------------------------------------------------------------
  py::class_<soma::Rational>(m, "Rational",
                             "Exact rational time (spec 4.3). Never a double: "
                             "1/3 and 1/7 subcycling must not drift.")
      .def(py::init<>())
      .def(py::init<std::int64_t>(), py::arg("numerator"))
      .def(py::init<std::int64_t, std::int64_t>(), py::arg("numerator"),
           py::arg("denominator"))
      .def_static(
          "coerce",
          [](py::handle v) { return coerce_rational(v, "Rational.coerce"); },
          py::arg("value"),
          "Rational from a soma.Rational, int, '1/3' string, or "
          "fractions.Fraction. A float is refused on purpose.")
      .def_property_readonly("num", &soma::Rational::num)
      .def_property_readonly("den", &soma::Rational::den)
      .def_property_readonly("numerator", &soma::Rational::num)
      .def_property_readonly("denominator", &soma::Rational::den)
      .def("is_zero", &soma::Rational::is_zero)
      .def("is_integer", &soma::Rational::is_integer)
      .def("to_double", &soma::Rational::to_double)
      .def("to_string", &soma::Rational::to_string)
      .def("__float__", &soma::Rational::to_double)
      .def(py::self + py::self)
      .def(py::self - py::self)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(py::self < py::self)
      .def(py::self <= py::self)
      .def(py::self > py::self)
      .def(py::self >= py::self)
      .def("__hash__",
           [](const soma::Rational& r) {
             return py::hash(py::make_tuple(r.num(), r.den()));
           })
      .def("__str__", &soma::Rational::to_string)
      .def("__repr__", [](const soma::Rational& r) {
        return "Rational(" + std::to_string(r.num()) + ", " +
               std::to_string(r.den()) + ")";
      });

  // --- Units ---------------------------------------------------------------
  py::class_<soma::Units>(m, "Units",
                          "SI dimension exponents (spec 4.2). Exponent vectors "
                          "compose and compare; unit strings do not.")
      .def(py::init<>())
      .def(py::init([](int length, int mass, int time, int current,
                       int temperature, int amount, int luminosity) {
             const auto narrow = [](int v, const char* which) {
               if (v < -128 || v > 127) {
                 raise_invalid(std::string("Units: exponent '") + which +
                               "' does not fit int8_t");
               }
               return static_cast<std::int8_t>(v);
             };
             return soma::Units{narrow(length, "length"),
                                narrow(mass, "mass"),
                                narrow(time, "time"),
                                narrow(current, "current"),
                                narrow(temperature, "temperature"),
                                narrow(amount, "amount"),
                                narrow(luminosity, "luminosity")};
           }),
           py::arg("length") = 0, py::arg("mass") = 0, py::arg("time") = 0,
           py::arg("current") = 0, py::arg("temperature") = 0,
           py::arg("amount") = 0, py::arg("luminosity") = 0)
      .def_readwrite("length", &soma::Units::length)
      .def_readwrite("mass", &soma::Units::mass)
      .def_readwrite("time", &soma::Units::time)
      .def_readwrite("current", &soma::Units::current)
      .def_readwrite("temperature", &soma::Units::temperature)
      .def_readwrite("amount", &soma::Units::amount)
      .def_readwrite("luminosity", &soma::Units::luminosity)
      .def("is_dimensionless", &soma::Units::is_dimensionless)
      .def("to_string", &soma::Units::to_string)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &soma::Units::to_string)
      .def("__repr__", [](const soma::Units& u) {
        return "Units('" + u.to_string() + "')";
      });

  py::module_ units = m.def_submodule(
      "units", "Named SI unit vectors from include/soma/base/units.hpp.");
  units.attr("dimensionless") = soma::units::kDimensionless;
  units.attr("m") = soma::units::kMetre;
  units.attr("metre") = soma::units::kMetre;
  units.attr("kg") = soma::units::kKilogram;
  units.attr("kilogram") = soma::units::kKilogram;
  units.attr("s") = soma::units::kSecond;
  units.attr("second") = soma::units::kSecond;
  units.attr("K") = soma::units::kKelvin;
  units.attr("kelvin") = soma::units::kKelvin;
  units.attr("temperature") = soma::units::kTemperature;
  units.attr("N") = soma::units::kNewton;
  units.attr("newton") = soma::units::kNewton;
  units.attr("J") = soma::units::kJoule;
  units.attr("joule") = soma::units::kJoule;
  units.attr("W") = soma::units::kWatt;
  units.attr("watt") = soma::units::kWatt;
  units.attr("Pa") = soma::units::kPascal;
  units.attr("pascal") = soma::units::kPascal;
  units.attr("W_per_m2") = soma::units::kHeatFluxDensity;
  units.attr("heat_flux_density") = soma::units::kHeatFluxDensity;

  // --- graph value types ---------------------------------------------------
  py::class_<soma::MeshDecl>(m, "MeshDecl")
      .def(py::init([](std::string name, std::string source) {
             return soma::MeshDecl{std::move(name), std::move(source)};
           }),
           py::arg("name"), py::arg("source"))
      .def_readwrite("name", &soma::MeshDecl::name)
      .def_readwrite("source", &soma::MeshDecl::source)
      .def("__repr__", [](const soma::MeshDecl& d) {
        return "MeshDecl('" + d.name + "', '" + d.source + "')";
      });

  py::class_<soma::Region>(m, "Region")
      .def(py::init([](std::string name, soma::MeshId mesh,
                       std::string entity_set, int dimension) {
             return soma::Region{std::move(name), mesh, std::move(entity_set),
                                 dimension};
           }),
           py::arg("name"), py::arg("mesh"), py::arg("entity_set") = "",
           py::arg("dimension") = -1)
      .def_readwrite("name", &soma::Region::name)
      .def_readwrite("mesh", &soma::Region::mesh)
      .def_readwrite("entity_set", &soma::Region::entity_set)
      .def_readwrite("dimension", &soma::Region::dimension)
      .def("__repr__", [](const soma::Region& r) {
        return "Region('" + r.name +
               "', dimension=" + std::to_string(r.dimension) + ")";
      });

  py::class_<soma::RankSet>(m, "RankSet")
      .def(py::init([](int first, int count) {
             return soma::RankSet{first, count};
           }),
           py::arg("first") = 0, py::arg("count") = 1)
      .def_readwrite("first", &soma::RankSet::first)
      .def_readwrite("count", &soma::RankSet::count)
      .def("contains", &soma::RankSet::contains, py::arg("rank"))
      .def("overlaps", &soma::RankSet::overlaps, py::arg("other"))
      .def("last", &soma::RankSet::last)
      .def("__repr__", [](const soma::RankSet& r) {
        return "RankSet(first=" + std::to_string(r.first) +
               ", count=" + std::to_string(r.count) + ")";
      });

  py::class_<soma::ComponentDecl>(m, "ComponentDecl")
      .def(py::init<>())
      .def_readwrite("name", &soma::ComponentDecl::name)
      .def_readwrite("ranks", &soma::ComponentDecl::ranks)
      .def_readwrite("interval", &soma::ComponentDecl::interval)
      .def_readonly("fields", &soma::ComponentDecl::fields)
      .def("__repr__", [](const soma::ComponentDecl& c) {
        return "ComponentDecl('" + c.name +
               "', interval=" + c.interval.to_string() + ")";
      });

  py::class_<soma::FieldSpec>(m, "FieldSpec")
      .def(py::init<>())
      .def_readwrite("name", &soma::FieldSpec::name)
      .def_readwrite("region", &soma::FieldSpec::region)
      .def_readwrite("disc", &soma::FieldSpec::disc)
      .def_readwrite("order", &soma::FieldSpec::order)
      .def_readwrite("components", &soma::FieldSpec::components)
      .def_readwrite("units", &soma::FieldSpec::units)
      .def_readwrite("conservation", &soma::FieldSpec::conservation)
      .def_readwrite("intent", &soma::FieldSpec::intent)
      .def("__repr__", [](const soma::FieldSpec& f) {
        return "FieldSpec('" + f.name +
               "', components=" + std::to_string(f.components) +
               ", units=" + f.units.to_string() + ")";
      });

  py::class_<soma::Window> window(m, "Window");
  py::enum_<soma::Window::Kind>(window, "Kind")
      .value("CouplingPeriod", soma::Window::Kind::CouplingPeriod)
      .value("FromComponent", soma::Window::Kind::FromComponent)
      .value("Explicit", soma::Window::Kind::Explicit);
  window.def(py::init<>())
      .def_static("coupling_period", &soma::Window::coupling_period)
      .def_static("from_component", &soma::Window::from, py::arg("component"))
      .def_static(
          "explicit_length",
          [](py::handle len) {
            return soma::Window::explicit_length(
                coerce_rational(len, "Window.explicit_length"));
          },
          py::arg("length"))
      .def_readwrite("kind", &soma::Window::kind)
      .def_readwrite("component", &soma::Window::component)
      .def_readwrite("length", &soma::Window::length);

  py::class_<soma::Channel>(m, "Channel")
      .def_readonly("name", &soma::Channel::name)
      .def_readonly("source", &soma::Channel::source)
      .def_readonly("target", &soma::Channel::target)
      .def_readonly("lag", &soma::Channel::lag)
      .def_property_readonly(
          "accum_op", [](const soma::Channel& c) { return c.accumulate.op; })
      .def_property_readonly(
          "remap_flags", [](const soma::Channel& c) { return c.remap.flags; })
      .def_property_readonly(
          "relax_enabled",
          [](const soma::Channel& c) { return c.relax.enabled; })
      .def_property_readonly(
          "relax_omega", [](const soma::Channel& c) { return c.relax.omega; })
      .def_property_readonly(
          "spacetime_basis",
          [](const soma::Channel& c) { return c.spacetime.basis; })
      .def_property_readonly(
          "spacetime_degree",
          [](const soma::Channel& c) { return c.spacetime.degree; })
      .def("__repr__", [](const soma::Channel& c) {
        return "Channel('" + c.name + "')";
      });

  py::class_<soma::IterationRegionDecl>(m, "IterationRegionDecl")
      .def(py::init([](std::string name, std::vector<soma::ChannelId> channels,
                       int max_iterations) {
             return soma::IterationRegionDecl{
                 std::move(name), std::move(channels), max_iterations};
           }),
           py::arg("name"), py::arg("channels"), py::arg("max_iterations") = 1)
      .def_readwrite("name", &soma::IterationRegionDecl::name)
      .def_readwrite("channels", &soma::IterationRegionDecl::channels)
      .def_readwrite("max_iterations",
                     &soma::IterationRegionDecl::max_iterations);

  // --- diagnostics ---------------------------------------------------------
  py::class_<soma::Diagnostic>(m, "Diagnostic")
      .def_readonly("severity", &soma::Diagnostic::severity)
      .def_readonly("code", &soma::Diagnostic::code)
      .def_readonly("message", &soma::Diagnostic::message)
      .def_readonly("context", &soma::Diagnostic::context)
      .def_property_readonly("is_error",
                             [](const soma::Diagnostic& d) {
                               return d.severity == soma::Severity::Error;
                             })
      .def("__repr__", [](const soma::Diagnostic& d) {
        return std::string(d.severity == soma::Severity::Error ? "error "
                                                               : "warning ") +
               d.code + ": " + d.message;
      });

  py::class_<soma::Diagnostics>(m, "Diagnostics")
      .def(py::init<>())
      .def_property_readonly("has_errors", &soma::Diagnostics::has_errors)
      .def_property_readonly("items", &soma::Diagnostics::items)
      .def("to_string", &soma::Diagnostics::to_string)
      .def("__len__", &soma::Diagnostics::size)
      .def("__bool__", [](const soma::Diagnostics& d) { return !d.empty(); })
      .def(
          "__iter__",
          [](const soma::Diagnostics& d) {
            return py::make_iterator(d.items().begin(), d.items().end());
          },
          py::keep_alive<0, 1>())
      .def("__str__", &soma::Diagnostics::to_string);

  // --- ChannelBuilder ------------------------------------------------------
  py::class_<soma::ChannelBuilder>(m, "ChannelBuilder")
      .def(
          "accumulate",
          [](soma::ChannelBuilder& b, soma::AccumOp op,
             const soma::Window& w) -> soma::ChannelBuilder& {
            return b.accumulate(op, w);
          },
          py::arg("op"), py::arg("window") = soma::Window::coupling_period(),
          py::return_value_policy::reference_internal)
      .def("remap", &soma::ChannelBuilder::remap, py::arg("flags"),
           py::return_value_policy::reference_internal)
      .def("remap_method", &soma::ChannelBuilder::remap_method,
           py::arg("method"), py::return_value_policy::reference_internal)
      .def("conservative", &soma::ChannelBuilder::conservative,
           py::return_value_policy::reference_internal)
      .def(
          "spacetime",
          [](soma::ChannelBuilder& b, soma::TemporalBasis basis, int degree,
             const soma::Window& w) -> soma::ChannelBuilder& {
            return b.spacetime(basis, degree, w);
          },
          py::arg("basis"), py::arg("degree"),
          py::arg("window") = soma::Window::coupling_period(),
          py::return_value_policy::reference_internal)
      .def("relax", &soma::ChannelBuilder::relax, py::arg("omega"),
           py::return_value_policy::reference_internal)
      .def("extrapolate", &soma::ChannelBuilder::extrapolate, py::arg("order"),
           py::return_value_policy::reference_internal)
      .def("lag", &soma::ChannelBuilder::lag, py::arg("steps"),
           py::return_value_policy::reference_internal)
      .def("allow_time_level_mismatch",
           &soma::ChannelBuilder::allow_time_level_mismatch, py::arg("reason"),
           py::return_value_policy::reference_internal);

  // --- CouplingGraph -------------------------------------------------------
  py::class_<soma::CouplingGraph>(
      m, "CouplingGraph",
      "Spec 2 layer 1: the authored representation. Pure data. The Python "
      "front-end builds exactly the object C++ and YAML build, and emits the "
      "same soma_graph_v1 descriptor (spec 3.2).")
      .def(py::init<>())
      .def(
          "add_mesh",
          [](soma::CouplingGraph& g, std::string name, std::string source) {
            return g.add_mesh(
                soma::MeshDecl{std::move(name), std::move(source)});
          },
          py::arg("name"), py::arg("source"))
      .def(
          "add_region",
          [](soma::CouplingGraph& g, std::string name, soma::MeshId mesh,
             std::string entity_set, int dimension) {
            return g.add_region(soma::Region{std::move(name), mesh,
                                             std::move(entity_set), dimension});
          },
          py::arg("name"), py::arg("mesh"), py::arg("entity_set") = "",
          py::arg("dimension") = -1)
      .def(
          "add_component",
          [](soma::CouplingGraph& g, std::string name, int first_rank,
             int rank_count, py::handle interval) {
            soma::ComponentDecl d;
            d.name = std::move(name);
            d.ranks = soma::RankSet{first_rank, rank_count};
            d.interval = coerce_rational(interval, "add_component(interval=)");
            return g.add_component(std::move(d));
          },
          py::arg("name"), py::arg("first_rank") = 0, py::arg("rank_count") = 1,
          py::arg("interval") = 1,
          "interval accepts an int, a '1/3' string, a fractions.Fraction or a "
          "soma.Rational. A float is refused: spec 4.3 requires exact times.")
      .def(
          "add_field",
          [](soma::CouplingGraph& g, soma::ComponentId owner, std::string name,
             soma::RegionId region, soma::Discretization disc, int order,
             int components, const soma::Units& units,
             soma::Conservation conservation, soma::Intent intent) {
            if (!owner.valid() || owner.value() >= g.component_count()) {
              raise_invalid("add_field: owner component id " +
                            std::to_string(owner.value()) +
                            " is not in this graph");
            }
            soma::FieldSpec f;
            f.name = std::move(name);
            f.region = region;
            f.disc = disc;
            f.order = order;
            f.components = components;
            f.units = units;
            f.conservation = conservation;
            f.intent = intent;
            return g.add_field(owner, std::move(f));
          },
          py::arg("owner"), py::arg("name"), py::arg("region"),
          py::arg("disc") = soma::Discretization::FV, py::arg("order") = 0,
          py::arg("components") = 1,
          py::arg("units") = soma::units::kDimensionless,
          py::arg("conservation") = soma::Conservation::None,
          py::arg("intent") = soma::Intent::Provides)
      .def("add_channel", &soma::CouplingGraph::add_channel, py::arg("name"),
           py::arg("source"), py::arg("target"))
      .def("add_iteration_region", &soma::CouplingGraph::add_iteration_region,
           py::arg("decl"))
      .def("channel", &soma::CouplingGraph::channel, py::arg("id"),
           py::keep_alive<0, 1>(),
           "A fluent ChannelBuilder for the channel. It re-resolves the "
           "channel on every call, so it survives later add_channel() calls.")
      .def("mesh", &soma::CouplingGraph::mesh, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("region", &soma::CouplingGraph::region, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("component", &soma::CouplingGraph::component, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("field", &soma::CouplingGraph::field, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("channel_at", &soma::CouplingGraph::channel_at, py::arg("id"),
           py::return_value_policy::reference_internal)
      .def("iteration_region", &soma::CouplingGraph::iteration_region,
           py::arg("id"), py::return_value_policy::reference_internal)
      .def_property_readonly("mesh_count", &soma::CouplingGraph::mesh_count)
      .def_property_readonly("region_count", &soma::CouplingGraph::region_count)
      .def_property_readonly("component_count",
                             &soma::CouplingGraph::component_count)
      .def_property_readonly("field_count", &soma::CouplingGraph::field_count)
      .def_property_readonly("channel_count",
                             &soma::CouplingGraph::channel_count)
      .def_property_readonly("iteration_region_count",
                             &soma::CouplingGraph::iteration_region_count)
      .def("owner_of", &soma::CouplingGraph::owner_of, py::arg("field"))
      .def(
          "find_component",
          [](const soma::CouplingGraph& g, const std::string& name) {
            const auto id = g.find_component(name);
            if (!id.has_value()) {
              raise_notfound("find_component: no component named '" + name +
                             "'");
            }
            return *id;
          },
          py::arg("name"))
      .def(
          "find_field",
          [](const soma::CouplingGraph& g, soma::ComponentId owner,
             const std::string& name) {
            if (!owner.valid() || owner.value() >= g.component_count()) {
              raise_invalid("find_field: component id " +
                            std::to_string(owner.value()) +
                            " is not in this graph");
            }
            const auto id = g.find_field(owner, name);
            if (!id.has_value()) {
              raise_notfound("find_field: component '" +
                             g.component(owner).name +
                             "' has no field named '" + name + "'");
            }
            return *id;
          },
          py::arg("owner"), py::arg("name"))
      .def("to_yaml",
           [](const soma::CouplingGraph& g) {
             const auto r = soma::to_yaml(g);
             if (!r.ok()) raise_core_error("to_yaml", r.error());
             return r.value();
           })
      .def(
          "to_descriptor",
          [](const soma::CouplingGraph& g) {
            const std::vector<std::byte> bytes = soma::serialize(g);
            return py::bytes(reinterpret_cast<const char*>(bytes.data()),
                             bytes.size());
          },
          "The flat soma_graph_v1 descriptor of spec 3.2 -- byte-identical to "
          "what the C++ and YAML front-ends produce for the same graph.")
      .def("__copy__",
           [](const soma::CouplingGraph& g) { return soma::CouplingGraph(g); })
      .def(
          "__deepcopy__",
          [](const soma::CouplingGraph& g, py::dict) {
            return soma::CouplingGraph(g);
          },
          py::arg("memo"))
      .def("__repr__", [](const soma::CouplingGraph& g) {
        return "CouplingGraph(components=" +
               std::to_string(g.component_count()) +
               ", fields=" + std::to_string(g.field_count()) +
               ", channels=" + std::to_string(g.channel_count()) + ")";
      });

  m.def("validate", &soma::validate, py::arg("graph"),
        "Every structural problem at once (spec Global Constraints), never "
        "just the first.");
  m.def("structurally_equal", &soma::structurally_equal, py::arg("a"),
        py::arg("b"),
        "True when the two graphs serialize to identical descriptors.");
  m.def(
      "from_descriptor",
      [](const py::bytes& blob) {
        const std::string      raw = blob.cast<std::string>();
        std::vector<std::byte> bytes(raw.size());
        if (!raw.empty()) std::memcpy(bytes.data(), raw.data(), raw.size());
        auto r = soma::deserialize(bytes);
        if (!r.ok()) {
          raise_invalid("from_descriptor: " + r.error().code + ": " +
                        r.error().message);
        }
        return std::move(r.value());
      },
      py::arg("descriptor"));
  m.def(
      "from_yaml",
      [](const std::string& text) {
        auto r = soma::from_yaml(text);
        if (!r.ok()) {
          raise_invalid("from_yaml: " + r.error().code + ": " +
                        r.error().message);
        }
        return std::move(r.value());
      },
      py::arg("text"));
  m.def(
      "load_yaml",
      [](const std::string& path) {
        auto r = soma::load_yaml_file(path);
        if (!r.ok()) {
          raise_invalid("load_yaml: " + r.error().code + ": " +
                        r.error().message);
        }
        return std::move(r.value());
      },
      py::arg("path"));

  // --- IR ------------------------------------------------------------------
  py::class_<soma::OperatorDAG>(m, "OperatorDAG",
                                "Spec 2 layer 2: the lowered operator DAG.")
      .def_property_readonly("node_count", &soma::OperatorDAG::node_count)
      .def_property_readonly("arc_count", &soma::OperatorDAG::arc_count)
      .def_property_readonly("region_count", &soma::OperatorDAG::region_count)
      .def("__repr__", [](const soma::OperatorDAG& d) {
        return "OperatorDAG(nodes=" + std::to_string(d.node_count()) +
               ", arcs=" + std::to_string(d.arc_count()) + ")";
      });

  py::class_<soma::LoweringResult>(m, "LoweringResult")
      .def_readonly("dag", &soma::LoweringResult::dag)
      .def_property_readonly(
          "diagnostics", [](const soma::LoweringResult& r) { return r.diags; })
      .def_property_readonly("ok", &soma::LoweringResult::ok);

  m.def(
      "lower",
      [](const soma::CouplingGraph& g) {
        py::gil_scoped_release release;  // pure C++, no Python reachable
        return soma::lower(g);
      },
      py::arg("graph"), "Spec 2: CouplingGraph -> OperatorDAG.");
  m.def("dump_ir", &soma::dump_text, py::arg("dag"),
        "The spec 10 level-2 textual IR dump, byte-for-byte the artifact "
        "test/golden/*.ir is compared against.");
  m.def("dump_graphviz", &soma::dump_graphviz, py::arg("dag"));

  // --- schedule ------------------------------------------------------------
  py::class_<soma::ExecutionPlan>(m, "ExecutionPlan",
                                  "Spec 2 layer 3: the flat instruction "
                                  "stream, immutable after construction.")
      .def_property_readonly(
          "instruction_count",
          [](const soma::ExecutionPlan& p) { return p.instructions().size(); })
      .def_property_readonly("period", &soma::ExecutionPlan::period)
      .def_property_readonly("lockstep", &soma::ExecutionPlan::lockstep)
      .def_property_readonly("region_count", &soma::ExecutionPlan::region_count)
      .def("to_string", &soma::ExecutionPlan::to_string);

  py::class_<soma::ScheduleResult>(m, "ScheduleResult")
      .def_readonly("plan", &soma::ScheduleResult::plan)
      .def_property_readonly(
          "diagnostics", [](const soma::ScheduleResult& r) { return r.diags; })
      .def_property_readonly("ok", &soma::ScheduleResult::ok);

  py::class_<soma::SerialMeshBackend>(
      m, "SerialMeshBackend",
      "The dependency-free structured backend: 'generated:line:N' and "
      "'generated:box:NxMxP'.")
      .def(py::init<>())
      .def(
          "load",
          [](soma::SerialMeshBackend& b, const soma::MeshDecl& d) {
            const auto r = b.load(d);
            if (!r.ok()) raise_core_error("SerialMeshBackend.load", r.error());
          },
          py::arg("mesh"))
      .def(
          "load_graph_meshes",
          [](soma::SerialMeshBackend& b, const soma::CouplingGraph& g) {
            // Declaration order, so the backend's MeshIds match the graph's.
            for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
              const auto r = b.load(g.mesh(soma::MeshId{i}));
              if (!r.ok()) {
                raise_core_error("SerialMeshBackend.load_graph_meshes",
                                 r.error());
              }
            }
          },
          py::arg("graph"))
      .def(
          "region_size",
          [](const soma::SerialMeshBackend& b, const soma::Region& r) {
            const auto s = b.region_size(r);
            if (!s.ok()) {
              raise_core_error("SerialMeshBackend.region_size", s.error());
            }
            return s.value();
          },
          py::arg("region"))
      .def("name", &soma::SerialMeshBackend::name);

  m.def(
      "schedule",
      [](const soma::CouplingGraph& g, const soma::OperatorDAG& dag,
         const soma::SerialMeshBackend& backend, soma::ExpandMode mode) {
        py::gil_scoped_release release;  // pure C++, no Python reachable
        const soma::ClockSet   clocks{g};
        return soma::schedule(g, dag, clocks,
                              soma::make_region_sizes(g, backend), mode);
      },
      py::arg("graph"), py::arg("dag"), py::arg("backend"),
      py::arg("mode") = soma::ExpandMode::Auto,
      "expand -> prove_acyclic -> plan_buffers -> emit. The clock set is "
      "derived from the graph's intervals.");
  m.def(
      "dump_plan", [](const soma::ExecutionPlan& p) { return p.to_string(); },
      py::arg("plan"), "The --soma-dump-plan artifact (spec 2.3).");

  // --- field registry ------------------------------------------------------
  py::class_<PyFieldRegistry>(
      m, "FieldRegistry",
      "Maps FieldId to component-owned storage. bind_numpy() records a pointer "
      "into the caller's array and a strong reference to it; SOMA never owns, "
      "copies or frees field data.")
      .def(py::init<>())
      .def("bind_numpy", &PyFieldRegistry::bind_numpy, py::arg("field"),
           py::arg("array"),
           "Bind a C-contiguous float64 numpy array. 1-D is (count,) with one "
           "component; 2-D is (count, components), interleaved. A strided "
           "view, a transpose, a non-float64 dtype or a read-only array is "
           "refused -- each of those would silently be the wrong memory.")
      .def("get_numpy", &PyFieldRegistry::get_numpy, py::arg("field"),
           "A numpy view that ALIASES the bound storage. Writing through it "
           "writes the component's array.")
      .def("is_bound", &PyFieldRegistry::is_bound, py::arg("field"))
      .def_property_readonly("bound_count", &PyFieldRegistry::bound_count);

  // --- components ----------------------------------------------------------
  py::class_<soma::StepContext>(m, "StepContext",
                                "The half-open interval a component must "
                                "integrate over, in exact rational time.")
      .def_readonly("t_begin", &soma::StepContext::t_begin)
      .def_readonly("t_end", &soma::StepContext::t_end)
      .def_readonly("step", &soma::StepContext::step)
      .def_readonly("iteration", &soma::StepContext::iteration)
      .def_property_readonly("dt", &soma::StepContext::dt)
      .def("valid", &soma::StepContext::valid)
      .def("__repr__", [](const soma::StepContext& c) {
        return "StepContext(t_begin=" + c.t_begin.to_string() +
               ", t_end=" + c.t_end.to_string() +
               ", step=" + std::to_string(c.step) +
               ", iteration=" + std::to_string(c.iteration) + ")";
      });

  py::class_<PyComponent>(
      m, "PyComponent",
      "Base class for an in-process Python peer component (spec 6). Subclass "
      "it, call super().__init__(name) with the name the graph declares, and "
      "override register_fields(self, registry, graph) and advance(self, ctx). "
      "Optional save_state()/restore_state() opt the component into rollback "
      "inside an iteration region.")
      .def(py::init<std::string>(), py::arg("name"))
      .def_property_readonly("name", &PyComponent::name)
      .def(
          "register_fields",
          [](PyComponent& self, py::object, py::object) {
            throw py::type_error(
                "soma.PyComponent subclass '" + self.name() +
                "' does not override register_fields(self, registry, graph)");
          },
          py::arg("registry"), py::arg("graph"))
      .def(
          "advance",
          [](PyComponent& self, py::object) {
            throw py::type_error("soma.PyComponent subclass '" + self.name() +
                                 "' does not override advance(self, ctx)");
          },
          py::arg("ctx"))
      .def("__repr__", [](const PyComponent& c) {
        return "PyComponent('" + c.name() + "')";
      });

  py::class_<soma::RunnerConfig>(m, "RunnerConfig")
      .def(py::init([](std::string transport, bool check_conservation,
                       double tolerance, bool abort_on_violation) {
             return soma::RunnerConfig{std::move(transport), check_conservation,
                                       tolerance, abort_on_violation};
           }),
           py::arg("transport") = "copy", py::arg("check_conservation") = true,
           py::arg("conservation_tolerance") = 1e-12,
           py::arg("abort_on_violation") = false)
      .def_readwrite("transport", &soma::RunnerConfig::transport)
      .def_readwrite("check_conservation",
                     &soma::RunnerConfig::check_conservation)
      .def_readwrite("conservation_tolerance",
                     &soma::RunnerConfig::conservation_tolerance)
      .def_readwrite("abort_on_violation",
                     &soma::RunnerConfig::abort_on_violation);

  py::class_<soma::RunStats>(m, "RunStats")
      .def_readonly("periods", &soma::RunStats::periods)
      .def_readonly("instructions_executed",
                    &soma::RunStats::instructions_executed)
      .def_readonly("conservation_checks", &soma::RunStats::conservation_checks)
      .def_readonly("conservation_violations",
                    &soma::RunStats::conservation_violations)
      .def_readonly("worst_conservation_error",
                    &soma::RunStats::worst_conservation_error)
      .def("__repr__", [](const soma::RunStats& s) {
        return "RunStats(periods=" + std::to_string(s.periods) +
               ", instructions_executed=" +
               std::to_string(s.instructions_executed) + ")";
      });

  py::class_<Session>(
      m, "Session",
      "Drives a coupled run of Python peer components through soma::Runner. "
      "Owns its own copy of the graph, because the Runner holds a reference to "
      "it for the whole run.")
      .def(py::init<const soma::CouplingGraph&>(), py::arg("graph"))
      .def("add_component", &Session::add_component, py::arg("component"))
      .def("initialize", &Session::initialize,
           py::arg("config") = soma::RunnerConfig{})
      .def("run_period", &Session::run_period)
      .def("run", &Session::run, py::arg("periods"))
      .def_property_readonly("stats", &Session::stats,
                             py::return_value_policy::copy)
      .def_property_readonly("graph", &Session::graph_object)
      .def_property_readonly("registry", &Session::registry_object)
      .def_property_readonly("dag", &Session::dag,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("plan", &Session::plan,
                             py::return_value_policy::reference_internal);

  // --- the C ABI -----------------------------------------------------------
  py::class_<Context>(
      m, "Context",
      "One soma_ctx from include/soma/soma_c.h -- the narrow waist of spec "
      "3.2. Every method is one ABI entry point, and every non-zero return "
      "code becomes an exception carrying soma_last_error()'s message.")
      .def(py::init<>())
      .def("close", &Context::close,
           "soma_destroy(). Idempotent from Python's point of view: a second "
           "close() is a no-op, not the ABI's double-destroy.")
      .def_property_readonly("closed", &Context::closed)
      .def("__enter__", [](py::object self) { return self; })
      .def("__exit__",
           [](Context& c, const py::object&, const py::object&,
              const py::object&) {
             c.close();
             return false;
           })
      .def("last_error", &Context::last_error,
           "soma_last_error(), copied into a Python str.")
      .def("load_graph", &Context::load_graph, py::arg("source"),
           "soma_load_graph() from a soma.CouplingGraph or a bytes "
           "descriptor.")
      .def("load_yaml", &Context::load_yaml, py::arg("path"),
           "soma_load_graph_yaml().")
      .def("save_graph", &Context::save_graph,
           "soma_save_graph(), two-call idiom, as bytes.")
      .def("field_handle", &Context::field_handle, py::arg("component"),
           py::arg("field"), "soma_field_handle().")
      .def("field_components", &Context::field_components, py::arg("handle"),
           "The component count the graph declares for this field. Answered "
           "from the graph this context loaded, NOT by parsing an error "
           "message: the C ABI has no soma_field_components().")
      .def("field_name", &Context::field_name, py::arg("handle"))
      .def("register_field", &Context::register_field, py::arg("handle"),
           py::arg("array"), py::arg("memspace") = SOMA_MEMSPACE_HOST,
           "soma_register_field(). The array must be C-contiguous float64 and "
           "must stay alive; this context holds a reference to it until it is "
           "closed.")
      .def("set_communicator", &Context::set_communicator,
           py::arg("fortran_comm"), "soma_set_communicator().")
      .def("set_end_time", &Context::set_end_time, py::arg("time"),
           "soma_set_end_time(). Takes an exact rational -- int, '1/3', "
           "fractions.Fraction or soma.Rational -- never a float.")
      .def("initialize", &Context::initialize, "soma_initialize().")
      .def("begin_step", &Context::begin_step, py::arg("component"),
           "soma_begin_step(); returns (t_begin, t_end).")
      .def("end_step", &Context::end_step, py::arg("component"),
           "soma_end_step().")
      .def("requires_checkpoint", &Context::requires_checkpoint,
           py::arg("component"), "soma_requires_checkpoint().")
      .def("requires_restore", &Context::requires_restore, py::arg("component"),
           "soma_requires_restore().")
      .def("is_coupling_ongoing", &Context::is_coupling_ongoing,
           "soma_is_coupling_ongoing().")
      .def("run_period", &Context::run_period,
           "soma_run_period(). Reserved by the ABI: raises soma.Unsupported, "
           "because a C ABI caller IS the component and drives with "
           "begin_step()/end_step(). Use soma.Session for Python peers.")
      .def("run", &Context::run, py::arg("periods"),
           "soma_run(). Reserved; see run_period().");
}
