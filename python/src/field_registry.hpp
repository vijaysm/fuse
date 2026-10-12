#pragma once

// The numpy-aware fuse::FieldRegistry wrapper, as a header: field_bindings.cpp binds it as
// fuse.FieldRegistry, and component_bindings.cpp's ForwardingComponent constructs one to pass
// to a Python Component's register_fields(). See docs/dev/python-surface.md, "The numpy
// buffer contract and the lifetime rule."

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>

#include "errors.hpp"
#include "fuse/base/error.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace py = pybind11;

namespace pfuse_py {

/// What a validated array tells us, in fuse::Layout's own vocabulary.
struct ArrayView {
  py::object owner;  // strong reference: the lifetime guarantee
  double* data = nullptr;
  std::size_t count = 0;
  std::size_t stride = 1;
  std::size_t component_stride = 0;
  int components = 1;
};

/// Every rejection here raises fuse::InvalidArgument via FUSE_THROW -- the same macro the
/// library itself uses, so these binding-level precondition checks are translated through the
/// SAME exception-translator path as any other fuse::InvalidArgument (module.cpp's own header
/// comment). Transplanted from upstream's inspect_array(): same five checks, same reasons.
inline ArrayView inspect_array(py::handle obj, const std::string& what) {
  if (!py::isinstance<py::array>(obj)) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: expected a numpy.ndarray, got {}. fuse binds the component's own storage; "
              "a list or a sequence would have to be copied, and a copy is not the "
              "component's storage.",
              what, py::str(py::type::of(obj).attr("__name__")).cast<std::string>());
  }
  py::array arr = py::reinterpret_borrow<py::array>(obj);

  if (!arr.dtype().equal(py::dtype::of<double>())) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: dtype must be float64, got {}. fuse fields are double; casting would "
              "silently copy.",
              what, py::str(arr.dtype()).cast<std::string>());
  }

  const py::object flags = arr.attr("flags");
  if (!flags.attr("c_contiguous").cast<bool>()) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: the array is not C-contiguous (shape={}, strides={}). A strided view such "
              "as a[::2] or a transpose does not describe a fuse::Layout run, and binding it "
              "would read the wrong doubles. Pass numpy.ascontiguousarray(a) -- and keep THAT "
              "array alive, since it is a different buffer.",
              what, py::str(arr.attr("shape")).cast<std::string>(),
              py::str(arr.attr("strides")).cast<std::string>());
  }
  if (!flags.attr("writeable").cast<bool>()) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: the array is read-only. fuse writes into a Requires field, so a read-only "
              "buffer cannot be bound.",
              what);
  }

  const auto ndim = static_cast<int>(arr.ndim());
  if (ndim != 1 && ndim != 2) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: expected a 1-D (count,) or 2-D (count, components) array, got {} dimensions",
              what, ndim);
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
      FUSE_THROW(fuse::InvalidArgument,
                "{}: a 2-D array must have at least one component; got shape {}", what,
                py::str(arr.attr("shape")).cast<std::string>());
    }
    if (ncomp > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      FUSE_THROW(fuse::InvalidArgument, "{}: component count does not fit an int", what);
    }
    v.components = static_cast<int>(ncomp);
    v.stride = static_cast<std::size_t>(ncomp);
    v.component_stride = 1;
  }
  v.data = static_cast<double*>(arr.mutable_data());
  if (v.data == nullptr && v.count != 0) {
    FUSE_THROW(fuse::InvalidArgument, "{}: the array has no data pointer", what);
  }
  return v;
}

/// A numpy array that ALIASES `ref`'s storage. `base` tells numpy not to own the memory.
inline py::object numpy_view(const fuse::FieldRef& ref, py::object base) {
  const fuse::Layout& l = ref.layout();
  auto* data = const_cast<double*>(ref.data());
  const auto elem = static_cast<py::ssize_t>(sizeof(double));

  if (base.is_none()) {
    base = py::capsule(data, [](void*) {});
  }

  if (l.components == 1) {
    return py::array_t<double>(std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.count)},
                               std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.stride) * elem},
                               data, base);
  }
  return py::array_t<double>(
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.count),
                               static_cast<py::ssize_t>(l.components)},
      std::vector<py::ssize_t>{static_cast<py::ssize_t>(l.stride) * elem,
                               static_cast<py::ssize_t>(l.effective_component_stride()) * elem},
      data, base);
}

/// Keeps every bound array alive. Shared (not owned outright) because a PyFieldRegistry
/// handed to Component.register_fields is a temporary view onto the session's registry and
/// must not take the references with it when it dies -- upstream's own KeepAlive design,
/// transplanted unchanged.
struct KeepAlive {
  std::unordered_map<std::uint32_t, py::object> refs;
};

/// fuse.FieldRegistry. Either owns a fuse::FieldRegistry (standalone use, e.g. in a test) or
/// points at somebody else's (the session's, during register_fields()).
class PyFieldRegistry {
 public:
  PyFieldRegistry()
      : owned_(std::make_shared<fuse::FieldRegistry>()),
        reg_(owned_.get()),
        keep_(std::make_shared<KeepAlive>()) {}

  PyFieldRegistry(fuse::FieldRegistry* reg, std::shared_ptr<KeepAlive> keep)
      : reg_(reg), keep_(std::move(keep)) {}

  [[nodiscard]] fuse::FieldRegistry& raw() { return *reg_; }
  [[nodiscard]] const fuse::FieldRegistry& raw() const { return *reg_; }

  /// The graph, when one is known, so a shape mismatch is diagnosed HERE -- against the
  /// declared component count -- rather than discovered from error text (same reasoning as
  /// upstream's own set_graph()).
  void set_graph(const fuse::CouplingGraph* g) { graph_ = g; }

  void bind_numpy(fuse::FieldId id, py::handle array) {
    const std::string what = "FieldRegistry.bind_numpy(field " + std::to_string(id.value()) + ")";
    ArrayView v = inspect_array(array, what);

    if (graph_ != nullptr) {
      if (!id.valid() || id.value() >= graph_->field_count()) {
        FUSE_THROW(fuse::InvalidArgument, "{}: field id {} is outside [0, {})", what, id.value(),
                  graph_->field_count());
      }
      const fuse::FieldSpec& spec = graph_->field(id);
      if (spec.components != v.components) {
        FUSE_THROW(fuse::InvalidArgument,
                  "{}: field '{}' is declared with {} component(s) but the array supplies {}. "
                  "A 1-D array is one component per entity; a 2-D (count, components) array "
                  "is C-contiguous interleaved.",
                  what, spec.name, spec.components, v.components);
      }
    }

    const fuse::Layout layout{v.stride, v.component_stride, v.count, v.components};
    const auto r =
        reg_->bind(id, fuse::FieldRef{v.data, layout, fuse::MemorySpace::Host});
    if (!r.ok()) pfuse_py::raise_for(r);
    keep_->refs[id.value()] = std::move(v.owner);
  }

  [[nodiscard]] py::object get_numpy(fuse::FieldId id) const {
    const auto r = reg_->get(id);
    if (!r.ok()) pfuse_py::raise_for(r);
    const auto it = keep_->refs.find(id.value());
    py::object base = it == keep_->refs.end() ? py::object(py::none()) : py::object(it->second);
    return numpy_view(r.value(), std::move(base));
  }

  [[nodiscard]] bool is_bound(fuse::FieldId id) const { return reg_->is_bound(id); }
  [[nodiscard]] std::size_t bound_count() const { return reg_->bound_count(); }
  [[nodiscard]] std::shared_ptr<KeepAlive> keep() const { return keep_; }

 private:
  std::shared_ptr<fuse::FieldRegistry> owned_;
  fuse::FieldRegistry* reg_ = nullptr;
  std::shared_ptr<KeepAlive> keep_;
  const fuse::CouplingGraph* graph_ = nullptr;
};



}  // namespace pfuse_py
