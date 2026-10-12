// fuse.FieldRegistry binding. The implementation (ArrayView/inspect_array/numpy_view/
// KeepAlive/PyFieldRegistry) lives in field_registry.hpp, shared with
// component_bindings.cpp's ForwardingComponent.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "field_registry.hpp"
#include "fuse/field/field_ref.hpp"

namespace py = pybind11;

namespace pfuse_py {

void bind_field(py::module_& m) {
  py::enum_<fuse::MemorySpace>(m, "MemorySpace")
      .value("Host", fuse::MemorySpace::Host)
      .value("Device", fuse::MemorySpace::Device);

  py::class_<PyFieldRegistry>(m, "FieldRegistry",
      "fuse::FieldRegistry, numpy-aware: bind_numpy()/get_numpy() replace fuse::FieldRef's "
      "raw-pointer interface with a zero-copy numpy view under the lifetime rule described in "
      "docs/dev/python-surface.md.")
      .def(py::init<>())
      .def("bind_numpy", &PyFieldRegistry::bind_numpy, py::arg("field"), py::arg("array"))
      .def("get_numpy", &PyFieldRegistry::get_numpy, py::arg("field"))
      .def("is_bound", &PyFieldRegistry::is_bound, py::arg("field"))
      .def_property_readonly("bound_count", &PyFieldRegistry::bound_count);
}

}  // namespace pfuse_py
