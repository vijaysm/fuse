// fuse/base/{ids,rational,units,result}.hpp -- the Id<Tag> family, Rational, Units,
// Severity/Diagnostic/Diagnostics. See docs/dev/python-surface.md's evidence table: every one
// of these carries over from upstream soma almost verbatim.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "errors.hpp"
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/base/units.hpp"

namespace py = pybind11;

namespace pfuse_py {

/// Binds one `fuse::Id<Tag>` specialisation under `name`. Transplanted from upstream's own
/// generic-Id binding (bindings.cpp): equality, ordering, a constant-by-value constructor and
/// `__hash__`/`__repr__`, nothing else -- an Id is an opaque handle, not a number to do
/// arithmetic on.
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
        return std::string(name) + "(" + std::to_string(id.value()) + ")";
      });
}

/// Accepts fuse.Rational, int, a "n/d" str, and anything exposing integral
/// .numerator/.denominator (fractions.Fraction). float is deliberately refused: spec 4.3
/// requires exact coupling times, and a binary double cannot represent 1/3 or 1/10 -- silently
/// taking the nearest double is exactly the drift the rational clock exists to prevent.
/// Transplanted from upstream's coerce_rational() verbatim (same rule, same message shape).
fuse::Rational coerce_rational(py::handle value, const char* what) {
  // FUSE_THROW, not a hand-constructed fuse::InvalidArgument: the macro runs fuse's own
  // on-error policy (capture a backtrace if the policy wants one, print/attach a debugger/
  // abort per FUSE_ON_ERROR) before throwing -- the SAME path every other fuse::Error in the
  // library goes through. Raising directly would silently skip that policy for exactly this
  // one call site.
  if (value.is_none()) {
    FUSE_THROW(fuse::InvalidArgument, "{}: expected a rational, got None", what);
  }
  if (py::isinstance<fuse::Rational>(value)) {
    return value.cast<fuse::Rational>();
  }
  if (py::isinstance<py::bool_>(value)) {
    FUSE_THROW(fuse::InvalidArgument, "{}: a bool is not a coupling interval", what);
  }
  if (py::isinstance<py::float_>(value)) {
    FUSE_THROW(fuse::InvalidArgument,
              "{}: a float is not accepted, because spec 4.3 requires exact coupling times "
              "and a binary double cannot represent 1/3 or 1/10. Pass an int, a string such "
              "as '1/3', or a fractions.Fraction.",
              what);
  }
  if (py::isinstance<py::int_>(value)) {
    return fuse::Rational{value.cast<std::int64_t>()};
  }
  if (py::isinstance<py::str>(value)) {
    const std::string text = value.cast<std::string>();
    const auto slash = text.find('/');
    try {
      if (slash == std::string::npos) {
        return fuse::Rational{static_cast<std::int64_t>(std::stoll(text))};
      }
      const std::int64_t n = std::stoll(text.substr(0, slash));
      const std::int64_t d = std::stoll(text.substr(slash + 1));
      return fuse::Rational{n, d};
    } catch (const std::exception& e) {
      FUSE_THROW(fuse::InvalidArgument, "{}: cannot read '{}' as a rational: {}", what, text,
                e.what());
    }
  }
  if (py::hasattr(value, "numerator") && py::hasattr(value, "denominator")) {
    const auto n = value.attr("numerator").cast<std::int64_t>();
    const auto d = value.attr("denominator").cast<std::int64_t>();
    return fuse::Rational{n, d};
  }
  FUSE_THROW(fuse::InvalidArgument, "{}: expected fuse.Rational, int, str or fractions.Fraction, got {}",
            what, py::str(py::type::of(value).attr("__name__")).cast<std::string>());
}

void bind_base(py::module_& m) {
  bind_id<fuse::ComponentId>(m, "ComponentId", "Strong-typed index into a graph's components.");
  bind_id<fuse::RegionId>(m, "RegionId", "Strong-typed index into a graph's regions.");
  bind_id<fuse::FieldId>(m, "FieldId", "Strong-typed index into a graph's fields.");
  bind_id<fuse::ChannelId>(m, "ChannelId", "Strong-typed index into a graph's channels.");
  bind_id<fuse::MeshId>(m, "MeshId", "Strong-typed index into a graph's meshes.");
  bind_id<fuse::RegionIdx>(m, "IterationRegionIdx",
                          "Strong-typed index into a graph's implicit coupling groups.");

  py::enum_<fuse::Discretization>(m, "Discretization")
      .value("FV", fuse::Discretization::FV)
      .value("ContinuousGLL", fuse::Discretization::ContinuousGLL)
      .value("DiscontinuousGLL", fuse::Discretization::DiscontinuousGLL)
      .value("PointCloud", fuse::Discretization::PointCloud);
  py::enum_<fuse::Conservation>(m, "Conservation")
      .value("None_", fuse::Conservation::None)  // "None" is a Python keyword
      .value("Extensive", fuse::Conservation::Extensive)
      .value("Intensive", fuse::Conservation::Intensive);
  py::enum_<fuse::Intent>(m, "Intent")
      .value("Provides", fuse::Intent::Provides)
      .value("Requires", fuse::Intent::Requires)
      .value("Bidirectional", fuse::Intent::Bidirectional);
  py::enum_<fuse::AccumOp>(m, "AccumOp")
      .value("Instant", fuse::AccumOp::Instant)
      .value("Sum", fuse::AccumOp::Sum)
      .value("Mean", fuse::AccumOp::Mean)
      .value("Min", fuse::AccumOp::Min)
      .value("Max", fuse::AccumOp::Max)
      .value("Integral", fuse::AccumOp::Integral);
  py::enum_<fuse::TemporalBasis>(m, "TemporalBasis")
      .value("None_", fuse::TemporalBasis::None)
      .value("Constant", fuse::TemporalBasis::Constant)
      .value("Linear", fuse::TemporalBasis::Linear)
      .value("DG", fuse::TemporalBasis::DG)
      .value("Spline", fuse::TemporalBasis::Spline);

  py::class_<fuse::Rational>(m, "Rational",
      "Exact rational time (spec 4.3): numerator/denominator, no float constructor.")
      .def(py::init<>())
      .def(py::init<std::int64_t>(), py::arg("numerator"))
      .def(py::init<std::int64_t, std::int64_t>(), py::arg("numerator"), py::arg("denominator"))
      .def_property_readonly("num", &fuse::Rational::num)
      .def_property_readonly("den", &fuse::Rational::den)
      .def("is_zero", &fuse::Rational::is_zero)
      .def("is_integer", &fuse::Rational::is_integer)
      .def("to_double", &fuse::Rational::to_double)
      .def("to_string", &fuse::Rational::to_string)
      .def("__float__", &fuse::Rational::to_double)
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
          [](const fuse::Rational& r) {
            return std::hash<std::int64_t>{}(r.num()) ^ (std::hash<std::int64_t>{}(r.den()) << 1);
          })
      .def("__str__", &fuse::Rational::to_string)
      .def("__repr__", [](const fuse::Rational& r) {
        return "Rational(" + std::to_string(r.num()) + ", " + std::to_string(r.den()) + ")";
      });

  m.def("rational", [](py::handle v) { return coerce_rational(v, "rational()"); },
       py::arg("value"),
       "Coerce int/str/fractions.Fraction/Rational to a fuse.Rational. Rejects float (spec 4.3).");

  // FUSE_VERSION_STRING (include/fuse/base/config.hpp, generated from CMakeLists.txt's own
  // PROJECT_VERSION by cmake/FuseDependencies.cmake) -- the project's semantic version, not
  // a separate ABI version counter: fuse has no soma_abi_version()-shaped integer binding
  // this mirrors, so abi_version() is not bound (unlike upstream's soma.abi_version()).
  m.def("version", []() { return std::string(FUSE_VERSION_STRING); },
       "The FUSE_VERSION_STRING this extension was built against.");

  py::class_<fuse::Units>(m, "Units", "SI dimension exponents plus an exact affine scale/offset.")
      .def(py::init<>())
      .def(py::init([](int length, int mass, int time, int current, int temperature,
                       int amount, int luminosity, py::object scale, py::object offset) {
             fuse::Units u{};
             u.length = static_cast<std::int8_t>(length);
             u.mass = static_cast<std::int8_t>(mass);
             u.time = static_cast<std::int8_t>(time);
             u.current = static_cast<std::int8_t>(current);
             u.temperature = static_cast<std::int8_t>(temperature);
             u.amount = static_cast<std::int8_t>(amount);
             u.luminosity = static_cast<std::int8_t>(luminosity);
             u.scale = scale.is_none() ? fuse::Rational{1} : coerce_rational(scale, "Units.scale");
             u.offset = offset.is_none() ? fuse::Rational{0}
                                         : coerce_rational(offset, "Units.offset");
             return u;
           }),
           py::arg("length") = 0, py::arg("mass") = 0, py::arg("time") = 0,
           py::arg("current") = 0, py::arg("temperature") = 0, py::arg("amount") = 0,
           py::arg("luminosity") = 0, py::arg("scale") = py::none(),
           py::arg("offset") = py::none())
      .def_readwrite("length", &fuse::Units::length)
      .def_readwrite("mass", &fuse::Units::mass)
      .def_readwrite("time", &fuse::Units::time)
      .def_readwrite("current", &fuse::Units::current)
      .def_readwrite("temperature", &fuse::Units::temperature)
      .def_readwrite("amount", &fuse::Units::amount)
      .def_readwrite("luminosity", &fuse::Units::luminosity)
      .def_readwrite("scale", &fuse::Units::scale)
      .def_readwrite("offset", &fuse::Units::offset)
      .def("is_dimensionless", &fuse::Units::is_dimensionless)
      .def("is_linear", &fuse::Units::is_linear)
      .def("same_dimensions", &fuse::Units::same_dimensions, py::arg("other"))
      .def("to_string", &fuse::Units::to_string)
      .def(py::self * py::self)
      .def(py::self / py::self)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", &fuse::Units::to_string)
      .def("__repr__", [](const fuse::Units& u) { return "Units(" + u.to_string() + ")"; });

  py::module_ units = m.def_submodule("units", "Named fuse::units:: constants.");
  units.attr("dimensionless") = fuse::units::kDimensionless;
  units.attr("K") = fuse::units::kKelvin;
  units.attr("metre") = fuse::units::kMetre;
  units.attr("kilogram") = fuse::units::kKilogram;
  units.attr("second") = fuse::units::kSecond;
  units.attr("newton") = fuse::units::kNewton;
  units.attr("joule") = fuse::units::kJoule;
  units.attr("watt") = fuse::units::kWatt;
  units.attr("pascal") = fuse::units::kPascal;
  units.attr("W_per_m2") = fuse::units::kHeatFluxDensity;

  py::enum_<fuse::Severity>(m, "Severity")
      .value("Warning", fuse::Severity::Warning)
      .value("Error", fuse::Severity::Error);

  py::class_<fuse::Diagnostic>(m, "Diagnostic")
      .def_readonly("severity", &fuse::Diagnostic::severity)
      .def_readonly("code", &fuse::Diagnostic::code)
      .def_readonly("message", &fuse::Diagnostic::message)
      .def_readonly("context", &fuse::Diagnostic::context)
      .def("__repr__", [](const fuse::Diagnostic& d) {
        return "Diagnostic(" + d.code + ": " + d.message + ")";
      });

  py::class_<fuse::Diagnostics>(m, "Diagnostics")
      .def(py::init<>())
      .def_property_readonly("has_errors", &fuse::Diagnostics::has_errors)
      .def_property_readonly("empty", &fuse::Diagnostics::empty)
      .def_property_readonly("items", &fuse::Diagnostics::items)
      .def("to_string", &fuse::Diagnostics::to_string)
      .def("__len__", &fuse::Diagnostics::size)
      .def("__bool__", [](const fuse::Diagnostics& d) { return !d.empty(); })
      .def("__iter__",
          [](const fuse::Diagnostics& d) { return py::make_iterator(d.items()); },
          py::keep_alive<0, 1>())
      .def("__str__", &fuse::Diagnostics::to_string);
}

}  // namespace pfuse_py
