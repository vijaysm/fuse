// fuse/model/{region,component,field_spec,channel,coupling_graph,iteration,descriptor,
// yaml_io,validate}.hpp and fuse/ir/{lowering,dump}.hpp, fuse/schedule/{clock,plan}.hpp.
// See docs/dev/python-surface.md's evidence table.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include "errors.hpp"
#include "fuse/base/error.hpp"
#include "fuse/ir/dump.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"
#include "fuse/model/validate.hpp"
#include "fuse/model/yaml_io.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/backend/mesh_view.hpp"

namespace py = pybind11;

namespace pfuse_py {

fuse::Rational coerce_rational(py::handle value, const char* what);  // base_bindings.cpp

void bind_model(py::module_& m) {
  py::class_<fuse::MeshDecl>(m, "MeshDecl")
      .def(py::init([](std::string name, std::string source) {
             return fuse::MeshDecl{std::move(name), std::move(source)};
           }),
           py::arg("name"), py::arg("source"))
      .def_readwrite("name", &fuse::MeshDecl::name)
      .def_readwrite("source", &fuse::MeshDecl::source)
      .def("__repr__", [](const fuse::MeshDecl& d) {
        return "MeshDecl('" + d.name + "', '" + d.source + "')";
      });

  py::class_<fuse::Region>(m, "Region")
      .def(py::init([](std::string name, fuse::MeshId mesh, std::string entity_set,
                       int dimension) {
             return fuse::Region{std::move(name), mesh, std::move(entity_set), dimension};
           }),
           py::arg("name"), py::arg("mesh"), py::arg("entity_set") = std::string{},
           py::arg("dimension") = -1)
      .def_readwrite("name", &fuse::Region::name)
      .def_readwrite("mesh", &fuse::Region::mesh)
      .def_readwrite("entity_set", &fuse::Region::entity_set)
      .def_readwrite("dimension", &fuse::Region::dimension)
      .def("__repr__", [](const fuse::Region& r) { return "Region('" + r.name + "')"; });

  py::class_<fuse::RankSet>(m, "RankSet",
      "fuse's own richer RankSet (docs/dev/python-surface.md): a contiguous block via "
      "RankSet(first, count), or an arbitrary rank list via RankSet([0, 2, 4]).")
      .def(py::init<>())
      .def(py::init<int, int>(), py::arg("first"), py::arg("count"))
      .def(py::init<std::vector<int>>(), py::arg("ranks"))
      .def_property_readonly("ranks", &fuse::RankSet::ranks)
      .def_property_readonly("size", &fuse::RankSet::size)
      .def_property_readonly("empty", &fuse::RankSet::empty)
      .def_property_readonly("min_rank", &fuse::RankSet::min_rank)
      .def_property_readonly("max_rank", &fuse::RankSet::max_rank)
      .def_property_readonly("is_contiguous", &fuse::RankSet::is_contiguous)
      .def("contains", &fuse::RankSet::contains, py::arg("rank"))
      .def("overlaps", &fuse::RankSet::overlaps, py::arg("other"))
      .def("same_as", &fuse::RankSet::same_as, py::arg("other"))
      .def("index_of", &fuse::RankSet::index_of, py::arg("rank"))
      .def(py::self == py::self)
      .def("__repr__", [](const fuse::RankSet& r) {
        std::ostringstream os;
        os << "RankSet([";
        bool first = true;
        for (int x : r.ranks()) { if (!first) os << ", "; os << x; first = false; }
        os << "])";
        return os.str();
      });

  py::enum_<fuse::ComponentKind>(m, "ComponentKind")
      .value("Physics", fuse::ComponentKind::Physics)
      .value("Coupler", fuse::ComponentKind::Coupler);

  py::class_<fuse::ComponentDecl>(m, "ComponentDecl")
      .def(py::init<>())
      .def_readwrite("name", &fuse::ComponentDecl::name)
      .def_readwrite("ranks", &fuse::ComponentDecl::ranks)
      .def_readwrite("interval", &fuse::ComponentDecl::interval)
      .def_readwrite("fields", &fuse::ComponentDecl::fields)
      .def_readwrite("kind", &fuse::ComponentDecl::kind)
      .def("__repr__", [](const fuse::ComponentDecl& c) {
        return "ComponentDecl('" + c.name + "')";
      });

  py::enum_<fuse::Window::Kind>(m, "_WindowKind")  // exposed via Window.Kind below
      .value("CouplingPeriod", fuse::Window::Kind::CouplingPeriod)
      .value("FromComponent", fuse::Window::Kind::FromComponent)
      .value("Explicit", fuse::Window::Kind::Explicit);

  py::class_<fuse::Window> window(m, "Window");
  window.def(py::init<>())
      .def_readwrite("kind", &fuse::Window::kind)
      .def_readwrite("component", &fuse::Window::component)
      .def_readwrite("length", &fuse::Window::length)
      .def_static("coupling_period", &fuse::Window::coupling_period)
      .def_static("from_component", &fuse::Window::from)
      .def_static("explicit_length",
                  [](py::handle len) {
                    return fuse::Window::explicit_length(coerce_rational(len, "Window.explicit_length"));
                  },
                  py::arg("length"));
  window.attr("Kind") = m.attr("_WindowKind");

  py::class_<fuse::FieldSpec>(m, "FieldSpec")
      .def(py::init<>())
      .def_readwrite("name", &fuse::FieldSpec::name)
      .def_readwrite("region", &fuse::FieldSpec::region)
      .def_readwrite("disc", &fuse::FieldSpec::disc)
      .def_readwrite("order", &fuse::FieldSpec::order)
      .def_readwrite("components", &fuse::FieldSpec::components)
      .def_readwrite("units", &fuse::FieldSpec::units)
      .def_readwrite("conservation", &fuse::FieldSpec::conservation)
      .def_readwrite("intent", &fuse::FieldSpec::intent)
      .def("__repr__", [](const fuse::FieldSpec& f) { return "FieldSpec('" + f.name + "')"; });

  // --- Channel and its fluent sub-structs (docs/dev/python-surface.md, "Signature
  // conventions": grouped by fuse::Channel's own C++ sub-struct, not re-flattened). -----------
  // pybind11 3.1's py::arithmetic() adds __or__/__and__ only for an enum implicitly
  // convertible to its underlying type, which a C++ enum class (fuse::RemapFlags) never is
  // -- bound explicitly here instead, using fuse's own operator|/operator& (channel.hpp) so
  // Python's `|`/`&` compute the identical bit pattern the C++ operators do.
  py::enum_<fuse::RemapFlags>(m, "RemapFlags")
      .value("None_", fuse::RemapFlags::None)
      .value("Conservative", fuse::RemapFlags::Conservative)
      .value("Monotone", fuse::RemapFlags::Monotone)
      .value("HighOrder", fuse::RemapFlags::HighOrder)
      .def("__or__", [](fuse::RemapFlags a, fuse::RemapFlags b) { return a | b; })
      .def("__and__", [](fuse::RemapFlags a, fuse::RemapFlags b) { return a & b; })
      .def("__bool__", [](fuse::RemapFlags f) { return static_cast<std::uint32_t>(f) != 0; });
  m.def("has", [](fuse::RemapFlags set, fuse::RemapFlags bit) { return fuse::has(set, bit); },
       py::arg("flags"), py::arg("bit"));

  py::class_<fuse::AccumulateSpec>(m, "AccumulateSpec")
      .def_readonly("op", &fuse::AccumulateSpec::op)
      .def_readonly("window", &fuse::AccumulateSpec::window);
  py::class_<fuse::RemapSpec>(m, "RemapSpec")
      .def_readonly("flags", &fuse::RemapSpec::flags)
      .def_readonly("method_override", &fuse::RemapSpec::method_override)
      .def_readonly("remap_on", &fuse::RemapSpec::remap_on);
  py::class_<fuse::SpacetimeSpec>(m, "SpacetimeSpec")
      .def_readonly("basis", &fuse::SpacetimeSpec::basis)
      .def_readonly("degree", &fuse::SpacetimeSpec::degree)
      .def_readonly("window", &fuse::SpacetimeSpec::window);
  py::class_<fuse::RelaxSpec>(m, "RelaxSpec")
      .def_readonly("enabled", &fuse::RelaxSpec::enabled)
      .def_readonly("omega", &fuse::RelaxSpec::omega);
  py::class_<fuse::ExtrapolateSpec>(m, "ExtrapolateSpec")
      .def_readonly("enabled", &fuse::ExtrapolateSpec::enabled)
      .def_readonly("order", &fuse::ExtrapolateSpec::order);

  py::class_<fuse::Channel>(m, "Channel")
      .def_readonly("name", &fuse::Channel::name)
      .def_readonly("source", &fuse::Channel::source)
      .def_readonly("target", &fuse::Channel::target)
      .def_readonly("accumulate", &fuse::Channel::accumulate)
      .def_readonly("remap", &fuse::Channel::remap)
      .def_readonly("spacetime", &fuse::Channel::spacetime)
      .def_readonly("relax", &fuse::Channel::relax)
      .def_readonly("extrapolate", &fuse::Channel::extrapolate)
      .def_readonly("lag", &fuse::Channel::lag)
      .def_readonly("time_level_mismatch_reason", &fuse::Channel::time_level_mismatch_reason)
      .def("__repr__", [](const fuse::Channel& c) { return "Channel('" + c.name + "')"; });

  py::class_<fuse::ChannelBuilder>(m, "ChannelBuilder")
      .def("accumulate",
          [](fuse::ChannelBuilder& b, fuse::AccumOp op, std::optional<fuse::Window> w) {
            return &b.accumulate(op, w.value_or(fuse::Window::coupling_period()));
          },
          py::arg("op"), py::arg("window") = py::none(), py::return_value_policy::reference_internal)
      .def("remap", &fuse::ChannelBuilder::remap, py::arg("flags"),
          py::return_value_policy::reference_internal)
      .def("remap_method", &fuse::ChannelBuilder::remap_method, py::arg("method"),
          py::return_value_policy::reference_internal)
      .def("remap_on", &fuse::ChannelBuilder::remap_on, py::arg("component"),
          py::return_value_policy::reference_internal)
      .def("conservative", &fuse::ChannelBuilder::conservative,
          py::return_value_policy::reference_internal)
      .def("spacetime",
          [](fuse::ChannelBuilder& b, fuse::TemporalBasis basis, int degree,
             std::optional<fuse::Window> w) {
            return &b.spacetime(basis, degree, w.value_or(fuse::Window::coupling_period()));
          },
          py::arg("basis"), py::arg("degree"), py::arg("window") = py::none(),
          py::return_value_policy::reference_internal)
      .def("relax", &fuse::ChannelBuilder::relax, py::arg("omega"),
          py::return_value_policy::reference_internal)
      .def("extrapolate", &fuse::ChannelBuilder::extrapolate, py::arg("order"),
          py::return_value_policy::reference_internal)
      .def("lag", &fuse::ChannelBuilder::lag, py::arg("steps"),
          py::return_value_policy::reference_internal)
      .def("allow_time_level_mismatch", &fuse::ChannelBuilder::allow_time_level_mismatch,
          py::arg("reason"), py::return_value_policy::reference_internal);

  // --- Convergence criteria, acceleration, Newton configuration (docs/dev/python-surface.md:
  // "new relative to soma" -- the whole vocabulary below has no upstream equivalent). ----------
  py::enum_<fuse::MeasureKind>(m, "MeasureKind")
      .value("Relative", fuse::MeasureKind::Relative)
      .value("Absolute", fuse::MeasureKind::Absolute)
      .value("AbsoluteOrRelative", fuse::MeasureKind::AbsoluteOrRelative)
      .value("ResidualRelative", fuse::MeasureKind::ResidualRelative);

  py::class_<fuse::ConvergenceMeasure>(m, "ConvergenceMeasure")
      .def_readonly("kind", &fuse::ConvergenceMeasure::kind)
      .def_readonly("tolerance", &fuse::ConvergenceMeasure::tolerance)
      .def_readonly("field", &fuse::ConvergenceMeasure::field)
      .def("__and__",
          [](const fuse::ConvergenceMeasure& a, const fuse::ConvergenceMeasure& b) {
            return a & b;
          })
      .def("__repr__", [](const fuse::ConvergenceMeasure& m_) {
        return "ConvergenceMeasure('" + m_.field + "')";
      });

  py::class_<fuse::ConvergenceCriterion>(m, "ConvergenceCriterion")
      .def(py::init<>())
      .def("add", &fuse::ConvergenceCriterion::add, py::arg("measure"))
      .def_property_readonly("empty", &fuse::ConvergenceCriterion::empty)
      .def_property_readonly("measures", &fuse::ConvergenceCriterion::measures)
      .def("__len__", &fuse::ConvergenceCriterion::size)
      .def("__and__",
          [](fuse::ConvergenceCriterion c, const fuse::ConvergenceMeasure& m_) {
            return c & m_;
          });

  py::module_ converged = m.def_submodule(
      "Converged", "fuse::Converged::{Relative,Absolute,AbsoluteOrRelative,ResidualRelative}.");
  converged.def("Relative", &fuse::Converged::Relative, py::arg("tol"), py::arg("field"));
  converged.def("Absolute", &fuse::Converged::Absolute, py::arg("tol"), py::arg("field"));
  converged.def("AbsoluteOrRelative", &fuse::Converged::AbsoluteOrRelative, py::arg("tol"),
               py::arg("field"));
  converged.def("ResidualRelative", &fuse::Converged::ResidualRelative, py::arg("tol"),
               py::arg("field"));

  py::enum_<fuse::AccelKind>(m, "AccelKind")
      .value("None_", fuse::AccelKind::None)
      .value("Constant", fuse::AccelKind::Constant)
      .value("Aitken", fuse::AccelKind::Aitken)
      .value("Secant", fuse::AccelKind::Secant)
      .value("Steffensen", fuse::AccelKind::Steffensen)
      .value("IQN_ILS", fuse::AccelKind::IQN_ILS)
      .value("IQN_IMVJ", fuse::AccelKind::IQN_IMVJ)
      .value("Newton", fuse::AccelKind::Newton);
  py::enum_<fuse::FilterKind>(m, "FilterKind")
      .value("None_", fuse::FilterKind::None)
      .value("QR1", fuse::FilterKind::QR1)
      .value("QR2", fuse::FilterKind::QR2)
      .value("QR3", fuse::FilterKind::QR3);

  py::class_<fuse::AccelConfig>(m, "AccelConfig")
      .def(py::init([](fuse::AccelKind kind, double initial_relaxation, double omega, int reuse,
                       fuse::FilterKind filter, double limit, bool scale_fields) {
             fuse::AccelConfig c{};
             c.kind = kind;
             c.initial_relaxation = initial_relaxation;
             c.omega = omega;
             c.reuse = reuse;
             c.filter = filter;
             c.limit = limit;
             c.scale_fields = scale_fields;
             return c;
           }),
           py::arg("kind") = fuse::AccelKind::Constant, py::arg("initial_relaxation") = 0.1,
           py::arg("omega") = 0.5, py::arg("reuse") = 8, py::arg("filter") = fuse::FilterKind::QR2,
           py::arg("limit") = 1.0e-2, py::arg("scale_fields") = true)
      .def_readwrite("kind", &fuse::AccelConfig::kind)
      .def_readwrite("initial_relaxation", &fuse::AccelConfig::initial_relaxation)
      .def_readwrite("omega", &fuse::AccelConfig::omega)
      .def_readwrite("reuse", &fuse::AccelConfig::reuse)
      .def_readwrite("filter", &fuse::AccelConfig::filter)
      .def_readwrite("limit", &fuse::AccelConfig::limit)
      .def_readwrite("scale_fields", &fuse::AccelConfig::scale_fields);

  py::enum_<fuse::LineSearchKind>(m, "LineSearchKind")
      .value("BT", fuse::LineSearchKind::BT)
      .value("CP", fuse::LineSearchKind::CP)
      .value("Basic", fuse::LineSearchKind::Basic);
  py::enum_<fuse::EisenstatWalkerChoice>(m, "EisenstatWalkerChoice")
      .value("Constant", fuse::EisenstatWalkerChoice::Constant)
      .value("Choice1", fuse::EisenstatWalkerChoice::Choice1)
      .value("Choice2", fuse::EisenstatWalkerChoice::Choice2);
  py::enum_<fuse::NewtonSubstrate>(m, "NewtonSubstrate")
      .value("Native", fuse::NewtonSubstrate::Native)
      .value("Petsc", fuse::NewtonSubstrate::Petsc);
  py::enum_<fuse::NewtonNorm>(m, "NewtonNorm")
      .value("Plain", fuse::NewtonNorm::Plain)
      .value("MeasureWeighted", fuse::NewtonNorm::MeasureWeighted);
  py::enum_<fuse::NewtonResidualPath>(m, "NewtonResidualPath")
      .value("Advance", fuse::NewtonResidualPath::Advance)
      .value("Direct", fuse::NewtonResidualPath::Direct);

  // All 28 fields, matching include/fuse/model/iteration.hpp's own NewtonConfig exactly --
  // test_graph.py's NEWTON_CONFIG_FIELDS list (extracted by grep from the same header) is
  // what checks this binding does not silently drop one.
  py::class_<fuse::NewtonConfig>(m, "NewtonConfig")
      .def(py::init([](fuse::LineSearchKind line_search_kind,
                       fuse::EisenstatWalkerChoice forcing_choice, double forcing_gamma,
                       double forcing_alpha, double eta_initial, double eta_max, double eta_min,
                       double forcing_safeguard_threshold, double forcing_eta_constant,
                       double atol, double rtol, double stol, double divergence_tolerance,
                       int max_nonlinear_iterations, std::size_t max_function_evaluations,
                       double armijo_alpha, double lambda_min, int max_backtracks,
                       double secant_tol, int max_secant_steps, double damping,
                       int gmres_restart, std::size_t gmres_max_iterations,
                       int preconditioner_lag, bool use_block_jacobi_preconditioner,
                       fuse::NewtonSubstrate substrate, fuse::NewtonNorm norm,
                       fuse::NewtonResidualPath residual_path) {
             fuse::NewtonConfig c{};
             c.line_search_kind = line_search_kind;
             c.forcing_choice = forcing_choice;
             c.forcing_gamma = forcing_gamma;
             c.forcing_alpha = forcing_alpha;
             c.eta_initial = eta_initial;
             c.eta_max = eta_max;
             c.eta_min = eta_min;
             c.forcing_safeguard_threshold = forcing_safeguard_threshold;
             c.forcing_eta_constant = forcing_eta_constant;
             c.atol = atol;
             c.rtol = rtol;
             c.stol = stol;
             c.divergence_tolerance = divergence_tolerance;
             c.max_nonlinear_iterations = max_nonlinear_iterations;
             c.max_function_evaluations = max_function_evaluations;
             c.armijo_alpha = armijo_alpha;
             c.lambda_min = lambda_min;
             c.max_backtracks = max_backtracks;
             c.secant_tol = secant_tol;
             c.max_secant_steps = max_secant_steps;
             c.damping = damping;
             c.gmres_restart = gmres_restart;
             c.gmres_max_iterations = gmres_max_iterations;
             c.preconditioner_lag = preconditioner_lag;
             c.use_block_jacobi_preconditioner = use_block_jacobi_preconditioner;
             c.substrate = substrate;
             c.norm = norm;
             c.residual_path = residual_path;
             return c;
           }),
           py::arg("line_search_kind") = fuse::LineSearchKind::BT,
           py::arg("forcing_choice") = fuse::EisenstatWalkerChoice::Choice2,
           py::arg("forcing_gamma") = 0.9, py::arg("forcing_alpha") = 2.0,
           py::arg("eta_initial") = 0.3, py::arg("eta_max") = 0.9, py::arg("eta_min") = 0.0,
           py::arg("forcing_safeguard_threshold") = 0.1, py::arg("forcing_eta_constant") = 0.1,
           py::arg("atol") = 1e-50, py::arg("rtol") = 1e-8, py::arg("stol") = 1e-8,
           py::arg("divergence_tolerance") = 1e4, py::arg("max_nonlinear_iterations") = 50,
           py::arg("max_function_evaluations") = 10000, py::arg("armijo_alpha") = 1e-4,
           py::arg("lambda_min") = 1e-12, py::arg("max_backtracks") = 40,
           py::arg("secant_tol") = 1e-10, py::arg("max_secant_steps") = 40,
           py::arg("damping") = 1.0, py::arg("gmres_restart") = 30,
           py::arg("gmres_max_iterations") = 1000, py::arg("preconditioner_lag") = 1,
           py::arg("use_block_jacobi_preconditioner") = false,
           py::arg("substrate") = fuse::NewtonSubstrate::Native,
           py::arg("norm") = fuse::NewtonNorm::Plain,
           py::arg("residual_path") = fuse::NewtonResidualPath::Advance)
      .def_readwrite("line_search_kind", &fuse::NewtonConfig::line_search_kind)
      .def_readwrite("forcing_choice", &fuse::NewtonConfig::forcing_choice)
      .def_readwrite("forcing_gamma", &fuse::NewtonConfig::forcing_gamma)
      .def_readwrite("forcing_alpha", &fuse::NewtonConfig::forcing_alpha)
      .def_readwrite("eta_initial", &fuse::NewtonConfig::eta_initial)
      .def_readwrite("eta_max", &fuse::NewtonConfig::eta_max)
      .def_readwrite("eta_min", &fuse::NewtonConfig::eta_min)
      .def_readwrite("forcing_safeguard_threshold", &fuse::NewtonConfig::forcing_safeguard_threshold)
      .def_readwrite("forcing_eta_constant", &fuse::NewtonConfig::forcing_eta_constant)
      .def_readwrite("atol", &fuse::NewtonConfig::atol)
      .def_readwrite("rtol", &fuse::NewtonConfig::rtol)
      .def_readwrite("stol", &fuse::NewtonConfig::stol)
      .def_readwrite("divergence_tolerance", &fuse::NewtonConfig::divergence_tolerance)
      .def_readwrite("max_nonlinear_iterations", &fuse::NewtonConfig::max_nonlinear_iterations)
      .def_readwrite("max_function_evaluations", &fuse::NewtonConfig::max_function_evaluations)
      .def_readwrite("armijo_alpha", &fuse::NewtonConfig::armijo_alpha)
      .def_readwrite("lambda_min", &fuse::NewtonConfig::lambda_min)
      .def_readwrite("max_backtracks", &fuse::NewtonConfig::max_backtracks)
      .def_readwrite("secant_tol", &fuse::NewtonConfig::secant_tol)
      .def_readwrite("max_secant_steps", &fuse::NewtonConfig::max_secant_steps)
      .def_readwrite("damping", &fuse::NewtonConfig::damping)
      .def_readwrite("gmres_restart", &fuse::NewtonConfig::gmres_restart)
      .def_readwrite("gmres_max_iterations", &fuse::NewtonConfig::gmres_max_iterations)
      .def_readwrite("preconditioner_lag", &fuse::NewtonConfig::preconditioner_lag)
      .def_readwrite("use_block_jacobi_preconditioner",
                     &fuse::NewtonConfig::use_block_jacobi_preconditioner)
      .def_readwrite("substrate", &fuse::NewtonConfig::substrate)
      .def_readwrite("norm", &fuse::NewtonConfig::norm)
      .def_readwrite("residual_path", &fuse::NewtonConfig::residual_path);

  py::class_<fuse::IterationRegionDecl>(m, "IterationRegionDecl")
      .def_readonly("name", &fuse::IterationRegionDecl::name)
      .def_readonly("channels", &fuse::IterationRegionDecl::channels)
      .def_readonly("max_iterations", &fuse::IterationRegionDecl::max_iterations)
      .def_readonly("convergence", &fuse::IterationRegionDecl::convergence)
      .def_readonly("acceleration", &fuse::IterationRegionDecl::acceleration)
      .def_readonly("requires_residual_capability",
                    &fuse::IterationRegionDecl::requires_residual_capability)
      .def_readonly("requires_linearization_capability",
                    &fuse::IterationRegionDecl::requires_linearization_capability)
      .def_readonly("newton_config", &fuse::IterationRegionDecl::newton_config);

  py::class_<fuse::IterationBuilder>(m, "IterationBuilder")
      .def("until",
          [](fuse::IterationBuilder& b, py::object crit_or_measure) {
            if (py::isinstance<fuse::ConvergenceCriterion>(crit_or_measure)) {
              return &b.until(crit_or_measure.cast<fuse::ConvergenceCriterion>());
            }
            return &b.until(crit_or_measure.cast<fuse::ConvergenceMeasure>());
          },
          py::arg("criterion_or_measure"), py::return_value_policy::reference_internal)
      .def("max_iterations", &fuse::IterationBuilder::max_iterations, py::arg("n"),
          py::return_value_policy::reference_internal)
      .def("accelerate", &fuse::IterationBuilder::accelerate, py::arg("config"),
          py::return_value_policy::reference_internal)
      .def("initial_relaxation", &fuse::IterationBuilder::initial_relaxation, py::arg("omega"),
          py::return_value_policy::reference_internal)
      .def("name", &fuse::IterationBuilder::name, py::arg("n"),
          py::return_value_policy::reference_internal)
      .def("newton",
          [](fuse::IterationBuilder& b, std::optional<fuse::NewtonConfig> cfg) {
            return &b.newton(cfg.value_or(fuse::NewtonConfig{}));
          },
          py::arg("config") = py::none(), py::return_value_policy::reference_internal)
      .def("requires_residual", &fuse::IterationBuilder::requires_residual, py::arg("v") = true,
          py::return_value_policy::reference_internal)
      .def("requires_linearization", &fuse::IterationBuilder::requires_linearization,
          py::arg("v") = true, py::return_value_policy::reference_internal)
      .def("id", &fuse::IterationBuilder::id);

  // --- CouplingGraph itself ------------------------------------------------------------
  py::class_<fuse::CouplingGraph> graph(m, "CouplingGraph",
      "Spec layer 1, the authored representation (docs/dev/python-surface.md).");
  graph.def(py::init<>())
      .def("add_mesh", [](fuse::CouplingGraph& g, std::string name, std::string source) {
             return g.add_mesh(fuse::MeshDecl{std::move(name), std::move(source)});
           }, py::arg("name"), py::arg("source"))
      .def("add_region",
          [](fuse::CouplingGraph& g, std::string name, fuse::MeshId mesh, std::string entity_set,
             int dimension) {
            return g.add_region(fuse::Region{std::move(name), mesh, std::move(entity_set),
                                             dimension});
          },
          py::arg("name"), py::arg("mesh"), py::arg("entity_set") = std::string{},
          py::arg("dimension") = -1)
      .def("add_component",
          [](fuse::CouplingGraph& g, std::string name, std::optional<fuse::RankSet> ranks,
             py::handle interval, fuse::ComponentKind kind) {
            fuse::ComponentDecl d{};
            d.name = std::move(name);
            if (ranks.has_value()) d.ranks = *ranks;
            d.interval = coerce_rational(interval, "add_component(interval=)");
            d.kind = kind;
            return g.add_component(std::move(d));
          },
          py::arg("name"), py::arg("ranks") = py::none(), py::arg("interval") = 1,
          py::arg("kind") = fuse::ComponentKind::Physics)
      .def("add_coupler",
          [](fuse::CouplingGraph& g, std::string name, std::optional<fuse::RankSet> ranks) {
            fuse::CouplerDecl d{};
            d.name = std::move(name);
            if (ranks.has_value()) d.ranks = *ranks;
            return g.add_coupler(std::move(d));
          },
          py::arg("name"), py::arg("ranks") = py::none())
      .def("add_field",
          [](fuse::CouplingGraph& g, fuse::ComponentId owner, std::string name,
             fuse::RegionId region, fuse::Discretization disc, int order, int components,
             fuse::Units units, fuse::Conservation conservation, fuse::Intent intent) {
            fuse::FieldSpec f{};
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
          py::arg("disc") = fuse::Discretization::FV, py::arg("order") = 0,
          py::arg("components") = 1, py::arg("units") = fuse::Units{},
          py::arg("conservation") = fuse::Conservation::None,
          py::arg("intent") = fuse::Intent::Provides)
      .def("add_channel", &fuse::CouplingGraph::add_channel, py::arg("name"), py::arg("source"),
          py::arg("target"))
      .def("iterate", &fuse::CouplingGraph::iterate, py::arg("channels"))
      .def("channel", &fuse::CouplingGraph::channel, py::arg("id"))
      .def("mesh", &fuse::CouplingGraph::mesh, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("region", &fuse::CouplingGraph::region, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("component", &fuse::CouplingGraph::component, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("field", &fuse::CouplingGraph::field, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("channel_at", &fuse::CouplingGraph::channel_at, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("iteration_region", &fuse::CouplingGraph::iteration_region, py::arg("id"),
          py::return_value_policy::reference_internal)
      .def("owner_of", &fuse::CouplingGraph::owner_of, py::arg("field"))
      .def("find_component",
          [](const fuse::CouplingGraph& g, std::string_view name) {
            const auto id = g.find_component(name);
            if (!id.has_value()) {
              FUSE_THROW(fuse::InvalidArgument,
                        "find_component: the graph declares no component named '{}'", name);
            }
            return *id;
          },
          py::arg("name"))
      .def("find_field",
          [](const fuse::CouplingGraph& g, fuse::ComponentId owner, std::string_view name) {
            const auto id = g.find_field(owner, name);
            if (!id.has_value()) {
              FUSE_THROW(fuse::InvalidArgument,
                        "find_field: component {} declares no field named '{}'", owner.value(),
                        name);
            }
            return *id;
          },
          py::arg("owner"), py::arg("name"))
      .def_property_readonly("mesh_count", &fuse::CouplingGraph::mesh_count)
      .def_property_readonly("region_count", &fuse::CouplingGraph::region_count)
      .def_property_readonly("component_count", &fuse::CouplingGraph::component_count)
      .def_property_readonly("field_count", &fuse::CouplingGraph::field_count)
      .def_property_readonly("channel_count", &fuse::CouplingGraph::channel_count)
      .def_property_readonly("iteration_region_count",
                             &fuse::CouplingGraph::iteration_region_count)
      .def("to_yaml",
          [](const fuse::CouplingGraph& g) {
            auto r = fuse::to_yaml(g);
            if (!r.ok()) pfuse_py::raise_for(r);
            return r.value();
          })
      .def("to_bytes",
          [](const fuse::CouplingGraph& g) {
            const auto bytes = fuse::serialize(g);
            return py::bytes(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          })
      .def("__copy__", [](const fuse::CouplingGraph& g) { return fuse::CouplingGraph(g); })
      .def("__repr__", [](const fuse::CouplingGraph& g) {
        return "CouplingGraph(" + std::to_string(g.component_count()) + " component(s), " +
               std::to_string(g.channel_count()) + " channel(s))";
      });

  m.def("structurally_equal", &fuse::structurally_equal, py::arg("a"), py::arg("b"));

  m.def("from_yaml", [](std::string_view text) {
         auto r = fuse::from_yaml(text);
         if (!r.ok()) pfuse_py::raise_for(r);
         return r.value();
       }, py::arg("text"));
  m.def("load_yaml_file", [](const std::string& path) {
         auto r = fuse::load_yaml_file(path);
         if (!r.ok()) pfuse_py::raise_for(r);
         return r.value();
       }, py::arg("path"));
  m.def("from_bytes", [](py::bytes blob) {
         const std::string s = blob;
         const auto* begin = reinterpret_cast<const std::byte*>(s.data());
         auto r = fuse::deserialize(std::span<const std::byte>(begin, s.size()));
         if (!r.ok()) pfuse_py::raise_for(r);
         return r.value();
       }, py::arg("blob"));

  m.def("validate", [](const fuse::CouplingGraph& g) { return fuse::validate(g); },
       py::arg("graph"));

  py::class_<fuse::OperatorDAG>(m, "OperatorDAG")
      .def("__repr__", [](const fuse::OperatorDAG&) { return std::string("OperatorDAG(...)"); });
  py::class_<fuse::LoweringResult>(m, "LoweringResult")
      .def_readonly("dag", &fuse::LoweringResult::dag)
      .def_readonly("diagnostics", &fuse::LoweringResult::diags)
      .def_property_readonly("ok", &fuse::LoweringResult::ok);
  m.def("lower", [](const fuse::CouplingGraph& g) { return fuse::lower(g); }, py::arg("graph"));
  m.def("dump_text", &fuse::dump_text, py::arg("dag"));
  m.def("dump_graphviz", &fuse::dump_graphviz, py::arg("dag"));

  py::class_<fuse::ExecutionPlan>(m, "ExecutionPlan")
      .def_property_readonly("period", &fuse::ExecutionPlan::period)
      .def_property_readonly("lockstep", &fuse::ExecutionPlan::lockstep)
      .def_property_readonly("region_count", &fuse::ExecutionPlan::region_count)
      .def("to_string", &fuse::ExecutionPlan::to_string)
      .def("__repr__", [](const fuse::ExecutionPlan& p) {
        return "ExecutionPlan(period=" + p.period().to_string() + ", lockstep=" +
               (p.lockstep() ? std::string("True") : std::string("False")) + ")";
      });
  py::class_<fuse::ScheduleResult>(m, "ScheduleResult")
      .def_readonly("plan", &fuse::ScheduleResult::plan)
      .def_readonly("diagnostics", &fuse::ScheduleResult::diags)
      .def_property_readonly("ok", &fuse::ScheduleResult::ok);

  py::class_<fuse::SerialMeshBackend>(m, "SerialMeshBackend",
      "fuse::SerialMeshBackend: the dependency-free structured backend v1 drives "
      "(docs/dev/python-surface.md -- no native/MOAB backend is bound).")
      .def(py::init<>())
      .def("load", [](fuse::SerialMeshBackend& b, const fuse::MeshDecl& decl) {
             auto r = b.load(decl);
             if (!r.ok()) pfuse_py::raise_for(r);
           }, py::arg("decl"))
      .def_property_readonly("name", &fuse::SerialMeshBackend::name);
}

}  // namespace pfuse_py
