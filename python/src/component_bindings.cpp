// fuse.Component (the Python-facing base class) and ForwardingComponent (the bridge into
// fuse::Component), StepContext, RunnerConfig, RunStats, and Session (owns a graph copy +
// registry + fuse::Runner, mirroring upstream soma.Session -- docs/dev/python-surface.md).
//
// -----------------------------------------------------------------------------------------
// THE GIL (docs/dev/python-surface.md, "The GIL")
// -----------------------------------------------------------------------------------------
// Released around exactly the calls that do real work and touch no Python object: fuse::
// validate()/lower()/schedule(), the portions of Session.initialize() that call them plus
// fuse::Runner::initialize(), and Session.run()/run_period()'s own instruction-stream walk.
// ForwardingComponent reacquires the GIL before every callback into the Python object, and
// catches py::error_already_set (which FETCHES the Python error, clearing the interpreter's
// indicator) to convert it into a fuse::Result<void> failure the C++ runtime can unwind
// through normally; the stashed exception is re-raised VERBATIM once control returns to the
// binding boundary (Session::rethrow_pending()).
//
// -----------------------------------------------------------------------------------------
// NEWTON: NATIVE COMPONENTS ONLY
// -----------------------------------------------------------------------------------------
// Component::implicit() is never exposed to a Python Component subclass: ForwardingComponent
// always returns nullptr from implicit(). The EXISTING fuse::verify_newton_capability_support()
// (src/runtime/checkpoint_support.cpp) already refuses a Newton region whose member's
// capabilities() does not set kCapabilityResidual, with E-NEWTON-CAPABILITY -- this covers a
// ForwardingComponent with the default (unclaimed) capabilities honestly, with NO new code
// needed here (see test_newton.py's own test that makes this existing refusal fire; note that the EXISTING
// message's C++-specific guidance -- "override Component::implicit()" -- does not apply to a
// Python reader).
// What IS new here: a Python component that tries to CLAIM kCapabilityResidual/
// kCapabilityLinearization via its own CAPABILITIES attribute (lying about a capability
// ForwardingComponent can never actually back, since implicit() always returns nullptr) is
// refused immediately, at Session.add_component() time, with a message that says why in terms
// a Python reader can act on.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "errors.hpp"
#include "field_registry.hpp"
#include "fuse/base/error.hpp"
#include "fuse/backend/serial_backend.hpp"
#include "fuse/ir/lowering.hpp"
#include "fuse/model/capabilities.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/validate.hpp"
#include "fuse/runtime/component.hpp"
#include "fuse/runtime/runner.hpp"
#include "fuse/schedule/clock.hpp"
#include "fuse/schedule/plan.hpp"

namespace py = pybind11;

namespace pfuse_py {

/// The Python-facing base class. Deliberately NOT a pybind11 trampoline over fuse::Component:
/// fuse::ComponentSet::add() takes a unique_ptr<Component>, and handing it a trampoline whose
/// real owner is the Python object would be double ownership for one lifetime.
/// ForwardingComponent (below) holds a strong reference to the Python object instead.
class PyComponentBase {
 public:
  explicit PyComponentBase(std::string name) : name_(std::move(name)) {
    if (name_.empty()) {
      FUSE_THROW(fuse::InvalidArgument,
                "Component: the name must be non-empty and must match a component name in "
                "the coupling graph");
    }
  }
  virtual ~PyComponentBase() = default;
  [[nodiscard]] const std::string& name() const noexcept { return name_; }

 private:
  std::string name_;
};

/// Bridges a Python object into fuse::Component. See this file's own header comment for the
/// GIL policy and the Newton-capability restriction.
class ForwardingComponent final : public fuse::Component {
 public:
  ForwardingComponent(py::object obj, std::string name, PyFieldRegistry registry)
      : obj_(std::move(obj)),
        name_(std::move(name)),
        registry_(std::move(registry)),
        keep_(registry_.keep()) {
    has_save_ = py::hasattr(obj_, "save_state");
    has_restore_ = py::hasattr(obj_, "restore_state");
    rollback_ = has_save_ && has_restore_;
    capabilities_ = compute_capabilities();
  }

  ~ForwardingComponent() override {
    py::gil_scoped_acquire gil;
    obj_ = py::object();
    pending_ = py::object();
    registry_ = PyFieldRegistry{};
    keep_.reset();
  }

  [[nodiscard]] std::string name() const override { return name_; }

  fuse::Result<void> register_fields(fuse::FieldRegistry& registry,
                                     const fuse::CouplingGraph& graph) override {
    return call("register_fields", [&](py::object& fn) {
      PyFieldRegistry view = (&registry == &registry_.raw())
                                 ? registry_
                                 : PyFieldRegistry{&registry, keep_};
      view.set_graph(&graph);
      fn(py::cast(std::move(view)), py::cast(graph, py::return_value_policy::reference));
    });
  }

  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    return call("advance", [&](py::object& fn) { fn(py::cast(ctx)); });
  }

  fuse::Result<void> save_state() override {
    if (!has_save_) return fuse::Result<void>::success();
    return call("save_state", [](py::object& fn) { fn(); });
  }

  fuse::Result<void> restore_state() override {
    if (!has_restore_) return fuse::Result<void>::success();
    return call("restore_state", [](py::object& fn) { fn(); });
  }

  [[nodiscard]] bool supports_rollback() const { return rollback_; }
  [[nodiscard]] std::uint64_t capabilities() const override { return capabilities_; }

  /// Always false: a Python component is always an in-process peer, never a participant-ABI
  /// ExternalStepComponent. Not an override point in v1.
  [[nodiscard]] bool is_externally_driven() const noexcept override { return false; }

  /// Always nullptr: see this file's own header comment, "NEWTON: NATIVE COMPONENTS ONLY".
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return nullptr; }

  [[nodiscard]] bool has_pending() const { return static_cast<bool>(pending_); }

  /// Re-raises the Python exception this component swallowed, if any. Called at the binding
  /// boundary with the GIL held.
  void rethrow_pending() {
    if (!pending_) return;
    py::object value = pending_;
    pending_ = py::object();
    PyErr_SetObject(reinterpret_cast<PyObject*>(Py_TYPE(value.ptr())), value.ptr());
    throw py::error_already_set();
  }

 private:
  /// Reads the Python object's own CAPABILITIES attribute (class or instance; default 0),
  /// ORs in kCapabilityRollback from the inferred save_state/restore_state pair, and REFUSES
  /// immediately if the result claims kCapabilityResidual or kCapabilityLinearization: those
  /// two bits mean "Component::implicit() returns a usable ImplicitComponent*", which
  /// ForwardingComponent can never back (it always returns nullptr -- see this file's header
  /// comment). A component that is honest and says nothing about these two bits is NOT
  /// refused here; it is refused later, by fuse::verify_newton_capability_support() itself,
  /// only if it is actually placed in a Newton-selected region (test_newton.py's own test
  /// proves that existing refusal fires for the honest-default case).
  [[nodiscard]] std::uint64_t compute_capabilities() const {
    std::uint64_t declared = 0;
    if (py::hasattr(obj_, "CAPABILITIES")) {
      declared = obj_.attr("CAPABILITIES").cast<std::uint64_t>();
    }
    if ((declared & (fuse::kCapabilityResidual | fuse::kCapabilityLinearization)) != 0) {
      FUSE_THROW(
          fuse::InvalidArgument,
          "component '{}' declares CAPABILITIES including kCapabilityResidual and/or "
          "kCapabilityLinearization, but pfuse does not expose Component.implicit() to a "
          "Python subclass in v1 (docs/dev/python-surface.md, 'Components implemented in "
          "Python') -- there is no fuse::ImplicitComponent behind this component for a "
          "Newton-solved region to call back into. Use a native (C++) component for a member "
          "of a Newton-solved implicit coupling group; a Python component may still be a "
          "member of a Picard/Aitken/IQN-ILS-accelerated one.",
          name_);
    }
    return declared | (rollback_ ? fuse::kCapabilityRollback : std::uint64_t{0});
  }

  template <class F>
  fuse::Result<void> call(const char* method, F&& body) {
    py::gil_scoped_acquire gil;
    try {
      py::object fn = obj_.attr(method);
      body(fn);
      return fuse::Result<void>::success();
    } catch (py::error_already_set& e) {
      pending_ = e.value();
      return fuse::Failure{"E-PY-CALLBACK",
                          "component '" + name_ + "'." + method + "() raised: " + e.what()};
    } catch (const std::exception& e) {
      return fuse::Failure{"E-PY-CALLBACK",
                          "component '" + name_ + "'." + method + "() failed: " + e.what()};
    } catch (...) {
      return fuse::Failure{"E-PY-CALLBACK",
                          "component '" + name_ + "'." + method + "() threw a non-std exception"};
    }
  }

  py::object obj_;
  std::string name_;
  PyFieldRegistry registry_;
  std::shared_ptr<KeepAlive> keep_;
  py::object pending_;
  bool has_save_ = false;
  bool has_restore_ = false;
  bool rollback_ = false;
  std::uint64_t capabilities_ = 0;
};

}  // namespace pfuse_py

namespace pfuse_py {

/// fuse.Session: owns ONE copy of the graph and everything derived from it (the copy
/// matters -- fuse::Runner holds const CouplingGraph& for its whole life, so a session that
/// aliased the caller's Python object would dangle the moment that object was rebound; same
/// reasoning as upstream's own soma.Session).
class Session {
 public:
  explicit Session(const fuse::CouplingGraph& graph) {
    graph_obj_ = py::cast(fuse::CouplingGraph(graph));  // an explicit copy -- see class comment
    registry_obj_ = py::cast(PyFieldRegistry{});
    registry().set_graph(&this->graph());
  }

  fuse::CouplingGraph& graph() { return graph_obj_.cast<fuse::CouplingGraph&>(); }
  PyFieldRegistry& registry() { return registry_obj_.cast<PyFieldRegistry&>(); }
  [[nodiscard]] py::object graph_object() const { return graph_obj_; }

  void add_component(py::object obj) {
    if (initialized_) {
      FUSE_THROW(fuse::InvalidArgument, "Session.add_component: the session is already initialized");
    }
    if (!py::isinstance<PyComponentBase>(obj)) {
      FUSE_THROW(fuse::InvalidArgument,
                "Session.add_component: expected a pfuse.Component subclass instance, got {}",
                py::str(py::type::of(obj).attr("__name__")).cast<std::string>());
    }
    const std::string name = obj.cast<PyComponentBase&>().name();
    if (!graph().find_component(name).has_value()) {
      FUSE_THROW(fuse::InvalidArgument,
                "Session.add_component: the graph declares no component named '{}'", name);
    }
    auto owned = std::make_unique<ForwardingComponent>(std::move(obj), name, registry());
    ForwardingComponent* raw = owned.get();
    try {
      components_.add(std::move(owned));
    } catch (const std::invalid_argument& e) {
      FUSE_THROW(fuse::InvalidArgument, "Session.add_component: {}", e.what());
    }
    forwarders_.push_back(raw);
  }

  void initialize(const fuse::RunnerConfig& config) {
    if (initialized_) FUSE_THROW(fuse::InvalidArgument, "Session.initialize: already initialized");
    fuse::CouplingGraph& g = graph();

    for (std::uint32_t i = 0; i < g.mesh_count(); ++i) {
      const auto r = backend_.load(g.mesh(fuse::MeshId{i}));
      if (!r.ok()) pfuse_py::raise_for(r);
    }

    fuse::LoweringResult lowered{fuse::OperatorDAG{}, fuse::Diagnostics{}};
    fuse::ScheduleResult scheduled;
    fuse::Diagnostics validation;
    {
      py::gil_scoped_release release;
      validation = fuse::validate(g);
      if (!validation.has_errors()) {
        lowered = fuse::lower(g);
        if (lowered.ok()) {
          const fuse::ClockSet clocks{g};
          scheduled = fuse::schedule(g, lowered.dag, clocks,
                                     fuse::make_region_sizes(g, lowered.dag, backend_));
        }
      }
    }
    if (validation.has_errors()) {
      raise_validation_failure("Session.initialize: the graph does not validate",
                               py::cast(validation));
    }
    if (!lowered.ok()) {
      raise_validation_failure("Session.initialize: the graph does not lower",
                               py::cast(lowered.diags));
    }
    if (!scheduled.ok()) {
      raise_validation_failure("Session.initialize: the graph does not schedule",
                               py::cast(scheduled.diags));
    }
    dag_ = std::move(lowered.dag);
    plan_ = std::move(scheduled.plan);

    // Component::register_fields(): calls into Python, so the GIL is held; a raising
    // component is re-raised verbatim below via rethrow_pending().
    for (fuse::Component& c : components_) {
      const auto r = c.register_fields(registry().raw(), g);
      if (!r.ok()) {
        rethrow_pending();
        pfuse_py::raise_for(r);
      }
    }

    runner_ = std::make_unique<fuse::Runner>(g, dag_, plan_, backend_);
    fuse::Result<void> r = fuse::Result<void>::success();
    {
      py::gil_scoped_release release;
      r = runner_->initialize(components_, registry().raw(), config);
    }
    if (!r.ok()) {
      runner_.reset();
      rethrow_pending();
      pfuse_py::raise_for(r);
    }
    initialized_ = true;
  }

  void run_period() {
    require_initialized("run_period");
    fuse::Result<void> r = fuse::Result<void>::success();
    {
      py::gil_scoped_release release;
      r = runner_->run_period(components_);
    }
    if (!r.ok()) {
      rethrow_pending();
      pfuse_py::raise_for(r);
    }
  }

  void run(std::int64_t periods) {
    require_initialized("run");
    fuse::Result<void> r = fuse::Result<void>::success();
    {
      py::gil_scoped_release release;
      r = runner_->run(components_, periods);
    }
    if (!r.ok()) {
      rethrow_pending();
      pfuse_py::raise_for(r);
    }
  }

  [[nodiscard]] const fuse::RunStats& stats() const {
    require_initialized_const("stats");
    return runner_->stats();
  }
  [[nodiscard]] const fuse::ExecutionPlan& plan() const { return plan_; }
  [[nodiscard]] const std::vector<fuse::IterationStats>& iteration_stats() const {
    require_initialized_const("iteration_stats");
    return runner_->iteration_stats();
  }

 private:
  void require_initialized(const char* what) {
    if (!initialized_) FUSE_THROW(fuse::InvalidArgument, "Session.{}: not yet initialized", what);
  }
  void require_initialized_const(const char* what) const {
    if (!initialized_) FUSE_THROW(fuse::InvalidArgument, "Session.{}: not yet initialized", what);
  }

  void rethrow_pending() {
    for (ForwardingComponent* f : forwarders_) {
      if (f->has_pending()) f->rethrow_pending();
    }
  }

  fuse::SerialMeshBackend backend_;
  py::object graph_obj_;
  py::object registry_obj_;
  fuse::OperatorDAG dag_;
  fuse::ExecutionPlan plan_;
  fuse::ComponentSet components_;
  std::vector<ForwardingComponent*> forwarders_;
  std::unique_ptr<fuse::Runner> runner_;
  bool initialized_ = false;
};

void bind_component(py::module_& m) {
  // fuse::k* bit constants (fuse/model/capabilities.hpp), for a Python Component subclass
  // that wants to declare CAPABILITIES explicitly (docs/dev/python-surface.md, "Components
  // implemented in Python"). CAP_RESIDUAL/CAP_LINEARIZATION are exposed so the REFUSAL that
  // claiming them triggers (ForwardingComponent::compute_capabilities(), above) has
  // something concrete to construct from -- not so a Python author can set them and
  // succeed.
  m.attr("CAP_ROLLBACK") = py::int_(fuse::kCapabilityRollback);
  m.attr("CAP_HIDDEN_STATE") = py::int_(fuse::kCapabilityHiddenState);
  m.attr("CAP_RESIDUAL") = py::int_(fuse::kCapabilityResidual);
  m.attr("CAP_LINEARIZATION") = py::int_(fuse::kCapabilityLinearization);

  py::class_<fuse::StepContext>(m, "StepContext")
      .def_readonly("t_begin", &fuse::StepContext::t_begin)
      .def_readonly("t_end", &fuse::StepContext::t_end)
      .def_readonly("step", &fuse::StepContext::step)
      .def_readonly("iteration", &fuse::StepContext::iteration)
      .def("dt", &fuse::StepContext::dt)
      .def_property_readonly("valid", &fuse::StepContext::valid)
      .def("__repr__", [](const fuse::StepContext& c) {
        return "StepContext(t_begin=" + c.t_begin.to_string() + ", t_end=" +
               c.t_end.to_string() + ", step=" + std::to_string(c.step) + ")";
      });

  py::class_<PyComponentBase>(m, "Component",
      "The Python-facing peer-component base class (docs/dev/python-surface.md).")
      .def(py::init<std::string>(), py::arg("name"))
      .def_property_readonly("name", &PyComponentBase::name)
      .def("register_fields",
          [](PyComponentBase&, py::object, py::object) -> void {
            throw py::type_error(
                "Component.register_fields() must be overridden by the subclass; the base "
                "class refuses to pretend it has storage to bind.");
          })
      .def("advance",
          [](PyComponentBase&, py::object) -> void {
            throw py::type_error(
                "Component.advance() must be overridden by the subclass; the base class "
                "refuses to silently no-op a coupling step.");
          })
      .def("__repr__", [](const PyComponentBase& c) {
        return "Component('" + c.name() + "')";
      });

  py::class_<fuse::RunnerConfig>(m, "RunnerConfig",
      "fuse::RunnerConfig. base_communicator/force_this_process/transport_setup are not "
      "bound (MPI is out of scope for pfuse v1 -- docs/dev/python-surface.md).")
      .def(py::init([](std::string transport, bool check_conservation,
                       double conservation_tolerance, bool abort_on_violation,
                       bool abort_on_nonconvergence, bool force_distributed_nodal_remap) {
             fuse::RunnerConfig c{};
             c.transport = std::move(transport);
             c.check_conservation = check_conservation;
             c.conservation_tolerance = conservation_tolerance;
             c.abort_on_violation = abort_on_violation;
             c.abort_on_nonconvergence = abort_on_nonconvergence;
             c.force_distributed_nodal_remap = force_distributed_nodal_remap;
             return c;
           }),
           py::arg("transport") = "copy", py::arg("check_conservation") = true,
           py::arg("conservation_tolerance") = 1e-12, py::arg("abort_on_violation") = false,
           py::arg("abort_on_nonconvergence") = false,
           py::arg("force_distributed_nodal_remap") = false)
      .def_readwrite("transport", &fuse::RunnerConfig::transport)
      .def_readwrite("check_conservation", &fuse::RunnerConfig::check_conservation)
      .def_readwrite("conservation_tolerance", &fuse::RunnerConfig::conservation_tolerance)
      .def_readwrite("abort_on_violation", &fuse::RunnerConfig::abort_on_violation)
      .def_readwrite("abort_on_nonconvergence", &fuse::RunnerConfig::abort_on_nonconvergence)
      .def_readwrite("force_distributed_nodal_remap",
                     &fuse::RunnerConfig::force_distributed_nodal_remap);

  py::class_<fuse::RunStats>(m, "RunStats")
      .def_readonly("periods", &fuse::RunStats::periods)
      .def_readonly("instructions_executed", &fuse::RunStats::instructions_executed)
      .def_readonly("conservation_checks", &fuse::RunStats::conservation_checks)
      .def_readonly("conservation_violations", &fuse::RunStats::conservation_violations)
      .def_readonly("worst_conservation_error", &fuse::RunStats::worst_conservation_error)
      .def_readonly("total_iterations", &fuse::RunStats::total_iterations)
      .def_readonly("windows_not_converged", &fuse::RunStats::windows_not_converged)
      .def_readonly("remap_weight_builds", &fuse::RunStats::remap_weight_builds)
      .def_readonly("remap_weight_nonzeros", &fuse::RunStats::remap_weight_nonzeros)
      .def_readonly("checkpoint_verifications", &fuse::RunStats::checkpoint_verifications)
      .def_readonly("epochs_reconciled", &fuse::RunStats::epochs_reconciled)
      .def("__repr__", [](const fuse::RunStats& s) {
        return "RunStats(periods=" + std::to_string(s.periods) + ", total_iterations=" +
               std::to_string(s.total_iterations) + ")";
      });

  py::class_<fuse::IterationStats>(m, "IterationStats")
      .def_readonly("iterations", &fuse::IterationStats::iterations)
      .def_readonly("converged", &fuse::IterationStats::converged)
      .def_readonly("residual_history", &fuse::IterationStats::residual_history);

  py::class_<Session>(m, "Session",
      "Owns one copy of the graph plus everything derived from it, and drives fuse::Runner "
      "(docs/dev/python-surface.md).")
      .def(py::init<const fuse::CouplingGraph&>(), py::arg("graph"))
      .def("add_component", &Session::add_component, py::arg("component"))
      .def("initialize",
          [](Session& s, std::optional<fuse::RunnerConfig> cfg) {
            s.initialize(cfg.value_or(fuse::RunnerConfig{}));
          },
          py::arg("config") = py::none())
      .def("run_period", &Session::run_period)
      .def("run", &Session::run, py::arg("periods"))
      .def_property_readonly("graph", &Session::graph_object)
      .def_property_readonly("stats", &Session::stats, py::return_value_policy::reference_internal)
      .def_property_readonly("plan", &Session::plan, py::return_value_policy::reference_internal)
      .def_property_readonly("iteration_stats", &Session::iteration_stats,
                             py::return_value_policy::reference_internal);
}

}  // namespace pfuse_py
