# Python bindings (`pfuse`)

:::{note}
**Built.** Stage A (2026-10-12) specified this surface before `python/` existed. Stage B
(2026-10-12) built `python/` under `FUSE_ENABLE_PYTHON` against that specification, with three
scope decisions the owner made between the stages (flat error hierarchy, native-only Newton
confirmed, the `dn_heat` driver reproduced by Python reimplementation rather than a test-only
extension — see the sections below, each updated in place rather than left as the Stage A
proposal). Every acceptance test cited on this page is measured to pass against the built
`_pfuse` extension, not merely designed to.
:::

## Scope and binding choice

Two bindings could expose FUSE to Python: a `pybind11` module over the C++ model and runtime
directly, or a `cffi`/`ctypes` shim over the flat C ABI (`fuse_coupler_abi()`, the same table
`cfuse.h` names). The repository owner chose `pybind11` over the C++ model and `fuse::Runner`.
A `cffi` shim over the ABI is an explicitly later, separate decision — not addressed here.

This means `pfuse` binds `fuse::CouplingGraph`, `fuse::Component`, `fuse::Runner` and friends
directly (spec mode 1: in-process peer components), **not** `fuse_coupler_abi()`. Upstream's
`soma.Context` — a one-method-per-entry-point mirror of `soma_c.h` — has no `pfuse` counterpart
for that reason; see the evidence table below (row `Context`, classification **drop**). An
external driver that wants the ABI surface from Python should use the future `cffi` shim, not
`pfuse`.

Package name: **`pfuse`**, both the PyPI distribution name and the `import` name. Ruled
2026-09-17 in `KNOWN_ISSUES.md` A3 and `docs/dev/porting.md`; re-confirmed for this track with
the climate-vis interpreter's pip (`pip index versions pfuse` → "No matching distribution
found", the same signal that cleared the name the first time, checked here against both a
known-existing package and a known-nonexistent control so the absence is informative and not a
network artefact).

## Module layout

Stage A proposed one extension submodule per `fuse::` layer (ten files). Built, that split added
file boundaries with nothing layer-specific to enforce across them — every binding file links
the same five `fuse::` libraries and none is ever compiled or used without the others — so Stage
B consolidated to five source files, named by what they bind rather than by a layer that turned
out not to gate anything:

```
python/
  CMakeLists.txt              # FUSE_ENABLE_PYTHON gate; find_package(pybind11); pybind11_add_module;
                               # ctest registration (see Build/CMake integration, below)
  src/
    module.cpp                 # PYBIND11_MODULE(_pfuse, m); registers every binding function below;
                                # the exception-translator table (see Errors, below) and the
                                # Kokkos::initialize()/std::atexit(Kokkos::finalize) pair live here
    errors.hpp                 # raise_result_failure()/raise_for<T>()/raise_validation_failure(),
                                # shared by every .cpp file below so there is one place that decides
                                # which fuse::Result failure becomes which Python exception
    base_bindings.cpp          # Id<Tag> family, Rational, Units, the base-layer enums, Diagnostic,
                                # Diagnostics, pfuse.rational(), pfuse.version(), pfuse.units
    model_bindings.cpp         # everything under fuse/model/: MeshDecl, Region, RankSet,
                                # ComponentDecl, CouplerDecl, Window, FieldSpec, RemapFlags, Channel/
                                # ChannelBuilder, ConvergenceMeasure/Criterion, AccelConfig,
                                # NewtonConfig, IterationBuilder, CouplingGraph (incl. to_bytes/
                                # from_bytes/from_yaml/validate), plus OperatorDAG/lower()/
                                # dump_text()/dump_graphviz() (fuse/ir/) and ExecutionPlan/schedule()
                                # (fuse/schedule/) and SerialMeshBackend (fuse/backend/) — none of
                                # these three layers' Python surface is larger than a handful of
                                # functions, and each depends on CouplingGraph types already bound
                                # in this file, so splitting them out gained nothing
    field_registry.hpp         # ArrayView, inspect_array(), numpy_view(), KeepAlive, PyFieldRegistry
                                # — header-only, included by both files below
    field_bindings.cpp         # fuse.FieldRegistry, fuse.MemorySpace
    component_bindings.cpp     # PyComponentBase, ForwardingComponent, Session (owns graph copy +
                                # registry + Runner, mirrors upstream's soma.Session), StepContext,
                                # RunnerConfig, RunStats, IterationStats, the CAP_* module constants
  pfuse/
    __init__.py                 # from _pfuse import *
  tests/
    test_buffers.py
    test_graph.py
    test_yaml.py
    test_runtime.py
    test_newton.py
    test_dn_heat_rows.py
    test_analytic_problem.py
    dn_heat_components.py       # Python DirichletSlab/NeumannSlab reimplementation (see "The
                                 # dn_heat driver test", below) -- shared by test_dn_heat_rows.py
    golden/
      dn_heat_full_output.txt       # all 12 examples/dn_heat/main.cpp rows, captured by actually
                                     # running the built example binary
      newton_descriptor_reference.bin  # serialized native-built Newton graph, compared byte-for-
                                        # byte against the Python-authored equivalent
```

`descriptor_bindings.cpp`/`yaml_bindings.cpp`/`ir_bindings.cpp`/`schedule_bindings.cpp`/
`backend_bindings.cpp`/`runtime_bindings.cpp` from the Stage A sketch do not exist as separate
files; their functions are in `model_bindings.cpp` and `component_bindings.cpp` as described
above. The compiled extension is still named `_pfuse`, and `pfuse/__init__.py` is `from _pfuse
import *` with no pure-Python additions — none of the acceptance tests ended up needing a
Python-level convenience the compiled module could not provide directly (`pfuse.rational()` and
`pfuse.units` are themselves bindings, in `base_bindings.cpp`, not `__init__.py` code).

**Build integration, as built.** `FUSE_ENABLE_PYTHON`, default `OFF`. When `ON`:
`find_package(pybind11 REQUIRED)`, `pybind11_add_module(_pfuse MODULE ${sources})`, linking
`fuse::model`, `fuse::runtime`, `fuse::backend`, `fuse::ir`, `fuse::schedule`. No MPI link:
`pfuse` v1 drives only `fuse::SerialMeshBackend` and leaves `RunnerConfig::base_communicator`
unset (see "What this does not expose", below; `KNOWN_ISSUES.md` D37).

**A `find_package` gotcha measured in this sandbox, not in general.** `pybind11` 3.1's own
`pybind11Tools.cmake` calls `find_package(Python ...)` — the un-suffixed name, not `Python3` —
to locate the interpreter/headers/libraries it builds against. A `VIRTUAL_ENV` environment
variable pointing at an unrelated virtualenv (left set by this sandbox's own Python tooling)
hijacks that search even when `-DPython3_EXECUTABLE=...` is passed, because CMake's `FindPython`
module consults `VIRTUAL_ENV` before `Python3_EXECUTABLE`. The fix applied to reconfigure
correctly: `unset VIRTUAL_ENV`, put the target interpreter's `bin/` first on `PATH`, and pass
**both** `-DPython_EXECUTABLE=`/`-DPython_ROOT_DIR=` and `-DPython3_EXECUTABLE=`/
`-DPython3_ROOT_DIR=` plus `-DPYBIND11_FINDPYTHON=ON` — belt and suspenders because
`pybind11Tools.cmake` and this project's own `find_package(Python3 COMPONENTS Interpreter
REQUIRED)` (for `ctest` registration, below) each resolve the interpreter through a different
CMake variable name.

**Kokkos must be initialized before `fuse::Runner` is touched.** Every native `fuse::` example
wraps `main()` in a `Kokkos::ScopeGuard`; a Python process has no equivalent single entry point
to wrap, so `module.cpp`'s `PYBIND11_MODULE` body calls `Kokkos::initialize()` directly (guarded
by `Kokkos::is_initialized()`/`is_finalized()`, so importing `pfuse` twice in the same
interpreter is harmless) and registers `std::atexit([]{ if (Kokkos::is_initialized())
Kokkos::finalize(); })` rather than ever constructing a `Kokkos::ScopeGuard` — its destructor
calls `Kokkos::finalize()` unconditionally, which is the wrong lifetime for an object with no
defined destruction point in a Python process. Omitting the `atexit` registration was tried and
measured: the process still exits 0, but prints `libc++abi: terminating due to uncaught
exception ... mutex lock failed: Invalid argument` at interpreter shutdown, because Kokkos' own
static destructors run against an already-torn-down runtime. The `atexit` finalize removes the
message entirely.

**Layering.** No `pybind11` type (`py::object`, `py::array`, …) may appear in any header under
`include/fuse/` — `tools/check-layering.py`'s grep-based gate enforces this project-wide, and
every `pfuse` binding file lives under `python/src/`, never under `include/fuse/`, so the gate
sees nothing to flag. `fuse::Component`'s existing virtual interface (`register_fields`,
`advance`, `save_state`, `restore_state`, `capabilities`, `implicit`) is wide enough that the
Python binding's `ForwardingComponent` never needs a new C++-side hook — exactly the point of a
pure-virtual base with no `pybind11` in sight.

## Per-file evidence: upstream `soma` Python bindings mapped onto `fuse`

Source: `upstream/soma/python/src/bindings.cpp` (2064 lines, 155 `.def(...)` call sites across
26 `py::class_<...>` and 10 `py::enum_<...>` declarations — counted directly, `grep -c
'\.def('`/`grep -n 'py::class_<\|py::enum_<'`, not estimated) and its three test files
(`tests/test_buffers.py` 217 lines, `test_graph.py` 319 lines, `test_runtime.py` 592 lines).
These bind the **old `soma::` API** (`docs/dev/porting.md`'s naming map) and cannot be reused
as-is; every row below is a judgement call made by reading both the upstream binding and the
current `fuse::` header it would have to bind against instead.

Classification: **carry over** (same shape, same tests, different namespace) · **rename**
(same concept, renamed per `docs/dev/porting.md`) · **new** (`fuse` has it, `soma` had no
equivalent) · **drop** (deliberately not bound, with the reason).

| Upstream (`soma::`) | `fuse::` equivalent | Header | Classification |
|---|---|---|---|
| `Id<Tag>` + `ComponentId/FieldId/MeshId/ChannelId/RegionId` | `fuse::Id<Tag>` + same aliases, plus `IterationRegionIdx` | `base/ids.hpp` | carry over |
| `Rational` (float constructor rejected; `str`/`Fraction` coercion) | `fuse::Rational` — identical API (`num/den/is_zero/is_integer/to_double/to_string`, full comparison and arithmetic operators) | `base/rational.hpp` | carry over |
| `Units` (dimension exponents only, by the binding's `py::init([](length, mass, time, current, ...))`) | `fuse::Units` — dimension exponents **plus** an affine `scale`/`offset` pair (`Rational`, exact) so e.g. degrees Celsius is representable and dimensional checking (`same_dimensions`) is explicitly separated from affine equality (`operator==`) | `base/units.hpp` | carry over + **new** (`scale`, `offset`, `is_linear()`) |
| `MeshDecl` | `fuse::MeshDecl` | `model/region.hpp` | carry over |
| `Region` | `fuse::Region` | `model/region.hpp` | carry over |
| `RankSet` (`{first, count}` only) | `fuse::RankSet` — `{first, count}` **plus** an arbitrary sorted/de-duplicated rank list (`RankSet(std::vector<int>)`), `overlaps`, `same_as`, `is_contiguous`, `index_of`, `min_rank`/`max_rank` | `model/component.hpp` | carry over + **new** (arbitrary rank lists; the richer query set spec's processor-set overlap reasoning needs, `docs/user/glossary.md`'s "disjoint / partially overlapping / coincident processor sets") |
| `ComponentDecl` | `fuse::ComponentDecl` — plus `kind` (`ComponentKind::Physics`/`Coupler`) and `launch` (`LaunchSpec`, MPMD) | `model/component.hpp` | carry over + **new** (`kind`, `launch`) |
| *(none — no coupler concept)* | `fuse::CouplerDecl`, `CouplingGraph::add_coupler()` — a placement with no fields and no clock that a channel's `RemapSpec::remap_on` can name (the CESM/E3SM hub-and-spoke pattern, `docs/user/glossary.md` "coupler processor set") | `model/component.hpp` | **new** |
| `FieldSpec` | `fuse::FieldSpec` — plus `hdf5_path` (LowFive track) | `model/field_spec.hpp` | carry over + new field, **not bound** in v1 (see "What this does not expose") |
| `Window` (+ `Kind`) | `fuse::Window` (+ `Kind`) — identical shape (`CouplingPeriod`/`FromComponent`/`Explicit`) | `model/component.hpp` | carry over |
| `Channel`/`ChannelBuilder` (`accumulate`/`remap`/`remap_method`/`conservative`/`spacetime`/`relax`/`extrapolate`/`lag`/`allow_time_level_mismatch`) | `fuse::Channel`/`ChannelBuilder` — the same fluent setters **plus** `remap_on(ComponentId)` (route a transfer's remap through a separately-placed coupler) | `model/channel.hpp` | carry over + **new** (`remap_on`) |
| `CouplingGraph` (`add_mesh/region/component/field/channel/iteration_region`, accessors, `to_yaml`, `__copy__`) | `fuse::CouplingGraph` — same accessors plus `add_coupler()` | `model/coupling_graph.hpp` | carry over + new (`add_coupler`) |
| *(`IterationRegionDecl`: channels + a convergence criterion only — no `.accelerate()`/`.newton()` on its builder)* | `fuse::IterationBuilder` — `until()`, `max_iterations()`, `accelerate(AccelConfig)`, `newton(NewtonConfig)`, `initial_relaxation()`, `name()`, `requires_residual()`, `requires_linearization()` | `model/coupling_graph.hpp`, `model/iteration.hpp` | **new** (the whole convergence-acceleration and Newton-selection vocabulary; `soma`'s iteration region had no accelerator and no Newton path at all) |
| *(none)* | `fuse::ConvergenceMeasure`/`ConvergenceCriterion`, `fuse::Converged::Relative/Absolute/AbsoluteOrRelative/ResidualRelative`, `operator&` composition | `model/iteration.hpp` | **new** |
| *(none)* | `fuse::AccelConfig` (`AccelKind::None/Constant/Aitken/Secant/Steffensen/IQN_ILS/IQN_IMVJ/Newton`, `FilterKind::None/QR1/QR2/QR3`, `reuse`, `initial_relaxation`, `omega`) | `model/iteration.hpp` | **new** |
| *(none)* | `fuse::NewtonConfig` — 28 fields: line search (`LineSearchKind::BT/CP/Basic` + its tuning constants), Eisenstat-Walker forcing (`EisenstatWalkerChoice::Constant/Choice1/Choice2` + 6 tuning constants), stopping (`atol/rtol/stol/divergence_tolerance/max_nonlinear_iterations/max_function_evaluations`), GMRES (`gmres_restart/gmres_max_iterations`), `preconditioner_lag`, `use_block_jacobi_preconditioner`, `substrate` (`NewtonSubstrate::Native/Petsc`), `norm` (`NewtonNorm::Plain/MeasureWeighted` — the "measure-weighted" norm named in this task's brief), `residual_path` (`NewtonResidualPath::Advance/Direct`) | `model/iteration.hpp` | **new** |
| *(none)* | `fuse::ImplicitComponent` (`evaluate_residual`), `fuse::LinearizationToken` (`apply_jacobian`/`apply_jacobian_transpose`/`apply_preconditioner`/`rows`/`cols`/`signature`), `fuse::verify_linearization_shape()` | `runtime/component.hpp` | **new**, deliberately **not exposed** to a Python `Component` subclass in v1 (see "What this does not expose") |
| `Diagnostic`/`Diagnostics` | `fuse::Diagnostic`/`Diagnostics` (`Severity::Warning/Error`) | `base/result.hpp` | carry over |
| `soma::validate(graph)` | `fuse::validate(graph, policy={})` | `model/validate.hpp` | carry over (fuse's `ConnectPolicy` default-constructs; not bound as a tunable in v1 — every graph the acceptance tests build uses the default policy) |
| `OperatorDAG`, `LoweringResult`, `dump_ir`, `dump_graphviz` | `fuse::OperatorDAG`, `fuse::lower()`, `fuse::dump_text()`, `fuse::dump_graphviz()` | `ir/lowering.hpp`, `ir/dump.hpp` | carry over (`dump_ir` renamed `dump_text` — matches the C++ name already) |
| `ExecutionPlan`, `ScheduleResult`, `soma.build_plan()` | `fuse::ExecutionPlan`, `fuse::schedule()` | `schedule/plan.hpp`, `schedule/clock.hpp` | carry over |
| `SerialMeshBackend` | `fuse::SerialMeshBackend` | `backend/serial_backend.hpp` | carry over |
| descriptor `to_descriptor()`/`from_descriptor()`/`structurally_equal()` | `fuse::serialize()`/`fuse::deserialize()`/`fuse::structurally_equal()` — free functions in C++; bound as `graph.to_bytes()`/`pfuse.from_bytes(blob)` to keep upstream's ergonomic method-on-graph spelling | `model/descriptor.hpp` | carry over (renamed `to_descriptor`→`to_bytes`/`from_descriptor`→`from_bytes`: `fuse`'s own vocabulary calls this "the descriptor" as a noun but never verbs it `descriptor()`; `to_bytes`/`from_bytes` says what the method returns) |
| `to_yaml`/`from_yaml`/`load_yaml` | `fuse::to_yaml()`/`fuse::from_yaml()`/`fuse::load_yaml_file()` | `model/yaml_io.hpp` | carry over (`load_yaml`→`load_yaml_file`, matches the C++ name) |
| `PyFieldRegistry` (numpy-aware `bind_numpy`/`get_numpy`/`is_bound`/`bound_count`) | needs a fresh implementation over `fuse::FieldRegistry`/`FieldRef`/`Layout` — **the buffer-validation and lifetime logic transplants verbatim** (same checks, same tests, see "The numpy buffer contract" below); only the C++ types underneath differ | `field/registry.hpp`, `field/field_ref.hpp` | carry over (design and tests), new implementation |
| `StepContext` (`t_begin/t_end/step/iteration`, `.valid()`) | `fuse::StepContext` — identical fields plus `.dt()` | `runtime/component.hpp` | carry over |
| `PyComponent`/`ForwardingComponent` (`register_fields`/`advance`/`save_state`/`restore_state`, GIL reacquire + exception stash-and-rethrow) | needs a fresh `ForwardingComponent : fuse::Component` — **the GIL and exception-forwarding design transplants directly**, `fuse::Component`'s four core virtuals have the identical shape; three members have no upstream counterpart (next row) | `runtime/component.hpp` | carry over (design and tests) for the four core virtuals |
| *(`PyComponent` had no capability bits — only an inferred `supports_rollback`)* | `fuse::Component::capabilities()` — one `std::uint64_t` bitmask encoding rollback, `DeterminismLevel` (`Undeclared`/`RunToRun`/`Bitwise`/`Nondeterministic`) and hidden-state, replacing `soma`'s single inferred bool | `model/capabilities.hpp`, `runtime/component.hpp` | **new**; `ForwardingComponent` infers the rollback bit from `hasattr(save_state)`/`hasattr(restore_state)` exactly as upstream did, and leaves determinism **undeclared** and hidden-state **unset** unless the Python subclass states otherwise (see "Components implemented in Python", below) |
| *(none)* | `fuse::Component::is_externally_driven()` — true only for the participant-ABI's `ExternalStepComponent`; no Python subclass is ever driven that way | `runtime/component.hpp` | **new**, not overridable from Python — `ForwardingComponent` returns `false` unconditionally |
| *(none)* | `fuse::Component::implicit()` → `ImplicitComponent*` — the Newton Jacobian seam | `runtime/component.hpp` | **new**, deliberately **not exposed** to Python `Component` subclasses in v1 (see below) |
| `RunnerConfig` (`transport`, conservation fields) | `fuse::RunnerConfig` — same fields, plus `force_distributed_nodal_remap`, `base_communicator`, `force_this_process`, `transport_setup` | `runtime/runner.hpp` | carry over the conservation/transport fields; the four MPI/transport-setup fields are **not bound** in v1 (see below) |
| `RunStats` | `fuse::RunStats` — same accounting plus `total_iterations`, `windows_not_converged` | `runtime/runner.hpp` | carry over + new fields |
| `Session` (owns a graph copy + registry + `Runner`; `add_component`/`initialize`/`run_period`/`run`) | needs a fresh `Session` wrapper over `fuse::Runner` + `fuse::ComponentSet` — **the ownership pattern (the session holds its own copy of the graph, because `Runner` holds a `const CouplingGraph&` for its whole life) transplants directly** | `runtime/runner.hpp`, `model/coupling_graph.hpp` | carry over (design) |
| `Context` (one Python method per `soma_c.h` entry point, driven with `begin_step`/`end_step`) | *(none bound)* — `fuse`'s C ABI is `fuse_coupler_abi()`, the `cfuse` binding target, not `pfuse`'s | `(soma_c.h has no fuse analogue bound here)` | **drop** — the owner chose `pybind11` over the C++ model and `Runner`, explicitly not the ABI; an external driver wanting the ABI surface from Python is a `cffi` shim's job, a separate, later decision |
| 5-way `SOMA_ERR_*` → 5 Python exception classes | `fuse::Result<T>`'s free-form string codes (dozens, e.g. `E-FIELD-REBIND`, `E-UNITS`, `E-RT-UNBOUND`) **and** the separate `fuse::Error` C++ exception hierarchy (`InvariantViolation`/`InvalidArgument`/`NullPointer`/`Refusal`/`NotImplemented`/`IoError`) | `base/result.hpp`, `base/error.hpp` | **new shape**, no 1:1 rename possible — see "Errors", below (the owner's ruling: a flat `pfuse.FuseError` plus six subclasses mirroring `fuse::Error`'s own tree, no triage of `Result` failure codes) |

### Why `soma`'s iteration-region and Newton vocabulary is "new," not "renamed"

Checked directly against the upstream binding rather than assumed: `CouplingGraph`'s bound
methods are `add_iteration_region`, `channel_at`, `iteration_region`, and the bound
`IterationRegionDecl` class has only a constructor and `__repr__` — no `accelerate`, no
`newton`, no `until`. `soma`'s implicit coupling groups (the glossary's term) could declare
*which channels* iterate together, but not *how* the fixed point is accelerated or solved; every
run converged by plain Picard sweeps. The entire `AccelConfig`/`NewtonConfig`/
`ConvergenceCriterion` vocabulary — Aitken, IQN-ILS, IQN-IMVJ, and the Newton-Krylov path this
task's desired final outputs ask for ("Window Newton running...") — is new in `fuse` relative to
what the upstream binding ever had to express.

## Errors: two sources, one Python hierarchy

`fuse` reports failure two ways, and both must cross the `pybind11` boundary as a Python
exception — never a return code a caller can forget to check (the project's own invariant I4,
`docs/dev/invariants.md`):

1. **`fuse::Result<T>`** — the explicit outcome of `validate()`, `lower()`, `schedule()`,
   `Session.initialize()`/`run()`, `Component::register_fields/advance/save_state/restore_state`,
   `FieldRegistry::bind/get`, `serialize`/`deserialize`, `to_yaml`/`from_yaml`. A failure is a
   `Failure{code, message}` with a free-form string code (`E-FIELD-REBIND`, `E-UNITS`,
   `E-RT-UNBOUND`, dozens more, growing as `fuse` grows). There is no enum to switch on: each
   binding call site must decide, by reading the C++ comment at that call site (exactly as
   upstream's `bindings.cpp` already does, case by case — `raise_state` for `E-FIELD-REBIND`
   because the comment calls it "a sequencing mistake," `raise_invalid` for everything else at
   that call site, `raise_notfound` for `get()` of an unbound field), which Python exception the
   code means.
2. **`fuse::Error`** (`FUSE_ASSERT`/`FUSE_THROW`/`FUSE_REFUSE`) — a C++ exception thrown across
   the call. Unregistered, `pybind11` would translate it into a generic `RuntimeError` built
   from `what()`, losing `.code()`/`.message()`/`.where()`/`.backtrace()`/`.rank()` — exactly the
   structured information upstream's own exception design exists to keep. `pfuse` installs a
   `py::register_exception_translator` in `module.cpp` that catches `const fuse::Error&` and
   re-raises the matching Python type below with `.code`, `.message` and `.report()` populated.

**The owner's ruling (2026-10-12): flat, not triaged.** Stage A proposed four behavioural
`Result`-failure classes (`InvalidArgument`/`StateError`/`NotFoundError`, plus the `fuse::Error`
side) as a default ergonomics/precision trade-off and left the choice open. Decided instead:
every `fuse::Result` failure, with no exception, becomes a **bare** `pfuse.FuseError` carrying
`.code` and `.message` only — no triage of the free-form `E-*` string into a behavioural
subclass, and no `StateError`/`NotFoundError`/`ValidationError` classes. This removes the
open-ended "which category does this new `E-*` code belong in" maintenance burden Stage A's
proposal would have created as `fuse` grows, at the cost of upstream's `except ValueError:`
ergonomics for a `Result` failure specifically (a thrown `fuse::Error` still subclasses the
matching builtin — see the table below).

**The six-class hierarchy, mirroring `fuse::Error`'s own C++ class tree exactly** (one Python
class per C++ exception type `fuse::Error` defines, no more, no fewer — every subclass derives
from both `pfuse.FuseError` and the closest Python builtin, so `except ValueError:` keeps
working for a caller who never imports `pfuse`'s own hierarchy):

| `pfuse` class | Bases | `fuse::Error` type |
|---|---|---|
| `pfuse.FuseError` | `RuntimeError` | (the common base; also what every bare `Result` failure raises) |
| `pfuse.InvariantViolation` | `FuseError` | `fuse::InvariantViolation` |
| `pfuse.InvalidArgument` | `FuseError, ValueError` | `fuse::InvalidArgument` |
| `pfuse.NullPointer` | `pfuse.InvalidArgument` | `fuse::NullPointer` — mirrors `error.hpp`'s own nesting, which documents `NullPointer` as "a specific `InvalidArgument`" |
| `pfuse.Refusal` | `FuseError, ValueError` | `fuse::Refusal` |
| `pfuse.NotImplementedFeature` | `FuseError, NotImplementedError` | `fuse::NotImplemented` |
| `pfuse.IoError` | `FuseError, OSError` | `fuse::IoError` |

`pfuse.NotImplementedFeature` keeps the `Feature` suffix from the Stage A proposal rather than
reusing `NotImplementedError`'s own name: `NotImplemented` is already a Python builtin
**singleton value** (not an exception), and shadowing that name in `pfuse`'s own namespace
would be a standing trap for a `from pfuse import *` caller.

**Binding-level precondition checks raise through the same path as the library itself.** A
numpy-buffer-contract violation (`field_registry.hpp`'s `inspect_array()`) or a "not found"
lookup (`find_component()` of an unknown name) is not a `fuse::Error` the C++ library threw —
it is a check the Python binding itself makes before calling into `fuse::`. Both raise via
`FUSE_THROW(fuse::InvalidArgument, ...)`, the same macro `fuse::` itself uses, so they cross the
exception translator by the identical path as a genuine C++-thrown error rather than through a
second, binding-only raising mechanism. A "not found" lookup is therefore `pfuse.
InvalidArgument`, not a dedicated lookup class — the fixed six-class list above has none, by the
same flat-hierarchy ruling.

**`Session.initialize()`'s `validate()`/`lower()`/`schedule()` failures keep their `Diagnostics`
payload without a seventh exception class.** These three calls return a `fuse::Diagnostics`
collection, not a `fuse::Error`; `Session.initialize()` raises a bare `pfuse.FuseError` for any
of them, with `.diagnostics` set to the bound `fuse::Diagnostics` object (`None` for every other
`FuseError` — a thrown `fuse::Error` or any other `Result` failure has no `Diagnostics` to
attach). This was measured, not assumed, while writing `test_newton.py`: the capability-gate
refusal raised by `fuse::Runner::initialize()` (`E-NEWTON-CAPABILITY`) is itself a plain
`Result<void>` failure, not a `Diagnostics`-carrying one, so `.diagnostics` reads `None` there
too and the refusal's text is in `.message` instead — a distinction a reader has to check per
call site, the same way the C++ API itself draws it.

**`pfuse.InvariantViolation` is not always seen.** `error.hpp`'s `OnError` policy
(`Throw`/`PrintBacktrace`/`AttachDebugger`/`Abort`) runs **before** the throw, and
`OnError::Abort` calls `std::abort()` unconditionally without ever throwing. `pfuse` does not
call `fuse::set_error_policy()` — a process embedding `pfuse` keeps whatever policy
`fuse::initialize()` or the environment (`FUSE_ON_ERROR=...`) already set. Concretely: with the
library default (`PrintBacktrace`, which throws after printing) a Python `try`/`except` works
normally; under `FUSE_ON_ERROR=abort` the interpreter terminates with no exception and no
chance to catch it, by the same design that makes `FUSE_ASSERT` "checked in every build" for a
C++ caller. This is a correctness property, not a gap: an `InvariantViolation` means the
library's own logic is wrong (`error.hpp`'s own description), and papering over an abort policy
so Python always gets a catchable exception would let a Python caller's `except Exception:
pass` hide exactly the class of bug the policy exists to surface loudly.

## The GIL

Released, with `py::gil_scoped_release`, around exactly the calls that do real work and touch
no Python object — the same list upstream releases around, transplanted onto `fuse`'s matching
entry points: `fuse::validate()`, `fuse::lower()`, `fuse::schedule()`, the portions of
`Session.initialize()` that call them plus `fuse::Runner::initialize()`, and `Session.run()`/
`run_period()`'s own instruction-stream walk. `Session.run()` is the one that calls back into
Python through `ForwardingComponent`, which reacquires with `py::gil_scoped_acquire` before
touching anything Python — exactly upstream's design, and necessary for the identical reason:
`fuse::Runner`'s hot loop may run with the GIL released, and a Python `Component`'s `advance()`
cannot run without it.

**Exceptions across the callback boundary**, mirroring `ForwardingComponent`'s existing
design exactly: each of `register_fields`/`advance`/`save_state`/`restore_state` is called
inside a `try` that catches `py::error_already_set` (whose constructor *fetches* the Python
error, clearing the interpreter's error indicator) and converts it to a `fuse::Result<void>`
failure so `fuse::Runner`'s own C++ error channel unwinds normally; the original exception
object is stashed and re-raised **verbatim** (same type, same traceback) once control returns
to the binding boundary (`Session.initialize()`/`run()`/`run_period()`), via
`rethrow_pending()`. A Python exception therefore never has to be re-expressed as a `fuse::Error`
and back — it round-trips as the exact object the user's code raised, which is what upstream's
own acceptance test (`test_an_exception_from_advance_reaches_the_caller_verbatim`) checks and
what `test_runtime.py` (below) checks again against `fuse::Runner`.

## The numpy buffer contract and the lifetime rule

Transplanted from upstream's `inspect_array()`/`numpy_view()` almost unchanged — the contract
is about what a `numpy.ndarray` must be to alias `fuse::FieldRef`-compatible storage, which does
not depend on which C++ library owns the `FieldRegistry` on the other side:

A `fuse::FieldRegistry.bind_numpy(field_id, array)` accepts **only** an array that is (1) a
real `numpy.ndarray` (not a list, not a `memoryview`, not a scalar); (2) `dtype == float64`,
native byte order, no casting; (3) C-contiguous (`a[::2]`, `.T`, any strided view refused); (4)
writeable; (5) 1-D `(count,)` for a scalar field or 2-D `(count, components)` — C-contiguous
2-D is interleaved (AoS), matching `fuse::Layout`'s `component_stride = 0` convention exactly.
Every one of upstream's `RejectsWhatWouldBeTheWrongMemory` tests is carried over onto
`fuse::FieldRegistry` unchanged (see `test_buffers.py`, below) — the underlying type changed,
the rejected inputs did not.

**The lifetime rule.** The registry retains a strong reference to every bound array (a
`KeepAlive` map, keyed by `FieldId`, exactly upstream's design) for as long as the registry — or,
for a component's own view during `register_fields()`, the owning `Session` — is alive. This is
what makes `reg.bind_numpy(f, np.zeros(4) + 1)` — a temporary with no other name — safe rather
than a dangling pointer: the registry's reference is what keeps the temporary alive, not the
caller's.

**The rule cannot be violated, not merely discouraged, and a test makes that failure mode
fire.** `numpy.ndarray.resize()` checks the array's reference count before reallocating
in-place, and refuses (`ValueError`) whenever it is not exactly 1 — `refcheck=True` is resize's
own default, needing no cooperation from the caller. The registry's `KeepAlive` reference is
exactly one such extra reference, so **every** attempt to grow or shrink a bound array in place
fails structurally, by numpy's own allocator, not by a check `pfuse` has to add. `test_buffers.py`
(carried over from upstream's `Lifetime.test_numpy_refuses_to_resize_a_bound_array`) proves this
with a **control case first**: it resizes an *unbound* array of the identical shape and asserts
that succeeds, so a numpy version that silently stopped doing the refcount check is caught as an
environment skip rather than passing a test that has quietly stopped testing anything, before
binding the same shape and asserting the identical call now raises.

## Components implemented in Python

A Python `Component` subclass (`pfuse.Component`, mirroring upstream's `PyComponent`) is
bridged into `fuse::Component` by a `ForwardingComponent` that holds a strong reference to the
Python object — never a `pybind11` trampoline sharing ownership with a `unique_ptr<Component>`,
for the identical reason upstream's own comment gives: `fuse::ComponentSet::add()` takes a
`unique_ptr<Component>`, and a trampoline whose real owner is the Python object is two owners
for one lifetime.

The four core virtuals (`register_fields`, `advance`, `save_state`, `restore_state`) transplant
directly from `ForwardingComponent` — same GIL policy, same exception stash-and-rethrow, same
shapes. Three members `soma::PyComponent` never had to address:

- **`capabilities()`.** `ForwardingComponent` infers the rollback bit exactly as upstream
  inferred `supports_rollback()` — `hasattr(obj, "save_state") and hasattr(obj, "restore_state")`
  — but `fuse::Component::capabilities()` is a *single* 64-bit answer also covering
  `DeterminismLevel` (bits 1–2) and hidden-state (bit 3), neither of which can be inferred from
  the object's shape. Built as proposed: determinism stays **undeclared** (triggers an iteration
  region's `W-ITER-DETERMINISM-UNDECLARED` warning, not a refusal — the honest default
  `fuse::Component`'s own base class documents) and hidden-state stays **unset**, unless the
  Python subclass sets a `CAPABILITIES` class attribute (an `int`, OR of `pfuse.CAP_ROLLBACK`/
  `pfuse.CAP_HIDDEN_STATE`/`pfuse.CAP_RESIDUAL`/`pfuse.CAP_LINEARIZATION`) that
  `ForwardingComponent::compute_capabilities()` reads and ORs with the inferred rollback bit.
  `CAP_RESIDUAL`/`CAP_LINEARIZATION` are bound for exactly one reason: so a Python component
  that claims either without being able to back it up is refused (next bullet), not so a Python
  component can usefully set them.
- **`is_externally_driven()`.** Always `false` for `ForwardingComponent` — a Python component is
  always an in-process peer, never a participant-ABI `ExternalStepComponent`; not an override
  point in v1.
- **`implicit()` → `ImplicitComponent*`.** **Not exposed to a Python `Component` subclass —
  confirmed by the owner (2026-10-12) as a v1 scope decision, not left open.** `ForwardingComponent
  ::implicit()` always returns `nullptr`. Reasons, stated rather than assumed: (1)
  `LinearizationToken::apply_jacobian()`/`apply_preconditioner()` are called from inside a Krylov
  inner loop — potentially hundreds of times per Newton step — and each call would reacquire the
  GIL and cross into Python, the kind of per-call overhead that turns a Newton solve into the
  slowest thing in the process; (2) `evaluate_residual()`'s purity contract (bit-identical output
  for repeated calls at the same point, no RNG, no unordered reduction, no mutation of anything
  `save_state()` would need) is harder to guarantee from Python than from C++, and nothing
  downstream checks it, only shape (`verify_linearization_shape()`). The consequence: a
  Newton-solved implicit coupling group **may be configured and run from a Python-authored
  graph**, but every component that is a *member* of that region must be native (C++); `pfuse`
  does not restrict which components a non-Newton region may contain.

**Two refusal paths cover the restriction, not one, and `test_newton.py` exercises both.**
(1) A Python component honest about its capabilities (no `CAPABILITIES` attribute claiming
`CAP_RESIDUAL`) added to a Newton region is refused by `fuse::Runner::initialize()`'s own
pre-existing `E-NEWTON-CAPABILITY` check (`src/runtime/checkpoint_support.cpp`) — this track did
not add a second check for that case; the existing one already covers it, measured by
`TestNewtonCapabilityGate`. That check's own message says "override `Component::implicit()` to
return it" — correct advice for a C++ author, impossible advice for a Python one, since `pfuse`
exposes no such override point; `test_newton.py` records the exact message rather than treating
the mismatch as fixed, for the owner to judge whether a Python-specific wording belongs in that
C++ message (`KNOWN_ISSUES.md` D37). (2) A Python component that *claims* `CAP_RESIDUAL` or
`CAP_LINEARIZATION` via `CAPABILITIES` without the C++ hooks to back it — the case the honest
check above cannot see, since it only tests for an *unclaimed* capability — is refused
immediately when `add_component()` is called, before `initialize()` ever runs
(`ForwardingComponent::compute_capabilities()`; `TestClaimingTheCapabilityIsRefusedImmediately`).

## What this does not expose, and why

- **`fuse_coupler_abi()` / `soma.Context`'s mirror.** The owner's own choice of `pybind11` over
  the C++ model, not the ABI (Scope, above). A `cffi` shim is the ABI's binding, separately.
- **MPI.** `RunnerConfig::base_communicator`/`force_this_process`/`transport_setup` are not
  bound. mpi4py is not installed in the target interpreter and MPI-from-Python is explicitly out
  of this task's scope; `RunnerConfig{}`'s defaults (`transport = "copy"`, no communicator) move
  whole arcs without ever touching MPI, which is what every acceptance test below runs against,
  and matches how `examples/dn_heat`'s own binary runs (`examples/CMakeLists.txt` links it
  against `fuse::runtime`/`fuse::backend` only — no MPI, no PETSc, no Fortran; confirmed by
  configuring and building it in this track's own tier-0 tree, see "The `dn_heat` driver test").
- **`fuse::NativeMeshBackend`, `fuse::MoabBackend`.** Only `fuse::SerialMeshBackend` is bound in
  v1 — every upstream test and every acceptance test below needs nothing richer than
  `"generated:line:N"`/`"generated:box:NxMxP"`, and the native/MOAB backends' own Python surface
  (if ever wanted) is a separate decision with its own dependency story.
- **`FieldSpec::hdf5_path`** (the LowFive track). Nothing in `pfuse` v1 launches a
  `LaunchMode::LowFive` component, so a bound field could never legally set this; binding an
  always-`None` field would be dead surface.
- **`ImplicitComponent`/`LinearizationToken` for Python `Component` subclasses.** See
  "Components implemented in Python," above — confirmed by the owner, not left open.
- **`validate()`'s `ConnectPolicy` tunables.** Bound with its default; no acceptance test needs
  a non-default policy. Extending the binding to accept one is additive and can wait.

See also `KNOWN_ISSUES.md` D37, which records this list (plus the `dump_graphviz()` naming bug
found while writing `test_graph.py`, below) as the v1/v2 scope boundary for the owner.

## Build/CMake integration

```cmake
option(FUSE_ENABLE_PYTHON "Build the pfuse pybind11 module" OFF)
if(FUSE_ENABLE_PYTHON)
  find_package(pybind11 REQUIRED)   # -Dpybind11_DIR=<interpreter>/.../pybind11/share/cmake/pybind11
  add_subdirectory(python)
endif()
```

`python/CMakeLists.txt` builds `_pfuse` as a `pybind11_add_module`, links `fuse::model`,
`fuse::runtime`, `fuse::backend`, `fuse::ir`, `fuse::schedule` (no `fuse::transport`'s MPI
variant, no `fuse::petsc`), and registers all **seven** `python/tests/test_*.py` files
(`buffers`, `graph`, `yaml`, `runtime`, `newton`, `dn_heat_rows`, `analytic_problem`) as `ctest`
rows — `add_test(NAME python.buffers COMMAND ${CMAKE_COMMAND} -E env
PYTHONPATH=$<TARGET_FILE_DIR:_pfuse> ${Python3_EXECUTABLE} -m pytest -q
${CMAKE_CURRENT_SOURCE_DIR}/tests/test_buffers.py ...)`, one row per file via a `foreach`, so a
failure names the file without the whole suite needing to run to isolate it — gated on
`FUSE_ENABLE_PYTHON` and on `FUSE_ENABLE_TESTS`, each row labelled `"python"`
(`set_tests_properties(... PROPERTIES LABELS "python")`) so `ctest -L python` runs exactly this
suite. Registration uses `find_package(Python3 COMPONENTS Interpreter REQUIRED)` and
`Python3_EXECUTABLE`, resolved independently of the `pybind11_add_module` call above (which
resolves the interpreter through `find_package(Python ...)` inside `pybind11Tools.cmake` — see
the `VIRTUAL_ENV` gotcha above); both must name the same interpreter or the built extension and
the interpreter running the tests disagree on ABI. `tools/check-layering.py`'s grep-based gate
needed no change: it already refuses any `pybind11` token under `include/fuse/`, and nothing
built here puts one there.

## Acceptance tests and what they establish

All seven files under `python/tests/` pass against the built `_pfuse` extension: **91 passed, 1
skipped (a big-endian `float64` check, correctly skipped on this little-endian arm64 host), 1
xfailed (the `dump_graphviz()` naming bug below, recorded rather than worked around)**, exit
code 0.

- **`test_buffers.py`** — the numpy buffer contract and the lifetime rule, transplanted from
  upstream's `test_buffers.py` with `soma.FieldId`/`soma.FieldRegistry` renamed to `pfuse`'s.
- **`test_graph.py`** — graph authoring (`add_mesh/region/component/field/channel`), `Rational`
  coercion (string/`Fraction`/int accepted, `float` rejected per spec §4.3), YAML and descriptor
  round trips, `validate()`'s diagnostics, the exception hierarchy, a field-by-field coverage
  check of `NewtonConfig`/`AccelConfig` against the exact field lists extracted from
  `include/fuse/model/iteration.hpp` by `grep` (28 and 7 fields respectively, not retyped from
  memory), and an `xfail(strict=True)` case for the `dump_graphviz()` bug below.
- **`test_yaml.py`** — split out of `test_graph.py`'s `YamlRoundTrip` class into its own file
  because `fuse`'s YAML front-end has a materially larger surface than upstream's —
  `IterationBuilder`'s accelerator/Newton configuration round-trips through YAML too, which
  upstream's YAML tests never had to cover.
- **`test_runtime.py`** — driving a real coupled run through `Session`/`fuse::Runner` with two
  Python peer components, transplanted from upstream's `SessionDrivesPythonPeers` (the numbers
  that arrive in the consumer's array are checked, not just "did not crash"), with upstream's
  `ContextIsTheCAbi` class **not** carried over (no `Context` binding — see "What this does not
  expose").
- **`test_newton.py`** — `IterationBuilder.newton()`/`.accelerate()` configuration; the
  honest-default capability-gate refusal (`TestNewtonCapabilityGate`, firing through
  `fuse::Runner::initialize()`'s pre-existing `E-NEWTON-CAPABILITY` check, with the C++-specific
  wording recorded rather than patched around); the claimed-capability refusal
  (`TestClaimingTheCapabilityIsRefusedImmediately`, firing at `add_component()` time); and a
  byte-for-byte comparison of a Python-authored Newton descriptor against one built and
  serialized natively in C++ (`TestDescriptorMatchesNativeReference`, below).
- **`test_dn_heat_rows.py`** — the `dn_heat` byte-for-byte driver test (below).
- **`test_analytic_problem.py`** — the second, hand-solvable coupled problem with a derived
  tolerance (below).

### The `dn_heat` driver test

`examples/dn_heat/main.cpp` prints a 12-row comparison table (explicit/Picard/Aitken/IQN-ILS/
IQN-ILS+QR2/severed/two "long" variants/Newton/Newton-nonlinear) plus residual histories and
summary statistics. The reference was captured by actually building and running the example in
this track's tier-0 tree (flags per the task's standing rules, plus two
`FETCHCONTENT_SOURCE_DIR` overrides pointing at the owner's own already-populated `_deps/` to
avoid a network clone of r3d/yaml-cpp), with `OMP_PROC_BIND=false OMP_PLACES=threads`: Kokkos'
own OpenMP initialization prints a 5-line, environment-dependent advisory banner only when
`OMP_PROC_BIND` is unset, and with it set the remaining 48 lines reproduced byte for byte across
three separate runs. The full captured output is `python/tests/golden/dn_heat_full_output.txt`.

**The owner's ruling (2026-10-12): reimplement the slabs in Python, not a test-only native
extension.** Stage A proposed exposing the existing native `dn::DirichletSlab`/`NeumannSlab`
(`examples/dn_heat/dn_component.hpp`) through a small test-only pybind11 target as the only way
to honestly claim "byte for byte," leaving a Python reimplementation with a derived tolerance as
the cheaper alternative (open question 3). Decided instead: reimplement the two components'
closed-form physics as Python `pfuse.Component` subclasses (`python/tests/
dn_heat_components.py`), drive them through the real `fuse::Runner` via `pfuse.Session`, and
measure whether the result is bit-identical before writing a single assertion — rather than
assuming a tolerance is needed and deriving one for it.

**What was measured before any row comparison ran.** A standalone C++ probe
(`-ffp-contract=off`, the same flag this build applies project-wide) computed
`dn::T_outer_1`/`T_outer_2` at the example's own grid of `t` values and compared the raw IEEE-754
bit patterns against Python's `math.sin`/`math.pi` for the identical inputs: every bit pattern
matched exactly, including the `kPi` literal against `math.pi`. Only given that result did the
row comparison go ahead expecting equality rather than a tolerance.

**Result: all 8 non-Newton rows reproduce byte for byte, including sweep counts.** `explicit`,
`picard` (14918 sweeps), `aitken`, `iqn-ils`, `iqn+qr2`, `severed`, `expl-long`, `iqn-long` — every
summary row and every captured residual-history line matches
`golden/dn_heat_full_output.txt` with exact string equality, checked in `test_dn_heat_rows.py`.
The three Newton rows (`newton`/`newton-repeat`/`newton-nl`) are explicitly out of scope, not
silently missing: no Python-authored graph can supply the native `ImplicitComponent` a
Newton-solved region needs ("Components implemented in Python," above), and
`TestNewtonRowsAreOutOfScope` asserts the golden file still has a `newton` row so that claim
stays checked against the fixture rather than going stale.

**The Newton descriptor is still checked, separately, without running Newton.** A standalone
C++ probe (`newton_descriptor_probe.cpp`) builds the identical mesh/region/component/field/
channel/remap/spacetime/relax/Newton-configuration graph through the raw C++ API and serializes
it (`fuse::serialize()`, 769 bytes, `golden/newton_descriptor_reference.bin`). The equivalent
graph authored from Python produces `g.to_bytes()` that matches this reference byte for byte
(`TestDescriptorMatchesNativeReference`, `test_newton.py`) — checking that a Python-authored
Newton *configuration* compiles to the identical descriptor a native one would, independent of
the fact that running it is out of scope.

### A naming bug found while writing these tests, not fixed

`include/fuse/ir/dump.hpp:126`, inside `dump_graphviz()`, still emits the literal header
`"digraph soma_ir {"` — a leftover from the soma → fuse rename that `dump_text()` (line 73, same
header) already carries correctly as `"# fuse-ir v1"`. This track does not edit `src/`/
`include/`; `test_graph.py` records the finding as an `xfail(strict=True)` test citing both line
numbers (`KNOWN_ISSUES.md` D37), so the suite documents it rather than silently working around
it, and will fail loudly (an unexpected XPASS) the day someone fixes `dump_graphviz()` without
updating that test.

## The analytic problem

A second coupled problem (`test_analytic_problem.py`), per the owner's third ruling
(2026-10-12): small enough to solve by hand, with a tolerance *derived* from that solution
rather than picked by trial and error. Two single-station components — `Source` (`x_new = RHO *
y_in + C`) and `Mirror` (`y_new = x_in`, a pure relay) — wired through the same "one channel
lagged, one not" two-channel shape `dn_heat_components.build_graph()` uses, Picard-only
(`AccelKind.None_`). The fixed point is solved by substitution (`y* = C / (1 - RHO)`), and
instrumenting both components' `advance()` with `fuse::StepContext.iteration` before writing any
assertion showed that neither channel delivers within the sweep it was produced — `fwd`'s
default (zero) `Channel.lag` still only reaches `Mirror.x_in` the sweep *after* `Source` computed
it, exactly as `back`'s explicit `.lag(1)` does for the reverse direction — so the pair
`(Source.x, Mirror.y)` retraces the fixed-point recursion at half the raw sweep rate, and
whichever field just changed on a given sweep is exactly the nonzero term in `fuse`'s own
combined residual history. Because the recursion is exactly affine, the standard fixed-point
a-posteriori identity (`|value − y*| = RHO/(1−RHO) · |value_k − value_{k−1}|`) is exact here, not
approximate, and the test checks it as measured equality (with an explicit, stated floating-point
allowance — not a fitted margin) rather than an inequality loose enough to always pass.

## Signature conventions fixed while writing the acceptance tests

Two binding-signature choices the test files above already assume, recorded here so Stage B
does not have to re-derive them from the tests alone:

- **`add_component(name, ranks, interval=1, kind=ComponentKind.Physics)` takes a `RankSet`
  object for `ranks`, not upstream's `first_rank=`/`rank_count=` keyword pair.** `fuse::RankSet`
  (unlike `soma::RankSet`) can express an arbitrary sorted rank list, not only a contiguous
  block (the evidence table's "carry over + new" row). A `first_rank`/`rank_count` convenience
  pair cannot express that case at all, so the binding accepts the richer type directly:
  `pfuse.RankSet(0, 2)` for the contiguous case (mirrors `RankSet(int first, int count)`),
  `pfuse.RankSet([0, 2, 4])` for an arbitrary list. This is a deliberate departure from
  upstream's kwarg shape, not an oversight.
- **Channel accessors are grouped by the C++ sub-struct, not flattened.** `fuse::Channel`
  composes `AccumulateSpec`/`RemapSpec`/`SpacetimeSpec`/`RelaxSpec`/`ExtrapolateSpec` as named
  members (`channel.accumulate.op`, `channel.remap.flags`, `channel.relax.omega`,
  `channel.relax.enabled`, …) where `soma::Channel` held the same information as flat scalar
  members (`accum_op`, `remap_flags`, `relax_omega`, `relax_enabled`, …) and upstream's binding
  mirrored that flatness. Binding `fuse::Channel`'s own nesting directly, rather than
  re-flattening it, keeps the Python binding a thin wrapper instead of inventing names the C++
  header does not have; `test_graph.py::test_channel_builder_recorded_what_it_was_told` is
  written against the nested form.
