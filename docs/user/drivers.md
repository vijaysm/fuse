# Writing a driver: command line, YAML and configuration

A *driver* is the executable that owns `main()`: it reads options, builds or loads the coupling
graph, compiles it, creates the components and runs the `fuse::Runner`. This page is for users who
want to run their own coupled problem from a program they control, rather than from one of the
shipped examples. It shows a complete generic YAML driver, how to pass options through to the
graph and the Runner, and the conventions the examples follow.

:::{note}
FUSE ships no generic driver executable. There is no `fuse run graph.yaml`. The examples each
hand-roll their own argument loop, and the closest thing to a reusable driver is the C participant
`examples/brusselator/participants/coupler_main.c`, which hardcodes its component name. This page
shows how to write a driver of your own in about a hundred lines.
:::

## What a driver does

Every FUSE program passes through the same five stages ([](getting-started.md#6-write-your-own)):

1. **Describe**: a `fuse::CouplingGraph`, built in code ([](graph-programming.md)) or loaded
   from YAML ([](yaml.md)).
2. **Load**: give each `MeshDecl` to a `MeshBackend` (`SerialMeshBackend` for generated meshes in
   one process; see [](geometry.md) for the others).
3. **Compile**: `fuse::validate(graph)` reports every defect, then `fuse::lower(graph)` produces
   the operator DAG.
4. **Schedule**: `fuse::schedule(graph, dag, clocks, sizes)` produces the fixed instruction order.
5. **Run**: create one `fuse::Component` per graph component, bind storage through
   `register_fields`, `Runner::initialize`, then `Runner::run` or `run_period`.

A driver comes in two shapes:

| Shape | Who owns the time loop | Read |
|---|---|---|
| A C++ program on `fuse::Runner` | FUSE: the Runner calls `advance()` on every component | this page |
| A C or Fortran program on the participant ABI | the application: it calls `fuse_create`, loads the graph from `fuse_graph_source.yaml_path`, then calls `run` | [](c-abi.md), [](fortran.md), [](multiple-executables.md) |

In the second shape each executable is one participant and the YAML file is the shared contract;
the C ABI loads it for you. The rest of this page is about the first shape.

## A generic YAML driver

The program below loads any graph whose components it knows how to build, compiles it, runs it
and prints the run statistics. It handles `--graph <file>`, `--periods N`, `--dump`,
`--transport copy|mpi`, `--no-check-conservation` and Kokkos' own `--kokkos-*` flags, and it calls
`fuse::initialize()` so that the `FUSE_ON_ERROR` family of variables takes effect
([](environment.md#error-policy-variables)).

Two things are yours to change: the `ArrayComponent` class (a placeholder with no physics) and
the `kRegistry` map from graph component name to factory. Everything else is the fixed pipeline.

```cpp
// yaml_driver.cpp: load a coupling from a file, compile it, run it.
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <fuse/base/config.hpp>
#include <fuse/base/error.hpp>
#include <fuse/backend/serial_backend.hpp>
#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/validate.hpp>
#include <fuse/model/yaml_io.hpp>
#include <fuse/runtime/runner.hpp>
#include <fuse/schedule/clock.hpp>
#if FUSE_HAVE_MPI
#include <mpi.h>
#include <fuse/transport/transport.hpp>
#endif

namespace {

// One array, one field, no physics: the producer writes sin(0.1 i + t), the
// consumer stores what arrives. Replace these with your own components.
class ArrayComponent final : public fuse::Component {
 public:
  ArrayComponent(std::string name, std::size_t n, bool produces)
      : name_(std::move(name)), data_(n, 0.0), produces_(produces) {}
  std::string name() const override { return name_; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = g.find_component(name_);
    if (!cid) return fuse::Failure{"E-DRIVER-COMP", "'" + name_ + "' is not in the graph"};
    for (const fuse::FieldId f : g.component(*cid).fields) {
      auto r = reg.bind(f, fuse::FieldRef{data_.data(),
                                          fuse::Layout{.stride = 1, .count = data_.size()},
                                          fuse::MemorySpace::Host});
      if (!r.ok()) return r;
    }
    return fuse::Result<void>::success();
  }
  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    if (produces_) {
      const double t = ctx.t_end.to_double();
      for (std::size_t i = 0; i < data_.size(); ++i) data_[i] = std::sin(0.1 * double(i) + t);
    }
    return fuse::Result<void>::success();
  }
  double sum() const { double s = 0; for (double v : data_) s += v; return s; }
 private:
  std::string name_;
  std::vector<double> data_;
  bool produces_;
};

// name -> factory. Fill this with your own components.
using Factory = std::function<std::unique_ptr<fuse::Component>(std::size_t cells)>;
const std::map<std::string, Factory> kRegistry = {
    {"producer", [](std::size_t n) { return std::make_unique<ArrayComponent>("producer", n, true); }},
    {"consumer", [](std::size_t n) { return std::make_unique<ArrayComponent>("consumer", n, false); }},
};

struct Options {
  std::string graph;
  std::int64_t periods = 10;
  bool dump = false;
  bool check_conservation = true;
  std::string transport = "copy";
};

int usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " --graph <file.yaml> [--periods N] [--dump]"
            << " [--transport copy|mpi] [--no-check-conservation] [--kokkos-*]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  // Kokkos strips its own --kokkos-* flags here, so the loop below never sees them.
  Kokkos::ScopeGuard kokkos_guard(argc, argv);
  // Read FUSE_ON_ERROR / FUSE_DEBUGGER* / FUSE_BACKTRACE for this process. Call it first.
  fuse::initialize();
  fuse::install_terminate_handler();

  Options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&](std::string& out) { if (i + 1 >= argc) return false; out = argv[++i]; return true; };
    std::string v;
    if (a == "--graph" && value(v)) opt.graph = v;
    else if (a == "--periods" && value(v)) opt.periods = std::stoll(v);
    else if (a == "--transport" && value(v)) opt.transport = v;
    else if (a == "--dump") opt.dump = true;
    else if (a == "--no-check-conservation") opt.check_conservation = false;
    else return usage(argv[0]);
  }
  if (opt.graph.empty()) return usage(argv[0]);

  // 1. Describe: the graph comes from the file.
  auto loaded = fuse::load_yaml_file(opt.graph);
  if (!loaded.ok()) {
    std::cerr << loaded.error().code << ": " << loaded.error().message << "\n";
    return 1;
  }
  const fuse::CouplingGraph& graph = loaded.value();
  if (const auto d = fuse::validate(graph); d.has_errors()) { std::cerr << d.to_string(); return 1; }

  // 2. Load meshes.
  fuse::SerialMeshBackend backend;
  for (std::uint32_t m = 0; m < graph.mesh_count(); ++m) {
    if (const auto r = backend.load(graph.mesh(fuse::MeshId{m})); !r.ok()) {
      std::cerr << r.error().code << ": " << r.error().message << "\n";
      return 1;
    }
  }

  // 3. Compile and 4. schedule.
  const auto lowered = fuse::lower(graph);
  if (!lowered.ok()) { std::cerr << lowered.diags.to_string(); return 1; }
  const fuse::ClockSet clocks{graph};
  const auto sched = fuse::schedule(graph, lowered.dag, clocks,
                                    fuse::make_region_sizes(graph, lowered.dag, backend));
  if (!sched.ok()) { std::cerr << sched.diags.to_string(); return 1; }
  if (opt.dump) {
    std::cout << "=== IR ===\n" << fuse::dump_text(lowered.dag)
              << "\n=== PLAN ===\n" << sched.plan.to_string() << "\n";
    return 0;
  }

  // 5. Run: one component object per graph component, looked up by name.
  fuse::ComponentSet components;
  for (std::uint32_t c = 0; c < graph.component_count(); ++c) {
    const auto& decl = graph.component(fuse::ComponentId{c});
    const auto it = kRegistry.find(decl.name);
    if (it == kRegistry.end()) {
      std::cerr << "E-DRIVER-UNKNOWN: no factory registered for component '" << decl.name << "'\n";
      return 1;
    }
    const auto& region = graph.region(graph.field(decl.fields.at(0)).region);
    components.add(it->second(backend.region_size(region).value()));
  }
  fuse::FieldRegistry registry;
  for (auto& c : components) {
    if (const auto r = c.register_fields(registry, graph); !r.ok()) {
      std::cerr << r.error().code << ": " << r.error().message << "\n";
      return 1;
    }
  }

  fuse::RunnerConfig cfg;
  cfg.transport          = opt.transport;
  cfg.check_conservation = opt.check_conservation;
#if FUSE_HAVE_MPI
  if (opt.transport == "mpi") {
    cfg.base_communicator = MPI_Comm_c2f(MPI_COMM_WORLD);
    cfg.transport_setup   = [](fuse::Transport& t) -> fuse::Result<void> {
      auto* mpi = dynamic_cast<fuse::MpiTransport*>(&t);
      if (!mpi) return fuse::Failure{"E-DRIVER-TRANSPORT", "expected an MpiTransport"};
      return mpi->set_communicator(MPI_Comm_c2f(MPI_COMM_WORLD));
    };
  }
#endif
  fuse::Runner runner{graph, lowered.dag, sched.plan, backend};
  if (const auto r = runner.initialize(components, registry, cfg); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }
  if (const auto r = runner.run(components, opt.periods); !r.ok()) {
    std::cerr << r.error().code << ": " << r.error().message << "\n";
    return 1;
  }

  const auto& s = runner.stats();
  std::cout << "periods              : " << s.periods << "\n"
            << "instructions         : " << s.instructions_executed << "\n"
            << "conservation checks  : " << s.conservation_checks << "\n"
            << "violations           : " << s.conservation_violations << "\n"
            << "remap weight builds  : " << s.remap_weight_builds << "\n"
            << "windows not converged: " << s.windows_not_converged << "\n";
  const auto* prod = dynamic_cast<ArrayComponent*>(components.find("producer"));
  const auto* cons = dynamic_cast<ArrayComponent*>(components.find("consumer"));
  std::cout << "sum producer / consumer: " << prod->sum() << " / " << cons->sum() << "\n";
  return s.conservation_violations == 0 ? 0 : 1;
}
```

Points worth noting in the code:

- `Kokkos::ScopeGuard` is constructed *before* the argument loop. Kokkos removes its own
  `--kokkos-*` flags from `argv`, so a strict parser that rejects unknown flags still accepts
  them. A driver that parses first and then constructs the guard rejects `--kokkos-num-threads=4`
  as an unknown argument.
- `FUSE_HAVE_MPI` is a `0`/`1` macro from `<fuse/base/config.hpp>`, so the test is
  `#if FUSE_HAVE_MPI`, not `#ifdef`. In an MPI build the driver must also call `MPI_Init` before
  the guard and `MPI_Finalize` at the end, as `examples/layout_driver/main.cpp` does.
- `validate()` runs before `lower()`. `load_yaml_file` checks the file's syntax and references,
  not the graph-wide rules; only `validate()` does that.
- The component count comes from the backend (`region_size`) rather than from a flag, so the same
  binary serves any mesh size written in the YAML.
- Every refusal is a `Result` carrying an `E-...` code and a message; printing both is what makes
  the failure searchable in [](troubleshooting.md#error-codes).

A graph for it, `pair.yaml`, in the same shape as `mms_pair` (a producer on a fine step, a
consumer on a coarse one, the mean of the fine steps delivered each period):

```yaml
version: 1
meshes:
  - {name: m_left,  source: "generated:line:64"}
  - {name: m_right, source: "generated:line:64"}
regions:
  - {name: left,  mesh: m_left,  dimension: 1}
  - {name: right, mesh: m_right, dimension: 1}
components:
  - name: producer
    ranks: {first: 0, count: 1}
    interval: 1/4000
    fields:
      - {name: T, region: left, disc: FV, components: 1, units: K,
         conservation: Extensive, intent: Provides}
  - name: consumer
    ranks: {first: 1, count: 1}
    interval: 1/1000
    fields:
      - {name: T_in, region: right, disc: FV, components: 1, units: K,
         conservation: Extensive, intent: Requires}
channels:
  - name: T_link
    source: producer.T
    target: consumer.T_in
    accumulate: {op: Mean}
    remap: {flags: [Conservative]}
iteration_regions: []
```

Build it against an installed FUSE ([](build.md#using-fuse-from-your-own-project)) and run:

```sh
./yaml_driver --graph pair.yaml --periods 100
```

Expected output (exact digits can vary by platform):

```
periods              : 100
instructions         : 1600
conservation checks  : 100
violations           : 0
remap weight builds  : 0
windows not converged: 0
sum producer / consumer: 0.126369 / 0.125933
```

The consumer's sum lags the producer's by one period, which is the explicit coupling's one-cycle
delay ([](mms-pair.md)). `remap weight builds` is 0 because the two generated lines coincide and
the remap is the proved identity. `--dump` prints the IR and plan and exits 0 without running;
the plan shows four `Source`/`Sample` pairs per period followed by one `Accumulate`, `Transport`,
`Epoch`, `Remap`, `Constrain`, `Diagnose` and `Sink` ([](ir.md)).

What the driver prints when something is wrong:

```
$ ./yaml_driver --graph missing.yaml
E-YAML-IO: cannot open 'missing.yaml'
$ ./yaml_driver --graph pair.yaml --transport mpi        # serial build
E-TRANSPORT-BACKEND: no transport backend named 'mpi'
$ FUSE_EXECUTION_SPACE=bogus ./yaml_driver --graph pair.yaml
E-RT-SPACE: FUSE_EXECUTION_SPACE='bogus' is not a Runner execution space; use 'host' or 'device' (or unset it), or set RunnerConfig::execution_space
```

If `remap: {flags: [Conservative]}` is left out of the YAML, `validate()` refuses the channel with
`E-CONN-CONSERVATION`, because the target field is `Extensive` and the default transfer makes no
conservation guarantee. The message says which of the two to change.

:::{tip}
With `transport: copy` and no `base_communicator`, the `ranks` in the YAML are not used for
placement: a copy transport moves whole arcs inside one process. They matter as soon as you run
`--transport mpi` under `mpiexec`, where each component's `RankSet` must lie inside the
communicator you pass ([](placement.md)).
:::

## Propagating options

What is configurable, and where it belongs:

| Setting | Lives in | How a driver exposes it |
|---|---|---|
| Component rank sets, time steps (`interval`), channel treatments (accumulate, remap flags, lag), iteration regions and accelerators | The graph: YAML file or `CouplingGraph` code | Edit the YAML, or build the graph in code from parsed flags |
| Transport (`copy`, `mpi`, `lowfive`), conservation check and tolerance, abort on violation or non-convergence, base communicator, execution space | `fuse::RunnerConfig` ([](environment.md#runnerconfig)) | One CLI flag per field you want to vary |
| Execution space | `RunnerConfig::execution_space`, or `FUSE_EXECUTION_SPACE` when left at `Default` | A flag that sets the field, or leave it to the environment |
| Thread count, device id, Kokkos tuning | Kokkos | `--kokkos-num-threads=N`, `--kokkos-device-id=N`, parsed by `Kokkos::ScopeGuard` |
| Process count and placement on nodes | The MPI launcher | `mpiexec -n P`, host files, binding options |
| Error policy, debug traces | Environment variables | [](environment.md); a driver only needs the three-line prologue |

### Overriding graph fields after loading YAML

`CouplingGraph` has no setter for a component's ranks or interval once it is built: the accessors
`graph.component(id)` and friends are `const`. So there are two honest ways to make a rank layout
a command-line matter:

1. **Generate the YAML.** Write the file from a template with the per-run values filled in, then
   load it. This keeps one description per run on disk, which is also what you want for
   reproducibility.
2. **Build the graph in code from parsed flags**, as `examples/layout_driver/main.cpp` does. Its
   `parse_range` turns `a:b` into an inclusive `RankSet`:

```cpp
#include <optional>
#include <string>
#include <fuse/model/component.hpp>

// "a:b" (inclusive) -> RankSet{first, count}. nullopt on a malformed range.
std::optional<fuse::RankSet> parse_range(const std::string& text) {
  const auto colon = text.find(':');
  if (colon == std::string::npos) return std::nullopt;
  const int lo = std::stoi(text.substr(0, colon));
  const int hi = std::stoi(text.substr(colon + 1));
  if (lo > hi) return std::nullopt;
  return fuse::RankSet{lo, hi - lo + 1};
}

// ... later, while building the graph:
//   g.add_component(fuse::ComponentDecl{.name = "comp1", .ranks = *parse_range(arg_comp1), ...});
```

The layout driver checks the result against the world size before any collective call
(`E-LAYOUT-RANK-OUT-OF-RANGE`, `E-LAYOUT-BOUNDS`), so a bad layout is rejected on every rank
instead of hanging on some of them ([](layout-driver.md)).

## Conventions the examples follow

The examples have no shared option parser. Each one hand-rolls a `for (i = 1; i < argc; ++i)`
loop, so the conventions below are habits, not an API; copy the ones that suit you.

### Flags per example

| Executable | Flags | Unknown flag |
|---|---|---|
| `fuse_example_mms_pair` | `--dump` | ignored |
| `fuse_example_dn_heat` | `--dump`, `--newton-petsc` (needs PETSc) | ignored |
| `fuse_example_gray_scott` | `--n`, `--dt`, `--ratio m` or `p:q`, `--policy fixed|adjustable`, `--accel`, `--t-end`, `--check`, `--output <csv>`, `--F`, `--k`, `--Du`, `--Dv`, `--L` | ignored |
| `fuse_example_fsi_channel` | `--nx-fluid`, `--ny-fluid`, `--nx-beam`, `--mass-ratio`, `--accel`, `--dt`, `--t-end`, `--csv`, `--check` | `argument error`, exit 2 |
| `fuse_example_brusselator` | `--nx-u`, `--ny-u`, `--nx-v`, `--ny-v`, `--dt-u`, `--dt-v`, `--window`, `--end`, `--alpha`, `--tolerance`, `--max-iterations`, `--no-forcing`, `--save-history`, `--check`, `--accel`, `--out`, `--driver window|runner`, `--iteration-mode jacobi|gauss-seidel`, `--no-block-jacobi`, `--substrate native|petsc`, `--help` | exit 1 |
| `fuse_example_petsc_reaction_diffusion` | `--petsc` (exit 1 without PETSc) | ignored |
| `fuse_example_layout_driver` | `--comp1 a:b`, `--comp2 c:d`, `--coupler e:f`, `--entities N`, `--steps K`, `--check`, `--mesh <spec>`, `--migrate`, `--timing <csv>` | `E-LAYOUT-PARSE` |
| `brusselator_graph_tool` | `--ranks-u`, `--ranks-v`, `--ranks-coupler`, `--launch mpmd|accept`, `--out graph.yaml`, `--port-file`, `--transport fuse|lowfive` | |
| `brusselator_u_main` (C) | `--graph`, `--log`, `--ranks-u N`, `--periods`, `--transport fuse|lowfive`, `--solver auto|direct|schur|pcg|gmres` | exit 1 |
| `brusselator_v_main` (Fortran) | `--graph`, `--log`, `--periods`, `--transport`, `--ranks-u`, `--ranks-v`, `--solver` | |
| `brusselator_coupler_main` (C) | `--graph`, `--periods` | |

Two cautions that follow from the table. `gray_scott` ignores a misspelled flag silently, so
`--t_end 10` runs with the default. `brusselator` parses its arguments before constructing
`Kokkos::ScopeGuard`, so it rejects `--kokkos-*` flags with `FAIL: unknown argument`; every other
C++ example constructs the guard first and accepts them.

### `--accel` names

The examples that iterate share one name table (`examples/common/accel_names.hpp`):

| Name | Accelerator |
|---|---|
| `none` | plain fixed-point (Picard) |
| `constant` | constant under-relaxation |
| `aitken` | Aitken dynamic relaxation |
| `secant`, `steffensen` | scalar secant methods |
| `iqn-ils` (alias `iqn`) | interface quasi-Newton, inverse least squares |
| `iqn-imvj` | interface quasi-Newton, inverse multi-vector Jacobian |

An unknown name is refused with `E-EXAMPLE-ACCEL`. `brusselator` additionally accepts `newton`.
[](dn-heat.md) and [](convergence.md) compare them.

### Exit codes as checks

Every example exits non-zero when its own gate fails, so `ctest` can run it with no output
parsing. Do the same: compare against a reference, inspect `runner.stats()`, and `return 1` on
any failed check. A `--dump` mode should exit 0 without running, and should print *something*;
the examples' dump tests assert on the output because an empty dump also exits 0.

### MPI-agreed exit

In an MPI driver a failure on one rank must not leave the others blocked in a collective. The
layout driver's pattern is a file-local

```cpp
int agree_and_exit(int local_status) {
  int agreed = local_status;
  MPI_Allreduce(&local_status, &agreed, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  MPI_Finalize();
  return agreed;
}
```

called after every stage, with the local `Result` reduced to 0 or 1 first. Validate arguments
*before* the first collective so that a bad command line is rejected identically on every rank.

## Reproducibility checklist

- **Times on a fixed-point grid.** The examples convert command-line `--dt` values to exact
  `fuse::Rational`s on a `1e-6` grid before building the graph; `fsi_channel` refuses a `--dt`
  that rounds to zero with `E-EXAMPLE-DT`. Never build a `Rational` from a raw `double`.
- **Keep `fuse-resolved-dependencies.txt`.** Every configure writes it into the build directory;
  it records which Kokkos, fmt, Eigen and MPI the binary was built against.
- **Keep a golden `--dump`.** The IR and plan are a function of the graph and the build; diff
  them before and after any change to the coupling or to FUSE itself
  ([](cookbook.md#see-what-fuse-built)).
- **Record the launch.** The `mpiexec` line, `OMP_NUM_THREADS`, `FUSE_EXECUTION_SPACE` and any
  `--kokkos-*` flags, next to the output.
- **Leave `FUSE_FP_REPRODUCIBLE` on** unless you have decided otherwise
  ([](power-users.md#reproducibility)).

## See also

- [](environment.md): every environment variable and `RunnerConfig` field.
- [](component-interface.md): the contract behind `register_fields` and `advance`.
- [](yaml.md): the file format the driver loads.
- [](layout-driver.md): a complete MPI driver with ranges, timing and agreed exits.
- [](multiple-executables.md), [](c-abi.md), [](fortran.md): drivers that are participants.
- [](troubleshooting.md): what to do with the error codes a driver prints.
