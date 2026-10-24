# Cookbook

Short recipes for common tasks. Each one shows the smallest piece of code that does the job and
points to the page or example that covers it in full. Examples named here live under `examples/` and
run in `ctest`.

## Describe a coupling

**Two components exchanging one field.** [](quickstart.md) and `examples/mms_pair/main.cpp`.
Every declaration and treatment, with its validation codes: [](graph-programming.md). The same graph
as a file: [](yaml.md).

**Convert units on the way.** Declare each field's unit. If the two are compatible (for example `K`
and `degC`, or `W` and `kW`) FUSE inserts the scale conversion; if they are not, `validate()` refuses
the channel.

**Make one transfer conserve its integral.**

```cpp
const auto ch = g.add_channel("heat_flux", q_out, q_in);
g.channel(ch).conservative();
```

Declare `Conservation::Extensive` on both fields. FUSE adds a check to the transfer and counts it in
`runner.stats()`.

**Couple components with different time steps.** Give each its own `fuse::Rational` step. Choose how
the fast side is reduced with `.accumulate(...)`: `Mean` for a state, `Integral` for a flux
([](concepts.md#time)).

**Break a cycle.** Mark one channel of the loop `.lag(1)`, or put the loop in an iteration region.
`schedule()` names the cycle if you do neither.

**Merge several sources into one field.** [](combining-writers.md).

**Keep the description in a file.**

```cpp
const fuse::Result<std::string> text = fuse::to_yaml(g);
if (!text.ok()) { std::cerr << text.error().message << "\n"; return 1; }
auto loaded = fuse::from_yaml(text.value());      // or fuse::load_yaml_file(path)
```

Both return a `fuse::Result`; `loaded.value()` is the graph once `loaded.ok()`. Building and
editing graphs in code is covered in [](graph-programming.md).

## See what FUSE built

```cpp
std::cout << fuse::dump_text(lowered.dag);        // the dataflow graph
std::cout << sched.plan.to_string();              // the instructions that run
```

Or run an example with `--dump`. Diff the output across a change to see exactly what the change did
to the coupling ([](getting-started.md#3-look-at-what-fuse-built)). [](ir.md) explains every line of
both dumps and how to walk the compiled structures from code.

## Iterate to convergence

```cpp
g.iterate({flux, temp})
    .name("interface")
    .until(std::move(crit))
    .max_iterations(200)
    .accelerate(accel);
```

Measure every exchanged field, and implement `save_state()` / `restore_state()` in each member. Start
with Aitken; move to IQN-ILS for harder problems and to Newton when you need a Jacobian-free Newton
solve. [](dn-heat.md) compares them on one problem and [](convergence.md) explains the measures.
Strongly coupled fluid-structure: `examples/fsi_channel` ([](fsi-channel.md)). Reacting flow with
PETSc: [](petsc-reaction-diffusion.md).

## Couple incommensurate clocks

Components whose steps share no useful period are advanced over a window, iterated as a group
([](coupling-windows.md)).

## Move data between different meshes

| You need | Read |
|---|---|
| A conserved flux between non-matching meshes | [](geometry.md) |
| Values one way and loads back across an interface | [](interface-transfer.md) |
| Scattered points with no mesh | [](point-cloud-remap.md) |
| Reuse expensive weights | [](transfer-maps.md) |
| Field operations on a mesh (merge, derive) | [](field-pipeline.md) |

## Run on several ranks

Give each component a rank set in its declaration. Overlapping, identical and disjoint sets all work,
and a separate coupler can do the remapping ([](placement.md)). Launch with `mpiexec -n N` on an MPI
build. A coupling in separately built executables: [](multiple-executables.md).

## Stop and resume

Write state to disk and resume, even on a different number of ranks: [](restart.md).

## Write your own executable

A driver that loads a YAML file, takes command-line options, installs the environment error policy
and runs: [](drivers.md). Every environment variable and `RunnerConfig` field: [](environment.md).
The full component contract: [](component-interface.md); the Runner API: [](runtime.md).

## Couple scales, or a learned model

Macro-micro coupling with the features that exist today: [](multiscale.md). A PyTorch,
scikit-learn or any callable model as a component: [](surrogates.md).

## Bring your own code

| Your code is | Use |
|---|---|
| A C++ solver you can edit | Implement `fuse::Component` ([](quickstart.md)) |
| C, with its own time loop | [](c-abi.md) |
| Fortran | [](fortran.md) |
| A plugin FUSE should drive | [](providers.md) |
| Python | [](python.md) |
| Unmodified HDF5 codes | [](lowfive.md) |
| Several of the above together | [](mixing-languages.md) |

## Study a full application

Each example is a complete program with its problem statement, expected output and pass criteria.

| Example | Shows |
|---|---|
| [](mms-pair.md) | Multirate coupling checked against an exact solution |
| [](dn-heat.md) | Dirichlet-Neumann iteration and accelerators |
| [](gray-scott.md) | Reaction-diffusion coupled across components |
| [](brusselator.md) | Oscillating chemistry with a stiff partner |
| [](fsi-channel.md) | Fluid-structure interaction |
| [](petsc-reaction-diffusion.md) | Newton with PETSc |
| [](demo-b.md), [](layout-driver.md) | Placement and layout |

[](examples.md) lists them with their ctest names.
