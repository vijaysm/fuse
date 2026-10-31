# FUSE documentation

**F**lexible, **U**nified, **S**calable **E**xchange: a C++20 library for coupling multiphysics and
multiscale simulations.

You describe a coupled simulation as a graph of components, the fields they exchange and the
transfers between them. FUSE checks the graph, compiles it into an inspectable schedule with exact
rational time and a start-up proof that it cannot deadlock, and runs it on Kokkos: on one process or
many MPI ranks, in one executable or several.

## Where to start

| You want to | Read |
|---|---|
| Build FUSE, run a coupling and learn to read what it prints | [Getting started](user/getting-started.md), then [Quickstart](user/quickstart.md) and [Concepts](user/concepts.md) |
| Do a specific task | [Cookbook](user/cookbook.md), [Examples](user/examples.md), [Portfolio](user/portfolio.md), [Troubleshooting](user/troubleshooting.md) |
| Run at scale or reproducibly | [Power-user guide](user/power-users.md), [Placement](user/placement.md), [Build guide](user/build.md) |
| Write your own coupled problem | [Graph programming](user/graph-programming.md), [Writing a component](user/component-interface.md), [Drivers](user/drivers.md), [YAML reference](user/yaml.md) |
| Understand what FUSE compiles and runs | [IR and execution plan](user/ir.md), [Runtime](user/runtime.md), [Environment variables](user/environment.md) |
| Couple codes in C, Fortran or Python | [Mixing languages](user/mixing-languages.md), [C ABI](user/c-abi.md), [Fortran](user/fortran.md), [Python](user/python.md) |
| Use a Runge-Kutta or Rosenbrock integrator inside a component | [Time integrators](user/integrators.md) |
| Look up a word | [Glossary](user/glossary.md) |
| Contribute | [CONTRIBUTING](../CONTRIBUTING.md) and the [developer guide](dev/index.md) |

The full list of pages is the [table of contents](contents.md). The same documentation is built as a
website with Sphinx (`sphinx-build -W -b html docs docs/_build/html`) and published to the
[project wiki](https://github.com/vijaysm/fuse/wiki).
