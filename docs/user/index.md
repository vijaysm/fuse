# User guide

[](glossary.md) defines every term used in this documentation and gives the literature term for each.

Start with [](build.md) for what you need installed — FUSE builds as a ladder of build
configurations, and the default one
needs only a C++20 compiler, Kokkos and a small linear-algebra backend. Then [](quickstart.md) for a
complete two-component coupling, [](fluent-api.md) for the chaining idiom and its rules,
[](combining-writers.md) when several sources write one field,
[](convergence.md) for how an implicit coupling decides it is done,
[](coupling-windows.md) for coupling two components that do not share a
time grid, and [](c-abi.md) if you are coupling an existing code through C or
Fortran — [](fortran.md) is the Fortran side's own page, for the `ffuse`
module's generated names and type mapping in full. Those three pages are
about a foreign code driving FUSE; [](providers.md) is the other direction —
FUSE driving a plugin you write, in C or Fortran, named by path in the graph.
