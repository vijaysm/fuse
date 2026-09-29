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
Fortran.
