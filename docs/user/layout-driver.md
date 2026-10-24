# layout_driver: try processor-set layouts from the command line

`examples/layout_driver/` is not a physics example. It is a test driver for processor-set
placement: an MPI program that builds a two-component coupling (three with a coupler) over
processor sets you give on the command line, runs it through `fuse::Runner` with
`fuse::transport::MpiTransport`, and checks that the data arrives. Use it to try disjoint, partially
overlapping and coincident placements (see [](placement.md) and the glossary's *processor set*) at
any rank count without writing a new fixture for each shape. It is the command-line version of the
`TwoComponentJob` fixture in `test/parallel/runner_transport_test.cpp`.

## Build and run

Needs MPI:

```sh
cmake --preset tier1 && cmake --build --preset tier1
mpiexec -n 2 <build-dir>/examples/fuse_example_layout_driver --comp1 0:0 --comp2 1:1 --check
mpiexec -n 3 <build-dir>/examples/fuse_example_layout_driver --comp1 0:1 --comp2 1:2 --coupler 1:2 --check
mpiexec -n 4 <build-dir>/examples/fuse_example_layout_driver --comp1 0:1 --comp2 1:3 --mesh hexbox:4x4x4 --migrate --check
```

Usage:

```{literalinclude} ../../examples/layout_driver/main.cpp
:language: cpp
:start-at: "mpiexec -n P fuse_example_layout_driver"
:end-at: "[--coupler e:f] [--entities N] [--steps K] [--check] [--timing out.csv]"
```

Ranges `a:b` are inclusive world ranks. The layout must cover the whole world exactly:
`min(a,b,c,d[,e,f]) == 0` and `max(...) == P-1`.

## What it couples

`comp1` provides a field (`"link"`) and `comp2` requires it, over `PartitionedTestBackend`
(`test/support/partitioned_test_backend.hpp`), a test-support backend that gives each graph
component its own decomposition of a shared mesh. `--entities N` sets the field's element count
(default 100); `--steps K` sets how many coupling cycles to run (default 1). The components have one
field and no physics: the driver writes the source side's storage with each entity's global id, and
with `--check` confirms every destination entity holds its own global id after redistribution. This
tests the transport and remap machinery, not any simulated physics.

**`--coupler e:f`** routes the field transfer's remap through a separately placed coupler processor
set (`comp1 -> coupler -> comp2`). The `coupler -> comp2` redistribution is omitted when `[e:f]` is
identical to `comp2`'s own range. The transfer always carries `RemapFlags::Conservative`, since
`comp1` and `comp2` share one mesh entity set here.

**`--mesh <spec>`** switches to a separate mode. Instead of the field transfer, `comp1` and `comp2`
each get a `fuse::native` generated mesh (`hexbox:NxMxK`, `tetbox:...` or `prismpyramidmix:...`),
optionally `--migrate`d from `comp1`'s decomposition onto `comp2`'s processor set by recursive
coordinate bisection (RCB). `--check` then verifies that global cell count, measure and id
uniqueness are conserved across the migration. `--migrate` requires `--mesh`.

## Output and errors

`--timing out.csv` writes one row of step timings. Rank 0 always prints one
`shape(comp1, comp2): ...` line and, under `--check`, one further line:
`check: N/N destination entities delivered (job entities = ...)` in field-transfer mode, or
`check: comp1 cells=... measure=...` in `--mesh` mode. Nothing else is printed on success. Any
rejection or failed check prints an `E-LAYOUT-*` message and exits non-zero.

| Code | Meaning |
|---|---|
| `E-LAYOUT-PARSE` | A flag is unrecognised, missing its value, not of the form `a:b`, or has a non-integer; `--comp1` or `--comp2` is missing; `--migrate` is used without `--mesh` |
| `E-LAYOUT-RANGE-EMPTY`, `E-LAYOUT-RANGE-REVERSED` | A range is missing an endpoint, or has `lo > hi` |
| `E-LAYOUT-RANK-OUT-OF-RANGE` | A range names a rank outside `[0, P-1]` |
| `E-LAYOUT-BOUNDS` | The layout does not start at rank 0 and end at rank `P-1` |
| `E-LAYOUT-COUPLER` | `--coupler` was combined with `--mesh` |
| `E-LAYOUT-MESH` | Unrecognised `--mesh` kind |
| `E-LAYOUT-TIMING` | `--timing` file could not be written |

Argument validation is deterministic and happens before any collective call, so a bad layout is
rejected on every rank rather than hanging, and each later collective step is followed by an
agreement step before any rank acts on its local result.

## Example layouts

The `layout.*` tests in `examples/CMakeLists.txt` are a good set of starting points:

| Test family | Ranks | Shapes |
|---|---:|---|
| `layout.2.*` | 2 | disjoint (both orders) and identical |
| `layout.3.*` | 3 | disjoint, partial overlap (two orderings), full overlap |
| `layout.4.*` | 4 | disjoint, partial overlap (two orderings), full overlap |
| `layout.coupler.*` | 3-4 | coupler disjoint from both components; coupler identical to `comp2` (redistribution elided); coupler overlapping `comp1` |
| `layout.migrate.*` | 2-4 | `--mesh hexbox:4x4x4 --migrate`, disjoint and partially overlapping pairs |
| `layout.refuse.*` | 2 | deliberate rejections: min not 0, max not on the last rank, reversed range, `--coupler` with `--mesh` |

Run them with `ctest -R layout`.

## Limitations

The driver uses `PartitionedTestBackend` and `fuse::native`; there is no MOAB-backed path. Covered
by the automated test suite at multiple rank counts.

## See also

- [](placement.md): processor sets, overlap and coupler placement
- [](layouts.md): processor-element layouts from the command line
- [](multiple-executables.md): placing components in separate executables
- [](examples.md): the full list of examples
