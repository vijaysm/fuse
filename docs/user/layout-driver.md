# layout_driver: the processor-layout test driver

`examples/layout_driver/` is not a physics example. It is the *processor-layout test driver*
(glossary term): an MPI-SPMD program that builds a two- (or three-, with a coupler) component
coupling specification over processor sets given on the command line and runs it through
`fuse::Runner` with `fuse::transport::MpiTransport`, so that disjoint, partially overlapping, and
coincident processor-set placements (glossary: *processor set*) can be exercised at an arbitrary
rank count without writing a new fixture for each shape. It is the same recipe
`test/parallel/runner_transport_test.cpp`'s `TwoComponentJob` fixture exercises with
`Overlap::{Disjoint,Partial,Full}`, driven here from the command line instead.

## What it couples

`comp1` provides a field (`"link"`), `comp2` requires it, over `PartitionedTestBackend`
(`test/support/partitioned_test_backend.hpp`) — the only `MeshBackend` in the tree that gives each
graph component its own decomposition of a shared mesh, which is exactly the capability a processor
layout test needs. `--entities N` sets the field's element count (default 100); `--steps K` how many
coupling cycles to run (default 1). `LayoutComponent` (`main.cpp`) is a minimal component with one
field and no physics: the driver itself writes the source side's storage by each entity's own global
id and reads the destination side's directly, so `--check` can confirm every destination entity holds
its own global id after the redistribution — a correctness property of the transport and remap
machinery, not of any simulated physics.

```{literalinclude} ../../examples/layout_driver/main.cpp
:language: cpp
:start-at: "mpiexec -n P fuse_example_layout_driver"
:end-at: "[--coupler e:f] [--entities N] [--steps K] [--check] [--timing out.csv]"
```

**`--coupler e:f`** routes the field transfer's remap through a separately placed coupler processor
set (glossary: *coupler processor set*) — `comp1 -> coupler -> comp2` — with the `coupler -> comp2`
redistribution elided whenever `[e:f]` is identical to `comp2`'s own range (glossary: *redistribution
omitted*). It always attaches `RemapFlags::Conservative`, since `comp1` and `comp2` share one mesh
entity set here and nothing else would make `remap_on` do anything. Not supported together with
`--mesh` — that mode never builds the field-transfer graph at all.

**`--mesh <spec>`** switches the driver into a *separate* mode: instead of the synthetic
field-transfer demonstration, `comp1` and `comp2` each get a real `fuse::native` generated mesh
(`hexbox:NxMxK`, `tetbox:...`, or `prismpyramidmix:...`), optionally **`--migrate`**d from `comp1`'s
decomposition onto `comp2`'s processor set via recursive coordinate bisection (RCB). `--check` in
this mode re-verifies mesh data transfer — global cell count, measure, and id uniqueness conserved
across the migration — not the field transfer the other mode checks, which this mode does not build
at all. `--coupler` together with `--mesh` is rejected (`E-LAYOUT-COUPLER`): that mode never builds
a field-transfer graph a coupler could attach to.

**The PE layout must cover the whole world exactly**: `min(a,b,c,d[,e,f]) == 0` and
`max(...) == P-1`. A violation, a reversed or unparseable range, a rank outside `[0, P-1]`, or
`--coupler` with `--mesh` is a rejection on **every** rank, never a hang — `argv` is identical on
every rank under `mpiexec`, so every validation decision up to the graph build is deterministic and
reached before any collective call, and every collective call this binary makes afterwards
(`Runner::initialize`, `run_period`, the `--check` reduction) is followed by an agreement step before
any rank acts on its local half of the answer — the same rank-agreement discipline
([](../dev/testing.md#the-five-states); `Epoch`, glossary) applied end to end in a user-facing
driver.

## Building and running it

Needs tier 1 (MPI):

```sh
cmake --preset tier1 && cmake --build --preset tier1
mpiexec -n 2 <build-dir>/examples/fuse_example_layout_driver --comp1 0:0 --comp2 1:1 --check
mpiexec -n 3 <build-dir>/examples/fuse_example_layout_driver --comp1 0:1 --comp2 1:2 --coupler 1:2 --check
mpiexec -n 4 <build-dir>/examples/fuse_example_layout_driver --comp1 0:1 --comp2 1:3 --mesh hexbox:4x4x4 --migrate --check
```

This sandbox cannot start MPI at all, so every `layout.*` row below is a `ctest` registration
(`examples/CMakeLists.txt`'s `fuse_add_layout_test`) that is build-verified here and
**host-verification-needed** for its run outcome.

## What it prints

`--timing out.csv` writes one row of step timings. Rank 0 always prints one `shape(comp1,
comp2): ...` line regardless of `--timing` (`examples/layout_driver/main.cpp`'s own
`shape_of()` call) and, under `--check`, one further `check: N/N destination entities
delivered (job entities = ...)` line — the driver is silent only apart from those two lines on
success, and prints an `E-LAYOUT-*`-prefixed message and returns non-zero on any rejection or a
failed `--check`.

## Measured against (host-recorded, KNOWN_ISSUES.md)

Twenty-one `layout.*` `ctest` rows are registered, covering every combination `fuse_add_layout_test`
in `examples/CMakeLists.txt` declares:

| Row family | Ranks | Shape |
|---|---:|---|
| `layout.2.*` (3 rows) | 2 | identical, disjoint, and the only non-trivial 2-rank partial-overlap is impossible at 2 ranks, so these three cover identical/disjoint/the other identical ordering |
| `layout.3.*` (4 rows) | 3 | disjoint, partial overlap (two orderings), full overlap |
| `layout.4.*` (4 rows) | 4 | disjoint, two partial-overlap orderings, full overlap |
| `layout.coupler.*` (3 rows) | 3-4 | coupler disjoint from both components; coupler identical to `comp2` (the elided-hop case); coupler overlapping `comp1` |
| `layout.migrate.*` (3 rows) | 2-4 | `--mesh hexbox:4x4x4 --migrate`, disjoint/partial/partial processor-set pairs |
| `layout.refuse.*` (4 rows) | 2 | deliberate rejections: a zero-width range at both ends, a max not on the last rank, a reversed range, `--coupler` with `--mesh` |

Host runs recorded in `KNOWN_ISSUES.md`:

- **D5, 2026-09-23**: partial overlap delivers every entity at 3, 4 and 5 ranks, full overlap at
  every rank count, and all 18 `layout.*` rows registered at that date pass, including the 3
  `layout.migrate.*` rows.
- **D-coupler-placement, 2026-09-27**: the three `layout.coupler.*` rows registered that day FAILED
  at `initialize()` with `E-MESH-PLACEMENT` — the driver had not been updated to read one processor
  set per graph component (including the coupler) from the partitioned backend. Fixed the same
  session: the driver now reads one processor set per graph component, exactly as
  `runner_transport_test.cpp`'s own harness does.
- **D4, 2026-09-28**: `ctest -R 'layout.coupler|parallel.runner_transport|parallel.remap_halo|
  backend.remap_halo'` and `ctest -L omp` all pass on the host — the `layout.coupler.*` fix above is
  host-confirmed, not merely built.

`tools/check-parallel-matrix.py` tracks this example's rows as part of its cross-check that every
claimed rank count in `build-evidence/parallel-matrix.toml` is really run; as of the `D4` entry above
it reported 70 claims verified, 0 broken, 1 gap (the one gap belongs to a different track, D8's
`A-NODAL-HT-PARALLEL`).

## State

**Verified** (host-recorded, above) for `layout.2.*`, `layout.3.*`, `layout.4.*`, `layout.coupler.*`,
`layout.migrate.*`, and `layout.refuse.*` at tier 1, as of the 2026-09-28 host run. Build-verified
only in this sandbox, which cannot start MPI at all (`KNOWN_ISSUES.md` C1) — confirmed here by a
direct `mpiexec` attempt failing at the process manager's own port bind, consistent with that
record. No tier-2 (MOAB) path exists for this driver: `PartitionedTestBackend` is a test-support
backend, not a MOAB-backed one, and the `--mesh`/`--migrate` mode uses `fuse::native`, not MOAB.
