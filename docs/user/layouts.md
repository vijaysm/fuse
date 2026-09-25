# PE layouts from the command line

`fuse_example_layout_driver` (`examples/layout_driver/main.cpp`, Tier 1 only) builds the same
two-component job [](placement.md) describes — comp1 provides a field, comp2 requires it, one
channel — over rank ranges you give it on the command line, rather than in C++. It exists to make a
placement's shape and its transport cost visible without writing a fixture, and to give the parallel
matrix ([](../dev/testing.md)) an explicit, CLI-driven axis of coverage at 2, 3 and 4 ranks.

## Command line

```bash
mpiexec -n P fuse_example_layout_driver \
    --comp1 a:b --comp2 c:d [--coupler e:f] \
    [--entities N] [--steps K] [--check] [--timing out.csv]
```

`--comp1`/`--comp2` are **inclusive** rank ranges (`0:3` means ranks 0, 1, 2 and 3) and are
required. Every rank of the job must be covered at its two ends: the smallest endpoint named by any
range must be `0` and the largest must be `P - 1`, where `P` is the size of `MPI_COMM_WORLD`. That
is a weaker condition than "every rank in `[0, P-1]` is claimed by some placement" — it is exactly
what it says, the two ends — and the driver checks nothing stronger.

| Flag | Meaning |
|---|---|
| `--comp1 a:b`, `--comp2 c:d` | required; inclusive rank ranges for the two components |
| `--coupler e:f` | routes the "link" channel's Remap through a separately placed coupler on ranks `[e:f]` — `comp1 -> coupler -> comp2` (`[](placement.md#a-separately-placed-coupler)`); refused only together with `--mesh` |
| `--entities N` | total entities in the shared region; default 100 |
| `--steps K` | coupling periods to run; default 1 |
| `--check` | verify every destination entity holds its own source value; non-zero exit and a message on every rank if not |
| `--timing out.csv` | write a per-phase timing CSV (schema below) from rank 0 |

## Refusals

The driver refuses, with a distinct `E-LAYOUT-*` code and a message on rank 0, rather than
guessing or hanging:

- a range that is not `a:b`, has a non-integer or missing endpoint, or has `a > b` (reversed) —
  `E-LAYOUT-PARSE`, `E-LAYOUT-RANGE-EMPTY`, `E-LAYOUT-RANGE-REVERSED`;
- any endpoint outside `[0, P-1]` — `E-LAYOUT-RANK-OUT-OF-RANGE`;
- a layout whose two ends are not `0` and `P-1` — `E-LAYOUT-BOUNDS`;
- `--coupler` together with `--mesh` — `E-LAYOUT-COUPLER: --coupler is not supported with --mesh`.
  `--mesh`/`--migrate` mode never builds the field-transfer coupling graph a coupler's `remap_on`
  could attach to (see `Args::mesh_spec`'s own comment), so the two flags together are refused
  rather than silently ignoring one. Plain `--coupler` (no `--mesh`) is not a refusal: it builds the
  graph with a coupler on `[e:f]` and routes the channel's Remap through it. `--coupler`'s own range
  is still parsed and validated first, and folded into the `0`/`P-1` check above, so a malformed or
  out-of-bounds coupler range is reported as that specific problem rather than masked by anything
  else.

Every one of these is reached before any MPI collective call: the command line is identical on
every rank under `mpiexec`, so the refusal decision is identical on every rank too, and nothing can
diverge and leave a peer blocked in a later collective. After that point — `Runner::initialize()`,
each `run_period()`, and the `--check` reduction — the driver takes an `MPI_Allreduce(MAX)` of a
local pass/fail flag before any rank acts on it, the same discipline
`test/support/fuse_test_main.cpp` uses to fail a whole job uniformly.

## Shapes

Printed on rank 0 before the run, for the `comp1`/`comp2` pair: `identical` (both ranges are the
same set), `disjoint` (no rank in common), or `partially overlapping` (some ranks in common, some
not — see [](placement.md#shared-identical-and-disjoint-placements)). A partial overlap needs at
least 3 ranks to be genuine: at 2 ranks, `comp1 = 0:0` and `comp2 = 1:1` is disjoint, not partial.

## Registered layouts

`examples/CMakeLists.txt` registers 11 valid comp1/comp2 layouts, 3 + 4 + 4 across 2, 3 and 4 ranks:
at 2 ranks, identical plus disjoint in both orderings (partial overlap is not possible below 3
ranks); at 3 and 4 ranks, identical once, disjoint once (comp1 lower), and partial overlap in both
orderings (comp1 lower, comp2 lower).

Plus 3 `--coupler` rows (`layout.coupler.*`): a coupler disjoint from both comp1 and comp2 (3
ranks), a coupler identical to comp2's own range (3 ranks — the coupler->target hop elides), and a
coupler overlapping comp1 without being identical to it (4 ranks).

Plus 4 `WILL_FAIL` refusal cases (one each for `min != 0`, `max != P-1`, a reversed range, and
`--coupler` together with `--mesh`), all labelled `mpi;parallel;layout`:

```bash
ctest -L layout            # list them
ctest -L layout -R layout\\.4\\.   # just the 4-rank cell
```

`build-evidence/parallel-matrix.toml`'s `A-LAYOUT` cell records this as a `registration`-kind claim
against that list — see [](../dev/testing.md) for what the parallel matrix does and does not
guarantee.

## The backend behind it

The driver needs a `MeshBackend` that decomposes the shared mesh independently for each component
(`MeshBackend::placed_view`) — without that, a partially overlapping pair cannot both cover the
region, because the runtime would hand them the same single decomposition. The only such backend in
the tree today is `PartitionedTestBackend`
(`test/support/partitioned_test_backend.hpp`), shared verbatim between
`test/parallel/runner_transport_test.cpp` and this driver rather than duplicated.

## Timing CSV

`--timing out.csv` writes one row per phase — `initialize` (mesh load through
`Runner::initialize()`, including `CommGraph` construction) and `period_1` .. `period_K` — reduced
over every rank with `MPI_MIN`, `MPI_SUM / ranks` and `MPI_MAX`:

```text
# fuse_example_layout_driver timing CSV
# schema: phase,ranks,layout,n,min_s,mean_s,max_s
phase,ranks,layout,n,min_s,mean_s,max_s
initialize,4,c1_0-2_c2_1-3,200,0.000412,0.000501,0.000633
period_1,4,c1_0-2_c2_1-3,200,0.000021,0.000024,0.000029
...
```

`layout` is `c1_<lo>-<hi>_c2_<lo>-<hi>`, the same two ranges named on the command line.

## Scaling sweeps

`tools/scale-sweep.py` runs the driver over a list of rank counts and placement shapes
(`identical`, `disjoint`, `partial`), in **strong** mode (a fixed total `--entities` at every rank
count) or **weak** mode (`--entities` scaled by the rank count, so the work per rank stays fixed),
and concatenates every run's `--timing` CSV into one file with its own documented schema (run
`--help` or read the module docstring for the exact columns):

```bash
python3 tools/scale-sweep.py --dry-run --ranks 2,3,4 --shapes identical,disjoint,partial \
    --mode strong --entities 200 --steps 5
```

`--dry-run` prints the `mpiexec` command line for every cell and runs nothing — the way to check the
sweep before spending any wall time on it. Without `--dry-run` it needs `--driver` pointing at a
built `fuse_example_layout_driver` and an `mpiexec` on `PATH` or given via `--mpiexec`.

:::{important}
**This machine has 12 cores.** A sweep that asks for more ranks than that is oversubscribed —
`mpiexec` may still run it, but the timing numbers measure contention, not scaling. Real strong- or
weak-scaling numbers past a handful of ranks need a cluster; this script only shapes the sweep and
the CSV, it does not manufacture cores.
:::
