# LowFive: coupling through intercepted HDF5

Use this page when two components already exchange data through HDF5 (one calls
`H5Fcreate`/`H5Dwrite`, the other `H5Fopen`/`H5Dread`) and you want to couple them without
touching either code and without writing a file. LowFive, Argonne's HDF5 Virtual Object Layer
(VOL) connector, intercepts those calls and keeps the data in memory (or sends it over MPI);
FUSE contributes the schedule, the coupling-window semantics, remapping and convergence
control around them. `fuse::LowFiveTransport` (`fuse/transport/transport.hpp`) is the
C++ entry point.

## When to use it

| You have | Use |
|---|---|
| Two codes that already speak HDF5 and must not change | LowFive-bound fields (`hdf5_path`), this page |
| Components that can call FUSE (C++, C, Fortran, Python) | An ordinary transport; see [](concepts.md) and [](c-abi.md) |
| One FUSE-owned process that wants to push a buffer through the HDF5 stack and back | `LowFiveTransport::move()`, see [Single process](#single-process-the-smallest-working-setup) |
| Restart files on disk | [](restart.md) (plain HDF5, never the LowFive VOL) |

There are three ways to use LowFive, in increasing scope:

1. **`LowFiveTransport::move()`** in one process: FUSE itself writes and reads back through
   the connector. Useful as a test and a demonstration that interception works.
2. **Two executables joined by an MPI intercommunicator**, using `set_intercomm()`,
   `serve_as_producer()` and `receive_as_consumer()`. This is [Demo B](demo-b.md).
3. **LowFive-bound channels in a coupling graph**: a field names an `hdf5_path`; the two
   applications perform the actual `H5Dwrite`/`H5Dread`; FUSE schedules around them. This is
   the [Brusselator](brusselator.md) LowFive configuration.

## Prerequisites and enabling it in a build

Configure with `-DFUSE_ENABLE_LOWFIVE=ON` (see [](build.md)). Requirements:

- A LowFive installation that CMake can find (`LowFive_DIR`/`LowFive_ROOT` or
  `CMAKE_PREFIX_PATH`). It must be a **shared** library; a static LowFive is refused at
  configure time.
- A **shared** HDF5 build, the same one LowFive was built against.
- For anything that crosses processes (usage modes 2 and 3), `-DFUSE_ENABLE_MPI=ON` and a
  LowFive that ships the separate `liblowfive-dist` archive (recorded as `LOWFIVE_HAVE_DIST`).
  The single-process path needs LowFive but not MPI.
- Without `FUSE_ENABLE_LOWFIVE`, `LowFiveTransport::set_location()` refuses with
  `E-TRANSPORT-NOLOWFIVE`, and a graph that uses `hdf5_path` cannot be run through LowFive.

```bash
cmake --preset tier1 -DFUSE_ENABLE_LOWFIVE=ON -DFUSE_ENABLE_MPI=ON
cmake --build --preset tier1
```

### One HDF5 in the process

If you also enable MOAB (`FUSE_ENABLE_MOAB=ON`), the MOAB and the LowFive you link must be
built against the **same** HDF5 installation. Otherwise two independent copies of `libhdf5`
load into one process. Neither the linker nor the loader reports that; each copy has its own
property-list id table and VOL registry, so LowFive's `H5Pset_vol` fails with HDF5's
`"not a file VOL ID"` and interception silently stops applying to MOAB's own HDF5 traffic.

FUSE checks this at configure time. When `FUSE_ENABLE_LOWFIVE` and `FUSE_ENABLE_MOAB` are both
on, `cmake/FuseDependencies.cmake` runs `tools/probe-hdf5-images.sh` against the MOAB, HDF5 and
LowFive that this configure resolved. It counts the distinct `libhdf5` and `libhdf5_hl` images
that actually load and stops with `FATAL_ERROR` if there is more than one. With MOAB off, the
check is skipped.

You can run the probe yourself:

```bash
tools/probe-hdf5-images.sh <lowfive_prefix> [moab_dir] [hdf5_prefix]
```

It exits 0 when exactly one image of each library loads, 1 when the hazard is present (it
names the duplicate paths), and 2 when the probe could not be built or run for an unrelated
reason (so "untested" is never reported as "passed"). Expected output on a good pairing
(exact paths vary by platform):

```
probe-hdf5-images.sh: libhdf5 images loaded (1):
  <prefix>/lib/libhdf5.310.dylib
probe-hdf5-images.sh: libhdf5_hl images loaded (1):
  <prefix>/lib/libhdf5_hl.310.dylib
probe-hdf5-images.sh: PASS -- exactly one libhdf5 and one libhdf5_hl image
```

On a failing pairing the output lists two images per library and ends with
`FAIL -- the two-HDF5-image hazard is present`.

:::{tip}
If the probe fails, rebuild LowFive against the exact HDF5 prefix MOAB uses, not against a
separately built "shared HDF5":

```bash
cmake -B lowfive-build -S LowFive-src \
      -DCMAKE_INSTALL_PREFIX=<new-prefix> \
      -DCMAKE_PREFIX_PATH="<moab-hdf5-prefix>;<diy-prefix>" \
      -DCMAKE_CXX_COMPILER=<mpicxx> \
      -DBUILD_SHARED_LIBS=ON -Dbuild_examples=OFF
cmake --build lowfive-build -j8 && cmake --install lowfive-build
```

CMake's own `FindHDF5` may still pick up a different HDF5 through its CONFIG-mode search of
prefixes such as `/opt/hdf5*`, whatever `CMAKE_PREFIX_PATH` says. If that happens, configure
with `-DHDF5_NO_FIND_PACKAGE_CONFIG_FILE=ON -DHDF5_ROOT=<the HDF5 you want>`.
:::

## Single process: the smallest working setup

`LowFiveTransport` registers LowFive's VOL connector once per process, builds a file-access
property list carrying it, and proves that interception is live before trusting it: it writes
and reads back a probe dataset through the connector, checks that LowFive's own
`after_dataset_write` callback fired, and checks that `stat()` on the path fails (no real
file). `move()` repeats both checks on every call, so a connector that stops intercepting
fails loudly instead of quietly writing files.

```cpp
#include <fuse/transport/transport.hpp>

fuse::LowFiveTransport t;
// The file name is never created, but its directory must exist and be writable.
if (auto r = t.set_location("/tmp/exchange.h5", "/grid/values"); !r.ok()) { /* r.error().code */ }
if (auto r = t.build_plan(fuse::TransportSpec{.elements = 8}); !r.ok())  { /* ... */ }

double src[8] = {1, 2, 3, 5, 8, 13, 21, 34}, dst[8];
if (auto r = t.move(src, dst, 8); !r.ok()) { /* ... */ }   // dst == src, nothing on disk
```

Things to know:

- `n` passed to `move()` must be exactly `elements * components` of the planned spec
  (`E-TRANSPORT-SIZE` otherwise).
- `access_plist_handle()` returns the intercepting file-access property list as a
  `long long` (an HDF5 `hid_t`; negative before `build_plan`). Hand it as the `fapl_id` of
  `H5Fcreate`/`H5Fopen` in an otherwise unmodified HDF5 program and that program's I/O is
  intercepted. `intercepted_writes()` counts the dataset writes the connector has seen; zero
  means nothing went through it.
- The transport is serial and not thread-safe. The VOL singleton and its in-memory file map
  are process-wide, so two transports must name different files; `set_location()` rejects a
  name another live transport already holds.
- `LowFiveTransport` attaches the connector programmatically. Do not load it through
  `HDF5_VOL_CONNECTOR`/`HDF5_PLUGIN_PATH`; see [Troubleshooting](#troubleshooting).

## Wire it into a coupling graph

A field becomes LowFive-bound by naming an HDF5 location. The writing application will
`H5Dwrite` there, the reading application will `H5Dread` there, and FUSE never touches the
bytes.

```yaml
components:
  - name: U
    ranks: {first: 0, count: 1}
    launch: {mode: mpmd, appnum: 0}
    fields:
      - {name: u_out, region: u_all, disc: FV, components: 1, units: 1,
         conservation: Extensive, intent: Provides,
         hdf5_path: brusselator_u.h5/grid/u_out}
  # ... V declares v_out the same way, with hdf5_path: brusselator_v.h5/grid/v_out
couplers:
  - name: lowfive            # marker: "this graph uses LowFive"
    ranks: {first: 0, count: 3}
    launch: {mode: lowfive}
channels:
  - {name: u2v, source: U.u_out, target: V.v_in, remap: {flags: [Conservative]}}
```

- **`hdf5_path`** has the form `"<file>.h5/<dataset path>"`, split at the first `.h5/`. The
  separator must appear exactly once, with a non-empty file name before it and a non-empty
  dataset path after it. A leading `*` in the file component is a wildcard (LowFive's
  `set_memory("*", "*")` pattern).
- **The marker component** declares `launch: {mode: lowfive}` and must be a bare placement
  (`kind: Coupler`, no fields). It is never stepped and owns no fields. Its only job is to
  state that this graph relies on LowFive interception.
- Launch modes combine: the component that holds the field keeps its own `mpmd`/`connect` launch; `lowfive` is
  an independent fifth mode on the marker, not a replacement for those. See
  [](multiple-executables.md).
- Remap, scaling, extrapolation and convergence on a LowFive-bound channel work as on any
  channel: they act on the data the application delivered, so two sides on different meshes
  still get conservative remap. A channel carrying an `accumulate` window keeps its window
  semantics (how many samples per window) too.

`validate()` enforces three rules before anything runs:

| Code | Meaning | Fix |
|---|---|---|
| `E-LOWFIVE-NO-COMPONENT` | A field sets `hdf5_path` but no component declares `launch: {mode: lowfive}` | Add the marker component |
| `E-LOWFIVE-COMPONENT-KIND` | A `lowfive`-launched component is not a bare placement (it has fields, or is not a coupler) | Make it `kind: Coupler` with no fields |
| `E-LOWFIVE-DUPPATH` | Two fields set the same non-wildcard `hdf5_path`; they would overwrite each other's dataset | Give each field its own file/dataset |

`fuse::build_lowfive_bindings()` (`fuse/transport/lowfive_graph.hpp`) turns the graph's fields
into a `LowFiveFieldBindingTable` of qualified `"<component>.<field>"` names, so two components
may reuse a local field name. A hand-built `LowFiveFieldBindingTable` (`fuse/transport/lowfive.hpp`)
applies the same rules and adds `E-LOWFIVE-PORT` (malformed `hdf5_path`, or an empty field
name) and `E-LOWFIVE-DUPFIELD` (a field bound twice).

### What FUSE does for a LowFive-bound channel

FUSE does **not** move the data of such a channel, and `LowFiveTransport::move()` is not
involved. It compiles the channel to an `ExternalDelivery` step that sits where an ordinary
`Transport` would sit: after the producing export, before the channel's remap, diagnostics
and sink. At that point the Runner hands control back to the application, which performs its
ordinary `H5Dwrite` and `H5Dread`. FUSE schedules *when* the writer must have written before
the reader may read, applies the window semantics and any remap, and drives convergence.
No `Transport` and no `Epoch` operation is emitted for the channel's data hop.

## The application side

The two applications keep their ordinary HDF5 code. Three points matter in practice.

1. **Pass the intercepting property list.** In the bundled Brusselator participants the
   `fapl_id` goes literally into every `H5Fcreate`/`H5Fopen` in place of `H5P_DEFAULT`.
   The property list comes from the attach step described under
   [Launch two executables](#launch-two-executables).
2. **Create the parent group before the dataset.** LowFive's `dataset_create` does not create
   intermediate groups. Writing `/grid/u_out` without `H5Gcreate2` for `/grid` leaves a flat
   child literally named `"grid/u_out"`, which a remote reader cannot open correctly (the
   read fails with a dataspace error such as `Unexpected no class in dataspace = -1`).
   Create the group first.
3. **Read with an explicit dataspace.** For a remote dataset, use the dataset's own
   `H5Dget_space()` handle as both the file and memory dataspace in `H5Dread`, not `H5S_ALL`.
   LowFive does not support `H5S_ALL` on a remote read.

A window member that exchanges through a LowFive-bound channel must also honour the C ABI
suspension protocol: check `fuse_step_report::delivery_pending`/`delivery_field` to know when
to do the HDF5 call, and `step_opened` to know whether `fuse_begin_step()` opened a step. See
[](c-abi.md) ("`step_opened`") and [](coupling-windows.md).

## Launch two executables

LowFive-bound channels work across two programs that share one MPI world, launched either as
one MPMD job or as two independent jobs. How the base communicator is formed is the job of the
launch modes described in [](multiple-executables.md); LowFive builds on it.

For a cross-process exchange LowFive uses `DistMetadataVOL`, which carries an MPI
**intercommunicator** between the writer's process group and the reader's. FUSE does not form
that intercommunicator for you in the low-level API: you hand it one.

### Low-level API (Demo B)

- `LowFiveTransport::set_intercomm(local_comm, intercomm)`: call before `set_location()`.
  Switches the process onto `DistMetadataVOL`. Both communicators cross the interface as
  Fortran handles (`MPI_Comm_c2f`), so the header never includes `mpi.h`. The choice is
  process-wide and one-time. Errors: `E-TRANSPORT-NOLOWFIVE`, `E-TRANSPORT-NOMPI` (built
  without `FUSE_ENABLE_MPI`), `E-TRANSPORT-MPIINIT`, `E-TRANSPORT-COMM` (not a valid
  intercommunicator, for example an ordinary intracommunicator), `E-TRANSPORT-LF-LOCATION`
  (called after `set_location`/`build_plan`).
- `serve_as_producer()`: the producer calls it after its own unmodified `H5Fclose`; it pushes
  the file across the intercommunicator.
- `receive_as_consumer()`: the consumer calls it before its own unmodified `H5Fopen`.
- Both refuse with `E-TRANSPORT-LF-NOTDIST` if `set_intercomm()` was never called, and
  `move()` refuses with `E-TRANSPORT-LF-DISTRIBUTED` once it has been: writing and reading
  back in one call is meaningless when the two ends are different processes.

[Demo B](demo-b.md) is the complete example: `producer_hdf5.cpp`/`consumer_hdf5.cpp` are plain
HDF5 with no FUSE header, and `producer_main.cpp`/`consumer_main.cpp` form the rendezvous,
attach the transport and call into the unmodified payload.

### Graph-driven launch (Brusselator)

For a graph-driven run, the participants build the connector from the same `--graph` file
they already load. Their attach step reads the field bindings and the writer/reader rank sets
from the graph, so no separate bindings variable is needed. The intercommunicator is formed
over the merged base communicator that FUSE sets up for the component (see
[](multiple-executables.md) and `base_communicator()` in [](c-abi.md)). Every rank sharing that
communicator, including the coupler rank that hosts no LowFive channel, takes part in one
collective split; only ranks that host a qualifying channel go on to attach.

World size 3 (one U rank, one V rank, one coupler rank), one MPMD job:

```bash
mpiexec -n 1 brusselator_u_main --graph graph.yaml --log u.log --periods 2 --transport lowfive \
    : -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --transport lowfive \
    : -n 1 brusselator_coupler_main --graph graph.yaml
brusselator_check --u-log u.log --v-log v.log
```

The same coupling as two independent jobs (accept/connect), U's executable carrying the
coupler's bundled rank:

```bash
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 --transport lowfive &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --transport lowfive &
wait
brusselator_check --u-log u.log --v-log v.log
```

`--transport fuse|lowfive` selects between the ordinary transports and LowFive in each
participant. `brusselator_graph_tool` writes the matching graph (`--transport lowfive`,
`--launch mpmd`, `--ranks-u`, `--ranks-v`, `--ranks-coupler`, `--out`, `--port-file`). A
LowFive-bound run produces logs identical to the same run over an ordinary transport, which
`brusselator_check` verifies. See [](brusselator.md).

### Serve and receive ordering

With `DistMetadataVOL`, a writer's file is held until the peer has opened **and** closed it.
A process blocked serving a file is released only by that peer-side open-and-close, not by the
peer merely reaching an `H5Fopen`. Two components that each serve first and read second would
therefore wait on each other forever. The bundled attach step avoids this:

- it disables serve-on-close and keeps the file after `H5Fclose`, so the application's own
  close neither serves nor discards it; it serves explicitly at a defined point;
- a first-server serves immediately after each write; a first-reader defers its cold-start
  write and serves it after its own first read of the peer's file has closed; every later
  write, by either side, serves immediately;
- each exported file is dropped right after it has been served, before the same component's
  next `H5Fcreate` of the same name, so a later serve never delivers a previous period's data;
- at exit, a deferred serve that was never flushed is refused.

The writer/reader order that results follows the order in which the graph declares its
channels. The helper checks that order statically at attach time and again before every
blocking call (see the table below).

## What is supported

- Interception of `H5Fcreate`/`H5Dwrite`/`H5Fopen`/`H5Dread` between two unmodified HDF5
  programs, in one process or across processes over an MPI intercommunicator.
- In the bundled distributed participants, one dataset per bound field, with a distinct file
  for each direction.
- Coupling-window semantics, interpolation, conservative remap, extrapolation, and convergence
  on LowFive-bound channels.
- MPMD launch (one `mpiexec`) and separately launched accept/connect jobs.

## Limitations

- **Static HDF5 or LowFive:** refused at configure time (the VOL must be loadable).
- **Two HDF5 images:** refused at configure time when MOAB and LowFive are both enabled and
  the probe finds more than one `libhdf5`.
- **`move()` in distributed mode:** refused (`E-TRANSPORT-LF-DISTRIBUTED`).
- **Cross-process runs are exercised with one rank per side.** More than one rank per side,
  more than one file or dataset per `LowFiveTransport`, and a second exchange in the same
  `LowFiveTransport` process are not covered.
- **The bundled graph-driven attach is validated for one shape:** two components (U and V),
  one rank each, exchanging two LowFive-bound channels in the order U to V, then V to U. Any
  other graph is refused at attach time with `E-LOWFIVE-ATTACH-ORDER-UNVALIDATED`, naming the
  first differing property.
- **More than two Connect-mode components** in a LowFive graph are refused
  (`E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY`).
- **One intercommunicator per process:** every LowFive-bound channel a rank takes part in must
  split the world into the same two groups (`E-LOWFIVE-ATTACH-INCONSISTENT` otherwise).
- **Writers and readers must be disjoint rank sets.** A rank that is both writer and reader
  is the single-process case, which uses `MetadataVOL` and not the distributed path
  (`E-LOWFIVE-RENDEZVOUS-OVERLAP`).
- **Not thread-safe:** the VOL and its in-memory file map are process-wide. The bundled
  participants serialize the two directions of the exchange rather than overlapping them.
- **No total-length check in the graph:** `validate()` checks a field's per-point component
  count, not the total runtime array length. The bundled shim checks each read's extent
  against the dataset's actual size and refuses a mismatch.

## Error codes

| Code | Meaning | What to do |
|---|---|---|
| `E-TRANSPORT-NOLOWFIVE` | Build has no LowFive | Configure with `-DFUSE_ENABLE_LOWFIVE=ON` |
| `E-TRANSPORT-LF-LOCATION` | Bad file/dataset name, unwritable directory, name already claimed by another live transport, or a call out of order (`build_plan` before `set_location`, `set_intercomm` after `set_location`) | Give a non-empty name in an existing, writable directory; follow the call order |
| `E-TRANSPORT-LF-VOL` | The connector could not be registered or attached | Check the LowFive/HDF5 pairing (see the probe above) |
| `E-TRANSPORT-LF-NOINTERCEPT` | The probe or a `move()` showed the connector did not intercept | The connector is not live; do not proceed with file I/O assumptions |
| `E-TRANSPORT-LF-ONDISK` | A real file appeared at the path | Interception was bypassed; check the property list is the one passed to HDF5 |
| `E-TRANSPORT-LF-NOTDIST` | `serve_as_producer`/`receive_as_consumer` without `set_intercomm` | Call `set_intercomm()` first |
| `E-TRANSPORT-LF-DISTRIBUTED` | `move()` after `set_intercomm()` | Use `serve_as_producer`/`receive_as_consumer` |
| `E-TRANSPORT-SIZE` | `n` is not `elements * components` of the planned spec | Pass the planned length |
| `E-LOWFIVE-PORT` | Malformed `hdf5_path` (no or repeated `.h5/`, empty file or dataset part) or empty field name | Use `"<file>.h5/<dataset path>"` |
| `E-LOWFIVE-DUPFIELD` | A field bound to two paths | One `hdf5_path` per field |
| `E-LOWFIVE-DUPPATH` | Two fields share one non-wildcard path | Distinct paths |
| `E-LOWFIVE-NO-COMPONENT` | `hdf5_path` set but no `lowfive` marker component | Add the marker |
| `E-LOWFIVE-COMPONENT-KIND` | The `lowfive` marker owns fields or is not a coupler | Make it a bare coupler placement |
| `E-LOWFIVE-RENDEZVOUS-OVERLAP` | Writer and reader rank sets share a rank | Use disjoint rank sets |
| `E-LOWFIVE-RENDEZVOUS-UNPLACED` | A rank that is in neither rank set asked for the rendezvous | Only writer and reader ranks attach; the coupler rank must not |
| `E-LOWFIVE-ATTACH-NOCHANNELS` | Attach called by a rank with no LowFive-bound channel | Do not call attach on such a rank |
| `E-LOWFIVE-ATTACH-ACCEPT-NO-BRIDGE` | Connect-mode attach with no base communicator formed | Call attach after `fuse_setup()` succeeds, with the base communicator |
| `E-LOWFIVE-ATTACH-ACCEPT-UNSUPPORTED-TOPOLOGY` | More than two Connect-mode components | Reduce to two |
| `E-LOWFIVE-ATTACH-INCONSISTENT` | Channels do not all partition the world into the same two groups | Make all channels share one writer/reader partition |
| `E-LOWFIVE-ATTACH-ORDER-UNVALIDATED` | The graph's channel order is not the validated shape | See Limitations |
| `E-LOWFIVE-ATTACH-ORDER-DEVIATION` | A blocking call was about to run out of the expected write/read order (names expected vs. observed event and slot) | Check the participant's HDF5 call order against the graph's channel order |
| `E-LOWFIVE-ATTACH-DATASET-EMPTY-PATTERN` | A binding has an empty dataset component | Give the dataset a non-empty path |
| `E-LOWFIVE-ATTACH-SERVE-OVERLAP` / `-SERVE-EMPTY` | The two components' rank sets overlap, or one is empty | Fix the rank sets |
| `E-LOWFIVE-ATTACH-SERVE-DOUBLE-DEFER` / `-SERVE-UNFLUSHED` / `-SERVE-BADEVENT` | The serve sequence was violated: a second deferral, a deferred serve left unflushed at exit, or an unrecognised event | Indicates a call-order error in the participant; check it against the ordering rules above |

## Diagnostics

Both variables below are read by the bundled Brusselator attach step and are off by default.

- **`FUSE_LOWFIVE_ATTACH_TRACE=1`** (exactly the character `1`) prints `TRACE(...)` lines to
  stderr: one at attach time (world rank, component, role `first-server`/`first-reader`,
  whether the attach is distributed, the registered file and dataset patterns, the
  `serve_on_close` value actually set); one before and after every `bruss_lowfive_write`/
  `bruss_lowfive_read`; one when a write or read is skipped because a window has no samples;
  and one before and after every explicit serve, with elapsed seconds. A serve that logs an
  "enter" line but never an "exit" line is the hang.
- **`FUSE_LOWFIVE_LOG=trace|debug|info`** turns on LowFive's own logger (for example
  `enter serve_all, files.size = N`). Use `info` first; `trace` is large.

Under `ctest` the variables pass through to every rank because the MPI launcher forwards the
launching shell's environment:

```bash
FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info \
    ctest -R 'participants\.lowfive\.mpmd\.3' --output-on-failure
```

Per-rank stdout and stderr land in the test's scratch directory (`u_stdout.<rank>.log`,
`v_stdout.<rank>.log`, `coupler_stdout.<rank>.log`).

`FUSE_LOWFIVE_BINDINGS=<graph.yaml>` is read only by the optional loadable VOL plugin
`fuse-lowfive-vol` (`H5PLget_plugin_type`/`H5PLget_plugin_info`), which HDF5 loads through
`HDF5_VOL_CONNECTOR` and `HDF5_PLUGIN_PATH`. The bundled participants do not use it. If it is
unset the plugin warns on stderr and applies no bindings, so HDF5 calls then reach real files.

## Troubleshooting

**A process aborts after `main` returns with `mutex lock failed: Invalid argument` (exit 134).**
This is the abort seen when LowFive's connector is loaded through `HDF5_VOL_CONNECTOR`/
`HDF5_PLUGIN_PATH` with no teardown fix: HDF5's `atexit` handler terminates the connector after
LowFive's own C++ statics (its logger and that logger's mutex) are already destroyed. All
program output has been produced by then. `LowFiveTransport` avoids it by registering
`std::atexit(+[]{ H5close(); })` when it constructs the VOL singleton, which runs ahead of
HDF5's own handler. The bundled participants call `H5close()` explicitly before
`MPI_Finalize()`. If you attach the connector yourself, do the same, and attach it
programmatically (`LowFive::H5VOLProperty::apply()` on a file-access property list) rather
than through the environment variables.

**`H5Pset_vol` reports `not a file VOL ID`.** Two HDF5 images are loaded. See
[One HDF5 in the process](#one-hdf5-in-the-process).

**A LowFive-linked binary fails to start with `Library not loaded` (a `libfmt`, `libmpi` or
`libmpicxx` version).** Your LowFive install was built against a different fmt or MPI than the
one present. Rebuild LowFive against the toolchain you build FUSE with.

**A cross-process run hangs.**
1. Run with `FUSE_LOWFIVE_ATTACH_TRACE=1 FUSE_LOWFIVE_LOG=info` and look for a serve with an
   "enter" and no "exit". Both sides blocked in a serve, or both blocked in `H5Fopen`, means
   the write/read order does not match what the graph's channel order produces; see
   [Serve and receive ordering](#serve-and-receive-ordering).
2. Confirm every rank that shares the base communicator joined the collective split,
   including the coupler rank. A rank that skips it blocks the others.
3. Confirm the coupler rank does not attach (`E-LOWFIVE-RENDEZVOUS-UNPLACED`).

**A reader gets the previous period's data.** The writing side did not drop its served file
before creating the same file again; the bundled attach step does this automatically. A custom
attach must do the same.

**`file_open(): no intercomm found for <file>`.** The reader's attach did not register an
intercommunicator for that file name. Register one per binding with
`DistMetadataVOL::set_intercomm()` and give the dataset pattern in LowFive's slashed form
(`/grid/u_out`).

**A reader fails with `Unexpected no class in dataspace = -1`.** The writer did not create the
parent group; see [The application side](#the-application-side).

**Data never crosses (reader sees an empty file) and `H5Fclose` did not serve.** With
`serve_on_close` off, LowFive only keeps the file if `keep` is on. Set both together:
`serve_on_close=false` requires `keep=true`.

## Maturity

Single-process interception is covered by the automated test suite (`test/transport/lowfive_test.cpp`).
The cross-process path is exercised by the Demo B example (one producer process and one consumer
process, values compared bit for bit with no file on disk) and by the Brusselator LowFive MPMD
and accept/connect configurations at the rank counts above. Both need MPI; see
[](build.md).

## See also

- [](build.md): enabling LowFive, MPI and MOAB in a build.
- [](multiple-executables.md): MPMD and accept/connect launches that LowFive builds on.
- [](demo-b.md): two unmodified HDF5 programs coupled by interception alone.
- [](brusselator.md): the graph-driven LowFive configuration.
- [](c-abi.md): the suspension protocol for window members (`step_opened`, `delivery_pending`).
- [](restart.md): restart files use plain HDF5, never the LowFive VOL.
- [](troubleshooting.md): general diagnostics.
