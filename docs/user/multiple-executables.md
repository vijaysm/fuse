# Coupling across separately built executables

Use this page when the components of one coupling live in different executables: for example a
Fortran ocean model and a C++ atmosphere model that each have their own build. It shows the two ways
to launch them (one `mpiexec` with several programs, or independently launched programs that find
each other at run time), the graph declarations each needs, and what `connect()` checks.

Every other page in this guide assumes one `mpiexec` launches one executable that contains every
component of the graph. `fuse::transport::connect()` (`include/fuse/transport/connect.hpp`) is for
the two cases that put more than one *executable* into the same coupled run. Both need MPI; see
[](build.md). The worked example is [](brusselator.md).

## Launcher-style MPMD: one `mpiexec`, several programs

```
mpiexec -n 2 ./ocean : -n 3 ./atmosphere
```

`MPI_COMM_WORLD` already spans both programs — there is one world, numbered 0 through 4 in launch
order, and `MPI_APPNUM` tells each rank which segment of the command line it came from. Declare
each component's launch style on the graph:

```cpp
g.add_component({.name = "ocean", .ranks = {0, 2},
                 .launch = {.mode = fuse::LaunchMode::Mpmd, .appnum = 0}});
g.add_component({.name = "atmosphere", .ranks = {2, 3},
                 .launch = {.mode = fuse::LaunchMode::Mpmd, .appnum = 1}});
```

Every executable in the launch links the whole program and builds this SAME graph — exactly as a
single-executable run does; only the component each one actually drives differs. Call `connect()`
once, early, with this process's own `MPI_COMM_WORLD`:

```cpp
fuse::transport::ConnectOptions opts;
opts.my_component = my_component_id;        // which of the two this executable plays
auto base = fuse::transport::connect(graph, MPI_Comm_c2f(MPI_COMM_WORLD), opts);
if (!base.ok()) { /* E-CONNECT-SPEC, E-CONNECT-APPNUM: see below */ }
```

`connect()` reads `MPI_APPNUM`, gathers it across the world, and checks it against the graph: every
appnum some rank reports must name a declared `Mpmd` component, and that component's `ranks` must be
exactly the ranks reporting it — not a superset, not a subset. On agreement it returns a duplicate of
`MPI_COMM_WORLD`, already numbered correctly; hand it to the Runner exactly as a single-executable run
does:

```cpp
fuse::RunnerConfig cfg;
cfg.transport = "mpi";
cfg.base_communicator = base.value().base_comm_f;
cfg.transport_setup = [&](fuse::Transport& t) {
  return static_cast<fuse::MpiTransport&>(t).set_communicator(base.value().base_comm_f);
};
runner.initialize(components, registry, cfg);
```

## True separate-executable MPMD: independent launches

```
mpiexec -n 2 ./ocean &
mpiexec -n 3 ./atmosphere &
wait
```

Here the two programs do not share a parent communicator at all — each has its own, private
`MPI_COMM_WORLD` numbered from 0. `connect()` forms the shared communicator itself, over an
MPI intercommunicator built with `MPI_Open_port`/`MPI_Comm_accept`/`MPI_Comm_connect`. Declare which
side publishes the port and where:

```cpp
// ocean's declaration:
g.add_component({.name = "ocean", .ranks = {0, 2},
                 .launch = {.mode = fuse::LaunchMode::Connect,
                            .is_acceptor = true,
                            .port_file = "/shared/fs/ocean-atmosphere.port"}});
// atmosphere's declaration (same graph, same port_file, is_acceptor left false):
g.add_component({.name = "atmosphere", .ranks = {2, 3},
                 .launch = {.mode = fuse::LaunchMode::Connect,
                            .port_file = "/shared/fs/ocean-atmosphere.port"}});
```

`port_file` must name a path both executables' processes can see (a shared filesystem, or a
directory a job script creates before either launches). The acceptor's rank 0 calls
`MPI_Open_port`, writes the resulting name to that file, and every rank of its own world calls
`MPI_Comm_accept`; the other side polls for the file and calls `MPI_Comm_connect` once it appears.
`connect()`'s call is otherwise identical to the MPMD case above — same `ConnectOptions`, same
return type, same way of handing the result to `RunnerConfig`.

:::{warning}
`ConnectOptions::timeout_s` (default 30 seconds) bounds how long the non-acceptor side polls for the
port file before refusing with `E-CONNECT-TIMEOUT`; that side can give up and exit. The acceptor has
no such bound: once its rank 0 has published the port name, every rank of its world blocks in
`MPI_Comm_accept`, so a connector that times out and exits leaves the acceptor waiting. End that wait
with job control (a wall-clock limit on the launching script or scheduler). The `MPI_Comm_connect`
handshake itself, once the port file is found, cannot be bounded from one side either.
:::

## What `connect()` checks, and what it refuses

Both launch styles are checked by the same rule before any MPI call happens:
**the launched components' `RankSet`s must be consecutive blocks, in the graph's own component
order, with nothing but in-process/plugin ranks between them.** `MPI_Intercomm_merge` concatenates
one side's ranks after the other's in whichever order the two sides agree to merge in, and
launcher-style MPMD assigns rank 0..k-1 to the first `mpiexec` segment and so on in command-line
order; both agree with the graph's own order only if that order is what the graph declares. A
graph that declares an `Mpmd` or `Connect` component on an empty, strided, or out-of-sequence
`RankSet` is refused with `E-CONNECT-SPEC` before any process even calls `MPI_Open_port` or reads
`MPI_APPNUM`.

**A gap between two launched blocks is legal exactly when an in-process component fills it exactly**:
an `InProcess`/`Plugin` component whose `RankSet` covers every rank of the gap and none outside it.
This is how a coupler bundled into one of the two executables is declared. For example, the
acceptor executable carries the coupler's ranks as well as its own (only two `Connect`-mode
components are supported, see `E-CONNECT-SPEC`); the acceptor's declared `RankSet` is just its
physics ranks, and the coupler's `RankSet` is the gap between that block and the connector's. A
declaration where the acceptor's range spans the coupler's ranks too, with no gap, also validates. An
in-process component may share ranks with one launched block (overlap is legal), but it may not
straddle the boundary between a gap and a block, or between two gaps: its ranks must sit entirely on
one side.

| Code | When |
|---|---|
| `E-LAUNCH-APPNUM` | (graph build) an `Mpmd` component has no `appnum` set |
| `E-LAUNCH-CONNECT-TARGET` | (graph build) a `Connect` component has neither `port_file` nor `service_name` |
| `E-CONNECT-SPEC` | the launched components' `RankSet`s are not consecutive blocks (with in-process-covered gaps) in graph order; a gap between two launched blocks has a rank no in-process/plugin component covers, or one an in-process component's own ranks straddle the boundary of; two `Mpmd` components share one `appnum`; a non-`Mpmd`/`Connect` component was handed to `connect()`; or more than two components declare `Connect` (`Mpmd` has no such limit) |
| `E-CONNECT-APPNUM` | the real `MPI_APPNUM` layout at runtime disagrees with the graph (an undeclared appnum, or a declared one whose ranks do not match) |
| `E-CONNECT-TIMEOUT` | the non-acceptor side waited longer than `timeout_s` for the port file |
| `E-CONNECT-ACCEPT-UNAVAILABLE` | a `Connect`-mode component was declared but this build has accept/connect disabled (see [Which launch modes this build offers](#which-launch-modes-this-build-offers)) |

See [](placement.md) for `RankSet` itself, and [](c-abi.md) if the other executable is not written
against this library's own `CouplingGraph` API at all — a participant driven through the C or
Fortran ABI declares its launch style the same way, through the descriptor it was handed.

## A C or Fortran participant never calls `connect()` itself

Everything above is a C++ call against a `fuse::CouplingGraph` the executable built or loaded
itself. A participant written in C or Fortran loads a graph someone else authored
(`fuse_load_graph`) and never touches `fuse::transport::connect()` or any other C++.
`fuse_setup_options::my_component` makes this possible: set it to the name of the component THIS
process plays, with `communicator` holding this executable's own `MPI_COMM_WORLD` (as a Fortran
handle, `MPI_Comm_c2f`).

:::{warning}
On Open MPI, the Fortran handle of `MPI_COMM_WORLD` is 0, which is also the original "no
communicator supplied" sentinel of this struct. Set `has_communicator = 1` so `fuse_setup()` reads a
`communicator` of 0 as this process's own world. A caller whose `struct_size` predates
`has_communicator` keeps the original sentinel (`communicator == 0` means none).
:::

```c
fuse_setup_options opts = {.struct_size = sizeof(opts)};
opts.communicator      = MPI_Comm_c2f(MPI_COMM_WORLD);
opts.has_communicator  = 1;  /* 0 is a real handle on Open MPI */
opts.my_component      = "ocean";
fuse_coupler_abi(FUSE_ABI_VERSION)->setup(ctx, &opts, NULL);
```

`fuse_setup()` reads `my_component`'s `LaunchSpec` from the loaded graph and does what the C++ call
above does: nothing extra for `InProcess`/`Plugin`; `MPI_APPNUM` checking for `Mpmd`; the full
accept/connect/merge handshake for `Connect`. The `communicator` you pass is the world communicator
the executable started with; what `fuse_setup()` stores internally is the real base communicator.
See [](c-abi.md) and [](fortran.md).

### Worked example

`examples/brusselator/participants/{u_main.c,v_main.f90,coupler_main.c}` are the worked example: a
three-component graph authored once by a small C++ tool and loaded by three executables that link no
C++ of their own (only the physics, `shim.cpp`, is C++, behind a plain C wrapper).

```sh
brusselator_graph_tool --ranks-u N --ranks-v N --ranks-coupler N --launch mpmd|accept --out graph.yaml
```

`--launch accept` bundles the coupler's ranks into the acceptor executable (`u_main.c`) rather than
giving it a third `Connect`-mode executable. The exact `mpiexec` lines, as `ctest` runs them (world
size 3):

```
# mpmd: one mpiexec, three ':'-separated app segments
mpiexec -n 1 brusselator_u_main --graph graph.yaml --log u.log --periods 2 \
    : -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 \
    : -n 1 brusselator_coupler_main --graph graph.yaml

# accept: two mpiexec invocations; U's executable is the acceptor and
# carries the coupler's bundled rank in its local MPI_COMM_WORLD
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 &
wait

# both modes
brusselator_check --u-log u.log --v-log v.log
```

`brusselator_check` reads both executables' per-window logs and reproduces the in-process example's
`--check` verdict. `test/parallel/brusselator_participants/run_participants.sh` runs the whole
sequence. Expected agreement with the single-process example: the 3-rank MPMD run matches
bit-for-bit; the 5-rank run (U on two ranks, V on two, coupler on one) differs by about 9e-14 in
each mean. See [](brusselator.md) for the details.

## Releasing the base communicator

`connect()` forms a communicator; `disconnect()` releases it, and every caller must call it
before `MPI_Finalize`:

```cpp
auto connected = fuse::transport::connect(graph, world_comm_f, opts);
{
  // everything that uses connected.value().base_comm_f -- the Runner, the
  // transport, anything that duplicated it -- lives in here
}
if (auto r = fuse::transport::disconnect(connected.value()); !r.ok()) { /* ... */ }
MPI_Finalize();
```

The order is not housekeeping. In `connect` mode the base communicator is merged from an
intercommunicator, which leaves the two programs *connected*: `MPI_Finalize` is then collective
across both of them, and a program that finalizes while still connected leaves an operation
half-finished on a peer that is no longer listening. In MPMD mode the handle is a duplicate of
the caller's own world and `disconnect()` simply frees it; the call is the same either way, which
is the point — a caller does not have to know which mode it ended up in. It is idempotent and safe
on the result of a failed `connect()`.

A participant that reaches FUSE through the C or Fortran ABI has nothing to do: `fuse_destroy()`
releases the connection the same way, after the runner it owns is gone.

## Which launch modes this build offers

The two modes do not ask the same things of an MPI. **MPMD** needs only `MPI_APPNUM`, which every
launcher provides, and is never gated. **Accept/connect** needs dynamic process management *and*
needs it to work between two jobs that share a node, which not every installation can do.

At configure time, with MPI enabled, `tools/probe-mpi-dynamic.sh` compiles a small program and
launches it twice (one job accepting, one connecting), performing the same sequence `connect()`
performs: open a port, accept, connect, merge, collective, disconnect. The verdict sets
`FUSE_ENABLE_CONNECT_ACCEPT`, which you can force on the command line:

```sh
cmake -S . -B build -DFUSE_ENABLE_MPI=ON -DFUSE_ENABLE_CONNECT_ACCEPT=OFF
```

| probe says | `FUSE_ENABLE_CONNECT_ACCEPT` | what you see |
|---|---|---|
| the two jobs connected | `ON` | the accept tests are registered and run |
| they did not | `OFF` | the accept tests are not registered, and a `connect`-mode component is refused at `connect()` with `E-CONNECT-ACCEPT-UNAVAILABLE` |
| could not be tested | `ON` | a configure message says so; the tests run and report for themselves |

The third case is the ordinary one on a cluster login node, where `mpiexec` cannot start a job
without an allocation; an untested probe leaves the mode enabled.

If you see `E-CONNECT-ACCEPT-UNAVAILABLE`, either switch to MPMD launch (one `mpiexec`), or rebuild
against an MPI whose dynamic process management works between two jobs and reconfigure without
forcing `FUSE_ENABLE_CONNECT_ACCEPT=OFF`.

### MPI implementation

Accept/connect needs an MPI whose dynamic process management works between two jobs on the same
node. MPICH's `ch3:nemesis` device does not: both programs abort inside `MPI_Intercomm_merge`. Use
an MPICH built with `--with-device=ch4` (the default since MPICH 4.0) or Open MPI. MPMD is
unaffected, since every rank belongs to one job.

## See also

- [](placement.md): `RankSet` and processor-set placement
- [](c-abi.md), [](fortran.md): driving a participant through the C or Fortran interface
- [](lowfive.md): file-based coupling between executables through HDF5 interception
- [](brusselator.md): the worked example
- [](build.md): enabling MPI and the configure-time probes
