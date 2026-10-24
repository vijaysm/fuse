# Coupling across separately built executables

Use this page when the components of one coupling live in different executables: for example a
Fortran ocean model and a C++ atmosphere model that each have their own build. It shows how to
choose between the ways of putting several programs (or several languages) into one run, the two
launch styles for separate executables (one `mpiexec` with several programs, or independently
launched programs that find each other at run time), the graph declarations each needs, what
`connect()` checks, and how a C or Fortran participant gets the same result without calling any
C++.

Every other page in this guide assumes one `mpiexec` launches one executable that contains every
component of the graph. `fuse::transport::connect()` (`include/fuse/transport/connect.hpp`) is for
the cases that put more than one *executable* into the same coupled run. Both need MPI; see
[](build.md). The worked example is [](brusselator.md).

## Which arrangement to use

| Arrangement | Processes | Launch | When to use it | Declared as |
|---|---|---|---|---|
| In-process | one executable holds every component | `mpiexec -n N ./app` | all components are C++ (or Python) and can be linked into one program | no `launch:` block (`InProcess`) |
| Plugin | the host executable `dlopen()`s the component | same as the host | a C or Fortran solver that FUSE should drive, hosted by any program that uses the participant ABI | `launch: {mode: plugin, plugin_path: ...}` |
| MPMD | one MPI world, several programs | `mpiexec -n 2 ./a : -n 3 ./b` | separately built executables, one launcher invocation | `launch: {mode: mpmd, appnum: k}` |
| Connect | independent MPI worlds joined at run time | `mpiexec ./a &` then `mpiexec ./b &` | separately launched jobs, or launchers without MPMD; needs dynamic process management | `launch: {mode: connect, is_acceptor: ..., port_file: ...}` |
| LowFive | as MPMD or Connect | as MPMD or Connect | two unmodified HDF5 codes exchanging through intercepted `H5Dwrite`/`H5Dread` | `hdf5_path` on the exchanged fields, plus a fieldless marker coupler with `launch: {mode: lowfive}`; the field-holding components keep their `mpmd`/`connect` launch; see [](lowfive.md) |

A graph may mix these per component: a plugin next to an MPMD executable is fine; a plugin next to
a Connect executable has one restriction ([](providers.md#a-known-failure-case)). Python components
run in-process only: the Python module has no MPI binding and no participant ABI, so a Python
program cannot be one of the executables of an MPMD or Connect run and cannot host plugins. It can
author and validate the YAML the other programs load; see [](python.md) and
[](mixing-languages.md).

## The launch blocks in YAML

The same declarations in the YAML a C or Fortran participant loads (`examples/brusselator/
participants/reference_mpmd_3.yaml` and `reference_accept_3.yaml` are complete files):

```yaml
components:
  - name: ocean
    ranks: {first: 0, count: 2}
    launch: {mode: mpmd, appnum: 0}
  - name: atmosphere
    ranks: {first: 2, count: 3}
    launch: {mode: mpmd, appnum: 1}
```

```yaml
components:
  - name: ocean
    ranks: {first: 0, count: 2}
    launch: {mode: connect, is_acceptor: true, port_file: /shared/fs/ocean-atmosphere.port}
  - name: atmosphere
    ranks: {first: 2, count: 3}
    launch: {mode: connect, is_acceptor: false, port_file: /shared/fs/ocean-atmosphere.port}
```

The keys a `launch:` block accepts are `mode`, `appnum`, `is_acceptor`, `port_file`,
`service_name` and `plugin_path`; any other key is refused at load. An `mpmd` component without
`appnum` is refused with `E-LAUNCH-APPNUM`; a `connect` component with neither `port_file` nor
`service_name` with `E-LAUNCH-CONNECT-TARGET`. A coupler placement (`couplers:`) takes the same
block.

## Launcher-style MPMD: one `mpiexec`, several programs

```
mpiexec -n 2 ./ocean : -n 3 ./atmosphere
```

`MPI_COMM_WORLD` already spans both programs: there is one world, numbered 0 through 4 in launch
order, and `MPI_APPNUM` tells each rank which segment of the command line it came from. Declare
each component's launch style on the graph:

```cpp
g.add_component({.name = "ocean", .ranks = {0, 2},
                 .launch = {.mode = fuse::LaunchMode::Mpmd, .appnum = 0}});
g.add_component({.name = "atmosphere", .ranks = {2, 3},
                 .launch = {.mode = fuse::LaunchMode::Mpmd, .appnum = 1}});
```

Every executable in the launch builds (or loads) this same graph, exactly as a single-executable
run does; only the component each one drives differs. Call `connect()` once, early, with this
process's own `MPI_COMM_WORLD`:

```cpp
fuse::transport::ConnectOptions opts;
opts.my_component = my_component_id;        // which of the two this executable plays
auto base = fuse::transport::connect(graph, MPI_Comm_c2f(MPI_COMM_WORLD), opts);
if (!base.ok()) { /* E-CONNECT-SPEC, E-CONNECT-APPNUM: see below */ }
```

`connect()` reads `MPI_APPNUM`, gathers it across the world, and checks it against the graph: every
appnum some rank reports must name a declared `Mpmd` component, and that component's `ranks` must be
exactly the ranks reporting it, not a superset, not a subset. On agreement it returns a duplicate of
`MPI_COMM_WORLD`, already numbered correctly; hand it to the Runner exactly as a single-executable
run does:

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

Here the two programs do not share a parent communicator at all; each has its own private
`MPI_COMM_WORLD` numbered from 0. `connect()` forms the shared communicator itself, over an MPI
intercommunicator built with `MPI_Open_port`/`MPI_Comm_accept`/`MPI_Comm_connect`. Declare which
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
The two sides are then joined with `MPI_Intercomm_merge`, ordered by the graph's component order,
not by which side accepted. `connect()`'s call is otherwise identical to the MPMD case: same
`ConnectOptions`, same return type, same way of handing the result to `RunnerConfig`.

`ConnectOptions` has three members:

| Member | Default | Meaning |
|---|---|---|
| `my_component` | (required) | the `ComponentId` this executable's world plays |
| `timeout_s` | `30.0` | how long the non-acceptor side polls for the port file before `E-CONNECT-TIMEOUT` |
| `poll_interval_s` | `0.05` | how often it re-checks for the file |

:::{warning}
`timeout_s` bounds only the wait for the port file; that side can give up and exit. The acceptor
has no such bound: once its rank 0 has published the port name, every rank of its world blocks in
`MPI_Comm_accept`, so a connector that times out and exits leaves the acceptor waiting. End that wait
with job control (a wall-clock limit on the launching script or scheduler). The `MPI_Comm_connect`
handshake itself, once the port file is found, cannot be bounded from one side either.
:::

### More than two connect-mode executables

A graph may declare any number of `connect` components. They are joined one at a time, in the
graph's component order: step 1 joins the first two as above (one of them is the acceptor);
at every later step the joining component must itself declare `is_acceptor: true` and its own
`port_file`, and the already merged group is the connector waiting on that file. A later joining
component declared with `is_acceptor: false`, or with `service_name` instead of `port_file`, is
refused with `E-CONNECT-SPEC` naming the component and the step. `service_name` is accepted for
the two-component case only. `test/parallel/connect/run_chain.sh` launches three- and four-way
chains.

Set the environment variable `FUSE_CONNECT_CHAIN_TRACE=1` (exactly `1`) in every program's
environment to print one line per rendezvous step per rank to `stderr`
(`FUSE_CONNECT_CHAIN_TRACE: <function> step=<k> rank=<r>: <what>`), which tells a chain that
deadlocks apart from an accept/connect that never completed. It is off by default and must be
set in the environment the launched processes inherit, not only in the launching shell's.

## What `connect()` checks, and what it refuses

Both launch styles are checked by the same rule, `fuse::transport::validate_launch_layout(g)`,
before any MPI call happens:

- Only `mpmd` and `connect` components form rank blocks. Each such block must be a single
  contiguous, non-empty rank set.
- The blocks must be consecutive in the graph's own component order, starting at rank 0:
  `MPI_Intercomm_merge` concatenates one side's ranks after the other's, and launcher-style MPMD
  assigns ranks to `mpiexec` segments in command-line order, so the graph's order is the one both
  agree with.
- A gap between two blocks is legal exactly when in-process or plugin components fill it exactly:
  their rank sets together cover every rank of the gap and none outside it, and no single
  in-process component straddles the boundary of a gap (its ranks sit entirely inside one gap, or
  entirely inside blocks). This is how a coupler bundled into one of the executables is declared:
  the acceptor executable carries the coupler's ranks as well as its own, its declared `ranks` are
  just its physics ranks, and the coupler's `ranks` are the gap between that block and the
  connector's. An in-process component may also overlap a launched block (sharing its ranks).
- A `lowfive` component's rank set is a cross-cutting annotation and takes no part in this check.
- A graph with no `mpmd` or `connect` component at all is refused: there is nothing to connect.

| Code | When |
|---|---|
| `E-LAUNCH-APPNUM` | (graph build or load) an `mpmd` component has no `appnum` |
| `E-LAUNCH-CONNECT-TARGET` | (graph build or load) a `connect` component has neither `port_file` nor `service_name` |
| `E-CONNECT-SPEC` | the layout rule above fails; two `mpmd` components share one `appnum`; `connect()` was asked for an in-process, plugin or lowfive component; fewer than two `connect` components; a chain step whose joining component is not an acceptor with a `port_file` |
| `E-CONNECT-APPNUM` | the real `MPI_APPNUM` layout at run time disagrees with the graph (an undeclared appnum, or a declared one whose ranks do not match) |
| `E-CONNECT-TIMEOUT` | the non-acceptor side waited longer than `timeout_s` for the port file |
| `E-CONNECT-ACCEPT-UNAVAILABLE` | a `connect` component was declared but this build has accept/connect disabled; see [Which launch modes this build offers](#which-launch-modes-this-build-offers) |
| `E-CONNECT-MPI` | an MPI call or the port-file write failed; the message names the call and MPI's own error text |
| `E-CONNECT-FINALIZED` | `disconnect()` was called after `MPI_Finalize` |

A failure that cannot be agreed across ranks (one that happens inside a collective every rank has
already entered) aborts the job with a message on `stderr` instead of returning, so that no rank is
left waiting in a collective its peers have abandoned.

See [](placement.md) for `RankSet` itself.

## A C or Fortran participant never calls `connect()` itself

Everything above is a C++ call against a `fuse::CouplingGraph` the executable built or loaded
itself. A participant written in C or Fortran loads a graph someone else authored
(`fuse_load_graph`) and never touches `fuse::transport::connect()` or any other C++.
`fuse_setup_options::my_component` makes this possible: set it to the name of the component this
process plays, with `communicator` holding this executable's own `MPI_COMM_WORLD` (as a Fortran
handle, `MPI_Comm_c2f` in C; already a Fortran handle in Fortran).

:::{warning}
On Open MPI, the Fortran handle of `MPI_COMM_WORLD` is 0, which is also the original "no
communicator supplied" sentinel of this struct. Set `has_communicator = 1` so `setup()` reads a
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

```fortran
character(len=8), target :: my_component_name
type(fuse_setup_options) :: setup
my_component_name = "ocean" // c_null_char
setup%communicator     = int(MPI_COMM_WORLD, c_int64_t)
setup%has_communicator = 1_c_int32_t
setup%my_component     = c_loc(my_component_name)
status = fuse_setup(ctx, setup, nothing)
```

`setup()` reads `my_component`'s launch mode from the loaded graph and does what the C++ call above
does: nothing extra for in-process or plugin; `MPI_APPNUM` checking for `mpmd`; the full
accept/connect/merge handshake for `connect`, with the default `ConnectOptions` (a 30 s port-file
timeout). The `communicator` you pass is the world communicator the executable started with; what
`setup()` stores internally is the real base communicator, which `base_communicator()` reads back.
`destroy()` releases it. A rank that hosts only a coupler placement calls `run(periods)` instead of
stepping; see [](c-abi.md) and [](fortran.md).

### Worked example

`examples/brusselator/participants/{u_main.c,v_main.f90,coupler_main.c}` are the worked example: a
three-component graph authored once by a small C++ tool and loaded by three executables that link no
C++ of their own (only the physics, `shim.cpp`, is C++, behind a plain C wrapper).

```sh
brusselator_graph_tool --ranks-u N --ranks-v N [--ranks-coupler N] --launch mpmd|accept \
    --out graph.yaml [--port-file path] [--transport fuse|lowfive]
```

`--launch accept` bundles the coupler's ranks into the acceptor executable (`u_main.c`) rather than
giving it a third `connect`-mode executable. The exact `mpiexec` lines, as `ctest` runs them (world
size 3):

```
# mpmd: one mpiexec, three ':'-separated app segments
mpiexec -n 1 brusselator_u_main --graph graph.yaml --log u.log --periods 2 \
    : -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 --ranks-u 1 --ranks-v 1 \
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
`--check` verdict. Expected agreement with the single-process example: the 3-rank MPMD run matches
bit for bit; the 5-rank run (U on two ranks, V on two, coupler on one) differs by about 9e-14 in
each mean. See [](brusselator.md) for the details.

### The launch-script pattern

`test/parallel/brusselator_participants/run_participants.sh` runs the whole sequence and is the
template for a job script of your own:

```sh
run_participants.sh <mode: mpmd|accept> <mpiexec> <ranks_u> <ranks_v> <ranks_coupler> \
    <graph_tool_exe> <u_main_exe> <v_main_exe> <coupler_main_exe> <check_exe> <scratch_dir> \
    [periods] [timeout_s] [transport] [solver]
```

1. Write the graph for this run's layout (`brusselator_graph_tool ... --out graph.yaml
   --port-file port.txt`), so the rank counts in the YAML and on the `mpiexec` line come from the
   same arguments.
2. MPMD: one `mpiexec` with one `:` segment per executable, in graph order. The script wraps each
   segment in `sh -c '... > "${f}.${PMI_RANK:-0}.log" 2>&1'` so that every rank of every program
   writes its own log; with three programs failing at once, a single merged stream does not say
   which one was the cause.
3. Accept: start the acceptor (`u_main`, with `ranks_u + ranks_coupler` ranks) in the background,
   then the connector (`v_main`), then `wait` on both and fail if either did.
4. Run the checker on the logs.

Pass the same `--periods` to every program: every rank must step the same number of periods.

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

The order matters. In `connect` mode the base communicator is merged from an intercommunicator,
which leaves the two programs *connected*: `MPI_Finalize` is then collective across both of them,
and a program that finalizes while still connected leaves an operation half-finished on a peer that
is no longer listening. In MPMD mode the handle is a duplicate of the caller's own world and
`disconnect()` frees it; the call is the same either way, so a caller does not have to know which
mode it ended up in. `ConnectedWorld` records which (`Teardown::Free`, `Teardown::Disconnect`, or
`Teardown::None` for a failed or already released handle); `disconnect()` is idempotent, safe on
the result of a failed `connect()`, and refuses with `E-CONNECT-FINALIZED` after `MPI_Finalize`.

A participant that reaches FUSE through the C or Fortran ABI has nothing to do: `destroy()` releases
the connection the same way, after the runner it owns is gone, and before the program's own
`MPI_Finalize`.

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

## Limitations

- Python programs cannot take part in an MPMD or Connect run; see [](python.md).
- `service_name` is accepted only for a two-component connect; chains need `port_file`.
- A plugin component in the same graph as a `connect` component must be hosted by a process whose
  own `my_component` connects (`E-ABI-PLUGIN-NO-SPANNING-COMM` otherwise).
- The `MPI_Comm_connect`/`MPI_Comm_accept` handshake cannot be bounded from one side once the
  port file is found.
- Accept/connect requires `FUSE_ENABLE_CONNECT_ACCEPT` and an MPI with working dynamic process
  management between jobs.

## See also

- [](mixing-languages.md): which languages and entry points can be combined, and how
- [](placement.md): `RankSet` and processor-set placement
- [](c-abi.md), [](fortran.md): driving a participant through the C or Fortran interface
- [](providers.md): the plugin launch mode
- [](lowfive.md): file-based coupling between executables through HDF5 interception
- [](brusselator.md): the worked example
- [](build.md): enabling MPI and the configure-time probes
