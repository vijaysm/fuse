# Coupling across separately built executables

:::{note}
`fuse::transport::connect()` (`include/fuse/transport/connect.hpp`) is implemented and tested
(`build-evidence/mutations/transport-connect.toml`). It needs a real multi-process launch to
execute, so the executable-pair test under `test/parallel/connect/` is compiled here and run on
the host; see [](../dev/testing.md#the-five-states) for what "implemented" means while a claim is
still a host-run item rather than a measured one.
:::

Every other page in this guide assumes one `mpiexec` launches one executable that contains every
component of the graph. Two further cases put more than one *executable* into the same coupled run.

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

**A partner that never launches is a timeout, not a hang.** `ConnectOptions::timeout_s` (default 30
seconds) bounds how long the non-acceptor side polls for the port file before refusing with
`E-CONNECT-TIMEOUT`. The one blocking window this does not cover is `MPI_Comm_connect` itself, once
the port file is found: that call is a synchronous MPI handshake with the acceptor's own
`MPI_Comm_accept`, and neither this library nor the MPI standard offers a way to bound it from one
side alone.

## What `connect()` checks, and what it refuses

Both launch styles are checked by the same rule before any MPI call happens:
**the launched components' `RankSet`s must be consecutive blocks, in the graph's own component
order.** `MPI_Intercomm_merge` concatenates one side's ranks after the other's in whichever order
the two sides agree to merge in, and launcher-style MPMD assigns rank 0..k-1 to the first `mpiexec`
segment and so on in command-line order; both agree with the graph's own order only if that order
is what the graph declares. A graph that declares an `Mpmd` or `Connect` component on an empty,
strided, or out-of-sequence `RankSet` is refused with `E-CONNECT-SPEC` before any process even calls
`MPI_Open_port` or reads `MPI_APPNUM`.

| Code | When |
|---|---|
| `E-LAUNCH-APPNUM` | (graph build) an `Mpmd` component has no `appnum` set |
| `E-LAUNCH-CONNECT-TARGET` | (graph build) a `Connect` component has neither `port_file` nor `service_name` |
| `E-CONNECT-SPEC` | the launched components' `RankSet`s are not consecutive blocks in graph order; two `Mpmd` components share one `appnum`; a non-`Mpmd`/`Connect` component was handed to `connect()`; or more than TWO components declare `Connect` (KNOWN_ISSUES.md D18 -- `Mpmd` has no such limit, only true separate-executable `Connect` launches) |
| `E-CONNECT-APPNUM` | the real `MPI_APPNUM` layout at runtime disagrees with the graph (an undeclared appnum, or a declared one whose ranks do not match) |
| `E-CONNECT-TIMEOUT` | the non-acceptor side waited longer than `timeout_s` for the port file |

See [](placement.md) for `RankSet` itself, and [](c-abi.md) if the other executable is not written
against this library's own `CouplingGraph` API at all — a participant driven through the C or
Fortran ABI declares its launch style the same way, through the descriptor it was handed.

## A C or Fortran participant never calls `connect()` itself

Everything above is a C++ call against a `fuse::CouplingGraph` the executable built or loaded
itself. A participant written in C or Fortran has neither: it loads a graph someone else authored
(`fuse_load_graph`) and never touches `fuse::transport::connect()`, or any other C++, at all —
`fuse_setup_options::my_component` is the one field that makes this possible. Set it to the name of
the component THIS process plays, with `communicator` holding this executable's own
`MPI_COMM_WORLD` (as a Fortran handle, `MPI_Comm_c2f`):

```c
fuse_setup_options opts = {.struct_size = sizeof(opts)};
opts.communicator  = MPI_Comm_c2f(MPI_COMM_WORLD);
opts.my_component  = "ocean";
fuse_coupler_abi(FUSE_ABI_VERSION)->setup(ctx, &opts, NULL);
```

`fuse_setup()` reads `my_component`'s own `LaunchSpec` straight off the already-loaded graph and
does exactly what the C++ call above does: nothing extra for `InProcess`/`Plugin` (`communicator`
is already the base communicator); `MPI_APPNUM` checking for `Mpmd`; the full accept/connect/merge
handshake for `Connect`. Either way, `communicator` as GIVEN is the WORLD communicator this
executable started with, and what `fuse_setup()` stores internally afterward is the real base one
— the caller never sees the difference, or the intercommunicator machinery, at all.
`examples/brusselator/participants/{u_main.c,v_main.f90,coupler_main.c}` (Conformance Unit 2) are the
worked example: the same three-component graph (the coupler included) as every other page in this
guide, authored once by a small C++ tool (`brusselator_graph_tool --ranks-u N --ranks-v N
--ranks-coupler N --launch mpmd|accept --out graph.yaml`) and loaded by `u_main.c`/`v_main.f90`/
`coupler_main.c`, none of which link any C++ of their own (`shim.cpp`, the Brusselator physics
itself, is the one C++ translation unit either participant's own loop body calls into, through a
plain C-ABI wrapper — [](../dev/abi-design.md) distinguishes this from the ABI's own ban on a
foreign caller touching `fuse::` C++ directly). `--launch accept` bundles the coupler's own ranks
into the acceptor executable's (`u_main.c`'s) own `MPI_COMM_WORLD` rather than giving it a third
`Connect`-mode executable of its own — see `E-CONNECT-SPEC`'s row below and KNOWN_ISSUES.md D18 for
why. `brusselator_check` (same directory) reads both executables' own printed per-window logs and
reproduces `examples/brusselator/main.cpp`'s own `--check` verdict from them.
`test/parallel/brusselator_participants/run_participants.sh` is the host driver
(`verification.brusselator.participants.{mpmd,accept}.{3,5}`, `host_verification_needed`: MPI cannot
start in this sandbox, KNOWN_ISSUES.md C1 — every executable involved is confirmed to BUILD at tier 1,
not yet run). What that script's own `mpiexec` lines actually are, exactly as `ctest` runs them (the
world-size-3 case; `{3,5}`'s own `ranks_u`/`ranks_v`/`ranks_coupler` are `1/1/1` and `2/2/1`):

```
# mpmd.3: one mpiexec, three ':'-separated app segments
mpiexec -n 1 brusselator_u_main --graph graph.yaml --log u.log --periods 2 \
    : -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 \
    : -n 1 brusselator_coupler_main --graph graph.yaml

# accept.3: two mpiexec invocations -- U's executable is the acceptor and
# carries the coupler's own bundled rank in its local MPI_COMM_WORLD
mpiexec -n 2 brusselator_u_main --graph graph.yaml --log u.log --ranks-u 1 --periods 2 &
mpiexec -n 1 brusselator_v_main --graph graph.yaml --log v.log --periods 2 &
wait

# both modes, every row
brusselator_check --u-log u.log --v-log v.log
```
