# Coupling codes written in different languages

This page is for people who have two or more codes in different languages (C++, C, Fortran,
Python, or an HDF5 program they cannot modify) and want to know how each one can take part in a
FUSE coupling, which combinations work, and how to pick. It is a decision guide; the mechanics of
each entry point are on their own pages, linked from every row.

## The five ways into a coupling

| Entry point | Who owns the time loop | Process model | Can share a process with | MPI | Restart to disk | Iteration regions |
|---|---|---|---|---|---|---|
| **C++ `fuse::Component`** driven by a `fuse::Runner` ([](quickstart.md)) | FUSE | linked into one C++ executable; several such executables join with `connect()` ([](multiple-executables.md)) | other C++ components | yes | yes ([](restart.md)) | yes (`save_state`/`restore_state`) |
| **C or Fortran participant** through `fuse_coupler_abi` ([](c-abi.md), [](fortran.md)) | your code | its own executable, or a process that also hosts plugins; joins other executables through `setup().my_component` | provider plugins; other participants in the same process | yes | no file: `request_restart_point` only raises a flag | yes (`requires_checkpoint`/`requires_restore`) |
| **C or Fortran provider plugin** through `fuse_provider_abi` ([](providers.md)) | FUSE | a shared library `dlopen()`ed by a participant-ABI host at `initialize()` | the host participant and other plugins | runs on its component's ranks of the host's communicator | restart blobs (`save_restart_blob`/`load_restart_blob`); no participant-ABI host writes the file | yes (`save_state`/`restore_state` + `FUSE_CAP_CHECKPOINT`) |
| **Python `pfuse.Component`** ([](python.md)) | FUSE (the Python session's runner) | in the Python interpreter, one process | other Python components | no | no | yes (`save_state`/`restore_state`) |
| **Unmodified HDF5 code** through LowFive ([](lowfive.md)) | the code itself | its own executable; FUSE-aware participants on each side perform the delivery | nothing | yes | n/a | coupling windows and convergence on the LowFive-bound channels |

Every row consumes the same graph: a `CouplingGraph` built in C++ or Python, or the same thing as a
YAML file or canonical bytes. Launch mode is a per-component declaration in that graph, not a
property of the program ([](multiple-executables.md#the-launch-blocks-in-yaml)).

## Combinations that work

**A C++ driver and plugins written in C or Fortran.** The driver is a C or C++ program that uses
the participant ABI: it loads a graph whose components carry `launch: {mode: plugin, plugin_path:
...}`, calls `initialize` (which `dlopen()`s every plugin on its ranks) and then `run(periods)`.
The driver can also be a participant itself, stepping its own component with
`begin_step`/`end_step` while the plugins advance in between. [](providers.md#a-driver-that-runs-plugins)
shows the driver and its output. This is the path for "FUSE drives my Fortran solver".

**A C participant and a Fortran participant in different executables.** Each keeps its own loop,
loads the same YAML, and passes its own `MPI_COMM_WORLD` plus its component name to `setup()`;
FUSE forms the base communicator (MPMD or connect/accept) from the launch block in the graph. The
Brusselator pair (`u_main.c`, `v_main.f90`, plus the C `coupler_main.c`) is the complete example
([](multiple-executables.md#worked-example), [](brusselator.md)). The same arrangement works for
two C programs, two Fortran programs, or a C++ program that uses the participant ABI.

**A Fortran participant and a plugin in one process.** The participant's graph names a plugin
component; `fuse_initialize` mounts it in the Fortran process, and FUSE advances it inside the
`fuse_begin_step`/`fuse_end_step` calls the Fortran code makes for its own component
([](fortran.md#workflows)). The plugin may be C or Fortran.

**A C++ executable (native components, `connect()`) and a C or Fortran participant executable.**
The C++ side builds or loads the graph and calls `fuse::transport::connect()`; the participant side
loads the same graph and calls `setup()` with `my_component`. Both end up on the same base
communicator with the same rank numbering, because both follow the graph's component order
([](multiple-executables.md)).

**LowFive between two unmodified HDF5 codes.** The codes keep their `H5Dwrite`/`H5Dread`; a
FUSE-aware participant on each side (C, Fortran or C++) performs the delivery when `begin_step`
reports `delivery_pending` ([](c-abi.md#windows-over-lowfive-bound-channels-step_opened-and-delivery_pending),
[](lowfive.md)).

**Python for authoring, C/C++/Fortran for running.** Python builds or loads the graph, validates
it, and writes it (`to_yaml()`, `to_bytes()`); the compiled programs load that file. See
[the shared-graph example](#one-graph-three-consumers) below.

## Combinations that do not work

These are plain facts about the current interfaces, each with the refusal or the missing surface:

- **A native C++ `fuse::Component` cannot be added to a context created through
  `fuse_coupler_abi`.** The participant ABI has no entry that takes a C++ object; the components of
  an ABI context are externally driven participants and `dlopen()`ed plugins. To run native C++
  components together with a C or Fortran code, put them in their own executable and couple across
  executables, or package them as a plugin.
- **A C++ `fuse::Runner` program cannot mount a plugin directly.** The loader lives behind the
  participant ABI's `initialize()`. A C++ program that wants plugins calls `fuse_coupler_abi()`
  (the header is C-compatible) and becomes a participant-ABI driver.
- **Python components cannot join an MPI or multi-executable run and cannot host plugins.** `pfuse`
  binds the C++ model and runner in one process with no MPI, no participant ABI and no plugin
  loader. `Session.add_component` accepts only `pfuse.Component` instances.
- **Python, C++ and Fortran do not share a running coupling.** Mixing them happens by exchanging the
  graph (YAML or canonical bytes), never at run time: a Python-authored graph runs in a C/C++/Fortran
  program, and a Python session runs only Python components.
- **A plugin component next to a `connect`-launched component is refused** unless the hosting
  process's own `my_component` is one of the connecting components
  (`E-ABI-PLUGIN-NO-SPANNING-COMM`; [](providers.md#a-known-failure-case)). MPMD has no such
  restriction.
- **Restart files are not written through the C or Fortran ABI.** A participant's
  `request_restart_point` raises a flag that no ABI entry reads; a plugin's restart blobs are
  collected only by a C++ host that owns a `fuse::Runner` and a restart writer.

## Which should I choose

1. **Is the code C++ and can you link it with the other components into one executable?** Use
   `fuse::Component` and a `fuse::Runner` ([](quickstart.md)). Several such executables join with
   `connect()`.
2. **Is it Python?** Use `pfuse.Component` for a single-process, non-MPI coupling of Python
   components ([](python.md)); use Python to author and validate graphs for everything else.
3. **Does the code have a time loop you cannot give up (a legacy driver, its own I/O and restart
   cadence)?** Make it a participant: C through `fuse_coupler_abi` ([](c-abi.md)), Fortran through
   `ffuse` ([](fortran.md)). It can be its own executable or host plugins.
4. **Is it a solver you can call once per step from outside?** Make it a provider plugin in C or
   Fortran ([](providers.md)) and host it from any participant-ABI program.
5. **Can the code not be modified at all, but it reads and writes HDF5?** Use LowFive
   ([](lowfive.md)), with a small participant on each side to drive the delivery.
6. **Several executables?** Prefer MPMD (one `mpiexec`, works everywhere) over connect/accept
   (needs dynamic process management; [](multiple-executables.md#which-launch-modes-this-build-offers)).

## One graph, three consumers

The same YAML file is consumed, unchanged, by a C program, a Fortran program and Python.
`test/abi/fortran/install_consumer/two_component.yaml` declares an `upstream` component providing
`out` and a `downstream` component requiring `in`, on two generated line meshes of four cells:

```yaml
version: 1
meshes:
  - {name: m1, source: "generated:line:4"}
  - {name: m2, source: "generated:line:4"}
regions:
  - {name: r1, mesh: m1, dimension: 1}
  - {name: r2, mesh: m2, dimension: 1}
components:
  - name: upstream
    ranks: {first: 0, count: 1}
    interval: 1
    fields:
      - {name: out, region: r1, disc: FV, components: 1, units: K,
         conservation: None, intent: Provides}
  - name: downstream
    ranks: {first: 0, count: 1}
    interval: 1
    fields:
      - {name: in, region: r2, disc: FV, components: 1, units: K,
         conservation: None, intent: Requires}
channels:
  - name: link
    source: upstream.out
    target: downstream.in
    remap: {flags: [Conservative]}
iteration_regions: []
```

**C** (`test/abi/install_consumer/main.c`, shown in full in [](c-abi.md#a-complete-program))
loads it with `load_graph`, registers one array per component and drives both with
`begin_step`/`end_step`. **Fortran** (`test/abi/fortran/consumer.f90`, shown in full in
[](fortran.md#a-complete-program)) does the same through `ffuse`. **Python** validates and inspects
it, and can rewrite it:

```python
import sys
import pfuse

g = pfuse.load_yaml_file(sys.argv[1])      # the same file the C and Fortran programs load
diags = pfuse.validate(g)
print(diags.to_string(), end="")
print(g)
for i in range(g.component_count):
    print("component", g.component(pfuse.ComponentId(i)).name)
sys.exit(1 if diags.has_errors else 0)
```

Expected output for `two_component.yaml`:

```text
CouplingGraph(2 component(s), 1 channel(s))
component upstream
component downstream
```

`g.to_yaml()` writes the graph back out, launch blocks included, and `g.to_bytes()` produces the
canonical descriptor that `load_graph` accepts through `bytes`/`byte_count`. Adding
`launch: {mode: plugin, plugin_path: ...}` to `upstream` turns the same file into one the driver in
[](providers.md#a-driver-that-runs-plugins) runs with no code change on the C side; adding
`launch: {mode: mpmd, appnum: ...}` to both components turns it into a two-executable run
([](multiple-executables.md)). The programs do not change; the graph does. [](drivers.md) covers
reusable drivers and [](graph-programming.md) building graphs in code.

## See also

- [](c-abi.md), [](fortran.md): participants.
- [](providers.md): plugins and the driver that runs them.
- [](multiple-executables.md): launch modes and the decision table for processes.
- [](python.md): the Python module and what it can and cannot do.
- [](lowfive.md): unmodified HDF5 codes.
- [](yaml.md): the graph file format.
