# Coupling an existing C or C++ code

This page is for people who have a simulation with its own time loop and want to couple it to other
components through FUSE with as little change as possible. Your code keeps ownership of the loop and
calls FUSE at the start and end of each step, through the **participant ABI**: one C header, one
exported symbol, no C++ in your program. Fortran codes use the same ABI through a module; see
[](fortran.md). If you want the opposite arrangement, FUSE owning the loop and calling your solver
as a plugin, see [](providers.md). [](mixing-languages.md) compares the options.

## One symbol, one call shape

`fuse/abi/participant.h` declares exactly one function:

```c
const fuse_participant_abi_v1* fuse_coupler_abi(uint32_t abi_version);
```

It returns a static table of function pointers for the version you ask for, or `NULL` when the
library does not speak that version. `FUSE_ABI_VERSION` is `1`. `libfuse-abi` exports no other
symbol, so nothing in it can collide with another library's `fuse_*` names.

Every entry in the table has the shape

```c
fuse_status f(fuse_ctx* ctx, const IN* in, OUT* out);
```

`IN` and `OUT` are plain structs from `fuse/abi/types.h`. Each begins with `uint32_t struct_size`,
which you set to `sizeof` of the struct your program was compiled against; FUSE reads only the
members that size says are present and uses documented defaults for the rest. Pass `NULL` where an
entry's type is `fuse_null`. Two value types, `fuse_exact_time` and `fuse_semantic_version`, have no
`struct_size` and never grow. In prose below, `begin_step()` means the `begin_step` member of the
table.

## The smallest working program

```c
#include <fuse/abi/participant.h>
#include <string.h>

const fuse_participant_abi_v1* fuse = fuse_coupler_abi(FUSE_ABI_VERSION);
if (!fuse) { /* this build does not speak the version you asked for */ }

fuse_context_out co = {.struct_size = sizeof(co)};
fuse->create(NULL, NULL, &co);
fuse_ctx* ctx = co.ctx;

fuse_graph_source src = {.struct_size = sizeof(src)};
src.yaml_path = "coupling.yaml";          /* or set .bytes/.byte_count instead */
fuse->load_graph(ctx, &src, NULL);

fuse_field_name fname = {.struct_size = sizeof(fname)};
fname.component = "fluid";
fname.field     = "heat_flux";
fuse_field_ref fref = {.struct_size = sizeof(fref)};
fuse->field_handle(ctx, &fname, &fref);

fuse_field_view view = {.struct_size = sizeof(view)};
view.handle       = fref.handle;
view.element_type = FUSE_ELEMENT_F64;
view.memspace     = FUSE_MEMSPACE_HOST;
view.ownership    = FUSE_OWNERSHIP_OWNED;
view.components   = 1;
view.count        = n;
view.owned_count  = n;
view.stride       = 1;
view.data         = q;                     /* your own storage; FUSE never copies it */
fuse->register_field(ctx, &view, NULL);

fuse->initialize(ctx, NULL, NULL);

fuse_step_request req = {.struct_size = sizeof(req)};
req.component = "fluid";
fuse_ongoing ongoing = {.struct_size = sizeof(ongoing)};
for (fuse->is_coupling_ongoing(ctx, NULL, &ongoing); ongoing.ongoing;
     fuse->is_coupling_ongoing(ctx, NULL, &ongoing)) {
  fuse_step_report rep = {.struct_size = sizeof(rep)};
  fuse->begin_step(ctx, &req, &rep);        /* rep.t_begin / rep.t_end: exact times */
  advance_my_solver();                      /* read/write q between the two calls */
  fuse->end_step(ctx, &req, &rep);
}
fuse->destroy(ctx, NULL, NULL);
```

This loop runs forever unless `setup()` set an `end_time` (see
[is_coupling_ongoing and the end time](#is_coupling_ongoing-and-the-end-time)); the real programs
below bound their loops with a period count instead.

### A complete program

`test/abi/install_consumer/main.c` is a complete C11 program that links only `libfuse-abi`. It
writes a two-component graph as YAML, drives both components from one process, and prints what the
downstream component received:

```{literalinclude} ../../test/abi/install_consumer/main.c
:language: c
:lines: 68-155
```

Expected output (exact digits can vary by platform; `%a` prints hexadecimal floating point, so
`0x1.4p+3` is 10 and `0x1.6p+3` is 11):

```text
installed C consumer ok
period 0 downstream.in[0] = 0x1.4p+3
period 0 downstream.in[1] = 0x1.4p+3
period 0 downstream.in[2] = 0x1.4p+3
period 0 downstream.in[3] = 0x1.4p+3
period 1 downstream.in[0] = 0x1.6p+3
period 1 downstream.in[1] = 0x1.6p+3
period 1 downstream.in[2] = 0x1.6p+3
period 1 downstream.in[3] = 0x1.6p+3
```

The program reads `down_storage` after `end_step(downstream)` rather than between the two calls,
because for this graph (two components on the same interval, no lag) the compiled schedule places
the channel's transfer after both components' ticks; see
[What is current between begin_step and end_step](#what-is-current-between-begin_step-and-end_step).

## The lifecycle

A context moves through three states: **Empty** after `create`, **Loaded** after `load_graph`,
**Initialized** after `initialize`. The order of calls is:

1. `fuse_coupler_abi(FUSE_ABI_VERSION)`, once. Check for `NULL`.
2. `create(NULL, NULL, &co)`. `ctx` must be `NULL` here (`E-ABI-CREATE`).
3. `load_graph`, once per context, from the Empty state (`E-ABI-LOAD-STATE`). Set exactly one of
   `yaml_path` or `bytes`/`byte_count` (`E-ABI-LOAD-SOURCE`).
4. `setup`, optional, from the Loaded state: communicator, end time, and which component this
   process plays. Collective over the communicator it is given.
5. `field_handle` and `register_field` for every field of the components this process drives, from
   the Loaded state. `setup` and `register_field` can come in either order.
6. Write each export's initial value into its registered storage.
7. `initialize`, collective. Validates, lowers and schedules the graph, mounts plugin components,
   wires every other component as externally driven.
8. The loop: `is_coupling_ongoing`, then `begin_step(name)`, your solver, `end_step(name)`. A
   process that hosts no externally driven component calls `run` instead.
9. `destroy`, once. A second `destroy` on the same handle returns `FUSE_ERR_INVALID`.

:::{warning}
**Your export must hold its initial value before `initialize()`.** For a component in a coupling
window ([](coupling-windows.md)), `initialize()` primes the first window's t = 0 sample from
whatever the storage you registered for the export holds at that moment, and the partner integrates
that sample. An in-process component's constructor has already written its initial value there;
your program has to do the same by hand: compute the export at t = 0 and store it before calling
`initialize()`. If the export depends on the partner's initial state, you need to know the partner's
initial export too; `examples/brusselator/participants/u_main.c` rebuilds a throwaway partner for
exactly this. Left at zero, the first window's exchange is measurably short on one side and nothing
refuses it. The storage must also be the field's size on this process: a shorter or longer
registration is refused at `initialize()` with `E-RT-SIZE`.
:::

## Reading an error

Every entry point returns a `fuse_status`: `FUSE_OK` (`0`) or a negative code.

| Status | Value | Meaning |
|---|---|---|
| `FUSE_ERR_INVALID` | -1 | malformed argument: null pointer, out-of-range value, bad or destroyed handle |
| `FUSE_ERR_NOTFOUND` | -2 | a well-formed lookup found nothing: no such component or field |
| `FUSE_ERR_STATE` | -3 | a valid call in the wrong state: no graph yet, already initialized, step already open |
| `FUSE_ERR_RUNTIME` | -4 | an operation failed or an exception was contained at the boundary; `last_error` has the reason |
| `FUSE_ERR_UNSUPPORTED` | -5 | the entry exists but this build or this process cannot perform it |
| `FUSE_ERR_VERSION` | -6 | `struct_size` is below the oldest layout this library knows or above `FUSE_ABI_MAX_STRUCT_SIZE` (65536) |

No C++ exception crosses the boundary: a `fuse::Error` becomes `FUSE_ERR_RUNTIME` with that error's
`E-...` code, and any other exception becomes `E-ABI-EXCEPTION`. A null, foreign or already destroyed
`ctx` is detected without dereferencing it and refused with `FUSE_ERR_INVALID` / `E-ABI-HANDLE`.

The status does not carry the message. `last_error(ctx, NULL, &err)` fills a caller-owned
`fuse_error` with:

- `err.status`: the failed call's own `fuse_status` (`FUSE_OK` if nothing has failed yet on this
  context);
- `err.code`: its `E-...` code, valid until the next failing call on this context (or on the thread,
  when `ctx` is `NULL`). It is not static storage, and it is not valid after `destroy()`;
- `err.length`: the full message length;
- `err.buffer`/`err.capacity`: as much of the message as fits, always NUL-terminated.
  `err.truncated` is `1` if it did not fit, and `err.length` is then the size a second call needs.
  Pass `buffer = NULL` to query the length alone.

```c
fuse_error err = {.struct_size = sizeof(err)};
char buf[256];
err.buffer = buf;
err.capacity = sizeof(buf);
if (fuse->begin_step(ctx, &req, &rep) != FUSE_OK) {
  fuse->last_error(ctx, NULL, &err);
  fprintf(stderr, "%s: %s%s\n", err.code, buf, err.truncated ? " [...]" : "");
}
```

Errors are stored per context and per thread. `ctx` may be `NULL` to read the thread-local error of
a failed `create()`, the one call that has no context to attach an error to, and of a failed
`destroy()`.

## Entry points

The table, in order. "State" is the context state the entry requires; "collective" means every
process of the communicator (or of the component) must make the call.

| Entry | In → Out | State | What it does; notable codes |
|---|---|---|---|
| `version` | `fuse_null` → `fuse_version_info` | any, `ctx` may be `NULL` | library semantic version, `abi_version`, and a descriptive string |
| `capabilities` | `fuse_null` → `fuse_capability_set` | any, `ctx` may be `NULL` | `FUSE_CAP_*` bits; see [What capabilities reports](#what-capabilities-reports) |
| `create` | `fuse_null` → `fuse_context_out` | `ctx` must be `NULL` | new context; initialises Kokkos once per process (`E-ABI-CREATE`) |
| `destroy` | `fuse_null` → `fuse_null` | any | ends the context, disconnects a connect-formed world; second call `FUSE_ERR_INVALID` |
| `last_error` | `fuse_null` → `fuse_error` | any, `ctx` may be `NULL` | see above |
| `load_graph` | `fuse_graph_source` → `fuse_null` | Empty | YAML path or canonical bytes, exactly one (`E-ABI-LOAD-SOURCE`, `E-ABI-LOAD-STATE`) |
| `save_graph` | `fuse_null` → `fuse_graph_sink` | Loaded or later | canonical bytes of the loaded graph; query with `buffer = NULL`, then fill (`E-ABI-SAVE-STATE`) |
| `setup` | `fuse_setup_options` → `fuse_null` | Loaded; collective | communicator, end time, `my_component` (`E-ABI-SETUP-STATE`, `-COMPONENT`, `-NOMPI`, `-RESERVED`) |
| `field_handle` | `fuse_field_name` → `fuse_field_ref` | Loaded or later | handle, declared `components`, `intent` (0 Provides, 1 Requires, 2 both), `conservation`; any component of the graph, not only yours (`E-ABI-FIELD-HANDLE-{STATE,NULL,COMPONENT,FIELD}`) |
| `register_field` | `fuse_field_view` → `fuse_null` | Loaded | binds your storage to a handle (`E-ABI-REGISTER-{STATE,ELEMENT,MEMSPACE,OWNERSHIP,OWNED-COUNT}`) |
| `initialize` | `fuse_null` → `fuse_null` | Loaded; collective | validate, lower, schedule, mount plugins, wire components (`E-ABI-INIT-{STATE,VALIDATE,LOWER,SCHEDULE,COMM}`, `E-ABI-PLUGIN-NO-SPANNING-COMM`, `E-ABI-DELIVERY-INPROCESS`, `E-RT-SIZE`) |
| `is_coupling_ongoing` | `fuse_null` → `fuse_ongoing` | Initialized | side-effect free: `ongoing`, `period`, `now`, `next_component` (`E-ABI-ONGOING-STATE`) |
| `begin_step` | `fuse_step_request` → `fuse_step_report` | Initialized; collective over the component | opens the named component's step; imports are current on return (`E-ABI-STEP-{STATE,NULL,COMPONENT,IN-PROCESS,ALREADY-OPEN,ORDER}`) |
| `end_step` | `fuse_step_request` → `fuse_step_report` | Initialized; collective over the component | closes it; exports are read on entry (`E-ABI-STEP-NOT-OPEN`) |
| `run` | `fuse_run_request` → `fuse_null` | Initialized; collective | runs `periods` coupling periods (or `-1`: until `end_time`) in process (`E-ABI-RUN-{STATE,EXTERNAL,PERIODS,NO-END-TIME}`) |
| `view_field` | `fuse_field_ref` → `fuse_field_view` | Loaded or later | FUSE's own view of a field's storage, including window buffers (`E-ABI-VIEW-FIELD-HANDLE`, `FUSE_ERR_NOTFOUND`) |
| `base_communicator` | `fuse_null` → `fuse_base_communicator_info` | any; local | the base communicator `setup()` formed, and whether one was formed |
| `request_restart_point` | `fuse_null` → `fuse_null` | Initialized | raises a flag asking for a restart point; no effect before `initialize` |

Field intents and the other enumerators are defined in `fuse/abi/types.h`:
`FUSE_ELEMENT_{F64,I64}`, `FUSE_MEMSPACE_{HOST,DEVICE}`,
`FUSE_OWNERSHIP_{OWNED,GHOSTED,REPLICATED}`, and the `FUSE_CAP_*` bits.

## Setup: communicator, end time and `my_component`

`setup` is optional. Without it the context runs on one process, with no communicator and no end
time. Its members:

`communicator`, `has_communicator`
: The Fortran handle (`MPI_Comm_c2f`) of the communicator this process couples on. A
  communicator counts as supplied when `communicator != 0` **or** `has_communicator == 1`. Always
  set `has_communicator = 1` when you pass one: under Open MPI the Fortran handle of
  `MPI_COMM_WORLD` is `0`, which without the flag reads as "none". A build without MPI refuses a
  supplied communicator with `E-ABI-SETUP-NOMPI`.

`has_end_time`, `end_time`
: An exact rational time at which `is_coupling_ongoing` reports `0` and `run(-1)` stops. The graph
  carries no end time of its own.

`my_component`
: The name of the graph component this process plays. An unknown name is refused with
  `E-ABI-SETUP-COMPONENT`. If that component's launch mode is `mpmd` or `connect`, `setup()` calls
  `fuse::transport::connect()` for you with the communicator you passed (which must then be this
  executable's own `MPI_COMM_WORLD`), and stores the resulting base communicator: a duplicate of the
  shared MPMD world, or the intracommunicator merged from an accept/connect handshake. An `mpmd` or
  `connect` component with no communicator is refused with `E-ABI-SETUP-COMPONENT`. For an
  in-process or plugin component, `my_component` changes nothing about the communicator, but it is
  still what lets FUSE reconcile statuses across the component's ranks (see
  [Collective agreement](#collective-agreement)). A C or Fortran participant never calls `connect()`
  or any other C++ itself; see [](multiple-executables.md).

`reserved`
: Must be `0` (`E-ABI-SETUP-RESERVED`).

`examples/brusselator/participants/u_main.c` sets this up for an MPI run of component `U`:

```{literalinclude} ../../examples/brusselator/participants/u_main.c
:language: c
:lines: 324-346,351-354
```

### Recovering the base communicator

`base_communicator(ctx, NULL, &info)` reads back what `setup()` stored:

```c
fuse_base_communicator_info info = {.struct_size = sizeof(info)};
fuse->base_communicator(ctx, NULL, &info);
if (info.formed) {
  MPI_Comm base = MPI_Comm_f2c((MPI_Fint)info.communicator);
  /* this process's own base communicator, for every launch mode */
}
```

It is legal in any state, including before `setup()` (`formed == 0`, `communicator == 0`). Check
`formed` first: `communicator == 0` is also a valid handle for `MPI_COMM_WORLD` under some MPI
implementations, and `formed` is `1` whenever `setup()` was given a communicator, including handle
0 with `has_communicator = 1`. The call is local and non-collective.

## Fields: `field_handle` and `register_field`

`field_handle` resolves a `(component, field)` pair from the loaded graph into a handle that is
stable for the context's lifetime, together with what the graph declares about the field. It works
for any component in the graph, which is how a participant learns the handle of a peer's field (used
by `delivery_field`, below).

`register_field` binds your own storage to a handle. FUSE keeps the pointer and never copies the
data; the storage must stay valid and at the same address until `destroy()`. The rules:

- `element_type` must be `FUSE_ELEMENT_F64` (`E-ABI-REGISTER-ELEMENT`).
- `memspace` must be `FUSE_MEMSPACE_HOST` (`E-ABI-REGISTER-MEMSPACE`). Device pointers are
  accepted only on the provider side ([](providers.md)).
- `ownership` must not be `FUSE_OWNERSHIP_GHOSTED` (`E-ABI-REGISTER-OWNERSHIP`), and `owned_count`
  must equal `count` (`E-ABI-REGISTER-OWNED-COUNT`).
- `components` must equal the `components` the graph declares (`fuse_field_ref::components`).
- `count` is the number of entities on this process. It must match the size the compiled plan
  gives the field on this rank, otherwise `initialize()` refuses with `E-RT-SIZE`. For a component
  on several ranks, that is the rank's share of the region under the mesh's decomposition, not the
  whole region.
- `stride` is the number of elements between consecutive entities (`1` = packed), and
  `component_stride` the number of elements between the components of one entity: `1` is
  array-of-structures, `count * stride` is structure-of-arrays.
- `window_samples` is ignored as an input; register your own single-sample storage.
- Register before `initialize()` (`E-ABI-REGISTER-STATE`), once per handle.

## The step loop

`begin_step(name)` resumes FUSE's compiled instruction stream for the current period up to the
named component's own step and stops there. `end_step(name)` resumes it past that step, to the next
externally driven component's step or to the period's end. Between the two calls your solver reads
its imports and writes its exports directly in the storage you registered. FUSE snapshots a
participant's exports when an iteration region needs to roll back, so a participant can sit inside
one.

`fuse_step_request` carries only `component` for these two entries; the window members are used by
the provider ABI. `fuse_step_report` reports:

| Member | Meaning |
|---|---|
| `t_begin`, `t_end` | the step's exact bounds, as `fuse_exact_time` rationals; your solver advances exactly this interval and never chooses `dt` itself |
| `step` | step index within the run |
| `iteration` | 0-based sweep index inside an iteration region, `0` outside one |
| `requires_checkpoint` | `1` on `begin_step` when FUSE has just captured this component's registered fields: save the private state those fields do not cover |
| `requires_restore` | `1` on `begin_step` after a re-sweep rolled this component back: restore that private state before advancing (registered fields are already restored) |
| `next_component` | which component this process must call `begin_step` for next; see below |
| `step_opened` | `0` when `begin_step` did not open a step because the period rolled over; see below |
| `period` | the period counter, the same value `fuse_ongoing::period` reports |
| `delivery_pending`, `delivery_field` | a LowFive-bound transfer is owed; see below |

### Exact time

```c
typedef struct fuse_exact_time { int64_t numerator; int64_t denominator; } fuse_exact_time;
```

No entry point accepts or returns a floating-point time. If your solver keeps `double` time, convert
at the boundary and keep the exact value as the reference; otherwise a step count of a few million
drifts your schedule off the time window.

### What is current between `begin_step()` and `end_step()`

Every exchange that the compiled schedule places before this component's tick has happened by the
time `begin_step()` returns, so a `Requires` field fed by one of those is current to read right
there. What the schedule places before the tick is a property of the graph. A channel between two
components of the same interval, with no lag, is ordered after both components' ticks for the
period, so its import still holds the previous period's value at `begin_step()` and the new value
arrives inside the `end_step()` that closes the period (the complete program above reads after
`end_step()` for this reason). A `lag(1)` channel behaves the same way by definition.

### Several components in one process: the call order

A process hosting more than one externally driven component must call `begin_step()`/`end_step()`
for each in the order FUSE's compiled schedule visits their steps, and must finish each component's
pair before opening the next (a second open step is refused with `E-ABI-STEP-ALREADY-OPEN`). Calling
`begin_step()` for a component out of turn is refused with `E-ABI-STEP-ORDER`, and the message
names the component the schedule expects next. `end_step()` with no matching open step is refused
with `E-ABI-STEP-NOT-OPEN`. `begin_step()` on a plugin component is refused with
`E-ABI-STEP-IN-PROCESS`: FUSE drives those itself.

### Knowing the order in advance: `next_component`

A fixed alternation (A, then B, then A again) is right only when every externally driven component
ticks once per coupling window. When a window spans several substeps of a member, no single fixed
order is right for every sweep. `fuse_ongoing` and `fuse_step_report` both carry `next_component`:
the graph-declared component index that the compiled instruction stream expects at this process's
next `begin_step()`.

- At a period's start, `is_coupling_ongoing()` answers from the compiled plan. `-1` means this
  process hosts no externally driven component (`run()` drives such processes).
- Mid-period, `end_step()` reports the answer in its report, or `-1` if that call closed the period.
- `begin_step()` cannot resolve it yet and reports `-2` ("undetermined; call `end_step()` first").
  This is the only place `-2` appears.

A process that hosts one component, like each Brusselator executable, loops on that one component
until `end_step()` reports `next_component == -1`.

### Windows over LowFive-bound channels: `step_opened` and `delivery_pending`

This section applies only if your graph has a channel whose source field sets `hdf5_path` (see
[](lowfive.md)) and a time-window member on one side of it. For any other graph, `begin_step()`
always opens the step it is asked for, `step_opened` is `1`, and `delivery_pending` is `0`.

A channel with `hdf5_path` compiles to an external-delivery instruction that suspends FUSE between
the exporting member's step and the importing member's step. The application performs the transfer
at that point (an `H5Dwrite`/`H5Dread` pair, or `view_field()` plus `memcpy()`).

**`step_opened`.** Resuming past a suspension can discover that the current period has converged
and run straight into the next period's first request, all within one `begin_step()` call.
`step_opened` is `1` when the step you asked for opened, and `0` when resuming instead ran a period
to its end and primed the next one without opening a step. On `step_opened == 0`:

- No step is open on any component; `end_step()` on the component you named is refused with
  `E-ABI-STEP-NOT-OPEN`.
- `next_component` names the component whose request is now pending. Call `begin_step()` with that
  name next.
- `period` is set; the remaining fields are meaningless.

This is a legitimate outcome, not an error. Your loop needs one extra `begin_step()` round trip at
the period boundary.

**`delivery_pending` and `delivery_field`.** `delivery_pending` is `1` on the report of the
`begin_step()` call that returned because FUSE suspended at a LowFive-bound channel's delivery
point, and `0` on every other report (including every `end_step()` report). `delivery_field` names
the channel's source field, in the same handle convention as `fuse_field_ref::handle`; it is `-1`
whenever `delivery_pending` is `0`. Each suspension is reported on its own `begin_step()` call.
The protocol:

> On a report with `delivery_pending == 1` naming field `F`: if you produce `F`, `H5Dwrite` it; if
> you consume `F`, `H5Dread` it; then call `begin_step(next_component)`. Do this at no other point
> in the loop.

Resolve the handles of the fields you produce and consume once with `field_handle()`, compare
`delivery_field` against them, and act only on a match. `u_main.c` does exactly this:

```{literalinclude} ../../examples/brusselator/participants/u_main.c
:language: c
:lines: 625-633,642-645,657-666
```

## `view_field` and window buffers

`view_field(ctx, &ref, &view)` is the reverse of `register_field`: FUSE hands back a view of its own
storage for the handle. For an ordinary field this is the storage you registered (`window_samples`
is `0`). For a field that is part of a coupling window, after `initialize()` it is the window path's
private buffer: `window_samples` is the sample count K, `count` is unchanged, and `components` is
widened to the declared width times K, array-of-structures. Element e's K samples are contiguous at
`data[e*components ... e*components+components-1]`, with sample 0 the window-start value. It returns
`FUSE_ERR_NOTFOUND` if the handle names no field in the loaded graph. It also works for a plugin
component's fields, whose storage FUSE allocates.

## Processes without a component of their own: `run()`

`run(ctx, &run_req, NULL)` drives the graph in process. `fuse_run_request::periods` is the number
of coupling periods, or `-1` to run until the end time (`E-ABI-RUN-NO-END-TIME` if none was set;
`E-ABI-RUN-PERIODS` for any other negative value). It is refused with `FUSE_ERR_UNSUPPORTED` /
`E-ABI-RUN-EXTERNAL` only when an externally driven component is hosted on this process, never
merely because the graph declares one somewhere.

Two kinds of process use it:

- A rank that hosts only a coupler placement in a multi-executable coupling. It has no
  `begin_step()`/`end_step()` to call, but it still owes its partners its share of every period's
  collectives. Drive it with `run()` and the same `periods` count the physics ranks step, registering
  no field. `examples/brusselator/participants/coupler_main.c` is this program:

  ```{literalinclude} ../../examples/brusselator/participants/coupler_main.c
  :language: c
  :lines: 170-187,192-195,227-228,234-244
  ```

- A program whose graph names only plugin components (`launch: {mode: plugin, ...}`).
  `initialize()` mounts each plugin on the ranks of its `ranks` set, and `run()` advances them all.
  This is how a C or C++ program hosts solvers written as plugins; see
  [A driver that runs plugins](providers.md#a-driver-that-runs-plugins).

A process may also host a plugin component next to an externally driven one: `initialize()` mounts
the plugin, and FUSE advances it inside the `begin_step()`/`end_step()` calls you make for your own
component.

## `is_coupling_ongoing` and the end time

`is_coupling_ongoing` is side-effect free and can be called any number of times between steps. It
reports `period` (periods completed so far), `now` (that many periods times the plan's period, as an
exact time), `next_component`, and `ongoing`: `0` only when `setup()` set an `end_time` and `now` has
reached it (or the graph has no components). **Without an end time, `ongoing` stays `1` forever**,
and your own loop must bound the run, as the Brusselator participants do with `--periods`.

## Collective agreement

`setup`, `initialize`, `begin_step`, `end_step` and `run` are collective: every rank involved must
make the same call in the same order.

Once `setup()` has named `my_component` and supplied a communicator, `initialize()` splits a
component-local communicator from the base communicator, and from then on `initialize`,
`begin_step`, `end_step` and `run` each reconcile their status across the component's ranks with one
`MPI_Allreduce`: the most negative status wins, ties go to the lowest rank, and every rank returns
the same code. The originating rank keeps its own message; the others get
`E-ABI-COLLECTIVE-REFUSED`, naming the rank that refused. An `MPI_Allreduce` failure is
`E-ABI-COLLECTIVE-COMM`. Without `my_component`, or without a communicator, every refusal is decided
from the calling process's own state only, and a rank that is refused while its peers proceed into a
collective leaves them blocked there; the library does not detect that divergence.

When the graph declares a plugin component, `initialize()` additionally agrees the plugin mount
across every rank of the base communicator (`E-ABI-PLUGIN-AGREEMENT-REFUSED` names the refusing
rank). A plugin component together with a `connect`-launched component is refused with
`E-ABI-PLUGIN-NO-SPANNING-COMM` unless this process's own `my_component` is one of the connecting
components; see [](providers.md#a-known-failure-case).

## Iteration regions and restart requests

A participant can take part in an implicit coupling (iteration) region. FUSE captures and restores
the registered fields itself; `requires_checkpoint` and `requires_restore` on the `begin_step`
report tell you when to save and restore the private state those fields do not cover. Both are `0`
for a component no iteration region checkpoints, which is every component in a graph with no
iteration region.

`request_restart_point(ctx, NULL, NULL)` sets a flag asking that a restart point be taken, and
writes no file: writing needs a file path and layout, which the entry's fixed shape cannot carry.
Repeated calls before the flag is read leave one pending request. Nothing in the participant ABI
reads the flag or writes the file, so a program that reaches FUSE only through this table cannot
produce a restart file; restart-to-disk is available to C++ programs that own a `fuse::Runner`
([](restart.md)) and, on the provider side, through `save_restart_blob`/`load_restart_blob`
([](providers.md)).

## What `capabilities()` reports

`capabilities(NULL, NULL, &caps)` reports `FUSE_CAP_YAML`, plus `FUSE_CAP_MPI` when the library was
built with MPI. It reports nothing else: `FUSE_CAP_MPI_CONNECT`, `FUSE_CAP_CHECKPOINT`,
`FUSE_CAP_DEVICE_FIELDS` and `FUSE_CAP_LOWFIVE` are never set by this table, although MPMD and
connect launches work through `setup().my_component` and iteration regions work through
`requires_checkpoint`/`requires_restore`. Test for those by calling the entries; do not gate on
these bits.

## Kokkos and device selection

`create()` initialises Kokkos the first time it is called in a process and never finalises it per
context; an exit handler finalises it when the process ends. There is no `argv` at `create()`, so
the device a rank uses comes from the environment variable `KOKKOS_DEVICE_ID` when set; otherwise
FUSE picks `local_rank % device_count`, or device 0 when no device is visible. A participant's own
fields are host memory in any case.

## Version compatibility

The table begins with its own size and version:

```c
struct fuse_participant_abi_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  /* entry points follow, append-only across versions */
};
```

A binary compiled against v1 keeps working against a later FUSE, because later tables only append.
If you compiled against a newer header than the library you are running, compare the table's
`struct_size` with `sizeof(fuse_participant_abi_v1)` before calling an entry that a newer version
added. Every argument struct carries its own `struct_size` for the same reason: a callee reads only
the members your `struct_size` says are present and takes documented defaults for the rest, and a
`struct_size` below the oldest known layout or above `FUSE_ABI_MAX_STRUCT_SIZE` returns
`FUSE_ERR_VERSION`. Appended output members are written only when your `struct_size` covers them.

Current sizes of the structs this page discusses (`test/abi/struct_sizes.golden`, 64-bit pointers):

| Struct | `sizeof` |
|---|---|
| `fuse_participant_abi_v1` | 152 |
| `fuse_step_request` | 72 |
| `fuse_step_report` | 104 |
| `fuse_ongoing` | 40 |
| `fuse_setup_options` | 56 |
| `fuse_field_view` | 72 |

A caller built against an older header, with a smaller `struct_size`, sees the documented
defaults: `next_component` is not written, `step_opened` reads as `1`, `delivery_pending` as `0`,
and `has_communicator` as "supplied if `communicator != 0`".

## Building and linking

The participant ABI is built with the library; see [](build.md). `fuse::abi` is the shared
`libfuse-abi` (also available as `fuse::abi-static`):

```cmake
find_package(fuse-coupler CONFIG REQUIRED)
add_executable(my_participant main.c)
target_link_libraries(my_participant PRIVATE fuse::abi)
```

A pure C program needs no C++ compiler of its own; `test/abi/install_consumer/` is a complete
project that links only `fuse::abi`. For an MPI participant, link your MPI as usual and pass
`MPI_Comm_c2f(MPI_COMM_WORLD)` to `setup()`.

## The participant loop in a real code

This is the loop an actual unmodified external code runs: `u_loop_step()`, which
`examples/brusselator/participants/u_main.c` calls once per substep. It drives the Brusselator's `U`
species through `begin_step`, `step_opened`, `requires_checkpoint`/`requires_restore` and
`end_step`, with no FUSE-specific logic anywhere else in that program:

```{literalinclude} ../../examples/brusselator/participants/u_loop.c
:language: c
:lines: 7-31,56-71
```

The Fortran side of the same Brusselator pair, `V`'s `v_loop_step()`, is in [](fortran.md). The
two executables and the coupler are launched together as described in
[](multiple-executables.md).

## How `begin_step()` and `end_step()` drive FUSE

`fuse::Runner::resume_period()` (`include/fuse/runtime/runner.hpp`) runs the current period's
instruction stream up to the next externally driven component's step and returns there, or to the
period's end if no externally driven component remains due. `begin_step(name)` resumes it and checks
the step it stops at against `name`; `end_step(name)` resumes it again, past this component's step,
to the following one. Because the Runner is one shared, ordered instruction stream with exactly one
current suspension point, at most one externally driven component can have an open step on a given
context at a time.

## Fortran

Fortran gets a small set of flat `bind(C)` symbols rather than the table, because dereferencing a C
struct of function pointers from Fortran is unpleasant. They are generated from the same entry list
as the table, so the two cannot disagree. See [](fortran.md).

## Limitations

- `register_field` accepts only `FUSE_ELEMENT_F64` fields (`E-ABI-REGISTER-ELEMENT`), only
  `FUSE_MEMSPACE_HOST` storage (`E-ABI-REGISTER-MEMSPACE`), and does not accept
  `FUSE_OWNERSHIP_GHOSTED` (`E-ABI-REGISTER-OWNERSHIP`).
- At most one externally driven component has an open step per context, and `begin_step()` calls
  must follow the schedule order (`E-ABI-STEP-ALREADY-OPEN`, `E-ABI-STEP-ORDER`).
- `run()` is refused on a process hosting an externally driven component (`E-ABI-RUN-EXTERNAL`).
- Without an `end_time`, `is_coupling_ongoing` never reports `0`; the caller bounds the run.
- `request_restart_point` only raises a flag; no entry of this table writes or reads a restart
  file.
- `capabilities()` reports only `FUSE_CAP_YAML` and `FUSE_CAP_MPI`.
- Kokkos is initialised once per process; the device comes from `KOKKOS_DEVICE_ID` or the rank.
- A native C++ `fuse::Component` cannot be added to a context created through this table; the
  components of an ABI context are externally driven participants and plugins.
- The table and the argument structs grow by appending only. `include/fuse/abi/entries.def` is
  the single source they and the Fortran module are generated from.

## See also

- [](fortran.md): the same ABI from Fortran.
- [](providers.md): the opposite direction, FUSE driving a plugin you write.
- [](mixing-languages.md): which entry point to use for which code, and what can be combined.
- [](multiple-executables.md): launching participants as separate executables.
- [](restart.md): restart points.
- [](build.md): building the ABI libraries.
- [](troubleshooting.md): diagnosing `E-...` codes.
