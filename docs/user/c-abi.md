# Coupling an existing C or C++ code

This page is for people who have a simulation with its own time loop and want to couple it to other
components through FUSE with as little change as possible. Your code keeps ownership of the loop and
calls FUSE at the start and end of each step, through the **participant ABI**. Fortran codes use the
same ABI through a module; see [](fortran.md).

FUSE has two C ABIs for two opposite situations:

**Participant ABI** (this page)
: Your code owns its time loop and drives FUSE.

**Provider ABI**
: FUSE owns the loop and drives your component as a plugin, loaded at runtime from a path named in
  the graph. See [](providers.md).

Both work the same way: **one exported symbol returns a versioned table of function pointers.** There
is no prefixed family of symbols to collide with anything else on your system, and the entry points
have short names because the table already namespaces them. The design rationale is in
[](../dev/abi-design.md).

## The smallest working program

Every entry point has the shape `fuse_status f(fuse_ctx* ctx, const IN* in, OUT* out)`. `IN` and
`OUT` are plain structs from `fuse/abi/types.h`, each beginning with `uint32_t struct_size`, which
the caller sets to `sizeof` of the struct it was compiled against (see
[Version compatibility](#version-compatibility)). Pass `NULL` where the entry's type is `fuse_null`:
there is nothing to fill in. In prose below, `fuse_begin_step()` and `fuse->begin_step` both mean the
`begin_step` member of the table.

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

The call order is: `create`, `load_graph`, optionally `setup` (communicator and end time),
`field_handle` and `register_field` for each field your component imports or exports, `initialize`,
then `begin_step`/`end_step` once per step until `is_coupling_ongoing` reports `ongoing == 0`, then
`destroy`. `setup`, `initialize`, `begin_step` and `end_step` are collective over the processes they
involve.

The table also has `version`, `capabilities`, `save_graph`, `run`, `view_field`,
`base_communicator` and `request_restart_point`, described below.

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

Every entry point returns a `fuse_status`: `FUSE_OK` (`0`) or a negative `FUSE_ERR_*`
(`FUSE_ERR_INVALID`, `FUSE_ERR_NOTFOUND`, `FUSE_ERR_STATE`, `FUSE_ERR_RUNTIME`,
`FUSE_ERR_UNSUPPORTED`, `FUSE_ERR_VERSION`). The status does not carry the message.
`fuse->last_error(ctx, NULL, &err)` fills a caller-owned `fuse_error` with:

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

`ctx` may be `NULL` here specifically to read the thread-local error of a failed `fuse->create()`,
the one call that cannot yet have a context to attach an error to.

Two further entries: `fuse->capabilities(NULL, NULL, &caps)` reports what this build can do as a
`FUSE_CAP_*` bit set (`fuse/abi/types.h`). A caller that must work across more than one FUSE build
checks this before relying on an optional feature. `fuse->save_graph(ctx, NULL, &sink)` writes the
loaded graph's canonical bytes back out (`fuse_graph_sink::buffer`/`capacity`/`written`, the mirror
of `load_graph`'s `fuse_graph_source`), for a caller that wants to pass the canonical descriptor to a
second process or store it alongside a checkpoint.

## What is current between `begin_step()` and `end_step()`

`fuse_begin_step()` resumes FUSE's own instruction stream up to this component's own step and stops
there. Every exchange that the compiled schedule places before this tick (every feeding partner's
`Transport`/`Remap`) has already happened by the time `begin_step()` returns, so a `Requires` field
fed by one of those is current to read right there, between `begin_step()` and `end_step()`.

What the schedule places before this tick is a property of the graph. A `lag(1)` channel between two
components of the same interval is by construction ordered after both components' ticks for the
period, so its import is still the previous period's value at `begin_step()` and becomes this
period's at the start of the next period. That is the intended behaviour of `lag(1)`.

## Several components in one process: the call order

A process hosting more than one externally driven component must call `begin_step()`/`end_step()` for
each in the order FUSE's compiled schedule visits their steps, and must finish each component's
`begin_step()`/`end_step()` pair before opening the next (a second open step is refused with
`E-ABI-STEP-ALREADY-OPEN`). Calling `begin_step()` for a component out of turn is refused with
`E-ABI-STEP-ORDER`, and the message names the component the schedule expects next. `end_step()` with
no matching open step is refused with `E-ABI-STEP-NOT-OPEN`.

### Knowing the order in advance: `next_component`

A fixed alternation (A, then B, then A again) is right only when every externally driven component
ticks once per coupling window. When a window spans several substeps of a member (a window of
`4*dt_u` against a member that ticks every `dt_u`, say), no single fixed order is right for every
sweep. `fuse_ongoing` and `fuse_step_report` both carry a `next_component` member for this: the
graph-declared component index that FUSE's compiled instruction stream expects at this process's
next `begin_step()`.

- At a period's start, `is_coupling_ongoing()` answers from the compiled plan. `-1` means this
  process hosts no externally driven component (`run()` drives such processes), either because the
  graph has none or because this process hosts none of them.
- Mid-period, `end_step()` reports the answer in its own `fuse_step_report`.
- `begin_step()` cannot resolve it yet and reports `-2` ("undetermined; call `end_step()` first").
  This is the only place `-2` appears.

Query `is_coupling_ongoing()` once at the start of each period, then read `next_component` off each
`end_step()` report.

## Windows over LowFive-bound channels: `step_opened` and `delivery_pending`

This section applies only if your graph has a channel whose source field sets `hdf5_path` (see
[](lowfive.md)) and a time-window member on one side of it. For any other graph, `begin_step()`
always opens the step it is asked for, `step_opened` is `1`, and `delivery_pending` is `0`; you can
skip this section.

A channel with `hdf5_path` compiles to an external-delivery instruction that suspends FUSE between
the exporting member's step and the importing member's step. The application performs the transfer
at that point (an `H5Dwrite`/`H5Dread` pair, or `view_field()` plus `memcpy()`).

### `step_opened`

Resuming past a suspension can discover that the current period has converged and run straight into
the next period's first request, all within the same `begin_step()` call. `fuse_step_report::
step_opened` is `1` when the step you asked for opened, and `0` when resuming instead ran a period to
its end and primed the next one without opening a step. On `step_opened == 0`:

- No step is open on any component; `end_step()` on the component you named is refused with
  `E-ABI-STEP-NOT-OPEN`.
- `next_component` names the component whose request is now pending. Call `begin_step()` with that
  name next. If it is the same component you just asked for, the next call opens its step normally;
  otherwise the other component's step opens.
- `period` is set on this report (on all other reports where no step opened, the remaining
  fields are meaningless).

This is a legitimate outcome, not an error. Your loop needs one extra `begin_step()` round trip at
the period boundary.

### `delivery_pending` and `delivery_field`

`step_opened == 0` does not say whether a transfer is owed. `fuse_step_report::delivery_pending` does:
it is `1` on the report of the `begin_step()` call that returned because FUSE suspended at a
LowFive-bound channel's delivery point, and `0` on every other report (including every `end_step()`
report). `delivery_field` names which channel the suspension is about, as the channel's source field
in the same handle convention as `fuse_field_ref::handle`; it is `-1` whenever `delivery_pending` is
`0`.

A graph can have several LowFive-bound channels, and the two ends of one channel can be on different
processes, so each suspension is reported on its own `begin_step()` call. The protocol:

> On a report with `delivery_pending == 1` naming field `F`: if you produce `F`, `H5Dwrite` it; if
> you consume `F`, `H5Dread` it; then call `begin_step(next_component)`. Do this at no other point
> in the loop.

Resolve the handles of the fields you produce and consume once at setup with `field_handle()`, compare
`delivery_field` against them, and act only on a match. Reading or writing on every ordinary substep
is wrong on a graph with more than one LowFive-bound channel running at different paces, because it
can write or read a value before its producer has finished the sweep it belongs to.

`view_field(ctx, &ref, &view)` is the reverse of `register_field`: FUSE hands back a view of its own
storage for the handle. For an ordinary field this is the storage you registered (`window_samples`
is `0`). For a field that is part of a coupling window, after `initialize()` it is the window path's
private buffer, `window_samples` is the sample count K, and `components` is widened to the declared
width times K, array-of-structures. Element e's K samples are contiguous at
`data[e*components ... e*components+components-1]`, with sample 0 the window-start value. It returns
`FUSE_ERR_NOTFOUND` if the handle names no field in the loaded graph.

The full loop of a real participant is shown in
[The participant loop in a real code](#the-participant-loop-in-a-real-code).

## Processes without a component of their own: `run()`

`fuse->run(ctx, &run_req, NULL)` drives a graph in process, for a caller that does not need to step
any component from outside. `fuse_run_request::periods` is the number of coupling periods, or `-1`
to run until the end time. It is refused (`FUSE_ERR_UNSUPPORTED`, `E-ABI-RUN-EXTERNAL`) only when an
externally driven component is hosted on this process, never merely because the graph declares one
somewhere. A process whose components are all driven in process, and a process that hosts no
component at all, can both call `run()`.

This matters for a bundled multi-executable coupling. A rank that hosts only a coupler placement has
no `begin_step()`/`end_step()` to call, but it still owes its partners its share of every period's
collectives. Drive it with `run()` and the same `periods` count the physics ranks step, from
`initialize()` straight through to `destroy()`, registering no field.

## Every rank of a component must make the same calls

For a component spanning more than one MPI rank, every rank must call `begin_step()`, `end_step()`,
`run()`, `setup()` and `initialize()` in the same order as its peers. A refusal from one of these
entries (`E-ABI-STEP-ORDER`, `E-ABI-STEP-ALREADY-OPEN`, a version or struct-size mismatch, or any
other code on this page) is decided from this process's own state only. It is not reconciled across
the component's ranks.

If one rank calls an entry out of turn, or omits a call its peers make, it is refused and returns
promptly, while its peers block in the collective it never joins. The library does not detect or
correct such a divergence. Keep every rank's call sequence identical.

## Exact time

```c
typedef struct fuse_exact_time { int64_t numerator; int64_t denominator; } fuse_exact_time;
```

No entry point accepts or returns a floating-point time. `fuse_step_report::t_begin`/`t_end` are the
schedule's exact rational bounds. If your solver keeps `double` time, convert at the boundary and keep
the exact value as the reference; otherwise a step count of a few million drifts your schedule off
the time window.

## Iteration regions and restart requests

A participant can take part in an implicit coupling (iteration) region.

- `fuse_step_report::requires_checkpoint` is `1` on the step immediately after FUSE captured this
  component's registered fields. Save whatever private state your solver keeps that FUSE's
  registered fields do not cover.
- `requires_restore` is `1` on every step after a re-sweep rolled this component back. Restore your
  saved private state (and nothing else; registered fields are already restored) before advancing.
- Both are `0` for a component no iteration region checkpoints, which is every component in a
  graph with no iteration region.
- `iteration` is the 0-based sweep index inside the region, and `0` outside one.

`fuse->request_restart_point(ctx, NULL, NULL)` asks that a restart point be taken. It sets a flag and
writes no file: the embedding host process, which owns the restart file path and layout, polls
`fuse::Runner::take_restart_point_request()` after each `run_period()`/`resume_period()` call and
decides whether to write. Repeated calls before the host polls leave one pending request. It has no
effect before `initialize()`. See [](restart.md).

## Recovering the base communicator

`fuse_setup_options::communicator` is an input. When `my_component` names an `mpmd`- or
`connect`-launched component, `setup()` forms the real base communicator from it (a duplicate of
the shared MPMD world, or the merged intracommunicator from an accept/connect handshake); see
[](multiple-executables.md). `fuse->base_communicator(ctx, NULL, &info)` reads it back:

```c
fuse_base_communicator_info info = {.struct_size = sizeof(info)};
fuse->base_communicator(ctx, NULL, &info);
if (info.formed) {
  MPI_Comm base = MPI_Comm_f2c((MPI_Fint)info.communicator);
  /* this process's own base communicator, for every launch mode */
}
```

It is legal in any context state, including before `setup()` (`formed == 0`, `communicator == 0`).
Check `formed` first: `communicator == 0` is also a valid Fortran handle for `MPI_COMM_WORLD` under
some MPI implementations, and `formed` is `1` when `setup()` was given a communicator, including
handle 0 with `has_communicator = 1`. The call is local and non-collective, and the handle value is
process-local like any MPI communicator handle.

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
`FUSE_ERR_VERSION`.

Current sizes of the structs this page discusses by member name (`test/abi/struct_sizes.golden`):

| Struct | `sizeof` |
|---|---|
| `fuse_step_request` | 72 |
| `fuse_step_report` | 104 |
| `fuse_ongoing` | 40 |
| `fuse_setup_options` | 56 |
| `fuse_field_view` | 72 |

A caller built against an older header, with a smaller `struct_size`, never has the newer members
written into its memory and sees their documented defaults: `next_component` is not written,
`step_opened` reads as `1`, `delivery_pending` as `0`, and `has_communicator` as "supplied if
`communicator != 0`".

## The participant loop in a real code

This is the loop an actual unmodified external code runs: `u_loop_step()`, which
`examples/brusselator/participants/u_main.c` calls once per substep. It drives the Brusselator's `U`
species through `begin_step`, `step_opened`, `requires_checkpoint`/`requires_restore` and
`end_step`, with no FUSE-specific logic anywhere else in that program:

```{literalinclude} ../../examples/brusselator/participants/u_loop.c
:language: c
```

The Fortran side of the same Brusselator pair, `V`'s `v_loop_step()`, is in [](fortran.md).

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
- `run()` is refused on a process hosting an externally driven component
  (`E-ABI-RUN-EXTERNAL`).
- `request_restart_point` only raises a flag; the embedding host performs the write.

## See also

- [](fortran.md): the same ABI from Fortran.
- [](providers.md): the opposite direction, FUSE driving a plugin you write.
- [](multiple-executables.md): launching participants as separate executables.
- [](restart.md): restart points.
- [](build.md): building the ABI libraries.
- [](troubleshooting.md): diagnosing `E-...` codes.
