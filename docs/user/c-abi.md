# Coupling an existing code (C and Fortran)

:::{note}
Both ABIs below are implemented and tested (`build-evidence/abi-matrix.toml`'s "C ABI" and
"S6 phase 4" tracks). The design rationale for both is in [](../dev/abi-design.md).
:::

FUSE exposes two C ABIs for two opposite situations:

**Participant ABI**
: Your code owns its time loop and drives FUSE. Use this to couple an existing simulation with
  minimal intrusion. This is the one described below.

**Provider ABI**
: FUSE owns the loop and drives your component as a plugin, loaded at runtime by path named in
  the graph. See [](providers.md).

Both work the same way: **one exported symbol returns a versioned table of function pointers.** There
is no prefixed family of symbols to collide with anything else on your system, and the entry points
have short names because the table already namespaces them.

## Driving FUSE from C

Every entry point has the shape `fuse_status f(fuse_ctx* ctx, const IN* in, OUT* out)`: `IN`/`OUT`
are plain structs from `fuse/abi/types.h`, each beginning with `uint32_t struct_size` that the
caller sets to `sizeof` of the struct it was compiled against (see "Version compatibility" below).
Pass `NULL` where the entry's documented type is `fuse_null` — there is nothing to fill in.

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
  fuse->begin_step(ctx, &req, &rep);        /* rep.t_begin / rep.t_end: see below */
  advance_my_solver();                      /* read/write q between the two calls */
  fuse->end_step(ctx, &req, &rep);
}
fuse->destroy(ctx, NULL, NULL);
```

**Your export must hold its initial value before `initialize()`.** For a component in a
coupling window (`docs/user/coupling-windows.md`), `initialize()` primes the first window's t = 0 sample
from whatever the storage you registered for the export holds at that moment, and the partner
integrates that sample. An in-process component's constructor has already written its initial
value there; your program has to do the same by hand — compute the export at t = 0 (for a
reaction rate that depends on the partner's initial state, that means knowing the partner's
initial export too; `examples/brusselator/participants/u_main.c` rebuilds a throwaway partner
for it) and store it before calling `initialize()`. Left at zero, the first window's exchange is
measurably short on one side and nothing refuses it. The storage must also be the field's size on
this process: a shorter or longer registration is refused at `initialize()` with `E-RT-SIZE`.
This startup contract — a participant knowing its partner's initial export — is recorded as a
limitation in `KNOWN_ISSUES.md` D25; a coupler-delivered initial exchange would remove it.

**Reading an error.** Every entry point above returns a `fuse_status` — `FUSE_OK` (`0`) or a
negative `FUSE_ERR_*` — but not the message. `fuse->last_error(ctx, NULL, &err)` fills a caller-owned
`fuse_error` with the failed call's own `E-...` code (`err.code`, static storage, valid until the
next failing call), its full message length (`err.length`), and as much of the message text as fits
in `err.buffer`/`err.capacity`, always NUL-terminated (`err.truncated` is `1` if it did not fit, and
`err.length` is then the length a second call would need; pass `buffer = NULL` to query the length
alone, without copying anything):

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

`ctx` may be `NULL` here specifically to read the thread-local error of a failed `fuse->create()` —
the one call that cannot yet have a context to attach an error to.

**Two entries this document otherwise has no occasion to use.** `fuse->capabilities(NULL, NULL,
&caps)` reports what this build can do as a `FUSE_CAP_*` bit set (`fuse/abi/types.h`) — a caller
that must work across more than one FUSE build checks this before relying on an optional feature,
the same way it checks `struct_size` before calling an entry a newer header added ("Version
compatibility" below). `fuse->save_graph(ctx, NULL, &sink)` writes the loaded graph's own canonical
bytes back out (`fuse_graph_sink::buffer`/`capacity`/`written`, the mirror of `load_graph`'s
`fuse_graph_source`) — for a caller that loaded a graph from YAML and wants the canonical descriptor
bytes to pass along (to a second process, or to store alongside a checkpoint), not a round trip any
example above needs.

**What "imports are current" actually means, measured rather than assumed (KNOWN_ISSUES.md D13).**
`fuse_begin_step()` resumes FUSE's own instruction stream up to this component's own step and stops
there: every exchange the SAME schedule the in-process `fuse::Runner` would run places before this
tick — every feeding partner's `Transport`/`Remap` the compiled plan orders earlier — has already
happened by the time `begin_step()` returns, and a `Requires` field fed by one of those is current
to read **right there**, between `begin_step()` and `end_step()`, not just after `end_step()`. What
the schedule places before this tick is a property of the graph, not of this ABI: a `lag(1)`
channel between two components of the SAME interval is, by construction, ordered *after* both
components' ticks for the period (the exchange is one period behind on purpose), so its import is
still the previous period's value at `begin_step()` and only becomes this period's at the START of
the NEXT period — correct for that graph, not a staleness bug, and unchanged by this fix (measured
directly; see `AbiParticipant.SuspensionHappensExactlyAtTheScheduledTickNeitherEarlyNorLate`,
`test/abi/participant_test.cpp`, for the exact before/after). Where the fix changes what a caller
actually sees: `AbiParticipant.SubcycledMeanAccumulation` (a time-averaging `Requires` field is
correct mid-period, not just at the period's end) and `AbiParticipant.ImplicitCouplingConvergesLikeInProcess`
(an iteration region converges through the ABI exactly as it does in-process). The only ordering
constraint is on the CALLER: a process hosting more than one externally-driven component must call
`begin_step()`/`end_step()` for each in the order FUSE's own compiled schedule visits their steps —
calling `begin_step()` for a component out of turn is refused (`E-ABI-STEP-ORDER`), naming the
component the schedule actually expects next.

**Knowing the order in advance, not just on refusal (`next_component`).** A fixed alternation
(call A, then B, then A again, ...) is only right when every externally-driven component ticks
once per coupling window; once a window spans more than one of a member's own substeps — a
coupling window of `4*dt_u` against a member that ticks every `dt_u`, say — there is no single
fixed order that is right for every Picard sweep, and guessing wrong just to learn the real order
from `E-ABI-STEP-ORDER`'s own error message does not scale past the first mistake. `fuse_ongoing`
and `fuse_step_report` both carry a `next_component` member for exactly this: the graph-declared
component index FUSE's own compiled instruction stream expects at THIS process's own next
`begin_step()`. A caller never has to guess, not even for the very first step of a period:

- **At a period's own start** (nothing resolved yet this period), `fuse_is_coupling_ongoing()`
  answers straight from the compiled plan — the first externally driven component of a period is
  a fixed property of the schedule, the same every period, with no need to resume the instruction
  stream to find out. `-1` means THIS PROCESS hosts no externally-driven component of its own
  (`fuse_run()` drives those processes) — either because the graph has none at all, or because
  this process happens to host none of the ones it does declare (a dedicated coupler rank, or any
  other rank whose own components are all driven in-process).
- **Mid-period**, after at least one step, `fuse_end_step()` resolves the real answer for free —
  it already resumes the instruction stream internally to find out what comes next, so reading
  `next_component` off its own report costs nothing extra.
- **`fuse_begin_step()`** alone cannot resolve it synchronously (this component's own step has not
  run yet, so the stream has not been asked to look past it) and reports `-2` ("undetermined —
  call `fuse_end_step()` first"); this is the ONLY place `-2` appears — `fuse_is_coupling_ongoing()`
  always has a real answer, from the plan or from a prior step's own cache.

A caller driving several externally-driven window members therefore never guesses: query
`fuse_is_coupling_ongoing()` once at the start of each period, then read `next_component` off
each `end_step()`'s own report for every step after that. `next_component` was appended after
`fuse_step_report`/`fuse_ongoing` first shipped (see "Version compatibility" below): an older
build that only allocated the pre-growth struct never has this member written into its memory at
all, and keeps
working exactly as it did before this field existed.

## `step_opened`: the one report field a window member sharing a LowFive-bound channel must check

A time-window member on one side of an implicit coupling iteration cannot know from outside
whether that iteration has converged — the decision is made by the sweep loop itself, after this
component's own step runs, and nothing in `fuse_step_report`/`fuse_ongoing` can tell a caller in
advance whether `fuse_begin_step()` is about to open an ordinary step or instead resume straight
through a period's own end. This matters in exactly one case: a channel whose source field sets
`hdf5_path` (S6 phase 3, "LowFive" track, `lowfive.md`) compiles to an `ExternalDelivery`
instruction, which suspends the Runner between the exporting member's own step and the importing
member's — the one place an application-performed transfer (an `H5Dwrite`/`H5Dread` pair, or
`view_field()` + `memcpy()` in the tier-0 harness) must run. Resuming past that suspension can
legitimately discover the current period has converged and run straight into the next one's own
first request, all within the SAME `fuse_begin_step()` call.

`fuse_step_report::step_opened` is `1` on every ordinary step — the step this call was asked for
actually opened, exactly as every call before this field existed always did. It is `0` when
resuming instead ran a period to its end and primed the next one's first request, WITHOUT opening
a step on the component `fuse_begin_step()` was asked for: that component's real next turn may
belong to a different period, or a different component, than the one a caller watching only
`next_component`'s own mid-period hint would expect (that hint is a CONDITIONAL answer — the
component due if the current iteration continues — never a promise of what the next
`fuse_begin_step()` call will actually do, because whether the iteration continues is a
convergence decision not yet taken when the hint is reported). On `step_opened == 0`:

- No step is open on ANY component — a matching `fuse_end_step()` on the component this call named
  is refused (`E-ABI-STEP-NOT-OPEN`), the same refusal any unmatched `end_step()` already gets.
- `next_component` names the component whose request is now pending — the caller's own loop body
  must call `fuse_begin_step()` with THAT name next, before doing anything else. If it happens to
  be the SAME component the caller just asked for (a *matched* rollover — the new period's own
  first-due component coincides with what was asked), the very next call opens its step normally;
  if it names a DIFFERENT component (a *mismatched* rollover), that one opens instead.
- `period` is set on this report too (unlike every other `fuse_step_report` field, which is
  meaningless when no step opened) — the one signal that distinguishes a rollover reached this way
  from an ordinary first call of a period a caller already knew about from `fuse_is_coupling_
  ongoing()`'s own, separately-read `period`.

This is a legitimate outcome, not an error: unlike the ordinary out-of-turn mistake
(`E-ABI-STEP-ORDER`, above), nothing was asked wrong here, and the caller's own loop simply needs
one extra, cheap `begin_step()`/`end_step()` round trip at the period boundary. `step_opened` is
always `1` on the ordinary path (no time window, or no LowFive-bound channel feeding one), which
still always opens the step it is asked for — this field changes nothing there, and an old caller
whose `struct_size` predates it reads it as `1`, matching every call's behavior before it existed
(the growth-only rule, "Version compatibility" below). A caller driving a LowFive-bound window
member's loop body therefore reads `step_opened` on every `fuse_begin_step()` report and, on `0`,
calls `fuse_begin_step(next_component)` again immediately rather than treating `end_step()` as
reachable.

## `delivery_pending`: when the application-performed transfer must actually run

`step_opened == 0` says a rollover reached this call; it does not, by itself, say whether that
rollover passed through a LowFive-bound channel's own suspension (the one place an
application-performed transfer is required at all, this document's own `step_opened` section
above) or simply primed an ordinary period's first request with nothing for the caller to move.
`fuse_step_report::delivery_pending` resolves that directly: it is `1` on the report of the
`fuse_begin_step()` call that returned because the Runner suspended at a LowFive-bound channel's
own `ExternalDelivery` proxy, and `0` on every other report — the "run is over" report, every
ordinary substep's `begin_step()`/`end_step()` report, and every `fuse_end_step()` report without
exception (mirroring `step_opened`'s own always-`1`-from-`end_step()` precedent: a matching
reason, the opposite value, since `end_step()` only ever echoes a step that WAS open).

**One proxy, one report, one field.** A graph can have more than one LowFive-bound channel, each
with its own `ExternalDelivery` proxy, and the two ends of the SAME channel can be hosted on
different processes — so `deliver_external()` (the Runner's own suspension point) suspends once
for the exporting process (right after that channel's own producer finishes a sweep: "your export
is complete, write it") and once for the importing process ("this channel's import is due"), and
every one of those suspensions is reported on its OWN `fuse_begin_step()` call — never collapsed
or drained into a single report the way an earlier version of this protocol did. `fuse_step_report
::delivery_field` names WHICH channel a given `delivery_pending == 1` report is about: the
channel's own SOURCE field, as the same `FieldId`-derived handle `fuse_field_ref::handle` uses
(`fuse_field_handle()`'s own convention) — `-1` whenever `delivery_pending` is `0`.

The protocol this makes exact: **on a report with `delivery_pending == 1` naming field `F`
(`delivery_field`): if you produce `F`, `H5Dwrite` it; if you consume `F`, `H5Dread` it; then call
`fuse_begin_step(next_component)` — never at any other point in the loop.** A caller checks `F`
against the field handles of whatever it produces and consumes (resolved once, at setup, via
`fuse_field_handle()`) and acts only on a match — no guessing which of a graph's possibly several
independently-paced LowFive-bound channels this particular suspension is about, and no assuming
every owned import/export is simultaneously ready. Reading or writing on every ordinary substep
(the simplification Unit 3b's own harness and both Brusselator mains used before this field
existed) is no longer just unnecessary — on a graph with more than one LowFive-bound channel
running at different paces it is wrong, since it can write or read a value before its own
producer has actually finished the sweep it belongs to. An old caller whose `struct_size` predates
`delivery_pending`/`delivery_field` never has either field written into its memory at all, and
keeps working exactly as it did before (the growth-only rule, "Version compatibility" below) —
which is also, not coincidentally, why `step_opened` alone was never enough to tell a NEW caller a
transfer is owed: a graph can roll over for reasons `step_opened` reports identically whether or
not a LowFive-bound channel was involved, and only `delivery_pending`/`delivery_field` distinguish
the two and name which one.

## `fuse_run()` and ranks without a component of their own

`fuse_run()` drives a graph in process, for a caller that does not need to step any component from
outside at all. It refuses (`FUSE_ERR_UNSUPPORTED`, `E-ABI-RUN-EXTERNAL`) only when **an externally
driven component is hosted on this process** — never merely because the graph declares one
*somewhere*. A process whose own components are all driven in process, and a process that hosts no
component at all (a dedicated coupler placement, or any other rank with nothing of its own to
step), can both call `fuse_run()` freely, even in a graph that has externally driven components on
*other* ranks.

This matters for a bundled, multi-executable coupling: a rank that hosts only a coupler placement
has no `begin_step()`/`end_step()` of its own to call, but it still owes its partners their own
share of every period's collectives (the Transport/Remap a channel routes through it). `fuse_run()`
is how that rank executes its own period, with the same `periods` argument every physics rank steps
— drive it exactly as a fully in-process caller would, from `fuse_initialize()` straight through to
`fuse_destroy()`, registering no field at all.

## Time is exact across the boundary too

```c
typedef struct fuse_exact_time { int64_t numerator; int64_t denominator; } fuse_exact_time;
```

No FUSE entry point accepts or returns a floating-point time: `fuse_step_report::t_begin`/`t_end`
are the schedule's own `Rational` bounds, copied field for field, never rounded through a `double`.
If your solver keeps `double` time, convert at the boundary and keep the exact value as the
reference — otherwise a step count of a few million drifts your schedule off the time window.

## Checkpoints and restarts

An externally-driven component may take part in an implicit coupling region (spec §5.3)
(KNOWN_ISSUES.md D13). `fuse_step_report::requires_checkpoint` is `1` on the step immediately after
FUSE's own `save_state()` captured this component — save whatever private state your solver keeps
that FUSE's registered fields do not cover. `requires_restore` is `1` on every step after a
re-sweep's `restore_state()` rolled this component back — restore that saved state (and nothing
else; registered fields are already restored) before advancing. Both are `0` for a component never
checkpointed by any region, which is every component in a graph with no iteration region at all.
`fuse_step_report::iteration` is the 0-based sweep index inside the region (0 outside one), exactly
as `StepContext::iteration` (`include/fuse/runtime/component.hpp`) documents for an in-process
component.

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
If you compiled against a newer header than the library you are running, compare `struct_size`
before calling an entry point that a newer version added; the same comparison applies to every
individual argument struct (`fuse/abi/types.h`), each of which carries its own `struct_size` for
exactly this reason.

**Growth history, stated once.** The three structs this document discusses members of by name have
each grown since they were first frozen — every append documented, by commit, where it landed:

| Struct | `sizeof` | Commit | What that append added |
|---|---|---|---|
| `fuse_step_request` | 16 → 72 | `612b153` | the provider direction's own window/flags members (`t_begin`, `t_end`, `step`, `iteration`, `requires_checkpoint`, `requires_restore`, `reserved`) — unused by `begin_step`/`end_step`, read only by a provider's own `advance()` |
| `fuse_ongoing` | 32 → 40 | `da33572` | `next_component` ("Knowing the order in advance" above) |
| `fuse_step_report` | 64 → 72 | `da33572` | `next_component`, alongside `fuse_ongoing`'s own append |
| `fuse_step_report` | 72 → 88 | `f768fb3` | `step_opened` and `period` together (the "`step_opened`" section above) |
| `fuse_step_report` | 88 → 104 | `dfcd9d1` | `delivery_pending` and `delivery_field` together (the "`delivery_pending`" section above) |

Every intermediate append also claims its own few bytes of trailing padding as an explicitly named
`reserved*` member (`abi-design.md` rule 3: a struct whose size does not change when a member is
appended would let that member hide inside existing padding, invisible to the `struct_size`
comparison every caller is supposed to make) — visible in `include/fuse/abi/entries.def` itself,
not repeated member by member here.

## The participant loop in a real external code

The example above is a minimal, hand-written illustration. This is the loop an actual unmodified
external code runs — `u_loop_step()`, the function `examples/brusselator/participants/u_main.c`
calls once per substep, driving the Brusselator's own `U` species through exactly the entry points
documented above (`begin_step`, `step_opened`, `requires_checkpoint`/`requires_restore`,
`end_step`), with no FUSE-specific logic anywhere else in that program:

```{literalinclude} ../../examples/brusselator/participants/u_loop.c
:language: c
```

The Fortran side of the SAME Brusselator pair — `V`'s own `v_loop_step()` — is in
[](fortran.md)'s own matching section, member for member.

## How begin_step()/end_step() really drive FUSE

`fuse::Runner::resume_period()` (`include/fuse/runtime/runner.hpp`) is the real `fuse::Runner`,
resumed from wherever it last suspended rather than driven in one call: it runs the current
period's instruction stream up to the next externally-driven component's own step and returns there
— or to the period's end, if no externally-driven component remains due. `fuse_begin_step(name)`
resumes it and checks the step it stops at against `name`; `fuse_end_step(name)` resumes it again,
past this component's step, to the following one. `src/abi/context.hpp`'s header comment has the
full mechanism, including why at most one externally-driven component may have an open step at a
time on a given context (`E-ABI-STEP-ALREADY-OPEN`) — the Runner is one shared, ordered
instruction stream, with exactly one current suspension point to resume from, so a process hosting
several externally-driven components must complete each one's `begin_step()`/`end_step()` pair
before opening the next.

## Fortran

Fortran gets a small set of flat `bind(C)` symbols rather than the table, because dereferencing a C
struct of function pointers from Fortran is genuinely unpleasant. They are generated from the same
entry list as the table, so the two cannot disagree. See [](fortran.md) for the module, the calling
convention, and a worked example.
