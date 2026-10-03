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
