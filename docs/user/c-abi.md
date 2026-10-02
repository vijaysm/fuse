# Coupling an existing code (C and Fortran)

:::{note}
The participant ABI below (`libfuse-abi`, `include/fuse/abi/{types.h,participant.h}`) is
implemented and tested (`build-evidence/abi-matrix.toml`'s "C ABI" track). The provider ABI (FUSE
owning the loop and driving your component as a plugin) is still planned; see the
[roadmap](../dev/roadmap.md). The design rationale for both is in [](../dev/abi-design.md).
:::

FUSE exposes (or will expose) two C ABIs for two opposite situations:

**Participant ABI**
: Your code owns its time loop and drives FUSE. Use this to couple an existing simulation with
  minimal intrusion. This is the one described below.

**Provider ABI**
: FUSE owns the loop and drives your component as a plugin, possibly loaded at runtime. Not yet
  implemented.

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
