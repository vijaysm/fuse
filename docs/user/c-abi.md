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

**What "imports are current" actually means, measured rather than assumed.** A graph with more
than one externally-driven component defers the real field exchange to the point every one of
them has completed its step for the current coupling period — known only once the LAST such
component's `end_step()` returns. Read a `Requires` field after your own `end_step()`, not between
your `begin_step()`/`end_step()` pair, if another externally-driven component's export feeds it;
`test/abi/install_consumer/main.c` and `src/abi/context.hpp`'s own header comment work through the
exact ordering this requires and the open question it leaves (closing it for every possible calling
order needs a change to `fuse::Runner` itself, not attempted in this pass).

## Time is exact across the boundary too

```c
typedef struct fuse_exact_time { int64_t numerator; int64_t denominator; } fuse_exact_time;
```

No FUSE entry point accepts or returns a floating-point time: `fuse_step_report::t_begin`/`t_end`
are the schedule's own `Rational` bounds, copied field for field, never rounded through a `double`.
If your solver keeps `double` time, convert at the boundary and keep the exact value as the
reference — otherwise a step count of a few million drifts your schedule off the time window.

## Checkpoints and restarts

Not yet implemented through this ABI: `fuse_step_report::requires_checkpoint` /
`requires_restore` are always `0` today, because `begin_step()`/`end_step()` do not yet support an
iteration region driven from outside. An open question in the task that implemented the rest of
this ABI, not a silent gap — a component that needs implicit coupling's rollback contract must, for
now, be driven in-process (through `fuse_run()`) rather than through `begin_step()`/`end_step()`.

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

## Known limitation

`fuse_begin_step()`/`fuse_end_step()` drive the real `fuse::Runner` by deferring one
`run_period()` call to the point every externally-driven component in the graph has completed a
period (`src/abi/context.hpp`'s header comment has the full mechanism). This is exact for the
ordinary case — reading a `Requires` field after your own `end_step()` — but is not a literal
implementation of entries.def's "imports are current on return [of begin_step]" for every possible
interleaving of calls across multiple externally-driven components. Closing that gap fully would
need `fuse::Runner` itself to support pausing mid-period, which this task's scope did not include;
see its reported open questions.

## Fortran

Fortran gets a small set of flat `bind(C)` symbols rather than the table, because dereferencing a C
struct of function pointers from Fortran is genuinely unpleasant. They are generated from the same
entry list as the table, so the two cannot disagree. See [](fortran.md) for the module, the calling
convention, and a worked example.
