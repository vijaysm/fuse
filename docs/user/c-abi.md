# Coupling an existing code (C and Fortran)

:::{note}
**Planned.** Specified and scheduled; see the [roadmap](../dev/roadmap.md). The design rationale is
in [](../dev/abi-design.md).
:::

FUSE exposes two C ABIs for two opposite situations:

**Participant ABI**
: Your code owns its time loop and drives FUSE. Use this to couple an existing simulation with
  minimal intrusion.

**Provider ABI**
: FUSE owns the loop and drives your component as a plugin, possibly loaded at runtime.

Both work the same way: **one exported symbol returns a versioned table of function pointers.** There
is no prefixed family of symbols to collide with anything else on your system, and the entry points
have short names because the table already namespaces them.

## Driving FUSE from C

```c
#include <fuse/abi/participant.h>

const fuse_participant_abi_v1* fuse = fuse_coupler_abi(1);
if (!fuse) { /* this build does not speak v1 */ }

fuse_ctx* ctx = fuse->create();
fuse->load_graph_yaml(ctx, "coupling.yaml");
fuse->set_communicator(ctx, (int64_t)MPI_Comm_c2f(MPI_COMM_WORLD));
fuse->register_field(ctx, "fluid", "heat_flux", q, n, 1, 0);
fuse->initialize(ctx);

int ongoing = 0;
for (fuse->is_coupling_ongoing(ctx, &ongoing); ongoing;
     fuse->is_coupling_ongoing(ctx, &ongoing)) {
  fuse->begin_step(ctx, "fluid");
  advance_my_solver();
  fuse->end_step(ctx, "fluid");
}
fuse->destroy(ctx);
```

## Time is exact across the boundary too

```c
typedef struct fuse_time { int64_t num; int64_t den; } fuse_time;
```

No FUSE entry point accepts or returns a floating-point time. If your solver keeps `double` time,
convert at the boundary and keep the exact value as the reference — otherwise a step count of a few
million drifts your schedule off the coupling window.

## Checkpoints and restarts

When FUSE is iterating a coupling to convergence, your code may be asked to roll back:

```c
int yes = 0;
fuse->requires_checkpoint(ctx, "fluid", &yes);  if (yes) save_my_state();
fuse->requires_restore(ctx, "fluid", &yes);     if (yes) restore_my_state();
```

Declare the capability honestly. A component that cannot roll back is not eligible for implicit
coupling, and FUSE will tell you so at initialisation rather than producing a wrong answer.

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
before calling an entry point that a newer version added.

## Fortran

Fortran gets a small set of flat `bind(C)` symbols rather than the table, because dereferencing a C
struct of function pointers from Fortran is genuinely unpleasant. They are generated from the same
entry list as the table, so the two cannot disagree.

```fortran
use fuse_coupler
type(fuse_ctx) :: ctx
call fuse_create(ctx)
call fuse_set_communicator(ctx, MPI_COMM_WORLD)
call fuse_register_field(ctx, "fluid", "heat_flux", q)
call fuse_initialize(ctx)
```
