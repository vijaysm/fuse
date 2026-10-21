# Writing a provider plugin (C and Fortran)

The provider ABI is the other direction from [](c-abi.md): instead of your code driving FUSE,
**FUSE drives your component**, loaded at runtime from a shared object your graph names by path.
Implemented and tested — `build-evidence/abi-matrix.toml`'s S6 phase 4 rows, run four ways against
a native in-process reference by `test/abi/conformance_test.cpp`. The design rationale, including
why this is a *second* table rather than a flag on the participant one, is in
[](../dev/abi-design.md#provider-direction).

## The one symbol your plugin exports

```c
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version);
```

Exactly this, from `#include <fuse/abi/provider.h>`. FUSE's own `fuse::abi_detail::Registry`
`dlopen()`s your shared object and `dlsym()`s this one name — the same "one exported symbol"
discipline the participant direction uses (`docs/dev/abi-design.md`, point 1), applied to a plugin
instead of to `libfuse-abi` itself. Return `NULL` for any `abi_version` you do not speak; FUSE
refuses the mount with `E-ABI-VERSION` rather than guessing at a subset.

## The table

```c
typedef struct fuse_provider_abi_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  fuse_status (*version)(fuse_provider_ctx* self, const fuse_null* in, fuse_version_info* out);
  fuse_status (*create)(fuse_provider_ctx* self, const fuse_null* in, fuse_provider_instance* out);
  fuse_status (*destroy)(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out);
  fuse_status (*describe)(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out);
  fuse_status (*bind)(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out);
  fuse_status (*advance)(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out);
  fuse_status (*save_state)(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out);
  fuse_status (*restore_state)(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out);
  fuse_status (*capabilities)(fuse_provider_ctx* self, const fuse_null* in, fuse_capability_set* out);
  fuse_status (*save_restart_blob)(fuse_provider_ctx* self, const fuse_null* in, fuse_restart_blob_sink* out);
  fuse_status (*load_restart_blob)(fuse_provider_ctx* self, const fuse_restart_blob_source* in, fuse_null* out);
} fuse_provider_abi_v1;
```

Every member has the shape `fuse_status f(fuse_provider_ctx* self, const IN* in, OUT* out)` — THE
ONE RULE, same as the participant table, with `self` (your own opaque instance, `create()`'s own
output) taking the `ctx` slot. `self` is a *different* C type from the participant direction's
`fuse_ctx*`, so the two tables' handles cannot be passed to each other by accident even though both
are plain pointers.

The last two entries, `save_restart_blob`/`load_restart_blob` (appended per this table's own
append-only rule, never inserted earlier), are the provider-direction counterpart of restart-to-disk
Option B (`docs/dev/restart.md`) -- an opaque, versioned byte record for whatever `save_state()`'s
in-memory rollback cannot reach. A provider that does not implement them reports `written = 0` and
`blob_version = 0` from `save_restart_blob()`, which the caller treats as "nothing to capture", not
as an error, exactly as the C++ `Component::save_restart_blob()` default already does.

**The struct catalogue is shared.** `fuse_field_view` (what `bind()` hands you — a pointer, element
type, and layout, identical to what `register_field` takes on the participant side) and the window
carried by `fuse_step_request`/`fuse_step_report` (what `advance()` reads and must echo back
exactly) are the SAME structs, grown — not duplicated — from `entries.def`. A field or a time bound
looks like the same bytes on either side of the coupler.

FUSE calls your table in this order, once per instance: `create()`, then `describe()` (query the
field count with `out->buffer = NULL`, then fill it), then `bind()` once per field `describe()`
reported, then `advance()` repeatedly (and `save_state()`/`restore_state()` if your component is
inside an iteration region), then `destroy()`.

**`bind()`'s own call order, for a plugin with more than one field, follows the GRAPH's declared
field order for your component (the order its `add_field()` calls were made in), not the order
your own `describe()` rows happen to list them in.** `describe()`'s rows are matched against the
graph by NAME, never by position, so they may list fields in any order you like; `bind()` is
called once per field in the order the GRAPH declares them, and `fuse_field_view` carries no field
name for you to tell them apart at that point. A plugin with only one field (`test/abi/plugins/
linear_provider.c`) never has to care; a plugin with more than one does
(`test/abi/plugins/affine_provider.c`'s own header comment measures this against
`src/abi/provider_adapter.cpp`'s `register_fields()`).

## Writing a plugin in C

```c
#include <fuse/abi/provider.h>
#include <stdlib.h>

typedef struct { double* out_ptr; } instance_t;

static fuse_status p_create(fuse_provider_ctx* self, const fuse_null* in, fuse_provider_instance* out) {
  instance_t* inst = calloc(1, sizeof(instance_t));
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = (fuse_provider_ctx*)inst;
  return FUSE_OK;
}
static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  free((instance_t*)self);
  return FUSE_OK;
}
static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  out->struct_size = sizeof(fuse_field_table);
  out->written = 1;
  if (out->buffer && out->capacity >= 1) {
    out->buffer[0].struct_size = sizeof(fuse_field_spec);
    out->buffer[0].name = "out";
    out->buffer[0].components = 1;
    out->buffer[0].intent = 0;              /* Provides */
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
  }
  return FUSE_OK;
}
static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  ((instance_t*)self)->out_ptr = (double*)in->data;   /* keep the pointer; write through it later */
  return FUSE_OK;
}
static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  if (inst->out_ptr) *inst->out_ptr = /* your physics, over [in->t_begin, in->t_end) */ 0.0;
  out->struct_size = sizeof(fuse_step_report);
  out->step = in->step; out->iteration = in->iteration;
  out->t_begin = in->t_begin; out->t_end = in->t_end;  /* echo the window EXACTLY (rule 6) */
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1), FUSE_ABI_VERSION,
    NULL /* version */, p_create, p_destroy, p_describe, p_bind, p_advance,
    NULL /* save_state */, NULL /* restore_state */, NULL /* capabilities */,
};

const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  return abi_version == FUSE_ABI_VERSION ? &g_table : NULL;
}
```

A full, tested version (with `save_state`/`restore_state`/`capabilities` filled in and every
`struct_size`/null check FUSE's own adapter actually exercises) is
`test/abi/plugins/linear_provider.c`.

## Writing a plugin in Fortran

Generated module `ffuse_provider` (from the same `entries.def` as `ffuse` — see [](fortran.md))
gives you a `fuse_provider_abi_v1` derived type with one `c_funptr` member per entry, an abstract
interface per entry so your own procedures are checked against the right signature, and
`ffuse_provider_set_<entry>()` to assign each one via `c_funloc`:

```fortran
module my_provider
  use, intrinsic :: iso_c_binding
  use ffuse, only: fuse_null, fuse_field_view, fuse_field_table, fuse_provider_instance, &
                   fuse_step_request, fuse_step_report, FUSE_OK, FUSE_ELEMENT_F64
  use ffuse_provider
  implicit none
  type(fuse_provider_abi_v1), save, target :: g_table
  logical, save :: g_ready = .false.
contains
  function p_advance(self, in, out) result(status) bind(C)
    type(c_ptr), value :: self
    type(fuse_step_request), intent(in) :: in
    type(fuse_step_report), intent(inout) :: out
    integer(c_int32_t) :: status
    ! ... your physics, writing through the pointer p_bind() stashed ...
    out%t_begin = in%t_begin; out%t_end = in%t_end   ! echo the window EXACTLY (rule 6)
    status = FUSE_OK
  end function p_advance
  ! p_create, p_destroy, p_describe, p_bind: same pattern, bind(C) procedures.

  function my_provider_bootstrap(abi_version) result(ptr)
    integer(c_int32_t), value :: abi_version
    type(c_ptr) :: ptr
    if (.not. g_ready) then
      g_table%struct_size = int(c_sizeof(g_table), c_int32_t)
      g_table%abi_version = FUSE_ABI_VERSION
      call ffuse_provider_set_advance(g_table, p_advance)
      ! ... one ffuse_provider_set_<entry> call per entry you implement ...
      g_ready = .true.
    end if
    ptr = ffuse_provider_bootstrap(g_table, abi_version)
  end function my_provider_bootstrap
end module my_provider

function fuse_provider_abi(abi_version) result(ptr) bind(C, name="fuse_provider_abi")
  use, intrinsic :: iso_c_binding, only: c_ptr, c_int32_t
  use my_provider, only: my_provider_bootstrap
  implicit none
  integer(c_int32_t), value :: abi_version
  type(c_ptr) :: ptr
  ptr = my_provider_bootstrap(abi_version)
end function fuse_provider_abi
```

Build against `fuse::abi-fortran` (CMake target `fuse-abi-fortran`, which bundles `ffuse` and
`ffuse_provider`) and `fuse::abi` (for `fuse_coupler_abi`, the shim's own undefined symbol — never
called by a provider-only plugin, but still resolved at link time). A full, tested version is
`test/abi/plugins/linear_provider.f90`.

## Naming your plugin in the graph

```yaml
components:
  - name: ideal-gas
    ranks: {first: 3, count: 1}
    launch:
      mode: plugin
      plugin_path: /opt/fuse-plugins/ideal-gas.so
    fields:
      - name: pressure
        region: cell
        components: 1
        intent: provides
```

A `plugin`-launched component shares the host process and its communicator exactly like an
`InProcess` one (it is never a separate executable, and `fuse::transport::connect()` has nothing to
do for it); the only thing that differs is where its code comes from. `fuse_initialize()` mounts it
automatically — nothing else in your calling code needs to know the component is a plugin rather
than a component you wrote in C++. An empty or missing `plugin_path` is refused at validation
(`E-LAUNCH-PLUGIN-PATH`), and a `dlopen()`/`dlsym()` failure, a version mismatch, or a `describe()`
that disagrees with the graph's own field declaration are refused at `fuse_initialize()` (codes
`E-ABI-LOAD`, `E-ABI-VERSION`, `E-ABI-DESCRIBE` respectively) — never silently skipped or
substituted.

## Error codes

| Code | Meaning |
|---|---|
| `E-ABI-LOAD` | `dlopen()`/`dlsym()` failed; the message carries `dlerror()`'s own text |
| `E-ABI-VERSION` | the plugin's table does not speak `FUSE_ABI_VERSION` (absent, too small, or a different `abi_version`), or `create()`/`advance()` wrote a `struct_size` this host does not recognise |
| `E-ABI-DESCRIBE` | `describe()` disagrees with the graph's own field declaration for a named field — including a field `describe()` reports that the graph does not declare for this component at all (a superset, refused the same as a mismatch or an omission) |
| `E-ABI-PROVIDER-CREATE` / `-BIND` / `-ADVANCE` / `-SAVE` / `-RESTORE` | the named entry returned a non-`FUSE_OK` status |
| `E-ABI-PROVIDER-CAPABILITIES` | the optional `capabilities()` entry exists but returned a non-`FUSE_OK` status (absence, not failure, is what "no checkpoint" means) |
| `E-ABI-PROVIDER-WINDOW` | `advance()` echoed a different window than the one it was asked to run (rule 6) |
| `E-ABI-PROVIDER-ELEMENT` | a field is not `FUSE_ELEMENT_F64`, the only element type this host supports through this ABI |
| `E-ABI-PROVIDER-TABLE` | the provider's table is absent, or missing one of `create`/`destroy`/`describe`/`bind`/`advance` |
| `E-ABI-PROVIDER-DESCRIBE` | the `describe()` call itself (the size query or the fill) returned a non-`FUSE_OK` status — distinct from `E-ABI-DESCRIBE`, which is the cross-check against the graph's own declaration |
| `E-ABI-PROVIDER-GRAPH` | the provider's own component name is not a component of the loaded graph |
| `E-ABI-PROVIDER-EXCEPTION` | `register_fields()` or `advance()` caught a `std::exception` the plugin or FUSE's own backend/graph calls threw |
| `E-ABI-REGISTRY-NULL` | `mount_component()`/`mount_table()` was given a null component or table |
| `E-ABI-REGISTRY-DUPLICATE` | a second provider was mounted under a name already in use |
| `E-ABI-REGISTRY-INSTALL` | `ComponentSet::add()` threw while installing a mounted provider |
| `E-ABI-REGISTRY-NOTFOUND` | `extract()` was asked for a name no provider is mounted under |
| `E-LAUNCH-PLUGIN-PATH` | a `plugin`-launched component declares an empty `plugin_path` |

## Open questions

**Device fields (`FUSE_MEMSPACE_DEVICE`).** Not yet exercised by any test, and not yet a design
decision: `register_field`'s own `FUSE_MEMSPACE_DEVICE` refusal (`E-ABI-REGISTER-MEMSPACE`) has no
provider-direction counterpart, `ProviderAdapter` only ever allocates host storage, and `describe()`'s
own `fuse_field_spec` has no `memspace` field for a plugin to report one through even if it wanted to
(KNOWN_ISSUES.md D15, item 1).

**Checkpoint/restore through an iteration region mixing native and plugin components** is now
tested: `ConformanceTest.PluginBackedComponentInAnIterationRegionConvergesLikeAllNative`
(`test/abi/conformance_test.cpp`) puts a plugin-backed component (`test/abi/plugins/
affine_provider.c`) inside an `IterationRegionDecl` alongside a native one and confirms the sweep
converges to bit-identical values, in the same number of sweeps, as an all-native reference -- and
that the runtime's own byte-level rollback verification (`RunStats::checkpoint_verifications`) ran
against the plugin's own registered storage, not merely that `fuse_run()` returned `FUSE_OK`.

**A plugin launched on a *subset* of a component's ranks** is tested (KNOWN_ISSUES.md D15, item 3,
closed 2026-10-19 across rounds 3-5 and confirmed on an MPI host); a SEPARATE question is still
open, not about test coverage but about design: `fuse::abi_detail::Registry::mount_dynamic()` loads
a `LaunchMode::Plugin` component's shared object on EVERY rank of the process set, whether or not
that rank hosts the component, and whether mounting it only on hosting ranks is worth the added
coupling to the graph's own placement model remains undecided.

## A note on exported symbols

A plugin written in C exports exactly one symbol, `fuse_provider_abi`, and the conformance
suite checks that with `nm`. A plugin written in Fortran also exports the module-procedure
symbols gfortran generates for `ffuse_provider` and for the plugin's own module
(`__ffuse_provider_MOD_*`, `__<module>_MOD_*`); these are the compiler's mangled names, not part
of the ABI, and a host never looks them up. Hiding them needs a linker version script or
`-fvisibility=hidden` on the Fortran objects, which this release does not apply.
