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

Only a rank that is a member of the component's own `ranks` actually loads the shared object
(`dlopen()`/`dlsym()`/`fuse_provider_abi()`) and calls into it at all; a non-hosting rank of a
multi-rank base communicator never touches it (KNOWN_ISSUES.md D15, "Registry hosting" track), and
still registers every field that component declares, as an empty view, so field ordering and the
runtime's own rank agreement are unaffected by which ranks actually host the plugin. A refusal from
the plugin itself (`E-ABI-LOAD`, `E-ABI-VERSION`, `E-ABI-DESCRIBE`, or a device gate) is seen first
by a hosting rank alone, but `fuse_initialize()` agrees it across every rank of the base
communicator before any rank's own call returns (KNOWN_ISSUES.md D15, "Registry agreement" track):
one `MPI_Allreduce(MPI_MINLOC)`, run exactly once per `fuse_initialize()` call and only when the
graph declares a Plugin component, so every rank -- hosting the plugin or not -- returns the SAME
code, with a non-originating rank's own message rewritten to name the rank that actually refused.
See "A known failure case" below for the one launch shape this agreement cannot be formed for, and
which is refused outright, before any plugin is touched, instead.

## Device fields

`fuse_field_spec`'s `memspace` entry (`FUSE_MEMSPACE_HOST` or `FUSE_MEMSPACE_DEVICE`, the same enum
`fuse_register_field` already uses on the participant direction) lets a field's `describe()` row ask
for device-resident storage instead of the host storage every field got before this entry existed.
Defaulted to `FUSE_MEMSPACE_HOST` (value 0) for a plugin compiled before this entry was added, so an
old plugin's behaviour does not change.

Asking for `FUSE_MEMSPACE_DEVICE` needs two things to both be true, checked in this order:

1. Your plugin's own `capabilities()` entry must report `FUSE_CAP_DEVICE_FIELDS` in its `flags`. A
   field that asks for a device space your own `capabilities()` did not declare is refused by name
   (`E-ABI-PROVIDER-DEVICE-UNDECLARED`) before FUSE ever looks at whether a device backend exists.
   The opposite -- `capabilities()` sets the bit but no field ever asks for `FUSE_MEMSPACE_DEVICE` --
   is unremarkable: a provider is always free to offer more than a graph uses.
2. This FUSE build must have been compiled with a real device backend (Kokkos configured with CUDA,
   HIP, or SYCL). A correctly-declared device field on a build with none of those is refused by name
   (`E-ABI-PROVIDER-NO-DEVICE`) rather than silently handed host-addressable memory under the device
   label -- `Kokkos::DefaultExecutionSpace::memory_space` legally equals `Kokkos::HostSpace` on such a
   build, and nothing would crash if the library used it anyway, but that is not what the field asked
   for.

An accepted device field is allocated and owned by FUSE, the same as a host field: your plugin's own
`bind()` entry still receives a `fuse_field_view` whose `data` pointer it writes through, except that
pointer now addresses `Kokkos::DefaultExecutionSpace::memory_space` rather than host memory, and
`view->memspace` reports which. After a successful `advance()` on a provider with at least one
accepted device field, FUSE fences the default execution space before returning -- unconditionally,
never left to the plugin's own discretion, and at zero cost to a provider with no device fields.

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
| `E-ABI-PROVIDER-MEMSPACE` | a field's `memspace` is neither `FUSE_MEMSPACE_HOST` nor `FUSE_MEMSPACE_DEVICE` |
| `E-ABI-PROVIDER-DEVICE-UNDECLARED` | a field's `describe()` row asks for `FUSE_MEMSPACE_DEVICE` but this provider's own `capabilities()` did not report `FUSE_CAP_DEVICE_FIELDS` |
| `E-ABI-PROVIDER-NO-DEVICE` | a field correctly declared and capability-granted for `FUSE_MEMSPACE_DEVICE`, but this FUSE build has no CUDA/HIP/SYCL backend to honour it |
| `E-ABI-PROVIDER-TABLE` | the provider's table is absent, or missing one of `create`/`destroy`/`describe`/`bind`/`advance` |
| `E-ABI-PROVIDER-DESCRIBE` | the `describe()` call itself (the size query or the fill) returned a non-`FUSE_OK` status — distinct from `E-ABI-DESCRIBE`, which is the cross-check against the graph's own declaration |
| `E-ABI-PROVIDER-GRAPH` | the provider's own component name is not a component of the loaded graph |
| `E-ABI-PROVIDER-EXCEPTION` | `register_fields()` or `advance()` caught a `std::exception` the plugin or FUSE's own backend/graph calls threw |
| `E-ABI-REGISTRY-NULL` | `mount_component()`/`mount_table()` was given a null component or table |
| `E-ABI-REGISTRY-DUPLICATE` | a second provider was mounted under a name already in use |
| `E-ABI-REGISTRY-INSTALL` | `ComponentSet::add()` threw while installing a mounted provider |
| `E-ABI-REGISTRY-NOTFOUND` | `extract()` was asked for a name no provider is mounted under |
| `E-LAUNCH-PLUGIN-PATH` | a `plugin`-launched component declares an empty `plugin_path` |
| `E-ABI-PLUGIN-AGREEMENT-REFUSED` | a DIFFERENT rank's plugin-mount refusal was agreed across the base communicator (the message names which rank and which status; see "Naming your plugin in the graph") |
| `E-ABI-PLUGIN-NO-SPANNING-COMM` | a `Plugin` component and a `Connect`-launched one share a graph, and this process's own `my_component` does not itself span them -- see "A known failure case" |

## Open questions

**Device fields (`FUSE_MEMSPACE_DEVICE`)** are built and tested at tier 0 (KNOWN_ISSUES.md D15,
item 1; see the "Device fields" section above for the mechanism). The positive path, a host field
and a device field on one provider, ran on a real CUDA backend on 2026-10-21 (ssh:higgs, GTX 1050 Ti,
Kokkos CUDA): `ConformanceTest.MixedHostAndDeviceFieldsOnOneProviderConvergeThroughChannels` passes
there and skips by name on a build with no CUDA/HIP/SYCL backend, and `abi.device_layout` checks the
array-of-structures layout element by element and both copy directions bit for bit. A plugin that writes a
device field must do it through a kernel launch, not a host store, and must complete that device work
(fence its own execution space) before `advance()` returns. The adapter also fences after every
`advance()` of a provider that has a device field, as a defensive measure: this library never depends on
that fence, or on a device pointer being visible across streams, and no behaviour is specified in terms
of either.
`register_field`'s own `FUSE_MEMSPACE_DEVICE` refusal
(`E-ABI-REGISTER-MEMSPACE`) is the PARTICIPANT direction's separate, still-open gap (a C or Fortran
caller invoking `fuse_register_field()` itself has no way to ask for a device field at all); this
item was the PROVIDER direction only.

**Checkpoint/restore through an iteration region mixing native and plugin components** is now
tested: `ConformanceTest.PluginBackedComponentInAnIterationRegionConvergesLikeAllNative`
(`test/abi/conformance_test.cpp`) puts a plugin-backed component (`test/abi/plugins/
affine_provider.c`) inside an `IterationRegionDecl` alongside a native one and confirms the sweep
converges to bit-identical values, in the same number of sweeps, as an all-native reference -- and
that the runtime's own byte-level rollback verification (`RunStats::checkpoint_verifications`) ran
against the plugin's own registered storage, not merely that `fuse_run()` returned `FUSE_OK`.

**A plugin launched on a *subset* of a component's ranks** is tested (KNOWN_ISSUES.md D15, item 3,
closed 2026-10-19 across rounds 3-5 and confirmed on an MPI host). The SEPARATE design question this
raised -- whether `fuse::abi_detail::Registry::mount_dynamic()` should load a `LaunchMode::Plugin`
component's shared object on every rank regardless of placement, or only on hosting ranks -- is
CLOSED (owner decision 2026-10-22, verbatim: "NO. This should not happen."; "Registry hosting"
track): only a hosting rank loads it now (see "Naming your plugin in the graph" above). The
collective-discipline cost this introduced (a plugin-side refusal visible only to a hosting rank)
is ALSO CLOSED (owner decision 2026-10-22, option (b); "Registry agreement" track): see "Naming
your plugin in the graph" above for the one-collective agreement this adds, and "A known failure
case" immediately below for the one launch shape it cannot cover.

## A known failure case

A graph that declares a `Plugin` component AND a `Connect`-launched one (true separate-executable
MPMD -- independent MPI universes, no common parent communicator until `fuse::transport::connect()`
itself performs its own `MPI_Open_port`/`MPI_Comm_accept`/`MPI_Comm_connect` handshake) is refused
at `fuse_initialize()`, before the plugin is ever touched, whenever the CALLING process's own
`my_component` is not itself the `Connect` half of that handshake (unset, or naming the `Plugin`
component, or any other non-`Mpmd`/non-`Connect` component): `E-ABI-PLUGIN-NO-SPANNING-COMM`. The
reason is the same one that makes the one-collective agreement above possible at all: agreeing a
plugin-mount verdict needs a communicator that spans every rank with an opinion on it, and a process
whose own `my_component` never calls `connect()` never gains one -- its own `communicator` is
exactly what `fuse_setup_options` was handed, unchanged. `Mpmd` does not have this problem
(`MPI_COMM_WORLD` already spans every program before `fuse_setup()` is ever called, launcher-style),
so a `Plugin` component alongside an `Mpmd` one is never refused this way, regardless of
`my_component`.

Workaround: host the plugin component in a process that is itself `Connect`-launched (so its own
`connect()` call forms the spanning communicator before `fuse_initialize()` needs one), or declare
the multi-executable component `Mpmd` instead of `Connect` if your launcher supports it (a single
`mpiexec ... : ...` invocation, not two separately launched programs).

## A note on exported symbols

A plugin written in C exports exactly one symbol, `fuse_provider_abi`, and the conformance
suite checks that with `nm`. A plugin written in Fortran also exports the module-procedure
symbols gfortran generates for `ffuse_provider` and for the plugin's own module
(`__ffuse_provider_MOD_*`, `__<module>_MOD_*`); these are the compiler's mangled names, not part
of the ABI, and a host never looks them up. Hiding them needs a linker version script or
`-fvisibility=hidden` on the Fortran objects, which this release does not apply.
