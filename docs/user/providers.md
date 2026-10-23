# Writing a provider plugin (C and Fortran)

This page is for people who want to supply a component to a FUSE coupling as a shared library:
FUSE loads it at runtime from a path named in the graph and calls it once per step. The component can
be written in C, Fortran, or anything that can export one C symbol. This is the **provider ABI**, the
opposite direction from [](c-abi.md), where your code drives FUSE.

## The smallest plugin, in C

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
  out->t_begin = in->t_begin; out->t_end = in->t_end;  /* echo the window EXACTLY */
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

A fuller version, with `save_state`/`restore_state`/`capabilities` filled in and the `struct_size`
and null checks FUSE exercises, is `test/abi/plugins/linear_provider.c`.

## The one symbol your plugin exports

```c
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version);
```

Exactly this, from `#include <fuse/abi/provider.h>`. FUSE `dlopen()`s your shared object and
`dlsym()`s this one name. Return `NULL` for any `abi_version` you do not speak; FUSE refuses the
mount with `E-ABI-VERSION` rather than guessing at a subset. The design rationale, including why this
is a second table rather than a flag on the participant one, is in
[](../dev/abi-design.md#provider-direction).

## The table

The table has this layout. A plugin must provide `create`, `destroy`, `describe`, `bind` and
`advance`; the others may be `NULL`.

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

The last two entries, `save_restart_blob`/`load_restart_blob`, are the provider-side hooks for
restart-to-disk ([](restart.md#restart-through-the-c-and-fortran-abi)): an opaque, versioned byte
record for whatever `save_state()`'s in-memory rollback cannot reach. A provider that does not
implement them reports `written = 0` and `blob_version = 0` from `save_restart_blob()`, which the
caller treats as "nothing to capture", not as an error.

**The structs are shared with the participant ABI.** `fuse_field_view` (what `bind()` hands you: a
pointer, element type and layout, identical to what `register_field` takes on the participant side)
and the window carried by `fuse_step_request`/`fuse_step_report` (what `advance()` reads and must
echo back exactly) are the same structs on both sides. Time is exact: `t_begin`/`t_end` are
`fuse_exact_time` rationals, and `advance()` must copy them into its report unchanged
(`E-ABI-PROVIDER-WINDOW` otherwise).

FUSE calls your table in this order, once per instance: `create()`, then `describe()` (query the
field count with `out->buffer = NULL`, then fill it), then `bind()` once per field `describe()`
reported, then `advance()` repeatedly (and `save_state()`/`restore_state()` if your component is
inside an iteration region), then `destroy()`.

:::{important}
For a plugin with more than one field, `bind()` is called in the order the **graph** declares the
fields for your component, not in the order your `describe()` rows list them. `describe()` rows are
matched to the graph by name, so they may come in any order, but `fuse_field_view` carries no field
name, so inside `bind()` you can tell fields apart only by call order. A plugin with one field
(`test/abi/plugins/linear_provider.c`) never has to care; `test/abi/plugins/affine_provider.c` is a
two-field example.
:::

## Writing a plugin in Fortran

The generated module `ffuse_provider` (from the same `entries.def` as `ffuse`; see [](fortran.md))
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
    out%t_begin = in%t_begin; out%t_end = in%t_end   ! echo the window EXACTLY 
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
`ffuse_provider`) and `fuse::abi` (for `fuse_coupler_abi`, which the Fortran shim references; a provider-only plugin never
calls it, but it must resolve at link time). A fuller version is `test/abi/plugins/linear_provider.f90`.
See [](build.md) for enabling the Fortran bindings.

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
in-process one. It is never a separate executable, and `fuse::transport::connect()` has nothing to do
for it; only where its code comes from differs. `fuse_initialize()` mounts it automatically, and
nothing else in your calling code needs to know the component is a plugin.

An empty or missing `plugin_path` is refused at validation (`E-LAUNCH-PLUGIN-PATH`). A
`dlopen()`/`dlsym()` failure, a version mismatch, or a `describe()` that disagrees with the graph's
field declaration is refused at `fuse_initialize()` (`E-ABI-LOAD`, `E-ABI-VERSION`,
`E-ABI-DESCRIBE`), never silently skipped or substituted.

### Plugins on multi-rank components

Only ranks that are members of the component's `ranks` load the shared object and call into it. A
non-hosting rank of a multi-rank base communicator never touches it, but still registers every field
the component declares, as an empty view, so field ordering and the runtime's rank agreement do not
depend on which ranks host the plugin.

A refusal from the plugin (`E-ABI-LOAD`, `E-ABI-VERSION`, `E-ABI-DESCRIBE`, or a device gate) is
seen first by a hosting rank alone, but `fuse_initialize()` agrees it across every rank of the base
communicator before any rank returns, using one `MPI_Allreduce(MPI_MINLOC)` per
`fuse_initialize()` call, and only when the graph declares a plugin component. Every rank, hosting or
not, then returns the same code, and a non-originating rank's message names the rank that refused
(`E-ABI-PLUGIN-AGREEMENT-REFUSED`). See [A known failure case](#a-known-failure-case) for the one
launch shape this agreement cannot be formed for.

## Device fields

`fuse_field_spec`'s `memspace` member (`FUSE_MEMSPACE_HOST` or `FUSE_MEMSPACE_DEVICE`, the same enum
`register_field` uses on the participant side) lets a field's `describe()` row ask for
device-resident storage. It defaults to `FUSE_MEMSPACE_HOST` for a plugin compiled against an older
header.

Asking for `FUSE_MEMSPACE_DEVICE` needs two things, checked in this order:

1. Your plugin's `capabilities()` entry must report `FUSE_CAP_DEVICE_FIELDS` in its `flags`. A field
   that asks for a device space your `capabilities()` did not declare is refused with
   `E-ABI-PROVIDER-DEVICE-UNDECLARED`. Setting the bit without any field asking for a device space is
   fine.
2. This FUSE build must have been compiled with a real device backend (Kokkos configured with CUDA,
   HIP or SYCL). A correctly declared device field on a build with none is refused with
   `E-ABI-PROVIDER-NO-DEVICE` rather than silently handed host memory under a device label.

An accepted device field is allocated and owned by FUSE, the same as a host field: your plugin's own
`bind()` entry still receives a `fuse_field_view` whose `data` pointer it writes through, except that
pointer now addresses `Kokkos::DefaultExecutionSpace::memory_space` rather than host memory, and
`view->memspace` reports which. A plugin that writes a device field must do so through a kernel launch, not a host store, and must
complete that device work (fence its own execution space) before `advance()` returns. FUSE also
fences the default execution space after every successful `advance()` of a provider that has a
device field, but no behaviour is specified in terms of that fence or of a device pointer being
visible across streams.

## Error codes

| Code | Meaning |
|---|---|
| `E-ABI-LOAD` | `dlopen()`/`dlsym()` failed; the message carries `dlerror()`'s own text |
| `E-ABI-VERSION` | the plugin's table does not speak `FUSE_ABI_VERSION` (absent, too small, or a different `abi_version`), or `create()`/`advance()` wrote a `struct_size` this host does not recognise |
| `E-ABI-DESCRIBE` | `describe()` disagrees with the graph's own field declaration for a named field — including a field `describe()` reports that the graph does not declare for this component at all (a superset, refused the same as a mismatch or an omission) |
| `E-ABI-PROVIDER-CREATE` / `-BIND` / `-ADVANCE` / `-SAVE` / `-RESTORE` | the named entry returned a non-`FUSE_OK` status |
| `E-ABI-PROVIDER-CAPABILITIES` | the optional `capabilities()` entry exists but returned a non-`FUSE_OK` status (absence, not failure, is what "no checkpoint" means) |
| `E-ABI-PROVIDER-WINDOW` | `advance()` echoed a different window than the one it was asked to run  |
| `E-ABI-PROVIDER-ELEMENT` | a field is not `FUSE_ELEMENT_F64`, the only element type this host supports through this ABI |
| `E-ABI-PROVIDER-MEMSPACE` | a field's `memspace` is neither `FUSE_MEMSPACE_HOST` nor `FUSE_MEMSPACE_DEVICE` |
| `E-ABI-PROVIDER-DEVICE-UNDECLARED` | a field's `describe()` row asks for `FUSE_MEMSPACE_DEVICE` but this provider's own `capabilities()` did not report `FUSE_CAP_DEVICE_FIELDS` |
| `E-ABI-PROVIDER-NO-DEVICE` | a field correctly declared and capability-granted for `FUSE_MEMSPACE_DEVICE`, but this FUSE build has no CUDA/HIP/SYCL backend to honour it |
| `E-ABI-PROVIDER-TABLE` | the provider's table is absent, or missing one of `create`/`destroy`/`describe`/`bind`/`advance` |
| `E-ABI-PROVIDER-DESCRIBE` | the `describe()` call itself (the size query or the fill) returned a non-`FUSE_OK` status — distinct from `E-ABI-DESCRIBE`, which is the cross-check against the graph's own declaration |
| `E-ABI-PROVIDER-GRAPH` | the provider's own component name is not a component of the loaded graph |
| `E-ABI-PROVIDER-EXCEPTION` | `register_fields()` or `advance()` caught a `std::exception` the plugin or FUSE's own backend/graph calls threw |
| `E-ABI-REGISTRY-NULL` | a null component or table was mounted |
| `E-ABI-REGISTRY-DUPLICATE` | a second provider was mounted under a name already in use |
| `E-ABI-REGISTRY-INSTALL` | `ComponentSet::add()` threw while installing a mounted provider |
| `E-ABI-REGISTRY-NOTFOUND` | `extract()` was asked for a name no provider is mounted under |
| `E-LAUNCH-PLUGIN-PATH` | a `plugin`-launched component declares an empty `plugin_path` |
| `E-ABI-PLUGIN-AGREEMENT-REFUSED` | a DIFFERENT rank's plugin-mount refusal was agreed across the base communicator (the message names which rank and which status; see [Plugins on multi-rank components](#plugins-on-multi-rank-components)) |
| `E-ABI-PLUGIN-NO-SPANNING-COMM` | a `Plugin` component and a `Connect`-launched one share a graph, and this process's own `my_component` does not itself span them -- see [A known failure case](#a-known-failure-case) |

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
`-fvisibility=hidden` on the Fortran objects; FUSE does not apply either to plugins you build.

## Limitations

- Provider fields must be `FUSE_ELEMENT_F64` (`E-ABI-PROVIDER-ELEMENT`) and `FUSE_MEMSPACE_HOST` or
  `FUSE_MEMSPACE_DEVICE` (`E-ABI-PROVIDER-MEMSPACE`).
- `FUSE_MEMSPACE_DEVICE` needs a build with a CUDA, HIP or SYCL backend.
- A graph that declares a plugin component together with a `Connect`-launched component is refused in
  some process configurations; see [A known failure case](#a-known-failure-case).
- A plugin inside an iteration region needs real `save_state()`/`restore_state()` entries and
  `FUSE_CAP_CHECKPOINT` in its `capabilities()`.


## See also

- [](c-abi.md): the opposite direction, your code driving FUSE.
- [](fortran.md): the `ffuse` module used by Fortran plugins.
- [](restart.md): restart blobs for providers.
- [](build.md): building the ABI libraries, including the Fortran bindings.
- [](multiple-executables.md): launch modes other than `plugin`.
- [](troubleshooting.md): diagnosing `E-...` codes.
