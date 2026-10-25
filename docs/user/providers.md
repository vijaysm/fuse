# Writing a provider plugin (C and Fortran)

This page is for people who want to supply a component to a FUSE coupling as a shared library:
FUSE loads it at run time from a path named in the graph and calls it once per step. The component
can be written in C, Fortran, or anything that can export one C symbol. This is the **provider
ABI**, the opposite direction from [](c-abi.md), where your code drives FUSE. The page also shows
the program on the other side: a driver that loads a graph of plugins and runs it.

## The smallest plugin, in C

```c
#include <fuse/abi/provider.h>
#include <stdlib.h>

typedef struct { double* out_ptr; uint64_t count; } instance_t;

static fuse_status p_create(fuse_provider_ctx* self, const fuse_null* in, fuse_provider_instance* out) {
  (void)self; (void)in;
  instance_t* inst = calloc(1, sizeof(instance_t));
  out->struct_size = sizeof(fuse_provider_instance);
  out->self = (fuse_provider_ctx*)inst;
  return FUSE_OK;
}
static fuse_status p_destroy(fuse_provider_ctx* self, const fuse_null* in, fuse_null* out) {
  (void)in; (void)out;
  free((instance_t*)self);
  return FUSE_OK;
}
static fuse_status p_describe(fuse_provider_ctx* self, const fuse_null* in, fuse_field_table* out) {
  (void)self; (void)in;
  out->struct_size = sizeof(fuse_field_table);
  out->written = 1;                          /* how many rows this plugin has */
  if (out->buffer && out->capacity >= 1) {   /* NULL buffer: the size query */
    out->buffer[0].struct_size  = sizeof(fuse_field_spec);
    out->buffer[0].name         = "out";
    out->buffer[0].components   = 1;
    out->buffer[0].intent       = 0;         /* Provides */
    out->buffer[0].element_type = FUSE_ELEMENT_F64;
    out->buffer[0].memspace     = FUSE_MEMSPACE_HOST;
  }
  return FUSE_OK;
}
static fuse_status p_bind(fuse_provider_ctx* self, const fuse_field_view* in, fuse_null* out) {
  (void)out;
  instance_t* inst = (instance_t*)self;
  inst->out_ptr = (double*)in->data;         /* FUSE owns this storage; keep the pointer */
  inst->count   = in->count;
  return FUSE_OK;
}
static fuse_status p_advance(fuse_provider_ctx* self, const fuse_step_request* in, fuse_step_report* out) {
  instance_t* inst = (instance_t*)self;
  double t_end = (double)in->t_end.numerator / (double)in->t_end.denominator;
  for (uint64_t i = 0; i < inst->count; ++i) inst->out_ptr[i] = t_end;  /* your physics */
  out->struct_size = sizeof(fuse_step_report);
  out->step = in->step; out->iteration = in->iteration;
  out->t_begin = in->t_begin; out->t_end = in->t_end;  /* echo the window EXACTLY */
  return FUSE_OK;
}

static const fuse_provider_abi_v1 g_table = {
    sizeof(fuse_provider_abi_v1), FUSE_ABI_VERSION,
    NULL /* version */, p_create, p_destroy, p_describe, p_bind, p_advance,
    NULL /* save_state */, NULL /* restore_state */, NULL /* capabilities */,
    NULL /* save_restart_blob */, NULL /* load_restart_blob */,
};

const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version) {
  return abi_version == FUSE_ABI_VERSION ? &g_table : NULL;
}
```

Build it as a shared library against `fuse/abi/provider.h` alone (a plugin calls nothing in FUSE,
so it links no FUSE library):

```sh
cc -shared -fPIC -fvisibility=hidden -I<prefix>/include my_provider.c -o libmy_provider.so
```

A complete plugin with two fields, real `save_state`/`restore_state` and `capabilities` is
[shown below](#a-complete-plugin-in-c).

## The one symbol your plugin exports

```c
const fuse_provider_abi_v1* fuse_provider_abi(uint32_t abi_version);
```

Exactly this, from `#include <fuse/abi/provider.h>`. FUSE `dlopen()`s your shared object and
`dlsym()`s this one name. Return `NULL` for any `abi_version` you do not speak; FUSE refuses the
mount with `E-ABI-VERSION` rather than guessing at a subset. Several plugins, each its own shared
object, may all export this same name: FUSE resolves it through each object's own `dlopen` handle
(`RTLD_NOW | RTLD_LOCAL`), never through the global symbol table.

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

Every member has the shape `fuse_status f(fuse_provider_ctx* self, const IN* in, OUT* out)`, the
same rule as the participant table, with `self` (your own opaque instance, `create()`'s output)
taking the `ctx` slot. `self` is a different C type from the participant direction's `fuse_ctx*`,
so the two tables' handles cannot be passed to each other by accident. `struct_size` and
`abi_version` must be filled in: `sizeof(fuse_provider_abi_v1)` (96 on a 64-bit platform) and
`FUSE_ABI_VERSION`.

| Entry | Required | Called |
|---|---|---|
| `version` | no | never by the runtime; a diagnostic hook |
| `create` | yes | once per instance, with `self == NULL`; returns your instance in `out->self` |
| `destroy` | yes | once, at context teardown |
| `describe` | yes | twice: with `buffer == NULL` to learn `written`, then with a buffer of `capacity` rows |
| `bind` | yes | once per field the graph declares for your component, in graph order, before the first `advance` |
| `advance` | yes | once per step of your component |
| `save_state`, `restore_state` | no | by iteration regions, when `capabilities` reports `FUSE_CAP_CHECKPOINT` |
| `capabilities` | no | once, after `create` |
| `save_restart_blob`, `load_restart_blob` | no | by restart-to-disk; see [Restart blobs](#restart-blobs) |

A table missing one of the required entries is refused with `E-ABI-PROVIDER-TABLE`.

**The structs are shared with the participant ABI.** `fuse_field_view` (what `bind()` hands you:
a pointer, element type and layout, the same struct `register_field` takes on the participant side)
and `fuse_step_request`/`fuse_step_report` (what `advance()` reads and must echo) are the same
structs on both sides.

## The call sequence

FUSE calls your table in this order, once per instance:

1. `create(NULL, NULL, &inst)`.
2. `capabilities`, if present (a non-`FUSE_OK` status is refused with
   `E-ABI-PROVIDER-CAPABILITIES`; absence means "no optional capability"). FUSE always assumes the
   plugin has hidden state.
3. `describe`, as a size query and then a fill. Each row is a `fuse_field_spec`: `name`,
   `components`, `intent` (0 Provides, 1 Requires, 2 both), `element_type`, `memspace`. Rows are
   matched to the graph's field declarations for your component **by name**, and must agree in
   `components` and `intent`. A row the graph does not declare, a declared field without a row, or a
   mismatch is refused with `E-ABI-DESCRIBE`. A `describe` call that itself fails is
   `E-ABI-PROVIDER-DESCRIBE`.
4. `bind`, once per graph field, **in the order the graph declares the fields** for your component.
   FUSE allocates and owns the storage (sized to this rank's share of the region, see
   [Plugins on multi-rank components](#plugins-on-multi-rank-components)) and hands you a
   `fuse_field_view` on it: keep `data` and `count`; write exports and read imports through them at
   every `advance`. On a rank whose share of a field is empty, `bind` is skipped for that field.
5. `advance(req)`, once per step. `req.t_begin`/`req.t_end` are the exact window to integrate,
   `req.step` the step index, `req.iteration` the sweep index inside an iteration region (0
   outside). `req.requires_checkpoint`/`requires_restore` are always 0 here: FUSE calls
   `save_state`/`restore_state` as separate entries instead.
6. `destroy`.

:::{important}
For a plugin with more than one field, `bind()` is called in the order the **graph** declares the
fields for your component, not in the order your `describe()` rows list them. `fuse_field_view`
carries no field name, so inside `bind()` you can tell fields apart only by call order. A plugin with
one field never has to care; the two-field plugin below counts its `bind` calls.
:::

### `advance` must echo the window exactly

`advance()` copies `in->t_begin`, `in->t_end`, `in->step` and `in->iteration` into its report
unchanged. Time is exact (`fuse_exact_time` rationals); a report whose window differs from the
request in any way is refused with `E-ABI-PROVIDER-WINDOW`. A non-`FUSE_OK` status from `advance` is
`E-ABI-PROVIDER-ADVANCE`. Your physics decides nothing about the step size: the graph's schedule
does.

### `save_state` and `restore_state`

A plugin inside an iteration region ([](convergence.md)) is advanced over the same window more than
once. FUSE snapshots and restores the fields it owns itself; `save_state()` must capture whatever
private state `advance()` mutates beyond those fields, and `restore_state()` must put it back. Only
one checkpoint needs to be retained. Declare `FUSE_CAP_CHECKPOINT` in `capabilities()` when these
two entries are real; without the bit a plugin placed inside an iteration region is refused before
the first sweep (`E-ITER-NOROLLBACK`). After every `restore_state` FUSE verifies that the restored
fields match the snapshot byte for byte (`E-ITER-ROLLBACK` otherwise). Failures of the entries
themselves are `E-ABI-PROVIDER-SAVE` and `E-ABI-PROVIDER-RESTORE`.

### What `capabilities` can report

| Bit | Meaning for a provider |
|---|---|
| `FUSE_CAP_CHECKPOINT` | `save_state`/`restore_state` are real, so the component may sit inside an iteration region |
| `FUSE_CAP_RESTART_FORM_B` | `save_restart_blob`/`load_restart_blob` are real |
| `FUSE_CAP_DEVICE_FIELDS` | `describe` rows may ask for `FUSE_MEMSPACE_DEVICE`; see [Device fields](#device-fields) |

Other bits are ignored on the provider side.

### Restart blobs

`save_restart_blob`/`load_restart_blob` are the provider-side hooks for restart-to-disk
([](restart.md#restart-through-the-c-and-fortran-abi)): an opaque, versioned byte record for
whatever `save_state()`'s in-memory rollback does not need but a restart from a file does.
`save_restart_blob` uses the same query-then-fill idiom as `describe`: report `written` (the size)
and `blob_version` on every call, and copy the bytes only when `buffer` is non-NULL and `capacity`
suffices. `load_restart_blob` receives the bytes and the version back and returns
`FUSE_ERR_VERSION` for a version it does not recognise. A provider that does not implement them
(NULL entries, or entries reporting `written = 0`, `blob_version = 0`) is treated as having nothing
to capture, not as an error. Failures are `E-ABI-PROVIDER-RESTART-SAVE` and
`E-ABI-PROVIDER-RESTART-LOAD`.

## Naming your plugin in the graph

```yaml
components:
  - name: ideal-gas
    ranks: {first: 3, count: 1}
    interval: 1
    launch:
      mode: plugin
      plugin_path: /opt/fuse-plugins/libideal-gas.so
    fields:
      - name: pressure
        region: cell
        components: 1
        units: Pa
        intent: Provides
```

Enumerator names in the YAML are case-sensitive: `intent: Provides`, `Requires` or
`Bidirectional`; a lowercase `provides` is refused at load with `E-YAML-PARSE` ("unknown intent").
See [](yaml.md) for the file format. The same declaration in C++ is
`.launch = {.mode = fuse::LaunchMode::Plugin, .plugin_path = "..."}`.

`plugin_path` is handed to `dlopen()` as given, so an absolute path is the predictable choice; a
bare file name is searched the way `dlopen()` searches (`LD_LIBRARY_PATH` and the system
directories). An empty or missing `plugin_path` is refused at validation (`E-LAUNCH-PLUGIN-PATH`).

A `plugin`-launched component shares the host process and its communicator exactly like an
in-process one. It is never a separate executable, `fuse::transport::connect()` has nothing to do
for it, and it needs no `appnum` or `port_file`; only where its code comes from differs.

### How a plugin is loaded

Plugins are mounted by `initialize()` of the **participant ABI** (`fuse_coupler_abi`), and nowhere
else: a program that calls `create`, `load_graph`, `initialize` on a graph with plugin components
gets them `dlopen()`ed on the ranks that host them, with no further call on its part. A `dlopen()`
or `dlsym()` failure is `E-ABI-LOAD` (the message carries `dlerror()`'s text), a `NULL` table is
`E-ABI-VERSION`, and a `describe()` that disagrees with the graph is `E-ABI-DESCRIBE`, all reported
from `initialize()`; nothing is silently skipped or substituted. The shared object stays open for the
context's lifetime and is closed after your `destroy()` has run.

A C++ program that owns a `fuse::Runner` directly, and the Python module, have no public way to
mount a plugin: the loader lives behind the participant ABI. To host plugins from C++, call
`fuse_coupler_abi()` from C++ (the header is C-compatible) as the driver below does.

### Plugins on multi-rank components

Only ranks that are members of the component's `ranks` load the shared object and call into it. A
non-hosting rank of the base communicator never touches it, but still registers every field the
component declares, as an empty view, so field ordering and the runtime's rank agreement do not
depend on which ranks host the plugin.

On the hosting ranks the region is split evenly, in rank order: with `n` hosting ranks and `total`
entities, every rank gets `total / n`, and the first `total % n` ranks one more. `bind()` receives
that rank's `count`; a rank whose share is empty gets no `bind()` call for the field.

A refusal from the plugin (`E-ABI-LOAD`, `E-ABI-VERSION`, `E-ABI-DESCRIBE`, or a device gate) is
seen first by a hosting rank alone, but `initialize()` agrees it across every rank of the base
communicator before any rank returns, with one `MPI_Allreduce(MPI_MINLOC)` per `initialize()` call,
and only when the graph declares a plugin component. Every rank then returns the same code, and a
non-originating rank's message names the rank that refused (`E-ABI-PLUGIN-AGREEMENT-REFUSED`). See
[A known failure case](#a-known-failure-case) for the one launch shape this agreement cannot be
formed for.

## A driver that runs plugins

The program on the other side of a plugin is any program that uses the participant ABI. The
sequence is the one `examples/brusselator/participants/coupler_main.c` runs for a coupler-only rank:
`create`, `load_graph`, optionally `setup`, `initialize`, `run(periods)`, `destroy`, registering no
field. Loaded with a graph whose components are all plugins, the same sequence runs them:

```c
#include <fuse/abi/participant.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void report(const fuse_participant_abi_v1* t, fuse_ctx* ctx, const char* what, fuse_status rc) {
  char buf[1024];
  fuse_error err;
  memset(&err, 0, sizeof(err));
  err.struct_size = sizeof(err);
  err.buffer = buf;
  err.capacity = sizeof(buf);
  buf[0] = '\0';
  t->last_error(ctx, NULL, &err);
  fprintf(stderr, "%s: status=%d %s: %s\n", what, (int)rc, err.code ? err.code : "?", buf);
}

int main(int argc, char** argv) {
  const char* graph_path = argc > 1 ? argv[1] : "plugins.yaml";
  int periods = argc > 2 ? atoi(argv[2]) : 3;
  const fuse_participant_abi_v1* t = fuse_coupler_abi(FUSE_ABI_VERSION);
  if (!t) return 1;

  fuse_context_out co;
  memset(&co, 0, sizeof(co));
  co.struct_size = sizeof(co);
  fuse_status rc = t->create(NULL, NULL, &co);
  if (rc != FUSE_OK) { report(t, NULL, "create", rc); return 1; }
  fuse_ctx* ctx = co.ctx;

  fuse_graph_source src;
  memset(&src, 0, sizeof(src));
  src.struct_size = sizeof(src);
  src.yaml_path = graph_path;
  rc = t->load_graph(ctx, &src, NULL);
  if (rc != FUSE_OK) { report(t, ctx, "load_graph", rc); return 1; }

  rc = t->initialize(ctx, NULL, NULL);          /* mounts every plugin the graph names */
  if (rc != FUSE_OK) { report(t, ctx, "initialize", rc); return 1; }

  fuse_run_request req;
  memset(&req, 0, sizeof(req));
  req.struct_size = sizeof(req);
  req.periods = periods;
  rc = t->run(ctx, &req, NULL);                 /* advances them, period by period */
  if (rc != FUSE_OK) { report(t, ctx, "run", rc); return 1; }

  /* Read a plugin's export back through FUSE's own view of its storage. */
  fuse_field_name fname;
  memset(&fname, 0, sizeof(fname));
  fname.struct_size = sizeof(fname);
  fname.component = "downstream";
  fname.field = "b_out";
  fuse_field_ref ref;
  memset(&ref, 0, sizeof(ref));
  ref.struct_size = sizeof(ref);
  rc = t->field_handle(ctx, &fname, &ref);
  if (rc != FUSE_OK) { report(t, ctx, "field_handle", rc); return 1; }
  fuse_field_view view;
  memset(&view, 0, sizeof(view));
  view.struct_size = sizeof(view);
  rc = t->view_field(ctx, &ref, &view);
  if (rc != FUSE_OK) { report(t, ctx, "view_field", rc); return 1; }
  for (uint64_t i = 0; i < view.count; ++i)
    printf("downstream.b_out[%llu] = %g\n", (unsigned long long)i, ((double*)view.data)[i]);

  rc = t->destroy(ctx, NULL, NULL);
  if (rc != FUSE_OK) { report(t, NULL, "destroy", rc); return 1; }
  return 0;
}
```

With this graph, naming the two test plugins `test/abi/plugins/linear_provider.c` (one field,
`out = t_end`) and `test/abi/plugins/affine_provider.c` (`b_out = 0.5 * b_in`), built as shared
libraries:

```yaml
version: 1
meshes:
  - {name: m1, source: "generated:line:4"}
  - {name: m2, source: "generated:line:4"}
regions:
  - {name: r1, mesh: m1, dimension: 1}
  - {name: r2, mesh: m2, dimension: 1}
components:
  - name: upstream
    ranks: {first: 0, count: 1}
    interval: 1
    launch: {mode: plugin, plugin_path: /abs/path/liblinear_provider.so}
    fields:
      - {name: out, region: r1, disc: FV, components: 1, units: K,
         conservation: None, intent: Provides}
  - name: downstream
    ranks: {first: 0, count: 1}
    interval: 1
    launch: {mode: plugin, plugin_path: /abs/path/libaffine_provider.so}
    fields:
      - {name: b_in, region: r2, disc: FV, components: 1, units: K,
         conservation: None, intent: Requires}
      - {name: b_out, region: r2, disc: FV, components: 1, units: K,
         conservation: None, intent: Provides}
channels:
  - name: link
    source: upstream.out
    target: downstream.b_in
    remap: {flags: [Conservative]}
iteration_regions: []
```

Expected output of `./plugin_driver plugins.yaml 3` (exact digits can vary by platform; the linear
plugin writes only entity 0, and `downstream` sees the previous period's `out`, which is 2 after
three periods):

```text
downstream.b_out[0] = 1
downstream.b_out[1] = 0
downstream.b_out[2] = 0
downstream.b_out[3] = 0
```

The driver names no physics and no plugin: everything comes from the YAML. For an MPI run add
`MPI_Init`, pass `MPI_Comm_c2f(MPI_COMM_WORLD)` with `has_communicator = 1` to `setup()` before
`initialize()`, and launch as many ranks as the graph's rank sets need; each plugin is mounted on
its own ranks. The driver may also drive a component of its own with `begin_step`/`end_step`
instead of `run()`, with the plugins advancing in between ([](c-abi.md)). There is no shipped
executable that does this generically; the Brusselator `coupler_main` is the same sequence with a
fixed role. See [](drivers.md) for drivers in general and [](mixing-languages.md) for which
combinations of languages this supports.

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
`view->memspace` reports which. A plugin that writes a device field must do so through a kernel
launch, not a host store, and must complete that device work (fence its own execution space) before
`advance()` returns. FUSE also fences the default execution space after every successful `advance()`
of a provider that has a device field, but no behaviour is specified in terms of that fence or of a
device pointer being visible across streams. The device a rank uses is chosen at `create()` of the
host context from `KOKKOS_DEVICE_ID`, or `local_rank % device_count` when the variable is unset.

## A complete plugin in C

`test/abi/plugins/affine_provider.c` provides `b_out` and requires `b_in`, computes
`b_out[i] = offset + scale * b_in[i]` over every bound entity, keeps a real snapshot for
`save_state`/`restore_state`, declares `FUSE_CAP_CHECKPOINT`, and implements the restart-blob
entries as the explicit "nothing to capture" default. The two-field `bind()` is the one to study:

```{literalinclude} ../../test/abi/plugins/affine_provider.c
:language: c
:lines: 97-236,258-291,309-348
```

## Writing a plugin in Fortran

The generated module `ffuse_provider` (from the same `entries.def` as `ffuse`) gives you the
`fuse_provider_abi_v1` derived type with one `c_funptr` member per entry, an abstract interface per
entry so your own procedures are checked against the right signature, `ffuse_provider_set_<entry>()`
to assign each one via `c_funloc`, and `ffuse_provider_bootstrap()` to answer the version check.
[](fortran.md#writing-a-provider-plugin-the-ffuse_provider-module) lists the module's interface.
`test/abi/plugins/linear_provider.f90` is the complete Fortran twin of the C linear provider:

```{literalinclude} ../../test/abi/plugins/linear_provider.f90
:language: fortran
:lines: 7-200
```

Build it against `fuse::abi-fortran` and `fuse::abi` (a provider-only plugin never calls
`fuse_coupler_abi`, but the Fortran shim references it, so it must resolve at link time):

```cmake
add_library(my_provider SHARED my_provider.f90)
target_link_libraries(my_provider PRIVATE fuse::abi-fortran fuse::abi)
```

See [](build.md) for enabling the Fortran bindings.

## Error codes

| Code | Meaning |
|---|---|
| `E-ABI-LOAD` | `dlopen()`/`dlsym()` failed; the message carries `dlerror()`'s own text |
| `E-ABI-VERSION` | the plugin's table does not speak `FUSE_ABI_VERSION` (absent, too small, or a different `abi_version`), or `create()`/`advance()` wrote a `struct_size` this host does not recognise |
| `E-ABI-DESCRIBE` | `describe()` disagrees with the graph's own field declaration for a named field, including a field `describe()` reports that the graph does not declare for this component at all |
| `E-ABI-PROVIDER-CREATE` / `-BIND` / `-ADVANCE` / `-SAVE` / `-RESTORE` | the named entry returned a non-`FUSE_OK` status |
| `E-ABI-PROVIDER-CAPABILITIES` | the optional `capabilities()` entry exists but returned a non-`FUSE_OK` status |
| `E-ABI-PROVIDER-WINDOW` | `advance()` echoed a different window than the one it was asked to run |
| `E-ABI-PROVIDER-ELEMENT` | a field is not `FUSE_ELEMENT_F64`, the only element type this host supports through this ABI |
| `E-ABI-PROVIDER-MEMSPACE` | a field's `memspace` is neither `FUSE_MEMSPACE_HOST` nor `FUSE_MEMSPACE_DEVICE` |
| `E-ABI-PROVIDER-DEVICE-UNDECLARED` | a `describe()` row asks for `FUSE_MEMSPACE_DEVICE` but `capabilities()` did not report `FUSE_CAP_DEVICE_FIELDS` |
| `E-ABI-PROVIDER-NO-DEVICE` | a correctly declared device field, but this FUSE build has no CUDA/HIP/SYCL backend |
| `E-ABI-PROVIDER-TABLE` | the provider's table is absent, or missing one of `create`/`destroy`/`describe`/`bind`/`advance` |
| `E-ABI-PROVIDER-DESCRIBE` | the `describe()` call itself (the size query or the fill) returned a non-`FUSE_OK` status |
| `E-ABI-PROVIDER-GRAPH` | the provider's component name is not a component of the loaded graph |
| `E-ABI-PROVIDER-RESTART-SAVE` / `-RESTART-LOAD` | `save_restart_blob()`/`load_restart_blob()` returned a non-`FUSE_OK` status, or the size query and the fill disagreed |
| `E-ABI-PROVIDER-EXCEPTION` | a `std::exception` was caught around a plugin call |
| `E-ABI-REGISTRY-NULL` / `-DUPLICATE` / `-INSTALL` / `-NOTFOUND` | the mount table was given a null table, a second provider under a name already in use, failed to install a mounted provider, or was asked for a name nothing is mounted under |
| `E-LAUNCH-PLUGIN-PATH` | a `plugin`-launched component declares an empty `plugin_path` |
| `E-ABI-PLUGIN-AGREEMENT-REFUSED` | a different rank's plugin-mount refusal was agreed across the base communicator; the message names which rank and which status |
| `E-ABI-PLUGIN-NO-SPANNING-COMM` | a plugin component and a `connect`-launched one share a graph, and this process's own `my_component` does not itself connect; see below |

## A known failure case

A graph that declares a plugin component **and** a `connect`-launched one (independent MPI
universes, no common communicator until `fuse::transport::connect()` performs its
`MPI_Open_port`/`MPI_Comm_accept`/`MPI_Comm_connect` handshake) is refused at `initialize()`,
before the plugin is touched, whenever the calling process's own `my_component` is not itself one
of the connecting components: `E-ABI-PLUGIN-NO-SPANNING-COMM`. Agreeing a plugin-mount verdict needs
a communicator that spans every rank with an opinion on it, and a process whose `my_component`
never calls `connect()` never gains one. `mpmd` does not have this problem, because
`MPI_COMM_WORLD` already spans every program, so a plugin component alongside an `mpmd` one is never
refused this way.

Workaround: host the plugin component in a process that is itself `connect`-launched (so its own
`connect()` call forms the spanning communicator before `initialize()` needs one), or declare the
multi-executable components `mpmd` instead of `connect` if your launcher supports one `mpiexec`
with several segments.

## A note on exported symbols

A plugin written in C exports exactly one symbol, `fuse_provider_abi`, and the conformance suite
checks that with `nm`. A plugin written in Fortran also exports the module-procedure symbols
gfortran generates for `ffuse_provider` and for the plugin's own module (`__ffuse_provider_MOD_*`,
`__<module>_MOD_*`); these are the compiler's mangled names, not part of the ABI, and a host never
looks them up. Hiding them needs a linker version script or `-fvisibility=hidden` on the Fortran
objects; FUSE does not apply either to plugins you build.

## Limitations

- Provider fields must be `FUSE_ELEMENT_F64` (`E-ABI-PROVIDER-ELEMENT`) and `FUSE_MEMSPACE_HOST` or
  `FUSE_MEMSPACE_DEVICE` (`E-ABI-PROVIDER-MEMSPACE`).
- `FUSE_MEMSPACE_DEVICE` needs a build with a CUDA, HIP or SYCL backend.
- `bind()` carries no field name; a multi-field plugin relies on the graph's declared field order.
- `create()` takes no arguments: a plugin's run-time configuration comes from its own environment
  or files, not from the graph.
- Plugins are mounted only by the participant ABI's `initialize()`; a C++ `fuse::Runner` program
  or a Python program cannot host one.
- A graph that declares a plugin component together with a `connect`-launched component is refused
  in some process configurations; see [A known failure case](#a-known-failure-case).
- A plugin inside an iteration region needs real `save_state()`/`restore_state()` entries and
  `FUSE_CAP_CHECKPOINT` in its `capabilities()`.

## See also

- [](c-abi.md): the opposite direction, your code driving FUSE, and the driver's entry points.
- [](fortran.md): the `ffuse` and `ffuse_provider` modules.
- [](mixing-languages.md): which languages can be combined, and how.
- [](restart.md): restart blobs for providers.
- [](build.md): building the ABI libraries, including the Fortran bindings.
- [](multiple-executables.md): launch modes other than `plugin`.
- [](troubleshooting.md): diagnosing `E-...` codes.
