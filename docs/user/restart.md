# Restart-to-disk

This page is for people who want a run to survive the end of its process: write the run's state to a
file at a period boundary, then start a later process, possibly with a different number of ranks, and
continue from that file.

Restart is different from the in-memory rollback that `Component::save_state()`/`restore_state()`
provide for iteration regions (see [](glossary.md)). Rollback re-advances one sweep inside the same
process; restart writes state to a file so a later process can resume.

## The smallest working example

```cpp
#include <fuse/runtime/restart.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>

// Write, at a period boundary (never mid-period):
fuse::restart_io::Provenance prov;
prov.descriptor_hash = fuse::restart_descriptor_hash(graph);
prov.written_by_rank_count = 1;  // your MPI_Comm_size(); 1 for a single-process run
auto pw = fuse::restart_io::create_restart_point(
    fuse::restart_io::FileLayout::SeparateMeshAndPoints, "run.3.h5", prov);
fuse::write_restart_point(runner, components, pw.value()->writer(),
                          current_time, runner.current_period_index());
pw.value()->finish();  // atomic rename; the point does not exist until this returns success

// Restore, before the first run_period()/run() call:
auto pr = fuse::restart_io::open_restart_point(
    fuse::restart_io::FileLayout::SeparateMeshAndPoints, "run.3.h5");
// Check pr.value()->provenance().descriptor_hash against your own current graph's
// fuse::restart_descriptor_hash() (or call fuse::restart_io::check_restart_descriptor_hash())
// before trusting anything else.
fuse::Rational restored_time; std::int64_t restored_period = 0;
bool any_accelerator_reset = false;
fuse::restore_restart_point(runner, components, pr.value()->reader(),
                            /*allow_accelerator_reset_on_mismatch=*/false,
                            any_accelerator_reset, restored_time, restored_period);
// runner.current_period_index() now reads restored_period; the next
// run_period()/run() call resumes the Runner's clock from there.
```

Every call returns a `Result`; check each one. A point exists only after `finish()` returns success.
A `PointWriter` destroyed without `finish()` leaves no point behind.

## Build requirement

Restart-to-disk needs `-DFUSE_ENABLE_HDF5=ON`, against a plain HDF5 install (point
`CMAKE_PREFIX_PATH` at it). The option shares its HDF5 resolution with `FUSE_ENABLE_LOWFIVE`, so
enabling both never links two different HDF5 libraries into one process. If that HDF5 was built
parallel (`H5_HAVE_PARALLEL`), its public header includes `<mpi.h>` unconditionally, and the build
finds an MPI include path for that reason even when `FUSE_ENABLE_MPI` is off. See [](build.md).

## Make a component restartable

A component that does nothing is not restartable. The default `write_restart()`/`read_restart()`/
`save_restart_blob()`/`load_restart_blob()` are no-ops, and `Runner::write_restart_fields()` skips a
component whose `capabilities()` sets neither `kCapabilityRestartFormA` nor
`kCapabilityRestartFormB`.

**Form A** (`kCapabilityRestartFormA`): override `write_restart(RestartWriter&)` (and
`read_restart(RestartReader&)`) to call
`writer.put_field(name, ids, values, components, declared_global_size)` for every array the component
owns, keyed by the component's own global ids. A component that lives on one rank numbers its entries
`0..n-1`; there is nothing to re-partition. Form A is what allows restart at a different rank count.

:::{important}
The field `name` must equal the field's name in the coupling graph (`FieldSpec::name`) for any field
that an accelerated iteration region's channel feeds. The accelerator resolves a
`RegionBlock::target_field` by (owning component name, that exact field name).
`examples/dn_heat/dn_component.hpp` has a worked `write_restart()`.
:::

**Form B** (`kCapabilityRestartFormB`): override `save_restart_blob()`, `load_restart_blob()` and
`restart_blob_version()` to write an opaque byte record. It is cheaper to write than Form A but
restores only at the same decomposition: a restart at a different rank count is refused,
collectively, if any component in the graph sets Form B without also setting Form A
(`E-RESTART-FORMB-ONLY-RANK-CHANGE`, from `fuse::check_restart_rank_count()`).

## Typed fields

`RestartWriter::put_field<T>()`/`put_scalar<T>()` and `RestartReader::get_field<T>()`/`get_scalar<T>()`
are templates over one byte-based virtual pair. `T` resolves to a `fuse::ElementType` at compile
time, and a `static_assert` fires for a `T` outside the supported set: `std::int8_t` to
`std::int64_t`, `std::uint8_t` to `std::uint64_t`, `fuse::bfloat16`, `float`, `double`,
`long double`, `bool`, and `_Float16` where the compiler provides it. Every field and scalar carries
its element-type metadata in the file (see "Element types" in [](glossary.md)), and the reader never
assumes its `T` matches what was written.

```cpp
// Writing an integer and a bool alongside double fields:
writer.put_field<std::int32_t>("dirichlet.sweep_count", ids, sweep_counts, 1);
writer.put_scalar<bool>("dirichlet.parity_flag", parity);

// Reading back. The default conversion is ReadConversion::Exact, so a mismatched T
// refuses by name (E-RESTART-TYPE-MISMATCH) instead of reinterpreting bytes.
std::vector<std::int32_t> sweep_counts(ids.size());
auto rr = reader.get_field<std::int32_t>("dirichlet.sweep_count", ids,
                                         std::span(sweep_counts), 1);
bool parity = false;
auto rs = reader.get_scalar<bool>("dirichlet.parity_flag", parity);

// Reading a widened or narrowed type must be requested explicitly:
double widened;
auto rw = reader.get_scalar<double>("some.bfloat16.scalar", widened,
                                    fuse::ReadConversion::Widening);
if (rw.ok() && rw.value().max_absolute_change > 0.0) {
  // With AllowNarrowing, the largest change actually measured over the values read
  // is reported here; report and act on it.
}
```

Accelerator and waveform histories (`dn_heat`'s `DirichletSlab`/`NeumannSlab` state) are stored as
`double` (binary64). `DirichletSlab` also carries an `std::int32_t` sweep count and a `bool` parity
flag that round-trip through the `Runner`.

## The two file layouts

- `FileLayout::SeparateMeshAndPoints`: one file per restart point, created fresh and atomically
  renamed into place on `finish()`.
- `FileLayout::GrowingFile`: one file that grows by copy-append-rename per point. A crash during a
  write can never damage an earlier point, at the cost of one file copy per point. `point_label`
  names the new point's group and is required on both `create_restart_point()` and
  `open_restart_point()` for this layout.

## Refusals

Every refusal is a named `fuse::Failure` code, never a silent default.

| Code | Meaning |
|---|---|
| `E-RESTART-VERSION` | unsupported `@format_version` in the file |
| `E-RESTART-DESCRIPTOR-VERSION` | the point was written under a different graph descriptor version; raised by `open_restart_point()` |
| `E-RESTART-DESCRIPTOR-HASH` | same descriptor version but a different graph; raised by `check_restart_descriptor_hash()` |
| `E-RESTART-FORMB-ONLY-RANK-CHANGE` | rank count differs and a component offers Form B without Form A |
| `E-RESTART-MISSING-FIELD`, `-SCALAR`, `-BLOB`, `-POINT` | nothing by that name in the file |
| `E-RESTART-MISSING-ID` | the field exists, but not this row |
| `E-RESTART-DUPLICATE-ID`, `E-RESTART-BAD-NUMBERING` | the batched or per-call id check on write failed |
| `E-RESTART-TRUNCATED` | not a valid HDF5 file |
| `E-RESTART-ACCEL-NOT-IMPLEMENTED` | the region's accelerator kind has no restart state (see below) |
| `E-RESTART-TYPE-MISMATCH` | `Exact` found an element-type difference, or `Widening` was asked for a conversion not proven exact (a float narrowing or any signed-to-unsigned conversion is never widening) |
| `E-RESTART-TYPE-METADATA-MISMATCH` | the file's per-field attributes (`element_type`, `byte_size`, `significand_bits`, `exponent_bits`, `exponent_bias`, `byte_order`, `is_signed`) disagree with the file's own HDF5 datatype |
| `E-RESTART-UNKNOWN-ELEMENT-TYPE` | an `element_type` name this build does not recognise |
| `E-RESTART-BAD-BOOL-BYTE` | a `Bool` field's stored byte is neither `0` nor `1` |
| `E-RESTART-INTEGER-OVERFLOW` | an integer narrowing does not fit the target type; refused even under `AllowNarrowing` |
| `E-RESTART-SIGNED-TO-UNSIGNED` | refused under every `ReadConversion` mode |

The reader hands back the provenance recorded in the file (`format_version`, `descriptor_hash`,
`written_by_rank_count`, `mesh_checksum`). `open_restart_point()` itself checks the format and
descriptor versions; comparing `descriptor_hash` (`check_restart_descriptor_hash()`) and
`mesh_checksum` with your current graph and mesh is the caller's job, and should happen before
`reader()` is used.

### Accelerators

An accelerator whose current stacked-vector layout does not match the file's refuses restart with a
named error, unless you pass `allow_accelerator_reset_on_mismatch = true`. In that case
`Accelerator::reset()` runs instead and `any_accelerator_reset` is set. Always check and report that
flag.

A separate refusal, `E-RESTART-ACCEL-NOT-IMPLEMENTED`, applies to an accelerator kind that has no
restart state (`Accelerator::restart_state_implemented()` returns `false`). `None`, `Constant`,
`Aitken`, `Steffensen` and the quasi-Newton accelerators (IQN-ILS and IQN-IMVJ) all implement
restart state, so this applies to a custom accelerator that does not. Without the opt-in flag it
refuses rather than silently restarting the accelerator cold; with it, `reset()` runs as above.

## Change the rank count between write and restore

A restart point written at one rank count can be restored at a different one, provided every
component in the graph uses Form A and keys its rows by a stable global id
(`fuse::MeshView::global_id(dim, i)`), not by position in its own local array. The reader resolves
each row through an id-to-position map built from the file's stored ids, never from the order the
caller requests them in. A reader at a different rank count asks for a different set of ids than the
writer owned and gets exactly those rows.

`examples/brusselator/species_component.hpp` is the worked example. `UComponent` and `VComponent` both
set `kCapabilityRestartFormA` and key every field, including each component's private hidden state,
by `NativeGrid::global_id()`. `VComponent` is the case to copy: it owns two id spaces in one
component (U-grid ids for the fields it exchanges, V-grid ids for its private `v_state`) and keys
each by the grid it belongs to.

Accelerator state and window-waveform history move across a rank-count change the same way, keyed by
(field, global id). A window member's history is restored keyed by the member's exported field's
global ids, one triple per retained sample. A window-scoped region's accelerator state is stored one
sub-dataset per sample, keyed by (field, global id, sample index).

### Writing from several ranks

`fuse::restart_io::create_restart_point_collective()`
(`include/fuse/restart_io/hdf5_restart_io_mpi.hpp`, available with `FUSE_ENABLE_MPI`) lets every rank
write its own rows of a field at an `MPI_Exscan`-derived offset in one collective hyperslab write.
It takes the communicator as a Fortran handle (`int comm_f`), and every rank must call `finish()`
together. Limits:

- Only `ElementType::Binary64` and `Int64` fields; other types refuse with
  `E-RESTART-TYPE-MISMATCH`.
- Only `FileLayout::SeparateMeshAndPoints`; `GrowingFile` refuses with
  `E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED`.
- Scalars, rationals and blobs refuse with `E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED`.

Reading needs nothing special, because rows are resolved by id regardless of which rank wrote them.

### What to expect after a rank-count change

A restored field, accelerator or window-history row compares bit-identically by global id across any
change in rank count, because restoring is a file read. State produced by advancing one further
window after the restore is different: a solver with cross-rank inner products (such as the
Brusselator's Schur-decomposed solve) sums in an order that depends on the partition. A run continued
at a different rank count therefore differs from one that never changed count at the roughly 1e-13
level. This is round-off, not a restart defect.

## Restart through the C and Fortran ABI

A provider plugin (C or Fortran, through the `fuse_provider_abi_v1` table) implements Form B through
two entries. `save_restart_blob` uses a query-then-fill idiom like `save_graph`: call with
`buffer = NULL` to get `written`, then again with a buffer. `load_restart_blob` takes the bytes back
and a `blob_version`, and returns `FUSE_ERR_VERSION` for a version it does not recognise. A plugin
that does not implement them reports `written = 0` and `blob_version = 0`, which the caller treats
as "nothing to capture", not as an error. A plugin that implements them sets `FUSE_CAP_RESTART_FORM_B`
in its `capabilities()` flags. See [](providers.md).

An externally driven (participant) code asks its embedding host for a restart point with
`request_restart_point`. That entry sets a flag and writes no file, because the ABI's fixed
`(ctx, in, out)` shape cannot carry a file path or layout. The embedding host polls
`fuse::Runner::take_restart_point_request()` after each `run_period()`/`resume_period()` call and
performs the write itself. Repeated calls before the host polls leave exactly one pending request.
See [](c-abi.md).

## Limitations

- Restart points are written and restored at period boundaries only.
- Restart at a different rank count requires Form A on every component.
- Collective writing is limited as described above.

## See also

- [](build.md): `FUSE_ENABLE_HDF5` and `FUSE_ENABLE_MPI`.
- [](c-abi.md) and [](providers.md): restart through the ABIs.
- [](glossary.md): element types and rollback checkpoints.
- [](troubleshooting.md): diagnosing `E-...` codes.
