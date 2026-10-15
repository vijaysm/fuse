# Restart-to-disk

Restart is a different problem from the in-memory rollback checkpoint
`Component::save_state()`/`restore_state()` already provide for iteration
regions (see `docs/user/glossary.md`'s own entry on that pair): rollback
re-advances one sweep inside the SAME process; restart writes a run's own
state to a file so a LATER process -- possibly at a different rank count --
can resume it. The full design is `docs/dev/restart.md`; this page is the
practical "how do I use it" companion, current as of Stage B Slice 1
(same-rank-count round trip, both file layouts, every refusal; a
rank-count-independent restart is Slice 2 work, not yet implemented).

## Building it

Restart-to-disk needs `-DFUSE_ENABLE_HDF5=ON`, against a plain HDF5 install
(`CMAKE_PREFIX_PATH` pointing at it, the same way `FUSE_ENABLE_LOWFIVE`
already resolves HDF5 -- the two options SHARE that resolution, so turning
both on never links two different HDF5 images into one process). If that
HDF5 was built parallel (`H5_HAVE_PARALLEL`), its own public header pulls in
`<mpi.h>` unconditionally; the build finds an MPI include path for this
reason even when `FUSE_ENABLE_MPI` is OFF, the identical situation
`FUSE_ENABLE_PETSC` already handles for `<petsc.h>`.

## What a component must do to be restartable

A component that does nothing is simply not restartable -- the default
`write_restart()`/`read_restart()`/`save_restart_blob()`/`load_restart_blob()`
are no-ops, and `Runner::write_restart_fields()` skips a component whose
`capabilities()` sets neither `kCapabilityRestartFormA` nor
`kCapabilityRestartFormB`.

**Form A** (`kCapabilityRestartFormA`): override `write_restart(RestartWriter&)`
to call `writer.put_field(name, ids, values, components, declared_global_size)`
for every array this component owns, keyed by ITS OWN global ids. A
component that lives on one rank numbers its own entries `0..n-1` -- there
is nothing to re-partition, so this is always available. **The field `name`
must equal the field's own name in the coupling graph** (`FieldSpec::name`)
for any field an accelerated iteration region's own channel feeds, because
the accelerator's row-to-component resolution looks a `RegionBlock::
target_field` up by (owning component name, that exact field name) --
see `examples/dn_heat/dn_component.hpp`'s own `write_restart()` for a
worked example and the measured reason this naming rule exists.

**Form B** (`kCapabilityRestartFormB`): override `save_restart_blob()`/
`load_restart_blob()`/`restart_blob_version()` for an opaque,
`memcpy`-cheap byte record. Cheaper to write than Form A, but
same-decomposition-only: a restart at a DIFFERENT rank count is refused,
collectively, if ANY component in the graph sets this bit without ALSO
setting Form A's.

## Writing and restoring a point

```cpp
#include <fuse/runtime/restart.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>

// Write, at a period boundary (never mid-period -- see docs/dev/
// restart.md Section 1.1 and KNOWN_ISSUES.md D63):
fuse::restart_io::Provenance prov;
prov.descriptor_hash = fuse::restart_descriptor_hash(graph);
prov.written_by_rank_count = 1;  // your own MPI_Comm_size(), at tier 0 always 1
auto pw = fuse::restart_io::create_restart_point(
    fuse::restart_io::FileLayout::SeparateMeshAndPoints, "run.3.h5", prov);
fuse::write_restart_point(runner, components, pw.value()->writer(),
                          current_time, runner.current_period_index());
pw.value()->finish();  // atomic rename; the point does not exist until this returns success

// Restore, before the first run_period()/run() call:
auto pr = fuse::restart_io::open_restart_point(
    fuse::restart_io::FileLayout::SeparateMeshAndPoints, "run.3.h5");
// Check pr.value()->provenance().descriptor_hash against your OWN current
// graph's fuse::restart_descriptor_hash() before trusting anything else.
fuse::Rational restored_time; std::int64_t restored_period = 0;
bool any_accelerator_reset = false;
fuse::restore_restart_point(runner, components, pr.value()->reader(),
                            /*allow_accelerator_reset_on_mismatch=*/false,
                            any_accelerator_reset, restored_time, restored_period);
// runner.current_period_index() now reads restored_period; the NEXT
// run_period()/run() call resumes the Runner's own clock from there.
```

## The two file layouts

`FileLayout::SeparateMeshAndPoints`: one file per restart point, created
fresh and atomically renamed into place on `finish()`. `FileLayout::
GrowingFile`: one file, growing by copy-append-rename per point (the
owner's robustness-first choice: a crash mid-write can never damage an
earlier point, at the cost of one file copy per point) -- `point_label`
names the new point's own group, and is required on both `create_restart_
point()`/`open_restart_point()` calls for this layout.

## Refusals

Every refusal is a named `fuse::Failure` code, never a silent default:
`E-RESTART-VERSION` (unsupported `@format_version`), a caller-side
`descriptor_hash`/`mesh_checksum` comparison (the reader hands back its own
recorded values; the CALLER decides whether they match its current graph
and mesh), `E-RESTART-FORMB-ONLY-RANK-CHANGE` (`fuse::check_restart_rank_
count()`), `E-RESTART-MISSING-FIELD`/`E-RESTART-MISSING-SCALAR`/
`E-RESTART-MISSING-BLOB`/`E-RESTART-MISSING-POINT` (nothing by that name in
the file), `E-RESTART-MISSING-ID` (the field exists but not this row),
`E-RESTART-DUPLICATE-ID`/`E-RESTART-BAD-NUMBERING` (owner decision 2's
batched check, or its per-call local pre-check), `E-RESTART-TRUNCATED` (not
a valid HDF5 file at all). An accelerator whose current stacked-vector
layout does not match the file's own refuses the same way, UNLESS the
caller passes `allow_accelerator_reset_on_mismatch = true`, in which case
`Accelerator::reset()` runs instead and `any_accelerator_reset` is set --
always check it and report it, never ignore it.

A SEPARATE refusal, added this round, covers an accelerator that has never
implemented restart state at all (`Accelerator::restart_state_implemented()`
returns `false` -- true today only for `QuasiNewtonAccel`/IQN-ILS and
IQN-IMVJ; `None`, `Constant`, `Aitken`, and `Steffensen` all implement it):
`E-RESTART-ACCEL-NOT-IMPLEMENTED`, again UNLESS `allow_accelerator_reset_on_
mismatch = true`, in which case `reset()` runs and `any_accelerator_reset`
is set exactly as above. The distinction matters: the layout-mismatch
refusal above means the file's own accelerator state exists but does not fit
the CURRENT run's shape; this one means no accelerator state was ever
written for that region in the first place, because the accelerator kind
does not support it -- restoring into an IQN-accelerated region without
opting in always refuses, never silently restarts the accelerator cold.

## Restart through the C/Fortran ABI

A provider plugin (C or Fortran, via the generated `fuse_provider_abi_v1`
table) implements restart-to-disk Option B -- an opaque, versioned byte
blob, same-decomposition restore only -- through two table entries:
`save_restart_blob` (query-then-fill, the same two-call idiom
`save_graph` already uses) and `load_restart_blob`. A plugin that does not
implement them reports `written = 0`/`blob_version = 0` from
`save_restart_blob`, which the caller treats as "nothing to capture", not
as an error; a plugin that does implement them sets capability bit
`FUSE_CAP_RESTART_FORM_B` from its own `capabilities()` entry.

An externally-driven (ABI) participant asks its embedding host to take a
restart point through `request_restart_point` -- a flag, not a file write:
the ABI's fixed `(ctx, in, out)` entry shape cannot carry a file path or a
layout choice, so the embedding host must poll (after every
`run_period()`/`resume_period()` call) and perform the actual write itself
once it sees the request. Calling `request_restart_point` more than once
before the host polls leaves exactly one pending request, never a queue.
