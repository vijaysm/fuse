# Restart-to-disk

Restart is a different problem from the in-memory rollback checkpoint
`Component::save_state()`/`restore_state()` already provide for iteration
regions (see `docs/user/glossary.md`'s own entry on that pair): rollback
re-advances one sweep inside the SAME process; restart writes a run's own
state to a file so a LATER process -- possibly at a different rank count --
can resume it. The full design is `docs/dev/restart.md`; this page is the
practical "how do I use it" companion, current as of Stage B Slice 2
(same-rank-count round trip, both file layouts, every refusal, plus a
rank-count-INDEPENDENT restore -- see "Changing the rank count between
write and restore" below for exactly what that covers and what it still
does not).

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

## Typed fields

`RestartWriter::put_field<T>()`/`put_scalar<T>()` and `RestartReader::
get_field<T>()`/`get_scalar<T>()` are templates over ONE virtual byte-based
pair (`*_bytes()`); `T` is resolved to a `fuse::ElementType` at compile
time via `ElementTypeOf<T>` (a `static_assert` fires for any `T` outside the
supported set: `std::int8_t`..`std::int64_t`, `std::uint8_t`..`std::
uint64_t`, `fuse::bfloat16`, `float`, `double`, `long double`, `bool`, and
`_Float16` where the compiler provides it). Every field and scalar carries
its own element-type metadata in the file (see `docs/user/glossary.md`'s
"Element types" section); the writer never assumes the reader's `T` matches
what was written.

```cpp
// Writing an integer and a bool field alongside the existing double ones:
writer.put_field<std::int32_t>("dirichlet.sweep_count", ids, sweep_counts, 1);
writer.put_scalar<bool>("dirichlet.parity_flag", parity);

// Reading back: the default conversion is ReadConversion::Exact, so a
// mismatched T refuses by name (E-RESTART-TYPE-MISMATCH) rather than
// silently reinterpreting bytes.
std::vector<std::int32_t> sweep_counts(ids.size());
auto rr = reader.get_field<std::int32_t>("dirichlet.sweep_count", ids,
                                         std::span(sweep_counts), 1);
bool parity = false;
auto rs = reader.get_scalar<bool>("dirichlet.parity_flag", parity);

// A caller that KNOWS it is reading a widened or narrowed type says so
// explicitly; it is never the default:
double widened;
auto rw = reader.get_scalar<double>("some.bfloat16.scalar", widened,
                                     fuse::ReadConversion::Widening);
if (rw.ok() && rw.value().max_absolute_change > 0.0) {
  // AllowNarrowing: report and act on the largest change actually
  // measured over the values read, never silently accepted.
}
```

`dn_heat`'s `DirichletSlab`/`NeumannSlab` keep their existing physical state
(`time_`, `history_`, `beta_`, `t_ref_`) as `double` (`Binary64`), per the
owner's decision 7 (accelerator/waveform histories stay binary64,
documented in `docs/dev/restart.md` Section 11). `DirichletSlab` ALSO
carries a demonstration `std::int32_t sweep_count_` and `bool
parity_flag_`, round-tripped end to end through the `Runner` by `test/
runtime/restart_test.cpp`'s `RunResult::dirichlet_sweep_count`/
`dirichlet_parity_flag`.

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

A SEPARATE family of refusals, added for typed fields, covers element-type
mismatches: `E-RESTART-TYPE-MISMATCH` (`ReadConversion::Exact`'s
refuse-by-name on any element-type difference, or `Widening`'s refusal when
the stored-to-requested conversion is not PROVEN exact from the two types'
own metadata -- a float narrowing or any signed-to-unsigned conversion is
never "widening"), `E-RESTART-TYPE-METADATA-MISMATCH` (the file's own
per-field attributes -- `element_type`, `byte_size`, `significand_bits`,
`exponent_bits`, `exponent_bias`, `byte_order`, `is_signed` -- disagree with
the file's own HDF5 dataset/attribute datatype; this is checked for
INTERNAL self-consistency against the dataset's own actual type, never
against this build's global `element_type_info()` table, so a genuinely
foreign-platform file is never structurally impossible to read),
`E-RESTART-UNKNOWN-ELEMENT-TYPE` (an `element_type` attribute name this
build's `element_type_from_name()` does not recognise -- a newer file
format reader should refuse by name, not guess), `E-RESTART-BAD-BOOL-BYTE`
(a `Bool` field's stored byte is neither `0` nor `1`), `E-RESTART-INTEGER-
OVERFLOW` (an integer narrowing that does not fit the target type, refused
even under `AllowNarrowing` -- this mode bounds silent error, it does not
disable the overflow check), and `E-RESTART-SIGNED-TO-UNSIGNED` (refused
unconditionally, under every `ReadConversion` mode including
`AllowNarrowing` -- a negative stored value reinterpreted as unsigned is
never an acceptable "narrowing").

## Changing the rank count between write and restore

A restart point written at one task count can be restored at a DIFFERENT
task count, provided every component in the graph uses Form A and keys its
own rows by a stable global id (`fuse::MeshView::global_id(dim, i)`, not a
position in the component's own local array). `HDF5GroupReader::
get_field_bytes()` (`src/restart_io/hdf5_restart_io.cpp`) resolves every row
by an id-to-position map built from the file's own stored ids, never from
the order the caller happens to request them in -- a reader at a different
rank count simply asks for a different, smaller or larger, set of ids than
the writer owned, and gets back exactly those rows regardless of where they
sit in the file. `examples/brusselator/species_component.hpp`'s
`UComponent`/`VComponent` are the worked example: both set
`kCapabilityRestartFormA` and key every field -- the registered `u_out`/
`u_in`/`v_out`/`v_in` AND each component's own private hidden state
(`UCore::state_`/`q_prev_`, `VDiffusionCore::state_`) -- by
`NativeGrid::global_id()`. `VComponent` is the case worth copying exactly:
it owns TWO different id spaces in one component (U-grid ids for the
fields it exchanges, V-grid ids for its own private `v_state`), and keys
each by the grid it actually belongs to, never by the other.

Accelerator state and window-waveform history move across a rank-count
change the same way -- keyed by (field, global id), not by position.
`QuasiNewtonAccel` (IQN-ILS and IQN-IMVJ) now implements restart state like
every other accelerator kind (`None`, `Constant`, `Aitken`, `Steffensen`
already did); `Accelerator::restart_state_implemented()` is `true` for all
of them, closing `KNOWN_ISSUES.md` D64. A window member's own `history`
(the per-sample record `detail::WindowMemberState` keeps for its waveform)
restarts the same way, keyed by the member's own exported field's global
ids, one triple per retained sample.

**The WRITE side is now collective.**
`fuse::restart_io::create_restart_point_collective()`
(`include/fuse/restart_io/hdf5_restart_io_mpi.hpp`, compiled under
`FUSE_ENABLE_MPI`) lets every task write its own rows of a field at an
`MPI_Exscan`-derived offset in one collective hyperslab write, so a
restart point can now genuinely be PRODUCED by more than one task, not
only read back by a different count than one. This round's own declared
scope: `ElementType::Binary64`/`Int64` fields only (anything else refuses
`E-RESTART-TYPE-MISMATCH`); `FileLayout::SeparateMeshAndPoints` only
(`FileLayout::GrowingFile` refuses `E-RESTART-COLLECTIVE-LAYOUT-
UNSUPPORTED`); scalars, rationals, and blobs all refuse
(`E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED`) -- HDF5's own metadata
operations are collective-required even for a value one task alone
would otherwise host, and this round's own time did not extend to that
path. The READ side needed no changes at all: `get_field_bytes()`
already resolved every row by id (above), regardless of which task, or
how many, wrote it.

A restored (pre-advance) field, accelerator, or window-waveform-history
row compares bit-identically BY GLOBAL ID across any change in task
count, because restoring is a file read, not a recomputation. A state
produced by advancing the simulation one FURTHER window after such a
restore is a different matter: Brusselator's own Schur-decomposed solve
path performs cross-task inner products whose floating-point summation
order depends on how the mesh is partitioned, so comparing a run
continued at a different task count against a reference that never
changed count will disagree at the ~1e-13 scale
(`test/verification/brusselator_species_core_decomposition_test.cpp`'s
own `TwoPartitionSchurMatchesSingleRankUCore` /
`ThreePartitionSchurMatchesSingleRankUCore` /
`OneWindowAtRestartCadenceMeasuresDecompositionOrderSensitivity`) --
expected, measured, round-off, not a restart defect.

What this does NOT yet cover, precisely:

- **A window-scoped region's own accelerator state now restarts.**
  `KNOWN_ISSUES.md` D69 (renumbered from round 1's own D68, which a later
  `main` commit took for an unrelated BoxIndex fix), CLOSED: a
  window-scoped channel's accelerated vector -- the whole stacked
  window-transfer buffer, not the single-sample size a component's
  `write_restart()` reports ids for -- is now demuxed into one sub-dataset
  per sample, keyed by (field, global id, sample index), and refuses by
  name rather than silently resetting when the run did not opt into
  `allow_accelerator_reset_on_mismatch`. **Still open** (`KNOWN_ISSUES.md`
  D72): a window-scoped region's own CONSUMER-side transfer buffer
  (`cfg.window_import`) is not correctly reconstructed after a restart, so
  a `Runner`-driven round trip with a real accelerator (Aitken, IQN-ILS)
  over a window-scoped region does not yet reproduce the reference run's
  own iteration count, even though the accelerator's own state round-trips
  correctly in isolation.
- **A window member's `last_window_waveform` is implemented but dead code
  under the `Runner` driver.** `KNOWN_ISSUES.md` D70 (renumbered from
  round 1's own D69): only the standalone `fuse::WindowCoupler` driver
  populates this member today; a `Runner`-driven restart point never has
  anything to write there, so restoring it is a no-op in practice until
  that gap closes.
- **A real accelerator over a `WaveformIteration::GaussSeidel` window
  refuses restart outright.** `KNOWN_ISSUES.md` D71, pre-existing, named
  precisely this round: `E-RT-WINDOW-ACCEL`
  (`src/runtime/iteration.cpp:1836-1851`). Out of restart's own scope to
  fix.

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
