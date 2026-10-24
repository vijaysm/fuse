# Restart-to-disk

This page is for people who want a run to survive the end of its process: write the run's state to a
file at a period boundary, then start a later process, possibly with a different number of ranks, and
continue from that file.

Restart is different from the in-memory rollback that `Component::save_state()`/`restore_state()`
provide for iteration regions (see [](glossary.md)). Rollback re-advances one sweep inside the same
process; restart writes state to a file so a later process can resume.

The pieces, by header:

| Header | Provides |
|---|---|
| `fuse/runtime/component.hpp` | the hooks a component implements: `write_restart()`/`read_restart()` (form A), `save_restart_blob()`/`load_restart_blob()`/`restart_blob_version()` (form B), and the `kCapabilityRestartFormA`/`kCapabilityRestartFormB` bits |
| `fuse/model/restart.hpp` | `RestartWriter`/`RestartReader` (what the hooks are handed), `fuse::restart_descriptor_hash(graph)`, `ReadConversion` |
| `fuse/runtime/restart.hpp` | `fuse::write_restart_point()`, `fuse::restore_restart_point()`, `fuse::check_restart_rank_count()` |
| `fuse/restart_io/hdf5_restart_io.hpp` | the HDF5 file backend: `create_restart_point()`/`open_restart_point()`, `PointWriter`/`PointReader`, `Provenance`, `FileLayout`, `check_restart_descriptor_hash()` |
| `fuse/restart_io/hdf5_restart_io_mpi.hpp` | `create_restart_point_collective()` for writing from several ranks |

## Build requirement

Restart-to-disk needs `-DFUSE_ENABLE_HDF5=ON`, against a plain HDF5 install (point
`CMAKE_PREFIX_PATH` at it). The option shares its HDF5 resolution with `FUSE_ENABLE_LOWFIVE`, so
enabling both never links two different HDF5 libraries into one process. If that HDF5 was built
parallel (`H5_HAVE_PARALLEL`), its public header includes `<mpi.h>` unconditionally, and the build
finds an MPI include path for that reason even when `FUSE_ENABLE_MPI` is off. See [](build.md).

## The smallest working example

Three parts: a component that implements the form A hooks, a function that writes a point, and a
function that restores one. The component here owns two arrays on one rank and keys them by
position; a partitioned component keys them by mesh global id instead (see "Change the rank count").

```cpp
#include <fuse/model/restart.hpp>
#include <fuse/restart_io/hdf5_restart_io.hpp>
#include <fuse/runtime/component.hpp>
#include <fuse/runtime/restart.hpp>
#include <fuse/runtime/runner.hpp>

class Slab final : public fuse::Component {
 public:
  explicit Slab(std::size_t n) : n_(n), q_(n, 0.0), u_in_(n, 0.0) {}
  [[nodiscard]] std::string name() const override { return "slab"; }
  fuse::Result<void> register_fields(fuse::FieldRegistry& reg,
                                     const fuse::CouplingGraph& g) override {
    const auto cid = *g.find_component("slab");
    const fuse::Layout layout{.stride = 1, .count = n_};
    auto r = reg.bind(*g.find_field(cid, "q_wall"),
                      fuse::FieldRef{q_.data(), layout, fuse::MemorySpace::Host});
    if (!r.ok()) return r;
    return reg.bind(*g.find_field(cid, "u_wall_in"),
                    fuse::FieldRef{u_in_.data(), layout, fuse::MemorySpace::Host});
  }
  fuse::Result<void> advance(const fuse::StepContext& ctx) override {
    time_ = ctx.t_end.to_double();
    ++steps_;
    return fuse::Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRestartFormA;      // or-ed with the component's other bits
  }

  // Form A: every array the component owns, keyed by its own global ids.
  fuse::Result<void> write_restart(fuse::RestartWriter& writer) override {
    const std::vector<std::int64_t> ids = global_ids();
    auto r = writer.put_field<double>("q_wall", ids, q_, 1, static_cast<std::int64_t>(n_));
    if (!r.ok()) return r;
    r = writer.put_field<double>("u_wall_in", ids, u_in_, 1, static_cast<std::int64_t>(n_));
    if (!r.ok()) return r;
    r = writer.put_scalar("time", time_);
    if (!r.ok()) return r;
    return writer.put_scalar<std::int32_t>("steps", steps_);
  }
  fuse::Result<void> read_restart(fuse::RestartReader& reader) override {
    const std::vector<std::int64_t> ids = global_ids();
    auto q = reader.get_field<double>("q_wall", ids, q_, 1);
    if (!q.ok()) return q.error();
    auto u = reader.get_field<double>("u_wall_in", ids, u_in_, 1);
    if (!u.ok()) return u.error();
    auto t = reader.get_scalar("time", time_);
    if (!t.ok()) return t.error();
    auto s = reader.get_scalar<std::int32_t>("steps", steps_);
    if (!s.ok()) return s.error();
    return fuse::Result<void>::success();
  }

 private:
  [[nodiscard]] std::vector<std::int64_t> global_ids() const {
    std::vector<std::int64_t> ids(n_);   // one rank: positions are the global ids
    for (std::size_t i = 0; i < n_; ++i) ids[i] = static_cast<std::int64_t>(i);
    return ids;
  }
  std::size_t n_;
  std::vector<double> q_, u_in_;
  double time_ = 0.0;
  std::int32_t steps_ = 0;
};
```

Writing a point, at a period boundary (after `run_period()` returns, never mid-period):

```cpp
fuse::Result<void> write_point(fuse::Runner& runner, fuse::ComponentSet& components,
                               const fuse::CouplingGraph& graph, const fuse::Rational& now,
                               std::uint32_t rank_count, const std::string& path) {
  fuse::restart_io::Provenance prov;
  prov.fuse_version = "my-app 1.0";
  prov.descriptor_hash = fuse::restart_descriptor_hash(graph);
  prov.written_by_rank_count = rank_count;      // MPI_Comm_size(); 1 for a serial run
  auto pw = fuse::restart_io::create_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path, prov);
  if (!pw.ok()) return pw.error();
  auto w = fuse::write_restart_point(runner, components, pw.value()->writer(),
                                     now, runner.current_period_index());
  if (!w.ok()) return w;
  return pw.value()->finish();   // the point exists only after this succeeds
}
```

`now` is the time of the boundary, `plan.period() * runner.current_period_index()` for a plan
without a declared end time. `write_restart_point()` records it exactly (as a rational), together
with the period index, every restartable component's state and every accelerated region's
accelerator memory, then calls the writer's `finish_point()`, which checks that no global id was
written twice and none was missed. `finish()` on the `PointWriter` then renames the file into
place. A `PointWriter` destroyed without `finish()` leaves no point behind.

Restoring, after `Runner::initialize()` and before the first `run_period()`/`run()` call:

```cpp
fuse::Result<void> restore_point(fuse::Runner& runner, fuse::ComponentSet& components,
                                 const fuse::CouplingGraph& graph, std::uint32_t rank_count,
                                 const std::string& path, fuse::Rational& time_out,
                                 std::int64_t& period_out) {
  auto pr = fuse::restart_io::open_restart_point(
      fuse::restart_io::FileLayout::SeparateMeshAndPoints, path);
  if (!pr.ok()) return pr.error();
  const fuse::restart_io::Provenance& prov = pr.value()->provenance();
  auto h = fuse::restart_io::check_restart_descriptor_hash(graph, prov);   // same graph?
  if (!h.ok()) return h;
  auto c = fuse::check_restart_rank_count(components, prov.written_by_rank_count, rank_count);
  if (!c.ok()) return c;
  bool any_accelerator_reset = false;
  auto r = fuse::restore_restart_point(runner, components, pr.value()->reader(),
                                       /*allow_accelerator_reset_on_mismatch=*/false,
                                       any_accelerator_reset, time_out, period_out);
  if (!r.ok()) return r;
  if (any_accelerator_reset) { /* report it: the accelerator starts cold */ }
  return fuse::Result<void>::success();
}
```

`open_restart_point()` itself checks the file's format version and descriptor version. Comparing
the descriptor hash and, if you wrote one, `prov.mesh_checksum`, against your current graph and mesh
is the caller's job and belongs before `reader()` is used. After `restore_restart_point()` returns,
`runner.current_period_index()` reads the restored index and the next `run_period()`/`run()`
resumes the Runner's clock from there; `time_out` and `period_out` are the point's own recorded
values. Every call returns a `Result`; check each one.

## Make a component restartable

A component that does nothing is not restartable. The default `write_restart()`/`read_restart()`/
`save_restart_blob()`/`load_restart_blob()` are no-ops, and `write_restart_point()` skips a
component whose `capabilities()` sets neither `kCapabilityRestartFormA` nor
`kCapabilityRestartFormB`. Pick one form (or both) and set the bit.

### Form A: arrays keyed by global id

Set `kCapabilityRestartFormA` and override `write_restart(RestartWriter&)` and
`read_restart(RestartReader&)`. In `write_restart()`, call
`writer.put_field<T>(name, ids, values, components, declared_global_size)` for every array the
component owns, with `ids` the global id of each row, and `put_scalar<T>()`/`put_scalar_rational()`
for scalars. In `read_restart()`, ask for exactly the ids this rank now owns with
`reader.get_field<T>(name, ids, out, components)`. The reader resolves each row through an
id-to-position map built from the file, never from the order the caller requests them in, so the
reading rank can own a different set of ids than the writing rank did. **Form A is what allows
restart at a different rank count.**

:::{important}
The field `name` passed to `put_field()` must equal the field's name in the coupling graph
(`FieldSpec::name`) for any field that an accelerated iteration region's channel feeds. The
accelerator resolves its rows by (owning component name, that exact field name); a private label
makes the accelerator's rows unresolvable (`E-RESTART-MISSING-FIELD`).
`examples/dn_heat/dn_component.hpp` is a worked `write_restart()`.
:::

`write_restart()` must not mutate the component: on a restore the Runner calls it a second time,
through an id-collecting writer, purely to learn the component's current global ids before
`read_restart()` is called for real.

### Form B: an opaque byte record

Set `kCapabilityRestartFormB` and override `save_restart_blob()`, `load_restart_blob()` and
`restart_blob_version()`:

```cpp
class Box final : public fuse::Component {
  // name(), register_fields(), advance() as usual ...
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::kCapabilityRestartFormB;
  }
  [[nodiscard]] std::uint32_t restart_blob_version() const override { return 1; }
  fuse::Result<std::vector<std::byte>> save_restart_blob() override {
    std::vector<std::byte> bytes(state_.size() * sizeof(double));
    std::memcpy(bytes.data(), state_.data(), bytes.size());
    return bytes;
  }
  fuse::Result<void> load_restart_blob(std::span<const std::byte> bytes,
                                       std::uint32_t written_by_version) override {
    if (written_by_version != 1) {
      return fuse::Failure{"E-BOX-BLOB-VERSION", "unknown blob version"};
    }
    if (bytes.size() != state_.size() * sizeof(double)) {
      return fuse::Failure{"E-BOX-BLOB-SIZE", "blob size does not match this decomposition"};
    }
    std::memcpy(state_.data(), bytes.data(), bytes.size());
    return fuse::Result<void>::success();
  }
  std::vector<double> state_ = std::vector<double>(16, 0.0);
};
```

Form B is `memcpy`-cheap to implement and needs no mesh, but the bytes mean something only on the
decomposition that wrote them. A restart at a different rank count is therefore refused,
collectively, if any component in the graph sets form B without also setting form A
(`E-RESTART-FORMB-ONLY-RANK-CHANGE`, from `fuse::check_restart_rank_count()`). The version your
`restart_blob_version()` returned at write time comes back as `written_by_version`, so an
implementation can refuse bytes it no longer understands by name.

| | Form A | Form B |
|---|---|---|
| Capability bit | `kCapabilityRestartFormA` | `kCapabilityRestartFormB` |
| Hooks | `write_restart()`, `read_restart()` | `save_restart_blob()`, `load_restart_blob()`, `restart_blob_version()` |
| What is stored | typed arrays keyed by global id, typed scalars | one opaque byte record per rank |
| Restart at another rank count | yes | no |
| Checked by the library | every id written once, none missing; element types on read | blob version handed back to you |

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
writer.put_field<std::int32_t>("dirichlet.sweep_count", ids, sweep_counts, 1, global_size);
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

Accelerator and waveform histories are stored as `double` (binary64). A `Rational` time level is
stored exactly, as its numerator and denominator (`put_scalar_rational()`/`get_scalar_rational()`),
never as a `double`.

## The two file layouts

- `FileLayout::SeparateMeshAndPoints`: one file per restart point, created fresh and atomically
  renamed into place on `finish()`.
- `FileLayout::GrowingFile`: one file that grows by copy-append-rename per point. A crash during a
  write can never damage an earlier point, at the cost of one file copy per point. `point_label`
  names the new point's group and is required on both `create_restart_point()` and
  `open_restart_point()` for this layout (`E-RESTART-USAGE` if it is missing):

```cpp
auto pw = fuse::restart_io::create_restart_point(
    fuse::restart_io::FileLayout::GrowingFile, "run.h5", prov, "period-0003");
auto pr = fuse::restart_io::open_restart_point(
    fuse::restart_io::FileLayout::GrowingFile, "run.h5", "period-0003");
```

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
| `E-RESTART-DUPLICATE-ID`, `E-RESTART-BAD-NUMBERING` | the batched or per-call id check on write failed: an id written twice, or a numbering that does not cover `declared_global_size` |
| `E-RESTART-DUPLICATE-CALL` | `put_field()` called twice under one name for the same point |
| `E-RESTART-FIELD-SHAPE` | `values.size()` is not `ids.size() * components` |
| `E-RESTART-AMBIGUOUS-FIELD-NAME` | two components in the graph declare a field of the same name, so an accelerator row cannot be attributed |
| `E-RESTART-USAGE` | `GrowingFile` used without a `point_label` |
| `E-RESTART-TRUNCATED` | not a valid HDF5 file, or missing `@format_version` |
| `E-RESTART-HDF5` | an HDF5 call failed; the message carries the call |
| `E-RESTART-ACCEL-NOT-IMPLEMENTED` | the region's accelerator kind has no restart state (see below) |
| `E-RESTART-TYPE-MISMATCH` | `Exact` found an element-type difference, or `Widening` was asked for a conversion not proven exact (a float narrowing or any signed-to-unsigned conversion is never widening) |
| `E-RESTART-TYPE-METADATA-MISMATCH` | the file's per-field attributes (`element_type`, `byte_size`, `significand_bits`, `exponent_bits`, `exponent_bias`, `byte_order`, `is_signed`) disagree with the file's own HDF5 datatype |
| `E-RESTART-UNKNOWN-ELEMENT-TYPE` | an `element_type` name this build does not recognise |
| `E-RESTART-BAD-BOOL-BYTE` | a `Bool` field's stored byte is neither `0` nor `1` |
| `E-RESTART-INTEGER-OVERFLOW` | an integer narrowing does not fit the target type; refused even under `AllowNarrowing` |
| `E-RESTART-SIGNED-TO-UNSIGNED` | refused under every `ReadConversion` mode |
| `E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED`, `E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED`, `E-RESTART-GLOBAL-SIZE-MISMATCH` | collective-writer limits, below |

The reader hands back the provenance recorded in the file (`format_version`, `fuse_version`,
`descriptor_version`, `descriptor_hash`, `written_by_rank_count`, `mesh_checksum`).

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

`fuse::restart_io::create_restart_point_collective(layout, path, provenance, comm_f, point_label)`
(`include/fuse/restart_io/hdf5_restart_io_mpi.hpp`, available with `FUSE_ENABLE_MPI`) lets every rank
write its own rows of a field at an `MPI_Exscan`-derived offset in one collective hyperslab write.
It takes the communicator as a Fortran handle (`int comm_f`, from `MPI_Comm_c2f`), and every rank
must call `finish()` together. Limits:

- Only `ElementType::Binary64` and `Int64` fields; other types refuse with
  `E-RESTART-TYPE-MISMATCH`.
- Only `FileLayout::SeparateMeshAndPoints`; `GrowingFile` refuses with
  `E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED`.
- Scalars, rationals and blobs refuse with `E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED`.
- Every rank must pass the same `declared_global_size` for a field
  (`E-RESTART-GLOBAL-SIZE-MISMATCH`).

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

An externally driven (participant) code asks its embedding host for a restart point with the
participant table's `request_restart_point` entry (`ffuse_request_restart_point` in Fortran). That
entry sets a flag and writes no file, because the ABI's fixed `(ctx, in, out)` shape cannot carry a
file path or layout. The embedding host polls `fuse::Runner::take_restart_point_request()` after
each `run_period()`/`resume_period()` call and performs the write itself:

```cpp
for (int p = 0; p < periods; ++p) {
  if (!runner.run_period(components).ok()) break;
  if (runner.take_restart_point_request()) { /* write_point(...) as above */ }
}
```

`take_restart_point_request()` consumes the flag. Repeated requests before the host polls leave
exactly one pending request, and a request made before `initialize()` has no effect. See
[](c-abi.md).

## Limitations

- Restart points are written and restored at period boundaries only.
- Restart at a different rank count requires Form A on every component.
- Collective writing is limited as described above.
- Only the HDF5 backend is provided; `RestartWriter`/`RestartReader` are abstract, so another
  file format means implementing that pair yourself.

## See also

- [](build.md): `FUSE_ENABLE_HDF5` and `FUSE_ENABLE_MPI`.
- [](c-abi.md) and [](providers.md): restart through the ABIs.
- [](component-interface.md): the full `Component` interface.
- [](glossary.md): element types and rollback checkpoints.
- [](troubleshooting.md): diagnosing `E-...` codes.
