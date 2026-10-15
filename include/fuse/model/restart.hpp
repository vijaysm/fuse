#pragma once

/// Restart-to-disk writer/reader interfaces (docs/dev/restart.md, Stage B
/// Slice 1). Abstract and backend-agnostic: no HDF5 type appears here.
///
/// WHY fuse::model, NOT fuse::runtime. `tools/check-layering.py` ranks
/// `model` below both `runtime` and `num` (docs/dev/layering.md). A
/// restart point writes BOTH a component's own field state
/// (`fuse::Component::write_restart()`, fuse/runtime/component.hpp, layer
/// `runtime`) AND an iteration region's accelerator memory
/// (`fuse::Accelerator::export_restart_state()`, fuse/num/accelerator.hpp,
/// layer `num`) through the SAME writer object, keyed the same way (owner
/// decision 4: accelerator rows are keyed by the OWNING component's own
/// global ids, so the accelerator and the component it accelerates must
/// agree on what a "row" means). `num` cannot reach up into `runtime` for
/// a shared type, and `runtime` cannot reach down and redefine the type
/// `num` already uses -- the only placement that serves both without a
/// layering violation is a layer BELOW both, which is `model`. The
/// concrete HDF5-backed implementation (Hdf5RestartWriter/Hdf5RestartReader)
/// lives ABOVE `runtime` instead, in its own library gated by
/// FUSE_ENABLE_HDF5 -- see docs/dev/restart.md Section 4 and Section 8.
///
/// NAME QUALIFICATION. Every name below is the CALLER's responsibility to
/// make unique within one restart point; this header does no qualifying
/// of its own. `fuse::runtime`'s orchestration (`write_restart_point()`,
/// fuse/runtime/restart.hpp) wraps the real writer in a small per-component
/// decorator that prefixes a component's own bare field name with that
/// component's own graph name before any call reaches here, so a
/// `Component::write_restart()` override never has to think about
/// collisions with another component's field of the same name -- see that
/// header's own `QualifyingRestartWriter`.
///
/// OWNER DECISION 2 (batched global-id verification). `put_field()` is a
/// LOCAL, per-rank, per-field call: it checks only that `ids` has no
/// duplicate WITHIN this one call (free, and catches an authoring mistake
/// before anything leaves this rank) and stages the data. The verification
/// the owner asked for -- "uniqueness and coverage of the declared global
/// size for ALL arrays of a restart point in ONE collective (one sort and
/// reduction over every array's ids together), never one collective per
/// array" -- happens once, in `finish_point()`, after every component (and
/// every accelerator) for this restart point has made its `put_field()`
/// calls: the writer concatenates every field's own `(field, id)` pairs
/// into ONE combined buffer, sorts it ONCE, and checks per field that (a)
/// no id repeats and (b) the number of ids received for that field,
/// summed across whatever this writer's own scope spans, equals the
/// `declared_global_size` every caller for that field reported (which
/// must therefore agree across every caller of the SAME field -- `finish_
/// point()` refuses, by name, if two calls for the same field declared
/// different global sizes). At a single rank (Slice 1, no MPI in this
/// sandbox) this degenerates to a purely local sort with no communication;
/// Slice 2 extends `finish_point()`'s own combined sort into one collective
/// (a single `Allgatherv` over the SAME combined buffer, not N separate
/// ones per field) rather than replacing it, so the tier-0 code path this
/// slice tests is not thrown away when MPI is added -- see docs/dev/
/// restart.md Section 7(iv)'s own tier split.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fuse/base/bytes.hpp"
#include "fuse/base/rational.hpp"
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/model/descriptor.hpp"

namespace fuse {

/// The write side of one restart point. One instance is used for exactly
/// one restart point (mesh-once data goes through a SEPARATE path -- see
/// fuse/runtime/restart.hpp's `write_mesh_once()`), across every
/// component and every accelerated region the Runner visits, in the
/// order `write_restart_point()` visits them; `finish_point()` is the
/// LAST call made, after every other call for this point has returned
/// successfully.
/// The canonical provenance hash a restart point's own `@descriptor_hash`
/// attribute is checked against (docs/dev/restart.md Section 4.1):
/// `fnv1a(serialize(graph))`, the SAME primitive three other call sites
/// already hash their own serialized bytes with (`src/backend/
/// weights_operator.cpp`, `src/model/semantics.cpp`, `src/model/
/// transform.cpp`) -- reused here rather than a new hash invented for
/// restart, and gated by `fuse::deserialize()`'s own `kGraphVersion` check
/// (fuse/model/descriptor.hpp), which already refuses a version mismatch
/// before this hash would even be compared.
///
/// CAVEAT (KNOWN_ISSUES.md D62, OPEN): `fuse::serialize()`'s own
/// `Writer::pod()` is host-endian, not little-endian like the rest of
/// `fuse/base/bytes.hpp` -- this hash is therefore only comparable
/// between hosts of the same endianness, a limitation this function
/// inherits rather than fixes (D62 is owned elsewhere).
[[nodiscard]] inline std::uint64_t restart_descriptor_hash(
    const CouplingGraph& graph) {
  return bytes::fnv1a(serialize(graph));
}

/// One named sub-range of an accelerated region's stacked interface
/// vector (docs/dev/restart.md, Stage B Slice 1 round 2, owner decision 4's
/// EXACT-KEY fix). `field_name` is the OWNING component's own qualified
/// field name -- already unique within one graph: `fuse::Runner`'s own
/// `build_field_owner_map()` (src/runtime/runner.cpp) refuses
/// E-RESTART-AMBIGUOUS-FIELD-NAME before any accelerator call is ever
/// reached if two components declared the same field name, so no two
/// `RestartRowBlock`s passed to the SAME `Accelerator::export_restart_state()`/
/// `import_restart_state()` call ever share a `field_name`. `ids` is that
/// field's own local/global ids, in the SAME order as the corresponding
/// slice of the accelerator's own flat (stacked) buffer -- never combined,
/// hashed, or otherwise mixed with another block's ids into one shared id
/// space. Each block's row data is written under a dataset path qualified
/// by `field_name` itself (`name_prefix + "/" + field_name`), so two
/// blocks' rows can never collide: they are not even candidates for the
/// same key, because they are not written under the same name. This
/// replaces round 1's `combine_field_and_local_id()` (a hash of
/// `field_name` folded into the id's high bits, flagged there as a
/// bridging measure, not collision-proof) with a construction that cannot
/// collide by definition, not merely "collision-free for every graph this
/// project builds so far".
struct RestartRowBlock {
  std::string field_name;
  std::vector<std::int64_t> ids;
};

class RestartWriter {
 public:
  virtual ~RestartWriter() = default;

  /// `ids.size() * components == values.size()`, `ids` carry no duplicate
  /// within THIS call (checked here, locally; the cross-caller duplicate
  /// and coverage check is `finish_point()`'s, not this call's).
  /// `declared_global_size` is the caller's own claim of how many entries
  /// this field has IN TOTAL across every caller that will write to
  /// `qualified_name` for this restart point (at a single rank, simply
  /// `static_cast<std::int64_t>(ids.size())`); every caller writing the
  /// SAME `qualified_name` must declare the SAME value, or `finish_point()`
  /// refuses by name (E-RESTART-GLOBAL-SIZE-MISMATCH).
  virtual Result<void> put_field(std::string_view qualified_name,
                                 std::span<const std::int64_t> ids,
                                 std::span<const double> values,
                                 int components,
                                 std::int64_t declared_global_size) = 0;

  /// A value with no spatial index -- replicated, written once per restart
  /// point. First writer for a given name wins; a SECOND call for the
  /// same name with a DIFFERENT value is `Failure` (E-RESTART-SCALAR-
  /// DISAGREE), never silently overwritten, since "replicated" means every
  /// caller agrees, not "whoever wrote last".
  virtual Result<void> put_scalar(std::string_view qualified_name,
                                  double value) = 0;

  /// Exact: the Rational's own `{numerator, denominator}` pair
  /// (`fuse::ExactTimeRecord`, fuse/base/rational.hpp), never only the
  /// lossy `to_double()` -- docs/dev/restart.md Section 4.1's own
  /// `@time_numerator`/`@time_denominator` vs `@time_double` split.
  virtual Result<void> put_scalar_rational(std::string_view qualified_name,
                                           const Rational& value) = 0;

  /// Form B (owner decision 1): an opaque, versioned byte record,
  /// same-decomposition only. `version` is the writer's own declared
  /// `restart_blob_version()` (fuse::Component, fuse/runtime/
  /// component.hpp), recorded so a reader can refuse an incompatible
  /// blob by name rather than by a corrupted read -- the same posture
  /// `kGraphVersion` (fuse/model/descriptor.hpp) already takes for the
  /// coupling graph.
  virtual Result<void> put_blob(std::string_view qualified_name,
                                std::span<const std::byte> bytes,
                                std::uint32_t version) = 0;

  /// Owner decision 2's own batched check -- see this file's header
  /// comment. Called exactly once, after every `put_field()`/`put_scalar()`
  /// `put_scalar_rational()`/`put_blob()` call for this restart point has
  /// returned successfully; refuses (collectively, when Slice 2 adds MPI)
  /// on a duplicate id within one field across callers, a global-size
  /// disagreement, or (E-RESTART-BAD-NUMBERING) a field whose received
  /// count never reaches its declared global size.
  virtual Result<void> finish_point() = 0;

  /// The ids a PRIOR `put_field()` call on THIS writer recorded for
  /// `qualified_name`, or `Failure` if nothing was ever recorded under
  /// that name. Exists for exactly one caller: fuse/runtime/restart.hpp's
  /// orchestration, which resolves an iteration region's accelerator rows
  /// to the SAME global ids the owning component just reported through
  /// its own `write_restart()` call, earlier in the SAME restart point --
  /// never for a `RestartWriter` implementer's own use, and never across
  /// restart points (a fresh `RestartWriter` instance is used per point).
  virtual Result<std::vector<std::int64_t>> recorded_ids(
      std::string_view qualified_name) const = 0;
};

/// The read side of `RestartWriter`. `get_field()` asks for exactly the
/// rows the CALLER's own ids name -- the reader resolves them against
/// whatever the file actually holds, which may have been written at a
/// DIFFERENT rank count (owner decision 1's whole point: Form A is
/// re-partitionable). A requested id absent from the file is `Failure`
/// (E-RESTART-MISSING-ID), never a default-filled row, because a
/// silently-zero row is numerically indistinguishable from a
/// legitimately-zero one.
class RestartReader {
 public:
  virtual ~RestartReader() = default;

  virtual Result<void> get_field(std::string_view qualified_name,
                                 std::span<const std::int64_t> ids,
                                 std::span<double> out, int components) = 0;
  virtual Result<double> get_scalar(std::string_view qualified_name) = 0;
  virtual Result<Rational> get_scalar_rational(std::string_view qualified_name) = 0;

  /// `version` receives the writer's own declared `restart_blob_version()`.
  virtual Result<std::vector<std::byte>> get_blob(std::string_view qualified_name,
                                                  std::uint32_t& version) = 0;

  /// Whether a field named `qualified_name` has ANY row for `id` at all --
  /// used by the accelerator-layout-hash refusal (docs/dev/restart.md
  /// Section 4.1) to decide, BEFORE calling `get_field()`, whether every
  /// row the current layout needs is actually present; `get_field()`
  /// itself refuses on a missing id rather than silently skip it, so a
  /// caller that may legitimately not find one (the accelerator path,
  /// when the owner's reset-on-mismatch option is set) checks first.
  virtual Result<bool> has_id(std::string_view qualified_name,
                             std::int64_t id) = 0;
};


/// Prefixes every name this writer sees with `prefix + "/"` before
/// forwarding to `inner`, so a `Component::write_restart()` override (or
/// an `Accelerator::export_restart_state()` call) can use its own bare
/// field names without colliding with another component's or region's
/// field of the same name in the SAME restart point. `fuse::runtime`'s
/// orchestration (fuse/runtime/restart.hpp) constructs one of these per
/// component (`prefix` = the component's own graph name) and per
/// accelerated region (`prefix` = the region's own name), never
/// constructed by a `Component`/`Accelerator` override itself.
/// `finish_point()` is refused here deliberately: it must be called
/// exactly once, on the TOP-LEVEL writer, after every qualified call for
/// the whole restart point has returned -- calling it through a
/// per-component decorator would run it once per component instead of
/// once per point, silently wrong in a way a caller could not see by
/// reading the per-component code they wrote.
class QualifyingRestartWriter final : public RestartWriter {
 public:
  QualifyingRestartWriter(RestartWriter& inner, std::string prefix)
      : inner_(inner), prefix_(std::move(prefix)) {}

  Result<void> put_field(std::string_view name, std::span<const std::int64_t> ids,
                         std::span<const double> values, int components,
                         std::int64_t declared_global_size) override {
    return inner_.put_field(qualify(name), ids, values, components,
                            declared_global_size);
  }
  Result<void> put_scalar(std::string_view name, double value) override {
    return inner_.put_scalar(qualify(name), value);
  }
  Result<void> put_scalar_rational(std::string_view name,
                                   const Rational& value) override {
    return inner_.put_scalar_rational(qualify(name), value);
  }
  Result<void> put_blob(std::string_view name, std::span<const std::byte> bytes,
                        std::uint32_t version) override {
    return inner_.put_blob(qualify(name), bytes, version);
  }
  Result<void> finish_point() override {
    return Failure{"E-RESTART-INTERNAL",
                   "QualifyingRestartWriter::finish_point(): must be called "
                   "on the top-level writer exactly once per restart point, "
                   "never through a per-component/per-region decorator"};
  }
  Result<std::vector<std::int64_t>> recorded_ids(
      std::string_view name) const override {
    return inner_.recorded_ids(qualify(name));
  }

 private:
  [[nodiscard]] std::string qualify(std::string_view name) const {
    return prefix_ + "/" + std::string(name);
  }
  RestartWriter& inner_;
  std::string    prefix_;
};

/// The read-side counterpart of `QualifyingRestartWriter`.
class QualifyingRestartReader final : public RestartReader {
 public:
  QualifyingRestartReader(RestartReader& inner, std::string prefix)
      : inner_(inner), prefix_(std::move(prefix)) {}

  Result<void> get_field(std::string_view name, std::span<const std::int64_t> ids,
                         std::span<double> out, int components) override {
    return inner_.get_field(qualify(name), ids, out, components);
  }
  Result<double> get_scalar(std::string_view name) override {
    return inner_.get_scalar(qualify(name));
  }
  Result<Rational> get_scalar_rational(std::string_view name) override {
    return inner_.get_scalar_rational(qualify(name));
  }
  Result<std::vector<std::byte>> get_blob(std::string_view name,
                                          std::uint32_t& version) override {
    return inner_.get_blob(qualify(name), version);
  }
  Result<bool> has_id(std::string_view name, std::int64_t id) override {
    return inner_.has_id(qualify(name), id);
  }

 private:
  [[nodiscard]] std::string qualify(std::string_view name) const {
    return prefix_ + "/" + std::string(name);
  }
  RestartReader& inner_;
  std::string    prefix_;
};

/// A `RestartWriter` that performs no I/O at all: `put_field()` records
/// `(qualified_name -> ids)` and returns success; every other call is a
/// successful no-op. `fuse::runtime`'s restore orchestration (fuse/
/// runtime/restart.hpp) runs every Form-A component's OWN `write_restart()`
/// through one of these, wrapped in a fresh `QualifyingRestartWriter`
/// scoped to that component, BEFORE restoring anything for real -- this
/// is how it learns, on a RESTORE, the exact global ids an accelerated
/// region's rows must be matched against, without going through the
/// restore file at all (those ids are a property of the CURRENT run's
/// own mesh/numbering, not of what a previous run wrote). Calling the
/// component's own `write_restart()` here therefore must not mutate
/// component state -- the same purity `write_restart()`'s own doc
/// comment (fuse/runtime/component.hpp) requires of every override,
/// restated here because THIS is the call site a violation would be
/// invisible at (a real write never needs the component's state to stay
/// put afterward; a restore immediately re-reads it).
class IdCollectingRestartWriter final : public RestartWriter {
 public:
  Result<void> put_field(std::string_view name, std::span<const std::int64_t> ids,
                         std::span<const double> /*values*/, int /*components*/,
                         std::int64_t /*declared_global_size*/) override {
    ids_by_name_[std::string(name)] =
        std::vector<std::int64_t>(ids.begin(), ids.end());
    return Result<void>::success();
  }
  Result<void> put_scalar(std::string_view, double) override {
    return Result<void>::success();
  }
  Result<void> put_scalar_rational(std::string_view, const Rational&) override {
    return Result<void>::success();
  }
  Result<void> put_blob(std::string_view, std::span<const std::byte>,
                        std::uint32_t) override {
    return Result<void>::success();
  }
  Result<void> finish_point() override { return Result<void>::success(); }
  Result<std::vector<std::int64_t>> recorded_ids(
      std::string_view name) const override {
    const auto it = ids_by_name_.find(std::string(name));
    if (it == ids_by_name_.end()) {
      return Failure{"E-RESTART-MISSING-FIELD",
                     "IdCollectingRestartWriter::recorded_ids: no put_field() "
                     "call was ever made for '" +
                         std::string(name) + "'"};
    }
    return it->second;
  }

 private:
  std::unordered_map<std::string, std::vector<std::int64_t>> ids_by_name_;
};

}  // namespace fuse
