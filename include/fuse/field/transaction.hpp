#pragma once

/// Staged, all-or-nothing field updates over the existing `fuse::FieldRef`/
/// `fuse::FieldRegistry` (field/field_ref.hpp, field/registry.hpp).
///
/// Semantics copied from KARMA's own `karma_field_transaction.h`
/// (upstream/karma/include/coupling/karma_field_transaction.h), the gap
/// `docs/dev/field-semantics-matrix.md` Section 2.2 names: the invariant is
/// that no caller-visible value changes until every staged value has passed
/// local validation, and in the collective form until every rank has agreed
/// (karma_field_transaction.h:11-23). Staging writes into transaction-private
/// memory, so a refused or abandoned update cannot leave a field
/// half-written.
///
/// NUMTYPES SEAM. Every value in this file is `double` -- the one scalar
/// type `fuse::FieldRef`/`fuse::FieldSpan` carry today (field/field_ref.hpp,
/// field/pipeline.hpp). The numtypes track (docs/dev/physics-kit-
/// disposition.md's D-E, landing in parallel) is expected to template
/// `FieldRef` into a `BasicFieldRef<T>`; WHEN IT DOES, the template
/// parameter goes HERE -- `FieldStorage`, `FieldRefStorage` and
/// `FieldTransaction` below all become `template <class Value = double>`,
/// and every `double` in this file becomes `Value`. Built over `double` now,
/// deliberately: there is no `BasicFieldRef<T>` to templatize against yet
/// (field-semantics-matrix.md Section 3), and templating ahead of that
/// landing would guess at a shape this track has no evidence for. The one
/// place this will NOT transfer unchanged is the canonical accumulation
/// order below (`Accumulate` mode): it is defined on a `double`'s own
/// magnitude, sign, and (as a final tie-break) raw bit pattern, none of
/// which has a single meaning for a value type carrying multiple components
/// (e.g. a forward-AD `dual<S,N>`'s tangent slots -- which component's
/// magnitude would order the whole value?) -- see that section's own
/// comment, src/field/transaction.cpp.
///
/// LAYERING (tools/check-layering.py, docs/dev/layering.md). `field` sits
/// below `transport`, where this project's own layering doc says MPI
/// enters, and `field`'s PUBLIC headers may name only `Kokkos_Core.hpp`
/// (ALLOWED_PUBLIC_THIRD_PARTY["field"]) -- `mpi.h` is in FORBIDDEN_PUBLIC at
/// every layer. `commit_collective()` below therefore takes a Fortran-style
/// communicator handle (`std::optional<int>`, the same type `fuse::
/// RunnerConfig::base_communicator`, include/fuse/runtime/runner.hpp, already
/// uses for exactly this reason), never a raw `MPI_Comm`; the actual
/// `MPI_Allreduce` lives in `src/field/transaction.cpp`, behind `#if
/// FUSE_HAVE_MPI`, in the same shape `reconcile_plugin_agreement()`
/// (src/abi/participant.cpp) already uses for a different collective
/// agreement. `std::nullopt` means single process: nothing to reconcile --
/// the same convention `reconcile_plugin_agreement()` uses for
/// `!c.communicator_set`.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"
#include "fuse/field/field_ref.hpp"

namespace fuse {

/// Three update modes, each with its own ambiguity rule -- copied from
/// KARMA's own `UpdateMode` (karma_field_transaction.h:26-37), renamed
/// `SparseReplace` to `Insert` per the owner's 2026-10-28 decision.
enum class UpdateMode {
  /// Every value must be staged exactly once; a second value for one slot
  /// is ambiguous and refused rather than resolved by arrival order.
  Replace,
  /// Staged slots replace the published value and unstaged slots keep it; a
  /// second value for one slot is still ambiguous and refused. This is the
  /// shape of a sparse insertion into an existing field.
  Insert,
  /// Contributions add to the published value; slots may be left unstaged.
  Accumulate,
};

/// Publication target. An implementation must publish all values or none.
/// See the file comment's NUMTYPES SEAM note for where a value-type template
/// parameter would replace every `double` here.
class FieldStorage {
 public:
  virtual ~FieldStorage() = default;

  [[nodiscard]] virtual std::size_t value_count() const = 0;

  /// Snapshot the currently published values, for accumulation bases and for
  /// `FieldTransaction::restore_committed()`.
  [[nodiscard]] virtual Result<void> read_into(std::span<double> destination) const = 0;

  /// Publish all values atomically from the caller's perspective: either
  /// every slot in `values` reaches the backing storage, or (on a `Failure`)
  /// none of them do.
  [[nodiscard]] virtual Result<void> publish(std::span<const double> values) = 0;
};

/// Publishes into a caller-owned `fuse::FieldRef` (field/field_ref.hpp), in
/// entry-major, component-minor order (`value_count() == layout().count *
/// layout().components`, slot `entry * components + component`) -- the same
/// dense-index convention `fuse::FieldSpan`'s own `at(i, c)` uses.
///
/// `FieldRef::at()` is bounds-checked per element (field/field_ref.hpp:154-
/// 177); this class relies on its OWN `value_count()`-vs-`destination.size()`
/// check instead of paying that per-element cost a second time on every
/// value of a full-field read or publish.
class FieldRefStorage final : public FieldStorage {
 public:
  explicit FieldRefStorage(FieldRef ref) noexcept : ref_(ref) {}

  [[nodiscard]] std::size_t value_count() const override {
    return ref_.layout().count * static_cast<std::size_t>(ref_.layout().components);
  }

  [[nodiscard]] Result<void> read_into(std::span<double> destination) const override;
  [[nodiscard]] Result<void> publish(std::span<const double> values) override;

 private:
  FieldRef ref_;
};

class FieldTransaction {
 public:
  /// `entries`/`components` describe the SAME shape `storage`'s own
  /// `value_count()` must equal (`entries * components`); a mismatch is
  /// reported by `validate_local()`/`commit()`, not by the constructor,
  /// which never fails (matching `FieldStorage`'s own contract that sizing
  /// disagreements are reported, not thrown, since a transaction may be
  /// constructed before `storage` is fully set up in a staged initialisation
  /// sequence).
  FieldTransaction(FieldStorage& storage, std::size_t entries, int components, UpdateMode mode);

  /// Dense-index staging: the hot path. Performs no key lookup -- KARMA's
  /// own `stage_by_key()` (karma_field_transaction.h:97-98, over a
  /// `FieldIndexMap`) has no counterpart here, because there is no stable
  /// per-entry key below `fuse::FieldRef`'s own dense index yet
  /// (field-semantics-matrix.md Section 2.1, gap 3, DEFERRED per the
  /// owner's 2026-10-28 decision) -- every caller of this transaction
  /// addresses a slot by its dense `(entry, component)` position.
  [[nodiscard]] Result<void> stage(std::size_t entry, int component, double value) noexcept;

  /// Local, total, and side-effect free.
  [[nodiscard]] Result<void> validate_local() const noexcept;

  /// Validates, then publishes; on any failure nothing is published and no
  /// caller-visible value changes.
  [[nodiscard]] Result<void> commit() noexcept;

  /// Discards staged state. Infallible by construction: it touches only
  /// transaction-private memory, which is why it is separate from
  /// `restore_committed()`.
  void rollback() noexcept;

  /// Republishes the snapshot taken before the last successful commit. This
  /// CAN fail -- it is a publication -- which is exactly why discarding
  /// staged state is a different operation.
  [[nodiscard]] Result<void> restore_committed() noexcept;

  /// Validate on every rank, reconcile, and publish only if every rank
  /// agreed. No rank publishes when any rank's validation failed. See the
  /// file comment's LAYERING note for why `communicator` is a Fortran handle
  /// and not an `MPI_Comm`.
  ///
  /// Every rank of `communicator` must call this. Every rank's RETURNED
  /// STATUS is identical -- the same code and the same message, naming the
  /// one rank whose local condition caused the refusal -- whether validation
  /// disagreed (nothing published anywhere) or publication disagreed after
  /// a successful validation agreement (every rank but the failing one has
  /// already published by the time that is discovered, the one case this
  /// API cannot make atomic, the same limitation KARMA's own
  /// `commit_collective()` documents, karma_field_transaction.h:117-124);
  /// in that second case the caller must restore. This is a DIFFERENT
  /// convention from `reconcile_plugin_agreement()`'s (src/abi/
  /// participant.cpp), which lets the originating rank's own raw message
  /// stand and rewrites it only for peers -- a deliberate choice for THIS
  /// collective (round 2, S8 Stage B, the lead's own instruction): a caller
  /// comparing every rank's return value sees one answer, not a mix of a
  /// detailed local diagnostic on one rank and a generic one everywhere
  /// else.
  [[nodiscard]] Result<void> commit_collective(std::optional<int> communicator) noexcept;

  [[nodiscard]] std::size_t staged_count() const noexcept;
  [[nodiscard]] std::size_t entries() const noexcept { return entries_; }
  [[nodiscard]] int components() const noexcept { return components_; }
  [[nodiscard]] UpdateMode mode() const noexcept { return mode_; }

 private:
  [[nodiscard]] Result<void> build_published_values(std::span<double> destination) const;

  FieldStorage* storage_ = nullptr;
  std::size_t entries_ = 0;
  int components_ = 0;
  UpdateMode mode_ = UpdateMode::Replace;

  /// Replacement/insertion staging: one value per slot plus a staged flag.
  std::vector<double> staged_;
  std::vector<bool> has_staged_;
  /// Accumulation staging: contributions kept until commit canonicalises
  /// them. See `build_published_values()`'s own comment (src/field/
  /// transaction.cpp) for the exact total order and what it is, and is not,
  /// a guarantee of.
  std::vector<std::pair<std::size_t, double>> contributions_;
  /// Snapshot of the values published before the last successful commit.
  std::vector<double> previous_;
  bool has_previous_ = false;
};

}  // namespace fuse
