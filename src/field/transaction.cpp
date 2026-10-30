#include "fuse/field/transaction.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

// Explicit, not transitive: the feature macro FUSE_HAVE_MPI is read below
// (docs/dev/invariants.md; tools/check-invariants.py enforces this).
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse {

// ---------------------------------------------------------------------
// FieldRefStorage
// ---------------------------------------------------------------------

Result<void> FieldRefStorage::read_into(std::span<double> destination) const {
  if (destination.size() != value_count()) {
    return Failure{"E-FIELD-TXN-SIZE",
                   "FieldRefStorage::read_into: destination size does not match the field"};
  }
  const Layout& layout = ref_.layout();
  std::size_t slot = 0;
  for (std::size_t entry = 0; entry != layout.count; ++entry) {
    for (int component = 0; component != layout.components; ++component) {
      destination[slot++] = ref_.at(entry, component);
    }
  }
  return Result<void>::success();
}

Result<void> FieldRefStorage::publish(std::span<const double> values) {
  if (values.size() != value_count()) {
    return Failure{"E-FIELD-TXN-SIZE",
                   "FieldRefStorage::publish: value count does not match the field"};
  }
  const Layout& layout = ref_.layout();
  std::size_t slot = 0;
  for (std::size_t entry = 0; entry != layout.count; ++entry) {
    for (int component = 0; component != layout.components; ++component) {
      ref_.at(entry, component) = values[slot++];
    }
  }
  return Result<void>::success();
}

// ---------------------------------------------------------------------
// FieldTransaction
// ---------------------------------------------------------------------

FieldTransaction::FieldTransaction(FieldStorage& storage, std::size_t entries, int components,
                                   UpdateMode mode)
    : storage_(&storage), entries_(entries), components_(components), mode_(mode) {
  const std::size_t values = entries_ * static_cast<std::size_t>(components_);
  if (mode_ != UpdateMode::Accumulate) {
    staged_.assign(values, 0.0);
    has_staged_.assign(values, false);
  }
}

Result<void> FieldTransaction::stage(std::size_t entry, int component, double value) noexcept {
  if (!std::isfinite(value)) {
    return Failure{"E-FIELD-TXN-NONFINITE",
                   "FieldTransaction::stage: a nonfinite value was staged"};
  }
  if (entry >= entries_ || component < 0 || component >= components_) {
    return Failure{"E-FIELD-TXN-UNKNOWN-ENTRY",
                   "FieldTransaction::stage: entry or component is out of range"};
  }
  const std::size_t slot = entry * static_cast<std::size_t>(components_) +
                            static_cast<std::size_t>(component);
  if (mode_ == UpdateMode::Accumulate) {
    contributions_.emplace_back(slot, value);
    return Result<void>::success();
  }
  if (has_staged_[slot]) {
    return Failure{"E-FIELD-TXN-DUPLICATE",
                   "FieldTransaction::stage: a second value for one slot is ambiguous under "
                   "Replace/Insert and is refused rather than resolved by arrival order"};
  }
  staged_[slot] = value;
  has_staged_[slot] = true;
  return Result<void>::success();
}

Result<void> FieldTransaction::validate_local() const noexcept {
  if (storage_ == nullptr) {
    return Failure{"E-FIELD-TXN-SIZE", "FieldTransaction::validate_local: no storage bound"};
  }
  const std::size_t values = entries_ * static_cast<std::size_t>(components_);
  if (storage_->value_count() != values) {
    return Failure{"E-FIELD-TXN-SIZE",
                   "FieldTransaction::validate_local: storage size disagrees with the "
                   "transaction's own entries*components"};
  }
  if (mode_ == UpdateMode::Replace) {
    // Replacement must cover every value. A partial replacement would
    // publish a mixture of this step's values and the previous step's,
    // which is the failure mode this check exists to prevent.
    for (const bool staged : has_staged_) {
      if (!staged) {
        return Failure{"E-FIELD-TXN-INCOMPLETE",
                       "FieldTransaction::validate_local: an incomplete replacement would "
                       "publish a mixture of this step's values and the previous step's"};
      }
    }
  }
  return Result<void>::success();
}

namespace {
/// The canonical accumulation order (`UpdateMode::Accumulate`'s commit
/// step), and precisely what it is, and is not, a guarantee of.
///
/// Round 2 of the S8 Stage B Field semantics track (the owner's own
/// standing rule: accuracy over efficiency) replaces round 1's raw-bit-
/// pattern order with KARMA's own magnitude-first scheme
/// (karma_field_transaction.C:48-54): TOTAL ORDER: contributions are
/// summed grouped by SLOT (entry index) -- `std::stable_sort` on `.first`
/// below, the same grouping KARMA's own `build_published_values()` uses
/// (karma_field_transaction.C:130-132) -- and, WITHIN one slot's group,
/// ordered ascending by |value|, then (for equal magnitude) by sign with
/// the NEGATIVE value first, then (for an exact duplicate -- equal
/// magnitude AND equal sign, which for a finite IEEE-754 `double` means
/// equal bits already) by the contribution's own raw bit pattern
/// (`std::bit_cast<std::uint64_t>`) as the final, formally-complete tie-
/// break. The within-slot order is therefore a function of WHICH VALUES
/// were staged to that slot, never of WHEN `stage()` happened to be called
/// for them: re-running the same logical set of contributions through a
/// loop that visits them in a different order (a different iteration order
/// over a caller's own container, a different interleaving of several
/// callers' stage() calls) produces the IDENTICAL sorted sequence for that
/// slot, and therefore the identical summed bits -- this is what "a total
/// order independent of staging order" means here, and it is the property
/// the ambiguity-removed mutation and the reversed-order mutation in
/// `build-evidence/mutations/field-transaction.toml` are built to catch;
/// swapping the comparator's magnitude/sign keys for raw arrival order
/// breaks the sort just as surely as it did for round 1's simpler key, so
/// that mutation stays killable unchanged.
///
/// WHAT THIS IS: summing the smallest-magnitude contributions to a slot
/// FIRST bounds the accumulated rounding by the classical recursive-
/// summation error bound (Higham, *Accuracy and Stability of Numerical
/// Algorithms*, 2nd ed., Theorem 4.2: summing n terms left-to-right has
/// forward error at most (n-1) * eps * SUM(|partial sum_i|) + O(eps^2), to
/// leading order) with each PARTIAL sum kept as small as the running total
/// allows -- a term whose magnitude is far below the running partial sum's
/// own ulp is not silently discarded until the partial sum has already
/// absorbed every other term of comparable or smaller size, which is
/// exactly how a large-magnitude term swamping many small ones one at a
/// time (the arrival-order failure mode) loses their CUMULATIVE
/// contribution entirely. The sign tie-break (negative first) is
/// arbitrary between two contributions of IDENTICAL magnitude -- it exists
/// only to make the order total and reproducible, not because negative-
/// before-positive has any accuracy advantage at equal magnitude.
///
/// WHAT THIS IS NOT:
///   - NOT a correctly-rounded sum. The summation itself (below) is a
///     plain left-to-right sequential fold -- not Kahan-compensated, not a
///     pairwise reduction tree -- so ordinary floating-point rounding error
///     still accumulates at each addition; magnitude-first ordering bounds
///     that error more tightly than an arbitrary order (see WHAT THIS IS,
///     above, and the worked case in
///     FieldTransaction.MagnitudeFirstOrderBeatsANaiveArrivalOrderSumOnACase
///     WhereItMatters, test/field/transaction_test.cpp), but it is not a
///     claim that the published value is the single closest representable
///     double to the mathematically exact sum in every case.
///   - NOT pairwise (tree) summation. Contributions are not split into
///     balanced halves and reduced hierarchically; they are folded
///     sequentially in the one sorted order above.
///   - NOT independent of WHICH SLOT a contribution targets, by
///     construction -- contributions to different slots never interact,
///     so grouping by slot first is exact, not an approximation.
double magnitude_of(double value) noexcept { return value < 0.0 ? -value : value; }

bool magnitude_then_sign_then_bits_before(double left, double right) noexcept {
  const double left_magnitude = magnitude_of(left);
  const double right_magnitude = magnitude_of(right);
  if (left_magnitude != right_magnitude) {
    return left_magnitude < right_magnitude;
  }
  const bool left_negative = std::signbit(left);
  const bool right_negative = std::signbit(right);
  if (left_negative != right_negative) {
    // Negative first: `true` (negative) sorts before `false`.
    return left_negative && !right_negative;
  }
  // Equal magnitude and equal sign already means `left == right` bit for
  // bit for any finite IEEE-754 double (magnitude and sign determine the
  // value completely), so this branch can never actually distinguish two
  // UNEQUAL bit patterns -- it is kept anyway, as the task's own
  // instruction asks, so the comparator is a formally complete strict
  // weak ordering rather than one that relies on an unstated invariant.
  return std::bit_cast<std::uint64_t>(left) < std::bit_cast<std::uint64_t>(right);
}
}  // namespace

Result<void> FieldTransaction::build_published_values(std::span<double> destination) const {
  const std::size_t values = entries_ * static_cast<std::size_t>(components_);
  if (destination.size() != values) {
    return Failure{"E-FIELD-TXN-SIZE",
                   "FieldTransaction::build_published_values: destination size mismatch"};
  }
  if (mode_ == UpdateMode::Replace) {
    std::copy(staged_.begin(), staged_.end(), destination.begin());
    return Result<void>::success();
  }

  const auto base = storage_->read_into(destination);
  if (!base.ok()) {
    return base;
  }

  if (mode_ == UpdateMode::Insert) {
    for (std::size_t slot = 0; slot != staged_.size(); ++slot) {
      if (has_staged_[slot]) {
        destination[slot] = staged_[slot];
      }
    }
    return Result<void>::success();
  }

  // Accumulate: group contributions by slot, then sum each group in the
  // canonical (slot, magnitude, sign, bit-pattern) order documented above,
  // so the published value does not depend on the order values were staged
  // in, and sums the smallest-magnitude terms of each slot first.
  std::vector<std::pair<std::size_t, double>> ordered(contributions_);
  std::stable_sort(ordered.begin(), ordered.end(),
                   [](const auto& left, const auto& right) { return left.first < right.first; });
  std::size_t index = 0;
  std::vector<double> slot_values;
  while (index != ordered.size()) {
    const std::size_t slot = ordered[index].first;
    slot_values.clear();
    while (index != ordered.size() && ordered[index].first == slot) {
      slot_values.push_back(ordered[index].second);
      ++index;
    }
    std::sort(slot_values.begin(), slot_values.end(), magnitude_then_sign_then_bits_before);
    double sum = destination[slot];
    for (const double value : slot_values) {
      sum += value;
    }
    if (!std::isfinite(sum)) {
      return Failure{"E-FIELD-TXN-NONFINITE",
                     "FieldTransaction::build_published_values: an accumulation produced a "
                     "nonfinite sum"};
    }
    destination[slot] = sum;
  }
  return Result<void>::success();
}

Result<void> FieldTransaction::commit() noexcept {
  const auto validation = validate_local();
  if (!validation.ok()) {
    return validation;
  }

  std::vector<double> publication(entries_ * static_cast<std::size_t>(components_), 0.0);
  const auto built = build_published_values(publication);
  if (!built.ok()) {
    return built;
  }

  // Snapshot before publishing, so a later restore has somewhere to go.
  std::vector<double> snapshot(entries_ * static_cast<std::size_t>(components_), 0.0);
  const auto read = storage_->read_into(snapshot);
  if (!read.ok()) {
    return read;
  }

  const auto published = storage_->publish(publication);
  if (!published.ok()) {
    return published;
  }

  previous_ = std::move(snapshot);
  has_previous_ = true;
  rollback();
  return Result<void>::success();
}

void FieldTransaction::rollback() noexcept {
  if (mode_ != UpdateMode::Accumulate) {
    std::fill(staged_.begin(), staged_.end(), 0.0);
    std::fill(has_staged_.begin(), has_staged_.end(), false);
  }
  contributions_.clear();
}

Result<void> FieldTransaction::restore_committed() noexcept {
  if (!has_previous_) {
    return Failure{"E-FIELD-TXN-ROLLBACK",
                   "FieldTransaction::restore_committed: there is no previous commit to "
                   "restore"};
  }
  const auto published = storage_->publish(previous_);
  if (!published.ok()) {
    return Failure{"E-FIELD-TXN-ROLLBACK",
                   "FieldTransaction::restore_committed: republishing the previous commit "
                   "failed: " + std::string(published.error().message)};
  }
  return Result<void>::success();
}

Result<void> FieldTransaction::commit_collective(std::optional<int> communicator) noexcept {
  // Preparation: agree that every rank can commit. No rank publishes while
  // any rank's validation has failed. Single process (no communicator):
  // nothing to reconcile, the same convention reconcile_plugin_agreement()
  // (src/abi/participant.cpp) uses for `!c.communicator_set`.
  if (!communicator.has_value()) {
    return commit();
  }

#if FUSE_HAVE_MPI
  const auto local_validation = validate_local();
  // MPI_MINLOC over {status-as-int32, rank}: the same shape
  // reconcile_plugin_agreement() (src/abi/participant.cpp) uses to agree a
  // verdict across a communicator, picking the MOST NEGATIVE/failing status
  // and the rank it came from deterministically, so every rank reaches the
  // identical decision from the identical reduced value -- a collective
  // path never returns on a locally computed condition.
  const MPI_Comm comm = MPI_Comm_f2c(static_cast<MPI_Fint>(*communicator));
  if (comm == MPI_COMM_NULL) {
    return Failure{"E-FIELD-TXN-COMM", "FieldTransaction::commit_collective: null communicator"};
  }
  int rank = 0;
  if (MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    return Failure{"E-FIELD-TXN-COMM",
                   "FieldTransaction::commit_collective: MPI_Comm_rank failed"};
  }
  static_assert(sizeof(int) == sizeof(std::int32_t),
               "MPI_2INT packs two `int`s; the status code below is int32_t -- the two must "
               "agree, the same static_assert reconcile_plugin_agreement() carries");
  struct {
    std::int32_t status;
    int rank;
  } in{local_validation.ok() ? 0 : 1, rank}, out{0, 0};
  if (MPI_Allreduce(&in, &out, 1, MPI_2INT, MPI_MAXLOC, comm) != MPI_SUCCESS) {
    return Failure{"E-FIELD-TXN-COMM",
                   "FieldTransaction::commit_collective: MPI_Allreduce(MPI_MAXLOC) failed "
                   "while reconciling the validation verdict"};
  }
  if (out.status != 0) {
    // At least one rank's local validation failed; no rank publishes.
    // Round 2 (S8 Stage B, real-communicator test, the lead's own
    // instruction): every rank -- including the one whose own local
    // validation is what failed -- returns the IDENTICAL Failure{code,
    // message} naming the failing rank, never the originating rank's own
    // raw local code. Round 1 special-cased `local_validation.ok()` to let
    // the originating rank's own message stand, mirroring
    // reconcile_plugin_agreement()'s (src/abi/participant.cpp) convention
    // for a DIFFERENT collective (that one deliberately keeps the origin's
    // own detailed diagnostic and rewrites it only for peers, host-verified
    // in KNOWN_ISSUES.md D85). FieldTransaction's own collective is
    // decided differently: a caller comparing every rank's return value
    // must see the SAME status everywhere, which is also what actually
    // completes the "a collective path never returns on a locally computed
    // condition" rule this function's own Allreduce comment already
    // invokes -- the pre-fix code violated that rule on exactly the one
    // rank whose local condition WAS the failure.
    return Failure{"E-FIELD-TXN-AGREEMENT-REFUSED",
                   "FieldTransaction::commit_collective: rank " + std::to_string(out.rank) +
                       " refused local validation; no rank publishes"};
  }

  // Publication: publish, then reconcile the publication outcome so every
  // rank learns of a failure that happened elsewhere. Every rank other than
  // the failing one has ALREADY published by the time this Allreduce runs
  // (`commit()` just above ran unconditionally, per rank, before the
  // agreement check) -- unlike the validation phase above, this phase
  // cannot be made atomic after the fact, the one limitation KARMA's own
  // commit_collective() documents and this file's own header repeats; what
  // this round's fix buys is a uniform REPORT of that inconsistent state,
  // not a way to undo it (restore_committed() is the caller's own recourse).
  const auto local = commit();
  struct {
    std::int32_t status;
    int rank;
  } pub_in{local.ok() ? 0 : 1, rank}, pub_out{0, 0};
  if (MPI_Allreduce(&pub_in, &pub_out, 1, MPI_2INT, MPI_MAXLOC, comm) != MPI_SUCCESS) {
    return Failure{"E-FIELD-TXN-COMM",
                   "FieldTransaction::commit_collective: MPI_Allreduce(MPI_MAXLOC) failed "
                   "while reconciling the publication outcome"};
  }
  if (pub_out.status != 0) {
    // Same uniformity fix as the validation branch above: every rank,
    // including the one that failed to publish, returns the identical
    // Failure naming that rank -- never the failing rank's own raw local
    // commit() error.
    return Failure{"E-FIELD-TXN-AGREEMENT-REFUSED",
                   "FieldTransaction::commit_collective: rank " + std::to_string(pub_out.rank) +
                       " failed publication after a successful agreement; the distributed "
                       "state is now inconsistent and must be restored"};
  }
  return Result<void>::success();
#else
  (void)communicator;
  return Failure{"E-FIELD-TXN-COMM",
                 "FieldTransaction::commit_collective: a communicator was supplied but this "
                 "build has no MPI (FUSE_HAVE_MPI=OFF)"};
#endif
}

std::size_t FieldTransaction::staged_count() const noexcept {
  if (mode_ != UpdateMode::Accumulate) {
    return static_cast<std::size_t>(std::count(has_staged_.begin(), has_staged_.end(), true));
  }
  return contributions_.size();
}

}  // namespace fuse
