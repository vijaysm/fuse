#include "fuse/field/transaction.hpp"

#include "fuse/field/collective_agreement.hpp"

#include <algorithm>
#include <bit>
#include <cmath>

#include "fuse/base/error.hpp"

// MPI is no longer named in this file: the two collective agreements of
// commit_collective() run through fuse::agree_outcome() in
// src/field/collective_agreement.cpp, now the one translation unit of
// `field` that calls MPI (docs/dev/layering.md, "MPI outside transport").

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
/// Accuracy takes priority over efficiency here, so the order is KARMA's own
/// magnitude-first scheme (karma_field_transaction.C:48-54), not a raw-bit-
/// pattern order: TOTAL ORDER: contributions are
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
/// breaks the sort just as surely as it would for a simpler key, so
/// that mutation is killable.
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
  // UNEQUAL bit patterns -- it is kept anyway, so
  // the comparator is a formally complete strict weak ordering rather than one that relies on an unstated invariant.
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
  if (!communicator.has_value()) {
    return commit();
  }
  // Two agreements, each through fuse::agree_outcome() (fuse/field/
  // collective_agreement.hpp): the verdict is reduced, the originating rank
  // broadcasts its own Failure, and every rank -- the originator included --
  // returns the identical Failure carrying the origin rank, the origin's own
  // code and its message (the original error and the rank it came from are
  // known everywhere).
  // Phase 1 agrees validation; no rank publishes unless every rank's local
  // validation passed. Phase 2 agrees publication; by the time a refusal is
  // discovered there every other rank has already published, the one
  // non-atomic case the header documents, and the caller must restore.
  const auto local_validation = validate_local();
  const auto validation = agree_outcome(*communicator, local_validation,
                                        "E-FIELD-TXN-AGREEMENT-REFUSED", "E-FIELD-TXN-COMM",
                                        "FieldTransaction::commit_collective (validation)");
  if (!validation.ok()) return validation;

  const auto local = commit();
  return agree_outcome(*communicator, local, "E-FIELD-TXN-AGREEMENT-REFUSED",
                       "E-FIELD-TXN-COMM",
                       "FieldTransaction::commit_collective (publication; every rank that "
                       "did not refuse has already published and must be restored)");
}

std::size_t FieldTransaction::staged_count() const noexcept {
  if (mode_ != UpdateMode::Accumulate) {
    return static_cast<std::size_t>(std::count(has_staged_.begin(), has_staged_.end(), true));
  }
  return contributions_.size();
}

}  // namespace fuse
