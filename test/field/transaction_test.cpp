// Staged field updates with all-or-nothing publication: FieldTransaction
// over fuse::FieldRef (field/transaction.hpp, src/field/transaction.cpp).
//
// Structure and the properties under test mirror KARMA's own
// field_transaction_test.C (upstream/karma/test/field_transaction_test.C),
// cited per case below, translated onto fuse::Result and fuse::FieldRef in
// place of KARMA's OutcomeReason and FieldView/FieldIndexMap -- there is no
// stable-key index map here (docs/dev/field-semantics-matrix.md Section
// 2.1, gap 3, DEFERRED), so every case addresses a slot by its dense
// (entry, component) position, matching FieldTransaction::stage()'s own
// contract.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

// Explicit, not transitive: the feature macro FUSE_HAVE_MPI is read below
// (docs/dev/invariants.md; tools/check-invariants.py enforces this).
#include "fuse/base/config.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/transaction.hpp"

namespace {
using fuse::FieldRef;
using fuse::FieldRefStorage;
using fuse::FieldStorage;
using fuse::FieldTransaction;
using fuse::Layout;
using fuse::MemorySpace;
using fuse::Result;
using fuse::UpdateMode;

/// A dense, interleaved (component-minor) view over a plain array -- the
/// shape every case below builds a FieldRef with.
FieldRef dense_ref(std::vector<double>& values, std::size_t entries, int components) {
  Layout layout;
  layout.count = entries;
  layout.components = components;
  layout.stride = static_cast<std::size_t>(components);
  layout.component_stride = 1;
  return FieldRef(values.data(), layout, MemorySpace::Host);
}

/// FieldRefStorage that refuses publication, to prove that a failed commit
/// leaves no partial mutation -- KARMA's own FailingHostArrayStorage
/// (karma_field_transaction.h:70-85): `failing_slot` names the slot at
/// which the refusal fires, so the refusal happens mid-write rather than up
/// front, the same reason that fixture's own comment gives.
class FailingFieldRefStorage final : public FieldStorage {
 public:
  FailingFieldRefStorage(std::vector<double>& target, std::size_t failing_slot)
      : ref_(dense_ref(target, target.size(), 1)), failing_slot_(failing_slot) {}

  [[nodiscard]] std::size_t value_count() const override { return ref_.size(); }

  [[nodiscard]] Result<void> read_into(std::span<double> destination) const override {
    if (destination.size() != value_count()) {
      return fuse::Failure{"E-FIELD-TXN-SIZE", "size mismatch"};
    }
    for (std::size_t i = 0; i != destination.size(); ++i) destination[i] = ref_.at(i, 0);
    return Result<void>::success();
  }

  [[nodiscard]] Result<void> publish(std::span<const double> values) override {
    if (values.size() != value_count()) {
      return fuse::Failure{"E-FIELD-TXN-SIZE", "size mismatch"};
    }
    if (failing_slot_ < values.size()) {
      return fuse::Failure{"E-FIELD-TXN-STORAGE-TEST", "deliberate test failure"};
    }
    for (std::size_t i = 0; i != values.size(); ++i) ref_.at(i, 0) = values[i];
    return Result<void>::success();
  }

 private:
  FieldRef ref_;
  std::size_t failing_slot_;
};

// --- FieldRefStorage itself -------------------------------------------

TEST(FieldRefStorage, RoundTripsThroughAFieldRef) {
  std::vector<double> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  FieldRef ref = dense_ref(values, 3, 2);
  FieldRefStorage storage(ref);
  EXPECT_EQ(storage.value_count(), 6u);

  std::vector<double> read(6, 0.0);
  ASSERT_TRUE(storage.read_into(read).ok());
  EXPECT_EQ(read, values);

  const std::vector<double> written{9.0, 8.0, 7.0, 6.0, 5.0, 4.0};
  ASSERT_TRUE(storage.publish(written).ok());
  EXPECT_EQ(values, written);
}

TEST(FieldRefStorage, SizeMismatchIsRefused) {
  std::vector<double> values{1.0, 2.0};
  FieldRef ref = dense_ref(values, 2, 1);
  FieldRefStorage storage(ref);
  std::vector<double> wrong(3, 0.0);
  EXPECT_FALSE(storage.read_into(wrong).ok());
  EXPECT_FALSE(storage.publish(wrong).ok());
}

// --- FieldTransaction: Replace mode -- karma_field_transaction_test.C's
// own replace_commits() (:78-101) and rejects_bad_staging() (:103-122) ----

TEST(FieldTransaction, ReplaceDemandsCompleteCoverage) {
  std::vector<double> target{0.0, 0.0, 0.0};
  FieldRef ref = dense_ref(target, 3, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 3, 1, UpdateMode::Replace);

  EXPECT_TRUE(transaction.stage(0, 0, 1.5).ok());
  EXPECT_TRUE(transaction.stage(1, 0, 2.5).ok());
  // Nothing may reach the caller-visible target before commit.
  EXPECT_EQ(target[0], 0.0);
  EXPECT_EQ(target[1], 0.0);

  // Replace demands complete coverage: a partial update would silently keep
  // stale values from the previous step.
  const auto incomplete = transaction.validate_local();
  ASSERT_FALSE(incomplete.ok());
  EXPECT_EQ(incomplete.error().code, "E-FIELD-TXN-INCOMPLETE");
  EXPECT_FALSE(transaction.commit().ok());
  EXPECT_EQ(target[0], 0.0);

  EXPECT_TRUE(transaction.stage(2, 0, 3.5).ok());
  EXPECT_TRUE(transaction.validate_local().ok());
  ASSERT_TRUE(transaction.commit().ok());
  EXPECT_EQ(target[0], 1.5);
  EXPECT_EQ(target[1], 2.5);
  EXPECT_EQ(target[2], 3.5);
}

TEST(FieldTransaction, RejectsBadStaging) {
  std::vector<double> target{0.0, 0.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);

  auto nan_result = transaction.stage(0, 0, std::numeric_limits<double>::quiet_NaN());
  ASSERT_FALSE(nan_result.ok());
  EXPECT_EQ(nan_result.error().code, "E-FIELD-TXN-NONFINITE");

  auto inf_result = transaction.stage(0, 0, std::numeric_limits<double>::infinity());
  ASSERT_FALSE(inf_result.ok());
  EXPECT_EQ(inf_result.error().code, "E-FIELD-TXN-NONFINITE");

  auto bad_entry = transaction.stage(5, 0, 1.0);
  ASSERT_FALSE(bad_entry.ok());
  EXPECT_EQ(bad_entry.error().code, "E-FIELD-TXN-UNKNOWN-ENTRY");

  auto bad_component = transaction.stage(0, 3, 1.0);
  ASSERT_FALSE(bad_component.ok());
  EXPECT_EQ(bad_component.error().code, "E-FIELD-TXN-UNKNOWN-ENTRY");

  EXPECT_TRUE(transaction.stage(0, 0, 1.0).ok());
  // In Replace mode a second value for one slot is ambiguous, refused by
  // name rather than resolved by arrival order.
  auto duplicate = transaction.stage(0, 0, 2.0);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, "E-FIELD-TXN-DUPLICATE");
  EXPECT_EQ(target[0], 0.0);
}

// --- Insert mode -- karma_field_transaction_test.C's sparse_insertion()
// (:124-144), renamed SparseReplace -> Insert ------------------------------

TEST(FieldTransaction, InsertKeepsUnstagedValuesAndStillRejectsDuplicates) {
  std::vector<double> target{1.0, 2.0, 3.0};
  FieldRef ref = dense_ref(target, 3, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 3, 1, UpdateMode::Insert);

  EXPECT_TRUE(transaction.stage(1, 0, 9.0).ok());
  EXPECT_TRUE(transaction.validate_local().ok());
  EXPECT_EQ(target[1], 2.0);  // staging must not mutate the target
  ASSERT_TRUE(transaction.commit().ok());
  EXPECT_EQ(target[0], 1.0);
  EXPECT_EQ(target[1], 9.0);
  EXPECT_EQ(target[2], 3.0);

  EXPECT_TRUE(transaction.stage(2, 0, 1.0).ok());
  auto duplicate = transaction.stage(2, 0, 2.0);
  ASSERT_FALSE(duplicate.ok());
  EXPECT_EQ(duplicate.error().code, "E-FIELD-TXN-DUPLICATE");
}

// --- Accumulate mode: the canonical total order -- karma_field_transaction_
// test.C's canonical_accumulation() (:146-178), translated onto this file's
// own (slot, magnitude, sign, bit-pattern) order, KARMA's own magnitude-
// first scheme as of round 2 (see src/field/transaction.cpp's own comment
// for exactly what guarantee this is and is not) ----------------------------

TEST(FieldTransaction, AccumulationIsBitwiseIndependentOfStagingOrder) {
  const std::vector<double> contributions{1e16, 1.0, -1e16, 0.5, 3.0, -2.5};

  std::vector<double> forward_target{0.0};
  FieldRef forward_ref = dense_ref(forward_target, 1, 1);
  FieldRefStorage forward_storage(forward_ref);
  FieldTransaction forward(forward_storage, 1, 1, UpdateMode::Accumulate);
  for (const double value : contributions) {
    ASSERT_TRUE(forward.stage(0, 0, value).ok());
  }
  ASSERT_TRUE(forward.commit().ok());

  std::vector<double> reverse_target{0.0};
  FieldRef reverse_ref = dense_ref(reverse_target, 1, 1);
  FieldRefStorage reverse_storage(reverse_ref);
  FieldTransaction reverse(reverse_storage, 1, 1, UpdateMode::Accumulate);
  for (auto it = contributions.rbegin(); it != contributions.rend(); ++it) {
    ASSERT_TRUE(reverse.stage(0, 0, *it).ok());
  }
  ASSERT_TRUE(reverse.commit().ok());

  EXPECT_EQ(std::bit_cast<std::uint64_t>(forward_target[0]),
            std::bit_cast<std::uint64_t>(reverse_target[0]))
      << "accumulation is not bitwise independent of staging order";

  // A third, arbitrarily shuffled order must also match.
  std::vector<double> shuffled_target{0.0};
  FieldRef shuffled_ref = dense_ref(shuffled_target, 1, 1);
  FieldRefStorage shuffled_storage(shuffled_ref);
  FieldTransaction shuffled(shuffled_storage, 1, 1, UpdateMode::Accumulate);
  const std::vector<std::size_t> order{3, 0, 5, 1, 4, 2};
  for (const std::size_t index : order) {
    ASSERT_TRUE(shuffled.stage(0, 0, contributions[index]).ok());
  }
  ASSERT_TRUE(shuffled.commit().ok());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(forward_target[0]),
            std::bit_cast<std::uint64_t>(shuffled_target[0]));
}

// Round 2 (S8 Stage B, the owner's own accuracy-over-efficiency instruction):
// the same bitwise-independence property, over contributions spanning 1e-16
// to 1e+0 -- sixteen orders of magnitude -- rather than round 1's
// 1e16-to-1.0 range, so the new magnitude-first comparator's own ordering
// among SMALL values (not just its behaviour against one huge term) is
// exercised by this property too.
TEST(FieldTransaction, AccumulationIsBitwiseIndependentOfStagingOrderAcrossSixteenOrdersOfMagnitude) {
  const std::vector<double> contributions{1e-16, -3e-16, 5e-16, 1e-8, -1e-8, 0.5, -0.25, 1.0};

  std::vector<double> forward_target{0.0};
  FieldRef forward_ref = dense_ref(forward_target, 1, 1);
  FieldRefStorage forward_storage(forward_ref);
  FieldTransaction forward(forward_storage, 1, 1, UpdateMode::Accumulate);
  for (const double value : contributions) {
    ASSERT_TRUE(forward.stage(0, 0, value).ok());
  }
  ASSERT_TRUE(forward.commit().ok());

  std::vector<double> reverse_target{0.0};
  FieldRef reverse_ref = dense_ref(reverse_target, 1, 1);
  FieldRefStorage reverse_storage(reverse_ref);
  FieldTransaction reverse(reverse_storage, 1, 1, UpdateMode::Accumulate);
  for (auto it = contributions.rbegin(); it != contributions.rend(); ++it) {
    ASSERT_TRUE(reverse.stage(0, 0, *it).ok());
  }
  ASSERT_TRUE(reverse.commit().ok());

  EXPECT_EQ(std::bit_cast<std::uint64_t>(forward_target[0]),
            std::bit_cast<std::uint64_t>(reverse_target[0]))
      << "accumulation is not bitwise independent of staging order across this magnitude range";

  std::vector<double> shuffled_target{0.0};
  FieldRef shuffled_ref = dense_ref(shuffled_target, 1, 1);
  FieldRefStorage shuffled_storage(shuffled_ref);
  FieldTransaction shuffled(shuffled_storage, 1, 1, UpdateMode::Accumulate);
  const std::vector<std::size_t> order{5, 1, 7, 0, 6, 2, 4, 3};
  for (const std::size_t index : order) {
    ASSERT_TRUE(shuffled.stage(0, 0, contributions[index]).ok());
  }
  ASSERT_TRUE(shuffled.commit().ok());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(forward_target[0]),
            std::bit_cast<std::uint64_t>(shuffled_target[0]));
}

// Round 2's own worked case (docs/dev/field-semantics-matrix.md Section 6.1
// carries the derivation this test's own numbers come from): magnitude-
// first order against a naive sequential (STAGING-order) sum, asserted on
// the EXACT published bits, not merely "the two differ".
//
// Field starts at 0.0 (Accumulate mode allows starting fresh). Three
// contributions are STAGED in this order: `huge = 2^60` (`ulp(huge) =
// 2^(60-52) = 256`, half-ulp = 128), then `100.0`, then `100.0` again --
// deliberately the ORDER a naive, staging-order sum would add them in, and
// deliberately NOT the canonical order FieldTransaction actually sums in
// (build_published_values() always starts its running sum from the
// EXISTING published value -- 0.0 here -- then folds the STAGED
// contributions in ascending-magnitude order, `huge` last since it is the
// largest). A single `100.0 < half-ulp`, so adding it alone to a
// `huge`-sized running sum rounds back to that running sum exactly: a
// NAIVE order that adds `huge` FIRST (0+huge=huge, +100->huge rounded
// away, +100->huge rounded away again) loses the ENTIRE `200.0`
// contribution, publishing exactly `huge`. The ACTUAL canonical order
// sums the two smaller, equal-magnitude contributions with EACH OTHER
// first, while the running sum is still `0.0` (`0+100+100=200` exactly,
// far below double precision's own limit at this magnitude), and only
// THEN adds the large one: `0+200+huge`; `200 > half-ulp`, so this rounds
// to the NEAREST representable double, `huge + 200` evaluated directly --
// which IS the correctly-rounded value of the true mathematical sum
// (MEASURED in Python IEEE-754 binary64 during this round's own
// derivation: `huge+200` rounds to the double one bit-pattern step above
// `huge`, landing on `huge`'s own next representable neighbour at this
// magnitude's `ulp`, not back on `huge` itself).
TEST(FieldTransaction, MagnitudeFirstOrderBeatsANaiveArrivalOrderSumOnACaseWhereItMatters) {
  const double huge = std::ldexp(1.0, 60);  // 2^60, ulp = 256, half-ulp = 128
  // Computed directly, the same single addition the canonical order's own
  // last step performs, so this is the correctly-rounded answer by
  // construction, not a hardcoded literal that could silently drift from
  // what the hardware actually rounds to.
  const double correctly_rounded = huge + 200.0;
  ASSERT_NE(std::bit_cast<std::uint64_t>(huge), std::bit_cast<std::uint64_t>(correctly_rounded))
      << "huge+200 must round to a distinct representable double from huge for this case to "
         "mean anything";

  // What FieldTransaction actually publishes: field starts at 0.0, and the
  // three contributions below are staged in NAIVE (large-magnitude-first)
  // order -- the canonical accumulation order re-sorts them by magnitude
  // regardless of this staging order, which is the whole point.
  std::vector<double> target{0.0};
  FieldRef ref = dense_ref(target, 1, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 1, 1, UpdateMode::Accumulate);
  ASSERT_TRUE(transaction.stage(0, 0, huge).ok());
  ASSERT_TRUE(transaction.stage(0, 0, 100.0).ok());
  ASSERT_TRUE(transaction.stage(0, 0, 100.0).ok());
  ASSERT_TRUE(transaction.commit().ok());
  EXPECT_EQ(std::bit_cast<std::uint64_t>(target[0]), std::bit_cast<std::uint64_t>(correctly_rounded))
      << "magnitude-first order must reproduce the correctly-rounded exact sum";

  // What a NAIVE sequential sum, folding the SAME three values in the
  // ORDER THEY WERE STAGED above (huge, then 100, then 100) with no
  // reordering at all, would have published instead.
  double naive = 0.0;
  naive += huge;
  naive += 100.0;
  naive += 100.0;
  EXPECT_EQ(std::bit_cast<std::uint64_t>(naive), std::bit_cast<std::uint64_t>(huge))
      << "the naive staging-order sum is expected to lose the entire 200.0 contribution";
  EXPECT_NE(std::bit_cast<std::uint64_t>(naive), std::bit_cast<std::uint64_t>(target[0]))
      << "magnitude-first order must differ from, and be more accurate than, the naive sum here";
}

TEST(FieldTransaction, AccumulationAddsToThePublishedValueAndAllowsPartialCoverage) {
  std::vector<double> target{5.0, 7.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Accumulate);

  EXPECT_TRUE(transaction.stage(1, 0, 3.0).ok());
  EXPECT_TRUE(transaction.validate_local().ok());  // no coverage requirement
  ASSERT_TRUE(transaction.commit().ok());
  EXPECT_EQ(target[0], 5.0);
  EXPECT_EQ(target[1], 10.0);
}

// --- Rollback and restore -- karma_field_transaction_test.C's
// rollback_and_restore() (:180-201) -----------------------------------------

TEST(FieldTransaction, RollbackDiscardsStagedStateWithoutTouchingTheTarget) {
  std::vector<double> target{1.0, 2.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction discarded(storage, 2, 1, UpdateMode::Replace);

  EXPECT_TRUE(discarded.stage(0, 0, 9.0).ok());
  discarded.rollback();
  EXPECT_EQ(target[0], 1.0);
  EXPECT_EQ(target[1], 2.0);
  EXPECT_EQ(discarded.staged_count(), 0u);

  // A rolled-back transaction is reusable rather than poisoned.
  EXPECT_TRUE(discarded.stage(0, 0, 4.0).ok());
  EXPECT_TRUE(discarded.stage(1, 0, 5.0).ok());
  ASSERT_TRUE(discarded.commit().ok());
  EXPECT_EQ(target[0], 4.0);
  EXPECT_EQ(target[1], 5.0);

  // Restoring the pre-commit snapshot is a separate, fallible operation:
  // discarding staged state cannot fail, republishing can.
  ASSERT_TRUE(discarded.restore_committed().ok());
  EXPECT_EQ(target[0], 1.0);
  EXPECT_EQ(target[1], 2.0);
}

TEST(FieldTransaction, RestoreWithNoPriorCommitIsRefused) {
  std::vector<double> target{1.0};
  FieldRef ref = dense_ref(target, 1, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 1, 1, UpdateMode::Replace);
  const auto restored = transaction.restore_committed();
  ASSERT_FALSE(restored.ok());
  EXPECT_EQ(restored.error().code, "E-FIELD-TXN-ROLLBACK");
}

// --- Storage failure leaves no partial mutation -- karma_field_transaction_
// test.C's storage_failure_leaves_no_partial_mutation() (:203-216) -----------

TEST(FieldTransaction, AFailingPublicationLeavesNoPartialMutation) {
  std::vector<double> target{1.0, 2.0, 3.0};
  // Refuses publication at the second value, which would leave a partial
  // write if the transaction published in place.
  FailingFieldRefStorage storage(target, 1);
  FieldTransaction transaction(storage, 3, 1, UpdateMode::Replace);
  for (std::size_t entry = 0; entry != 3; ++entry) {
    ASSERT_TRUE(transaction.stage(entry, 0, 10.0 + static_cast<double>(entry)).ok());
  }
  const auto outcome = transaction.commit();
  ASSERT_FALSE(outcome.ok());
  EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-STORAGE-TEST");
  EXPECT_EQ(target[0], 1.0);
  EXPECT_EQ(target[1], 2.0);
  EXPECT_EQ(target[2], 3.0);
}

// --- commit_collective: single-process path (no communicator) --------------

TEST(FieldTransaction, CommitCollectiveWithNoCommunicatorBehavesLikeCommit) {
  std::vector<double> target{0.0, 0.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);
  EXPECT_TRUE(transaction.stage(0, 0, 1.0).ok());
  EXPECT_TRUE(transaction.stage(1, 0, 2.0).ok());
  ASSERT_TRUE(transaction.commit_collective(std::nullopt).ok());
  EXPECT_EQ(target[0], 1.0);
  EXPECT_EQ(target[1], 2.0);
}

TEST(FieldTransaction, CommitCollectiveWithNoCommunicatorRefusesAnIncompleteReplaceLikeCommit) {
  std::vector<double> target{0.0, 0.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);
  EXPECT_TRUE(transaction.stage(0, 0, 1.0).ok());
  // Entry 1 never staged: validate_local() must refuse exactly as commit()
  // alone would, whether or not a communicator is involved.
  const auto outcome = transaction.commit_collective(std::nullopt);
  ASSERT_FALSE(outcome.ok());
  EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-INCOMPLETE");
  EXPECT_EQ(target[0], 0.0);
}

#if !FUSE_HAVE_MPI
// With no MPI compiled in at all, a REAL (non-null) communicator handle is
// refused by name rather than silently treated as single-process -- a
// caller that believes it is coordinating across ranks must learn that no
// coordination happened, not get a quiet no-op.
TEST(FieldTransaction, CommitCollectiveWithACommunicatorAndNoMpiIsRefused) {
  std::vector<double> target{0.0};
  FieldRef ref = dense_ref(target, 1, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 1, 1, UpdateMode::Replace);
  EXPECT_TRUE(transaction.stage(0, 0, 1.0).ok());
  const auto outcome = transaction.commit_collective(0);
  ASSERT_FALSE(outcome.ok());
  EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-COMM");
}
#endif

TEST(FieldTransaction, StagedCountDistinguishesModes) {
  std::vector<double> replace_target{0.0, 0.0};
  FieldRef replace_ref = dense_ref(replace_target, 2, 1);
  FieldRefStorage replace_storage(replace_ref);
  FieldTransaction replace(replace_storage, 2, 1, UpdateMode::Replace);
  EXPECT_EQ(replace.staged_count(), 0u);
  EXPECT_TRUE(replace.stage(0, 0, 1.0).ok());
  EXPECT_EQ(replace.staged_count(), 1u);

  std::vector<double> accum_target{0.0};
  FieldRef accum_ref = dense_ref(accum_target, 1, 1);
  FieldRefStorage accum_storage(accum_ref);
  FieldTransaction accumulate(accum_storage, 1, 1, UpdateMode::Accumulate);
  EXPECT_TRUE(accumulate.stage(0, 0, 1.0).ok());
  EXPECT_TRUE(accumulate.stage(0, 0, 2.0).ok());  // two CONTRIBUTIONS, not a duplicate refusal
  EXPECT_EQ(accumulate.staged_count(), 2u);
}
}  // namespace
