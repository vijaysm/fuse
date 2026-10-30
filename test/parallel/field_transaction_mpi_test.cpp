// KNOWN_ISSUES.md / docs/dev/field-semantics-matrix.md Section 6.1: the
// real-communicator twin of test/field/transaction_test.cpp's
// CommitCollectiveWith* cases, over an ACTUAL MPI_COMM_WORLD rather than
// `std::nullopt` or (compiled out under FUSE_HAVE_MPI) a locally-constructed
// non-null handle with no MPI behind it. S8 Stage B, Field semantics track,
// round 2 (the lead's own instruction, 2026-10-28/29): the round-1 tier-0
// suite built and linked `commit_collective()`'s `#if FUSE_HAVE_MPI` branch
// but never exercised it -- the round-1 `CommitCollectiveWithACommunicator
// AndNoMpiIsRefused` case is guarded `#if !FUSE_HAVE_MPI` and so is compiled
// OUT of an MPI build entirely, and the tier-0 mutation sweep's own
// `field-transaction-collective-agreement-skipped` row (build-evidence/
// mutations/field-transaction.toml) could only be marked NOT-ASSESSED for
// exactly that reason (tools/mutate.py --area field-transaction,
// `measured_results`). This file is that hole closed: a real
// MPI_Allreduce over MPI_COMM_WORLD, registered through the same
// fuse_add_mpi_test machinery as test/parallel/provider_subset_mpi_test.cpp
// (RANKS 2 3).
//
// MPI CANNOT START IN THIS SANDBOX (KNOWN_ISSUES.md C1): this file is
// compiled here, never run; host_pending names the exact `mpirun` commands
// the lead runs on higgs.
//
// THE COMMUNICATOR HANDLE. Every case below passes
// `MPI_Comm_c2f(MPI_COMM_WORLD)` as the `std::optional<int>` argument
// `FieldTransaction::commit_collective()` takes. On Open MPI that Fortran
// handle is 0 (KNOWN_ISSUES.md D85's own MEASURED finding, higgs, Open MPI
// 5.0.9, 2026-10-23) -- a value that the C ABI (`fuse_setup_options`) must
// be told is deliberate via a SEPARATE `has_communicator` flag, because a
// plain `int64_t communicator` field cannot otherwise distinguish "the
// handle is 0" from "no communicator was given" (D85's own "why not choose
// by MPI flavour" paragraph). `FieldTransaction::commit_collective()` does
// NOT have that ambiguity: its argument is `std::optional<int>`, and
// `std::optional<int>{0}` is a PRESENT optional holding the value 0, fully
// distinguishable from `std::nullopt` at the type level -- there is no
// second flag to set and no way to construct this call that reads as
// "unset" by accident. The comment is here anyway, matching D85's own
// precedent, because a reader unfamiliar with Open MPI's own numbering
// could otherwise mistake a logged "communicator = 0" for a bug.
#include "fuse/base/config.hpp"

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "fuse/field/field_ref.hpp"
#include "fuse/field/transaction.hpp"

#include "fuse_test_main.hpp"

namespace {

using fuse::FieldRef;
using fuse::FieldRefStorage;
using fuse::FieldStorage;
using fuse::FieldTransaction;
using fuse::Layout;
using fuse::MemorySpace;
using fuse::Result;
using fuse::UpdateMode;

FieldRef dense_ref(std::vector<double>& values, std::size_t entries, int components) {
  Layout layout;
  layout.count = entries;
  layout.components = components;
  layout.stride = static_cast<std::size_t>(components);
  layout.component_stride = 1;
  return FieldRef(values.data(), layout, MemorySpace::Host);
}

/// KARMA's own `FailingHostArrayStorage` (karma_field_transaction.h:70-85),
/// under the name test/field/transaction_test.cpp's own copy already uses
/// in this port (`FieldRefStorage` is this port's name for KARMA's
/// `HostArrayStorage`) -- redefined here rather than shared because that
/// file's own class is local to its own anonymous namespace, not exposed
/// through any header. `failing_slot` names the slot `publish()` refuses
/// at, so the refusal happens mid-write rather than up front.
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

/// MPI_Allgather of a fixed-width, null-padded copy of `outcome`'s own
/// error code (empty string for a success) and assert every rank's own
/// bytes are IDENTICAL to this rank's own -- the direct, collective check
/// behind this file's own "the SAME non-OK status on every rank"
/// requirement, not an assumption from each rank's own local EXPECT alone.
constexpr std::size_t kCodeBufferSize = 64;

void expect_identical_status_code_on_every_rank(const Result<void>& outcome, int world_size) {
  char mine[kCodeBufferSize] = {};
  if (!outcome.ok()) {
    std::snprintf(mine, sizeof(mine), "%.*s", static_cast<int>(sizeof(mine) - 1),
                 std::string(outcome.error().code).c_str());
  }
  std::vector<char> all(static_cast<std::size_t>(world_size) * kCodeBufferSize, '\0');
  MPI_Allgather(mine, static_cast<int>(kCodeBufferSize), MPI_CHAR, all.data(),
               static_cast<int>(kCodeBufferSize), MPI_CHAR, MPI_COMM_WORLD);
  for (int r = 0; r < world_size; ++r) {
    EXPECT_EQ(std::memcmp(mine, all.data() + static_cast<std::size_t>(r) * kCodeBufferSize,
                         kCodeBufferSize),
             0)
        << "rank " << r << " reported status code \"" << (all.data() + static_cast<std::size_t>(r) * kCodeBufferSize)
        << "\", this rank reported \"" << mine << "\" -- commit_collective() must return the "
           "SAME status on every rank";
  }
}

}  // namespace

// karma_field_transaction_test.C's own canonical_accumulation()-adjacent
// collective cases have no counterpart upstream (KARMA never had a
// collective commit over a REAL communicator in its own test suite,
// docs/dev/field-semantics-matrix.md Section 2.2) -- this file's own three
// cases are this round's own addition, not a translation of an existing one.

// --- Case 1: validation disagreement -----------------------------------
//
// Every rank stages a complete, valid Replace except the LAST rank (never
// rank 0, so the mutation this test kills cannot be mistaken for a
// rank-0-is-special special case): the last rank stages entry 0, then
// deliberately stages entry 0 a SECOND time (refused immediately, by name,
// as E-FIELD-TXN-DUPLICATE -- confirmed below), and never stages entry 1 at
// all, so its own validate_local() reports E-FIELD-TXN-INCOMPLETE.
//
// MUTATION KILLED HERE: build-evidence/mutations/field-transaction.toml's
// `field-transaction-collective-agreement-skipped` replaces
// `if (out.status != 0)` with `if (false)`, skipping the validation-
// agreement check entirely. Under that mutation every rank OTHER than the
// last falls through to publish unconditionally (each rank's own `commit()`
// succeeds locally, since each one's OWN staging is complete and valid),
// changing ITS OWN target values -- which this test's own "unchanged on
// every rank" assertion below catches directly. (The last rank's own
// `commit()` still fails locally regardless of the mutation, since
// `commit()` re-validates; the mutation's effect is visible on every OTHER
// rank, which is why this case, not a single-rank check, is what kills it.)
TEST(FieldTransactionMpi, ValidationDisagreementAgreesTheSameRefusalOnEveryRankAndPublishesNowhere) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int last_rank = world_size - 1;

  std::vector<double> target{11.0, 13.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);

  if (rank == last_rank) {
    ASSERT_TRUE(transaction.stage(0, 0, 100.0 + static_cast<double>(rank)).ok());
    const auto duplicate = transaction.stage(0, 0, 999.0);
    ASSERT_FALSE(duplicate.ok());
    EXPECT_EQ(duplicate.error().code, "E-FIELD-TXN-DUPLICATE");
    // Entry 1 is never staged: validate_local() below must fail with
    // E-FIELD-TXN-INCOMPLETE, which is what feeds this rank's own `status`
    // into the collective Allreduce.
    EXPECT_FALSE(transaction.validate_local().ok());
  } else {
    ASSERT_TRUE(transaction.stage(0, 0, 100.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.stage(1, 0, 200.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.validate_local().ok());
  }

  const auto outcome = transaction.commit_collective(MPI_Comm_c2f(MPI_COMM_WORLD));
  ASSERT_FALSE(outcome.ok()) << "rank " << rank;
  EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-AGREEMENT-REFUSED") << "rank " << rank;
  EXPECT_NE(outcome.error().message.find("rank " + std::to_string(last_rank)), std::string::npos)
      << "rank " << rank << " got message: " << outcome.error().message;
  expect_identical_status_code_on_every_rank(outcome, world_size);

  // No rank publishes when any rank's validation failed -- every rank's
  // own target is exactly its own starting values, whether or not THIS
  // rank's own staging was itself complete and valid.
  EXPECT_EQ(target[0], 11.0) << "rank " << rank;
  EXPECT_EQ(target[1], 13.0) << "rank " << rank;
}

// --- Case 2: every rank validates and all publish -----------------------

TEST(FieldTransactionMpi, EveryRankValidatesAndCommitsCollectivelyOnEveryRank) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  std::vector<double> target{11.0, 13.0};
  FieldRef ref = dense_ref(target, 2, 1);
  FieldRefStorage storage(ref);
  FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);
  ASSERT_TRUE(transaction.stage(0, 0, 100.0 + static_cast<double>(rank)).ok());
  ASSERT_TRUE(transaction.stage(1, 0, 200.0 + static_cast<double>(rank)).ok());

  const auto outcome = transaction.commit_collective(MPI_Comm_c2f(MPI_COMM_WORLD));
  ASSERT_TRUE(outcome.ok()) << "rank " << rank << ": " << outcome.error().message;
  expect_identical_status_code_on_every_rank(outcome, world_size);
  EXPECT_EQ(target[0], 100.0 + static_cast<double>(rank)) << "rank " << rank;
  EXPECT_EQ(target[1], 200.0 + static_cast<double>(rank)) << "rank " << rank;
}

// --- Case 3: validation agrees, one rank's publication fails -------------
//
// Every rank's OWN staging is complete and valid, so the validation-
// agreement phase succeeds everywhere; the LAST rank's own storage is
// `FailingFieldRefStorage`, so ITS OWN `commit()` (called unconditionally,
// once validation has agreed) fails at `publish()`. Every OTHER rank's own
// `commit()` has ALREADY succeeded by the time the publication-agreement
// Allreduce runs (FieldTransaction.hpp's own doc comment: this phase
// "cannot make atomic" a failure that happens elsewhere) -- so this case
// does NOT assert "unchanged on every rank" the way Case 1 does; it asserts
// the documented asymmetry directly: changed where publication succeeded,
// unchanged where `FailingFieldRefStorage` refused it (atomically, by its
// own contract, the same property
// FieldTransaction.AFailingPublicationLeavesNoPartialMutation pins with no
// MPI involved), and the SAME reported status, naming the failing rank, on
// every rank regardless of which side of that asymmetry it is on.
TEST(FieldTransactionMpi, PublicationDisagreementAgreesTheSameFailureNamingTheFailingRankOnEveryRank) {
  FUSE_REQUIRE_AT_LEAST_RANKS(2);
  int rank = 0, world_size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  const int last_rank = world_size - 1;

  std::vector<double> target{11.0, 13.0};
  const std::vector<double> original = target;

  if (rank == last_rank) {
    FailingFieldRefStorage storage(target, /*failing_slot=*/0);
    FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);
    ASSERT_TRUE(transaction.stage(0, 0, 100.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.stage(1, 0, 200.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.validate_local().ok());

    const auto outcome = transaction.commit_collective(MPI_Comm_c2f(MPI_COMM_WORLD));
    ASSERT_FALSE(outcome.ok()) << "rank " << rank;
    EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-AGREEMENT-REFUSED") << "rank " << rank;
    EXPECT_NE(outcome.error().message.find("rank " + std::to_string(last_rank)), std::string::npos)
        << "rank " << rank << " got message: " << outcome.error().message;
    expect_identical_status_code_on_every_rank(outcome, world_size);
    // FailingFieldRefStorage::publish() refuses before writing any slot --
    // this rank's own target is untouched, same as
    // FieldTransaction.AFailingPublicationLeavesNoPartialMutation.
    EXPECT_EQ(target[0], original[0]) << "rank " << rank;
    EXPECT_EQ(target[1], original[1]) << "rank " << rank;
  } else {
    FieldRef ref = dense_ref(target, 2, 1);
    FieldRefStorage storage(ref);
    FieldTransaction transaction(storage, 2, 1, UpdateMode::Replace);
    ASSERT_TRUE(transaction.stage(0, 0, 100.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.stage(1, 0, 200.0 + static_cast<double>(rank)).ok());
    ASSERT_TRUE(transaction.validate_local().ok());

    const auto outcome = transaction.commit_collective(MPI_Comm_c2f(MPI_COMM_WORLD));
    ASSERT_FALSE(outcome.ok()) << "rank " << rank;
    EXPECT_EQ(outcome.error().code, "E-FIELD-TXN-AGREEMENT-REFUSED") << "rank " << rank;
    EXPECT_NE(outcome.error().message.find("rank " + std::to_string(last_rank)), std::string::npos)
        << "rank " << rank << " got message: " << outcome.error().message;
    expect_identical_status_code_on_every_rank(outcome, world_size);
    // This rank's own commit() already succeeded before the agreement
    // Allreduce ran -- the documented non-atomicity of this phase.
    EXPECT_EQ(target[0], 100.0 + static_cast<double>(rank)) << "rank " << rank;
    EXPECT_EQ(target[1], 200.0 + static_cast<double>(rank)) << "rank " << rank;
  }
}
