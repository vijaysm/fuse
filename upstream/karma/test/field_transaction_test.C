// Storage-neutral field views and staged, all-or-nothing commits.
//
// Exercised over the host-array storage backend, which needs neither PETSc nor
// MPI. The PETSc and MOAB backends are the same transaction with a different
// publication step; their tests require those libraries and run separately.
#include "karma_field_transaction.h"
#include "karma_field_view.h"

#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

FieldEntryKey moab_key(std::uint64_t global_id, unsigned component)
{
    FieldEntryKey key;
    key.kind = FieldEntryKey::Kind::MoabGlobalId;
    key.global_id = global_id;
    key.component = component;
    return key;
}

std::vector<FieldEntryKey> three_entries()
{
    return {moab_key(10, 0), moab_key(20, 0), moab_key(30, 0)};
}

void index_map()
{
    const auto entries = three_entries();
    const FieldViewLayout layout{1, 2, 1};
    const auto built = FieldIndexMap::build(entries, layout);
    require(built.map.has_value() && built.reason == OutcomeReason::Success, "a valid index map was rejected");
    const auto& map = *built.map;
    require(map.entry_count() == 3, "index map lost entries");

    require(map.find(moab_key(20, 0)) == 1, "a known key resolved to the wrong dense index");
    require(!map.find(moab_key(99, 0)).has_value(), "an unknown key resolved");
    require(map.lookup_count() == 2, "key lookups are not counted");

    // Ownership is positional: owned entries precede ghosts.
    require(map.is_owned(0) && map.is_owned(1) && !map.is_owned(2), "owned and ghost entries are not separated");

    // A duplicate key would make dense resolution ambiguous.
    const std::vector<FieldEntryKey> duplicated{moab_key(10, 0), moab_key(10, 0)};
    require(FieldIndexMap::build(duplicated, {1, 2, 0}).reason == OutcomeReason::DuplicateEntry,
            "a duplicate key built an index map");
    // The layout must account for exactly the entries given.
    require(FieldIndexMap::build(entries, {1, 1, 1}).reason == OutcomeReason::InvalidArgument,
            "a layout that does not cover the entries was accepted");
}

void read_only_views()
{
    const std::vector<Real> values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    const FieldViewLayout layout{2, 2, 1};
    const FieldView view(values, layout);
    require(view.layout().total_entries() == 3 && view.layout().total_values() == 6, "view layout is wrong");
    const auto second = view.entry_values(1);
    require(second.size() == 2 && second[0] == 3.0 && second[1] == 4.0, "component-minor entry span is wrong");
    require(view.is_owned(1) && !view.is_owned(2), "view ownership is wrong");
    require(view.values().data() == values.data(), "the view copied instead of referencing");
}

void replace_commits()
{
    std::vector<Real> target{0.0, 0.0, 0.0};
    HostArrayStorage storage(target);
    const FieldViewLayout layout{1, 3, 0};

    FieldTransaction transaction(storage, layout, UpdateMode::Replace);
    require(transaction.stage(0, 0, 1.5) == OutcomeReason::Success, "staging was rejected");
    require(transaction.stage(1, 0, 2.5) == OutcomeReason::Success, "staging was rejected");
    // Nothing may reach the caller-visible target before commit.
    require(target[0] == 0.0 && target[1] == 0.0, "staging mutated the target");

    // Replace demands complete coverage: a partial update would silently keep
    // stale values from the previous step.
    require(transaction.validate_local() == OutcomeReason::IncompleteUpdate,
            "an incomplete replacement validated");
    require(!transaction.commit().ok(), "an incomplete replacement committed");
    require(target[0] == 0.0, "a rejected commit mutated the target");

    require(transaction.stage(2, 0, 3.5) == OutcomeReason::Success, "staging was rejected");
    require(transaction.validate_local() == OutcomeReason::Success, "a complete replacement failed validation");
    require(transaction.commit().ok(), "a complete replacement failed to commit");
    require(target[0] == 1.5 && target[1] == 2.5 && target[2] == 3.5, "commit published the wrong values");
}

void rejects_bad_staging()
{
    std::vector<Real> target{0.0, 0.0};
    HostArrayStorage storage(target);
    const FieldViewLayout layout{1, 2, 0};
    FieldTransaction transaction(storage, layout, UpdateMode::Replace);

    require(transaction.stage(0, 0, std::numeric_limits<Real>::quiet_NaN()) == OutcomeReason::NonfiniteValue,
            "a NaN was staged");
    require(transaction.stage(0, 0, std::numeric_limits<Real>::infinity()) == OutcomeReason::NonfiniteValue,
            "an infinity was staged");
    require(transaction.stage(5, 0, 1.0) == OutcomeReason::UnknownEntry, "an out-of-range entry was staged");
    require(transaction.stage(0, 3, 1.0) == OutcomeReason::UnknownEntry, "an out-of-range component was staged");

    require(transaction.stage(0, 0, 1.0) == OutcomeReason::Success, "a valid value was rejected");
    // In replacement mode a second value for one slot is ambiguous, so it is
    // rejected rather than resolved by arrival order.
    require(transaction.stage(0, 0, 2.0) == OutcomeReason::DuplicateEntry, "a duplicate replacement was accepted");
    require(target[0] == 0.0, "a rejected staging reached the target");
}

void sparse_insertion()
{
    std::vector<Real> target{1.0, 2.0, 3.0};
    HostArrayStorage storage(target);
    FieldTransaction transaction(storage, {1, 3, 0}, UpdateMode::SparseReplace);

    // Unstaged slots keep the published value rather than being zeroed, and no
    // coverage requirement applies.
    require(transaction.stage(1, 0, 9.0) == OutcomeReason::Success, "sparse staging was rejected");
    require(transaction.validate_local() == OutcomeReason::Success, "a sparse insertion failed validation");
    require(target[1] == 2.0, "sparse staging mutated the target");
    require(transaction.commit().ok(), "a sparse insertion failed to commit");
    require(target[0] == 1.0 && target[1] == 9.0 && target[2] == 3.0,
            "sparse insertion did not preserve unstaged values");

    // Ambiguity is still rejected: two values for one slot have no defined
    // resolution in a replacement.
    require(transaction.stage(2, 0, 1.0) == OutcomeReason::Success, "staging was rejected");
    require(transaction.stage(2, 0, 2.0) == OutcomeReason::DuplicateEntry,
            "a duplicate sparse replacement was accepted");
}

void canonical_accumulation()
{
    // Accumulation must not depend on the order contributions arrive in, so
    // the sum is formed in a canonical order at commit rather than in place.
    const std::vector<Real> contributions{1e16, 1.0, -1e16, 0.5, 3.0, -2.5};
    std::vector<Real> forward_target{0.0};
    std::vector<Real> reverse_target{0.0};
    HostArrayStorage forward_storage(forward_target);
    HostArrayStorage reverse_storage(reverse_target);
    const FieldViewLayout layout{1, 1, 0};

    FieldTransaction forward(forward_storage, layout, UpdateMode::Accumulate);
    for (const auto value : contributions) require(forward.stage(0, 0, value) == OutcomeReason::Success, "stage failed");
    require(forward.commit().ok(), "accumulation failed to commit");

    FieldTransaction reverse(reverse_storage, layout, UpdateMode::Accumulate);
    for (auto index = contributions.size(); index-- > 0;)
        require(reverse.stage(0, 0, contributions[index]) == OutcomeReason::Success, "stage failed");
    require(reverse.commit().ok(), "accumulation failed to commit");

    require(std::bit_cast<std::uint64_t>(forward_target[0]) == std::bit_cast<std::uint64_t>(reverse_target[0]),
            "accumulation is not bitwise independent of staging order");

    // Accumulation adds to what is already published rather than replacing it,
    // and does not require every slot to be staged.
    std::vector<Real> partial_target{5.0, 7.0};
    HostArrayStorage partial_storage(partial_target);
    FieldTransaction partial(partial_storage, {1, 2, 0}, UpdateMode::Accumulate);
    require(partial.stage(1, 0, 3.0) == OutcomeReason::Success, "stage failed");
    require(partial.validate_local() == OutcomeReason::Success, "a partial accumulation failed validation");
    require(partial.commit().ok(), "a partial accumulation failed to commit");
    require(partial_target[0] == 5.0 && partial_target[1] == 10.0, "accumulation did not add to the published value");
}

void rollback_and_restore()
{
    std::vector<Real> target{1.0, 2.0};
    HostArrayStorage storage(target);
    const FieldViewLayout layout{1, 2, 0};

    FieldTransaction discarded(storage, layout, UpdateMode::Replace);
    require(discarded.stage(0, 0, 9.0) == OutcomeReason::Success, "stage failed");
    discarded.rollback();
    require(target[0] == 1.0 && target[1] == 2.0, "rollback of staged state mutated the target");
    require(discarded.staged_count() == 0, "rollback did not discard staged values");
    // A rolled-back transaction is reusable rather than poisoned.
    require(discarded.stage(0, 0, 4.0) == OutcomeReason::Success && discarded.stage(1, 0, 5.0) == OutcomeReason::Success,
            "a rolled-back transaction refused new staging");
    require(discarded.commit().ok(), "a rolled-back transaction refused to commit");
    require(target[0] == 4.0 && target[1] == 5.0, "commit after rollback published the wrong values");

    // Restoring the pre-commit snapshot is a separate, fallible operation:
    // discarding staged state cannot fail, republishing can.
    require(discarded.restore_committed().ok(), "restoring the previous commit failed");
    require(target[0] == 1.0 && target[1] == 2.0, "restore did not republish the previous values");
}

void storage_failure_leaves_no_partial_mutation()
{
    std::vector<Real> target{1.0, 2.0, 3.0};
    // Refuses publication at the second value, which would leave a partial
    // write if the transaction published in place.
    FailingHostArrayStorage storage(target, 1);
    FieldTransaction transaction(storage, {1, 3, 0}, UpdateMode::Replace);
    for (std::size_t entry = 0; entry != 3; ++entry)
        require(transaction.stage(entry, 0, 10.0 + static_cast<Real>(entry)) == OutcomeReason::Success, "stage failed");
    const auto outcome = transaction.commit();
    require(!outcome.ok() && outcome.reason == OutcomeReason::StorageFailure, "a failing publication reported success");
    require(target[0] == 1.0 && target[1] == 2.0 && target[2] == 3.0,
            "a failed publication left a partially mutated target");
}
}

int main()
{
    try
    {
        index_map();
        read_only_views();
        replace_commits();
        rejects_bad_staging();
        sparse_insertion();
        canonical_accumulation();
        rollback_and_restore();
        storage_failure_leaves_no_partial_mutation();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
