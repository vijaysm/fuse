#include "karma_field_transaction.h"

#include <algorithm>
#include <bit>
#include <cmath>

namespace KARMA
{
FieldStorage::~FieldStorage() = default;

OutcomeReason HostArrayStorage::read_into(std::span<Real> destination) const
{
    if (destination.size() != _target.size()) return OutcomeReason::InvalidArgument;
    std::copy(_target.begin(), _target.end(), destination.begin());
    return OutcomeReason::Success;
}

OutcomeReason HostArrayStorage::publish(std::span<const Real> values)
{
    if (values.size() != _target.size()) return OutcomeReason::InvalidArgument;
    std::copy(values.begin(), values.end(), _target.begin());
    return OutcomeReason::Success;
}

OutcomeReason FailingHostArrayStorage::read_into(std::span<Real> destination) const
{
    if (destination.size() != _target.size()) return OutcomeReason::InvalidArgument;
    std::copy(_target.begin(), _target.end(), destination.begin());
    return OutcomeReason::Success;
}

OutcomeReason FailingHostArrayStorage::publish(std::span<const Real> values)
{
    if (values.size() != _target.size()) return OutcomeReason::InvalidArgument;
    // Refuse without writing anything: a storage backend that writes part of
    // the values and then fails would break the transaction's guarantee, so the
    // fixture models a backend that stages internally like the real ones.
    if (_failing_value < values.size()) return OutcomeReason::StorageFailure;
    std::copy(values.begin(), values.end(), _target.begin());
    return OutcomeReason::Success;
}

namespace
{
/** A total order on doubles that is independent of arrival order: ascending
 * magnitude first, then the bit pattern to break ties deterministically.
 * Summing smallest-magnitude first also limits cancellation error. */
bool canonical_before(Real left, Real right) noexcept
{
    const auto left_magnitude = std::fabs(left);
    const auto right_magnitude = std::fabs(right);
    if (left_magnitude != right_magnitude) return left_magnitude < right_magnitude;
    return std::bit_cast<std::uint64_t>(left) < std::bit_cast<std::uint64_t>(right);
}
}

FieldTransaction::FieldTransaction(FieldStorage& storage, FieldViewLayout layout, UpdateMode mode)
    : _storage(&storage), _layout(layout), _mode(mode)
{
    const auto values = _layout.total_values();
    if (_mode != UpdateMode::Accumulate)
    {
        _staged.assign(values, Real{0});
        _has_staged.assign(values, false);
    }
    _publication.assign(values, Real{0});
}

OutcomeReason FieldTransaction::stage(std::size_t entry, unsigned component, Real value) noexcept
{
    if (!std::isfinite(value)) return OutcomeReason::NonfiniteValue;
    if (entry >= _layout.total_entries() || component >= _layout.components) return OutcomeReason::UnknownEntry;
    const auto slot = entry * _layout.components + component;
    if (_mode != UpdateMode::Accumulate)
    {
        if (_has_staged[slot]) return OutcomeReason::DuplicateEntry;
        _staged[slot] = value;
        _has_staged[slot] = true;
        return OutcomeReason::Success;
    }
    _contributions.emplace_back(slot, value);
    return OutcomeReason::Success;
}

OutcomeReason FieldTransaction::stage_by_key(const FieldIndexMap& map, const FieldEntryKey& key, unsigned component,
                                             Real value)
{
    const auto entry = map.find(key);
    if (!entry) return OutcomeReason::UnknownEntry;
    return stage(*entry, component, value);
}

OutcomeReason FieldTransaction::validate_local() const noexcept
{
    if (_storage == nullptr) return OutcomeReason::InvalidArgument;
    if (_storage->value_count() != _layout.total_values()) return OutcomeReason::InvalidArgument;
    if (_mode == UpdateMode::Replace)
    {
        // Replacement must cover every value. A partial replacement would
        // publish a mixture of this step's values and the previous step's,
        // which is the failure mode this check exists to prevent.
        for (const bool staged : _has_staged)
            if (!staged) return OutcomeReason::IncompleteUpdate;
    }
    return OutcomeReason::Success;
}

OutcomeReason FieldTransaction::build_published_values(std::span<Real> destination) const noexcept
{
    if (destination.size() != _layout.total_values()) return OutcomeReason::InvalidArgument;
    if (_mode == UpdateMode::Replace)
    {
        std::copy(_staged.begin(), _staged.end(), destination.begin());
        return OutcomeReason::Success;
    }

    const auto base = _storage->read_into(destination);
    if (_mode == UpdateMode::SparseReplace)
    {
        if (base != OutcomeReason::Success) return base;
        for (std::size_t slot = 0; slot != _staged.size(); ++slot)
            if (_has_staged[slot]) destination[slot] = _staged[slot];
        return OutcomeReason::Success;
    }

    if (base != OutcomeReason::Success) return base;

    // Group contributions by slot, then sum each slot in canonical order, so
    // the published value does not depend on the order values were staged in.
    std::vector<std::pair<std::size_t, Real>> ordered(_contributions);
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const auto& left, const auto& right) { return left.first < right.first; });
    std::size_t index = 0;
    std::vector<Real> slot_values;
    while (index != ordered.size())
    {
        const auto slot = ordered[index].first;
        slot_values.clear();
        while (index != ordered.size() && ordered[index].first == slot)
        {
            slot_values.push_back(ordered[index].second);
            ++index;
        }
        std::sort(slot_values.begin(), slot_values.end(), canonical_before);
        Real sum = destination[slot];
        for (const auto value : slot_values) sum += value;
        if (!std::isfinite(sum)) return OutcomeReason::NonfiniteValue;
        destination[slot] = sum;
    }
    return OutcomeReason::Success;
}

LocalOutcome FieldTransaction::commit() noexcept
{
    const auto validation = validate_local();
    if (validation != OutcomeReason::Success) return {validation, 0};

    _publication.assign(_layout.total_values(), Real{0});
    const auto built = build_published_values(_publication);
    if (built != OutcomeReason::Success) return {built, 0};

    // Snapshot before publishing, so a later restore has somewhere to go.
    std::vector<Real> snapshot(_layout.total_values(), Real{0});
    const auto read = _storage->read_into(snapshot);
    if (read != OutcomeReason::Success) return {read, 0};

    const auto published = _storage->publish(_publication);
    if (published != OutcomeReason::Success) return {published, 0};

    _previous = std::move(snapshot);
    _has_previous = true;
    rollback();
    return {};
}

void FieldTransaction::rollback() noexcept
{
    if (_mode != UpdateMode::Accumulate)
    {
        std::fill(_staged.begin(), _staged.end(), Real{0});
        std::fill(_has_staged.begin(), _has_staged.end(), false);
    }
    _contributions.clear();
}

LocalOutcome FieldTransaction::restore_committed() noexcept
{
    if (!_has_previous) return {OutcomeReason::RollbackFailed, 0};
    const auto published = _storage->publish(_previous);
    if (published != OutcomeReason::Success) return {OutcomeReason::RollbackFailed, 0};
    return {};
}

ReconciledOutcome FieldTransaction::commit_collective(MPI_Comm communicator) noexcept
{
    // Preparation: agree that every rank can commit. No rank publishes while any
    // rank's validation has failed.
    const auto agreed = reconcile_outcome(communicator, {validate_local(), 0});
    if (!agreed.ok()) return agreed;

    // Publication: publish, then reconcile the publication outcome so every rank
    // learns of a failure that happened elsewhere.
    const auto local = commit();
    return reconcile_outcome(communicator, local);
}

std::size_t FieldTransaction::staged_count() const noexcept
{
    if (_mode != UpdateMode::Accumulate)
        return static_cast<std::size_t>(std::count(_has_staged.begin(), _has_staged.end(), true));
    return _contributions.size();
}
} // namespace KARMA
