#include "karma_transfer_map.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace KARMA
{
KARMATransferMap::KARMATransferMap(TransferMapRepresentation representation, unsigned components)
    : _representation(representation), _components(components)
{
    if (_components == 0) KARMA_THROW(InvalidArgument("transfer map requires components"));
}

void KARMATransferMap::add_row(SparseInterpolationRow row)
{
    if (row.source.size() != row.weights.size()) KARMA_THROW(InvalidArgument("transfer row dimensions differ"));
    if (!std::is_sorted(row.source.begin(), row.source.end()))
        KARMA_THROW(InvalidArgument("transfer row source keys are not sorted"));
    if (std::adjacent_find(row.source.begin(), row.source.end()) != row.source.end())
        KARMA_THROW(InvalidArgument("transfer row contains duplicate source keys"));
    for (const Real weight : row.weights)
        if (!std::isfinite(weight)) KARMA_THROW(InvalidArgument("transfer row contains a nonfinite weight"));
    if (std::find_if(_rows.begin(), _rows.end(), [&](const auto& existing) { return existing.target == row.target; }) !=
        _rows.end())
        KARMA_THROW(InvalidArgument("transfer map contains duplicate target keys"));
    _rows.push_back(std::move(row));
}

KARMATransferMap KARMATransferMap::select_targets(std::span<const FieldEntryKey> owned_targets) const
{
    KARMATransferMap selected(_representation, _components);
    for (const auto& row : _rows)
        if (std::find(owned_targets.begin(), owned_targets.end(), row.target) != owned_targets.end())
            selected.add_row(row);
    selected.verify();
    return selected;
}

void KARMATransferMap::validate_distribution(std::span<const FieldEntryKey> target_owned,
                                               std::span<const FieldEntryKey> source_local) const
{
    for (const auto& row : _rows)
    {
        if (std::find(target_owned.begin(), target_owned.end(), row.target) == target_owned.end())
            KARMA_THROW(InvalidArgument("transfer map row is not owned by current target rank"));
        for (const auto& source : row.source)
            if (std::find(source_local.begin(), source_local.end(), source) == source_local.end())
                KARMA_THROW(InvalidArgument("transfer map source column is unavailable on current source rank"));
    }
}

std::map<FieldEntryKey, Real> KARMATransferMap::apply_b(
    const std::map<FieldEntryKey, Real>& source_values,
    std::span<const FieldEntryKey> target_owned,
    std::span<const FieldEntryKey> source_local) const
{
    if (_representation != TransferMapRepresentation::ProjectionFactors)
        KARMA_THROW(InvalidArgument("apply_b requires projection-factor map representation"));
    validate_distribution(target_owned, source_local);
    std::map<FieldEntryKey, Real> rhs;
    for (const auto& row : _rows)
    {
        Real value = 0.0;
        for (std::size_t index = 0; index < row.source.size(); ++index)
        {
            const auto found = source_values.find(row.source[index]);
            if (found == source_values.end()) KARMA_THROW(InvalidArgument("B source column value is unavailable"));
            value += row.weights[index] * found->second;
        }
        rhs.emplace(row.target, value);
    }
    return rhs;
}

void KARMATransferMap::verify() const
{
    for (const auto& row : _rows)
    {
        if (row.source.size() != row.weights.size()) KARMA_THROW(InvalidArgument("transfer row dimensions differ"));
        if (row.coverage == CoverageClass::Uncovered && !row.source.empty())
            KARMA_THROW(InvalidArgument("uncovered transfer row has source weights"));
    }
}
}  // namespace KARMA
