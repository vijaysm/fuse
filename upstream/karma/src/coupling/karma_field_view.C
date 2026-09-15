#include "karma_field_view.h"

#include <algorithm>

namespace KARMA
{
std::span<const Real> FieldView::entry_values(std::size_t entry) const noexcept
{
    if (_layout.components == 0 || entry >= _layout.total_entries()) return {};
    const auto offset = entry * _layout.components;
    if (offset + _layout.components > _values.size()) return {};
    return _values.subspan(offset, _layout.components);
}

FieldIndexMapBuildResult FieldIndexMap::build(std::span<const FieldEntryKey> entries, FieldViewLayout layout)
{
    if (layout.components == 0) return {std::nullopt, OutcomeReason::InvalidArgument};
    if (layout.total_entries() != entries.size()) return {std::nullopt, OutcomeReason::InvalidArgument};

    FieldIndexMap map;
    map._layout = layout;
    map._entries.assign(entries.begin(), entries.end());
    map._sorted.reserve(entries.size());
    for (std::size_t index = 0; index != entries.size(); ++index) map._sorted.emplace_back(entries[index], index);
    std::sort(map._sorted.begin(), map._sorted.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });
    const auto duplicate = std::adjacent_find(map._sorted.begin(), map._sorted.end(),
                                              [](const auto& left, const auto& right) {
                                                  return left.first == right.first;
                                              });
    if (duplicate != map._sorted.end()) return {std::nullopt, OutcomeReason::DuplicateEntry};
    return {std::move(map), OutcomeReason::Success};
}

std::optional<std::size_t> FieldIndexMap::find(const FieldEntryKey& key) const
{
    ++_lookups;
    const auto position = std::lower_bound(_sorted.begin(), _sorted.end(), key,
                                           [](const auto& entry, const FieldEntryKey& sought) {
                                               return entry.first < sought;
                                           });
    if (position == _sorted.end() || !(position->first == key)) return std::nullopt;
    return position->second;
}
} // namespace KARMA
