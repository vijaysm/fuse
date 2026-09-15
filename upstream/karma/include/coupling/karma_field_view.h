#ifndef KARMA_COUPLING_FIELD_VIEW_H
#define KARMA_COUPLING_FIELD_VIEW_H

#include "karma_collective_outcome.h"
#include "karma_field_identity.h"

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

/** Storage-neutral read access to field values, and setup-time resolution of
 * stable keys to dense indices.
 *
 * Two rules shape this. First, a hot path must not look up a stable key: keys
 * are resolved once during setup into dense indices, and the lookup counter
 * exists so a test can prove a warmed path performed no lookups. Second,
 * ownership is positional - owned entries precede ghost entries - so a caller
 * never needs a predicate per entry to know what it may publish.
 */
namespace KARMA
{
struct FieldViewLayout
{
    unsigned components = 1;
    std::size_t owned_entries = 0;
    std::size_t ghost_entries = 0;

    [[nodiscard]] std::size_t total_entries() const noexcept { return owned_entries + ghost_entries; }
    [[nodiscard]] std::size_t total_values() const noexcept { return total_entries() * components; }
    bool operator==(const FieldViewLayout&) const = default;
};

/** Dense, component-minor view of field values. Non-owning. */
class FieldView
{
  public:
    FieldView(std::span<const Real> values, FieldViewLayout layout) noexcept : _values(values), _layout(layout) {}

    [[nodiscard]] std::span<const Real> values() const noexcept { return _values; }
    [[nodiscard]] const FieldViewLayout& layout() const noexcept { return _layout; }
    [[nodiscard]] bool is_owned(std::size_t entry) const noexcept { return entry < _layout.owned_entries; }
    [[nodiscard]] std::span<const Real> entry_values(std::size_t entry) const noexcept;

  private:
    std::span<const Real> _values;
    FieldViewLayout _layout;
};

/** Setup-time map from stable field-entry keys to dense indices. */
/** Declared before `FieldIndexMap` and defined after it, because it holds an
 * optional of that class. */
struct FieldIndexMapBuildResult;

class FieldIndexMap
{
  public:
    using BuildResult = FieldIndexMapBuildResult;

    /** Rejects duplicate keys, which would make dense resolution ambiguous,
     * and a layout whose entry count disagrees with the keys given. */
    [[nodiscard]] static FieldIndexMapBuildResult build(std::span<const FieldEntryKey> entries,
                                                        FieldViewLayout layout);

    [[nodiscard]] std::size_t entry_count() const noexcept { return _entries.size(); }
    [[nodiscard]] const FieldViewLayout& layout() const noexcept { return _layout; }
    [[nodiscard]] bool is_owned(std::size_t entry) const noexcept { return entry < _layout.owned_entries; }
    [[nodiscard]] const FieldEntryKey& key(std::size_t entry) const noexcept { return _entries[entry]; }

    /** Setup-path lookup. Counted, so a test can assert that a warmed hot path
     * performed none. */
    [[nodiscard]] std::optional<std::size_t> find(const FieldEntryKey& key) const;

    [[nodiscard]] std::size_t lookup_count() const noexcept { return _lookups; }
    void reset_lookup_count() noexcept { _lookups = 0; }

  private:
    FieldIndexMap() = default;

    std::vector<FieldEntryKey> _entries;
    /** Keys sorted with their dense index, for binary search without
     * disturbing the caller's entry order. */
    std::vector<std::pair<FieldEntryKey, std::size_t>> _sorted;
    FieldViewLayout _layout;
    mutable std::size_t _lookups = 0;
};

struct FieldIndexMapBuildResult
{
    std::optional<FieldIndexMap> map;
    OutcomeReason reason = OutcomeReason::Success;
};
} // namespace KARMA

#endif
