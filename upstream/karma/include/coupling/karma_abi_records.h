#ifndef KARMA_COUPLING_ABI_RECORDS_H
#define KARMA_COUPLING_ABI_RECORDS_H

#include "karma_component_abi.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

/** Operations on ABI records, implemented once for two callers.
 *
 * Why a header rather than a source file: `libkarma_abi` links `libkarma`
 * privately, so `libkarma` cannot call back into the ABI library without
 * creating a dependency cycle - and the host-side component runtime, which
 * lives in `libkarma`, needs exactly the same record validation the exported
 * C entry points perform. Inline definitions give both sides one
 * implementation with no cycle and no second copy of the rules.
 *
 * These are pure functions over plain records: no allocation, no state, and
 * nothing from PETSc, MOAB, or MPI. The exported `karma_*` symbols in
 * `libkarma_abi` are thin forwards to them.
 */
namespace KARMA
{
/** A zero or wild size is a corrupted record, not a very new one; separating
 * the two lets a caller tell "you are too old" from "this is not a record". */
[[nodiscard]] inline karma_status classify_abi_struct_size(std::uint32_t size, std::uint32_t core) noexcept
{
    if (size == 0 || size > KARMA_ABI_MAX_STRUCT_SIZE) return KARMA_STATUS_MALFORMED_SIZE;
    if (size < core) return KARMA_STATUS_TRUNCATED_PREFIX;
    return KARMA_STATUS_OK;
}

[[nodiscard]] inline std::size_t abi_element_size(std::int32_t element_type) noexcept
{
    switch (element_type)
    {
        case KARMA_ELEMENT_F64: return 8;
        case KARMA_ELEMENT_I64: return 8;
        case KARMA_ELEMENT_I32: return 4;
        default: return 0;
    }
}

[[nodiscard]] inline karma_status validate_abi_field_table(const karma_field_table* table) noexcept
{
    if (table == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto classification = classify_abi_struct_size(table->struct_size, sizeof(karma_field_table));
    if (classification != KARMA_STATUS_OK) return classification;

    if (table->field_id == nullptr) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    const auto width = abi_element_size(table->element_type);
    if (width == 0) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->rank == 0 || table->rank > KARMA_ABI_FIELD_MAX_RANK) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->components == 0) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->ownership < KARMA_OWNERSHIP_OWNED || table->ownership > KARMA_OWNERSHIP_REPLICATED)
        return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->access < KARMA_ACCESS_READ_ONLY || table->access > KARMA_ACCESS_WRITE_ONLY)
        return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->update < KARMA_UPDATE_REPLACE || table->update > KARMA_UPDATE_ACCUMULATE)
        return KARMA_STATUS_MALFORMED_FIELD_TABLE;

    std::uint64_t entries = 1;
    for (std::uint32_t dimension = 0; dimension != table->rank; ++dimension)
    {
        if (table->extents[dimension] <= 0) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
        // A stride narrower than one element would make two logical entries
        // overlap in memory, which no caller can mean.
        const auto stride = table->byte_strides[dimension];
        const auto magnitude = static_cast<std::uint64_t>(stride < 0 ? -stride : stride);
        if (stride == 0 || magnitude < width) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
        entries *= static_cast<std::uint64_t>(table->extents[dimension]);
    }
    if (table->element_count != entries * table->components) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->element_count != 0 && table->data == nullptr) return KARMA_STATUS_MALFORMED_FIELD_TABLE;

    if (table->key_bytes == nullptr)
    {
        if (table->key_bytes_size != 0 || table->key_record_size != 0) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
        return KARMA_STATUS_OK;
    }
    if (table->key_record_size == 0) return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    if (table->key_bytes_size != static_cast<std::uint64_t>(table->key_record_size) * entries)
        return KARMA_STATUS_MALFORMED_FIELD_TABLE;
    return KARMA_STATUS_OK;
}

[[nodiscard]] inline karma_status set_abi_diagnostic(karma_diagnostic* diagnostic, karma_status status,
                                                        const char* text) noexcept
{
    if (diagnostic == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto classification = classify_abi_struct_size(diagnostic->struct_size, sizeof(karma_diagnostic));
    if (classification != KARMA_STATUS_OK) return classification;

    diagnostic->status = status;
    diagnostic->truncated = 0;
    diagnostic->message[0] = '\0';
    if (text == nullptr) return KARMA_STATUS_OK;

    const std::size_t length = std::strlen(text);
    const std::size_t capacity = KARMA_ABI_DIAGNOSTIC_CAPACITY - 1;
    const std::size_t copied = length < capacity ? length : capacity;
    std::memcpy(diagnostic->message, text, copied);
    diagnostic->message[copied] = '\0';
    // Always NUL-terminated, and the flag says whether anything was lost: a
    // silently truncated diagnostic reads as complete, which is worse than
    // none.
    if (copied < length) diagnostic->truncated = 1;
    return KARMA_STATUS_OK;
}
} // namespace KARMA

#endif
