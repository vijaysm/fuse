#include "karma_abi.h"

/* For the host and component table sizes reported by negotiation. */
#include "karma_component_abi.h"

#include "karma_abi_records.h"

#include <cstddef>
#include <cstdint>

#include <cstring>

namespace
{
/* The smallest record this version can read. A caller may send more (a future
 * tail, ignored) but not less. */
constexpr std::uint32_t request_core_size = sizeof(karma_abi_request);
constexpr std::uint32_t agreement_core_size = sizeof(karma_abi_agreement);

constexpr std::uint64_t supported_capabilities =
    KARMA_CAPABILITY_HOST_ARRAY_FIELDS | KARMA_CAPABILITY_KEYED_FIELDS |
    KARMA_CAPABILITY_CHECKPOINT_RESTORE | KARMA_CAPABILITY_FRACTIONAL_STEPS |
    KARMA_CAPABILITY_CANCELLATION | KARMA_CAPABILITY_COMMUNICATOR;

} // namespace

extern "C" {

karma_version karma_abi_version(void)
{
    karma_version version;
    version.major = KARMA_ABI_MAJOR;
    version.minor = KARMA_ABI_MINOR;
    version.patch = KARMA_ABI_PATCH;
    return version;
}

std::uint64_t karma_abi_supported_capabilities(void) { return supported_capabilities; }

const char* karma_status_name(karma_status status)
{
    switch (status)
    {
        case KARMA_STATUS_OK: return "ok";
        case KARMA_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case KARMA_STATUS_TRUNCATED_PREFIX: return "truncated prefix";
        case KARMA_STATUS_MALFORMED_SIZE: return "malformed size";
        case KARMA_STATUS_INCOMPATIBLE_MAJOR: return "incompatible major version";
        case KARMA_STATUS_UNSUPPORTED_CAPABILITY: return "unsupported required capability";
        case KARMA_STATUS_TABLE_TOO_SMALL: return "table too small";
        case KARMA_STATUS_UNKNOWN_FIELD: return "unknown field";
        case KARMA_STATUS_MALFORMED_FIELD_TABLE: return "malformed field table";
        case KARMA_STATUS_NONFINITE_VALUE: return "nonfinite value";
        case KARMA_STATUS_INCOMPLETE_UPDATE: return "incomplete update";
        case KARMA_STATUS_STORAGE_FAILURE: return "storage failure";
        case KARMA_STATUS_UNKNOWN_CHECKPOINT: return "unknown checkpoint";
        case KARMA_STATUS_NOT_INITIALIZED: return "not initialized";
        case KARMA_STATUS_CANCELLED: return "cancelled";
        case KARMA_STATUS_INTERNAL_ERROR: return "internal error";
        default: return "";
    }
}

karma_status karma_diagnostic_set(karma_diagnostic* diagnostic, karma_status status, const char* text)
{
    return KARMA::set_abi_diagnostic(diagnostic, status, text);
}

karma_status karma_negotiate_abi(const karma_abi_request* request, karma_abi_agreement* agreement)
{
    if (request == nullptr || agreement == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;

    const auto agreement_classification = KARMA::classify_abi_struct_size(agreement->struct_size, agreement_core_size);
    if (agreement_classification != KARMA_STATUS_OK) return agreement_classification;
    const auto request_classification = KARMA::classify_abi_struct_size(request->struct_size, request_core_size);
    if (request_classification != KARMA_STATUS_OK) return request_classification;

    /* Fill what is knowable before any refusal, so a refused caller still
     * learns the sizes and the version it should have asked for. */
    agreement->abi_major = KARMA_ABI_MAJOR;
    agreement->abi_minor = request->abi_minor < KARMA_ABI_MINOR ? request->abi_minor : KARMA_ABI_MINOR;
    agreement->granted_capabilities = 0;
    agreement->unsupported_required = 0;
    agreement->required_host_table_size = static_cast<std::uint32_t>(sizeof(karma_host_table));
    agreement->plugin_table_size = static_cast<std::uint32_t>(sizeof(karma_component_table));

    if (request->abi_major != KARMA_ABI_MAJOR) return KARMA_STATUS_INCOMPATIBLE_MAJOR;

    const std::uint64_t unsupported = request->required_capabilities & ~supported_capabilities;
    if (unsupported != 0)
    {
        /* Reported rather than ignored: a participant that asked for a
         * guarantee must not be told it was granted. */
        agreement->unsupported_required = unsupported;
        return KARMA_STATUS_UNSUPPORTED_CAPABILITY;
    }

    agreement->granted_capabilities =
        (request->required_capabilities | request->optional_capabilities) & supported_capabilities;

    if (request->host_table_size < sizeof(karma_host_table)) return KARMA_STATUS_TABLE_TOO_SMALL;
    return KARMA_STATUS_OK;
}

} /* extern "C" */
