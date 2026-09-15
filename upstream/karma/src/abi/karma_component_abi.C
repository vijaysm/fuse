#include "karma_component_abi.h"

#include "karma_abi_records.h"
#include "karma_field_transaction.h"
#include "karma_semantics.h"

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <map>
#include <new>
#include <string>
#include <vector>

/* Exception containment.
 *
 * No exception may cross this boundary: the caller may be C, Fortran, or
 * Python, and unwinding through a frame that was not compiled to expect it is
 * undefined. Every ABI function body is wrapped, and the guard is unconditional
 * rather than dependent on KARMA_ENABLE_EXCEPTIONS, because that option does
 * not add -fno-exceptions - it only changes what KARMA_THROW does.
 *
 * A limitation worth stating plainly: with KARMA_ENABLE_EXCEPTIONS OFF,
 * KARMA_THROW calls std::abort(), so an internal KARMA failure terminates the
 * process before this guard can convert it to a status. The ABI implementation
 * therefore validates locally and returns statuses instead of relying on KARMA
 * internals to report errors. That keeps the boundary well-behaved in both
 * configurations for everything it validates itself.
 */
#define KARMA_ABI_GUARD_BEGIN                                                                                          \
    try                                                                                                                \
    {
#define KARMA_ABI_GUARD_END(instance)                                                                                  \
    }                                                                                                                  \
    catch (const std::exception& error)                                                                                 \
    {                                                                                                                  \
        return record_internal_error(instance, error.what());                                                          \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        return record_internal_error(instance, "unknown exception");                                                   \
    }

namespace
{
using namespace KARMA;

constexpr const char* reference_id = "karma.reference";
/* A fixture whose advance() throws, so exception containment is testable
 * rather than asserted. */
constexpr const char* throwing_id = "karma.reference.throwing";

struct ReferenceComponent
{
    std::string implementation_id;
    std::vector<Real> values;
    std::map<std::uint64_t, std::vector<Real>> checkpoints;
    std::uint64_t next_token = 1;
    Rational position{0, 1};
    bool initialized = false;
    bool cancelled = false;
    std::uint64_t granted_capabilities = 0;
    const karma_host_table* host = nullptr;
    karma_status last_status = KARMA_STATUS_OK;
    std::string last_message;
};

ReferenceComponent* cast(void* instance) noexcept { return static_cast<ReferenceComponent*>(instance); }

karma_status record_internal_error(void* instance, const char* what) noexcept
{
    if (instance != nullptr)
    {
        auto* component = cast(instance);
        component->last_status = KARMA_STATUS_INTERNAL_ERROR;
        /* Assigning a std::string can itself throw; a failure here must not
         * replace the status the caller is about to receive. */
        try
        {
            component->last_message = what != nullptr ? what : "unknown exception";
        }
        catch (...)
        {
        }
    }
    return KARMA_STATUS_INTERNAL_ERROR;
}

karma_status note(ReferenceComponent* component, karma_status status, const char* message) noexcept
{
    if (component != nullptr)
    {
        component->last_status = status;
        try
        {
            component->last_message = message != nullptr ? message : "";
        }
        catch (...)
        {
        }
    }
    return status;
}

} // namespace

extern "C" {

karma_status karma_validate_field_table(const karma_field_table* table)
{
    return KARMA::validate_abi_field_table(table);
}

static karma_status reference_create(const karma_component_config* config, void** instance)
{
    if (config == nullptr || instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    *instance = nullptr;
    const auto classification = classify_abi_struct_size(config->struct_size, sizeof(karma_component_config));
    if (classification != KARMA_STATUS_OK) return classification;
    if (config->implementation_id == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;

    const bool is_reference = std::strcmp(config->implementation_id, reference_id) == 0;
    const bool is_throwing = std::strcmp(config->implementation_id, throwing_id) == 0;
    if (!is_reference && !is_throwing) return KARMA_STATUS_UNKNOWN_FIELD;

    /* The reference component's configuration is one little-endian uint64
     * element count, which is enough to exercise the boundary without
     * inventing a configuration language before the configuration schema is defined. */
    if (config->configuration_size != sizeof(std::uint64_t) || config->configuration == nullptr)
        return KARMA_STATUS_INVALID_ARGUMENT;
    std::uint64_t count = 0;
    std::memcpy(&count, config->configuration, sizeof(count));
    if (count == 0 || count > (UINT64_C(1) << 24)) return KARMA_STATUS_INVALID_ARGUMENT;

    KARMA_ABI_GUARD_BEGIN
    auto* component = new ReferenceComponent();
    component->implementation_id = config->implementation_id;
    component->values.assign(static_cast<std::size_t>(count), Real{0});
    component->granted_capabilities = config->granted_capabilities;
    *instance = component;
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(nullptr)
}

static karma_status reference_bind_host(void* instance, const karma_host_table* host)
{
    if (instance == nullptr || host == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto classification = classify_abi_struct_size(host->struct_size, sizeof(karma_host_table));
    if (classification != KARMA_STATUS_OK) return classification;
    cast(instance)->host = host;
    return KARMA_STATUS_OK;
}

static karma_status reference_initialize(void* instance, karma_exact_time start)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    if (start.denominator == 0) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    KARMA_ABI_GUARD_BEGIN
    const auto position = make_rational(start.numerator, start.denominator);
    if (!position.value) return note(component, KARMA_STATUS_INVALID_ARGUMENT, "start time is not representable");
    component->position = *position.value;
    component->initialized = true;
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_metadata(void* instance, karma_component_metadata* metadata)
{
    if (instance == nullptr || metadata == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto classification = classify_abi_struct_size(metadata->struct_size, sizeof(karma_component_metadata));
    if (classification != KARMA_STATUS_OK) return classification;
    metadata->implementation_version = karma_abi_version();
    metadata->provided_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS |
                                      KARMA_CAPABILITY_CHECKPOINT_RESTORE |
                                      KARMA_CAPABILITY_FRACTIONAL_STEPS | KARMA_CAPABILITY_CANCELLATION;
    metadata->field_count = 1;
    metadata->supports_fractional_steps = 1;
    metadata->supports_checkpoint_restore = 1;
    return KARMA_STATUS_OK;
}

static karma_status reference_import_fields(void* instance, const karma_field_table* tables, std::uint32_t count)
{
    if (instance == nullptr || (count != 0 && tables == nullptr)) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    if (!component->initialized) return note(component, KARMA_STATUS_NOT_INITIALIZED, "import before initialize");

    KARMA_ABI_GUARD_BEGIN
    for (std::uint32_t index = 0; index != count; ++index)
    {
        const auto& table = tables[index];
        const auto valid = validate_abi_field_table(&table);
        if (valid != KARMA_STATUS_OK) return note(component, valid, "invalid import table");
        if (std::strcmp(table.field_id, "state") != 0)
            return note(component, KARMA_STATUS_UNKNOWN_FIELD, "unknown field identifier");
        if (table.element_type != KARMA_ELEMENT_F64)
            return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "state is a double field");
        if (table.access == KARMA_ACCESS_WRITE_ONLY)
            return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "cannot import from a write-only table");
        if (table.element_count != component->values.size())
            return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "import size differs from the state");

        /* Published through the transactional API, so a rejected import leaves
         * the component's state untouched rather than half-written. */
        HostArrayStorage storage(component->values);
        const auto mode = table.update == KARMA_UPDATE_ACCUMULATE
                              ? UpdateMode::Accumulate
                              : (table.update == KARMA_UPDATE_SPARSE_REPLACE ? UpdateMode::SparseReplace
                                                                               : UpdateMode::Replace);
        FieldTransaction transaction(storage, {1, component->values.size(), 0}, mode);
        const auto* base = static_cast<const unsigned char*>(table.data);
        for (std::size_t entry = 0; entry != component->values.size(); ++entry)
        {
            double value = 0;
            std::memcpy(&value, base + static_cast<std::size_t>(table.byte_strides[0]) * entry, sizeof(value));
            const auto staged = transaction.stage(entry, 0, value);
            if (staged == OutcomeReason::NonfiniteValue)
                return note(component, KARMA_STATUS_NONFINITE_VALUE, "imported value is not finite");
            if (staged != OutcomeReason::Success)
                return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "import staging was rejected");
        }
        const auto committed = transaction.commit();
        if (!committed.ok())
        {
            if (committed.reason == OutcomeReason::IncompleteUpdate)
                return note(component, KARMA_STATUS_INCOMPLETE_UPDATE, "import did not cover the state");
            return note(component, KARMA_STATUS_STORAGE_FAILURE, "import failed to publish");
        }
    }
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_export_fields(void* instance, const karma_field_table* tables, std::uint32_t count)
{
    if (instance == nullptr || (count != 0 && tables == nullptr)) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    if (!component->initialized) return note(component, KARMA_STATUS_NOT_INITIALIZED, "export before initialize");

    KARMA_ABI_GUARD_BEGIN
    for (std::uint32_t index = 0; index != count; ++index)
    {
        const auto& table = tables[index];
        const auto valid = validate_abi_field_table(&table);
        if (valid != KARMA_STATUS_OK) return note(component, valid, "invalid export table");
        if (std::strcmp(table.field_id, "state") != 0)
            return note(component, KARMA_STATUS_UNKNOWN_FIELD, "unknown field identifier");
        if (table.access == KARMA_ACCESS_READ_ONLY)
            return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "cannot export into a read-only table");
        if (table.element_count != component->values.size())
            return note(component, KARMA_STATUS_MALFORMED_FIELD_TABLE, "export size differs from the state");

        auto* base = static_cast<unsigned char*>(table.data);
        for (std::size_t entry = 0; entry != component->values.size(); ++entry)
        {
            const double value = static_cast<double>(component->values[entry]);
            std::memcpy(base + static_cast<std::size_t>(table.byte_strides[0]) * entry, &value, sizeof(value));
        }
    }
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_advance(void* instance, karma_exact_interval interval)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    if (!component->initialized) return note(component, KARMA_STATUS_NOT_INITIALIZED, "advance before initialize");
    if (component->cancelled) return note(component, KARMA_STATUS_CANCELLED, "component was cancelled");
    if (interval.begin.denominator == 0 || interval.end.denominator == 0)
        return note(component, KARMA_STATUS_INVALID_ARGUMENT, "interval denominator is zero");

    KARMA_ABI_GUARD_BEGIN
    if (component->implementation_id == throwing_id)
        throw std::runtime_error("the throwing reference component always throws in advance");

    const auto begin = make_rational(interval.begin.numerator, interval.begin.denominator);
    const auto end = make_rational(interval.end.numerator, interval.end.denominator);
    if (!begin.value || !end.value)
        return note(component, KARMA_STATUS_INVALID_ARGUMENT, "interval is not representable");
    const auto length = rational_subtract(*end.value, *begin.value);
    if (!length.value) return note(component, KARMA_STATUS_INVALID_ARGUMENT, "interval length overflows");
    if (rational_compare(*length.value, Rational{0, 1}) == std::strong_ordering::less)
        return note(component, KARMA_STATUS_INVALID_ARGUMENT, "interval ends before it begins");

    /* A step that is not a whole multiple of the clock needs the fractional
     * step capability. Checking it here rather than trusting the schedule is
     * the point of a capability bit. */
    if (length.value->denominator != 1 &&
        (component->granted_capabilities & KARMA_CAPABILITY_FRACTIONAL_STEPS) == 0)
        return note(component, KARMA_STATUS_UNSUPPORTED_CAPABILITY, "fractional steps were not granted");

    const double elapsed = rational_to_double(*length.value);
    for (auto& value : component->values) value += elapsed;
    component->position = *end.value;
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_advance_seconds(void* instance, double begin, double end)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    KARMA_ABI_GUARD_BEGIN
    /* The shadow converts and refuses what it cannot represent exactly within
     * a bounded denominator; it never rounds silently. */
    const auto exact_begin = rational_from_dyadic_bounded(begin, 32);
    const auto exact_end = rational_from_dyadic_bounded(end, 32);
    if (!exact_begin.value || !exact_end.value)
        return note(component, KARMA_STATUS_INVALID_ARGUMENT, "a seconds value is not exactly representable");
    karma_exact_interval interval;
    interval.begin.numerator = exact_begin.value->numerator;
    interval.begin.denominator = exact_begin.value->denominator;
    interval.end.numerator = exact_end.value->numerator;
    interval.end.denominator = exact_end.value->denominator;
    return reference_advance(instance, interval);
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_commit(void* instance)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    if (!component->initialized) return note(component, KARMA_STATUS_NOT_INITIALIZED, "commit before initialize");
    return KARMA_STATUS_OK;
}

static karma_status reference_checkpoint(void* instance, std::uint64_t* token)
{
    if (instance == nullptr || token == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    if ((component->granted_capabilities & KARMA_CAPABILITY_CHECKPOINT_RESTORE) == 0)
        return note(component, KARMA_STATUS_UNSUPPORTED_CAPABILITY, "checkpointing was not granted");
    KARMA_ABI_GUARD_BEGIN
    const auto assigned = component->next_token++;
    component->checkpoints[assigned] = component->values;
    *token = assigned;
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_restore(void* instance, std::uint64_t token)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    KARMA_ABI_GUARD_BEGIN
    const auto found = component->checkpoints.find(token);
    if (found == component->checkpoints.end())
        return note(component, KARMA_STATUS_UNKNOWN_CHECKPOINT, "no such checkpoint");
    component->values = found->second;
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_release_checkpoint(void* instance, std::uint64_t token)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    KARMA_ABI_GUARD_BEGIN
    if (component->checkpoints.erase(token) == 0)
        return note(component, KARMA_STATUS_UNKNOWN_CHECKPOINT, "no such checkpoint");
    return KARMA_STATUS_OK;
    KARMA_ABI_GUARD_END(instance)
}

static karma_status reference_diagnostics(void* instance, karma_diagnostic* diagnostic)
{
    if (instance == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    auto* component = cast(instance);
    return set_abi_diagnostic(diagnostic, component->last_status, component->last_message.c_str());
}

static karma_status reference_poll_cancellation(void* instance, std::int32_t* cancelled)
{
    if (instance == nullptr || cancelled == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    *cancelled = cast(instance)->cancelled ? 1 : 0;
    return KARMA_STATUS_OK;
}

static void reference_destroy(void* instance)
{
    /* Nonthrowing by contract: destruction cannot report and must not escape. */
    delete cast(instance);
}

karma_status karma_plugin_query(karma_plugin_query_result* result)
{
    if (result == nullptr) return KARMA_STATUS_INVALID_ARGUMENT;
    const auto classification = classify_abi_struct_size(result->struct_size, sizeof(karma_plugin_query_result));
    if (classification != KARMA_STATUS_OK) return classification;

    static const karma_component_table table = {
        static_cast<std::uint32_t>(sizeof(karma_component_table)),
        reference_create,
        reference_bind_host,
        reference_initialize,
        reference_metadata,
        reference_import_fields,
        reference_export_fields,
        reference_advance,
        reference_commit,
        reference_checkpoint,
        reference_restore,
        reference_release_checkpoint,
        reference_diagnostics,
        reference_poll_cancellation,
        reference_destroy,
        reference_advance_seconds};

    result->abi_major = KARMA_ABI_MAJOR;
    result->abi_minor = KARMA_ABI_MINOR;
    result->provided_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS |
                                    KARMA_CAPABILITY_CHECKPOINT_RESTORE |
                                    KARMA_CAPABILITY_FRACTIONAL_STEPS | KARMA_CAPABILITY_CANCELLATION;
    result->implementation_id = reference_id;
    result->implementation_version = karma_abi_version();
    result->component_table = &table;
    return KARMA_STATUS_OK;
}

} /* extern "C" */
