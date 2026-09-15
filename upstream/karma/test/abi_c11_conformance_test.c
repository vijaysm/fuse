/* Compiled as C11, on purpose: the ABI headers must be consumable by a C
 * compiler with no C++ in sight. This file would fail to build if a header
 * leaked a C++ construct, which is the point of having it.
 *
 * The file stem deliberately differs from `abi_layout_test.C`. On a
 * case-insensitive filesystem - the default on macOS, where this is developed -
 * a `.c` and a `.C` file with the same stem are the same file, and the second
 * one written silently replaces the first.
 *
 * It also exercises the whole component lifecycle from C, so the call sequence
 * a language adapter must implement is proven callable from C rather than only
 * from the C++ that defined it.
 */
#include "karma_abi.h"
#include "karma_component_abi.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void require(int condition, const char* message)
{
    if (!condition)
    {
        printf("FAIL: %s\n", message);
        ++failures;
    }
}

_Static_assert(sizeof(void*) == 8, "these checks assume a 64-bit pointer");
_Static_assert(sizeof(karma_exact_time) == 16, "exact time is two int64 fields");
_Static_assert(sizeof(karma_status) == 4, "status is a fixed 32-bit width");
_Static_assert(sizeof(karma_version) == 12, "version is three uint32 fields");
_Static_assert(sizeof(karma_communicator) == 8, "the communicator token is 64-bit");
_Static_assert(offsetof(karma_exact_interval, end) == 16, "interval end follows begin");
_Static_assert(offsetof(karma_diagnostic, struct_size) == 0, "every record starts with struct_size");
_Static_assert(offsetof(karma_abi_request, struct_size) == 0, "every record starts with struct_size");
_Static_assert(offsetof(karma_field_table, struct_size) == 0, "every record starts with struct_size");
_Static_assert(offsetof(karma_component_table, struct_size) == 0, "every record starts with struct_size");

static karma_status host_report(void* context, const karma_diagnostic* diagnostic)
{
    (void)context;
    (void)diagnostic;
    return KARMA_STATUS_OK;
}

int main(void)
{
    karma_version version = karma_abi_version();
    require(version.major == 1, "the ABI major version is not 1");

    karma_abi_request request;
    karma_abi_agreement agreement;
    memset(&request, 0, sizeof(request));
    memset(&agreement, 0, sizeof(agreement));
    request.struct_size = (uint32_t)sizeof(request);
    request.abi_major = 1;
    request.abi_minor = 0;
    request.required_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS | KARMA_CAPABILITY_CHECKPOINT_RESTORE;
    request.optional_capabilities = KARMA_CAPABILITY_FRACTIONAL_STEPS | KARMA_CAPABILITY_CANCELLATION;
    request.host_table_size = (uint32_t)sizeof(karma_host_table);
    agreement.struct_size = (uint32_t)sizeof(agreement);
    require(karma_negotiate_abi(&request, &agreement) == KARMA_STATUS_OK, "negotiation from C failed");
    require(agreement.unsupported_required == 0, "a supported capability was refused");

    karma_plugin_query_result query;
    memset(&query, 0, sizeof(query));
    query.struct_size = (uint32_t)sizeof(query);
    require(karma_plugin_query(&query) == KARMA_STATUS_OK, "plugin query from C failed");
    require(query.component_table != NULL, "the query returned no component table");
    require(query.component_table->struct_size >= sizeof(karma_component_table),
            "the component table is smaller than this header expects");

    const karma_component_table* table = query.component_table;

    uint64_t element_count = 4;
    karma_component_config config;
    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.implementation_id = query.implementation_id;
    config.configuration = &element_count;
    config.configuration_size = sizeof(element_count);
    config.communicator = KARMA_COMMUNICATOR_NONE;
    config.granted_capabilities = agreement.granted_capabilities;

    void* instance = NULL;
    require(table->create(&config, &instance) == KARMA_STATUS_OK, "create from C failed");
    require(instance != NULL, "create returned no instance");

    karma_host_table host;
    memset(&host, 0, sizeof(host));
    host.struct_size = (uint32_t)sizeof(host);
    host.report = host_report;
    require(table->bind_host(instance, &host) == KARMA_STATUS_OK, "bind_host from C failed");

    karma_exact_time start;
    start.numerator = 0;
    start.denominator = 1;
    require(table->initialize(instance, start) == KARMA_STATUS_OK, "initialize from C failed");

    karma_component_metadata metadata;
    memset(&metadata, 0, sizeof(metadata));
    metadata.struct_size = (uint32_t)sizeof(metadata);
    require(table->metadata(instance, &metadata) == KARMA_STATUS_OK, "metadata from C failed");
    require(metadata.supports_checkpoint_restore == 1, "the reference component lost checkpoint support");

    double values[4] = {1.0, 2.0, 3.0, 4.0};
    karma_field_table field;
    memset(&field, 0, sizeof(field));
    field.struct_size = (uint32_t)sizeof(field);
    field.field_id = "state";
    field.element_type = KARMA_ELEMENT_F64;
    field.ownership = KARMA_OWNERSHIP_OWNED;
    field.access = KARMA_ACCESS_READ_WRITE;
    field.update = KARMA_UPDATE_REPLACE;
    field.rank = 1;
    field.components = 1;
    field.extents[0] = 4;
    field.byte_strides[0] = (int64_t)sizeof(double);
    field.data = values;
    field.element_count = 4;
    require(karma_validate_field_table(&field) == KARMA_STATUS_OK, "a valid table was rejected");
    require(table->import_fields(instance, &field, 1) == KARMA_STATUS_OK, "import from C failed");

    uint64_t token = 0;
    require(table->checkpoint(instance, &token) == KARMA_STATUS_OK, "checkpoint from C failed");

    karma_exact_interval interval;
    interval.begin.numerator = 0;
    interval.begin.denominator = 1;
    interval.end.numerator = 1;
    interval.end.denominator = 2;
    require(table->advance(instance, interval) == KARMA_STATUS_OK, "advance from C failed");
    require(table->commit(instance) == KARMA_STATUS_OK, "commit from C failed");

    double exported[4] = {0, 0, 0, 0};
    field.data = exported;
    require(table->export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export from C failed");
    require(fabs(exported[0] - 1.5) < 1e-12, "advance did not add half a unit of time");

    require(table->restore(instance, token) == KARMA_STATUS_OK, "restore from C failed");
    require(table->export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export after restore failed");
    require(fabs(exported[0] - 1.0) < 1e-12, "restore did not recover the checkpointed state");
    require(table->release_checkpoint(instance, token) == KARMA_STATUS_OK, "release from C failed");
    require(table->release_checkpoint(instance, token) == KARMA_STATUS_UNKNOWN_CHECKPOINT,
            "releasing twice did not report an unknown checkpoint");

    require(table->restore(instance, 424242) == KARMA_STATUS_UNKNOWN_CHECKPOINT,
            "restoring an unknown token was not refused");
    karma_diagnostic diagnostic;
    memset(&diagnostic, 0, sizeof(diagnostic));
    diagnostic.struct_size = (uint32_t)sizeof(diagnostic);
    require(table->diagnostics(instance, &diagnostic) == KARMA_STATUS_OK, "diagnostics from C failed");
    require(diagnostic.status == KARMA_STATUS_UNKNOWN_CHECKPOINT, "the diagnostic lost the status");
    require(strlen(diagnostic.message) > 0, "the diagnostic carried no message");
    require(strcmp(karma_status_name(diagnostic.status), "unknown checkpoint") == 0, "status naming is wrong");

    /* The capability gate in both directions. A component that was not granted
     * fractional steps must refuse a step that is not a whole multiple of the
     * clock, rather than silently taking it. */
    karma_component_config restricted_config = config;
    restricted_config.granted_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS;
    void* restricted = NULL;
    require(table->create(&restricted_config, &restricted) == KARMA_STATUS_OK, "restricted create failed");
    require(table->initialize(restricted, start) == KARMA_STATUS_OK, "restricted initialize failed");
    require(table->advance(restricted, interval) == KARMA_STATUS_UNSUPPORTED_CAPABILITY,
            "a fractional step was taken without the fractional-step capability");
    karma_exact_interval whole;
    whole.begin.numerator = 0;
    whole.begin.denominator = 1;
    whole.end.numerator = 2;
    whole.end.denominator = 1;
    require(table->advance(restricted, whole) == KARMA_STATUS_OK, "a whole step was refused");
    uint64_t restricted_token = 0;
    require(table->checkpoint(restricted, &restricted_token) == KARMA_STATUS_UNSUPPORTED_CAPABILITY,
            "checkpointing was allowed without the capability");
    table->destroy(restricted);

    int32_t cancelled = 1;
    require(table->poll_cancellation(instance, &cancelled) == KARMA_STATUS_OK, "cancellation poll failed");
    require(cancelled == 0, "an uncancelled component reported cancellation");

    table->destroy(instance);

    if (failures == 0) printf("C11 lifecycle ok\n");
    return failures == 0 ? 0 : 1;
}
