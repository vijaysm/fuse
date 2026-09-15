/* A pure-C consumer of the installed KARMA ABI.
 *
 * This is the installation-check program for the candidate ABI. It is compiled as C11 with
 * one include path - the install prefix - and it links only libkarma_abi. It
 * uses the installed include spelling <karma/abi/...>, which works because the
 * ABI headers include each other as siblings in the same directory rather than
 * flatly like the rest of KARMA's headers.
 */
#include <karma/abi/karma_abi.h>
#include <karma/abi/karma_component_abi.h>

#include <math.h>
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

int main(void)
{
    karma_abi_request request;
    karma_abi_agreement agreement;
    memset(&request, 0, sizeof(request));
    memset(&agreement, 0, sizeof(agreement));
    request.struct_size = (uint32_t)sizeof(request);
    request.abi_major = KARMA_ABI_MAJOR;
    request.required_capabilities = KARMA_CAPABILITY_HOST_ARRAY_FIELDS | KARMA_CAPABILITY_CHECKPOINT_RESTORE;
    request.host_table_size = (uint32_t)sizeof(karma_host_table);
    agreement.struct_size = (uint32_t)sizeof(agreement);
    require(karma_negotiate_abi(&request, &agreement) == KARMA_STATUS_OK, "negotiation failed");

    karma_plugin_query_result query;
    memset(&query, 0, sizeof(query));
    query.struct_size = (uint32_t)sizeof(query);
    require(karma_plugin_query(&query) == KARMA_STATUS_OK, "plugin query failed");
    const karma_component_table* table = query.component_table;
    require(table != NULL, "no component table");

    uint64_t element_count = 3;
    karma_component_config config;
    memset(&config, 0, sizeof(config));
    config.struct_size = (uint32_t)sizeof(config);
    config.implementation_id = query.implementation_id;
    config.configuration = &element_count;
    config.configuration_size = sizeof(element_count);
    config.communicator = KARMA_COMMUNICATOR_NONE;
    config.granted_capabilities = agreement.granted_capabilities;

    void* instance = NULL;
    require(table->create(&config, &instance) == KARMA_STATUS_OK, "create failed");

    karma_exact_time start;
    start.numerator = 0;
    start.denominator = 1;
    require(table->initialize(instance, start) == KARMA_STATUS_OK, "initialize failed");

    double values[3] = {10.0, 20.0, 30.0};
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
    field.extents[0] = 3;
    field.byte_strides[0] = (int64_t)sizeof(double);
    field.data = values;
    field.element_count = 3;
    require(table->import_fields(instance, &field, 1) == KARMA_STATUS_OK, "import failed");

    uint64_t token = 0;
    require(table->checkpoint(instance, &token) == KARMA_STATUS_OK, "checkpoint failed");

    karma_exact_interval interval;
    interval.begin.numerator = 0;
    interval.begin.denominator = 1;
    interval.end.numerator = 3;
    interval.end.denominator = 1;
    require(table->advance(instance, interval) == KARMA_STATUS_OK, "advance failed");
    require(table->commit(instance) == KARMA_STATUS_OK, "commit failed");

    double exported[3] = {0, 0, 0};
    field.data = exported;
    require(table->export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export failed");
    require(fabs(exported[0] - 13.0) < 1e-12, "advance did not add three units of time");

    require(table->restore(instance, token) == KARMA_STATUS_OK, "restore failed");
    require(table->export_fields(instance, &field, 1) == KARMA_STATUS_OK, "export after restore failed");
    require(fabs(exported[0] - 10.0) < 1e-12, "restore did not recover the checkpoint");

    /* A deterministic error, named. */
    require(table->restore(instance, 999) == KARMA_STATUS_UNKNOWN_CHECKPOINT, "an unknown token was accepted");
    require(strcmp(karma_status_name(KARMA_STATUS_UNKNOWN_CHECKPOINT), "unknown checkpoint") == 0,
            "status naming is wrong");

    table->destroy(instance);
    if (failures == 0) printf("installed C consumer ok\n");
    return failures == 0 ? 0 : 1;
}
