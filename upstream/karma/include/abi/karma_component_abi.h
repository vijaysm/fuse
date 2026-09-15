/* KARMA stable C ABI, version 1: the component call boundary.
 *
 * Direction: the host calls the plugin. Every function returns a status; none
 * allocates on the caller's behalf; none takes ownership of caller memory.
 *
 * The table below is the *complete* shared call prefix. It is defined in full
 * before any freeze deliberately, because a prefix that grows after languages
 * have bound to it is not a prefix. A minor release may append to the tail and
 * a reader detects the tail through `struct_size`.
 */
#ifndef KARMA_COMPONENT_ABI_H
#define KARMA_COMPONENT_ABI_H

#include "karma_abi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KARMA_ELEMENT_F64 0
#define KARMA_ELEMENT_I64 1
#define KARMA_ELEMENT_I32 2

#define KARMA_OWNERSHIP_OWNED 0
#define KARMA_OWNERSHIP_GHOSTED 1
#define KARMA_OWNERSHIP_REPLICATED 2

#define KARMA_ACCESS_READ_ONLY 0
#define KARMA_ACCESS_READ_WRITE 1
#define KARMA_ACCESS_WRITE_ONLY 2

/* Update mode for an imported table, matching the transactional field API:
 * a replacement must cover every value, a sparse replacement need not, and an
 * accumulation adds to what is published. */
#define KARMA_UPDATE_REPLACE 0
#define KARMA_UPDATE_SPARSE_REPLACE 1
#define KARMA_UPDATE_ACCUMULATE 2

/* A caller-owned view of field values. The plugin reads or writes through it
 * during the call and must not retain the pointer afterwards: lifetime is the
 * call, which is the only rule a language adapter can enforce. */
typedef struct karma_field_table
{
    uint32_t struct_size;
    /* NUL-terminated, caller-owned, valid for the call. */
    const char* field_id;
    int32_t element_type;
    int32_t ownership;
    int32_t access;
    int32_t update;
    uint32_t rank;
    uint32_t components;
    int64_t extents[KARMA_ABI_FIELD_MAX_RANK];
    /* Byte strides, so a caller may present a non-contiguous array without
     * copying it into a packed buffer first. */
    int64_t byte_strides[KARMA_ABI_FIELD_MAX_RANK];
    void* data;
    uint64_t element_count;
    /* Canonical field-entry key bytes, one fixed-width record per entry, or
     * NULL for a positional table. */
    const void* key_bytes;
    uint64_t key_bytes_size;
    uint32_t key_record_size;
} karma_field_table;

/* Validates shape, strides, lifetime-relevant pointers, and key sizing.
 * Exposed so a language adapter can check a table it just built rather than
 * discovering the problem inside a component. */
KARMA_ABI_API karma_status karma_validate_field_table(const karma_field_table* table);

typedef struct karma_component_config
{
    uint32_t struct_size;
    const char* implementation_id;
    /* Opaque configuration bytes; the schema is the implementation's own. */
    const void* configuration;
    uint64_t configuration_size;
    karma_communicator communicator;
    uint64_t granted_capabilities;
} karma_component_config;

typedef struct karma_component_metadata
{
    uint32_t struct_size;
    karma_version implementation_version;
    uint64_t provided_capabilities;
    uint32_t field_count;
    int32_t supports_fractional_steps;
    int32_t supports_checkpoint_restore;
} karma_component_metadata;

/* Functions the plugin may call back into the host. Size-prefixed and passed
 * by pointer, so the host may extend it in a minor release. */
typedef struct karma_host_table
{
    uint32_t struct_size;
    void* host_context;
    karma_status (*report)(void* host_context, const karma_diagnostic* diagnostic);
    karma_status (*communicator_size)(void* host_context, karma_communicator communicator, int32_t* size);
    karma_status (*communicator_rank)(void* host_context, karma_communicator communicator, int32_t* rank);
} karma_host_table;

/* Functions the host calls on the plugin. */
typedef struct karma_component_table
{
    uint32_t struct_size;

    karma_status (*create)(const karma_component_config* config, void** instance);
    karma_status (*bind_host)(void* instance, const karma_host_table* host);
    karma_status (*initialize)(void* instance, karma_exact_time start);
    karma_status (*metadata)(void* instance, karma_component_metadata* metadata);

    karma_status (*import_fields)(void* instance, const karma_field_table* tables, uint32_t count);
    karma_status (*export_fields)(void* instance, const karma_field_table* tables, uint32_t count);

    karma_status (*advance)(void* instance, karma_exact_interval interval);
    karma_status (*commit)(void* instance);

    karma_status (*checkpoint)(void* instance, uint64_t* token);
    karma_status (*restore)(void* instance, uint64_t token);
    karma_status (*release_checkpoint)(void* instance, uint64_t token);

    karma_status (*diagnostics)(void* instance, karma_diagnostic* diagnostic);
    karma_status (*poll_cancellation)(void* instance, int32_t* cancelled);

    void (*destroy)(void* instance);

    /* Optional tail, added as a minor extension: a seconds-valued shadow of
     * `advance` for a participant with no rational arithmetic. Exact time
     * remains the primary form; this converts and may refuse a value it cannot
     * represent exactly. A reader detects its presence through `struct_size`. */
    karma_status (*advance_seconds)(void* instance, double begin, double end);
} karma_component_table;

typedef struct karma_plugin_query_result
{
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint64_t provided_capabilities;
    const char* implementation_id;
    karma_version implementation_version;
    const karma_component_table* component_table;
} karma_plugin_query_result;

/* The bootstrap symbol. A plugin exports exactly this name from its own
 * object and the loader resolves it through that object's handle, so several
 * plugins may each export it. The library also exports one, which is the
 * built-in reference component the conformance tests exercise. */
KARMA_ABI_API karma_status karma_plugin_query(karma_plugin_query_result* result);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KARMA_COMPONENT_ABI_H */
