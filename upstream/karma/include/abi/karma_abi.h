/* KARMA stable C ABI, version 1 (candidate).
 *
 * This header is C11 and deliberately self-contained: it includes only
 * <stdint.h> and <stddef.h>, and sibling ABI headers as "karma_abi.h" from
 * the same directory. A consumer needs one include path - the install prefix -
 * and nothing from KARMA's C++ headers, which include their siblings flat and
 * would otherwise drag in every subdirectory.
 *
 * Excluded from this boundary, permanently: C++ types of any kind, exceptions,
 * templates, PETSc and MOAB objects, and the platform's native MPI
 * communicator type. A communicator crosses as a framework-owned integer
 * handle, because MPI_Comm is a pointer on some implementations and an integer
 * on others, and a Fortran or Python participant cannot portably spell either.
 *
 * Compatibility policy: ABI major 1 stays compatible across minor releases of
 * the library. Every record begins with `struct_size`, and a reader accepts a
 * larger record than it knows by ignoring the tail. The C++ API is explicitly
 * NOT stable and is expected to keep changing.
 */
#ifndef KARMA_ABI_H
#define KARMA_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Only the symbols carrying this belong to the ABI. The library is built with
 * hidden visibility and an explicit export list, and a test asserts that the
 * exported set equals the documented set - an accidentally exported C++ symbol
 * is a compatibility promise nobody meant to make. */
#if defined(_WIN32)
#define KARMA_ABI_API
#else
#define KARMA_ABI_API __attribute__((visibility("default")))
#endif

#define KARMA_ABI_MAJOR 1u
#define KARMA_ABI_MINOR 0u
#define KARMA_ABI_PATCH 0u

/* Bounded so a diagnostic can cross the boundary without an allocator. */
#define KARMA_ABI_DIAGNOSTIC_CAPACITY 512
#define KARMA_ABI_FIELD_MAX_RANK 4
/* Any record larger than this is treated as malformed rather than as a very
 * new tail: a wild size is the shape a corrupted or hostile record takes. */
#define KARMA_ABI_MAX_STRUCT_SIZE 65536u

typedef int32_t karma_status;

#define KARMA_STATUS_OK 0
#define KARMA_STATUS_INVALID_ARGUMENT 1
#define KARMA_STATUS_TRUNCATED_PREFIX 2
#define KARMA_STATUS_MALFORMED_SIZE 3
#define KARMA_STATUS_INCOMPATIBLE_MAJOR 4
#define KARMA_STATUS_UNSUPPORTED_CAPABILITY 5
#define KARMA_STATUS_TABLE_TOO_SMALL 6
#define KARMA_STATUS_UNKNOWN_FIELD 7
#define KARMA_STATUS_MALFORMED_FIELD_TABLE 8
#define KARMA_STATUS_NONFINITE_VALUE 9
#define KARMA_STATUS_INCOMPLETE_UPDATE 10
#define KARMA_STATUS_STORAGE_FAILURE 11
#define KARMA_STATUS_UNKNOWN_CHECKPOINT 12
#define KARMA_STATUS_NOT_INITIALIZED 13
#define KARMA_STATUS_CANCELLED 14
#define KARMA_STATUS_INTERNAL_ERROR 15

/* Exact rational time. Never a double: a coupling schedule that rounds its
 * rendezvous points stops being a rendezvous. `denominator` is always > 0 and
 * the pair is always reduced. */
typedef struct karma_exact_time
{
    int64_t numerator;
    int64_t denominator;
} karma_exact_time;

typedef struct karma_exact_interval
{
    karma_exact_time begin;
    karma_exact_time end;
} karma_exact_interval;

typedef struct karma_version
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;
} karma_version;

/* A framework-owned communicator token. Zero means "no communicator". The
 * framework maps a token to whatever MPI handle the platform uses; the plugin
 * never sees, stores, or frees the underlying handle. */
typedef int64_t karma_communicator;
#define KARMA_COMMUNICATOR_NONE ((karma_communicator)0)

/* Capability bits. A required bit the library does not know is a refusal, not
 * a warning: silently ignoring an unknown requirement is how a participant
 * ends up believing it got a guarantee it did not get. */
#define KARMA_CAPABILITY_HOST_ARRAY_FIELDS (UINT64_C(1) << 0)
#define KARMA_CAPABILITY_KEYED_FIELDS (UINT64_C(1) << 1)
#define KARMA_CAPABILITY_CHECKPOINT_RESTORE (UINT64_C(1) << 2)
#define KARMA_CAPABILITY_FRACTIONAL_STEPS (UINT64_C(1) << 3)
#define KARMA_CAPABILITY_CANCELLATION (UINT64_C(1) << 4)
#define KARMA_CAPABILITY_COMMUNICATOR (UINT64_C(1) << 5)

typedef struct karma_diagnostic
{
    uint32_t struct_size;
    karma_status status;
    /* Nonzero when the message did not fit. The message is always
     * NUL-terminated regardless. */
    int32_t truncated;
    char message[KARMA_ABI_DIAGNOSTIC_CAPACITY];
} karma_diagnostic;

typedef struct karma_abi_request
{
    uint32_t struct_size;
    uint32_t abi_major;
    uint32_t abi_minor;
    uint64_t required_capabilities;
    uint64_t optional_capabilities;
    /* Bytes of host table the caller is prepared to provide. Too small is
     * answered with the size needed rather than a bare refusal. */
    uint32_t host_table_size;
} karma_abi_request;

typedef struct karma_abi_agreement
{
    uint32_t struct_size;
    uint32_t abi_major;
    /* min(requested, library): a newer caller talks down, an older caller is
     * not offered a tail it cannot read. */
    uint32_t abi_minor;
    uint64_t granted_capabilities;
    /* Nonzero if and only if the request was refused for capabilities. */
    uint64_t unsupported_required;
    uint32_t required_host_table_size;
    uint32_t plugin_table_size;
} karma_abi_agreement;

/* Library identity and what it can agree to. */
KARMA_ABI_API karma_version karma_abi_version(void);
KARMA_ABI_API uint64_t karma_abi_supported_capabilities(void);

/* Negotiate. Total and nonthrowing: every rejection is a status, and no
 * exception escapes even when the library is built with exceptions enabled. */
KARMA_ABI_API karma_status karma_negotiate_abi(const karma_abi_request* request, karma_abi_agreement* agreement);

/* Names a status for logging. Never NULL; empty for an unknown value, so an
 * unknown status reads as unknown rather than as something plausible. */
KARMA_ABI_API const char* karma_status_name(karma_status status);

/* Writes `text` into `diagnostic`, truncating at the capacity and setting the
 * truncation flag. Exposed because a plugin must be able to produce a
 * diagnostic without an allocator. */
KARMA_ABI_API karma_status karma_diagnostic_set(karma_diagnostic* diagnostic, karma_status status, const char* text);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* KARMA_ABI_H */
