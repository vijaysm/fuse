#ifndef KARMA_COUPLING_COLLECTIVE_OUTCOME_H
#define KARMA_COUPLING_COLLECTIVE_OUTCOME_H

#include <mpi.h>

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

/** The one collective failure-reconciliation primitive for coupling code.
 *
 * The hazard this exists to remove: one rank discovers a local problem and
 * throws while its peers proceed into the next MPI or PETSc collective, which
 * then hangs or aborts with a diagnostic that names the wrong cause. Every
 * coupling path that can fail locally must instead record a local outcome,
 * reconcile it across the transaction communicator, and act on one agreed
 * verdict.
 *
 * Selection is deterministic: the lowest failing rank decides, and within one
 * rank the lowest reason code decides. Determinism matters more than choosing
 * the "most interesting" failure, because a nondeterministic choice produces
 * error messages that change between runs of the same job.
 *
 * The reduction carries a numeric reason code, never a string. Each rank maps
 * the code to text locally through `outcome_reason_name()`, which avoids a
 * second collective to broadcast a message from the failing rank.
 */
namespace KARMA
{
enum class OutcomeReason : std::uint32_t
{
    Success = 0,
    InvalidArgument = 1,
    NonfiniteValue = 2,
    DuplicateEntry = 3,
    UnknownEntry = 4,
    IncompleteUpdate = 5,
    StorageFailure = 6,
    ValidationFailed = 7,
    RollbackFailed = 8,
    /** The reconciliation itself failed; no verdict could be agreed. */
    Internal = 9,

    // Everything below carries a distinction the C ABI makes. They exist so
    // that crossing the ABI boundary does not flatten a reason: a rank that
    // failed because a checkpoint token was unknown must not reconcile to
    // "invalid argument", or the agreed verdict describes a failure that did
    // not happen. Before these existed, four statuses collapsed onto
    // InvalidArgument and eight more onto Internal.
    TruncatedPrefix = 10,
    MalformedSize = 11,
    IncompatibleMajor = 12,
    UnsupportedCapability = 13,
    TableTooSmall = 14,
    UnknownField = 15,
    MalformedFieldTable = 16,
    UnknownCheckpoint = 17,
    NotInitialized = 18,
    Cancelled = 19,

    /** Not a reason. The highest named code, so coverage checks and the
     * name table cannot silently fall out of step with this enum. */
    HighestNamed = Cancelled
};

/** Empty for a value outside the enumeration, so an unknown code is reported
 * as unknown rather than defaulted to a plausible name. */
/** Reason codes are compared only to make the choice between two failing
 * ranks deterministic; a lower code is NOT a more severe failure. The rank is
 * the primary key and the reason breaks ties within one rank.
 */
[[nodiscard]] std::string_view outcome_reason_name(OutcomeReason reason) noexcept;

struct LocalOutcome
{
    OutcomeReason reason = OutcomeReason::Success;
    int rank = 0;
    [[nodiscard]] bool ok() const noexcept { return reason == OutcomeReason::Success; }
};

struct ReconciledOutcome
{
    OutcomeReason reason = OutcomeReason::Success;
    /** -1 when every rank succeeded. */
    int failing_rank = -1;
    int failing_rank_count = 0;
    [[nodiscard]] bool ok() const noexcept { return reason == OutcomeReason::Success; }
};

/** Sorts above every failure key, so a minimum reduction selects a failure
 * whenever any rank reports one. */
inline constexpr std::uint64_t outcome_success_key = ~std::uint64_t{0};

/** Packs rank into the high 32 bits and reason into the low 32, so an
 * unsigned minimum reduction implements "lowest rank, then lowest reason"
 * with one standard MPI operation and no custom operator. */
[[nodiscard]] std::uint64_t encode_outcome_key(LocalOutcome outcome) noexcept;
[[nodiscard]] LocalOutcome decode_outcome_key(std::uint64_t key) noexcept;

/** Pure selection over gathered outcomes, independent of their order.
 *
 * This is the whole determinism rule, factored out so it is testable without a
 * communicator and reusable where outcomes are already gathered.
 */
[[nodiscard]] ReconciledOutcome select_outcome(std::span<const LocalOutcome> outcomes) noexcept;

/** Reconcile one local outcome across `communicator`.
 *
 * Every rank of the communicator must call this, and callers must place it
 * where all ranks reach it, including ranks that failed. The local `rank` field
 * is ignored and replaced by this communicator's rank, so a caller cannot
 * misreport its own identity. Nonthrowing: an MPI failure is reported as a
 * failing outcome rather than an exception, because this primitive is called
 * precisely where an exception must not escape.
 */
[[nodiscard]] ReconciledOutcome reconcile_outcome(MPI_Comm communicator, LocalOutcome local) noexcept;

/** Reconcile a rank-local failure and return the agreed human-readable
 * message, or an empty string when every rank succeeded.
 *
 * This exists because five call sites each carried their own copy of the
 * reduction rule `reconcile_outcome()` implements - a max-reduce for "did
 * anyone fail", a min-reduce for "which rank", then a broadcast - which is one
 * rule maintained five times, and the transaction layer's invariant says there
 * is exactly one. The copies were not quite identical either, which is the
 * usual way duplicated collective code goes wrong.
 *
 * The message is what the primitive alone cannot give: a reason code is
 * reduced as a number and named locally, while these sites carry detail no
 * enumerator holds - which field, which interval, which instruction. So the
 * selected rank's text is broadcast, layered ON TOP of the shared selection
 * instead of reimplementing it.
 *
 * It returns rather than throws, so each caller keeps its own exception type
 * and wording; nothing here decides how a failure is reported. Every rank of
 * `communicator` must call it, including ranks that did not fail.
 *
 * When MPI itself fails the returned message says so and carries no broadcast
 * text: the primitive reports that case with a local rather than an agreed
 * rank, so using it as a broadcast root would have the ranks disagree about
 * the root. Callers still get a nonempty message and still fail.
 */
[[nodiscard]] std::string reconcile_failure_message(MPI_Comm communicator, bool failed,
                                                    const std::string& local_message);
} // namespace KARMA

#endif
