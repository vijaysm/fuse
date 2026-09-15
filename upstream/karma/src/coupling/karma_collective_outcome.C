#include "karma_collective_outcome.h"

#include <algorithm>
#include <limits>

namespace KARMA
{
namespace
{
constexpr std::uint32_t highest_named_reason = static_cast<std::uint32_t>(OutcomeReason::HighestNamed);

bool named(OutcomeReason reason) noexcept
{
    return static_cast<std::uint32_t>(reason) <= highest_named_reason;
}
}

std::string_view outcome_reason_name(OutcomeReason reason) noexcept
{
    switch (reason)
    {
        case OutcomeReason::Success: return "success";
        case OutcomeReason::InvalidArgument: return "invalid argument";
        case OutcomeReason::NonfiniteValue: return "nonfinite value";
        case OutcomeReason::DuplicateEntry: return "duplicate entry";
        case OutcomeReason::UnknownEntry: return "unknown entry";
        case OutcomeReason::IncompleteUpdate: return "incomplete update";
        case OutcomeReason::StorageFailure: return "storage failure";
        case OutcomeReason::ValidationFailed: return "validation failed";
        case OutcomeReason::RollbackFailed: return "rollback failed";
        case OutcomeReason::Internal: return "internal reconciliation failure";
        case OutcomeReason::TruncatedPrefix: return "truncated prefix";
        case OutcomeReason::MalformedSize: return "malformed size";
        case OutcomeReason::IncompatibleMajor: return "incompatible major version";
        case OutcomeReason::UnsupportedCapability: return "unsupported capability";
        case OutcomeReason::TableTooSmall: return "table too small";
        case OutcomeReason::UnknownField: return "unknown field";
        case OutcomeReason::MalformedFieldTable: return "malformed field table";
        case OutcomeReason::UnknownCheckpoint: return "unknown checkpoint";
        case OutcomeReason::NotInitialized: return "not initialized";
        case OutcomeReason::Cancelled: return "cancelled";
    }
    return {};
}

std::uint64_t encode_outcome_key(LocalOutcome outcome) noexcept
{
    if (outcome.ok()) return outcome_success_key;
    // A negative rank cannot come from MPI_Comm_rank; clamp rather than
    // sign-extend it into the rank field, where it would sort as the highest
    // possible rank and hide a real lowest-rank failure.
    const auto rank = static_cast<std::uint64_t>(outcome.rank < 0 ? 0 : outcome.rank);
    const auto reason = static_cast<std::uint64_t>(outcome.reason) & 0xffffffffULL;
    return (rank << 32) | reason;
}

LocalOutcome decode_outcome_key(std::uint64_t key) noexcept
{
    if (key == outcome_success_key) return {};
    LocalOutcome outcome;
    outcome.rank = static_cast<int>(key >> 32);
    outcome.reason = static_cast<OutcomeReason>(static_cast<std::uint32_t>(key & 0xffffffffULL));
    return outcome;
}

ReconciledOutcome select_outcome(std::span<const LocalOutcome> outcomes) noexcept
{
    std::uint64_t lowest = outcome_success_key;
    int failures = 0;
    for (const auto& outcome : outcomes)
    {
        if (outcome.ok()) continue;
        ++failures;
        lowest = std::min(lowest, encode_outcome_key(outcome));
    }
    if (failures == 0) return {};
    const auto selected = decode_outcome_key(lowest);
    return {selected.reason, selected.rank, failures};
}

ReconciledOutcome reconcile_outcome(MPI_Comm communicator, LocalOutcome local) noexcept
{
    int rank = 0;
    if (communicator == MPI_COMM_NULL || MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS)
        return {OutcomeReason::Internal, local.rank, 1};
    local.rank = rank;

    // Two small reductions rather than one custom operator: the minimum key
    // selects the verdict, and the sum reports how widespread the failure is.
    // The count matters diagnostically - one rank failing and every rank
    // failing are different problems with the same verdict.
    const std::uint64_t local_key = encode_outcome_key(local);
    std::uint64_t lowest_key = outcome_success_key;
    if (MPI_Allreduce(&local_key, &lowest_key, 1, MPI_UINT64_T, MPI_MIN, communicator) != MPI_SUCCESS)
        return {OutcomeReason::Internal, rank, 1};

    const int local_failure = local.ok() ? 0 : 1;
    int failures = 0;
    if (MPI_Allreduce(&local_failure, &failures, 1, MPI_INT, MPI_SUM, communicator) != MPI_SUCCESS)
        return {OutcomeReason::Internal, rank, 1};

    if (lowest_key == outcome_success_key) return {};
    const auto selected = decode_outcome_key(lowest_key);
    return {named(selected.reason) ? selected.reason : OutcomeReason::Internal, selected.rank, failures};
}

std::string reconcile_failure_message(MPI_Comm communicator, bool failed, const std::string& local_message)
{
    const auto outcome = reconcile_outcome(communicator, {failed ? OutcomeReason::ValidationFailed : OutcomeReason::Success, 0});
    if (outcome.ok()) return {};

    // MPI failed: `failing_rank` is this rank, not an agreed one, so there is
    // no safe broadcast root and no text to exchange.
    if (outcome.reason == OutcomeReason::Internal)
        return "collective reconciliation failed: " + std::string(outcome_reason_name(outcome.reason));

    int rank = 0;
    if (MPI_Comm_rank(communicator, &rank) != MPI_SUCCESS) rank = outcome.failing_rank;

    int length = rank == outcome.failing_rank ? static_cast<int>(local_message.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, outcome.failing_rank, communicator);
    std::string message(static_cast<std::size_t>(length), '\0');
    if (rank == outcome.failing_rank) message = local_message;
    if (length != 0) MPI_Bcast(message.data(), length, MPI_CHAR, outcome.failing_rank, communicator);

    std::string result = "rank " + std::to_string(outcome.failing_rank);
    // One rank failing and every rank failing are different problems with the
    // same verdict; the count is the only thing that distinguishes them, and
    // none of the five copies reported it.
    if (outcome.failing_rank_count > 1)
        result += " (and " + std::to_string(outcome.failing_rank_count - 1) + " other rank" +
                  (outcome.failing_rank_count > 2 ? "s" : "") + ")";
    return result + ": " + message;
}
} // namespace KARMA
