// Deterministic selection of one failure from many, and its wire encoding.
//
// The selection rule and the key packing are pure functions, so they are
// tested without MPI here; `collective_outcome_mpi_test.C` exercises the
// reduction itself. That split is deliberate: the selection rule is where the
// determinism lives, and it should not require four ranks to check.
#include "karma_collective_outcome.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

LocalOutcome failure(OutcomeReason reason, int rank) { return {reason, rank}; }
LocalOutcome success(int rank) { return {OutcomeReason::Success, rank}; }

void key_encoding()
{
    require(encode_outcome_key(success(0)) == outcome_success_key, "success did not encode to the success key");
    require(encode_outcome_key(success(7)) == outcome_success_key,
            "a successful rank's key depends on its rank");
    // Success must sort above EVERY representable failure, not merely above
    // the named ones, because the reduction is a minimum over all ranks: a
    // success key that sorted below a failure would mask that failure.
    require(encode_outcome_key(failure(OutcomeReason::Internal, 2147483647)) < outcome_success_key,
            "a failure on the highest rank sorts above success");
    require(encode_outcome_key({static_cast<OutcomeReason>(0xffffffffU), 2147483647}) < outcome_success_key,
            "the maximal representable failure key sorts above success");

    // A rank below zero cannot come from MPI_Comm_rank, and must not be
    // sign-extended into the rank field, where it would sort as the highest
    // possible rank and hide a real lowest-rank failure.
    const auto negative = encode_outcome_key(failure(OutcomeReason::NonfiniteValue, -1));
    require(negative == encode_outcome_key(failure(OutcomeReason::NonfiniteValue, 0)),
            "a negative rank was not clamped");
    require(negative < encode_outcome_key(failure(OutcomeReason::NonfiniteValue, 1)),
            "a negative rank does not sort below rank one");
    require(negative < outcome_success_key, "a negative-rank failure sorts above success");

    // Lower rank must sort first, then lower reason code.
    require(encode_outcome_key(failure(OutcomeReason::Internal, 1)) <
            encode_outcome_key(failure(OutcomeReason::InvalidArgument, 2)),
            "rank is not the primary sort key");
    require(encode_outcome_key(failure(OutcomeReason::InvalidArgument, 3)) <
            encode_outcome_key(failure(OutcomeReason::NonfiniteValue, 3)),
            "reason code is not the secondary sort key");

    for (const auto& outcome : {failure(OutcomeReason::NonfiniteValue, 0), failure(OutcomeReason::RollbackFailed, 5),
                               failure(OutcomeReason::Internal, 2147483647)})
    {
        const auto decoded = decode_outcome_key(encode_outcome_key(outcome));
        require(decoded.reason == outcome.reason && decoded.rank == outcome.rank, "key did not round-trip");
    }
    const auto decoded_success = decode_outcome_key(outcome_success_key);
    require(decoded_success.ok(), "the success key did not decode as success");
}

void deterministic_selection()
{
    require(select_outcome(std::vector<LocalOutcome>{success(0), success(1), success(2)}).ok(),
            "all-success was reported as a failure");
    require(select_outcome({}).ok(), "an empty outcome set was reported as a failure");

    // One failure among successes.
    const std::vector<LocalOutcome> one_failure{success(0), failure(OutcomeReason::NonfiniteValue, 1), success(2)};
    const auto single = select_outcome(one_failure);
    require(!single.ok() && single.reason == OutcomeReason::NonfiniteValue && single.failing_rank == 1 &&
            single.failing_rank_count == 1, "a single failure was not selected exactly");

    // Disagreeing reasons: the lowest rank decides, not the lowest reason.
    const std::vector<LocalOutcome> disagreeing{failure(OutcomeReason::RollbackFailed, 1),
                                                failure(OutcomeReason::InvalidArgument, 3),
                                                failure(OutcomeReason::NonfiniteValue, 2)};
    const auto chosen = select_outcome(disagreeing);
    require(!chosen.ok() && chosen.reason == OutcomeReason::RollbackFailed && chosen.failing_rank == 1 &&
            chosen.failing_rank_count == 3, "the lowest failing rank did not decide");

    // Selection must not depend on the order the records arrive in.
    std::vector<LocalOutcome> permuted{disagreeing[2], disagreeing[0], disagreeing[1]};
    const auto other = select_outcome(permuted);
    require(other.reason == chosen.reason && other.failing_rank == chosen.failing_rank &&
            other.failing_rank_count == chosen.failing_rank_count,
            "selection depends on the order of the records");

    // Two reasons from one rank: the lower reason code wins.
    const auto same_rank = select_outcome(std::vector<LocalOutcome>{
        failure(OutcomeReason::RollbackFailed, 4), failure(OutcomeReason::InvalidArgument, 4)});
    require(same_rank.reason == OutcomeReason::InvalidArgument && same_rank.failing_rank == 4,
            "two reasons from one rank did not resolve to the lower code");

    // All ranks failing with the same reason.
    const auto all_failed = select_outcome(std::vector<LocalOutcome>{
        failure(OutcomeReason::ValidationFailed, 0), failure(OutcomeReason::ValidationFailed, 1),
        failure(OutcomeReason::ValidationFailed, 2)});
    require(all_failed.failing_rank == 0 && all_failed.failing_rank_count == 3,
            "an all-rank failure was not counted");
}

// Two ranks failing for genuinely different reasons must agree on a verdict
// that names one of them. Before the vocabulary carried the ABI's
// distinctions, both of these arrived as InvalidArgument and the agreed
// reason described neither failure.
void distinct_reasons_survive_reconciliation()
{
    const std::vector<LocalOutcome> outcomes = {
        failure(OutcomeReason::MalformedFieldTable, 1),
        failure(OutcomeReason::UnknownCheckpoint, 2),
    };
    const auto selected = select_outcome(outcomes);
    require(!selected.ok(), "two failures reconciled to success");
    require(selected.reason == OutcomeReason::MalformedFieldTable ||
            selected.reason == OutcomeReason::UnknownCheckpoint,
            "the agreed reason names neither rank's actual failure");
    // Lowest rank is the primary key, so it is the rank-1 reason specifically.
    require(selected.failing_rank == 1, "the lowest failing rank did not win");
    require(selected.failing_rank_count == 2, "both failures were not counted");
    require(selected.reason == OutcomeReason::MalformedFieldTable,
            "the winning rank's reason was replaced by another rank's");
}

void reason_names()
{
    // Every reason must be nameable, because the primitive reduces a numeric
    // code and each rank maps it to text locally rather than broadcasting a
    // string from the failing rank.
    for (std::uint32_t code = 0; code <= static_cast<std::uint32_t>(OutcomeReason::HighestNamed); ++code)
        require(!outcome_reason_name(static_cast<OutcomeReason>(code)).empty(), "a reason code has no name");
    require(outcome_reason_name(static_cast<OutcomeReason>(9999)).empty(), "an unknown reason acquired a name");
    require(outcome_reason_name(OutcomeReason::Success) == "success", "the success name changed");
}
}

int main()
{
    try
    {
        key_encoding();
        deterministic_selection();
        distinct_reasons_survive_reconciliation();
    reason_names();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
