// Distributed reconciliation: one agreed verdict from disagreeing ranks.
//
// The property under test is the one that prevents a hang: every rank reaches
// the same conclusion, including the ranks that succeeded locally, so no rank
// proceeds into a later collective while a peer has bailed out.
//
// Only MPI is initialized; the primitive needs no PETSc.
#include "karma_collective_outcome.h"

#include <mpi.h>

#include <cstdint>
#include <iostream>
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

/** Every rank must agree bit-for-bit on the verdict, so each rank broadcasts
 * what it concluded and compares. */
void require_agreement(const ReconciledOutcome& outcome, int size, const char* message)
{
    std::vector<std::int64_t> encoded(static_cast<std::size_t>(size) * 3, 0);
    const std::int64_t local[3]{static_cast<std::int64_t>(outcome.reason), outcome.failing_rank,
                                outcome.failing_rank_count};
    require(MPI_Allgather(local, 3, MPI_INT64_T, encoded.data(), 3, MPI_INT64_T, MPI_COMM_WORLD) == MPI_SUCCESS,
            "allgather failed");
    for (int rank = 1; rank < size; ++rank)
        for (int field = 0; field != 3; ++field)
            require(encoded[static_cast<std::size_t>(rank) * 3 + field] == encoded[static_cast<std::size_t>(field)],
                    message);
}
}

int main(int argc, char** argv)
{
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) return 1;
    int rank = 0, size = 1, failure = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    try
    {
        // All ranks succeed.
        const auto clean = reconcile_outcome(MPI_COMM_WORLD, {});
        require(clean.ok() && clean.failing_rank == -1 && clean.failing_rank_count == 0,
                "an all-success reconciliation reported a failure");
        require_agreement(clean, size, "ranks disagreed on an all-success verdict");

        // One rank fails. The highest rank is chosen as the failing one so the
        // verdict cannot accidentally match "rank 0 decides".
        const int single_failing_rank = size - 1;
        const LocalOutcome one{rank == single_failing_rank ? OutcomeReason::NonfiniteValue : OutcomeReason::Success, 0};
        const auto single = reconcile_outcome(MPI_COMM_WORLD, one);
        require(!single.ok() && single.reason == OutcomeReason::NonfiniteValue &&
                single.failing_rank == single_failing_rank && single.failing_rank_count == 1,
                "a single-rank failure was not reconciled exactly");
        require_agreement(single, size, "ranks disagreed on a single-rank failure");

        // Every rank fails with a different reason. The lowest rank decides,
        // not the lowest reason code: rank 0 reports the highest code here.
        const auto descending = static_cast<OutcomeReason>(
            static_cast<std::uint32_t>(OutcomeReason::RollbackFailed) - static_cast<std::uint32_t>(rank % 8));
        const auto disagreeing = reconcile_outcome(MPI_COMM_WORLD, {descending, 0});
        require(!disagreeing.ok() && disagreeing.failing_rank == 0 &&
                disagreeing.reason == OutcomeReason::RollbackFailed && disagreeing.failing_rank_count == size,
                "the lowest failing rank did not decide a disagreeing reconciliation");
        require_agreement(disagreeing, size, "ranks disagreed on a disagreeing reconciliation");

        // All ranks fail with the same reason.
        const auto uniform = reconcile_outcome(MPI_COMM_WORLD, {OutcomeReason::ValidationFailed, 0});
        require(!uniform.ok() && uniform.reason == OutcomeReason::ValidationFailed && uniform.failing_rank == 0 &&
                uniform.failing_rank_count == size, "an all-rank failure was not counted");
        require_agreement(uniform, size, "ranks disagreed on an all-rank failure");

        // A rank cannot misreport its identity: the communicator's rank wins.
        const auto spoofed = reconcile_outcome(MPI_COMM_WORLD, {OutcomeReason::InvalidArgument, 12345});
        require(spoofed.failing_rank == 0 && spoofed.failing_rank_count == size,
                "a caller-supplied rank overrode the communicator rank");

        if (rank == 0) std::cout << "ranks=" << size << " reconciliations=5\n";
    }
    catch (const std::exception& error)
    {
        std::cerr << "rank " << rank << ": " << error.what() << '\n';
        failure = 1;
    }
    int global_failure = 0;
    MPI_Allreduce(&failure, &global_failure, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return global_failure;
}
