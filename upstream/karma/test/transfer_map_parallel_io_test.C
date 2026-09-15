#include "karma_transfer_map_io.h"

#include <mpi.h>

#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int ranks = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 0;
    try
    {
        const std::string filename = argc > 1 ? argv[1] : "transfer_map_parallel.bin";
        if (rank == 0)
        {
            KARMA::KARMATransferMap map(KARMA::TransferMapRepresentation::SparseInterpolation, 1);
            for (std::uint64_t id = 1; id <= 10; ++id)
            {
                KARMA::FieldEntryKey target;
                target.global_id = id;
                target.kind = KARMA::FieldEntryKey::Kind::MoabGlobalId;
                KARMA::FieldEntryKey source = target;
                map.add_row({target, {source}, {1.0}, KARMA::CoverageClass::Full});
            }
            KARMA::save_transfer_map(filename, map);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        const auto map = KARMA::load_transfer_map(filename);
        std::vector<KARMA::FieldEntryKey> owned;
        for (std::uint64_t id = 1; id <= 10; ++id)
            if (static_cast<int>((id - 1) % static_cast<std::uint64_t>(ranks)) == rank)
            {
                KARMA::FieldEntryKey target;
                target.global_id = id;
                target.kind = KARMA::FieldEntryKey::Kind::MoabGlobalId;
                owned.push_back(target);
            }
        const auto selected = map.select_targets(owned);
        selected.validate_distribution(owned, owned);
        int local = selected.rows().size() == owned.size() ? 1 : 0;
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        result = global ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        result = 1;
    }
    MPI_Finalize();
    return result;
}
