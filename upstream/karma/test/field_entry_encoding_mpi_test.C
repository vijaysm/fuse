// Distributed reassociation of the canonical FieldEntryKey encoding.
//
// The contract under test: a key encoded on one rank decodes to the same key
// on every other rank, so a field entry can be re-associated after an
// arbitrary change of ownership without consulting a PETSc ordinal, a MOAB
// handle, or the rank that produced it.
//
// Only MPI is initialized; PETSc is not needed to exercise a byte encoding.
#include "karma_field_entry_encoding.h"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <span>
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

constexpr int canonical_count = 37;

/** One deterministic global key set, independent of rank count. */
std::vector<FieldEntryKey> canonical_keys()
{
    std::vector<FieldEntryKey> keys;
    keys.reserve(canonical_count);
    for (int index = 0; index != canonical_count; ++index)
    {
        FieldEntryKey key;
        if (index % 3 == 0)
        {
            key.kind = FieldEntryKey::Kind::MoabGlobalId;
            key.global_id = 1000 + static_cast<std::uint64_t>(index);
            key.component = static_cast<unsigned>(index % 4);
        }
        else
        {
            key.kind = FieldEntryKey::Kind::Dof;
            key.dof[3] = static_cast<std::uint64_t>(index % 5);
            key.dof[5] = static_cast<std::uint64_t>(index % 3);
            key.dof[6] = 10 + static_cast<std::uint64_t>(index);
        }
        keys.push_back(key);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

/** Ownership that varies with rank count and leaves one rank empty when it can.
 *
 * The empty rank is the point: a rank owning nothing must still contribute a
 * well-formed zero-length buffer and must agree on the global set.
 */
std::vector<FieldEntryKey> owned_keys(const std::vector<FieldEntryKey>& all, int rank, int size)
{
    std::vector<FieldEntryKey> owned;
    if (size >= 3 && rank == size - 1) return owned;
    const int owners = size >= 3 ? size - 1 : size;
    for (int index = 0; index != static_cast<int>(all.size()); ++index)
    {
        // A stride-and-offset split so consecutive keys land on different ranks.
        if ((index * 7 + index / owners) % owners == rank) owned.push_back(all[index]);
    }
    return owned;
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
        const auto all = canonical_keys();
        const auto owned = owned_keys(all, rank, size);

        std::vector<std::byte> local(owned.size() * field_entry_key_encoded_size);
        for (std::size_t index = 0; index != owned.size(); ++index)
            require(encode_field_entry_key(
                        owned[index], std::span(local).subspan(index * field_entry_key_encoded_size,
                                                               field_entry_key_encoded_size)),
                    "encoding into a preallocated buffer failed");

        int local_bytes = static_cast<int>(local.size());
        std::vector<int> counts(static_cast<std::size_t>(size), 0);
        MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::vector<int> displacements(static_cast<std::size_t>(size), 0);
        for (std::size_t index = 1; index != counts.size(); ++index)
            displacements[index] = displacements[index - 1] + counts[index - 1];
        const int total = displacements.back() + counts.back();
        require(total == static_cast<int>(all.size() * field_entry_key_encoded_size),
                "ownership split lost or duplicated keys");

        std::vector<std::byte> gathered(static_cast<std::size_t>(total));
        MPI_Allgatherv(local.data(), local_bytes, MPI_BYTE, gathered.data(), counts.data(),
                       displacements.data(), MPI_BYTE, MPI_COMM_WORLD);

        std::vector<FieldEntryKey> decoded;
        decoded.reserve(all.size());
        for (std::size_t offset = 0; offset != gathered.size(); offset += field_entry_key_encoded_size)
        {
            const auto result = decode_field_entry_key(
                std::span(gathered).subspan(offset, field_entry_key_encoded_size));
            require(result.error == FieldEntryDecodeError::None && result.key.has_value(),
                    "a key encoded on a peer rank failed to decode");
            decoded.push_back(*result.key);
        }
        std::sort(decoded.begin(), decoded.end());
        require(decoded == all, "reassociated key set differs from the canonical set");

        // Hashes must agree bit-for-bit across ranks, so they can address a
        // remote owner without a prior exchange of the key itself.
        std::uint64_t local_hash = 0;
        for (const auto& key : all) local_hash ^= field_entry_key_hash(key);
        std::uint64_t minimum = 0, maximum = 0;
        MPI_Allreduce(&local_hash, &minimum, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&local_hash, &maximum, 1, MPI_UINT64_T, MPI_MAX, MPI_COMM_WORLD);
        require(minimum == maximum, "stable-key hash disagrees across ranks");

        if (rank == 0)
            std::cout << "ranks=" << size << " keys=" << all.size() << " hash=" << minimum << '\n';
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
