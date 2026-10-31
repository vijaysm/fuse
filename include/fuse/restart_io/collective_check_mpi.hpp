#pragma once

/// The cross-rank half of batched global-id verification
/// (docs/dev/restart.md): "FUSE verifies uniqueness and coverage of the
/// declared global size for ALL arrays of a restart point in ONE
/// collective (one sort and reduction over every array's ids together,
/// never one collective per array)". `Hdf5GroupWriter::finish_point()`
/// (src/restart_io/hdf5_restart_io.cpp) already does the SINGLE-RANK half
/// of this -- a local sort-and-check over this rank's own (name, id)
/// pairs, with no communication (tier 0, no MPI). This header is the cross-rank extension: it reaches the
/// SAME verdict a single combined rank would reach, by gathering every
/// rank's own local view into one shared global view FIRST (a single
/// `MPI_Allgatherv` over the combined buffer, not N separate collectives
/// per field, per the "never one collective per array" rule)
/// and then running the IDENTICAL check against that shared view on every
/// rank -- so every rank's verdict is the identical answer to the
/// identical question, never a separate agreement reduction on top of it
/// (the "collective path must never return on a locally computed
/// condition" rule is satisfied by construction here: nothing is locally
/// computed past the gather).
///
/// NOT WIRED INTO Hdf5GroupWriter. The tier-0
/// `create_restart_point()`/`Hdf5GroupWriter` surface (every test in
/// test/runtime/restart_test.cpp) is separate from this standalone
/// utility, which a parallel-aware writer calls before its own single-rank
/// check, extending `finish_point()`'s check into a single collective
/// rather than replacing it. `test/parallel/restart_collective_mpi_
/// test.cpp` exercises this function directly, not through a full
/// parallel HDF5 write (which would additionally need the collective
/// MPI-IO property lists docs/dev/restart.md Section 4.3 describes --
/// a separate piece of work this function does not attempt).
///
/// NO <mpi.h> HERE, by the SAME convention `fuse/linalg/comm.hpp` already
/// follows (tools/check-layering.py's FORBIDDEN_PUBLIC rejects `mpi.h` in
/// any public header, `restart_io`'s own ALLOWED_PUBLIC_THIRD_PARTY entry
/// is `set()`, same as `petsc`'s): `comm_f` is an `MPI_Comm_c2f` handle
/// (a plain, ABI-stable `int`), exactly what `fuse::linalg::split_comm()`
/// already hands across this same boundary (`comm.hpp:137-140`'s own
/// comment). `src/restart_io/collective_check_mpi.cpp` -- compiled ONLY
/// under FUSE_ENABLE_MPI, never in a public header -- resolves it back
/// with `MPI_Comm_f2c()` before calling anything collective.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "fuse/base/result.hpp"

namespace fuse::restart_io {

/// `comm_f` is the collective's own `MPI_Comm_c2f` handle (see this
/// header's own top comment for why an `int`, not `MPI_Comm`, crosses
/// this boundary). `local_ids` is THIS rank's own (qualified field name,
/// id) pairs -- the SAME pairs `Hdf5GroupWriter::finish_point()` already
/// builds from its own `ids_by_name_` map, concatenated across every
/// field this rank called `put_field()` for. `local_declared_sizes` is
/// this rank's own `(qualified field name -> declared_global_size)` map
/// (one entry per distinct field name this rank called `put_field()`
/// for; a field only SOME ranks contribute to is absent from the other
/// ranks' own maps, which is ordinary, not an error -- Section 2.3's own
/// dn_heat/Brusselator split: not every rank hosts every field).
///
/// Returns the SAME Failure, with the SAME `.code`, on every rank of the
/// collective when:
///   - E-RESTART-DUPLICATE-ID: the SAME (name, id) pair was reported by
///     two different ranks (or twice by the same rank, the single-rank
///     case `finish_point()` already catches locally, re-confirmed here
///     for free once the gather has happened);
///   - E-RESTART-GLOBAL-SIZE-MISMATCH: two ranks declared DIFFERENT
///     `declared_global_size` values for the same field name;
///   - E-RESTART-BAD-NUMBERING: a field's GLOBAL received count (summed
///     across every rank that reported any ids for it) never reaches
///     its agreed `declared_global_size`.
/// The underlying `MPI_Allgatherv` is called exactly twice (once for the
/// id buffer, once for the declared-size buffer), never once per field
/// -- the "never one collective per array" rule, restated
/// for this function specifically.
[[nodiscard]] Result<void> collective_duplicate_and_coverage_check(
    int comm_f,
    const std::vector<std::pair<std::string, std::int64_t>>& local_ids,
    const std::unordered_map<std::string, std::int64_t>& local_declared_sizes);

}  // namespace fuse::restart_io
