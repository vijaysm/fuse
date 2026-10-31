#pragma once

/// The collective multi-task WRITE half of rank-count-independent restart.
/// `fuse::restart_io::create_restart_point()` (hdf5_restart_io.hpp) is
/// fundamentally single-writer -- its own `Hdf5GroupWriter::
/// put_field_bytes()` refuses a SECOND `put_field()` call for the same
/// qualified name, which is exactly what two real tasks each
/// contributing their own partition of the SAME field need to do. This
/// header adds the collective counterpart: every task writes its own
/// rows of a field at an `MPI_Exscan`-derived offset in one collective
/// hyperslab write, with the pre-existing `collective_duplicate_and_
/// coverage_check()` (collective_check_mpi.hpp) run once, at `finish()`,
/// over every field together, never once per field.
///
/// The READ side needs NO new code: every task already independently
/// calls the existing, unchanged `open_restart_point()`/`get_field()`
/// with its own local ids (verified by
/// `Restart.GetFieldResolvesRowsByIdNotByFilePosition`,
/// test/runtime/restart_test.cpp) -- a reader at a different task count
/// than the writer simply asks for a different set of ids and gets back
/// exactly those rows, regardless of which task wrote them or where they
/// sit in the file.
///
/// Scope: `ElementType::Binary64`/`Int64`
/// fields only (anything else refuses E-RESTART-TYPE-MISMATCH); only
/// `FileLayout::SeparateMeshAndPoints` (GrowingFile refuses
/// E-RESTART-COLLECTIVE-LAYOUT-UNSUPPORTED); scalars, rationals, and
/// blobs all refuse with E-RESTART-COLLECTIVE-SCALAR-UNSUPPORTED --
/// HDF5's own metadata operations are collective-required even for a
/// value one task alone would otherwise host, and that path is not
/// implemented.
///
/// NO <mpi.h> HERE (tools/check-layering.py's FORBIDDEN_PUBLIC, the SAME
/// convention collective_check_mpi.hpp already follows): `comm_f` is an
/// `MPI_Comm_c2f` handle, a plain ABI-stable `int`. This header is
/// included from a TU compiled only under FUSE_ENABLE_MPI
/// (src/CMakeLists.txt's own `fuse-coupler-restart-io` target).

#include <string>

#include "fuse/base/result.hpp"
#include "fuse/restart_io/hdf5_restart_io.hpp"

namespace fuse::restart_io {

/// Collective counterpart of `create_restart_point()`. Every task of the
/// communicator named by `comm_f` must call this together, with the
/// IDENTICAL `layout`, `path`, and `provenance` (the collective
/// `H5Pset_fapl_mpio`/`H5Dcreate2` calls this performs require it); each
/// task's own subsequent `put_field()` calls on the returned writer
/// contribute that task's own disjoint rows of each field, and `finish()`
/// must likewise be called by every task together.
[[nodiscard]] Result<std::unique_ptr<PointWriter>> create_restart_point_collective(
    FileLayout layout, const std::string& path, const Provenance& provenance,
    int comm_f, const std::string& point_label = {});

}  // namespace fuse::restart_io
