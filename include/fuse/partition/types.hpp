#pragma once

#include <cstdint>

/// fuse::partition -- Kokkos-view RCB partitioner, ported from the user's
/// rcb_moab (a MOAB/MPI RCB implementation). General over any mesh's
/// centroids: 1D lines, 2D planar meshes, 2D manifolds embedded in 3D
/// (spherical/cubed-sphere included), and 3D volumetric meshes of any cell
/// type all go through the same `rcb_partition` entry point -- the geometry
/// is three doubles per element and nothing else. See
/// docs/dev/partitioning.md.
namespace fuse::partition {

/// Per-element weight. Matches rcb_moab's `Weight` (`int64_t`): element
/// counts on a large mesh overflow `int32`, and MPI has no portable unsigned
/// 64-bit datatype, so signed is what every collective below uses.
using Weight = std::int64_t;

/// A communicator handle that names no MPI type, so this header (and every
/// other public header in this layer) stays free of `mpi.h` -- the same
/// boundary convention `fuse::transport`'s `MpiTransport` uses
/// (`MPI_Comm_c2f`/`MPI_Comm_f2c`; see transport.hpp). `kNullComm` (0) means
/// "no communicator": exactly one participant, no MPI symbol touched at
/// all -- what makes `rcb_partition` callable in a Tier 0 (no-MPI) build.
/// A non-zero handle requires `FUSE_HAVE_MPI`; supplying one without it is
/// `E-PART-NO-MPI`.
using CommHandle = int;
inline constexpr CommHandle kNullComm = 0;

/// Histogram-median tunables, matching rcb_moab's Common.hpp defaults.
/// K=128 buckets, 4 refine iterations reaches sub-microscale bucket
/// resolution on any realistic mesh; see rcb_moab's Common.hpp comment for
/// the benchmark this was chosen against.
inline constexpr int kDefaultHistogramBuckets = 128;
inline constexpr int kDefaultHistogramRefineIters = 4;

}  // namespace fuse::partition
